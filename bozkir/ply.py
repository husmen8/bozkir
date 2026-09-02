"""Read 3D Gaussian Splatting PLY files into plain numpy arrays.

Field layout follows the reference implementation from Kerbl et al. 2023.
Values on disk are stored in the parameterisation used during optimisation,
not in the form a renderer wants:

    opacity   logit          -> sigmoid
    scale     log            -> exp
    rot       raw quaternion -> normalised, (w, x, y, z)
    f_rest    channel-major  -> (N, K, 3)

This module undoes all four so the rest of the codebase never has to think
about it again.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData

# Real spherical harmonic basis constants, degrees 0 through 3.
SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))
_C1 = 0.4886025119029199
_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
       -1.0925484305920792, 0.5462742152960396)
_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
       0.3731763325901154, -0.4570457994644658, 1.445305721320277,
       -0.5900435899266435)


def _numeric_fields(names, prefix):
    """Field names starting with `prefix`, ordered by trailing integer.

    A lexical sort is wrong here: as strings, 'f_rest_10' < 'f_rest_2'.
    Sorting by the integer suffix is what keeps the SH coefficients in the
    order the writer used.
    """
    picked = [n for n in names if n.startswith(prefix)]
    return sorted(picked, key=lambda n: int(n.rsplit("_", 1)[1]))


def quat_to_matrix(q):
    """Unit quaternions (N, 4) in (w, x, y, z) order -> rotation matrices (N, 3, 3).

    Also accepts a single (4,) quaternion and returns a (1, 3, 3) stack.
    """
    q = np.atleast_2d(np.asarray(q, dtype=np.float32))
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


@dataclass
class Splats:
    """A 3DGS scene, decoded and ready to render.

    xyz      (N, 3) world-space centres
    opacity  (N,)   in [0, 1]
    scale    (N, 3) world units, strictly positive
    rot      (N, 4) unit quaternions, (w, x, y, z)
    sh_dc    (N, 3) degree-0 SH coefficient per colour channel
    sh_rest  (N, K, 3) higher-order SH, K = (degree + 1)^2 - 1
    """

    xyz: np.ndarray
    opacity: np.ndarray
    scale: np.ndarray
    rot: np.ndarray
    sh_dc: np.ndarray
    sh_rest: np.ndarray
    sh_degree: int

    def __len__(self):
        return int(self.xyz.shape[0])

    @property
    def base_rgb(self):
        """Colour ignoring view direction. Enough to render a first image."""
        return np.clip(SH_C0 * self.sh_dc + 0.5, 0.0, 1.0)

    def subset(self, idx):
        """A copy containing only the selected splats.

        `idx` may be an integer index array or a boolean mask.
        """
        return Splats(
            xyz=self.xyz[idx], opacity=self.opacity[idx], scale=self.scale[idx],
            rot=self.rot[idx], sh_dc=self.sh_dc[idx], sh_rest=self.sh_rest[idx],
            sh_degree=self.sh_degree,
        )

    def rgb(self, view_dirs):
        """Colour seen from given view directions. Full SH evaluation.

        view_dirs is (3,) for one shared direction or (N, 3) for per-splat.
        Must be unit length. Returns (N, 3) in [0, 1].
        """
        d = np.atleast_2d(np.asarray(view_dirs, dtype=np.float32))
        if d.shape[0] == 1:
            d = np.broadcast_to(d, (len(self), 3))
        x, y, z = d[:, 0:1], d[:, 1:2], d[:, 2:3]

        c = SH_C0 * self.sh_dc
        r = self.sh_rest  # (N, K, 3); r[:, i] is SH index i + 1

        if self.sh_degree >= 1:
            c += (-_C1 * y * r[:, 0]
                  + _C1 * z * r[:, 1]
                  - _C1 * x * r[:, 2])
        if self.sh_degree >= 2:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            c += (_C2[0] * xy * r[:, 3]
                  + _C2[1] * yz * r[:, 4]
                  + _C2[2] * (2.0 * zz - xx - yy) * r[:, 5]
                  + _C2[3] * xz * r[:, 6]
                  + _C2[4] * (xx - yy) * r[:, 7])
        if self.sh_degree >= 3:
            c += (_C3[0] * y * (3.0 * xx - yy) * r[:, 8]
                  + _C3[1] * xy * z * r[:, 9]
                  + _C3[2] * y * (4.0 * zz - xx - yy) * r[:, 10]
                  + _C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * r[:, 11]
                  + _C3[4] * x * (4.0 * zz - xx - yy) * r[:, 12]
                  + _C3[5] * z * (xx - yy) * r[:, 13]
                  + _C3[6] * x * (xx - 3.0 * yy) * r[:, 14])

        return np.clip(c + 0.5, 0.0, 1.0)

    def truncate_sh(self, degree):
        """A copy keeping only SH up to `degree`. Cheap way to shrink the scene."""
        if degree > self.sh_degree:
            raise ValueError(f"cannot raise degree {self.sh_degree} to {degree}")
        k = (degree + 1) ** 2 - 1
        return Splats(
            xyz=self.xyz, opacity=self.opacity, scale=self.scale, rot=self.rot,
            sh_dc=self.sh_dc, sh_rest=self.sh_rest[:, :k].copy(),
            sh_degree=degree,
        )

    def covariance(self):
        """Per-splat 3x3 world-space covariance, Sigma = R S S^T R^T.

        Equation 6 of the 3DGS paper. Returns (N, 3, 3).
        """
        R = quat_to_matrix(self.rot)
        # M = R @ S with S diagonal is just a per-column scaling of R.
        M = R * self.scale[:, None, :]
        return M @ np.transpose(M, (0, 2, 1))


_REQUIRED = (
    "x", "y", "z",
    "opacity",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
)


def load_ply(path):
    """Load a 3DGS PLY. Raises ValueError if it is not one."""
    path = Path(path)
    ply = PlyData.read(str(path))

    if "vertex" not in {e.name for e in ply.elements}:
        raise ValueError(f"{path.name}: no 'vertex' element")
    v = ply["vertex"]
    names = [p.name for p in v.properties]

    missing = [f for f in _REQUIRED if f not in names]
    if missing:
        raise ValueError(
            f"{path.name}: missing {missing}. "
            "This looks like a plain point cloud (input.ply?), not a trained "
            "3DGS scene. Trained scenes live under point_cloud/iteration_*/."
        )

    n = int(v.count)

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    # Stored as a logit so the optimiser can range over all of R.
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float32)))

    # Stored in log space so the optimiser cannot produce a negative scale.
    scale_names = _numeric_fields(names, "scale_")
    if len(scale_names) != 3:
        raise ValueError(f"{path.name}: expected 3 scale fields, got {len(scale_names)}")
    scale = np.exp(
        np.stack([v[s] for s in scale_names], axis=1).astype(np.float32)
    )

    # Written unnormalised; the reference implementation normalises on use.
    rot_names = _numeric_fields(names, "rot_")
    if len(rot_names) != 4:
        raise ValueError(f"{path.name}: expected 4 rot fields, got {len(rot_names)}")
    rot = np.stack([v[r] for r in rot_names], axis=1).astype(np.float32)
    norm = np.linalg.norm(rot, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    rot = rot / norm

    sh_dc = np.stack(
        [v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1
    ).astype(np.float32)

    rest_names = _numeric_fields(names, "f_rest_")
    if rest_names:
        flat = np.stack([v[r] for r in rest_names], axis=1).astype(np.float32)
        if flat.shape[1] % 3 != 0:
            raise ValueError(
                f"{path.name}: {flat.shape[1]} f_rest fields is not a multiple of 3"
            )
        k = flat.shape[1] // 3
        # On disk: all K coefficients of channel 0, then channel 1, then 2.
        # Reshape to (N, 3, K), then move the channel axis last.
        sh_rest = flat.reshape(n, 3, k).transpose(0, 2, 1).copy()
    else:
        k = 0
        sh_rest = np.zeros((n, 0, 3), dtype=np.float32)

    # K = (degree + 1)^2 - 1
    degree = int(round(np.sqrt(k + 1))) - 1
    if (degree + 1) ** 2 - 1 != k:
        raise ValueError(
            f"{path.name}: {k * 3} f_rest fields do not correspond to any SH degree"
        )

    return Splats(
        xyz=xyz,
        opacity=opacity,
        scale=scale,
        rot=rot,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        sh_degree=degree,
    )


def save_ply(s, path):
    """Write a Splats scene back out in the standard 3DGS PLY layout.

    Re-encodes to the on-disk parameterisation: opacity -> logit,
    scale -> log, sh_rest -> channel-major. A load/save/load round trip
    should reproduce the scene exactly.
    """
    from plyfile import PlyElement

    path = Path(path)
    n = len(s)

    p = np.clip(s.opacity, 1e-6, 1.0 - 1e-6)
    opacity = np.log(p / (1.0 - p)).astype(np.float32)
    scale = np.log(np.maximum(s.scale, 1e-20)).astype(np.float32)

    k = s.sh_rest.shape[1]
    rest = (s.sh_rest.transpose(0, 2, 1).reshape(n, 3 * k).astype(np.float32)
            if k else np.zeros((n, 0), np.float32))

    names = (["x", "y", "z", "nx", "ny", "nz"]
             + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(3 * k)]
             + ["opacity"]
             + [f"scale_{i}" for i in range(3)]
             + [f"rot_{i}" for i in range(4)])

    cols = np.concatenate(
        [s.xyz, np.zeros((n, 3), np.float32), s.sh_dc, rest,
         opacity[:, None], scale, s.rot], axis=1).astype(np.float32)

    arr = np.empty(n, dtype=[(name, "f4") for name in names])
    for i, name in enumerate(names):
        arr[name] = cols[:, i]

    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))
    return path