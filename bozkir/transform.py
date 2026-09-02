"""Rotating a Gaussian scene.

3DGS scenes come out of COLMAP with an arbitrary world orientation - the
first camera decides which way is up, and there is no reason for it to be
level. Every later step (tiling, height fields, ground-plane parameterisation)
assumes a known up direction, so the scene has to be straightened first.

Rotating a Gaussian means rotating two things: where it is, and which way
it points. Doing only the first is a silent bug - the scene looks plausible
and every shape is wrong.
"""

import numpy as np

from .ply import Splats, quat_to_matrix


def quat_multiply(a, b):
    """Hamilton product of quaternions, (w, x, y, z) order.

    Composing rotations. `quat_multiply(a, b)` means "apply b, then a" -
    the same order convention as matrix multiplication. Swapping the
    arguments gives a different rotation, not an error, so it is worth
    being deliberate about.

    Accepts (4,) or (N, 4) for either argument and broadcasts.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float32))
    b = np.atleast_2d(np.asarray(b, dtype=np.float32))
    aw, ax, ay, az = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bw, bx, by, bz = b[:, 0:1], b[:, 1:2], b[:, 2:3], b[:, 3:4]
    return np.concatenate([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1).astype(np.float32)


def quat_between(a, b):
    """Shortest rotation taking unit vector `a` onto unit vector `b`.

    Axis-angle form: spin around the axis perpendicular to both (their
    cross product) by the angle between them (from their dot product).
    A quaternion for angle t about unit axis u is (cos(t/2), sin(t/2) * u).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    c = float(np.dot(a, b))

    if s < 1e-8:
        # Parallel: nothing to do. Antiparallel: 180 degrees about any
        # perpendicular axis, so pick one that is not degenerate.
        if c > 0:
            return np.float32([1, 0, 0, 0])
        seed = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, seed)
        axis /= np.linalg.norm(axis)
        return np.concatenate([[0.0], axis]).astype(np.float32)

    axis = axis / s
    angle = np.arctan2(s, c)
    return np.concatenate(
        [[np.cos(angle / 2)], np.sin(angle / 2) * axis]).astype(np.float32)


def rotate(s, q):
    """A copy of the scene rotated by quaternion `q`.

    Positions: xyz @ R.T   (row vectors, so the matrix goes on the right
                            and is transposed)
    Shapes:    Sigma -> R Sigma R^T. Since Sigma = R_q S S^T R_q^T, that
               expands to (R R_q) S S^T (R R_q)^T - so the new rotation is
               R R_q and the scale is untouched.
    """
    q = np.asarray(q, dtype=np.float32).reshape(4)
    q = q / np.linalg.norm(q)
    R = quat_to_matrix(q)[0]
    return Splats(
        xyz=(s.xyz @ R.T).astype(np.float32),
        opacity=s.opacity,
        scale=s.scale,
        rot=quat_multiply(q, s.rot),
        sh_dc=s.sh_dc,
        sh_rest=s.sh_rest,
        sh_degree=s.sh_degree,
    )


def ground_normal(s, core_pct=90.0, low_pct=30.0):
    """Estimate the ground plane normal, pointing up.

    The dense part of a captured outdoor scene is mostly ground, so its
    thinnest principal direction is the surface normal. This is plain PCA:
    eigenvectors of the covariance of the point positions, smallest
    eigenvalue first.

    Two things a plane fit cannot tell you on its own.

    Which end of the scene is the ground: `eigh` returns an arbitrary sign,
    so both ends are fitted and the flatter one wins, since ground is a
    plane and the tops of objects are not.

    Which side of that plane is up: a plane looks the same from both sides.
    The tie is broken by where the rest of the scene is - things sit on top
    of the ground, so up is the direction from the ground band toward
    everything else. Without this the scene comes out inverted about half
    the time, and every render is upside down.
    """
    centre = np.median(s.xyz, axis=0)
    r = np.linalg.norm(s.xyz - centre, axis=1)
    core = s.xyz[r <= np.percentile(r, core_pct)]

    def fit(pts):
        d = pts - pts.mean(axis=0)
        evals, evecs = np.linalg.eigh((d.T @ d) / len(d))
        return evecs[:, 0], float(evals[0] / max(evals[2], 1e-30))

    n, planarity = fit(core)
    axis = int(np.argmax(np.abs(n)))
    h = core[:, axis]

    best_n, best_p, best_band = n, planarity, core
    for band in (core[h <= np.percentile(h, low_pct)],
                 core[h >= np.percentile(h, 100.0 - low_pct)]):
        if len(band) > 100:
            cand_n, cand_p = fit(band)
            if cand_p < best_p:
                best_n, best_p, best_band = cand_n, cand_p, band

    if _points_down(s.xyz, best_n, core, best_band):
        best_n = -best_n

    return best_n.astype(np.float32), best_p


def _points_down(xyz, n, core, band):
    """Guess whether `n` points into the ground rather than out of it.

    A plane looks the same from both sides, so this cannot be read off the
    fit. Two weak signals are combined instead.

    Skew: a captured outdoor scene has a hard floor and a long tail of
    foliage, sky and floaters above it, so the distribution of heights is
    right-skewed when measured along a normal that points up.

    Mass: things stand on the ground, so the rest of the scene tends to sit
    on the outward side of the ground band. This one fails when the surface
    that got fitted is a raised platform - a table with a pot underneath -
    which is why it only breaks ties.

    Neither is reliable alone, and for some scenes neither is right. The
    caller can always override with `up_hint`.
    """
    t = xyz @ n
    lo, hi = np.percentile(t, [1, 99])          # floaters would swamp the mean
    inner = t[(t >= lo) & (t <= hi)]
    spread = inner.std()
    skew = (inner.mean() - np.median(inner)) / spread if spread > 1e-9 else 0.0

    away = core.mean(axis=0) - band.mean(axis=0)
    mass = float(np.dot(n, away))

    # Trust skew when it is decisive; otherwise fall back on the mass test.
    if abs(skew) > 0.05:
        return skew < 0
    return mass < 0


def align_to_ground(s, target_axis=2, up_hint=None):
    """Rotate the scene so the ground lies in a coordinate plane.

    Defaults to +z, which puts the ground in the XY plane - the convention
    GSWT tiles in (Section 3.1).

    Returns (aligned_scene, info_dict).
    """
    n, planarity = ground_normal(s)

    target = np.zeros(3, dtype=np.float32)
    target[target_axis] = 1.0

    # ground_normal already points up (away from the ground, toward the
    # rest of the scene). Overriding that sign is what turns a scene upside
    # down, so only do it when explicitly asked.
    if up_hint is not None:
        hint = np.asarray(up_hint, dtype=np.float32)
        if float(np.dot(n, hint)) < 0:
            n = -n

    q = quat_between(n, target)
    out = rotate(s, q)

    tilt_before = float(np.degrees(np.arccos(np.clip(abs(n[target_axis]), 0, 1))))
    n_after, planarity_after = ground_normal(out)
    if float(np.dot(n_after, target)) < 0:
        n_after = -n_after
    tilt_after = float(np.degrees(
        np.arccos(np.clip(abs(n_after[target_axis]), 0, 1))))

    return out, {
        "normal_before": n,
        "normal_after": n_after,
        "tilt_before_deg": tilt_before,
        "tilt_after_deg": tilt_after,
        "planarity": planarity,
        "planarity_after": planarity_after,
        "quaternion": q,
    }


def recentre(s, mode="ground"):
    """Move the scene so the origin sits somewhere useful.

    'ground' puts the origin at the horizontal median, vertically at the
    5th percentile - roughly on the ground under the middle of the scene.
    'median' just uses the median of all three axes.
    """
    if mode == "median":
        offset = np.median(s.xyz, axis=0)
    elif mode == "ground":
        offset = np.median(s.xyz, axis=0)
        offset[2] = np.percentile(s.xyz[:, 2], 5.0)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    return Splats(
        xyz=(s.xyz - offset).astype(np.float32),
        opacity=s.opacity, scale=s.scale, rot=s.rot,
        sh_dc=s.sh_dc, sh_rest=s.sh_rest, sh_degree=s.sh_degree,
    ), offset.astype(np.float32)