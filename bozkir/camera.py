"""Perspective camera and projection.

Orthographic projection is linear, so the 2D covariance is an exact
submatrix of the 3D one. Perspective is not linear: dividing by depth
bends straight lines, and a Gaussian does not stay a Gaussian under it.

3DGS (Section 4, Eq. 5) handles this the way EWA splatting does - take the
first-order Taylor expansion of the projection at each splat's centre, giving
a Jacobian J, and transform the covariance as

    Sigma_2D = J W Sigma W^T J^T

with W the world-to-camera rotation. This is exact only at the centre and
degrades toward the edge of the frame, which is why the projection is
clamped to a slightly enlarged frustum below.

Camera convention follows COLMAP and OpenCV: in camera space the camera
looks down +z, x points right, y points down. Anything with z <= near is
behind the camera.
"""

import numpy as np

from .render import DILATION, MIN_ALPHA


class Camera:
    """A pinhole camera.

    position  eye point in world space
    target    point being looked at
    up        world up direction, used to settle the roll angle
    fov_deg   vertical field of view
    """

    def __init__(self, position, target, up=(0.0, 0.0, 1.0),
                 fov_deg=60.0, width=800, height=600, near=0.01):
        self.position = np.asarray(position, dtype=np.float32).reshape(3)
        self.target = np.asarray(target, dtype=np.float32).reshape(3)
        self.width = int(width)
        self.height = int(height)
        self.near = float(near)

        forward = self.target - self.position
        norm = np.linalg.norm(forward)
        if norm < 1e-9:
            raise ValueError("camera position and target coincide")
        forward = forward / norm

        up = np.asarray(up, dtype=np.float32).reshape(3)
        if abs(float(np.dot(forward, up / np.linalg.norm(up)))) > 0.999:
            # Looking straight along `up`: roll is undefined, pick anything.
            up = np.float32([1.0, 0.0, 0.0])
            if abs(float(np.dot(forward, up))) > 0.999:
                up = np.float32([0.0, 1.0, 0.0])

        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)          # camera y points down

        # Rows are the camera axes in world space, so W maps world -> camera.
        self.W = np.stack([right, down, forward]).astype(np.float32)

        self.fov_deg = float(fov_deg)
        self.tan_half_y = float(np.tan(np.radians(fov_deg) / 2.0))
        self.fy = (height / 2.0) / self.tan_half_y
        self.fx = self.fy                        # square pixels
        self.tan_half_x = (width / 2.0) / self.fx
        self.cx = width / 2.0
        self.cy = height / 2.0

    def to_camera(self, xyz):
        """World points (N, 3) -> camera space (N, 3)."""
        return (xyz - self.position) @ self.W.T


def orbit_camera(target, distance, azimuth_deg, elevation_deg,
                 up_axis=2, **kw):
    """A camera on a sphere around `target`.

    elevation 0 looks along the ground (grazing), 90 looks straight down.
    azimuth sweeps around the up axis.
    """
    target = np.asarray(target, dtype=np.float32).reshape(3)
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)

    plane = [i for i in range(3) if i != up_axis]
    offset = np.zeros(3, dtype=np.float32)
    offset[plane[0]] = np.cos(el) * np.cos(az)
    offset[plane[1]] = np.cos(el) * np.sin(az)
    offset[up_axis] = np.sin(el)

    up = np.zeros(3, dtype=np.float32)
    up[up_axis] = 1.0
    return Camera(target + offset * distance, target, up=up, **kw)


def project_perspective(cam, s, sh_degree=None, guard=1.3):
    """Project a scene through a perspective camera.

    Returns the same dict shape as project_orthographic, so the rasterizer
    does not need to know which projection produced it.
    """
    scene = s.truncate_sh(sh_degree) if sh_degree is not None else s
    W, H = cam.width, cam.height

    p_cam = cam.to_camera(s.xyz)
    z = p_cam[:, 2]

    in_front = z > cam.near
    idx = np.flatnonzero(in_front)
    if len(idx) == 0:
        raise ValueError("every splat is behind the camera")

    p_cam = p_cam[idx]
    z = p_cam[:, 2]
    x, y = p_cam[:, 0], p_cam[:, 1]

    mean2d = np.stack([cam.fx * x / z + cam.cx,
                       cam.fy * y / z + cam.cy], axis=1).astype(np.float32)

    # The Taylor expansion is only good near the optical axis. 3DGS clamps
    # x/z and y/z to a slightly enlarged frustum so splats just outside the
    # frame do not get wildly distorted covariances.
    lim_x = guard * cam.tan_half_x
    lim_y = guard * cam.tan_half_y
    xc = np.clip(x / z, -lim_x, lim_x) * z
    yc = np.clip(y / z, -lim_y, lim_y) * z

    # J = d(screen) / d(camera position), evaluated at the splat centre.
    #     [ fx/z    0    -fx*x/z^2 ]
    #     [  0    fy/z   -fy*y/z^2 ]
    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z
    J = np.zeros((len(idx), 2, 3), dtype=np.float32)
    J[:, 0, 0] = cam.fx * inv_z
    J[:, 0, 2] = -cam.fx * xc * inv_z2
    J[:, 1, 1] = cam.fy * inv_z
    J[:, 1, 2] = -cam.fy * yc * inv_z2

    cov3 = s.covariance()[idx]
    T = J @ cam.W                                 # (N, 2, 3)
    cov2d = T @ cov3 @ np.transpose(T, (0, 2, 1))
    cov2d[:, 0, 0] += DILATION
    cov2d[:, 1, 1] += DILATION

    # Direction from camera to splat, for view-dependent colour. Evaluated
    # on the visible subset only - the direction array and the SH array
    # must have matching lengths or numpy will broadcast them wrongly.
    view_dirs = s.xyz[idx] - cam.position
    view_dirs /= np.maximum(np.linalg.norm(view_dirs, axis=1, keepdims=True), 1e-9)
    colour = scene.subset(idx).rgb(view_dirs.astype(np.float32))

    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    mid = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    disc = np.sqrt(np.maximum(mid * mid - det, 0.0))
    radius = 3.0 * np.sqrt(np.maximum(mid + disc, 1e-6))

    opacity = s.opacity[idx]
    keep = (
        (det > 1e-9)
        & (opacity > MIN_ALPHA)
        & (mean2d[:, 0] + radius >= 0) & (mean2d[:, 0] - radius < W)
        & (mean2d[:, 1] + radius >= 0) & (mean2d[:, 1] - radius < H)
    )

    return {
        "mean2d": mean2d[keep], "cov2d": cov2d[keep],
        # 3DGS sorts on camera-space z. Nearer means smaller z here, and the
        # rasterizer draws largest depth first, so negate.
        "depth": (-z[keep]).astype(np.float32),
        "colour": colour[keep], "opacity": opacity[keep],
        "radius": radius[keep], "W": W, "H": H,
        "kept": int(keep.sum()), "total": len(s),
        "behind": int(len(s) - len(idx)),
    }


def render_perspective(cam, s, sh_degree=None, **kw):
    """Project and rasterize in one call."""
    from .render import rasterize
    p = project_perspective(cam, s, sh_degree=sh_degree)
    return rasterize(p, **kw), p