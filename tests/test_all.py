"""Every check, in one place.

    python tests/test_all.py

Also works under pytest if you have it. No real PLY needed - scenes with
known answers are built in memory, so a failure means the code is wrong
rather than the data being odd.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bozkir.ply import Splats, quat_to_matrix, SH_C0  # noqa: E402
from bozkir.render import (project_orthographic, rasterize,   # noqa: E402
                           render_orthographic, DILATION)
from bozkir.camera import (Camera, orbit_camera,              # noqa: E402
                           project_perspective, render_perspective)
from bozkir.transform import (quat_between, quat_multiply,    # noqa: E402
                              rotate, align_to_ground, ground_normal)
from bozkir.select import (crop_box, crop_cylinder,           # noqa: E402
                           remove_large, remove_floaters)
from bozkir.render import rasterize_rgba, over, flatten        # noqa: E402
from bozkir.tile import (extract_patch, translate, merge,      # noqa: E402
                         grid, render_global, render_tiled, seam_camera)
from bozkir.scene import SceneConfig, config_from_args           # noqa: E402

RNG = np.random.default_rng(0)


def scene(xyz, scale=None, opacity=None, rot=None, rgb=None, sh_degree=0, k=0):
    """Build a Splats from whatever is specified; sensible defaults for the rest."""
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = len(xyz)
    if scale is None:
        scale = np.full((n, 3), 0.01, np.float32)
    if opacity is None:
        opacity = np.full(n, 0.5, np.float32)
    if rot is None:
        rot = np.tile(np.float32([1, 0, 0, 0]), (n, 1))
    if rgb is None:
        dc = np.zeros((n, 3), np.float32)
    else:
        dc = ((np.asarray(rgb, np.float32).reshape(-1, 3) - 0.5) / SH_C0)
    return Splats(xyz=xyz, opacity=np.asarray(opacity, np.float32).reshape(n),
                  scale=np.asarray(scale, np.float32).reshape(n, 3),
                  rot=np.asarray(rot, np.float32).reshape(n, 4),
                  sh_dc=dc.astype(np.float32),
                  sh_rest=np.zeros((n, k, 3), np.float32), sh_degree=sh_degree)


# ---------------------------------------------------------------- ply.py

def test_sh_rest_layout_is_channel_major():
    """The writer stores all K coefficients of R, then G, then B."""
    n, k = 100, 15
    true = RNG.normal(0, 0.1, (n, k, 3)).astype(np.float32)
    flat = true.transpose(0, 2, 1).reshape(n, k * 3)          # what goes to disk
    assert np.array_equal(flat.reshape(n, 3, k).transpose(0, 2, 1), true)
    assert not np.array_equal(flat.reshape(n, k, 3), true)    # the naive guess


def test_field_names_sort_numerically():
    names = [f"f_rest_{i}" for i in range(45)]
    assert sorted(names) != sorted(names, key=lambda x: int(x.rsplit("_", 1)[1]))
    assert sorted(names)[2] == "f_rest_10"                    # lexical is wrong


def test_covariance_eigenvalues_are_scale_squared():
    n = 500
    q = RNG.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    sc = np.exp(RNG.normal(-3, 1, (n, 3))).astype(np.float32)
    s = scene(RNG.normal(0, 1, (n, 3)), scale=sc, rot=q)
    ev = np.sort(np.linalg.eigvalsh(s.covariance().astype(np.float64)), axis=1)
    assert np.allclose(ev, np.sort(sc.astype(np.float64) ** 2, axis=1), rtol=1e-3)


def test_rotation_matrices_are_orthonormal():
    q = RNG.normal(size=(200, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    R = quat_to_matrix(q).astype(np.float64)
    assert np.allclose(R @ np.transpose(R, (0, 2, 1)), np.eye(3), atol=1e-5)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-5)


def test_sh_degree_inference():
    for deg in (0, 1, 2, 3):
        k = (deg + 1) ** 2 - 1
        assert int(round(np.sqrt(k + 1))) - 1 == deg


def test_sh_evaluation_along_pole():
    """Looking down +z, only the z term of degree 1 survives."""
    n = 50
    s = scene(RNG.normal(0, 1, (n, 3)), sh_degree=1, k=3)
    s.sh_rest[:] = RNG.normal(0, 0.1, (n, 3, 3))
    s.sh_dc[:] = RNG.normal(0, 0.3, (n, 3))
    C1 = 0.4886025119029199
    want = np.clip(SH_C0 * s.sh_dc + C1 * s.sh_rest[:, 1] + 0.5, 0, 1)
    assert np.allclose(s.rgb(np.float32([0, 0, 1])), want, atol=1e-6)


def test_truncate_sh():
    n = 50
    s = scene(RNG.normal(0, 1, (n, 3)), sh_degree=3, k=15)
    s.sh_rest[:] = RNG.normal(0, 0.1, (n, 15, 3))
    t = s.truncate_sh(1)
    assert t.sh_rest.shape[1] == 3
    assert np.array_equal(t.sh_rest, s.sh_rest[:, :3])
    assert np.allclose(s.truncate_sh(0).rgb(np.float32([0, 0, 1])), s.base_rgb)
    try:
        s.truncate_sh(5)
        raise AssertionError("should refuse to raise the degree")
    except ValueError:
        pass


def test_subset_carries_all_attributes():
    n = 200
    s = scene(RNG.normal(0, 1, (n, 3)), sh_degree=3, k=15)
    idx = RNG.choice(n, 50, replace=False)
    t = s.subset(idx)
    for name in ("xyz", "opacity", "scale", "rot", "sh_dc", "sh_rest"):
        assert np.array_equal(getattr(t, name), getattr(s, name)[idx]), name


# ------------------------------------------------------------- render.py

def test_single_splat_is_symmetric():
    s = scene([[0, 0, 0], [-1, 0, -1], [1, 0, 1]],
              scale=[[0.2] * 3, [1e-6] * 3, [1e-6] * 3],
              opacity=[1, 0, 0], rgb=[[1, 0, 0]] * 3)
    img, _ = render_orthographic(s, view_axis=1, resolution=200,
                                 bounds_pct=(0, 100))
    ink = 1.0 - img[:, :, 1]
    assert np.allclose(ink, ink[:, ::-1], atol=1e-5)
    assert np.allclose(ink, ink[::-1, :], atol=1e-5)


def test_alpha_compositing_arithmetic():
    """Half-opaque red over white must give exactly (1, 0.5, 0.5)."""
    s = scene([[0, 0, 0], [-1, 0, -1], [1, 0, 1]],
              scale=[[0.3] * 3, [1e-6] * 3, [1e-6] * 3],
              opacity=[0.5, 0, 0], rgb=[[1, 0, 0]] * 3)
    img, _ = render_orthographic(s, view_axis=1, resolution=101,
                                 bounds_pct=(0, 100))
    assert np.allclose(img[50, 50], [1.0, 0.5, 0.5], atol=0.02)


def test_occlusion_flips_with_camera_side():
    s = scene([[0, -1, 0], [0, 1, 0], [-1, 0, -1], [1, 0, 1]],
              scale=[[0.3] * 3, [0.3] * 3, [1e-6] * 3, [1e-6] * 3],
              opacity=[1, 1, 0, 0],
              rgb=[[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]])
    a, _ = render_orthographic(s, view_axis=1, up_sign=-1, resolution=101,
                               bounds_pct=(0, 100))
    b, _ = render_orthographic(s, view_axis=1, up_sign=+1, resolution=101,
                               bounds_pct=(0, 100))
    assert a[50, 50][0] > a[50, 50][2]
    assert b[50, 50][2] > b[50, 50][0]


def test_anisotropic_splat_keeps_its_aspect_ratio():
    s = scene([[0, 0, 0], [-1, 0, -1], [1, 0, 1]],
              scale=[[0.4, 0.05, 0.02], [1e-6] * 3, [1e-6] * 3],
              opacity=[1, 0, 0], rgb=[[1, 0, 0]] * 3)
    img, _ = render_orthographic(s, view_axis=1, resolution=200,
                                 bounds_pct=(0, 100))
    ink = 1.0 - img[:, :, 1] > 0.05
    ratio = ink.any(0).sum() / max(ink.any(1).sum(), 1)
    assert 10 < ratio < 30, ratio                             # world ratio is 20


def test_rotation_shows_up_on_screen():
    q = np.float32([np.cos(np.pi / 8), 0, np.sin(np.pi / 8), 0])   # 45 deg about y
    s = scene([[0, 0, 0], [-1, 0, -1], [1, 0, 1]],
              scale=[[0.4, 0.05, 0.02], [1e-6] * 3, [1e-6] * 3],
              opacity=[1, 0, 0], rgb=[[1, 0, 0]] * 3)
    s.rot[0] = q
    img, _ = render_orthographic(s, view_axis=1, resolution=200,
                                 bounds_pct=(0, 100))
    ys, xs = np.nonzero(1.0 - img[:, :, 1] > 0.05)
    _, evec = np.linalg.eigh(np.cov(np.stack([xs, ys]).astype(float)))
    ang = np.degrees(np.arctan2(evec[1, -1], evec[0, -1])) % 180
    assert 35 < ang < 55 or 125 < ang < 145, ang


# ------------------------------------------------------------- camera.py

def test_target_projects_to_image_centre():
    s = scene([[0, 0, 0]], scale=[[0.05] * 3], opacity=[1])
    cam = Camera([0, -5, 0], [0, 0, 0], up=[0, 0, 1], width=800, height=600)
    p = project_perspective(cam, s)
    assert np.allclose(p["mean2d"][0], [400, 300], atol=1e-3)


def test_size_falls_off_as_one_over_distance():
    s = scene([[0, 0, 0]], scale=[[0.05] * 3], opacity=[1])
    r = [project_perspective(
            Camera([0, -d, 0], [0, 0, 0], up=[0, 0, 1], width=800, height=600),
            s)["radius"][0] for d in (2.0, 4.0, 8.0)]
    assert np.allclose(r[0] / r[1], 2.0, rtol=0.05)
    assert np.allclose(r[1] / r[2], 2.0, rtol=0.05)


def test_splats_behind_the_camera_are_dropped():
    s = scene([[0, -2, 0], [0, 2, 0]], scale=[[0.05] * 3] * 2, opacity=[1, 1])
    p = project_perspective(
        Camera([0, 0, 0.001], [0, 1, 0], up=[0, 0, 1], width=400, height=300), s)
    assert p["kept"] == 1 and p["behind"] == 1


def test_jacobian_matches_finite_differences():
    """The linearisation in 3DGS Eq. 5 must be the real derivative."""
    cam = Camera([1.0, -6.0, 2.0], [0, 0, 0], up=[0, 0, 1],
                 fov_deg=55, width=800, height=600)
    pts = RNG.normal(0, 1, (100, 3))
    pc = (pts - cam.position) @ cam.W.T.astype(np.float64)

    def proj(p):
        return np.stack([cam.fx * p[..., 0] / p[..., 2] + cam.cx,
                         cam.fy * p[..., 1] / p[..., 2] + cam.cy], -1)

    eps = 1e-6
    num = np.zeros((len(pts), 2, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        num[:, :, i] = (proj(pc + d) - proj(pc - d)) / (2 * eps)

    z = pc[:, 2]
    ana = np.zeros_like(num)
    ana[:, 0, 0] = cam.fx / z
    ana[:, 0, 2] = -cam.fx * pc[:, 0] / z ** 2
    ana[:, 1, 1] = cam.fy / z
    ana[:, 1, 2] = -cam.fy * pc[:, 1] / z ** 2
    assert np.abs(ana - num).max() / np.abs(num).max() < 1e-6


def test_projected_covariance_matches_monte_carlo():
    """Sample the 3D Gaussian, project the samples, compare their spread."""
    cam = Camera([1.0, -6.0, 2.0], [0, 0, 0], up=[0, 0, 1],
                 fov_deg=55, width=800, height=600)
    for _ in range(3):
        mu = RNG.uniform(-1.5, 1.5, 3).astype(np.float32)
        sc = (np.float32([0.25, 0.12, 0.05]) * RNG.uniform(0.6, 1.6))
        q = RNG.normal(size=4).astype(np.float32)
        q /= np.linalg.norm(q)
        s = scene(mu[None], scale=sc[None], rot=q[None], opacity=[1])

        p = project_perspective(cam, s)
        ana = p["cov2d"][0].astype(np.float64).copy()
        ana[0, 0] -= DILATION
        ana[1, 1] -= DILATION

        samp = RNG.multivariate_normal(mu.astype(np.float64),
                                       s.covariance()[0].astype(np.float64),
                                       size=200_000)
        pc = (samp - cam.position) @ cam.W.T.astype(np.float64)
        scr = np.stack([cam.fx * pc[:, 0] / pc[:, 2] + cam.cx,
                        cam.fy * pc[:, 1] / pc[:, 2] + cam.cy], -1)
        emp = np.cov(scr.T)
        assert np.abs(ana - emp).max() / np.abs(emp).max() < 0.05


def test_perspective_converges_to_orthographic():
    """A long lens from far away is orthographic. Positions must agree."""
    n = 2000
    xy = RNG.uniform(-1, 1, (n, 2))
    xy -= (xy.max(0) + xy.min(0)) / 2
    s = scene(np.concatenate([xy, np.zeros((n, 1))], 1),
              scale=np.full((n, 3), 0.02, np.float32),
              opacity=np.full(n, 0.6, np.float32))

    op = project_orthographic(s, view_axis=2, up_sign=+1, resolution=300,
                              bounds_pct=(0, 100), sh_degree=0)
    D = 1000.0
    fy = op["px_per_unit"] * D
    cam = Camera([0, 0, D], [0, 0, 0], up=[0, 1, 0],
                 fov_deg=2 * np.degrees(np.arctan((op["H"] / 2) / fy)),
                 width=op["W"], height=op["H"])
    pp = project_perspective(cam, s)
    d = pp["mean2d"] - op["mean2d"]
    # A constant offset is framing (where each projection puts the origin);
    # only the spread around it would indicate a real disagreement.
    assert np.abs(d).max() < 0.5, np.abs(d).max()
    assert d.std(axis=0).max() < 1e-3, d.std(axis=0)
    assert np.allclose(pp["radius"] / op["radius"], 1.0, rtol=1e-3)


def test_orbit_elevation_controls_height():
    for el, want in ((0, 0.0), (90, 10.0)):
        c = orbit_camera([0, 0, 0], 10, 0, el, up_axis=2, width=10, height=10)
        assert np.allclose(c.position[2], want, atol=1e-4)


# ---------------------------------------------------------- transform.py

def test_quat_between_lands_on_target():
    for _ in range(300):
        a = RNG.normal(size=3); a /= np.linalg.norm(a)
        b = RNG.normal(size=3); b /= np.linalg.norm(b)
        R = quat_to_matrix(quat_between(a, b))[0]
        assert np.allclose(R @ a, b, atol=1e-4)


def test_quat_between_handles_degenerate_cases():
    for a, b in (([0, 0, 1], [0, 0, 1]), ([0, 0, -1], [0, 0, 1]),
                 ([1, 0, 0], [-1, 0, 0])):
        R = quat_to_matrix(quat_between(a, b))[0]
        assert np.allclose(R @ np.float32(a), b, atol=1e-5)


def test_quat_multiply_composes_like_matrices():
    q1 = quat_between([1, 0, 0], [0, 1, 0])
    q2 = quat_between([0, 1, 0], [0, 0, 1])
    R1, R2 = quat_to_matrix(q1)[0], quat_to_matrix(q2)[0]
    assert np.allclose(quat_to_matrix(quat_multiply(q2, q1))[0], R2 @ R1, atol=1e-5)
    assert not np.allclose(quat_multiply(q1, q2), quat_multiply(q2, q1), atol=1e-5)


def test_rotation_preserves_shape():
    """Rotating must move splats without resizing or reshaping them."""
    n = 2000
    q = RNG.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    long_ = np.exp(RNG.normal(-5, 1, n))
    ratio = 10 ** RNG.uniform(0, 5, n)                 # up to 1e5:1, like real data
    sc = np.stack([long_ / ratio, long_ / np.sqrt(ratio), long_], 1).astype(np.float32)
    s = scene(RNG.normal(0, 1, (n, 3)), scale=sc, rot=q)

    qr = quat_between([0, 0, 1], [0.4, 0, 0.9])
    r = rotate(s, qr)
    R = quat_to_matrix(qr)[0].astype(np.float64)
    C0 = s.covariance().astype(np.float64)
    C1 = r.covariance().astype(np.float64)
    # Comparing eigenvalues to themselves is ill-conditioned when splats are
    # this thin; compare the covariance directly instead.
    assert np.abs(C1 - R @ C0 @ R.T).max() / np.abs(C0).max() < 1e-4
    assert np.array_equal(s.scale, r.scale)


def test_forgetting_to_rotate_orientations_is_caught():
    n = 500
    q = RNG.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    s = scene(RNG.normal(0, 1, (n, 3)),
              scale=np.exp(RNG.normal(-3, 1, (n, 3))).astype(np.float32), rot=q)
    qr = quat_between([0, 0, 1], [0.4, 0, 0.9])
    good = rotate(s, qr)
    bad = Splats(xyz=good.xyz, opacity=s.opacity, scale=s.scale, rot=s.rot,
                 sh_dc=s.sh_dc, sh_rest=s.sh_rest, sh_degree=s.sh_degree)
    R = quat_to_matrix(qr)[0].astype(np.float64)
    C0 = s.covariance().astype(np.float64)
    err = np.abs(bad.covariance().astype(np.float64) - R @ C0 @ R.T).max()
    assert err / np.abs(C0).max() > 1e-2


def _tilted_ground(tilt_deg, azim_deg=0.0):
    n = 40_000
    ground = np.stack([RNG.uniform(-5, 5, n), RNG.uniform(-5, 5, n),
                       RNG.normal(0, 0.03, n)], 1)
    obj = np.stack([RNG.normal(0, 0.5, 6000), RNG.normal(0, 0.5, 6000),
                    RNG.uniform(0, 2, 6000)], 1)
    fl = RNG.uniform(-25, 25, (1200, 3))
    xyz = np.vstack([ground, obj, fl]).astype(np.float32)
    N = len(xyz)
    q = RNG.normal(size=(N, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    s = scene(xyz, scale=np.exp(RNG.normal(-3, 1, (N, 3))).astype(np.float32), rot=q)
    a, b = np.radians(tilt_deg), np.radians(azim_deg)
    tgt = [np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)]
    return rotate(s, quat_between([0, 0, 1], tgt))


def test_alignment_always_ends_up_right_side_up():
    """A plane fit cannot tell which side is up; the scene's mass can.

    Without this the ground normal's arbitrary sign leaves the scene
    inverted about half the time, and every render is upside down.
    """
    for tgt in ([0, 0, 1], [0.42, 0, 0.91], [0.9, 0, 0.42],
                [0, 0, -1], [0.42, 0, -0.91], [1, 0, 0], [-1, 0, 0]):
        s = rotate(_tilted_ground(0.0), quat_between([0, 0, 1], tgt))
        out, info = align_to_ground(s)
        assert info["tilt_after_deg"] < 0.5, (tgt, info)
        # Ground at the bottom means most mass sits above the floor.
        floor = np.percentile(out.xyz[:, 2], 5)
        assert out.xyz[:, 2].mean() - floor > 0.1, f"inverted from {tgt}"


def test_alignment_recovers_known_tilt():
    for tilt in (5, 25, 65):
        for azim in (0, 210):
            _, info = align_to_ground(_tilted_ground(tilt, azim))
            assert abs(info["tilt_before_deg"] - tilt) < 1.0, info
            assert info["tilt_after_deg"] < 0.5, info


def test_ground_fit_ignores_objects_standing_on_it():
    """The eigenvector sign is arbitrary, so the flatter band must win."""
    _, planarity = ground_normal(_tilted_ground(0.0))
    assert planarity < 0.01, planarity


# ------------------------------------------------------------- select.py

def test_crop_box_matches_brute_force():
    pts = RNG.uniform(-2, 2, (3000, 3))
    s = scene(pts)
    _, keep = crop_box(s, [-1, -1, -1], [1, 1, 1])
    assert np.array_equal(keep, np.all((pts >= -1) & (pts <= 1), axis=1))


def test_crop_box_on_two_axes_leaves_height_unbounded():
    """GSWT decides patch membership from the ground-plane position alone."""
    pts = RNG.uniform(-2, 2, (3000, 3))
    s = scene(pts)
    out, keep = crop_box(s, [-1, -1], [1, 1], axes=(0, 1))
    want = np.all((pts[:, :2] >= -1) & (pts[:, :2] <= 1), axis=1)
    assert np.array_equal(keep, want)
    assert out.xyz[:, 2].min() < -1.5 and out.xyz[:, 2].max() > 1.5


def test_crop_cylinder_matches_brute_force():
    pts = RNG.uniform(-2, 2, (3000, 3))
    _, keep = crop_cylinder(scene(pts), [0, 0, 0], 1.0, up_axis=2)
    assert np.array_equal(keep, np.linalg.norm(pts[:, :2], axis=1) <= 1.0)


def test_remove_large_drops_exactly_the_giants():
    sc = np.full((2000, 3), 0.01)
    sc[:50] = 2.0
    _, keep = remove_large(scene(RNG.uniform(-2, 2, (2000, 3)), scale=sc), 1.0)
    assert keep.sum() == 1950 and not keep[:50].any()


def test_remove_floaters_discriminates():
    dense = RNG.normal(0, 0.3, (15_000, 3))
    floats = RNG.uniform(-8, 8, (200, 3))
    s = scene(np.vstack([dense, floats]))
    _, keep = remove_floaters(s, k=8, std_ratio=2.0, weight_by_opacity=False)
    assert keep[:15_000].mean() > 0.95
    assert keep[15_000:].mean() < 0.35


def test_translate_moves_only_positions():
    n = 300
    q = RNG.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    s = scene(RNG.normal(0, 1, (n, 3)), rot=q)
    t = translate(s, [1.0, -2.0, 0.5])
    assert np.allclose(t.xyz - s.xyz, [1.0, -2.0, 0.5], atol=1e-6)
    assert np.array_equal(t.rot, s.rot)
    assert np.array_equal(t.scale, s.scale)
    # Shapes are unchanged, so the covariance must be identical.
    assert np.array_equal(t.covariance(), s.covariance())


def test_merge_preserves_everything():
    a = scene(RNG.normal(0, 1, (100, 3)), sh_degree=1, k=3)
    b = scene(RNG.normal(5, 1, (60, 3)), sh_degree=1, k=3)
    m = merge(a, b)
    assert len(m) == 160
    for name in ("xyz", "opacity", "scale", "rot", "sh_dc", "sh_rest"):
        want = np.concatenate([getattr(a, name), getattr(b, name)])
        assert np.array_equal(getattr(m, name), want), name


def test_merge_truncates_to_the_lowest_sh_degree():
    """Mixed degrees are common once tiles come from different captures."""
    a = scene(RNG.normal(0, 1, (10, 3)), sh_degree=1, k=3)
    b = scene(RNG.normal(0, 1, (10, 3)), sh_degree=3, k=15)
    m = merge(a, b)
    assert m.sh_degree == 1
    assert m.sh_rest.shape[1] == 3
    assert np.array_equal(m.sh_rest[10:], b.sh_rest[:, :3])


def test_naive_merge_produces_a_boundary():
    """Two copies of a patch placed flush must abut, not overlap or gap."""
    pts = RNG.uniform(-3, 3, (20_000, 3))
    s = scene(pts)
    size = 2.0
    a, _ = crop_box(s, [-size / 2, -size / 2], [size / 2, size / 2], axes=(0, 1))
    b = translate(a, [size, 0, 0])
    m = merge(a, b)

    assert len(m) == 2 * len(a)
    assert a.xyz[:, 0].max() <= size / 2 + 1e-5
    assert b.xyz[:, 0].min() >= size / 2 - 1e-5
    assert np.abs(m.xyz[:, 0]).max() <= 1.5 * size + 1e-5


# --------------------------------------------------------------- tile.py

def test_rgba_path_matches_rgb_path():
    n = 2000
    s = scene(RNG.normal(0, 0.5, (n, 3)),
              opacity=RNG.uniform(0.1, 0.9, n).astype(np.float32),
              scale=np.full((n, 3), 0.02, np.float32),
              rgb=RNG.uniform(0, 1, (n, 3)))
    cam = Camera([0, -4, 1], [0, 0, 0], up=[0, 0, 1], width=200, height=150)
    p = project_perspective(cam, s, sh_degree=0)
    for bg in ((0, 0, 0), (1, 1, 1)):
        assert np.abs(rasterize(p, background=bg)
                      - flatten(rasterize_rgba(p), bg)).max() < 1e-5


def test_extract_patch_uses_ground_plane_only():
    pts = RNG.uniform(-3, 3, (4000, 3))
    s = scene(pts)
    p = extract_patch(s, [1.0, 0.0], 2.0, up_axis=2, recentre=False)
    want = np.all((pts[:, :2] - [1, 0] >= -1) & (pts[:, :2] - [1, 0] <= 1), axis=1)
    assert len(p) == want.sum()
    assert p.xyz[:, 2].min() < -2 and p.xyz[:, 2].max() > 2   # height unbounded


def test_extract_patch_recentres_horizontally_only():
    pts = RNG.uniform(-3, 3, (4000, 3))
    s = scene(pts)
    a = extract_patch(s, [1.0, 0.5], 2.0, recentre=False)
    b = extract_patch(s, [1.0, 0.5], 2.0, recentre=True)
    assert np.abs(b.xyz[:, :2]).max() <= 1.001
    assert np.allclose(np.sort(a.xyz[:, 2]), np.sort(b.xyz[:, 2]))


def test_translate_moves_positions_only():
    s = scene(RNG.normal(0, 1, (500, 3)))
    t = translate(s, [1, 2, 3])
    assert np.allclose(t.xyz - s.xyz, [1, 2, 3])
    assert np.array_equal(t.scale, s.scale) and np.array_equal(t.rot, s.rot)


def test_merge_concatenates():
    a = scene(RNG.normal(0, 1, (300, 3)))
    b = scene(RNG.normal(0, 1, (200, 3)))
    m = merge(a, b)
    assert len(m) == 500
    assert np.array_equal(m.xyz[:300], a.xyz) and np.array_equal(m.xyz[300:], b.xyz)


def test_grid_lays_tiles_edge_to_edge():
    s = scene(RNG.uniform(-0.5, 0.5, (200, 3)))
    tiles = grid(s, 2, 1, 1.0, up_axis=2)
    assert len(tiles) == 2
    dx = tiles[1].xyz[:, 0].mean() - tiles[0].xyz[:, 0].mean()
    assert np.allclose(dx, 1.0, atol=1e-5)


def _two_rows(opacity, interleave=True):
    ys = np.linspace(-2, 2, 8)
    ay, by = (ys[0::2], ys[1::2]) if interleave else (ys[:4], ys[4:])
    mk = lambda yy, c: scene([[0, y, 0] for y in yy], opacity=[opacity] * 4,
                             scale=np.full((4, 3), 0.15, np.float32), rgb=[c] * 4)
    return mk(ay, [1, 0, 0]), mk(by, [0, 0, 1])


def test_tiled_matches_global_when_tiles_do_not_overlap():
    a = scene([[-1, 0, 0]], opacity=[0.8], scale=[[0.15] * 3], rgb=[[1, 0, 0]])
    b = scene([[1, 0, 0]], opacity=[0.8], scale=[[0.15] * 3], rgb=[[0, 0, 1]])
    cam = Camera([0, -12, 0], [0, 0, 0], up=[0, 0, 1], fov_deg=30,
                 width=200, height=150)
    g, _ = render_global(cam, [a, b], sh_degree=0)
    t, _ = render_tiled(cam, [a, b], sh_degree=0)
    assert np.abs(g - t).max() < 1e-6


def test_tiled_differs_from_global_when_tiles_interleave_in_depth():
    """The boundary artifact, isolated. Same geometry, only the sort differs."""
    cam = Camera([0, -12, 0], [0, 0, 0], up=[0, 0, 1], fov_deg=30,
                 width=100, height=100)

    a, b = _two_rows(0.5, interleave=True)
    g, _ = render_global(cam, [a, b], sh_degree=0)
    t, _ = render_tiled(cam, [a, b], sh_degree=0)
    mask = g.sum(2) > 0.01
    interleaved = np.abs(g - t)[mask].mean() / g[mask].mean()

    a, b = _two_rows(0.5, interleave=False)
    g, _ = render_global(cam, [a, b], sh_degree=0)
    t, _ = render_tiled(cam, [a, b], sh_degree=0)
    mask = g.sum(2) > 0.01
    separated = np.abs(g - t)[mask].mean() / g[mask].mean()

    assert interleaved > 0.10, interleaved
    assert separated < 1e-6, separated


def test_tile_entirely_behind_camera_is_skipped():
    """Normal once a grid is large enough - must not raise."""
    patch = scene(RNG.uniform(-0.5, 0.5, (300, 3)),
                  scale=np.full((300, 3), 0.03, np.float32))
    tiles = grid(patch, 6, 6, 1.0, up_axis=2)
    cam = seam_camera([0, 0, 0.05], 1.5, 3.0, 0.0, up_axis=2,
                      fov_deg=50, width=80, height=60)
    behind = [t for t in tiles
              if project_perspective(cam, t, sh_degree=0)["kept"] == 0]
    assert behind, "test needs at least one tile out of view"
    img, info = render_tiled(cam, tiles, sh_degree=0)
    assert np.isfinite(img).all() and info["layers"] < len(tiles)


def test_more_tiles_means_more_artifact():
    """More boundaries stacked along a grazing ray affect more pixels."""
    n = 4000
    xyz = np.stack([RNG.uniform(-0.5, 0.5, n), RNG.uniform(-0.5, 0.5, n),
                    RNG.normal(0, 0.03, n)], 1)
    q = RNG.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    patch = scene(xyz, rot=q,
                  scale=np.stack([np.full(n, 0.05), np.full(n, 0.04),
                                  np.full(n, 0.008)], 1).astype(np.float32),
                  opacity=RNG.uniform(0.15, 0.85, n).astype(np.float32),
                  rgb=RNG.uniform(0, 1, (n, 3)))
    cam = seam_camera([0, 0, 0.05], 2.5, 4.0, 0.0, up_axis=2,
                      fov_deg=50, width=160, height=120)

    def affected(nx):
        tiles = grid(patch, nx, 1, 1.0, up_axis=2)
        g, _ = render_global(cam, tiles, sh_degree=0)
        t, _ = render_tiled(cam, tiles, sh_degree=0)
        d = np.abs(g - t).sum(2) / 3.0
        lit = g.sum(2) > 0.01
        return (lit & (d > 1 / 255)).sum() / max(lit.sum(), 1)

    assert affected(4) > affected(2) * 1.5


def test_seam_camera_azimuth_convention():
    """0 degrees looks along the seam (the +y axis here), 90 across it."""
    along = seam_camera([0, 0, 0], 5.0, 0.0, 0.0, up_axis=2, width=10, height=10)
    across = seam_camera([0, 0, 0], 5.0, 0.0, 90.0, up_axis=2, width=10, height=10)
    assert abs(along.position[1]) > 4.9 and abs(along.position[0]) < 0.1
    assert abs(across.position[0]) > 4.9 and abs(across.position[1]) < 0.1


# -------------------------------------------------------------- scene.py

def test_config_key_is_stable_and_discriminating():
    assert SceneConfig().key() == SceneConfig().key()
    keys = {SceneConfig().key(),
            SceneConfig(clean=True).key(),
            SceneConfig(floater_std=1.5).key(),
            SceneConfig(align=False).key(),
            SceneConfig(sh_degree=0).key()}
    assert len(keys) == 5, "settings that change output must change the key"


def test_config_from_args_round_trip():
    import argparse
    from bozkir.scene import add_scene_args
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    add_scene_args(ap)

    cfg = config_from_args(ap.parse_args(["x.ply"]))
    assert cfg.align and cfg.recentre and not cfg.clean

    cfg = config_from_args(ap.parse_args(["x.ply", "--raw"]))
    assert not cfg.align and not cfg.recentre

    cfg = config_from_args(ap.parse_args(
        ["x.ply", "--clean", "--radius-pct", "40", "--sh", "0"]))
    assert cfg.clean and cfg.radius_pct == 40.0 and cfg.sh_degree == 0


# ----------------------------------------------------------- integration

def test_align_then_render_end_to_end():
    s = _tilted_ground(40.0, 120.0)
    out, info = align_to_ground(s)
    assert info["tilt_after_deg"] < 0.5
    cam = orbit_camera([0, 0, 0], 12.0, 30.0, 20.0, up_axis=2,
                       width=120, height=90)
    img, p = render_perspective(cam, out, sh_degree=0, background=(0, 0, 0))
    assert img.shape == (90, 120, 3)
    assert p["kept"] > 1000
    assert (img.sum(axis=2) > 0.01).mean() > 0.05          # something is drawn
    assert np.isfinite(img).all()


def main():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:                              # noqa: BLE001
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            failed.append(name)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())