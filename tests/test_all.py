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
from bozkir.pack import pack, STRIDE                             # noqa: E402
from bozkir.patches import (band_stats, clip_slab, coverage,     # noqa: E402
                            pick_patches, score_patch)
from bozkir.wang import (region_weights, build_tile,             # noqa: E402
                         build_tile_set, layout, check_layout,
                         edge_gaussians)

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


# --------------------------------------------------------------- wang.py

def _exemplar(tag, size=1.0, n=20_000):
    xy = RNG.uniform(-size / 2, size / 2, (n, 2))
    xyz = np.concatenate([xy, RNG.normal(0, 0.01, (n, 1))], 1)
    dc = np.zeros((n, 3), np.float32)
    dc[:, tag % 3] = 1.0
    s = scene(xyz, opacity=np.full(n, 0.7, np.float32),
              scale=np.full((n, 3), 0.01, np.float32))
    s.sh_dc[:] = dc
    return s


def test_region_weights_partition_the_square():
    pts = RNG.uniform(-0.5, 0.5, (5000, 2))
    for blend in (0.0, 0.05, 0.2):
        w = region_weights(pts, 1.0, blend)
        assert np.allclose(w.sum(axis=1), 1.0)
        assert w.min() >= 0.0 and w.max() <= 1.0 + 1e-6


def test_region_weights_hard_cut_assigns_the_right_triangle():
    pts = np.float32([[0, 0.4], [0.4, 0], [0, -0.4], [-0.4, 0]])
    assert list(np.argmax(region_weights(pts, 1.0, 0.0), axis=1)) == [0, 1, 2, 3]


def test_tiles_sharing_an_edge_colour_have_an_identical_edge():
    """The entire reason the construction exists.

    Corners are excluded: the diagonals reach the edge there, so the last
    sliver before a corner comes from the neighbouring triangle. Cohen's
    original construction has the same gap.
    """
    h = [_exemplar(0), _exemplar(1)]
    v = [_exemplar(2), _exemplar(3)]
    tiles, codes = build_tile_set(h, v, 1.0, blend=0.0)
    assert len(tiles) == 16

    for edge, col in (("n", 0), ("e", 1), ("s", 2), ("w", 3)):
        groups = {}
        for t, c in zip(tiles, codes):
            groups.setdefault(c[col], []).append(
                edge_gaussians(t, 1.0, edge=edge))
        assert len(groups) == 2
        for g in groups.values():
            assert len(g[0]) > 100, "edge sample is too small to mean anything"
            for other in g:
                assert np.array_equal(g[0], other), edge
        reps = [g[0] for g in groups.values()]
        assert not np.array_equal(reps[0], reps[1]), f"{edge} colours identical"


def test_feathered_tiles_still_match_away_from_the_diagonals():
    h = [_exemplar(0), _exemplar(1)]
    v = [_exemplar(2), _exemplar(3)]
    tiles, codes = build_tile_set(h, v, 1.0, blend=0.05)
    groups = {}
    for t, c in zip(tiles, codes):
        groups.setdefault(c[0], []).append(
            edge_gaussians(t, 1.0, edge="n", margin=0.1))
    for g in groups.values():
        assert all(np.array_equal(g[0], x) for x in g)


def test_layout_never_places_a_mismatched_edge():
    codes = [(n, e, s, w) for n in range(2) for e in range(2)
             for s in range(2) for w in range(2)]
    for nx, ny in ((4, 4), (16, 16), (32, 32)):
        assert check_layout(codes, layout(codes, nx, ny, seed=1)) == 0


def test_layout_is_aperiodic_and_uses_the_whole_set():
    codes = [(n, e, s, w) for n in range(2) for e in range(2)
             for s in range(2) for w in range(2)]
    g = layout(codes, 32, 32, seed=1)
    assert len(np.unique(g)) == 16
    row = g[0]
    for period in range(1, 17):
        assert not np.array_equal(row[:-period], row[period:]), period
    assert not np.array_equal(g, layout(codes, 32, 32, seed=2))


# ---------------------------------------------------------- graphcut.py

def _blobs(seed, r=96):
    """A texture with structure a cut can route around."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:r, 0:r]
    img = np.zeros((r, r, 3))
    for _ in range(80):
        cx, cy = rng.uniform(0, r, 2)
        rad = rng.uniform(r * 0.03, r * 0.09)
        img[((xx - cx) ** 2 + (yy - cy) ** 2) < rad ** 2] = rng.uniform(0.25, 0.8, 3)
    return np.clip(img + rng.normal(0, 0.03, (r, r, 3)) + 0.25, 0, 1)


def test_two_label_cut_respects_its_constraints():
    from bozkir.graphcut import two_label_cut
    r = 64
    a, b = _blobs(2, r), _blobs(7, r)
    take_a = np.zeros((r, r), bool); take_a[:, :3] = True
    take_b = np.zeros((r, r), bool); take_b[:, -3:] = True
    use_b = two_label_cut(a, b, take_a, take_b)
    assert not use_b[:, :3].any()
    assert use_b[:, -3:].all()


def test_two_label_cut_beats_a_straight_one():
    from bozkir.graphcut import two_label_cut, cut_cost
    r = 96
    a, b = _blobs(2, r), _blobs(7, r)
    take_a = np.zeros((r, r), bool); take_a[:, :3] = True
    take_b = np.zeros((r, r), bool); take_b[:, -3:] = True
    use_b = two_label_cut(a, b, take_a, take_b)
    straight = np.zeros((r, r), bool); straight[:, r // 2:] = True
    assert cut_cost(a, b, use_b) < 0.8 * cut_cost(a, b, straight)


def test_cut_routes_around_a_feature():
    """The point of the method: go round the stone, not through it."""
    from bozkir.graphcut import two_label_cut
    r = 96
    yy, xx = np.mgrid[0:r, 0:r]
    base = RNG.random((r, r, 3)) * 0.15 + 0.35
    a, b = base.copy(), base.copy()
    disc = ((xx - r * 0.5) ** 2 + (yy - r * 0.5) ** 2) < (r * 0.16) ** 2
    b[disc] = [0.95, 0.9, 0.2]
    take_a = np.zeros((r, r), bool); take_a[:, :3] = True
    take_b = np.zeros((r, r), bool); take_b[:, -3:] = True
    use_b = two_label_cut(a, b, take_a, take_b)

    def boundary(m):
        e = np.zeros_like(m)
        e[:, :-1] |= m[:, :-1] != m[:, 1:]
        e[:-1, :] |= m[:-1, :] != m[1:, :]
        return e

    assert (boundary(use_b) & disc).sum() == 0


def test_tile_labels_keep_the_edges_pure():
    """A cut leaking into an edge strip would break tile matching."""
    from bozkir.graphcut import tile_labels, edge_purity
    labels = tile_labels([_blobs(s, 96) for s in (2, 7, 11, 19)],
                         size=1.0, band=0.14)
    for name, purity in edge_purity(labels).items():
        assert purity == 1.0, (name, purity)
    assert set(np.unique(labels)) == {0, 1, 2, 3}


def test_cutting_at_the_square_leaves_nothing_overlapping():
    """A Gaussian in the neighbour's half is a strip both tiles draw.

    Reaching over on two sides does not help - the tile that reaches still
    lands on ground its neighbour covers. Only cutting at the square leaves
    nothing shared, and it costs almost no coverage because Gaussians still
    spread across the join from their own side.
    """
    from bozkir.graphcut import render_patch
    from bozkir.tile import translate, merge
    size = 1.5

    def over(seed, n=5000):
        r = np.random.default_rng(seed)
        xy = r.uniform(-size / 2 * 1.35, size / 2 * 1.35, (n, 2))
        s = scene(np.concatenate([xy, r.normal(0, 0.02, (n, 1))], 1),
                  scale=np.exp(r.normal(-4.6, 0.5, (n, 3))).astype(np.float32),
                  opacity=np.full(n, 0.8, np.float32))
        s.sh_dc[:] = r.normal(0, 0.4, (n, 3))
        return s

    tiles, codes = build_tile_set([over(1), over(2)], [over(3), over(4)],
                                  size, cut=True, resolution=48, band=0.14)
    h = size / 2

    def trim(t):
        return t.subset(np.abs(t.xyz[:, :2]).max(axis=1) <= h)

    cut = [trim(t) for t in tiles]

    # Two tiles side by side: neither may put a Gaussian in the other's half.
    a, b = cut[0], translate(cut[1], [size, 0, 0])
    assert not (a.xyz[:, 0] > h + 1e-6).any()
    assert not (b.xyz[:, 0] < h - 1e-6).any()

    # And the join has to be as well covered as it would be if both tiles
    # reached over it. An absolute threshold would only measure how dense
    # the test patches happen to be.
    def join_cover(left, right):
        m = merge(left, translate(right, [size, 0, 0]))
        band = m.subset(np.abs(m.xyz[:, 0] - h) < 0.12)
        band = band.subset(np.arange(len(band)))
        band.xyz = (band.xyz - np.float32([h, 0, 0])).astype(np.float32)
        return float(render_patch(band, 0.24, 64)[1].mean())

    def both_sides(t):
        allow = 2.5 * t.scale.max(axis=1)
        return t.subset((np.abs(t.xyz[:, 0]) <= h + allow)
                        & (np.abs(t.xyz[:, 1]) <= h + allow))

    # How much coverage the join loses depends on how dense the tiles are:
    # at the ~50k a real tile carries it is about a tenth of a percent, but
    # these test tiles are sparse enough that the same cut shows up larger.
    one = join_cover(cut[0], cut[1])
    two = join_cover(both_sides(tiles[0]), both_sides(tiles[1]))
    assert one > two - 0.15, (one, two)

    # And the overlapping version really does share ground, which is the
    # thing being avoided.
    ov = both_sides(tiles[0])
    assert (np.abs(ov.xyz[:, 0]) > h).any()

    for edge, col in (("n", 0), ("e", 1), ("s", 2), ("w", 3)):
        groups = {}
        for t, c in zip(cut, codes):
            groups.setdefault(c[col], []).append(
                edge_gaussians(t, size, edge=edge, margin=0.03))
        for g in groups.values():
            for other in g:
                assert np.array_equal(g[0], other), edge


def test_per_gaussian_overhang_beats_a_flat_distance():
    """Sizes have a long tail, and the big ones cover the most ground.

    One distance for everything keeps the small Gaussians that barely
    matter and cuts the large ones that do.
    """
    from bozkir.graphcut import render_patch
    size = 1.5

    def over(seed, n=4000):
        r = np.random.default_rng(seed)
        xy = r.uniform(-size / 2 * 1.35, size / 2 * 1.35, (n, 2))
        s = scene(np.concatenate([xy, r.normal(0, 0.02, (n, 1))], 1),
                  scale=np.exp(r.normal(-5.2, 0.8, (n, 3))).astype(np.float32),
                  opacity=np.full(n, 0.8, np.float32))
        s.sh_dc[:] = r.normal(0, 0.4, (n, 3))
        return s

    tiles, _ = build_tile_set([over(1), over(2)], [over(3), over(4)],
                              size, cut=True, resolution=48, band=0.14)
    t0 = tiles[0]

    def cover(keep):
        a = render_patch(t0.subset(keep), size, 96)[1]
        return float(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]).mean())

    everything = cover(np.ones(len(t0), bool))
    out = np.abs(t0.xyz[:, :2]).max(axis=1) - size / 2
    per_splat = cover(out <= 2.5 * t0.scale.max(axis=1))
    flat = cover(np.all(np.abs(t0.xyz[:, :2]) <= size * 0.509, axis=1))

    assert per_splat >= flat, (per_splat, flat)
    assert everything - per_splat < 0.02, (everything, per_splat)
    # and it must actually be cheaper than keeping the lot
    assert (out <= 2.5 * t0.scale.max(axis=1)).mean() < 0.9


def test_overhang_needed_is_set_by_splat_size_not_tile_size():
    """Coverage saturates after a couple of splat widths; overlap does not.

    Too little and the boundary strip is bare. Too much and both neighbours
    draw it, then swap which is in front as the camera turns.
    """
    from bozkir.graphcut import render_patch
    size = 1.5

    def over(seed, n=4000):
        r = np.random.default_rng(seed)
        xy = r.uniform(-size / 2 * 1.35, size / 2 * 1.35, (n, 2))
        s = scene(np.concatenate([xy, r.normal(0, 0.02, (n, 1))], 1),
                  scale=np.full((n, 3), 0.010, np.float32),
                  opacity=np.full(n, 0.8, np.float32))
        s.sh_dc[:] = r.normal(0, 0.4, (n, 3))
        return s

    tiles, _ = build_tile_set([over(1), over(2)], [over(3), over(4)],
                              size, cut=True, resolution=48, band=0.14)
    w = float(np.median(np.concatenate([t.scale.max(1) for t in tiles])))

    def edge_cover(frac):
        reach = size * (0.5 + frac)
        t = tiles[0].subset(np.all(np.abs(tiles[0].xyz[:, :2]) <= reach, axis=1))
        a = render_patch(t, size, 96)[1]
        return float(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]).mean())

    bare = edge_cover(0.0)
    enough = edge_cover(2.5 * w / size)
    lavish = edge_cover(0.30)
    assert enough > bare, (bare, enough)
    # Past a couple of splat widths there is nothing more to gain.
    assert abs(lavish - enough) < 0.02, (enough, lavish)


def test_trimming_overhang_keeps_edges_matching():
    """A finished tile needs some overhang to cover the seam, not all of it.

    Keeping the whole wide cut means both neighbours draw the same boundary
    strip and fight over which is in front as the camera moves.
    """
    size = 1.5

    def over(seed, n=5000):
        r = np.random.default_rng(seed)
        xy = r.uniform(-size / 2 * 1.35, size / 2 * 1.35, (n, 2))
        s = scene(np.concatenate([xy, r.normal(0, 0.02, (n, 1))], 1),
                  scale=np.full((n, 3), 0.008, np.float32),
                  opacity=np.full(n, 0.7, np.float32))
        s.sh_dc[:] = r.normal(0, 0.4, (n, 3))
        return s

    tiles, codes = build_tile_set([over(1), over(2)], [over(3), over(4)],
                                  size, cut=True, resolution=48, band=0.14)
    full = np.mean([len(t) for t in tiles])
    for overhang in (0.30, 0.06, 0.0):
        reach = size * (0.5 + overhang)
        cut = [t.subset(np.all(np.abs(t.xyz[:, :2]) <= reach, axis=1))
               for t in tiles]
        for edge, col in (("n", 0), ("e", 1), ("s", 2), ("w", 3)):
            groups = {}
            for t, c in zip(cut, codes):
                groups.setdefault(c[col], []).append(
                    edge_gaussians(t, size, edge=edge, margin=0.03))
            for g in groups.values():
                for other in g:
                    assert np.array_equal(g[0], other), (overhang, edge)
    tight = np.mean([len(t.subset(
        np.all(np.abs(t.xyz[:, :2]) <= size * 0.56, axis=1))) for t in tiles])
    assert tight < full, "trimming should actually remove something"


def test_overflowing_patches_do_not_break_edge_matching():
    """Levelling rotates a patch, so it no longer fits the tile square.

    Clamping the overflow to the nearest border pixel silently breaks
    matching: that pixel's label differs between tiles, so the same
    Gaussian is kept in one and dropped in another.
    """
    size = 1.5

    def over(seed, factor, n=6000):
        r = np.random.default_rng(seed)
        xy = r.uniform(-size / 2 * factor, size / 2 * factor, (n, 2))
        s = scene(np.concatenate([xy, r.normal(0, 0.03, (n, 1))], 1),
                  scale=np.full((n, 3), 0.006, np.float32),
                  opacity=np.full(n, 0.7, np.float32))
        s.sh_dc[:] = r.normal(0, 0.4, (n, 3))
        return s

    for factor in (1.0, 1.15, 1.4):
        h = [over(1, factor), over(2, factor)]
        v = [over(3, factor), over(4, factor)]
        tiles, codes = build_tile_set(h, v, size, cut=True, resolution=64,
                                      band=0.14)
        for edge, col in (("n", 0), ("e", 1), ("s", 2), ("w", 3)):
            groups = {}
            for t, c in zip(tiles, codes):
                groups.setdefault(c[col], []).append(
                    edge_gaussians(t, size, edge=edge, margin=0.03))
            for g in groups.values():
                for other in g:
                    assert np.array_equal(g[0], other), (factor, edge)

        # The overhang has to survive, or every edge gets a bare strip.
        beyond = (np.abs(tiles[0].xyz[:, :2]).max(axis=1) > size / 2).mean()
        if factor > 1.0:
            assert beyond > 0.02, (factor, beyond)


def test_graph_cut_tiles_still_match_at_their_edges():
    from bozkir.graphcut import render_patch  # noqa: F401
    h = [_exemplar(0, n=8000), _exemplar(1, n=8000)]
    v = [_exemplar(2, n=8000), _exemplar(3, n=8000)]
    tiles, codes = build_tile_set(h, v, 1.0, cut=True, resolution=64, band=0.14)
    assert len(tiles) == 16
    for edge, col in (("n", 0), ("e", 1), ("s", 2), ("w", 3)):
        groups = {}
        for t, c in zip(tiles, codes):
            groups.setdefault(c[col], []).append(
                edge_gaussians(t, 1.0, edge=edge))
        for g in groups.values():
            for other in g:
                assert np.array_equal(g[0], other), edge


# ---------------------------------------------------------------- pack.py

def test_stratified_keep_respects_its_budget_and_spreads():
    """The splat cap has to actually cap, and not pile up in one corner."""
    from bozkir.patches import stratified_keep
    size, n = 2.0, 60_000
    xy = RNG.uniform(-size / 2, size / 2, (n, 2))
    s = scene(np.concatenate([xy, RNG.normal(0, 0.02, (n, 1))], 1),
              scale=np.exp(RNG.normal(-4, 0.7, (n, 3))).astype(np.float32),
              opacity=RNG.uniform(0.2, 0.9, n).astype(np.float32))
    for budget in (50_000, 10_000, 2_000):
        keep = stratified_keep(s, budget, size)
        assert len(keep) <= budget
        q = s.subset(keep)
        g = 12
        ij = np.clip(((q.xyz[:, :2] + size / 2) / size * g).astype(int), 0, g - 1)
        filled = len(np.unique(ij[:, 0] * g + ij[:, 1])) / (g * g)
        assert filled > 0.9, (budget, filled)



def test_splat_pack_round_trips():
    """Positions and scales exact; colour and rotation within quantisation."""
    n = 500
    q = RNG.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    s = scene(RNG.normal(0, 3, (n, 3)),
              opacity=RNG.random(n).astype(np.float32),
              scale=np.exp(RNG.normal(-3, 1, (n, 3))).astype(np.float32),
              rot=q)
    s.sh_dc[:] = RNG.normal(0, 0.5, (n, 3))

    buf = pack(s).reshape(-1, STRIDE)
    assert buf.shape == (n, 32)

    pos = buf[:, 0:12].copy().view(np.float32).reshape(-1, 3)
    scl = buf[:, 12:24].copy().view(np.float32).reshape(-1, 3)
    assert np.array_equal(pos, s.xyz)
    assert np.array_equal(scl, s.scale)

    rgba = buf[:, 24:28].astype(np.float32) / 255.0
    assert np.abs(rgba[:, :3] - s.base_rgb).max() < 1.5 / 255
    assert np.abs(rgba[:, 3] - s.opacity).max() < 1.5 / 255

    rot = (buf[:, 28:32].astype(np.float32) - 128.0) / 128.0
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)
    dots = np.abs((rot * s.rot).sum(axis=1))       # q and -q are one rotation
    assert np.degrees(2 * np.arccos(np.clip(dots, 0, 1))).max() < 1.5


def test_subsampled_coverage_matches_a_full_render():
    """Rendering a subset and undoing the density has to give the same answer.

    Per pixel, not overall: correcting the average instead reports a
    half-empty patch as full, which is the case the check exists for.
    """
    from bozkir.patches import rendered_coverage
    size = 1.5
    for kind in ("full", "half"):
        n = 60_000
        xy = RNG.uniform(-size / 2, size / 2, (n, 2))
        if kind == "half":
            xy[:, 0] = np.abs(xy[:, 0]) - size / 2
        s = scene(np.concatenate([xy, RNG.normal(0, 0.02, (n, 1))], 1),
                  opacity=RNG.uniform(0.3, 0.9, n).astype(np.float32),
                  scale=np.exp(RNG.normal(-4.3, 0.6, (n, 3))).astype(np.float32))
        full = rendered_coverage(s, size, cap=n + 1)
        fast = rendered_coverage(s, size, cap=8000)
        assert abs(full - fast) < 0.05, (kind, full, fast)
    assert rendered_coverage(scene(np.zeros((0, 3))), size) == 0.0


def test_estimating_coverage_first_picks_the_same_patches():
    """The fast path has to agree with the slow one, or it is not a shortcut."""
    from bozkir.patches import pick_patches
    size, n = 1.5, 120_000
    xy = RNG.uniform(-5, 5, (n, 2))
    hole = np.zeros(len(xy), bool)
    for _ in range(10):
        c = RNG.uniform(-5, 5, 2)
        hole |= np.linalg.norm(xy - c, axis=1) < 1.0
    xy = xy[~hole]
    m = len(xy)
    s = scene(np.concatenate([xy, RNG.normal(0, 0.03, (m, 1))], 1),
              opacity=RNG.uniform(0.3, 0.9, m).astype(np.float32),
              scale=np.exp(RNG.normal(-4.3, 0.6, (m, 3))).astype(np.float32))

    def centres(margin):
        got = pick_patches(s, size, 5, 2, stride=0.6, thickness=0.4,
                           max_tilt=25, min_separation=0.5, min_cover=0.8,
                           cover_margin=margin, verbose=False)
        return [(round(x, 3), round(y, 3)) for _, (x, y), _, _ in got]

    assert centres(99.0) == centres(0.10)


# ------------------------------------------------------- feature tiles

def test_rim_tilt_does_not_depend_on_slab_thickness():
    """Fitting a plane to the whole rim band measures a shell, not ground.

    Once the slab is thick enough to hold anything standing up, that fit
    returns a near-random angle and every candidate gets rejected.
    """
    from bozkir.patches import band_stats, clip_slab
    from bozkir.tile import extract_patch

    size, n = 1.5, 120_000
    xyz = np.stack([RNG.uniform(-5, 5, n), RNG.uniform(-5, 5, n),
                    RNG.normal(0, 0.03, n)], 1)
    s = scene(xyz, scale=np.full((n, 3), 0.01, np.float32),
              opacity=np.full(n, 0.7, np.float32))

    tilts = []
    for thickness in (0.3, 0.6, 1.0, 1.5):
        p = extract_patch(s, [3.0, 3.0], size, up_axis=2)
        p, _ = clip_slab(p, 2, thickness)
        tilts.append(band_stats(p, 2, size)[1])
    assert max(tilts) < 5.0, tilts


def test_coverage_separates_a_solid_patch_from_a_holed_one():
    """Bin occupancy calls a half-empty patch full; area coverage does not."""
    size, n = 1.5, 40_000
    solid = scene(np.concatenate(
        [RNG.uniform(-size / 2, size / 2, (n, 2)),
         RNG.normal(0, 0.02, (n, 1))], 1),
        scale=np.full((n, 3), 0.02, np.float32),
        opacity=np.full(n, 0.8, np.float32))
    # Same splats, same density, half the area.
    xy = RNG.uniform(-size / 2, size / 2, (n, 2))
    xy[:, 0] = np.abs(xy[:, 0]) - size / 2
    holed = scene(np.concatenate([xy, RNG.normal(0, 0.02, (n, 1))], 1),
                  scale=np.full((n, 3), 0.02, np.float32),
                  opacity=np.full(n, 0.8, np.float32))

    assert coverage(solid, size) > 0.95
    assert coverage(holed, size) < coverage(solid, size)
    assert coverage(scene(np.zeros((0, 3))), size) == 0.0


def test_feature_scoring_prefers_something_in_the_middle():
    from bozkir.patches import score_patch
    from bozkir.tile import extract_patch

    size, n = 1.5, 200_000
    ground = np.stack([RNG.uniform(-5, 5, n), RNG.uniform(-5, 5, n),
                       RNG.normal(0, 0.02, n)], 1)
    m = 30_000
    a = RNG.uniform(0, 2 * np.pi, m)
    rad = RNG.uniform(0, 1, m) ** 0.5 * 0.45
    rock = np.stack([rad * np.cos(a), rad * np.sin(a),
                     RNG.uniform(0, 0.5, m)], 1)
    s = scene(np.vstack([ground, rock]),
              scale=np.full((n + m, 3), 0.01, np.float32),
              opacity=np.full(n + m, 0.7, np.float32))

    bare = extract_patch(s, [3.5, 3.5], size, up_axis=2)
    middle = extract_patch(s, [0.0, 0.0], size, up_axis=2)
    rim = extract_patch(s, [size / 2, 0.0], size, up_axis=2)

    flat_bare = score_patch(bare, 2, size, features=False)[0]
    flat_mid = score_patch(middle, 2, size, features=False)[0]
    feat_bare = score_patch(bare, 2, size, features=True)[0]
    feat_mid = score_patch(middle, 2, size, features=True)[0]
    feat_rim = score_patch(rim, 2, size, features=True)[0]

    assert flat_bare > flat_mid, "flat scoring should prefer bare ground"
    assert feat_mid > feat_bare, "feature scoring should prefer the object"
    assert feat_mid > feat_rim, "an object on the rim must not win"


def test_axis_split_keeps_both_directions_alike():
    """One set of patches runs north-south, the other east-west.

    Splitting by score puts the odd one out on a single axis, so every
    boundary running that way is made of different material from the ones
    across it - a grain in the grid that shows as the scene changing
    character each quarter turn.
    """
    from bozkir.patches import balanced_split, appearance

    def flat(mean, n=1500):
        r = np.random.default_rng(int(mean[0] * 1000))
        s = scene(r.uniform(-0.75, 0.75, (n, 3)),
                  scale=np.full((n, 3), 0.01, np.float32),
                  opacity=np.full(n, 0.7, np.float32))
        s.sh_dc[:] = (np.float32(mean) + r.normal(0, 0.01, (n, 3)) - 0.5) / SH_C0
        return s

    # three alike and one clearly darker, as a real scene gives
    ps = [flat([0.36, 0.36, 0.35]), flat([0.35, 0.36, 0.34]),
          flat([0.35, 0.35, 0.33]), flat([0.31, 0.32, 0.31])]
    f = [appearance(p) for p in ps]
    by_score = float(np.linalg.norm(np.mean(f[:2], axis=0)
                                    - np.mean(f[2:], axis=0)))
    (h, v), gap = balanced_split(ps, 2)

    assert len(h) == 2 and len(v) == 2
    assert {id(x) for x in h} | {id(x) for x in v} == {id(x) for x in ps}
    assert gap < by_score, (gap, by_score)

    # already-alike patches should not be made worse
    same = [flat([0.35, 0.35, 0.35]) for _ in range(4)]
    _, g2 = balanced_split(same, 2)
    assert g2 < 0.02


# --------------------------------------------------------------- presets

def test_presets_save_replay_and_yield_to_explicit_flags():
    import argparse
    import json
    import tempfile
    from bozkir.presets import add_preset_args, apply as apply_preset

    def parser():
        ap = argparse.ArgumentParser()
        ap.add_argument("path", type=Path)
        ap.add_argument("--size", type=float, default=1.5)
        ap.add_argument("--cut", action="store_true")
        ap.add_argument("--lod", type=int, default=4)
        add_preset_args(ap)
        return ap

    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "p.json"
        apply_preset(parser(), ["x.ply", "--size", "2.5", "--cut",
                                "--lod", "3", "--save-preset", "demo"], path=f)
        assert json.loads(f.read_text())["demo"]["size"] == 2.5

        a = apply_preset(parser(), ["y.ply", "--preset", "demo"], path=f)
        assert a.size == 2.5 and a.cut and a.lod == 3

        # An explicit flag has to beat the preset, or overriding is impossible.
        b = apply_preset(parser(), ["y.ply", "--preset", "demo",
                                    "--size", "9.0"], path=f)
        assert b.size == 9.0

        try:
            apply_preset(parser(), ["y.ply", "--preset", "nope"], path=f)
            raise AssertionError("unknown preset should fail")
        except SystemExit:
            pass


def test_preset_keys_a_script_lacks_are_skipped():
    """One preset per scene has to work with every script."""
    import argparse
    import json
    import tempfile
    from bozkir.presets import add_preset_args, apply as apply_preset

    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "p.json"
        f.write_text(json.dumps({"s": {"size": 3.0, "cut": True,
                                       "not_an_option_here": 7}}))
        ap = argparse.ArgumentParser()
        ap.add_argument("path", type=Path)
        ap.add_argument("--size", type=float, default=1.5)
        add_preset_args(ap)
        a = apply_preset(ap, ["x.ply", "--preset", "s"], path=f)
        assert a.size == 3.0


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


def test_every_script_declares_the_flags_it_reads():
    """Catch args.something with no matching add_argument.

    A helper gains a parameter, the callers gain a flag, and one script gets
    missed. Nothing notices until that script is run with the right options.
    """
    import ast as _ast
    root = Path(__file__).resolve().parents[1]
    problems = []
    for path in sorted((root / "scripts").glob("*.py")):
        tree = _ast.parse(path.read_text(encoding="utf-8"))
        declared = set()
        for node in _ast.walk(tree):
            if (isinstance(node, _ast.Call)
                    and isinstance(node.func, _ast.Attribute)
                    and node.func.attr == "add_argument"):
                for a in node.args:
                    if isinstance(a, _ast.Constant) and isinstance(a.value, str):
                        declared.add(a.value.lstrip("-").replace("-", "_"))
                for kw in node.keywords:
                    if kw.arg == "dest" and isinstance(kw.value, _ast.Constant):
                        declared.add(kw.value.value)
        # Flags the shared helpers attach.
        declared |= {"raw", "clean", "radius_pct", "floater_std",
                     "max_extent_pct", "sh", "up_axis", "flip", "no_cache",
                     "preset", "save_preset", "list_presets"}

        used = {n.attr for n in _ast.walk(tree)
                if isinstance(n, _ast.Attribute)
                and isinstance(n.value, _ast.Name) and n.value.id == "args"}
        missing = sorted(used - declared)
        if missing:
            problems.append(f"{path.name}: {', '.join(missing)}")
    assert not problems, "flags read but never declared -> " + "; ".join(problems)


def test_every_script_builds_its_parser():
    """Run each script with --help.

    Importing a module does not build its argument parser, so an import
    check misses a flag declared twice, a bad default, or a helper that
    attaches an option something else already added. Those only surface
    when the script is actually run.
    """
    import subprocess
    root = Path(__file__).resolve().parents[1]
    failures = []
    for path in sorted((root / "scripts").glob("*.py")):
        r = subprocess.run([sys.executable, str(path), "--help"],
                           cwd=root, capture_output=True, text=True,
                           timeout=120)
        if r.returncode != 0:
            tail = (r.stderr or r.stdout).strip().splitlines()[-1:]
            failures.append(f"{path.name}: {' '.join(tail)}")
    assert not failures, "; ".join(failures)


def test_every_script_imports():
    """Catch a script importing a name its module does not define.

    Modules and the scripts that use them drift apart: a helper gets moved
    or renamed on one side only, and nothing notices until the script is
    run. Importing each one here fails immediately instead.
    """
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    scripts = sorted((root / "scripts").glob("*.py"))
    assert scripts, "no scripts found"
    sys.path.insert(0, str(root / "scripts"))
    for path in scripts:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            pass                      # argparse in a __main__ guard is fine
        except ImportError as e:
            raise AssertionError(f"{path.name}: {e}") from e


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