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

import bozkir.cut  # noqa: E402
import bozkir.holes  # noqa: E402
import bozkir.materials  # noqa: E402
import bozkir.pick  # noqa: E402
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
from bozkir.wang import (region_weights,             # noqa: E402
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


def _source_patch(cx, cy, size=1.0, n=12_000):
    """A patch cut from one big ground at (cx, cy), each splat carrying
    where it lay in the ground in its colour (sh_dc x, y)."""
    xy = RNG.uniform(-size / 2, size / 2, (n, 2))
    s = scene(np.concatenate([xy, np.zeros((n, 1))], 1).astype(np.float32),
              opacity=np.full(n, 0.7, np.float32), scale=np.full((n, 3), 0.01, np.float32))
    s.sh_dc[:, 0], s.sh_dc[:, 1] = xy[:, 0] + cx, xy[:, 1] + cy
    return s


def _gap_across_edges(tiles, codes, size):
    """Median distance in the ground between splats just below a north
    edge and just above it, in the neighbour with that south colour: how
    far apart the ground on the two sides of an edge came from."""
    from scipy.spatial import cKDTree
    gaps = []
    for c in {k[0] for k in codes}:
        lo = tiles[next(i for i, k in enumerate(codes) if k[0] == c)]
        hi = tiles[next(i for i, k in enumerate(codes) if k[2] == c)]
        y_lo, y_hi = lo.xyz[:, 1] - size / 2, hi.xyz[:, 1] + size / 2
        a = lo.sh_dc[(y_lo > -0.03) & (y_lo <= 0) & (np.abs(lo.xyz[:, 0]) < 0.3)][:, :2]
        b = hi.sh_dc[(y_hi < 0.03) & (y_hi >= 0) & (np.abs(hi.xyz[:, 0]) < 0.3)][:, :2]
        gaps.append(np.median(cKDTree(a).query(b)[0]))
    return float(np.median(gaps))


def test_diamonds_let_the_ground_run_on_across_an_edge():
    """Cohen's diamond sits on its edge: the ground just above an edge
    continues the ground just below it. Centred on the tile instead, the
    two sides are a tile apart in the ground (the bug --diamonds fixes)."""
    h = [_source_patch(0, 0), _source_patch(5, 0)]
    v = [_source_patch(0, 5), _source_patch(5, 5)]
    for cut in (False, True):
        old, _ = build_tile_set(h, v, 1.0, cut=cut, resolution=64)
        new, codes = build_tile_set(h, v, 1.0, cut=cut, resolution=64, diamonds=True)
        assert _gap_across_edges(old, codes, 1.0) > 0.8, cut
        assert _gap_across_edges(new, codes, 1.0) < 0.03, cut
        for edge, col in (("n", 0), ("e", 1), ("s", 2), ("w", 3)):
            groups = {}
            for t, c in zip(new, codes):
                groups.setdefault(c[col], []).append(edge_gaussians(t, 1.0, edge=edge))
            for g in groups.values():
                assert all(np.array_equal(g[0], o) for o in g), (cut, edge)


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


def test_presets_stack():
    """A preset names its base and lists only changes; --preset a,b puts
    b over a; the command line still wins."""
    import argparse
    import json
    import tempfile
    from bozkir.presets import add_preset_args, apply as apply_preset

    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "p.json"
        f.write_text(json.dumps({
            "base": {"size": 1.5, "lod": 4, "cut": True},
            "child": {"base": "base", "lod": 3},
            "extra": {"cut": False},
            "loop": {"base": "loop"}}))

        def run(*argv):
            ap = argparse.ArgumentParser()
            ap.add_argument("path")
            ap.add_argument("--size", type=float, default=1.0)
            ap.add_argument("--lod", type=int, default=1)
            ap.add_argument("--cut", action="store_true")
            add_preset_args(ap, path=f)
            return apply_preset(ap, ["x.ply", *argv], path=f)

        a = run("--preset", "child")
        assert (a.size, a.lod, a.cut) == (1.5, 3, True)
        b = run("--preset", "child,extra")
        assert (b.size, b.lod, b.cut) == (1.5, 3, False)
        c = run("--preset", "child,extra", "--lod", "2")
        assert c.lod == 2
        try:
            run("--preset", "loop")
            raise AssertionError("a preset that is its own base should fail")
        except SystemExit:
            pass


def test_ground_is_the_published_method():
    """The tracked file holds one method, flat, that any capture can use."""
    from bozkir.presets import PRESETS, load_all
    p = load_all(PRESETS)
    assert set(p) == {"ground"}
    g = p["ground"]
    assert "base" not in g and g["classes"] == "auto" and g["class_scale"] == "auto"
    assert g["frames"] and g["level_tone"] and g["diamonds"] and g["pure_share"] > 0
    assert g["centre_shape"] > 2 and g["centre_size"] == 0 and 0 < g["pool_share"] < 1


def test_desert4_is_desert3_plus_todays_flags():
    """desert3 stays as it was, so v1 rebuilds exactly. These presets live
    in the working copy's local file; without it there is nothing to check."""
    from bozkir.presets import LOCAL, load_all, resolve
    if not LOCAL.exists():
        return
    p = load_all()
    d3, d4 = resolve(p, ["desert3"]), resolve(p, ["desert4"])
    assert "base" not in d4
    assert {k: d4[k] for k in d3} == d3
    assert d4["odd_spots"] > 0 and d4["decal_auto"] > 0
    # desert6: both materials, more and varied sand patches, decals off
    # until markers are told apart on the real capture.
    d6 = resolve(p, ["desert6"])
    assert d6["classes"] == 2 and d6["varied"]
    assert d6["decal_auto"] == 0
    # desert7: the missing layer instead of fill opacity, decals back.
    d7 = resolve(p, ["desert7"])
    assert d7["missing_layer"] and d7["fill_opacity"] == 0 and d7["decal_auto"] > 0
    # desert8: dense and alike (alike hides the triangles), no missing
    # layer (seen from the side it showed the capture's haze as sheets).
    d8 = resolve(p, ["desert8"])
    assert d8["dense"] and not d8["varied"] and not d8["missing_layer"]
    assert d8["fill_opacity"] > 0 and d8["decal_auto"] > 0
    # desert9: the capture's own materials in its own shares, centre patches.
    d9 = resolve(p, ["desert9"])
    assert d9["classes"] == 3 and d9["min_share"] > 0 and d9["centres"] and d9["cut"]
    assert len(d9["class_rotations"].split(",")) == d9["classes"]
    assert d9["flatten_bare"] > 0 and not d9["flatten_classes"]
    # desert10: the count chosen with reasons, the haze weighed down.
    d10 = resolve(p, ["desert10"])
    assert d10["classes"] == "auto" and 0 < d10["quality_weight"] <= 1
    assert 1 < d10["faint_fill"] <= 6                      # gentle, under the fill's own 6
    assert d10["tone_order"] and d10["seen_shares"] and d10["decal_class"] == -1
    # ground: everything decided by the capture, one preset for any of them.
    g = p["ground11"]
    assert "base" not in g and g["class_scale"] == "auto" and g["pure_share"] > 0
    assert g["classes"] == "auto" and resolve(p, ["desert11"]) == g
    # desert12: the same, with frames and levels true to their tiles.
    d12 = resolve(p, ["desert12"])
    assert d12["frames"] and d12["level_tone"] and not g.get("frames")
    assert {k: v for k, v in d12.items() if k not in ("frames", "level_tone")} == g
    # desert13: edge patches on their edges, centres as large rounded squares.
    d13 = resolve(p, ["ground"])
    assert d13["diamonds"] and d13["centre_shape"] > 2 and d13["centre_size"] == 0
    assert 0 < d13["pool_share"] < 1
    assert {k: v for k, v in d13.items()
            if k not in ("diamonds", "centre_shape", "centre_size", "pool_share",
                         "height_order", "pool_sharpness", "fill_spare_haze")} == d12


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
    for path in sorted((root / "scripts").rglob("*.py")):
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
                     "min_splat", "max_height",
                     "preset", "save_preset", "list_presets"}

        used = {n.attr for n in _ast.walk(tree)
                if isinstance(n, _ast.Attribute)
                and isinstance(n.value, _ast.Name) and n.value.id == "args"}
        missing = sorted(used - declared)
        if missing:
            problems.append(f"{path.name}: {', '.join(missing)}")
    assert not problems, "flags read but never declared -> " + "; ".join(problems)


def test_min_size_widens_thin_splats_only():
    from bozkir.select import min_size
    from bozkir.ply import Splats
    n = 3
    s = Splats(xyz=np.zeros((n, 3), np.float32), opacity=np.full(n, 0.5, np.float32),
               scale=np.float32([[0.003, 0.02, 0.0], [0.1, 0.1, 0.1], [0.0, 0.0, 0.0]]),
               rot=np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
               sh_dc=np.zeros((n, 3), np.float32),
               sh_rest=np.zeros((n, 0, 3), np.float32), sh_degree=0)
    out = min_size(s, 0.01)
    assert np.all(out.scale >= 0.01 - 1e-7)
    assert abs(out.scale[1, 0] - 0.1) < 0.001          # wide splats barely change
    assert np.allclose(out.opacity, s.opacity)          # nothing else touched
    assert np.allclose(s.scale[0, 0], 0.003)            # the input is not modified


def test_cap_height_keeps_the_view_from_above():
    from bozkir.select import cap_height, _matrix_to_quat
    from bozkir.ply import Splats, quat_to_matrix
    r = np.random.default_rng(0)
    n = 200
    q = r.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    # quaternion -> matrix -> quaternion round trip (up to sign)
    back = _matrix_to_quat(quat_to_matrix(q))
    assert np.allclose(np.abs((back * q).sum(axis=1)), 1.0, atol=1e-5)
    s = Splats(xyz=np.zeros((n, 3), np.float32), opacity=np.full(n, 0.5, np.float32),
               scale=r.uniform(0.002, 0.2, (n, 3)).astype(np.float32), rot=q,
               sh_dc=np.zeros((n, 3), np.float32),
               sh_rest=np.zeros((n, 0, 3), np.float32), sh_degree=0)
    out = cap_height(s, 0.01)
    a, b = s.covariance(), out.covariance()
    assert np.allclose(a[:, :2, :2], b[:, :2, :2], atol=1e-6)     # same from above
    assert np.all(b[:, 2, 2] <= 0.01 ** 2 + 1e-7)                   # no taller than 1 cm
    low = a[:, 2, 2] <= 0.01 ** 2
    assert np.allclose(a[low], b[low], atol=1e-7)                   # short ones untouched
    assert np.allclose(out.opacity, s.opacity)


def test_fill_opacity_makes_a_thin_layer_solid_without_adding_splats():
    from bozkir.tile import fill_opacity
    from bozkir.graphcut import render_patch
    from bozkir.ply import Splats
    r = np.random.default_rng(0)
    n = 3000
    t = Splats(xyz=np.c_[r.random((n, 2)) * 1.5 - 0.75, np.zeros(n)].astype(np.float32),
               opacity=np.full(n, 0.15, np.float32),        # a see-through layer
               scale=np.tile(np.float32([0.03, 0.03, 0.003]), (n, 1)),
               rot=np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
               sh_dc=np.zeros((n, 3), np.float32),
               sh_rest=np.zeros((n, 0, 3), np.float32), sh_degree=0)
    _, a0 = render_patch(t, 1.5, 48)
    f = fill_opacity(t, 1.5)
    _, a1 = render_patch(f, 1.5, 48)
    assert a0.mean() < 0.85 and a1.mean() > 0.95, (a0.mean(), a1.mean())
    assert len(f) == len(t) and np.array_equal(f.xyz, t.xyz)
    assert np.array_equal(f.scale, t.scale) and np.all(f.opacity >= t.opacity)


def _shrub_scene():
    from bozkir.ply import Splats, SH_C0
    # Thin needles at about the desert's density (2,800 per m^2, uneven
    # opacity): a scene of big opaque discs once hid a cover test that
    # found nothing on the real capture.
    r = np.random.default_rng(3)
    n = 100000
    xy = r.random((n, 2)) * 6.0 - 3.0
    z = r.normal(0, 0.004, n)
    rgb = np.clip([0.55, 0.5, 0.44] + r.normal(0, 0.08, (n, 3)), 0, 1)
    shrub = np.hypot(xy[:, 0] - 1.0, xy[:, 1] + 0.5) < 0.25
    rgb[shrub] = [0.22, 0.28, 0.14]
    z[shrub] += 0.1
    marker = (np.abs(xy[:, 0] + 1.5) < 0.12) & (np.abs(xy[:, 1] - 1.0) < 0.12)
    rgb[marker] = 0.97
    return Splats(xyz=np.c_[xy, z].astype(np.float32),
                  opacity=r.uniform(0.2, 0.9, n).astype(np.float32),
                  scale=np.c_[np.full(n, 0.02), np.full(n, 0.003), np.full(n, 0.003)].astype(np.float32),
                  rot=np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
                  sh_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
                  sh_rest=np.zeros((n, 0, 3), np.float32), sh_degree=0)


def _material_patch(seed, base, grey=None, marker=False, n=20000):
    """A 1.5 m patch of one material with blotches, centred on the origin
    (as candidates are), optionally with a pale grey hole or a marker."""
    from scipy.ndimage import zoom
    r = np.random.default_rng(seed)
    xy = r.random((n, 2)) * 1.5 - 0.75
    field = zoom(r.normal(0, 1, (8, 8)), 25, order=3)[:200, :200]
    ij = np.clip(((xy + 0.75) / 1.5 * 200).astype(int), 0, 199)
    rgb = np.clip(np.float64(base) + 0.1 * field[ij[:, 1], ij[:, 0]][:, None]
                  + r.normal(0, 0.05, (n, 3)), 0, 1)
    if grey is not None:
        rgb[np.hypot(xy[:, 0] - grey[0], xy[:, 1] - grey[1]) < 0.15] = 0.4
    if marker:
        sq = np.all(np.abs(xy - [0.3, -0.2]) < 0.15, axis=1)
        rgb[sq] = 0.04
        rgb[sq & np.any(np.abs(xy - [0.3, -0.2]) < 0.025, axis=1)] = [0.8, 0.85, 0.95]
    s = Splats(xyz=np.c_[xy, np.zeros(n)].astype(np.float32),
               opacity=np.full(n, 0.9, np.float32),
               scale=np.full((n, 3), 0.02, np.float32),
               rot=np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
               sh_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
               sh_rest=np.zeros((n, 0, 3), np.float32), sh_degree=0)
    return (1.0, (seed * 2.0, 0.0), s, {})


def test_odd_spots_find_the_hole_and_the_marker_not_blotches():
    from bozkir.patches import rank_clean
    bush = [_material_patch(i, [0.19, 0.2, 0.15]) for i in range(8)]
    bush.append(_material_patch(8, [0.19, 0.2, 0.15], grey=(0.2, 0.1)))
    ordered, clean, _ = rank_clean(bush, 1.5, 25)
    assert clean == 8 and ordered[-1] is bush[-1], "the grey hole was not found"
    sand = [_material_patch(20 + i, [0.42, 0.36, 0.28]) for i in range(8)]
    sand.append(_material_patch(40, [0.42, 0.36, 0.28], marker=True))
    ordered, clean, _ = rank_clean(sand, 1.5, 25)
    assert clean == 8 and ordered[-1] is sand[-1], "the marker was not found"
    # A patch overlapping another is not its own evidence: two shifted
    # copies of one hole must both be found.
    a = _material_patch(8, [0.19, 0.2, 0.15], grey=(0.2, 0.1))
    b = (1.0, (16.5, 0.0), a[2], {})
    ordered, clean, _ = rank_clean(bush[:8] + [a, b], 1.5, 25)
    assert clean == 8


def _field_scene(markers=8, seed=5, size=12.0):
    """A 12 m field like desert v1: thin needles on sand, a dark bush strip,
    a dozen shrubs, and survey markers (black squares with a blue or white
    cross, all alike, as on the real capture)."""
    r = np.random.default_rng(seed)
    n = int(2800 * size * size)
    xy = r.random((n, 2)) * size - size / 2
    z = r.normal(0, 0.004, n)
    rgb = np.clip([0.55, 0.5, 0.44] + r.normal(0, 0.08, (n, 3)), 0, 1)
    bush = xy[:, 0] > size / 2 - 3
    rgb[bush] = np.clip([0.2, 0.22, 0.14] + r.normal(0, 0.06, (bush.sum(), 3)), 0, 1)
    shrubs = r.random((12, 2)) * [size - 5, size - 2] - [size / 2 - 1, size / 2 - 1]
    for c in shrubs:
        d = np.hypot(*(xy - c).T)
        m = (d < 0.25) & ~bush
        rgb[m] = np.clip([0.24, 0.29, 0.14] + r.normal(0, 0.05, (m.sum(), 3)), 0, 1)
        z[m] += 0.1 * (1 - d[m] / 0.25)
    marks = r.random((markers, 2)) * [size - 5, size - 2] - [size / 2 - 1, size / 2 - 1]
    for k, c in enumerate(marks):
        sq = np.all(np.abs(xy - c) < 0.15, axis=1)
        rgb[sq] = 0.03
        rgb[sq & np.any(np.abs(xy - c) < 0.025, axis=1)] = \
            [0.45, 0.55, 0.92] if k % 2 == 0 else [0.85, 0.87, 0.92]
    s = Splats(xyz=np.c_[xy, z].astype(np.float32),
               opacity=r.uniform(0.2, 0.9, n).astype(np.float32),
               scale=np.c_[np.full(n, 0.02), np.full(n, 0.003), np.full(n, 0.003)].astype(np.float32),
               rot=np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
               sh_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
               sh_rest=np.zeros((n, 0, 3), np.float32), sh_degree=0)
    return s, shrubs, marks


def test_decals_find_the_shrubs_not_the_markers():
    """Markers stand out more than any shrub, so they topped every sheet
    (6 of 12 automatic picks on desert v1, then 3 of 12 with a test of
    colour rarity); no test of brightness caught them. Measured on two real
    markers and a shrub: a blue cast, and black right beside it."""
    from bozkir.decals import find
    for markers, seed in [(0, 4), (8, 5), (24, 8)]:
        s, shrubs, marks = _field_scene(markers, seed)
        found = find(s, 0.45, count=12)
        for x, y, _ in found:
            assert not len(marks) or np.hypot(*(marks - [x, y]).T).min() > 0.5, \
                f"a marker at ({x:.2f}, {y:.2f}) was offered as a decal"
        near = [np.hypot(*(shrubs - [x, y]).T).min() < 0.3 for x, y, _ in found[:8]]
        assert all(near), f"the best eight are not all shrubs: {near}"


def test_build_sorts_flags_and_backs_up():
    import importlib.util
    import tempfile
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("build", root / "scripts" / "build.py")
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    exp, dec = b.split_flags(["--odd-spots", "30", "--decals", "8",
                              "--exclude-at=-1.5,2", "--decal-seed", "4", "--cut"])
    assert exp == ["--odd-spots", "30", "--exclude-at=-1.5,2", "--cut"]
    assert dec == ["--decals", "8", "--decal-seed", "4"]
    with tempfile.TemporaryDirectory() as d:
        stem = Path(d) / "t"
        for s in (".splat", ".json"):
            (Path(d) / ("t" + s)).write_text("x")
        moved = b.back_up(stem)
        assert moved and sorted(p.name for p in moved.iterdir()) == ["t.json", "t.splat"]
        assert not (Path(d) / "t.splat").exists()
        assert b.back_up(stem) is None


def test_apart_keeps_a_materials_patches_off_shared_ground():
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("ew", root / "scripts" / "export_wang.py")
    ew = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ew)
    # Best first: a tight cluster, then patches further out.
    cands = [(0, (x, y), None, {}) for x, y in
             [(0, 0), (0.6, 0), (0, 0.6), (3, 0), (1.2, 1.2), (0, 3), (3, 3)]]
    got = [c[1] for c in bozkir.pick.apart(cands, 4, 1.5)]
    assert got == [(0, 0), (3, 0), (0, 3), (3, 3)], got
    # Too few apart: overlapping ones fill in, best first.
    got = [c[1] for c in bozkir.pick.apart(cands[:3], 3, 1.5)]
    assert got == [(0, 0), (0.6, 0), (0, 0.6)], got


def test_dense_first_leaves_thin_patches_out():
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("ew", root / "scripts" / "export_wang.py")
    ew = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ew)
    sizes = [25723, 6836, 22027, 5852, 20000, 5043, 15644, 6245]   # desert6's sand
    cands = [(0, (i * 2.0, 0.0), np.zeros(n), {}) for i, n in enumerate(sizes)]
    got = [len(c[2]) for c in bozkir.pick.dense_first(cands, 3)]
    assert got == [25723, 22027, 20000, 15644], got
    # Too few dense ones: the densest thin ones fill in.
    got = [len(c[2]) for c in bozkir.pick.dense_first(cands, 6)]
    assert got == [25723, 22027, 20000, 15644, 6836, 6245], got


def test_missing_layer_adds_only_what_the_capture_shows():
    from bozkir.tile import missing_layer, merge
    from bozkir.graphcut import render_patch
    r = np.random.default_rng(0)
    xy = r.random((8000, 2)) * 1.5 - 0.75
    xy = xy[np.hypot(xy[:, 0] - 0.2, xy[:, 1] + 0.1) > 0.25]      # a hole
    n = len(xy)

    def splats(xy, z, op, sc, rgb):
        k = len(xy)
        return Splats(xyz=np.c_[xy, z].astype(np.float32),
                      opacity=np.broadcast_to(np.float32(op), (k,)).copy(),
                      scale=np.broadcast_to(np.float32(sc), (k, 3)).copy(),
                      rot=np.tile(np.float32([1, 0, 0, 0]), (k, 1)),
                      sh_dc=((np.broadcast_to(np.float32(rgb), (k, 3)) - 0.5) / SH_C0).astype(np.float32),
                      sh_rest=np.zeros((k, 0, 3), np.float32), sh_degree=0)
    kept = splats(xy, r.normal(0.05, 0.004, n), 0.9, [0.02, 0.02, 0.004], [0.55, 0.5, 0.44])
    yy, xx = np.mgrid[0:60, 0:60]
    hole = np.hypot(-0.75 + (xx + 0.5) * 0.025 - 0.2, 0.75 - (yy + 0.5) * 0.025 + 0.1) < 0.2
    for wash in (0.8, 0.4):
        # A huge faint splat centred far off, as the capture's are.
        cap = merge(kept, splats([[3.0, 1.0]], [-0.2], wash, [20, 20, 0.05], [0.3, 0.25, 0.2]))
        layer = missing_layer(kept, cap, 1.5)
        inside = np.hypot(layer.xyz[:, 0] - 0.2, layer.xyz[:, 1] + 0.1) < 0.3
        assert inside.all(), "the layer went where the patch was solid"
        assert layer.xyz[:, 2].max() <= np.percentile(kept.xyz[:, 2], 10) + 0.003
        _, a_t = render_patch(cap, 1.5, 60, 2)
        _, a_f = render_patch(merge(kept, layer), 1.5, 60, 2)
        assert abs(a_f[hole].mean() - a_t[hole].mean()) < 0.08, \
            f"cover {a_f[hole].mean():.2f} against the capture's {a_t[hole].mean():.2f}"
    # Nothing missing, nothing added.
    assert len(missing_layer(kept, kept, 1.5)) == 0


def test_apart_alike_keeps_a_material_from_mixing_dark_and_pale():
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("ew", root / "scripts" / "export_wang.py")
    ew = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ew)
    from bozkir.ply import SH_C0

    def patch(v):
        n = 200
        rgb = np.full((n, 3), v, np.float32)
        return Splats(xyz=np.zeros((n, 3), np.float32), opacity=np.ones(n, np.float32),
                      scale=np.full((n, 3), 0.01, np.float32),
                      rot=np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
                      sh_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
                      sh_rest=np.zeros((n, 0, 3), np.float32), sh_degree=0)
    # desert7's bush: 0.20 first, then a pale half-sand one and a near twin.
    cands = [(0, (0, 0), patch(0.20), {}), (0, (3, 0), patch(0.32), {}),
             (0, (0, 3), patch(0.14), {}), (0, (3, 3), patch(0.21), {}),
             (0, (0.5, 0), patch(0.20), {})]
    got = [c[1] for c in bozkir.pick.apart_alike(cands, 3, 1.5)]
    assert got == [(0, 0), (3, 3), (0, 3)], got          # the pale one left out
    got = [c[1] for c in bozkir.pick.apart_alike(cands, 2, 1.5, apart_too=False)]
    assert got == [(0, 0), (0.5, 0)], got


def _flat_patch(seed, base, std=0.05, n=6000, half=0.95):
    from bozkir.ply import SH_C0
    r = np.random.default_rng(seed)
    xy = r.random((n, 2)) * 2 * half - half
    rgb = np.clip(np.float32(base) + r.normal(0, std, (n, 3)), 0, 1)
    return Splats(xyz=np.c_[xy, r.normal(0, 0.004, n)].astype(np.float32),
                  opacity=np.full(n, 0.9, np.float32),
                  scale=np.c_[np.full((n, 2), 0.03), np.full(n, 0.004)].astype(np.float32),
                  rot=np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
                  sh_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
                  sh_rest=np.zeros((n, 0, 3), np.float32), sh_degree=0)


def test_centre_patches_keep_edges_and_end_the_cross():
    """Each tile's interior from its own patch (a centre patch and four
    boundary strips, DAV-GSWT; a closed seam, omega-tiles): edges still
    identical between tiles that share a colour, and the four triangles,
    patches of different brightness here, no longer meet in the middle:
    what is left of their tones reaches the centre as a smooth grade (the
    seam membrane), with no step."""
    from scipy.ndimage import gaussian_filter
    from bozkir.wang import build_tile_set, minimal_codes, edge_gaussians
    from bozkir.graphcut import render_patch, tile_uv
    h = [_flat_patch(1, [0.55] * 3), _flat_patch(2, [0.40] * 3), _flat_patch(3, [0.65] * 3)]
    v = [_flat_patch(4, [0.48] * 3), _flat_patch(5, [0.60] * 3), _flat_patch(6, [0.45] * 3)]
    codes = minimal_codes(3, 3)[:6]
    centres = [_flat_patch(100 + i, [0.55] * 3) for i in range(6)]
    u, v_ = tile_uv(64)
    core = np.maximum(np.abs(u), np.abs(v_)) < 0.38
    step = {}
    for name, cs in (("cross", None), ("centre", centres)):
        tiles, got = build_tile_set(h, v, 1.5, cut=True, resolution=64,
                                    codes=codes, centres=cs)
        for edge, col in (("n", 0), ("e", 1), ("s", 2), ("w", 3)):
            same = {}
            for t, c in zip(tiles, got):
                same.setdefault(c[col], []).append(edge_gaussians(t, 1.5, 2, edge, margin=0.03))
            for g in same.values():
                assert all(np.array_equal(g[0], x) for x in g), f"{name}: edge {edge} differs"
        sharp = []
        for t in tiles:
            gy, gx = np.gradient(gaussian_filter(render_patch(t, 1.5, 64, 2)[0].mean(-1), 1.5))
            sharp.append(np.percentile(np.hypot(gx, gy)[core], 95))
        step[name] = np.mean(sharp)
    assert step["centre"] < 0.6 * step["cross"], step


def test_seam_membrane_meets_the_edge_and_stays_smooth():
    """Poisson-style correction: the offset equals the colour difference
    along the seam and has no step anywhere inside."""
    from bozkir.wang import seam_membrane
    r = 64
    yy, xx = np.mgrid[0:r, 0:r]
    inside = (np.abs(xx - 31.5) < 20) & (np.abs(yy - 31.5) < 20)
    diff = np.zeros((r, r, 3))
    diff[..., 0] = 0.1 * (xx > 31.5)            # the frame is redder on one side
    f = seam_membrane(diff, np.ones((r, r)), inside)
    assert abs(f[31, 11, 0]) < 0.03 and abs(f[31, 51, 0] - 0.1) < 0.03, (f[31, 11, 0], f[31, 51, 0])
    steps = np.abs(np.diff(f[..., 0], axis=1))[inside[:, 1:] & inside[:, :-1]]
    assert steps.max() < 0.02, steps.max()
    assert not f[~inside].any()


def test_even_out_moves_the_tone_and_keeps_the_detail():
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("ew", root / "scripts" / "export_wang.py")
    ew = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ew)
    p = _flat_patch(3, [0.40, 0.36, 0.30], std=0.06)
    q = bozkir.cut.even_out(p, np.array([0.45, 0.41, 0.35]), 0.08)
    assert np.allclose(bozkir.cut.even_mean(q), [0.45, 0.41, 0.35], atol=0.01)
    d = q.base_rgb - p.base_rgb
    assert np.allclose(d.std(axis=0), 0, atol=0.01), "detail changed, not only the tone"
    far = bozkir.cut.even_out(p, np.array([0.70, 0.70, 0.70]), 0.08)
    assert np.all(bozkir.cut.even_mean(far) - bozkir.cut.even_mean(p) < 0.09), "moved further than allowed"


def test_split_materials_is_not_led_by_a_few_markers():
    from bozkir.patches import split_materials

    def cand(seed, base, std):
        return (1.0, (float(seed), 0.0), _flat_patch(seed, base, std, n=1500), {})
    kinds = ([("bare", cand(i, [0.55, 0.51, 0.45], 0.04)) for i in range(60)]
             + [("scrub", cand(200 + i, [0.46, 0.43, 0.36], 0.10)) for i in range(50)]
             + [("green", cand(400 + i, [0.24, 0.28, 0.16], 0.06)) for i in range(8)]
             + [("marker", cand(600 + i, [0.15, 0.15, 0.20], 0.35)) for i in range(3)])
    kind = {id(c): k for k, c in kinds}
    groups = split_materials([c for _, c in kinds], 3)
    seen = [sorted({kind[id(c)] for c in g} - {"marker"}) for g in groups]
    assert seen == [["bare"], ["scrub"], ["green"]], seen


def test_an_outnumbered_material_is_not_set_aside_as_odd():
    """desert10 with 800 positions: the sand packed so tightly that all the
    bush looked odd and it vanished as a material."""
    from bozkir.patches import split_materials

    def cand(seed, base, std):
        return (1.0, (float(seed), 0.0), _flat_patch(seed, base, std, n=600), {})
    kinds = ([("light", cand(i, [0.52, 0.48, 0.44], 0.02)) for i in range(380)]
             + [("mid", cand(1000 + i, [0.44, 0.41, 0.37], 0.02)) for i in range(330)]
             + [("bush", cand(2000 + i, [0.23, 0.23, 0.18], 0.07)) for i in range(45)]
             + [("marker", cand(3000 + i, [0.15, 0.15, 0.20], 0.35)) for i in range(4)])
    kind = {id(c): k for k, c in kinds}
    groups = split_materials([c for _, c in kinds], 3)
    seen = sorted(tuple(sorted({kind[id(c)] for c in g} - {"marker"})) for g in groups)
    assert seen == [("bush",), ("light",), ("mid",)], seen


def test_class_shares_measure_the_capture_not_the_candidates():
    """A clump that gave plenty of candidates is still a small share of
    the ground."""
    from bozkir.patches import class_shares
    from bozkir.ply import SH_C0
    r = np.random.default_rng(0)
    n = 150000
    xy = r.random((n, 2)) * [12.0, 6.0] - [6.0, 3.0]
    clump = np.hypot(xy[:, 0] - 4.0, xy[:, 1]) < 1.0              # 4% of the ground
    rgb = np.where(clump[:, None], [0.24, 0.28, 0.16], [0.55, 0.51, 0.45])
    rgb = np.clip(rgb + r.normal(0, 0.04, (n, 3)), 0, 1)
    s = Splats(xyz=np.c_[xy, np.zeros(n)].astype(np.float32), opacity=np.ones(n, np.float32),
               scale=np.full((n, 3), 0.02, np.float32), rot=np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
               sh_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
               sh_rest=np.zeros((n, 0, 3), np.float32), sh_degree=0)
    sand = [(1, (0, 0), _flat_patch(i, [0.55, 0.51, 0.45], 0.04), {}) for i in range(5)]
    bush = [(1, (0, 0), _flat_patch(50 + i, [0.24, 0.28, 0.16], 0.04), {}) for i in range(5)]
    shares = class_shares(s, [sand, bush], 1.5, min_splats=500)
    assert shares[1] < 0.15 and shares[0] > 0.85, shares



def _export_module():
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("ew", root / "scripts" / "export_wang.py")
    ew = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ew)
    return ew


def test_centres_come_from_ground_of_their_own():
    """desert9: a material's 18 centres were shifted copies from one spot."""
    ew = _export_module()
    spot = [(1, (x, y), _flat_patch(int(x * 40 + y * 4), [0.53, 0.48, 0.43], 0.03, n=800), {})
            for x in np.arange(0, 3.01, 0.25) for y in np.arange(9, 11.41, 0.25)]
    edge = [(1, (-4.0, 0.0), _flat_patch(999, [0.53, 0.48, 0.43], 0.03, n=800), {})]
    got = bozkir.pick.pick_centres(edge, edge + spot, 18, 1.5)
    assert 2 <= len(got) < 18, len(got)
    pts = [c[1] for c in edge + got]
    for i in range(len(pts)):
        for j in range(i):
            gap = max(abs(pts[i][0] - pts[j][0]), abs(pts[i][1] - pts[j][1]))
            assert gap >= 1.05 - 1e-9, (pts[i], pts[j])


def test_centres_keep_their_materials_colour():
    """desert10: the bush ran out of ground and took brown rim sand as
    centres; every bush tile had a sand square."""
    ew = _export_module()
    bush = lambda i, x: (1, (x, 0.0), _flat_patch(i, [0.23, 0.25, 0.19], 0.03, n=900), {})
    edge = [bush(i, 2.0 * i) for i in range(6)]
    own = [bush(100 + i, 20.0 + 2.0 * i) for i in range(2)]
    brown = [(1, (40.0 + 2.0 * i, 0.0), _flat_patch(200 + i, [0.41, 0.36, 0.33], 0.03, n=900), {})
             for i in range(6)]
    got = bozkir.pick.pick_centres(edge, edge + own + brown, 18, 1.5)
    assert len(got) == 2 and all(c in own for c in got), [c[1] for c in got]


def test_a_material_keeps_its_common_tone():
    """desert11: bluish grass (as blue as red) had the bush's brightness and
    passed; a yellower bush, or a shrub in a sand tile, must still pass."""
    from bozkir.patches import tone_gate
    from bozkir.ply import SH_C0
    r = np.random.default_rng(8)

    def patch(i, base, shrub=0.0, other=None, part=0.0):
        p = _flat_patch(i, base, 0.05, n=3000)
        rgb = p.base_rgb.copy()
        k = len(rgb)
        if shrub:
            rgb[: int(shrub * k)] = np.clip([0.2, 0.3, 0.12] + r.normal(0, 0.03, (int(shrub * k), 3)), 0, 1)
        if other is not None:
            rgb[: int(part * k)] = np.clip(np.array(other) + r.normal(0, 0.03, (int(part * k), 3)), 0, 1)
        p.sh_dc = ((rgb - 0.5) / SH_C0).astype(np.float32)
        return (1, (float(i), 0.0), p, {})
    bush = [patch(i, [0.25 + 0.01 * (i % 3), 0.25, 0.20 - 0.02 * (i % 4)]) for i in range(30)]
    test = tone_gate(bush)
    assert test(patch(100, [0.27, 0.28, 0.27]))[0] is False                    # bluish
    assert "bluer" in test(patch(101, [0.27, 0.28, 0.27]))[1]
    assert test(patch(102, [0.27, 0.26, 0.16]))[0]                              # yellower bush
    sand = [patch(200 + i, [0.5 + 0.01 * (i % 3), 0.46, 0.42]) for i in range(30)]
    test = tone_gate(sand)
    assert test(patch(300, [0.51, 0.46, 0.42], shrub=0.08))[0]                  # one shrub
    ok, why = test(patch(301, [0.51, 0.46, 0.42], other=[0.15, 0.15, 0.2], part=0.45))
    assert not ok, why                                                          # half something else


def test_holes_from_above_are_found():
    ew = _export_module()
    whole = _flat_patch(1, [0.5, 0.45, 0.4], n=20000, half=0.75)
    p = _flat_patch(2, [0.5, 0.45, 0.4], n=20000, half=0.75)
    keep = np.flatnonzero(np.hypot(p.xyz[:, 0] - 0.3, p.xyz[:, 1] + 0.2) > 0.2)
    holed = p.subset(keep)
    n0 = bozkir.holes.holes(whole, 1.5, 2)[0]
    n1, area, mask = bozkir.holes.holes(holed, 1.5, 2)
    assert n0 == 0 and n1 == 1 and 300 < area < 1400, (n0, n1, area)


def test_patches_with_holes_are_passed_over():
    """desert11: holes in the patches showed in 16 of 18 bush tiles."""
    ew = _export_module()

    def cand(i, x, hole):
        p = _flat_patch(i, [0.5, 0.45, 0.4], n=20000, half=0.75)
        if hole:
            p = p.subset(np.flatnonzero(np.hypot(p.xyz[:, 0], p.xyz[:, 1]) > 0.2))
        return (1, (x, 0.0), p, {})
    cands = [cand(0, 0.0, False), cand(1, 2.0, True), cand(2, 4.0, False), cand(3, 6.0, False)]
    got = bozkir.pick.apart_alike(cands, 4, 1.5, accept=lambda c: bozkir.holes.hole_free(c, 1.5, 2, 0.99))
    assert sorted(c[1][0] for c in got) == [0.0, 4.0, 6.0], [c[1] for c in got]
    assert cands[1][3]["holed"] and not cands[0][3]["holed"]


def test_the_ground_slab_follows_the_ground_not_the_bush():
    """A plane fitted through the bush top cut away the ground under it
    (a third of the splats on this cut): holes in every bush tile."""
    import argparse
    from bozkir.patches import ground_level
    from bozkir.ply import SH_C0
    ew = _export_module()
    r = np.random.default_rng(9)
    n = 120000
    xy = r.random((n, 2)) * 4.0 - 2.0
    z = r.normal(0, 0.005, n)
    bush = xy[:, 0] > -1.0
    z[bush] += r.random(bush.sum()) * 0.4                        # 40 cm of bush
    s = Splats(xyz=np.c_[xy, z].astype(np.float32), opacity=np.full(n, 0.8, np.float32),
               scale=np.full((n, 3), 0.02, np.float32),
               rot=np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
               sh_dc=((np.full((n, 3), 0.4) - 0.5) / SH_C0).astype(np.float32),
               sh_rest=np.zeros((n, 0, 3), np.float32), sh_degree=0)
    args = argparse.Namespace(size=1.5, thickness=0.375, extract_margin=0.35, max_per_tile=0)
    inside = np.all(np.abs(xy) <= 1.37, axis=1)
    wide = s.subset(np.flatnonzero(inside))
    h = wide.xyz[:, 2]
    g = ground_level(h, 0.375)
    wide = wide.subset(np.flatnonzero((h >= g - 0.094) & (h <= g + 0.375)))
    got = bozkir.cut.prepare_patches([(1.0, (0.0, 0.0), s, {"relief": 0.2, "wide": wide, "cut": 2.03})],
                             1, args, 2, None, s, 0.74)[0]
    assert len(got) > 0.9 * inside.sum(), (len(got), int(inside.sum()))


def test_the_rim_is_not_taken_for_bush_when_shrinking():
    """desert11: a 2.42 m cut of the capture's dark rim (0.39) passed as bush
    on the material map, which judges over a tile."""
    ew = _export_module()
    bush = [(1, (2.0 * i, 0.0), _flat_patch(i, [0.23 + 0.01 * i, 0.25, 0.19], 0.03, n=900), {})
            for i in range(6)]
    alike = bozkir.pick.colour_alike(bush)
    rim = (1, (40.0, 0.0), _flat_patch(99, [0.39, 0.34, 0.31], 0.03, n=900), {})
    more = (1, (44.0, 0.0), _flat_patch(98, [0.26, 0.27, 0.21], 0.03, n=900), {})
    assert not alike(rim) and alike(more)


def test_a_material_can_be_cut_larger_and_shrunk():
    """--class-scale: a 3 m cut shrunk to the 1.5 m tile, splats with it."""
    import argparse
    from bozkir.ply import SH_C0
    ew = _export_module()
    r = np.random.default_rng(7)
    n = 80000
    xy = r.random((n, 2)) * 8.0 - 4.0
    rgb = np.clip([0.3, 0.32, 0.2] + r.normal(0, 0.05, (n, 3)), 0, 1)
    s = Splats(xyz=np.c_[xy, r.normal(0, 0.05, n)].astype(np.float32),
               opacity=np.full(n, 0.5, np.float32), scale=np.full((n, 3), 0.02, np.float32),
               rot=np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
               sh_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
               sh_rest=np.zeros((n, 0, 3), np.float32), sh_degree=0)
    args = argparse.Namespace(size=1.5, thickness=0.375, extract_margin=0.35, max_per_tile=0)
    near = lambda half: s.subset(np.flatnonzero(np.all(np.abs(xy) <= half, axis=1)))
    plain = bozkir.cut.prepare_patches([(1.0, (0.0, 0.0), s, {"relief": 0.1, "wide": near(1.0125)})],
                               1, args, 2)[0]
    small = bozkir.cut.prepare_patches([(1.0, (0.0, 0.0), s, {"relief": 0.1, "wide": near(2.025),
                                                      "cut": 3.0})], 1, args, 2, None, s, 0.5)[0]
    reach = lambda p: float(np.abs(p.xyz[:, :2]).max())
    assert abs(reach(small) - reach(plain)) < 0.1, (reach(small), reach(plain))
    assert len(small) > 3 * len(plain)                      # four times the ground
    assert np.allclose(small.scale, 0.01, atol=1e-6)
    assert float(np.std(small.xyz[:, 2])) < 0.6 * float(np.std(plain.xyz[:, 2]))
    # On a slope a level slab cut away the uphill and downhill sides (gaps in
    # desert10s); the slab follows the ground's plane instead.
    tilted = s.subset(np.arange(len(s)))
    tilted.xyz = tilted.xyz.copy()
    tilted.xyz[:, 2] += 0.14 * tilted.xyz[:, 0]                # 8 degrees
    wide = tilted.subset(np.flatnonzero(np.all(np.abs(xy) <= 2.025, axis=1)))
    h = wide.xyz[:, 2]
    clipped = wide.subset(np.flatnonzero((h > -0.1) & (h < 0.375)))   # a level slab's cut
    got = bozkir.cut.prepare_patches([(1.0, (0.0, 0.0), tilted, {"relief": 0.1, "wide": clipped,
                                                         "cut": 3.0})], 1, args, 2, None,
                             tilted, 0.5)[0]
    x = got.xyz[:, 0]
    assert (x < -0.6).sum() > 0.15 * len(got) and (x > 0.6).sum() > 0.15 * len(got), \
        ((x < -0.6).mean(), (x > 0.6).mean())


def test_material_map_and_the_look_picture():
    import tempfile
    from bozkir.patches import material_map
    from bozkir.ply import SH_C0
    ew = _export_module()
    r = np.random.default_rng(1)
    n = 150000
    xy = r.random((n, 2)) * [12.0, 6.0] - [6.0, 3.0]
    bush = xy[:, 0] > 2.0
    rgb = np.where(bush[:, None], [0.24, 0.28, 0.16], [0.55, 0.51, 0.45])
    rgb = np.clip(rgb + r.normal(0, 0.04, (n, 3)), 0, 1)
    s = Splats(xyz=np.c_[xy, np.zeros(n)].astype(np.float32), opacity=np.ones(n, np.float32),
               scale=np.full((n, 3), 0.02, np.float32), rot=np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
               sh_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
               sh_rest=np.zeros((n, 0, 3), np.float32), sh_degree=0)
    sand = [(1, (-4.0 + i, 0), _flat_patch(i, [0.55, 0.51, 0.45], 0.04), {}) for i in range(4)]
    scrub = [(1, (3.0 + i, 0), _flat_patch(50 + i, [0.24, 0.28, 0.16], 0.04), {}) for i in range(2)]
    labels, (x0, y0) = material_map(s, [sand, scrub], 1.5, min_splats=500)
    at = lambda x, y: labels[int((y - y0) / 0.5), int((x - x0) / 0.5)]
    assert at(-4, 0) == 0 and at(4.5, 0) == 1 and at(-1, 1) == 0, labels
    with tempfile.TemporaryDirectory() as d:
        dest = Path(d) / "classes_t.png"
        info = {"shares": [0.8, 0.2], "ground_shares": [0.67, 0.33], "thick": [0.01, 0.06],
                "odd": [(0.0, 0.0)], "viable": [(-4.0, 0.0), (3.0, 0.0)]}
        ew.look(s, [sand, scrub], [sand[:2], scrub[:1]], [sand[2:], []], info, 1.5, 2,
                dest, "t")
        assert dest.stat().st_size > 10000


def test_the_smeared_rim_is_not_counted_as_a_material():
    """desert v1's left rim is a dark smear of big splats; it counted as bush."""
    from bozkir.patches import class_shares, material_map
    from bozkir.ply import SH_C0
    r = np.random.default_rng(2)
    n = 200000
    xy = r.random((n, 2)) * [12.0, 6.0] - [6.0, 3.0]
    bush = xy[:, 0] > 4.5                                    # 12.5% of the ground
    rim = xy[:, 0] < -3.0                                    # dark, smeared
    rgb = np.where((bush | rim)[:, None], [0.24, 0.28, 0.16], [0.55, 0.51, 0.45])
    rgb = np.clip(rgb + r.normal(0, 0.04, (n, 3)), 0, 1)
    scale = np.where(rim, 0.08, 0.02)[:, None] * np.ones(3)
    s = Splats(xyz=np.c_[xy, np.zeros(n)].astype(np.float32), opacity=np.ones(n, np.float32),
               scale=scale.astype(np.float32), rot=np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
               sh_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
               sh_rest=np.zeros((n, 0, 3), np.float32), sh_degree=0)
    sand = [(1, (0, 0), _flat_patch(i, [0.55, 0.51, 0.45], 0.04), {}) for i in range(5)]
    scrub = [(1, (0, 0), _flat_patch(50 + i, [0.24, 0.28, 0.16], 0.04), {}) for i in range(5)]
    shares = class_shares(s, [sand, scrub], 1.5, min_splats=500)
    assert shares[1] < 0.25, shares                          # 0.4 with the rim
    labels, (x0, y0) = material_map(s, [sand, scrub], 1.5, min_splats=500)
    assert labels[int((0 - y0) / 0.5), int((-5 - x0) / 0.5)] == -1
    assert labels[int((0 - y0) / 0.5), int((5.5 - x0) / 0.5)] == 1


def test_thickness_tells_a_crust_from_a_haze():
    from bozkir.patches import thickness
    crust = _flat_patch(1, [0.5, 0.45, 0.4])
    haze = _flat_patch(2, [0.2, 0.22, 0.15])
    r = np.random.default_rng(3)
    haze.xyz = haze.xyz.copy()
    haze.xyz[:, 2] = r.random(len(haze.xyz)) * 0.3          # see-through, 30 cm deep
    assert thickness(crust) < 0.01 < 0.05 < thickness(haze), (thickness(crust), thickness(haze))


def test_haze_gets_a_small_part_of_the_map_not_none():
    ew = _export_module()
    got = bozkir.materials.weigh_shares([0.49, 0.27, 0.24], [0.01, 0.012, 0.10], 0.5)
    assert abs(sum(got) - 1) < 1e-9 and 0.04 < got[2] < 0.15 and got[0] > got[1] > got[2], got
    same = bozkir.materials.weigh_shares([0.49, 0.27, 0.24], [0.01, 0.012, 0.10], 0.0)
    assert np.allclose(same, [0.49, 0.27, 0.24])


def test_class_reasons_name_what_a_material_lacks():
    ew = _export_module()
    def mat(base, xs):
        return [(1, (x, 0.0), _flat_patch(int(x * 10) + int(base[0] * 100), base, 0.03, n=800), {})
                for x in xs]
    sand = mat([0.55, 0.51, 0.45], np.arange(0, 12, 1.6))
    twin = mat([0.56, 0.52, 0.46], np.arange(20, 32, 1.6))    # the same colour
    bush = mat([0.22, 0.25, 0.15], [40.0, 40.5, 41.0])         # one spot
    why = bozkir.materials.class_reasons([sand, twin, bush], [0.6, 0.35, 0.05], 1.5, 6, 0.1)
    assert why[0] == [] or all("colour" in w for w in why[0]), why
    assert any("colour" in w for w in why[1])
    assert any("ground" in w for w in why[2]) and any("apart" in w for w in why[2]), why


def test_the_search_says_where_it_rejected_and_why():
    """desert10: the sharp middle had few usable positions, and nothing said why."""
    from bozkir.patches import pick_patches
    from bozkir.ply import SH_C0
    r = np.random.default_rng(4)
    n = 120000
    xy = r.random((n, 2)) * [8.0, 4.0] - [4.0, 2.0]
    keep = (xy[:, 0] < 0) | (r.random(n) < 0.05)              # the right half sparse
    xy = xy[keep]
    n = len(xy)
    rgb = np.clip([0.55, 0.5, 0.44] + r.normal(0, 0.06, (n, 3)), 0, 1)
    s = Splats(xyz=np.c_[xy, r.normal(0, 0.003, n)].astype(np.float32),
               opacity=np.full(n, 0.8, np.float32), scale=np.full((n, 3), 0.02, np.float32),
               rot=np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
               sh_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
               sh_rest=np.zeros((n, 0, 3), np.float32), sh_degree=0)
    stats = {}
    pick_patches(s, 1.5, 10 ** 9, 2, verbose=False, stats=stats, min_separation=0.5)
    where = stats["rejected_at"]
    assert where and all(r == "sparse" for x, _, r in where if x > 1.0), where[:5]
    assert all(x > -0.9 for x, _, r in where if r == "sparse"), where[:5]
    one = [(0, (0.0, 0.0), _flat_patch(1, [0.5, 0.45, 0.4]), {}),
           (0, (0.1, 0.0), _flat_patch(2, [0.5, 0.45, 0.4]), {}),
           (0, (1.4, 0.0), _flat_patch(3, [0.5, 0.45, 0.4]), {})]
    got = [c[1] for c in _export_module().apart_alike(one, 2, 1.5)]
    assert got == [(0.0, 0.0), (1.4, 0.0)], got               # least overlap, not a twin


def test_faint_ground_is_told_from_ground_with_gaps():
    """desert10: 719 positions failed the hole test; the sharp middle is a
    faint but whole layer that the export's fill makes solid."""
    from bozkir.patches import pick_patches, rendered_coverage
    from bozkir.ply import SH_C0
    r = np.random.default_rng(5)

    def scene(gap):
        n = 60000
        xy = r.random((n, 2)) * [6.0, 3.0] - [3.0, 1.5]
        if gap:                                            # bare 30 cm stripes
            xy = xy[(np.floor((xy[:, 0] + 3.0) / 0.3) % 2) == 0]
        n = len(xy)
        rgb = np.clip([0.55, 0.5, 0.44] + r.normal(0, 0.05, (n, 3)), 0, 1)
        return Splats(xyz=np.c_[xy, r.normal(0, 0.003, n)].astype(np.float32),
                      opacity=np.full(n, 0.12, np.float32),
                      scale=np.full((n, 3), 0.015, np.float32),
                      rot=np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
                      sh_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
                      sh_rest=np.zeros((n, 0, 3), np.float32), sh_degree=0)
    def search(s, **kw):
        st = {}
        try:
            pick_patches(s, 1.5, 10 ** 9, 2, verbose=False, stats=st, min_separation=0.5, **kw)
        except SystemExit:                                 # nothing passed
            pass
        return st
    faint, gappy = scene(False), scene(True)
    now, filled = rendered_coverage(faint, 1.5, 2, boost=(1.0, 6.0))
    assert now < 0.8 <= filled, (now, filled)
    for s, want in ((faint, "faint"), (gappy, "gaps")):
        st = search(s)
        why = {w for _, _, w in st["rejected_at"]}
        assert st["viable"] == 0 and want in why, (want, why)
    st = search(faint, faint_fill=6.0)
    assert st["viable"] > 0, st                            # passes once judged filled
    st = search(gappy, faint_fill=6.0)
    assert st["viable"] == 0, st                           # gaps stay out


def test_ground_the_photos_saw_little_is_not_counted():
    """desert10: the faint, hazy left rim is dark and counted as bush."""
    from bozkir.patches import class_shares, material_map, seen_points
    from bozkir.ply import SH_C0
    r = np.random.default_rng(6)
    n = 200000
    xy = r.random((n, 2)) * [12.0, 6.0] - [6.0, 3.0]
    dark = (xy[:, 0] > 4.5) | (xy[:, 0] < -3.0)               # bush, and the rim
    rgb = np.where(dark[:, None], [0.24, 0.28, 0.16], [0.55, 0.51, 0.45])
    rgb = np.clip(rgb + r.normal(0, 0.04, (n, 3)), 0, 1)
    s = Splats(xyz=np.c_[xy, np.zeros(n)].astype(np.float32), opacity=np.ones(n, np.float32),
               scale=np.full((n, 3), 0.02, np.float32), rot=np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
               sh_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
               sh_rest=np.zeros((n, 0, 3), np.float32), sh_degree=0)
    sand = [(1, (0, 0), _flat_patch(i, [0.55, 0.51, 0.45], 0.04), {}) for i in range(5)]
    bush = [(1, (5.2, 0), _flat_patch(50 + i, [0.24, 0.28, 0.16], 0.04), {}) for i in range(5)]
    grid = [(x, y) for x in np.arange(-5.25, 5.3, 0.75) for y in np.arange(-2.25, 2.3, 0.75)]
    found = [(1, p, None, {}) for p in grid if p[0] >= -3.0]
    stats = {"rejected_at": [(x, y, "faint") for x, y in grid if x < -3.0]}
    seen = seen_points(found, stats)
    blind = class_shares(s, [sand, bush], 1.5, min_splats=500)
    shares = class_shares(s, [sand, bush], 1.5, min_splats=500, seen=seen)
    assert blind[1] > 0.4 and shares[1] < 0.3 and blind[1] - shares[1] > 0.12, (blind, shares)
    labels, (x0, y0) = material_map(s, [sand, bush], 1.5, min_splats=500, seen=seen)
    assert labels[int((0 - y0) / 0.5), int((-5 - x0) / 0.5)] == -1
    assert labels[int((0 - y0) / 0.5), int((5.5 - x0) / 0.5)] == 1


def test_decal_copy_keeps_its_tile_edges():
    from bozkir.decals import cut, place
    s = _shrub_scene()
    d = cut(s, (1.0, -0.5), 0.5)
    tile = s.subset(np.flatnonzero(np.all(np.abs(s.xyz[:, :2]) <= 0.75, axis=1)))
    for at in [(0.0, 0.0), (0.2, -0.15)]:
        v = place(tile, d, 0.45, 1.5, at=at)
        rv = np.hypot(*(v.xyz[:, :2] - np.float32(at)).T)
        rt = np.hypot(*(tile.xyz[:, :2] - np.float32(at)).T)
        assert np.array_equal(v.xyz[rv >= 0.45][:int((rt >= 0.45).sum())],
                              tile.xyz[rt >= 0.45])           # outside the disc: untouched
        edge = np.abs(v.xyz[:, :2]).max(1) > 0.65
        assert np.array_equal(np.sort(v.xyz[edge], axis=0),
                              np.sort(tile.xyz[np.abs(tile.xyz[:, :2]).max(1) > 0.65], axis=0))
    for bad in [dict(radius=0.7, at=(0, 0)), dict(radius=0.45, at=(0.3, 0))]:
        try:
            place(tile, d, bad["radius"], 1.5, at=bad["at"])
            raise AssertionError(f"a decal that leaves the tile was accepted: {bad}")
        except ValueError:
            pass



def _ew():
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("ew", root / "scripts" / "export_wang.py")
    ew = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ew)
    return ew


def _texture(kind, seed, n=60, px=0.025):
    """Top-down luminance of a made-up material: blobs without a direction,
    ripples at 30 degrees (each patch a little off), or lopsided ripples
    (a lit face that rises fast and a long shaded one)."""
    from scipy.ndimage import gaussian_filter
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] * px
    if kind == "blobs":
        g = gaussian_filter(r.normal(size=(n, n)), 1.5)
        return 0.5 + 0.05 * g / g.std()
    a = np.radians(30 + r.normal(0, 5) if kind != "every way" else r.uniform(0, 180))
    u = (xx * np.cos(a) + yy * np.sin(a)) / 0.25 + r.random()
    if kind == "lopsided":
        f = u % 1.0
        f = np.where(f < 0.15, f / 0.15, 1 - (f - 0.15) / 0.85)
    else:
        f = 0.5 + 0.5 * np.sin(2 * np.pi * u)
    return 0.4 + 0.2 * f + 0.05 * r.normal(size=(n, n))


def test_frames_keep_the_edges_and_centres_the_middle():
    from bozkir.frames import geometry, frame_of, centre_of
    g = geometry(1.5, 0.6, 15.0)
    assert abs(g["radius"] - 0.45) < 1e-9 and abs(g["half"] - 0.075) < 1e-9
    assert g["inner"] < g["radius"] < g["outer"] < 0.75 * (1 - 2 * 0.06)
    t = _flat_patch(1, [0.5, 0.5, 0.5], n=20000, half=0.75)
    f, c = frame_of(t, g["inner"]), centre_of(t, g["outer"])
    rf, rc = np.hypot(*f.xyz[:, :2].T), np.hypot(*c.xyz[:, :2].T)
    assert rf.min() >= g["inner"] and rc.max() <= g["outer"]
    edge = np.abs(t.xyz[:, :2]).max(axis=1) > 0.75 - 0.09
    assert int(edge.sum()) == int((np.abs(f.xyz[:, :2]).max(axis=1) > 0.75 - 0.09).sum())
    assert len(f) + len(c) > len(t)                      # they overlap in the ring
    try:
        geometry(1.5, 0.8, 15.0)
        raise AssertionError("a centre into the edge strip was allowed")
    except ValueError:
        pass


def test_how_far_centres_turn_is_measured():
    from bozkir.frames import turn_rule
    w = np.ones((60, 60))
    rule = lambda kind: turn_rule([(_texture(kind, s), w) for s in range(8)])
    assert rule("blobs")["any"]
    rip = rule("ripples")
    assert not rip["any"] and rip["half"] and 4.9 <= rip["spread"] <= 12, rip
    lop = rule("lopsided")
    assert not lop["any"] and not lop["half"], lop          # a half turn would show
    assert rule("every way")["any"]                         # patches disagree: any


def test_salient_spots_count_only_where_the_frame_shows():
    from bozkir.frames import salience, zone
    n, px = 60, 0.025
    yy, xx = np.mgrid[0:n, 0:n] * px

    def spot(img, cx, cy):
        return img - 0.25 * np.exp(-(np.hypot(xx - cx, yy - cy) / 0.12) ** 2)
    imgs = [_texture("blobs", s) for s in range(6)]
    imgs += [spot(_texture("blobs", 10), 0.3, 0.3), spot(_texture("blobs", 11), 0.75, 0.75)]
    ring = zone(n, 1.5, 0.34)
    sal = salience([(i, None) for i in imgs], [ring] * len(imgs))
    assert sal[6] > 3 * max(sal[:6].max(), 1e-3), sal      # a spot in the frame
    assert sal[7] <= max(sal[:6].max(), 1e-3) * 1.5, sal   # one in the middle: hidden


def test_height_order_measures_how_materials_follow_height():
    ew = _ew()
    r = np.random.default_rng(3)

    def pos(x, y, z):
        return (1.0, (x, y), scene(np.array([[x, y, z]] * 8, np.float32)), {})
    xy = r.uniform(0, 12, (300, 2))
    hill = 0.3 * np.sin(xy[:, 0]) * np.cos(xy[:, 1])        # hollows and crests
    low = [pos(x, y, h) for (x, y), h in zip(xy, hill) if h < 0]
    high = [pos(x, y, h) for (x, y), h in zip(xy, hill) if h >= 0]
    assert bozkir.materials.height_order([low, high]) < 0.1             # by height
    mixed = [pos(x, y, h) for (x, y), h in zip(xy, hill)]
    p = bozkir.materials.height_order([mixed[::2], mixed[1::2]])
    assert 0.35 < p < 0.65, p                               # unrelated
    assert bozkir.materials.height_order([high, low]) > 0.9               # the other way round


def test_a_tile_is_checked_for_holes_with_its_neighbours_edges():
    """The ground just past an edge belongs to the neighbour; seen alone, a
    tile is thin there. Its neighbours' edge bands come along, moved over."""
    ew = _ew()
    a = _source_patch(0, 0, n=4000)
    b = _source_patch(9, 9, n=4000)
    codes = [(0, 0, 0, 0), (1, 1, 1, 1)]
    seen = bozkir.holes.with_neighbours([a, b], codes, [0, 0], 1.0, band=0.1)
    x, y = seen[0].xyz[:, 0], seen[0].xyz[:, 1]
    assert len(seen[0]) > len(a)
    for far in (x > 0.5, x < -0.5, y > 0.5, y < -0.5):
        assert 0 < far.sum() and np.abs(np.r_[x[far], y[far]]).max() <= 0.6 + 1e-6
    alone = bozkir.holes.with_neighbours([a], [codes[0]], [0], 1.0)[0]       # its own edges match
    assert len(bozkir.holes.with_neighbours([a, b], codes, [0, 1], 1.0)[0]) == len(alone), \
        "a tile of another material lent its edge"


def test_the_fill_can_spare_large_faint_splats():
    """--fill-spare-haze: large faint splats keep their opacity; the rest
    are filled as before."""
    from bozkir.tile import fill_opacity, haze_splats
    r = np.random.default_rng(5)
    n = 3000
    t = scene(np.c_[r.uniform(-0.5, 0.5, (n, 2)), np.zeros(n)].astype(np.float32),
              opacity=np.full(n, 0.3, np.float32), scale=np.full((n, 3), 0.01, np.float32))
    big = np.arange(n) < 200
    t.scale[big] = 0.06
    t.opacity[big] = 0.1
    t.xyz[big, 2] = 0.2                                     # standing above the ground
    h = haze_splats(t, 1.0)
    assert not haze_splats(scene(t.xyz * [1, 1, 0], opacity=t.opacity, scale=t.scale), 1.0).any()
    assert h[big].all() and not h[~big].any()
    a = fill_opacity(t, 1.0, target=0.98)
    b = fill_opacity(t, 1.0, target=0.98, spare=h)
    assert (a.opacity[big] > t.opacity[big]).any()
    assert np.array_equal(b.opacity[big], t.opacity[big])
    assert np.array_equal(a.opacity[~big], b.opacity[~big])


def test_a_centre_is_judged_by_how_sharp_it_is_where_it_shows():
    """--pool-sharpness: a candidate with larger splats than the edges is
    left out; one with fewer splats over its whole cut but splats as fine is
    not (the old count left it out)."""
    ew = _ew()
    from bozkir.frames import geometry
    geo = geometry(1.5, 0, 15.0, 6.0)

    def cand(x, seed, n, size_m, half=0.75):
        p = _flat_patch(seed, [0.5, 0.45, 0.4], n=n, half=half)
        p.scale[:, :2] = size_m
        return (1.0, (x, 0.0), p, {})
    edges = [cand(0, 1, 3000, 0.01), cand(3, 2, 3000, 0.01)]
    blurry = cand(6, 3, 3300, 0.03)
    small_cut = cand(9, 4, 2000, 0.01, half=0.66)
    pool, _, _, why = bozkir.pick.pick_pool(edges, [blurry, small_cut], 5, 1.5, 1.5, geo,
                                   middles=False, by_sharpness=True)
    assert [c[1][0] for c in pool] == [9], why
    assert any("larger splats" in k and v == 1 for k, v in why.items()), why
    pool, _, _, why = bozkir.pick.pick_pool(edges, [blurry, small_cut], 5, 1.5, 1.5, geo, middles=False)
    assert 9 not in [c[1][0] for c in pool], "the old count let the small cut in"


def test_what_a_frame_shows_of_an_edge_patch():
    """Centred on the tile, a patch's frame is all but its middle; on its
    edge (--diamonds), its diamond but the tips the tile's middle falls on,
    on either axis (the axis is decided later)."""
    from bozkir.frames import geometry, shown, zone
    g = geometry(1.5, 0, 15.0, 6.0)
    assert np.array_equal(shown(60, 1.5, g["inner"], False, 6.0),
                          zone(60, 1.5, g["inner"], shape=6.0))
    d = shown(60, 1.5, g["inner"], True, 6.0)
    c = (np.arange(60) + 0.5) / 60 * 1.5 - 0.75
    diamond = np.abs(c[None, :]) + np.abs(c[:, None]) <= 0.75
    assert not (d & ~diamond).any() and d.sum() > 0.95 * diamond.sum()
    assert d[30, 30] and d[0 + 3, 30], "the middle and the north tip both show"


def test_the_centre_pool_stays_off_the_frames():
    ew = _ew()
    from bozkir.frames import geometry
    geo = geometry(1.5, 0.6, 15.0)

    def cand(x, y, seed):
        return (1.0, (x, y), _flat_patch(seed, [0.5, 0.45, 0.4], n=1500, half=0.75), {})
    edges = [cand(0, 0, 1), cand(3, 0, 2)]
    material = [cand(0.6, 0.0, 3),     # over edge 0's frame
                cand(0.02, 0.0, 4),    # on edge 0's middle, which the pool has
                cand(6.0, 0.0, 5),     # clear
                cand(6.3, 0.0, 6),     # overlaps the one before
                cand(0.0, 3.0, 7)]     # clear
    pool, mids, shared, why = bozkir.pick.pick_pool(edges, material, 10, 1.5, 1.5, geo)
    got = sorted(c[1] for c in pool)
    assert why["candidates"] == 5 and why["too close to a centre taken"] == 3, why
    assert mids == 2 and not any(shared) and got == [(0, 0), (0.0, 3.0), (3, 0), (6.0, 0.0)], got
    pool, _, _, _ = bozkir.pick.pick_pool(edges, material, 3, 1.5, 1.5, geo)
    assert len(pool) == 3


def test_a_short_pool_takes_ground_the_frames_share_only_when_asked():
    """--diamonds: an edge patch shows its diamond. A material with no
    ground clear of it gets no centre, unless --pool-share lets a centre
    share some with the frames: then the least shared first, and none
    beyond the share."""
    ew = _ew()
    from bozkir.frames import geometry, shared_with_frames
    geo = geometry(1.5, 0, 15.0, 6.0)

    def cand(x, y, seed):
        return (1.0, (x, y), _flat_patch(seed, [0.5, 0.45, 0.4], n=1500, half=0.75), {})
    edges = [cand(0, 0, 1), cand(1.5, 0, 2)]
    material = [cand(0.75, 0.0, 3),    # between the two diamonds' tips
                cand(0.0, 0.0, 4),     # on edge 0 itself
                cand(0.75, 1.5, 5)]    # half off
    shares = [shared_with_frames(c[1], [e[1] for e in edges], 1.5, geo, 1.5, True)
              for c in material]
    assert shares[1] > 0.8 and shares[2] < 0.02 < shares[0] < shares[1], shares
    pool, mids, _, why = bozkir.pick.pick_pool(edges, material, 10, 1.5, 1.5, geo, middles=False,
                                 diamonds=True)
    assert mids == 0 and not pool and why["on the frames' ground"] == 3, why
    pool, _, shared, _ = bozkir.pick.pick_pool(edges, material, 10, 1.5, 1.5, geo, middles=False,
                                   share=0.6, diamonds=True)
    got = [c[1] for c in pool]
    assert got == [(0.75, 1.5), (0.75, 0.0)] and shared[0] < 0.02 < shared[1] <= 0.6, shared


def test_quiet_edges_leave_spotted_frames_out():
    ew = _ew()
    from bozkir.ply import SH_C0

    def cand(x, seed, spotted):
        p = _flat_patch(seed, [0.55, 0.5, 0.42], std=0.02, n=9000, half=0.75)
        if spotted:
            d = np.hypot(p.xyz[:, 0] - 0.5, p.xyz[:, 1] - 0.5)
            rgb = p.base_rgb.copy()
            rgb[d < 0.12] = [0.2, 0.22, 0.12]
            p.sh_dc = ((rgb - 0.5) / SH_C0).astype(np.float32)
        return (1.0, (x, 0.0), p, {})
    cands = [cand(2 * k, k, k < 2) for k in range(6)]        # the first two spotted
    picked, line = bozkir.pick.quiet_edges(cands[:2], cands, 2, 1.5, 0.34, 1.5)
    assert line.startswith("quiet edges: yes"), line
    assert all(c[1][0] >= 4 for c in picked), [c[1] for c in picked]


def test_levels_of_detail_keep_the_tiles_colour():
    """desert11: the levels keep the largest splats, and a bush's small dark
    ones on top went, so distant bush came out pale."""
    from bozkir.pack import seen_colour, tile_levels
    from bozkir.ply import SH_C0
    r = np.random.default_rng(4)
    n_g, n_b = 6000, 4000
    xy = np.r_[r.random((n_g, 2)) * 1.5 - 0.75, (r.random((n_b, 2)) * 2 - 1) * 0.2]
    z = np.r_[np.zeros(n_g), 0.05 + 0.2 * r.random(n_b)]
    rgb = np.r_[np.tile([0.7, 0.65, 0.55], (n_g, 1)), np.tile([0.2, 0.3, 0.12], (n_b, 1))]
    sc = np.r_[np.full((n_g, 3), [0.04, 0.04, 0.005]), np.full((n_b, 3), 0.015)]
    t = Splats(xyz=np.c_[xy, z].astype(np.float32),
               opacity=np.r_[np.full(n_g, 0.95), np.full(n_b, 0.6)].astype(np.float32),
               scale=sc.astype(np.float32), rot=np.tile(np.float32([1, 0, 0, 0]), (n_g + n_b, 1)),
               sh_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
               sh_rest=np.zeros((n_g + n_b, 0, 3), np.float32), sh_degree=0)
    want = seen_colour(t)
    plain = tile_levels(t, 4, 1.5)
    toned = tile_levels(t, 4, 1.5, tone=True)
    drift = max(float(np.abs(seen_colour(p) - want).max()) for p in plain[1:])
    left = max(float(np.abs(seen_colour(p) - want).max()) for p in toned[1:])
    assert drift > 0.03 and left < 0.01, (drift, left)


def test_the_far_view_sees_what_stands_up_and_never_black():
    """The atlas: a bare spot lies over the entry's own colour, not black;
    the entries are the tiles then the centres; and each says its colour
    from low down, darker than from above where a bush stands."""
    from bozkir.pack import pack, read_splat, tile_atlas, seen_colour
    import json
    import tempfile
    bush = _flat_patch(2, [0.2, 0.3, 0.12], n=6000, half=0.3)
    bush.xyz[:, 2] += np.random.default_rng(3).random(len(bush)).astype(np.float32) * 0.3
    bush.scale[:] = 0.015
    ground = _flat_patch(1, [0.7, 0.65, 0.55], n=8000, half=0.75)
    ground = ground.subset(np.flatnonzero(np.hypot(*ground.xyz[:, :2].T) > 0.2))  # a hole
    tile = ground.subset(np.arange(len(ground)))
    from bozkir.tile import merge
    with_bush = merge(ground, bush)
    parts = [tile, with_bush, ground]
    tiles, cursor, bufs = [], 0, []
    for p in parts:
        bufs.append(pack(p))
        tiles.append({"start": cursor, "count": len(p), "levels": [[cursor, len(p)]]})
        cursor += len(p)
    m = {"size": 1.5, "up_axis": 2, "tiles": tiles[:2], "centres": tiles[2:], "total": cursor}
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "t.splat").write_bytes(np.concatenate(bufs).tobytes())
        (Path(d) / "t.json").write_text(json.dumps(m))
        m2, s = read_splat(Path(d) / "t.splat", Path(d) / "t.json")
    img, lay = tile_atlas(m2, s, res=32)
    assert lay["count"] == 3 and lay["tiles"] == 2 and len(lay["seen"]) == 3
    assert (img[..., 3] == 255).all()
    hole = img[16 - 3:16 + 3, 16 - 3:16 + 3, :3].astype(int)
    assert hole.min() > 100, "a bare spot came out dark"
    low = np.array(lay["seen"][1])
    top = img[:32, 32:64, :3].reshape(-1, 3).mean(axis=0) / 255
    assert low[0] < top[0] - 0.03, (low, top)               # more bush from low down


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
    for path in sorted((root / "scripts").rglob("*.py")):
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
    scripts = sorted((root / "scripts").rglob("*.py"))
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


def test_dem_window_is_square_in_metres():
    """A 1 arc-second DEM at 37 deg N has posts about 25 x 31 m; the window
    comes back square in metres, the size asked for, at the coarser spacing."""
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        return
    import tempfile
    from bozkir.terrain import read_dem_window
    step = 1 / 3600
    z = np.add.outer(np.arange(400.0), np.arange(400.0)).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "dem.tif"
        with rasterio.open(path, "w", driver="GTiff", width=400, height=400, count=1,
                           dtype="float32", crs="EPSG:4326",
                           transform=from_origin(-119.0, 37.1, step, step)) as dst:
            dst.write(z, 1)
        a, good, res = read_dem_window(path, 37.05, -118.95, 3.0)
    assert a.shape[0] == a.shape[1] and good.all()
    assert abs(res * a.shape[0] - 3000.0) < 1e-6
    assert 30.0 < res < 32.0, res                    # the 31 m north-south posts


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
