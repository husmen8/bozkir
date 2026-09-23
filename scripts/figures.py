"""The figures in the README, regenerated from the code that produced them.

    python scripts/figures.py            all of them, into docs/
    python scripts/figures.py rule       only those whose name contains 'rule'

Every number drawn here is computed on the spot - the rule by
bozkir/landform.py (held to the viewer's web/landform.js by the tests), the
ordering by the same probe as scripts/probe_order.py - so a figure cannot
drift from the code the way a pasted screenshot can.

Renders of the splats themselves still come from the viewer; these are the
maps and measurements behind them.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bozkir.erosion import generate                       # noqa: E402
from bozkir.landform import classify                      # noqa: E402
from bozkir.terrain import read_height_png                # noqa: E402

N = 24          # the grid the README's settings use
OVER = 4        # the viewer samples the terrain 4x per tile
SHARE = 0.30    # "first material 30%"
SAND, SCRUB = "#d9c9a9", "#4c573a"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10,
                         "figure.dpi": 130, "savefig.bbox": "tight"})
    return plt


def resample(a, m):
    """Bilinear resample of a square field onto m*m cell-centred samples,
    the whole field spanning the grid (relief scale = grid size)."""
    h, w = a.shape
    y = (np.arange(m) + 0.5) * h / m - 0.5
    x = (np.arange(m) + 0.5) * w / m - 0.5
    y0 = np.clip(np.floor(y).astype(int), 0, h - 2)
    x0 = np.clip(np.floor(x).astype(int), 0, w - 2)
    fy = np.clip(y - y0, 0, 1)[:, None]
    fx = np.clip(x - x0, 0, 1)[None, :]
    a00 = a[y0][:, x0]
    a01 = a[y0][:, x0 + 1]
    a10 = a[y0 + 1][:, x0]
    a11 = a[y0 + 1][:, x0 + 1]
    return (a00 * (1 - fx) + a01 * fx) * (1 - fy) + (a10 * (1 - fx) + a11 * fx) * fy


def hillshade(z, az=315, alt=40):
    gy, gx = np.gradient(z * z.shape[0] * 0.06)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, e = np.deg2rad(az), np.deg2rad(alt)
    return np.clip(np.sin(e) * np.cos(slope)
                   + np.cos(e) * np.sin(slope) * np.cos(a - aspect), 0, 1)


def terrain(seed):
    """The desert scene's terrain for seed 0 (read from web/data if it is
    there), otherwise generated with the same settings."""
    # Generated rather than read from web/data: the generator is
    # deterministic and bit-identical to the viewer's, so this is the same
    # ground `terrain_gen.py --profile desert` writes, and a figure cannot
    # silently depend on whichever height file happens to be on disk.
    z, sed = generate(size=512, seed=seed, profile="desert")
    return z, sed


def run_rule(z, sed, share=SHARE, **kw):
    m = N * OVER
    return classify(resample(z, m).ravel(), N, 2,
                    sediment=resample(sed, m).ravel(), balance=share,
                    sharpness=3, coherence=2, altitude=0.4, **kw)


def random_classes(share, seed=0):
    """What the viewer does with the rule off: each cell takes whichever
    class's tile the layout happens to pick. Drawn at the same share so the
    comparison is about arrangement, not amount."""
    rng = np.random.default_rng(seed)
    c = np.ones(N * N, int)
    c[rng.permutation(N * N)[:int(round(share * N * N))]] = 0
    return c


def class_image(cls):
    from matplotlib.colors import ListedColormap
    return cls.reshape(N, N), ListedColormap([SCRUB, SAND])


def overlay(ax, z, cls, title):
    img, cmap = class_image(cls)
    ax.imshow(hillshade(z), cmap="gray", extent=(0, N, N, 0))
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1, alpha=0.62,
              extent=(0, N, N, 0), interpolation="nearest")
    # Contours so the reader can see the first material in the low ground
    # without a second panel.
    h, w = z.shape
    ax.contour(np.linspace(0, N, w), np.linspace(0, N, h), z, levels=8,
               colors="k", linewidths=0.35, alpha=0.55)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_rule(out):
    """The contribution in one figure: the maps the rule reads, then the
    class map with the rule off and on."""
    plt = _plt()
    z, sed = terrain(0)
    r = run_rule(z, sed)
    off = random_classes(SHARE)
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.8))
    ax[0, 0].imshow(hillshade(z), cmap="gray")
    ax[0, 0].set_title("generated terrain (eroded)")
    maps = [("elevation", "terrain", "elevation"),
            ("slope", "magma", "slope"),
            ("tpi", "coolwarm", "position (TPI): hollow - ridge")]
    for a, (key, cmap, title) in zip(ax[0, 1:], maps):
        a.imshow(r["maps"][key].reshape(N, N), cmap=cmap)
        a.set_title(title)
    ax[1, 0].imshow(r["maps"]["sediment"].reshape(N, N), cmap="YlOrBr")
    ax[1, 0].set_title("sediment left by erosion")
    ax[1, 1].imshow(r["strength"].reshape(N, N), cmap="viridis", vmin=0,
                    vmax=1)
    ax[1, 1].set_title("confidence (decision margin)")
    overlay(ax[1, 2], z, off, "rule off: class per cell at random")
    overlay(ax[1, 3], z, r["cls"], f"rule on: {SHARE:.0%} collected, placed by terrain")
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Material coverage from terrain geometry "
                 f"({N}x{N} tiles; the class map is Hybrid GSWT's alpha_c, "
                 "computed instead of painted)")
    fig.savefig(out / "fig_rule.png")
    plt.close(fig)


def fig_seeds(out):
    """Same rule, same settings, different terrains: a rule, not a placement."""
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.3))
    for a, seed in zip(ax, (0, 1, 2)):
        z, sed = terrain(seed)
        r = run_rule(z, sed)
        overlay(a, z, r["cls"], f"seed {seed}")
    fig.suptitle("One rule, three terrains, no retuning")
    fig.savefig(out / "fig_seeds.png")
    plt.close(fig)


def fig_balance(out):
    """The requested share is the share given, and the boundary moves with
    the terrain rather than randomly as the share changes."""
    plt = _plt()
    z, sed = terrain(0)
    asks = np.linspace(0.05, 0.95, 19)
    got = [float((run_rule(z, sed, share=s)["cls"] == 0).mean()) for s in asks]
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6),
                           gridspec_kw={"width_ratios": [1.2, 1, 1, 1]})
    ax[0].plot(asks * 100, np.array(got) * 100, "o-", ms=4)
    ax[0].plot([0, 100], [0, 100], "k:", lw=0.8)
    ax[0].set_xlabel("first material asked for (%)")
    ax[0].set_ylabel("first material placed (%)")
    ax[0].set_title("share control is exact")
    for a, s in zip(ax[1:], (0.2, 0.5, 0.8)):
        overlay(a, z, run_rule(z, sed, share=s)["cls"], f"{s:.0%} first material")
    fig.savefig(out / "fig_balance.png")
    plt.close(fig)


def fig_ordering(out):
    """Tile ordering: depth key vs topological, reversed pairs per azimuth.
    Re-runs scripts/probe_order.py and parses its table."""
    plt = _plt()
    txt = subprocess.run([sys.executable, str(ROOT / "scripts" / "probe_order.py")],
                         capture_output=True, text=True, check=True).stdout
    az, key, topo = [], [], []
    for line in txt.splitlines():
        p = line.split()
        if len(p) > 3 and p[0].replace(".", "").isdigit():
            az.append(p[0])
            key.append(int(p[1]))
            topo.append(int(p[2]))
    x = np.arange(len(az))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(x - 0.2, key, 0.4, label="nearest-depth key", color="#c0504d")
    ax.bar(x + 0.2, topo, 0.4, label="topological (GSWT 3.4)", color="#3f6b45")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a} deg" for a in az])
    ax.set_ylabel("tile pairs that swap order\nover a 3 deg turn (9x9 grid)")
    ax.set_title("Depth keys tie along grid axes; the topological sort does not")
    ax.legend()
    ax.annotate("camera in a boundary plane:\nmerging's job",
                xy=(az.index("22.5") + 0.2, 8), xytext=(3.2, 30),
                arrowprops={"arrowstyle": "->", "lw": 0.8}, fontsize=9)
    fig.savefig(out / "fig_ordering.png")
    plt.close(fig)


def _tile_images(res=48):
    """Every tile of the desert tileset rendered straight down, north up."""
    import json
    from bozkir.graphcut import render_patch
    from bozkir.ply import Splats, SH_C0
    m = json.loads((ROOT / "web" / "data" / "desert.json").read_text())
    raw = np.fromfile(ROOT / "web" / "data" / "desert.splat", dtype=np.uint8)
    n = len(raw) // 32
    rec = raw[:n * 32].reshape(n, 32)
    f = rec[:, :24].copy().view(np.float32).reshape(n, 6)
    rgba = rec[:, 24:28].astype(np.float64) / 255
    q = (rec[:, 28:32].astype(np.float64) - 128) / 128
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
    imgs = []
    for t in m["tiles"]:
        a, c = t["levels"][0]
        sp = Splats(xyz=f[a:a + c, :3].astype(np.float64),
                    opacity=rgba[a:a + c, 3], scale=f[a:a + c, 3:6].astype(np.float64),
                    rot=q[a:a + c], sh_dc=(rgba[a:a + c, :3] - 0.5) / SH_C0,
                    sh_rest=np.zeros((c, 0, 3)), sh_degree=0)
        rgb, _ = render_patch(sp, m["size"], res, m.get("up_axis", 2))
        imgs.append(np.clip(rgb, 0, 1))
    return m["tiles"], imgs


def _wang_layout(tiles, want, seed=0):
    """Cells filled south to north, west to east, as the viewer does: each
    takes a tile whose west edge matches its neighbour's east and whose
    south edge matches the north of the cell below, from the class asked
    for. Every edge pair exists in every class, so the choice never fails."""
    rng = np.random.default_rng(seed)
    grid = np.zeros((N, N), int)
    for j in range(N):
        for i in range(N):
            ok = [k for k, t in enumerate(tiles)
                  if t["class"] == want[j * N + i]
                  and (i == 0 or t["w"] == tiles[grid[j, i - 1]]["e"])
                  and (j == 0 or t["s"] == tiles[grid[j - 1, i]]["n"])]
            grid[j, i] = ok[int(rng.integers(len(ok)))]
    return grid


def fig_map(out):
    """The rule off and on, rendered from the tiles themselves.

    Not a screenshot: every cell is the real tile the viewer's layout would
    put there, rendered straight down, with the terrain as shading. The
    only difference between the panels is where each class goes.
    """
    plt = _plt()
    tiles, imgs = _tile_images()
    res = imgs[0].shape[0]
    z, sed = terrain(0)
    rule = run_rule(z, sed)["cls"]
    off = random_classes(SHARE)
    shade = hillshade(resample(z, N * res))
    shade = 0.62 + 0.38 * shade
    panels = []
    for want in (off, rule):
        g = _wang_layout(tiles, want)
        mos = np.zeros((N * res, N * res, 3))
        for j in range(N):
            for i in range(N):
                r0 = (N - 1 - j) * res                  # north at the top
                mos[r0:r0 + res, i * res:(i + 1) * res] = imgs[g[j, i]]
        # The terrain grid has row 0 in the north too (image order).
        panels.append(np.clip(mos * shade[..., None] * 1.15, 0, 1))
    fig, ax = plt.subplots(1, 2, figsize=(14, 7.3))
    for a, img, title in zip(ax, panels,
                             ("rule off: each cell's material at random",
                              f"rule on: {SHARE:.0%} class 0 (collected ground) by terrain")):
        a.imshow(img)
        a.set_title(title)
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"The desert tileset on generated desert terrain, {N}x{N} "
                 "tiles, top-down - same tiles, same share, same layout rules")
    fig.savefig(out / "fig_map.png")
    plt.close(fig)


def fig_profiles(out):
    """The twelve named terrains at one seed: what the generator can make."""
    from bozkir.erosion import PROFILES
    plt = _plt()
    names = list(PROFILES)
    fig, ax = plt.subplots(3, 4, figsize=(13, 10))
    for a, name in zip(ax.ravel(), names):
        z, _ = generate(size=256, seed=0, profile=name)
        a.imshow(hillshade(z), cmap="gray")
        a.contour(z, levels=8, colors="k", linewidths=0.3, alpha=0.5)
        a.set_title(name)
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Named terrains, seed 0: noise, then stream power erosion "
                 "(FastScape scheme)")
    fig.savefig(out / "fig_profiles.png")
    plt.close(fig)


FIGURES = {"rule": fig_rule, "seeds": fig_seeds, "balance": fig_balance,
           "ordering": fig_ordering, "profiles": fig_profiles, "map": fig_map}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("only", nargs="*", help="names to make: " + ", ".join(FIGURES))
    ap.add_argument("--out", default=str(ROOT / "docs"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        if args.only and not any(o in name for o in args.only):
            continue
        fn(out)
        print(f"wrote {out / ('fig_' + name + '.png')}")


if __name__ == "__main__":
    main()