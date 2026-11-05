"""The figures in the README, regenerated from the code that produced them.

    python scripts/figures.py            all of them, into docs/
    python scripts/figures.py rule       only those whose name contains 'rule'

Everything is computed on the spot (the rule by bozkir/landform.py, the
ordering by scripts/probe_order.py), so a figure cannot drift from the code.
Screenshots of the viewer are separate (docs/hero*.jpg).
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
    # Generated, not read from web/data, so the figure does not depend on
    # whichever height file is on disk.
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


def _tile_images(res=48, name="desert", tiles=None, centres=None):
    """Tiles and centres of a web/data tileset rendered straight down, north
    up, on the same square: (manifest, tile images, centre images). All of
    them, or the indices in `tiles` and `centres`."""
    from bozkir.graphcut import render_patch
    from bozkir.pack import read_splat
    data = ROOT / "web" / "data"
    m, s = read_splat(data / f"{name}.splat", data / f"{name}.json")

    def render(entry):
        a, c = entry["levels"][0]
        rgb, _ = render_patch(s.subset(np.arange(a, a + c)), m["size"], res,
                              m.get("up_axis", 2))
        return np.clip(rgb, 0, 1)

    tl, cl = m["tiles"], m.get("centres", [])
    return (m, [render(tl[i]) for i in (range(len(tl)) if tiles is None else tiles)],
            [render(cl[i]) for i in (range(len(cl)) if centres is None else centres)])


def _centre_mask(m, res):
    """Where a centre covers its cell, as the splat shader blends it: a
    rounded square (p-norm ball) of the frame radius, faded over the ring."""
    fr = m.get("frames") or {}
    size, r = m["size"], fr.get("radius", 0.0)
    ring, p = max(fr.get("ring", 0.15), 1e-6), fr.get("shape", 2.0)
    x = (np.arange(res) + 0.5) / res * size - size / 2
    d = (np.abs(x)[None, :] ** p + np.abs(x)[:, None] ** p) ** (1 / p)
    return np.clip((r - d) / ring + 0.5, 0, 1)


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
    """The rule off and on, rendered from the real tiles and centres straight
    down with the terrain as shading; only where each material goes differs."""
    plt = _plt()
    m, imgs, cimgs = _tile_images()
    tiles, res = m["tiles"], imgs[0].shape[0]
    k = max(1, m.get("classes", 2))
    share = np.asarray(m.get("shares") or [1 / k] * k, float)
    share /= share.sum()
    z, sed = terrain(0)
    rule = classify(resample(z, N * OVER).ravel(), N, k,
                    sediment=resample(sed, N * OVER).ravel(), balance=list(share),
                    sharpness=3, coherence=2, altitude=0.4)["cls"]
    rng = np.random.default_rng(0)
    off = np.repeat(np.arange(k), np.round(share * N * N).astype(int))[:N * N]
    off = rng.permutation(np.pad(off, (0, N * N - len(off)), mode="edge"))
    # Decal copies (with a `base`) are placed sparsely by the viewer; left out.
    pools = [[c for c, e in enumerate(m.get("centres", []))
              if e["class"] == cls and "base" not in e]
             for cls in range(k)]
    mask = _centre_mask(m, res)[..., None] if cimgs else None
    shade = 0.62 + 0.38 * hillshade(resample(z, N * res))
    panels = []
    for want in (off, rule):
        g = _wang_layout(tiles, want)
        crng = np.random.default_rng(1)
        mos = np.zeros((N * res, N * res, 3))
        for j in range(N):
            for i in range(N):
                cell = imgs[g[j, i]]
                pool = pools[tiles[g[j, i]]["class"]]
                if mask is not None and pool:
                    # A centre from the cell's material, quarter-turned.
                    c = np.rot90(cimgs[pool[int(crng.integers(len(pool)))]],
                                 int(crng.integers(4)))
                    cell = cell * (1 - mask) + c * mask
                r0 = (N - 1 - j) * res                  # north at the top
                mos[r0:r0 + res, i * res:(i + 1) * res] = cell
        # The terrain grid has row 0 in the north too (image order).
        panels.append(np.clip(mos * shade[..., None] * 1.15, 0, 1))
    shares = " / ".join(f"{s:.0%}" for s in share)
    fig, ax = plt.subplots(1, 2, figsize=(14, 7.3))
    for a, img, title in zip(ax, panels,
                             ("rule off: each cell's material at random",
                              "rule on: placed by terrain")):
        a.imshow(img)
        a.set_title(title)
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"The desert tileset on generated desert terrain, {N}x{N} cells, "
                 f"top-down: {k} materials at {shares}, tiles with their centres, "
                 "no blending")
    fig.savefig(out / "fig_map.png")
    plt.close(fig)


WORLDS = ("desert", "river")


def fig_materials(out, worlds=WORLDS, edges=6, middles=4, res=160):
    """Each public tileset's materials as the viewer gets them: a row per
    material, its first edge tiles and centre pieces straight down, with the
    counts and the share of the ground it was given."""
    import json
    plt = _plt()
    about = json.loads((ROOT / "web" / "tilesets.json").read_text())
    rows = []
    for w in worlds:
        m = json.loads((ROOT / "web" / "data" / f"{w}.json").read_text())
        for k in range(m["classes"]):
            ti = [i for i, t in enumerate(m["tiles"]) if t["class"] == k]
            # Decal copies (with a `base`) are placed sparsely; not the pool.
            ci = [i for i, c in enumerate(m.get("centres", []))
                  if c["class"] == k and "base" not in c]
            _, timg, cimg = _tile_images(res, w, ti[:edges], ci[:max(middles, edges)])
            if cimg:
                # As the viewer shows them: each edge tile with a centre in.
                mask = _centre_mask(m, res)[..., None]
                timg = [t * (1 - mask) + cimg[i % len(cimg)] * mask
                        for i, t in enumerate(timg)]
            cimg = cimg[:middles]
            rgb = np.mean([im.reshape(-1, 3).mean(0) for im in timg], axis=0)
            names = about.get(w, {}).get("names")
            name = names[k] if names else None
            share = (m.get("shares") or [1 / m["classes"]] * m["classes"])[k]
            rows.append([w, name, timg, cimg, len(ti), len(ci), share, rgb])
        # Unnamed materials by what they look like: green, else by brightness.
        mine = [r for r in rows if r[0] == w and r[1] is None]
        bare = sorted((r for r in mine if r[7][1] <= max(r[7][0], r[7][2]) + 0.01),
                      key=lambda r: r[7].mean())
        for r in mine:
            r[1] = ("scrub" if not any(r is b for b in bare) else
                    "sand" if len(bare) == 1 else
                    "darker sand" if r is bare[0] else
                    "pale sand" if r is bare[-1] else "sand")
    cols = edges + 1 + middles
    fig, ax = plt.subplots(len(rows), cols, figsize=(1.6 * cols, 1.85 * len(rows)),
                           gridspec_kw={"width_ratios": [1] * edges + [0.25] + [1] * middles})
    for r, (w, name, timg, cimg, nt, nc, share, _) in enumerate(rows):
        for c in range(cols):
            a = ax[r, c]
            a.set_xticks([])
            a.set_yticks([])
            img = (timg[c] if c < edges and c < len(timg) else
                   cimg[c - edges - 1] if c > edges and c - edges - 1 < len(cimg) else None)
            if img is None:
                a.axis("off")
                continue
            a.imshow(img)
        ax[r, 0].set_ylabel(f"{w}\n{name}", fontsize=10)
        ax[r, 0].set_title(f"{nt} tiles, {share:.0%} of the ground", loc="left",
                           fontsize=9)
        ax[r, edges + 1].set_title(f"{nc} centre piece{'s' if nc > 1 else ''}" if nc else
                                   "no centres: whole tiles", loc="left", fontsize=9)
    fig.suptitle(f"The materials of both worlds, straight down, each square "
                 f"{rows and json.loads((ROOT / 'web' / 'data' / f'{worlds[0]}.json').read_text())['size']:g} m: "
                 "tiles as the viewer joins them (edge tile plus a centre piece), and the centre pieces",
                 fontsize=10)
    fig.subplots_adjust(wspace=0.05, hspace=0.35)
    fig.savefig(out / "fig_materials.png")
    plt.close(fig)


def fig_profiles(out):
    """Every named terrain at one seed, a row per group as the terrain
    window shows them: what the generator can make."""
    from bozkir.erosion import PROFILES
    # The grouping of GROUPS in web/terrain.js. Checked against PROFILES so
    # a profile added on one side cannot silently go missing here.
    groups = [("lowlands", ["plains", "steppe", "rolling"]),
              ("hills", ["hills", "terraced", "foothills"]),
              ("desert", ["desert", "dunes", "badlands", "buttes"]),
              ("highlands", ["mesa", "plateau", "canyon", "gorge", "piedmont"]),
              ("mountains", ["ridges", "alpine", "crags", "scree"])]
    listed = [n for _, names in groups for n in names]
    assert sorted(listed) == sorted(PROFILES), "figure groups out of date"
    plt = _plt()
    cols = max(len(names) for _, names in groups)
    fig, ax = plt.subplots(len(groups), cols, figsize=(2.6 * cols, 2.8 * len(groups)))
    for row, (group, names) in zip(ax, groups):
        for k, a in enumerate(row):
            a.set_xticks([])
            a.set_yticks([])
            if k >= len(names):
                a.axis("off")
                continue
            z, _ = generate(size=256, seed=0, profile=names[k])
            a.imshow(hillshade(z), cmap="gray")
            a.contour(z, levels=8, colors="k", linewidths=0.3, alpha=0.5)
            a.set_title(names[k])
        row[0].set_ylabel(group, fontsize=12)
    fig.suptitle("Named terrains, seed 0: noise, then stream power erosion "
                 "(FastScape scheme)")
    fig.savefig(out / "fig_profiles.png")
    plt.close(fig)


def fig_water(out):
    """The river's water as measured from the L2 drone photos
    (web/data/water.json, scripts/water_texture.py): the shallow and deep
    tiles at their ground size, and the share of the bed still seen
    against distance from the waterline with its fitted curve."""
    import json
    from PIL import Image
    plt = _plt()
    data = ROOT / "web" / "data"
    meta = json.loads((data / "water.json").read_text())
    lay = meta["layers"]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2), gridspec_kw={"width_ratios": [1, 1, 1.5]})
    for a, (img, m, title) in zip(ax, [("water.png", meta, "shallow water"),
                                       ("water_deep.png", meta["deep"], "deep channel")]):
        a.imshow(Image.open(data / img))
        a.set_title(title + "\n" + f"{m['source']}, {m['metres']:.2f} m across")
        a.axis("off")
    b = lay["bands"]
    mid = [(x["from_m"] + x["to_m"]) / 2 for x in b]
    wid = [x["to_m"] - x["from_m"] for x in b]
    ax[2].bar(mid, [x["t"] for x in b], width=[0.9 * w for w in wid], color="#9fb7b4",
              label="measured (stone contrast)")
    d = np.linspace(0, b[-1]["to_m"], 100)
    ax[2].plot(d, lay["t0"] * np.exp(-d / lay["length_m"]), color="#9a6b1f",
               label=f"T = {lay['t0']:.2f} exp(-d / {lay['length_m']:.2f} m)")
    ax[2].set_xlabel("distance from the waterline (m)")
    ax[2].set_ylabel("share of the bed seen, T")
    ax[2].set_ylim(0, 1.05)
    ax[2].set_title(f"bed under the water, {len(lay['photos'])} photos")
    ax[2].legend(frameon=False, fontsize=9)
    fig.subplots_adjust(wspace=0.35)
    fig.savefig(out / "fig_water.png")
    plt.close(fig)


FIGURES = {"rule": fig_rule, "seeds": fig_seeds, "balance": fig_balance,
           "ordering": fig_ordering, "profiles": fig_profiles, "map": fig_map,
           "water": fig_water, "materials": fig_materials}


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