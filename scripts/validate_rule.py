"""Score the material rule against where material really is.

    python scripts/validate_rule.py C:/odm/copr/odm_dem/dtm.tif ^
        C:/odm/copr/odm_orthophoto/odm_orthophoto.tif ^
        --dsm C:/odm/copr/odm_dem/dsm.tif

The viewer places scrub and sand by the shape of the ground. A drone survey
can check that: the DTM is the shape, the orthophoto is where the scrub
actually grows. This lines the two up on the viewer's cell grid, runs the
viewer's own rule on the DTM (bozkir/landform.py, held to web/landform.js by
the test suite), and reports how well the rule's map agrees with the real
one - with the chance level measured by shifting the truth around a torus,
because two clumpy maps overlap by chance far more than two random ones.

Truth comes from colour (nearest of the two tile classes' mean colours) and,
with --dsm, independently from vegetation height (DSM minus DTM). Both are
reported; they fail differently, colour on shadows and height on the ground
filter, so agreement between them is worth more than either.

Writes a figure and a JSON of every number. See bozkir/validate.py for what
each number means and why the test is set up this way.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bozkir.validate import (cell_fraction, colour_truth, evaluate,  # noqa: E402
                             height_truth, read_geotiff, sample_bilinear,
                             square_window, Raster)


def fine_grid(x_min, y_min, side, m):
    """World coordinates of an m*m sample grid, row-major from north-west."""
    c = (np.arange(m) + 0.5) * side / m
    xs, ys = np.meshgrid(x_min + c, y_min + side - c)
    return xs, ys


def truth_rasters(args, dtm, ortho):
    out = {}
    rgb = ortho.data
    out["colour"] = Raster(colour_truth(rgb), ortho.valid, ortho.x0,
                           ortho.y0, ortho.dx, ortho.dy)
    if args.dsm:
        dsm = read_geotiff(args.dsm)
        h, w = dtm.valid.shape
        rows, cols = np.mgrid[0:h, 0:w]
        x = dtm.x0 + (cols + 0.5) * dtm.dx
        y = dtm.y0 - (rows + 0.5) * dtm.dy
        top = sample_bilinear(dsm, x, y)
        out["height"] = Raster(height_truth(top, dtm.data, args.min_height),
                               dtm.valid, dtm.x0, dtm.y0, dtm.dx, dtm.dy)
    return out


def table(results):
    rows = [f"  {'':22s} {'kappa':>6s} {'acc':>6s} {'IoU':>6s} "
            f"{'rho':>6s} {'chance95':>9s} {'p':>7s}"]
    for name, r in results.items():
        if name.startswith("_"):
            continue
        rows.append(f"  {name:22s} {r['kappa']:6.3f} {r['accuracy']:6.3f} "
                    f"{r['iou']:6.3f} {r['spearman']:6.3f} "
                    f"{r['null_p95']:9.3f} {r['p']:7.4f}")
    return "\n".join(rows)


def figure(path, ortho, window, n, truth_frac, res, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    x_min, y_min, side = window
    xs, ys = fine_grid(x_min, y_min, side, 600)
    rows, cols = ortho.world_to_pixel(xs, ys)
    h, w = ortho.valid.shape
    rgb = ortho.data[np.clip(np.round(rows).astype(int), 0, h - 1),
                     np.clip(np.round(cols).astype(int), 0, w - 1)]

    rule = res["rule"]
    pred = (res["_cls"] == 0).reshape(n, n)
    truth = rule["truth"].reshape(n, n)
    agree = np.where(pred & truth, 3, np.where(pred, 2,
                     np.where(truth, 1, 0)))
    two = ListedColormap(["#d8c7a8", "#4f5a3a"])
    four = ListedColormap(["#eee6d6", "#c0504d", "#e3a33b", "#3f6b45"])
    ext = (0, side, 0, side)

    fig, ax = plt.subplots(2, 3, figsize=(13, 8.6))
    ax[0, 0].imshow(np.clip(rgb, 0, 1), extent=ext)
    ax[0, 0].set_title("orthophoto")
    ax[0, 1].imshow(truth_frac.reshape(n, n), cmap="Greens", vmin=0, vmax=1,
                    extent=ext)
    ax[0, 1].set_title("scrub fraction per cell (truth)")
    ax[0, 2].imshow(res["_maps"]["elevation"].reshape(n, n), cmap="terrain",
                    extent=ext)
    ax[0, 2].set_title("DTM elevation (what the rule reads)")
    ax[1, 0].imshow(truth, cmap=two, vmin=0, vmax=1, extent=ext)
    ax[1, 0].set_title(f"truth at {rule['share']:.0%} scrub")
    ax[1, 1].imshow(pred, cmap=two, vmin=0, vmax=1, extent=ext)
    ax[1, 1].set_title("rule, same share")
    ax[1, 2].imshow(agree, cmap=four, vmin=0, vmax=3, extent=ext)
    ax[1, 2].set_title("green both · orange rule only · red truth only")
    for a in ax.ravel():
        a.set_xlabel("m")
    fig.suptitle(f"{title}   kappa {rule['kappa']:.2f} "
                 f"(chance 95th pct {rule['null_p95']:.2f}, p {rule['p']:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cue_chart(path, results_by_truth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [k for k in next(iter(results_by_truth.values()))
             if not k.startswith("_")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / len(results_by_truth)
    for t, (truth_name, res) in enumerate(results_by_truth.items()):
        ks = [res[k]["kappa"] for k in names]
        chance = [res[k]["null_p95"] for k in names]
        x = np.arange(len(names)) + t * width
        ax.bar(x, ks, width, label=f"truth from {truth_name}")
        ax.scatter(x, chance, marker="_", s=300, color="k", zorder=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(np.arange(len(names)) + 0.4 - width / 2)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Cohen's kappa")
    ax.set_title("agreement with real scrub; black ticks = 95th percentile "
                 "of chance (torus shifts)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dtm", help="bare-ground model, odm_dem/dtm.tif")
    ap.add_argument("ortho", help="orthophoto, odm_orthophoto.tif")
    ap.add_argument("--dsm", help="surface model, for vegetation-height truth")
    ap.add_argument("--grid", type=int, default=24,
                    help="cells per side, as the viewer's grid (default 24)")
    ap.add_argument("--also", default="16,32",
                    help="extra grid sizes to check the result is not an "
                         "artefact of one cell size (default 16,32)")
    ap.add_argument("--altitude", type=float, default=0.4)
    ap.add_argument("--coherence", type=int, default=2)
    ap.add_argument("--sharpness", type=float, default=2.0)
    ap.add_argument("--min-height", type=float, default=0.25,
                    help="DSM-DTM above which a pixel is vegetation (m)")
    ap.add_argument("--out", default="docs",
                    help="where to write validation.png, validation_cues.png "
                         "and validation.json")
    args = ap.parse_args(argv)

    dtm = read_geotiff(args.dtm)
    ortho = read_geotiff(args.ortho)
    window = square_window(dtm, ortho)
    x_min, y_min, side = window
    print(f"window {side:.1f} m square at ({x_min:.1f}, {y_min:.1f}); "
          f"DTM {dtm.dx * 100:.1f} cm/px, ortho {ortho.dx * 100:.1f} cm/px")

    truths = truth_rasters(args, dtm, ortho)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"window": {"x_min": x_min, "y_min": y_min, "side_m": side},
              "grids": {}}

    grids = [args.grid] + [int(g) for g in args.also.split(",") if g.strip()
                           and int(g) != args.grid]
    main_results = {}
    for n in grids:
        m = 4 * n
        xs, ys = fine_grid(x_min, y_min, side, m)
        z = sample_bilinear(dtm, xs, ys).ravel()
        report["grids"][n] = {}
        for tname, traster in truths.items():
            frac = cell_fraction(traster, x_min, y_min, side, n)
            res = evaluate(z, frac, n, side / n, altitude=args.altitude,
                           coherence=args.coherence,
                           sharpness=args.sharpness)
            print(f"\ngrid {n} ({side / n:.2f} m cells), truth from "
                  f"{tname}, {res['rule']['share']:.0%} scrub")
            print(table(res))
            report["grids"][n][tname] = {
                k: {kk: vv for kk, vv in v.items() if kk != "truth"}
                for k, v in res.items() if not k.startswith("_")}
            if n == args.grid:
                main_results[tname] = res
                if tname == "colour":
                    figure(out / "validation.png", ortho, window, n, frac,
                           res, f"{Path(args.dtm).stem}, grid {n}")

    cue_chart(out / "validation_cues.png", main_results)
    (out / "validation.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out / 'validation.png'}, {out / 'validation_cues.png'}, "
          f"{out / 'validation.json'}")
    print("\nReading it: 'rule' is the stated hypothesis (scrub collects "
          "where water runs). It\nmeans something only if its kappa clears "
          "the chance column and holds across grids\nand both truths. A "
          "better score for 'rule, flipped' means the rule is wrong here.")


if __name__ == "__main__":
    main()
