# Captures: which ones work as tiling exemplars

A capture is only useful to bozkır if a few square metres of it can stand in
for kilometres of ground. This page records how each capture tried so far
screened, and why. The checks behind the columns live in
`scripts/inspect_ply.py` and `scripts/make_tileset.py` (the one-command
verdict: good / marginal / no).

## Screening (2026-09-23)

| capture | splats | planarity | tilt | anisotropy (median) | up axis (raw) | verdict |
|---|---|---|---|---|---|---|
| desert | 1.8M | good | low | 12.8 | Z | good: the reference tileset |
| beachrock | 3M | 0.013 | 4.4° | 12.8 | Y | good: flat, healthy geometry |
| bigsur | 10M | 0.168 | 81° raw, 3° aligned | 3.8 | Y, Z after alignment | works with tuning (flip, graph cut, splat cap) |
| bicycle | 1M | - | - | - | Z | not ground; viewed as a plain scene |
| garden | 730k | - | - | - | Z | not ground; viewed as a plain scene |
| italy | 5M | 0.078 | 18.9° | 9.7 | Y | marginal: steep, needs loose thresholds |
| minecraft | 3M | 0.060 | 6.9° | 14.6 | Y | marginal: synthetic, tight extent clipping |
| mitsukejima | 2.9M | 0.092 | 2.8° | **483.8** | Y | no: a vertical sea stack, not ground |

The verdicts above are what the numbers allowed. Looked at as tilesets,
only desert held up, so it is the only one used and the only one published;
the others stay as screening cases.

desert is the owner's own drone capture: aligned in RealityScan, trained in
Brush (SH degree 3), with an OpenDroneMap DTM and orthophoto of the same plot
used for the rule validation.

## What each difficult capture taught

**bigsur (10M splats, captured at 81°).**
- *Upside down after alignment.* The ground-normal estimate picked the wrong
  sign. Fixed with `flip`. The upside-down check (`upness`) is one-way: a
  warning means something, silence proves nothing.
- *200 MB tileset.* Tiles held 270k-440k splats each. `max_per_tile: 50000`
  brought the whole set to ~650k splats (21 MB) and the frame rate back up.
- *Slow search.* Floater removal builds a KD-tree over all 10M splats, and
  patch extraction indexed the whole scene per candidate. The plane index
  (`tile.PlaneIndex`) fixed the second; skip `floater_std` on scenes this
  dense.
- *X-shaped seams.* Straight feathered diagonals showed wherever adjoining
  patches differed. Graph-cut seams (`cut`, `cut_band 0.14`, `cut_res 160`)
  follow agreement between patches instead.

**mitsukejima.** Median anisotropy ~484 means the splats are needles, not
discs: the reconstruction is a vertical rock, not ground. Left out of the
presets and of the public data (the tileset is also 316 MB).

## Warning thresholds used by the exporter

- **Appearance spread** (`bozkir/wang.py`): the spread of mean colour and
  normal among the chosen patches. Below 0.05 seams are clean; above 0.10
  tile boundaries show colour and texture breaks. Above 0.05, keep graph cut
  on.
- **Axis balance**: divergence between the patches on horizontal and vertical
  edges. Large values point at directional light or a slope that will read
  as a repeating stripe. Skipped when rotations are on, because turned
  patches mix the two sets.

## What a good new capture looks like

From the desert experience and the null rule validation:
- **Overcast light.** Baked shadows turn with rotated patches and repeat
  with every tile. A low sun also inflated the colour truth in the
  validation.
- **Close and dense.** Low altitude (roughly 3-8 m) or handheld, nadir plus
  30-45° obliques, 80%+ overlap. Splat count is not the limit (the exporter
  caps patches at 60k); sharpness and few floaters are.
- **Each material over a generous area.** The desert scrub is only ~4 × 5 m
  of capture, so its patches overlap and are shifted copies.
- **A separate survey of the surroundings** (tens of metres up, 100 m+
  across several landforms) if the capture is also to test the material rule.
