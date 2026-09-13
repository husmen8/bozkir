# bozkır

Procedural terrain from captured Gaussian splats.

A drone photographs a few square metres of coastal ground. The capture is
reconstructed as 3D Gaussians, cut into square patches whose edges match,
and those patches are laid out as Wang tiles into terrain that runs to the
horizon without repeating visibly and without a seam.

Two halves, joined by a file rather than by a server: a Python **constructor**
that turns a capture into a tileset, and a browser **renderer** that lays the
tileset out and draws it. Drop a tileset zip on the page and it loads.

The work is a reimplementation of *Gaussian Splatting Wang Tiles* (Zeng, Ma
and Sander, SIGGRAPH Asia 2025) with the boundary artifacts their paper
names as unsolved taken further.

---

## Running it

**The renderer.** Any static server, since ES modules will not load from
`file://`:

```
cd web && python -m http.server 8000
```

Then open `http://localhost:8000`. A sample tileset loads by itself. To view
another, drop a `.zip` on the page, or the `.splat` and `.json` together.

**The constructor.** Needs `numpy`, `plyfile`, `Pillow`:

```
python scripts/export_wang.py --preset bigsur
```

writes `web/data/bigsur.splat` and `.json`. Settings live in `presets.json`;
`--help` on any script lists the rest.

**Tests.**

```
python tests/run_all.py
```

Three suites, because they need three different things: the package needs
numpy and plyfile, the popping metric needs numpy alone, the browser modules
need node. A missing dependency reports as skipped rather than failed.

---

## What it does

**Patch selection.** A capture is aligned to its ground plane, cleaned of
floaters and background, and searched for square regions that are flat
enough, dense enough, and similar enough to each other to be interchangeable.
Scoring and candidate search are in `bozkir/patches.py`.

**Edge matching.** Wang tiles need edges that pair up. Patches are cut and
recombined so each tile edge carries one of a small set of codes, and a
min-cut seam (`bozkir/graphcut.py`) places the join where the two patches
already agree, rather than down a straight line.

**Layout.** `bozkir/wang.py` builds the tile set and lays out a grid where
every shared edge carries the same code on both sides, with no backtracking
dead ends.

**Rendering.** `web/` is a WebGL2 splat renderer: per-patch counting sort in
a worker, surface warping onto a height field, LOD with cross-fade, sky and
haze, orbit and free-fly cameras, six debug views, and a frame capture for
measurement.

---

## Results

Each of these is a measurement, not an impression. The script that produced
it is named.

**Boundary gap from surface warping** is proportional to curvature times
span and exactly zero in the continuous limit. Halving the frames per tile
halves the gap, across a 16x range.

**One sorted order per tile is not enough** once there is any relief: 47% of
adjacent pairs come out in the wrong order at one frame per tile. GSWT caches
nine orders per tile to cover this. Sorting along `Wᵀv` — the view direction
expressed in the tile's own frame — gives exactly zero error instead, which
is an answer rather than an approximation.

**The tile frame must be the true Jacobian**, not a normalised rotation.
Normalising discards a stretch of `sqrt(1+|∇h|²)`, and the material thins on
slopes by exactly that factor.

**Ordering tiles by depth fails at axis-aligned views.** The minimum
projected depth over a cell's footprint is identical, bit for bit, for every
cell in a row when the view direction lines up with a grid axis — 81 cells
collapse to 9 distinct keys. Whatever breaks the tie then decides the whole
row at once, and rotating through alignment reverses nine cells in a single
frame. GSWT's topological sort over shared-boundary constraints removes it
(`python scripts/probe_order.py`):

```
    az    depth key    topological
   0.0         27            0
  22.5          0            8      <- a boundary plane genuinely crossed
  89.0         54            0
  90.0         27            0
```

The eight that remain at 22.5° are not a regression: the camera crosses that
boundary's plane during the turn, so the order must change. All eight sit at
`|n·(eye−edge)| = 0.254`, which is where selective merging takes over.

**Selective merging agrees with an exact sort** to under one quantisation
bucket of a 16-bit counting sort, checked end to end against splats placed in
world space exactly as the shader places them, including grid rotation.

---

## Layout

```
bozkir/          the constructor
  ply.py         load/save PLY, Splats, spherical harmonics, covariance
  transform.py   quaternions, ground normal, alignment
  select.py      crops, floater and outlier removal
  patches.py     ground detection, scoring, candidate search, LOD selection
  graphcut.py    min-cut seam placement between patches
  tile.py        patch extraction, translation, merge, grid
  wang.py        tile construction, layout, edge verification
  pack.py        the 32-byte .splat binary the browser reads
  scene.py       one load -> align -> clean path, cached by config key
  presets.py     named settings from presets.json
  camera.py      projection and view matrices
  render.py      CPU reference renderer, used as ground truth
  popping.py     motion-compensated frame difference

scripts/         thin CLI wrappers; none imports another
  export_wang.py     capture -> tileset
  probe_order.py     the ordering measurement above
  pop_metric.py      compare two captured sweeps

web/             the renderer
  index.html     page and panel
  viewer.js      WebGL2 renderer, layout, cameras, debug views
  sort-worker.js counting sort per patch, and per merged group
  grid.js        shared boundary geometry
  order.js       tile-level topological sorting      (GSWT 3.4)
  merge.js       selective merging                   (GSWT 3.4)
  tileset.js     read and write zips, open a tileset
  capture.js     scripted camera sweep to a zip of frames

tests/           run_all.py runs all three suites
```

---

## Measuring popping

`bozkir/popping.py` separates what changed because the camera moved from what
changed for any other reason: estimate the motion, undo it, and whatever
difference survives is the part motion cannot explain. Capture a sweep from
the panel with **capture both orderings**, then:

```
python scripts/pop_metric.py frames/topological --against frames/depth
```

This is a weaker instrument than the published one and the difference is
worth stating. StopThePop estimates motion with RAFT, a learned optical flow,
and weights the result with FLIP, a perceptual difference. Here the global
translation comes from phase correlation and the local refinement from block
matching, which is integer-accurate at best. On splat terrain — texture at
every scale — a sub-pixel registration error lights up a fifth of the frame
at a 0.02 threshold, and that floor is currently larger than the effect being
measured. The numbers compare orderings on one camera path; they are not
comparable to published figures, and at present they do not resolve the
ordering difference at all. Sub-pixel flow is the next step.

---

## Limitations

**Tiles are surfaces, so there is no underside.** Looking up from below shows
the far side of the same Gaussians. GSWT names this first among their own
limitations and points at Wang Cubes.

**Every capture so far is of a subject, not of ground.** Photographing
*around* something yields one good patch and three mediocre ones. A capture
of ground as the subject — a few metres of gravel, walked in a slow spiral —
is the thing that would fix this properly.

**The order within a merged group assumes flat tiles.** A cell contributes
its patch's depths plus one constant, which is exact under translation and
approximate once the surface is warped — the same assumption, for the same
reason, as the live per-tile order.

---

## Reference

Zeng, Ma, Sander. *Gaussian Splatting Wang Tiles.* SIGGRAPH Asia 2025.
Section 3.4 is the ordering and merging this project reimplements.

Radl, Steiner, Parger, Weinrauch, Kerbl, Steinberger. *StopThePop:
Sorted Gaussian Splatting for View-Consistent Real-time Rendering.*
SIGGRAPH 2024. The source of the popping metric's design.

Kerbl, Kopanas, Leimkühler, Drettakis. *3D Gaussian Splatting for Real-Time
Radiance Field Rendering.* SIGGRAPH 2023.