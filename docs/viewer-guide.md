# The viewer, control by control

Generated from `web/info.json` by `scripts/viewer_guide.py`; the
same text is the viewer's info drawer (the ⓘ marks, or I over a
control). Edit the JSON, not this page.

- [What bozkır is](#about)
- [Tilesets](#tileset)
- [Wang tiles](#tiles)
- [Terrain](#terrain)
- [World](#world)
- [Relief](#relief)
- [Detail relief](#detail)
- [The material rule](#rule)
- [Share and swap](#share)
- [Blending](#blend)
- [Show the decision](#decision)
- [Settings](#presets)
- [Camera](#camera)
- [Walk](#walk)
- [Look](#look)
- [Baked light](#lighting)
- [Land variation](#macro)
- [Renderer](#renderer)
- [Far field](#farfield)
- [Auto quality](#quality)
- [Tiles on terrain](#warp)
- [Level of detail](#lod)
- [Draw order](#ordering)
- [Selective merging](#merging)
- [Measurement](#measure)
- [Repetition and busyness](#repetition)

<a id="about"></a>
## What bozkır is

*Captured splat ground cut into Wang tiles and laid over terrain, with each material placed by the shape of the land.*

Hybrid GSWT chooses which material goes where from a coverage map a person paints. Here that map is computed from the terrain: drainage or sediment, topographic position, slope and altitude. The tiling, ordering and merging reimplement GSWT so the rule can be tested at scale.

Limits: lighting is baked into the capture; the desert scrub comes from about 4 × 5 m of capture; the rule's one validation so far was null.

- Zeng, Ma, Sander. GSWT: Gaussian Splatting Wang Tiles. SIGGRAPH Asia 2025.
- Zeng, Ma, Sander. Hybrid Gaussian Wang Tiles for Class-aware Authoring and Rendering. SIGGRAPH 2026.
- See also: [The material rule](#rule), [Wang tiles](#tiles)

<a id="tileset"></a>
## Tilesets

*A .splat of packed Gaussians and a .json saying which splats form which tile.*

The .json holds the tile size, levels of detail, Wang edge codes and each tile's material. Make one with `scripts/make_tileset.py` (gives a verdict) or `scripts/export_wang.py` (named presets). Without the .json the splats load as one plain scene.

The starter is generated in the browser: the same construction with procedural patches.

- In the code: `scripts/make_tileset.py`, `scripts/export_wang.py`, `web/starter.js`
- See also: [Wang tiles](#tiles)

<a id="tiles"></a>
## Wang tiles

*Tiles whose edges carry colours; neighbours must share the colour, so every layout is seamless.*

Each tile is four triangles, one per edge, cut from a captured patch of that edge's colour (Cohen et al. 2003). Seams between triangles follow a graph cut. Two tiles per west–south pair: 3 colours cost 18 tiles per material, not 81. Rotated patches count as extra colours.

A feature in a patch lands in the same place in every tile built from it, which the eye reads as a lattice.

- Cohen, Shade, Hiller, Deussen. Wang Tiles for Image and Texture Generation. SIGGRAPH 2003.
- In the code: `bozkir/wang.py`, `bozkir/graphcut.py`
- See also: [Repetition and busyness](#repetition)

<a id="terrain"></a>
## Terrain

*Nineteen landform profiles, generated and eroded in the browser, bit-identical to the Python version.*

Integer-hash noise per profile, Priority-Flood to fill depressions, then FastScape stream-power erosion with diffusion. Sediment is a routing proxy, not a deposition model. Closed basins become flat floors.

- Barnes, Lehman, Mulla. Priority-Flood. 2014.
- Braun, Willett. Implicit stream-power solver. 2013.
- docs/terrain.md has the full list.
- In the code: `web/terrain.js`, `bozkir/erosion.py`
- See also: [The material rule](#rule)

<a id="world"></a>
## World

*Size × size tiles laid out by the Wang rule and bent over the height field.*

Relief scale follows the size unless changed in Research; below it, the height field mirrors and landforms repeat (the scene card warns).

- See also: [Relief](#relief), [Far field](#farfield)

<a id="relief"></a>
## Relief

*Terrain height, and how many tiles one height field spans.*

Height grows with relief scale, so large worlds keep their hills. A relief scale below the world size makes landforms repeat.

- See also: [World](#world)

<a id="detail"></a>
## Detail relief

*Tile-scale creases on steep slopes and crests.*

A large world has about two height samples per tile, so close ground was smooth. The creases are masked to slopes and convex crests and computed identically for splats, far field and camera.

- See also: [Relief](#relief)

<a id="rule"></a>
## The material rule

*Material that collects goes to hollows and channels; the other to steep, convex ground.*

Inputs, sampled at 4× the tile grid: drainage (or sediment), topographic position (TPI), slope, elevation. Weights are in `web/rule.json`. A median filter makes regions; a rank threshold sets each material's exact share.

Validation: against a drone survey (desert plot, 15.6 m) kappa stayed between −0.13 and +0.02 — never above chance. The plot is a single slope (a plane explains 86% of its height). A survey across several landforms is the next test.

- Weiss. Topographic Position and Landforms Analysis. 2001.
- In the code: `web/landform.js`, `bozkir/landform.py`, `scripts/validate_rule.py`
- See also: [Share and swap](#share), [Blending](#blend), [Show the decision](#decision)

<a id="share"></a>
## Share and swap

*The share sets how much land a material gets; the rule sets where.*

Swap chooses which material collects. The exporter's material order is arbitrary (in the desert set, sand is first).

- See also: [The material rule](#rule)

<a id="blend"></a>
## Blending

*Near a tie both materials are drawn, dissolved splat by splat.*

Avoids the staircase of whole-tile boundaries, as Hybrid GSWT's per-Gaussian priority does. Costs a second draw call in the band.

- See also: [The material rule](#rule)

<a id="decision"></a>
## Show the decision

*Colours each cell by the class the rule chose: green collects, yellow exposed, grey a near tie.*

Separates a wrong decision from a tile that does not look like its class.

- See also: [The material rule](#rule)

<a id="presets"></a>
## Settings

*Named starting points from `web/views.json`; every control stays adjustable.*

Research adds the figure presets: rule on and off, land variation off. The story steps are presets with a camera and a caption.

- In the code: `web/views.json`

<a id="camera"></a>
## Camera

*Drag orbits, WASD moves, scroll zooms. Touch: one finger orbits, two pinch.*

The scale bar is measured at the orbit point.

- See also: [Walk](#walk)

<a id="walk"></a>
## Walk

*First person at 1.7 m eye height, walking pace 1.4 m/s.*

WASD walks, shift runs, click to look with the mouse, Esc frees it, V leaves. The eye follows the drawn ground height, so the terrain is the collider. Assumes the capture is in metres.

- See also: [Camera](#camera)

<a id="look"></a>
## Look

*Sky, haze and exposure. None of it relights the ground.*

- See also: [Baked light](#lighting), [Land variation](#macro)

<a id="lighting"></a>
## Baked light

*A capture's light and shadows are part of its colours.*

Ground shading only darkens hollows by how much sky they see. Overcast captures tile best: no shadows to repeat.

- See also: [Look](#look)

<a id="macro"></a>
## Land variation

*Slow brightness noise at about 4 and 11 tiles, in splats and far field alike.*

Breaks the one-tile period the eye picks up (macro variation in game landscape materials). A render-time tint, not in the capture; off in the figure presets.

- See also: [Repetition and busyness](#repetition)

<a id="renderer"></a>
## Renderer

*WebGL2, one instanced draw per tile, splats sorted in a worker.*

- See also: [Far field](#farfield), [Auto quality](#quality), [Level of detail](#lod), [Draw order](#ordering)

<a id="farfield"></a>
## Far field

*Distant ground as one mesh coloured from a top-down atlas of the tiles: one draw call.*

The limit was draw calls, not splat count, so this was chosen over hierarchical 3DGS. Opaque, dithered hand-over; with distance each cell fades part-way to its tile's mean. Tilesets without an atlas get a rougher one baked in the browser.

Limits: top-down colour differs from grazing views; a faint step can show at the hand-over.

- Kerbl et al. A Hierarchical 3D Gaussian Representation. SIGGRAPH 2024.
- In the code: `scripts/bake_atlas.py`
- See also: [Level of detail](#lod)

<a id="quality"></a>
## Auto quality

*Holds 30 fps: render resolution first (50–100%), level of detail only after.*

Moving both at once made the frame rate cycle between 20 and 45 fps.

- See also: [Level of detail](#lod)

<a id="warp"></a>
## Tiles on terrain

*Each splat is placed through the ground's tangent frame and shaped by its true Jacobian.*

Normalising the Jacobian had thinned slopes by √(1 + |∇h|²). One frame per tile, as in GSWT, leaves gaps on slopes.

<a id="lod"></a>
## Level of detail

*Tiles at 100, 25, 6 and 2% of their splats; each level starts twice as far out.*

- See also: [Far field](#farfield)

<a id="ordering"></a>
## Draw order

*Back to front, within each tile and between tiles.*

Within a tile: a counting sort in the tile's own frame (a single world order left 47% of pairs wrong on warped terrain; this leaves 0).

Between tiles: depth keys tie along whole rows at axis-aligned views; a topological order removes it (27/54/27 reversed pairs → 0).

- In the code: `web/order.js`, `web/sort-worker.js`
- See also: [Selective merging](#merging)

<a id="merging"></a>
## Selective merging

*Where no order between two tiles is right, they are drawn as one stream.*

Counting sort over the union: 3.8 ms against 9.5 ms for a k-way merge at 300k splats.

- In the code: `web/merge.js`
- See also: [Draw order](#ordering)

<a id="measure"></a>
## Measurement

*Pop meter, capture sweeps for pop_metric.py, and a scaling sweep.*

The popping metric does not yet resolve the ordering difference (integer-accurate matching). GTX 1650 Ti: 60 fps to grid 40; with far field and auto quality, grid 256 above 30 fps.

- In the code: `scripts/pop_metric.py`

<a id="repetition"></a>
## Repetition and busyness

*Two measures of tiled ground, checked against the eye.*

Repetition (whole-tile minus half-tile correlation) disagreed with the eye. Busyness (how different a tile's four triangles are) agreed: the set that repeated least had 3.6× busier scrub. Two runs; a perceptual study is the proper test.

- In the code: `scripts/repetition.py`
- See also: [Wang tiles](#tiles), [Land variation](#macro)
