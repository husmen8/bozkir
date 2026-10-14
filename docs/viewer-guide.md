# The viewer, control by control

Generated from `web/info.json` by `scripts/viewer_guide.py`; the
same text is the viewer's info drawer (the ⓘ marks, or I over a
control). Edit the JSON, not this page.

- [bozkır](#about)
- [Tilesets](#tileset)
- [Wang tiles](#tiles)
- [Terrain](#terrain)
- [World](#world)
- [Relief](#relief)
- [Detail relief](#detail)
- [The material rule](#rule)
- [Share and swap](#share)
- [Blending](#blend)
- [Decals](#decals)
- [Show the decision](#decision)
- [Views](#presets)
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
- [Frames and centres](#frames)

<a id="about"></a>
## bozkır

*Gaussian-splat ground cut into Wang tiles and laid over terrain, with each material placed from the terrain's geometry.*

Hybrid GSWT chooses a material class per Gaussian from a coverage map that an artist paints. Here that coverage is computed from the height field: drainage or erosion sediment, topographic position, slope and altitude. The tiling, draw ordering and merging reimplement GSWT so the rule can be tested over large terrain.

Limits: lighting is baked into the capture; the rule's one validation against survey data was inconclusive (see the material rule).

- Zeng, Ma, Sander. Gaussian Splatting Wang Tiles. SIGGRAPH Asia 2025.
- Zeng, Ma, Sander. Hybrid Gaussian Wang Tiles for Class-aware Authoring and Rendering. SIGGRAPH 2026.
- Desert tiles: OpenDroneMap COPR dataset (Coal Oil Point Reserve; Tobias and Mandel 2009), CC BY-SA 4.0.
- See also: [The material rule](#rule), [Wang tiles](#tiles)

<a id="tileset"></a>
## Tilesets

*A .splat of packed Gaussians and a .json manifest: tile size, edge codes, materials, levels of detail.*

`scripts/build.py <capture.ply> --preset ground -o web/data/<name>` runs the export, the decals and the far-field atlas. Every decision is measured from the capture and printed with its reason: number of materials, patch choice, hole checks, centre pool, turns, height order.

A .splat without a manifest loads as one plain scene. The starter tileset is generated in the browser with the same construction and procedural patches.

- In the code: `scripts/build.py`, `scripts/export_wang.py`, `web/starter.js`
- See also: [Wang tiles](#tiles), [Frames and centres](#frames)

<a id="tiles"></a>
## Wang tiles

*Edge-coloured tiles: neighbours share the colour of their common edge, so any valid layout is continuous.*

Each tile is four triangles, one per edge. The triangle on an edge comes from a patch of that edge's colour, placed with its centre on the edge (Cohen et al. 2003), so the ground continues across it. Seams between the triangles follow a minimum-error graph cut.

Two tiles per (west, south) colour pair: with 3 colours, 18 tiles per material instead of 81. Materials share the edge codes, so a cell can take any material without consulting its neighbours.

A feature in a patch appears at the same place in every tile built from that patch; frames and centres address this.

- Cohen, Shade, Hiller, Deussen. Wang Tiles for Image and Texture Generation. SIGGRAPH 2003.
- Kwatra, Schödl, Essa, Turk, Bobick. Graphcut Textures. SIGGRAPH 2003.
- In the code: `bozkir/wang.py`, `bozkir/graphcut.py`
- See also: [Frames and centres](#frames), [Repetition and busyness](#repetition)

<a id="terrain"></a>
## Terrain

*Nineteen landform profiles, generated and eroded in the browser; the Python generator gives the same heights bit for bit.*

Integer-hash fractal noise per profile, Priority-Flood depression filling, then stream-power erosion solved implicitly (FastScape) with hillslope diffusion. Sediment is a routing proxy, not a deposition model; closed basins become flat floors. The simulation runs 15% larger than the map and is cropped, so edges carry no boundary artefacts.

Height fields can also come from a drone DSM or from the capture (`scripts/heightmap.py`).

- Barnes, Lehman, Mulla. Priority-Flood. Computers & Geosciences 2014.
- Braun, Willett. An implicit, parallel solver for the stream power equation. Geomorphology 2013.
- In the code: `web/terrain.js`, `bozkir/erosion.py`, `docs/terrain.md`
- See also: [The material rule](#rule), [World](#world)

<a id="world"></a>
## World

*World size × world size cells, each one tile, laid out by the Wang rule and bent over the height field.*

Relief scale follows the world size unless set in Research. Below it the height field mirrors and landforms repeat; the scene section says so.

- See also: [Relief](#relief), [Far field](#farfield)

<a id="relief"></a>
## Relief

*Terrain height, and how many cells one height field spans.*

Height grows with relief scale, so a larger world keeps its hills. A relief scale below the world size repeats the landforms, mirrored.

- See also: [World](#world), [Detail relief](#detail)

<a id="detail"></a>
## Detail relief

*Ridged noise at tile scale, masked to steep slopes and convex crests.*

A large world has about two height samples per cell, so close ground is smooth without it. The same function runs in the splat shader, the far field and the camera.

- See also: [Relief](#relief)

<a id="rule"></a>
## The material rule

*The material that collects goes to hollows and channels, the exposed one to steep, convex ground.*

Inputs, sampled at 4× the cell grid: drainage, max-pooled so a channel narrower than a cell survives (or the erosion's sediment), topographic position index, slope, altitude. Weights are in `web/rule.json`, shared by the viewer and `bozkir/landform.py`. A median filter makes regions without moving their boundaries; a rank threshold gives each material exactly its share.

With three or more materials the height at which one gives way to the next wanders across the world by fractal noise. The capture sets the amount: how often the materials are out of height order over its ground (Mann-Whitney U over local relative height).

Validation against an ODM survey (15.6 m plot, DTM and orthophoto): Cohen's kappa between −0.13 and +0.02 on grids 16 to 32, never above a torus-shift null. The plot is one slope (a plane explains 86% of its height), so it cannot separate the rule from chance; a survey across several landforms is needed.

- Weiss. Topographic Position and Landforms Analysis. ESRI User Conference 2001.
- In the code: `web/landform.js`, `bozkir/landform.py`, `scripts/validate_rule.py`
- See also: [Share and swap](#share), [Blending](#blend), [Show the decision](#decision)

<a id="share"></a>
## Share and swap

*The share sets how much of the world a material covers; the rule sets where.*

A tileset that measured its capture starts at the capture's shares, weighted down for materials held only as haze. With three or more materials each has its own slider and the others keep their proportions. Swap reverses which material collects.

- See also: [The material rule](#rule)

<a id="blend"></a>
## Blending

*Each spot near a boundary belongs to one material; only a narrow strip between them dissolves, and there the taller splat wins.*

Each material has a field on a half-cell grid, read at a point displaced by value noise at three scales down to a sixth of a cell, so materials reach into one another instead of meeting along cell edges. Where three meet, every spot still goes to one of them. The far field reads the same fields.

Cells the strip can reach are drawn a second time with the neighbouring material. In the strip the taller and more opaque splat wins (height-based blending). With tones meet, a cell next to another material takes the centres whose own tone is nearest that material's; nothing is tinted.

- Zeng, Ma, Sander. Hybrid GSWT. SIGGRAPH 2026: per-Gaussian class choice at boundaries.
- Mishkinis. Advanced Terrain Texture Splatting. Gamasutra 2013: height-based blending.
- In the code: `web/shaders.js`, `web/frames.js`
- See also: [The material rule](#rule), [Frames and centres](#frames)

<a id="decals"></a>
## Decals

*Distinct pieces of the capture, a shrub and its shadow, baked into copies of plain tiles.*

Tiles are cut from the plainest ground, since anything distinctive repeats with them. `scripts/decals.py` finds distinct pieces and bakes each into copies of tiles or centres, inside a disc clear of the edges, so a copy matches its neighbours as the original does.

The layout places at most one decal per block of decal spacing cells each way, never the same decal within two blocks of itself. Decals are not turned, so their shadows keep the capture's direction, and they are not flattened. With decals follow the terrain, each block keeps the cell nearest the other material.

- Glanville. Texture Bombing. GPU Gems 2004.
- In the code: `bozkir/decals.py`, `scripts/decals.py`, `web/decals.js`
- See also: [Wang tiles](#tiles), [Repetition and busyness](#repetition)

<a id="decision"></a>
## Show the decision

*Each cell coloured by the material the rule chose; grey where it was nearly a tie.*

Separates a wrong decision from a tile that does not look like its material.

- See also: [The material rule](#rule)

<a id="presets"></a>
## Views

*Named settings from `web/views.json`; every control stays adjustable.*

A view sets the controls it names; controls another view set and this one does not go back to their defaults. Research adds the figure views (rule on and off, land variation off). The story steps are views with a camera and a caption.

- In the code: `web/views.json`

<a id="camera"></a>
## Camera

*Drag orbits, WASD moves, scroll zooms. Touch: one finger orbits, two pinch.*

The orbit point follows the ground and the eye stays above it. The scale bar is measured at the orbit point.

- See also: [Walk](#walk)

<a id="walk"></a>
## Walk

*First person at 1.7 m eye height and 1.4 m/s.*

WASD walks, shift runs, click to look with the mouse, Esc releases it, V leaves. The eye follows the rendered ground height, so the terrain is the collider. Assumes the capture is in metres.

- See also: [Camera](#camera)

<a id="look"></a>
## Look

*Sky, haze and exposure. None of it relights the ground.*

Haze is exponential in view depth. Its density is set for a 24-cell world; a larger world keeps the same haze at the same fraction of its width, so the haze slider means the same at any world size.

- See also: [Baked light](#lighting), [Land variation](#macro)

<a id="lighting"></a>
## Baked light

*A capture's light and shadows are part of its colours.*

Ground shading only darkens hollows by the share of sky they see. Overcast captures tile best: no shadows to repeat.

- See also: [Look](#look)

<a id="macro"></a>
## Land variation

*Slow brightness noise at about 4 and 11 cells, in the splats and the far field alike.*

Breaks the one-tile period (macro variation in landscape materials). A render-time tint, not in the capture; off in the figure views.

- See also: [Repetition and busyness](#repetition)

<a id="renderer"></a>
## Renderer

*WebGL2: one draw per visible cell or merged group, splats sorted in a worker.*

- See also: [Far field](#farfield), [Auto quality](#quality), [Level of detail](#lod), [Draw order](#ordering)

<a id="farfield"></a>
## Far field

*Distant ground as one mesh coloured from a top-down atlas of the tiles and centres: one draw call.*

The limit at scale is draw calls, not splat count, so this was chosen over a hierarchical 3DGS. Opaque, handed over to the splats by a dithered screen door; with distance each cell fades part-way to its tile's mean colour. Tilesets without an atlas get one baked in the browser.

Limits: a top-down atlas differs from grazing views, and a faint colour step can show at the hand-over.

- Kerbl et al. A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets. SIGGRAPH 2024.
- In the code: `scripts/bake_atlas.py`
- See also: [Level of detail](#lod)

<a id="quality"></a>
## Auto quality

*Holds about 30 fps: render resolution first (50–100%), level-of-detail distance second.*

Adjusting both at once made the frame rate oscillate between 20 and 45 fps.

- See also: [Level of detail](#lod)

<a id="warp"></a>
## Tiles on terrain

*Each splat is placed through the ground's tangent frame and shaped by the true Jacobian of the height map.*

A normalised Jacobian thins material on slopes by √(1 + |∇h|²). One frame per tile, as in GSWT, leaves gaps on slopes; off, the slider sets frames per tile.

<a id="lod"></a>
## Level of detail

*Tiles at 100, 25, 6 and 2% of their splats; each level starts twice as far out as the one before.*

- See also: [Far field](#farfield)

<a id="ordering"></a>
## Draw order

*Back to front, within each tile and between tiles.*

Within a tile: a counting sort along the view direction in the tile's own frame. A single world-space order per tile left 47% of adjacent pairs wrong on warped terrain; this leaves none.

Between tiles: nearest-depth keys tie along whole rows at axis-aligned views (81 cells, 9 keys) and reverse a row in one frame. A topological order over shared-edge constraints, as in GSWT, removes this.

- In the code: `web/order.js`, `web/sort-worker.js`, `scripts/probe_order.py`
- See also: [Selective merging](#merging)

<a id="merging"></a>
## Selective merging

*Where no order between two tiles is right, both are drawn as one interleaved stream.*

Counting sort over the union: 3.8 ms against 9.5 ms for a k-way merge at 300k splats, 25.6 against 76.8 ms at 1.2M.

- In the code: `web/merge.js`
- See also: [Draw order](#ordering)

<a id="measure"></a>
## Measurement

*Pop meter, camera sweeps for `scripts/pop_metric.py`, and a scaling sweep.*

The popping metric registers frames by phase correlation and block matching; integer-accurate matching does not yet resolve the ordering difference. GTX 1650 Ti: 60 fps to grid 40 (5.5M splats); with far field and auto quality, grid 256 above 30 fps.

- In the code: `scripts/pop_metric.py`, `bozkir/popping.py`

<a id="repetition"></a>
## Repetition and busyness

*Two measures of tiled ground, compared with visual judgement.*

Repetition: excess brightness correlation at whole-tile shifts over half-tile shifts. Busyness: the share of a tile's brightness variance explained by which triangle a pixel is in. On two tilesets of the same capture, the one that repeated less had 3.6× busier scrub and was judged worse; repetition alone did not match the judgement. Two tilesets are not a proof; a perceptual study is the test. The script lays out exported tiles only; centres from the pool are not modelled.

- In the code: `scripts/repetition.py`
- See also: [Wang tiles](#tiles), [Land variation](#macro)

<a id="frames"></a>
## Frames and centres

*A tile keeps only the band along its edges; each cell gets a centre from a per-material pool, chosen and turned in the viewer.*

The frame is what neighbours must agree on. The centre is a rounded square (unit ball of a p-norm, p = 6) as large as fits inside the edge strip, taken from the material's own ground where it shares little or none with the edge patches. Random placement and rotation hide a texture's period (Heitz and Neyret 2018).

How a material's centres may turn is measured with the structure tensor of its patches: ripples or rows keep their direction, isotropic ground turns in quarter turns. Frame and centre meet in a ring where the taller, more opaque splats win.

Pool candidates are judged where they show: no larger splat footprint, no thicker haze and no further in colour than the edge patches, and no see-through holes after the opacity fill.

- Heitz, Neyret. High-Performance By-Example Noise using a Histogram-Preserving Blending Operator. HPG 2018.
- Mikkelsen. Practical Real-Time Hex-Tiling. JCGT 2022.
- In the code: `bozkir/frames.py`, `bozkir/pick.py`, `web/frames.js`
- See also: [Wang tiles](#tiles), [Blending](#blend)
