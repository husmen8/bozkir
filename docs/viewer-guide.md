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
- [Atmosphere](#atmosphere)
- [Capture](#capture)
- [Frames and centres](#frames)
- [Distant land](#horizon)
- [River and water](#water)

<a id="about"></a>
## bozkır

*A few square metres of real ground, captured as Gaussian splats, made into tiles and laid over terrain, each material where the land puts it.*

The pipeline, end to end: a ground capture is trained as Gaussian splats (scripts/tools/groundsplat.py for drone photos, or any 3DGS trainer); it is levelled and cleaned; its materials are found and the cleanest patches of each chosen, checked for holes and haze; patches become edge-matched tiles with framed centres from a pool of the capture's own ground; in the viewer each cell's material comes from the terrain (drainage, position, slope, height, and a river where one runs), rivers are cut into the ground and drawn with water measured from drone photos, and everything is drawn in the browser with its own ordering, levels of detail and far field.

It began from Gaussian Splatting Wang Tiles (Zeng, Ma and Sander), whose idea of tiling splat ground with Wang tiles it keeps. Frames and centre pools, the material rule, the river and its water, and most of the export and renderer are this project's own.

Limits: lighting is baked into a capture; the rule's one test against survey data could not decide (see the material rule); a drone capture holds up to about a metre away and no closer.

- Zeng, Ma, Sander. Gaussian Splatting Wang Tiles. SIGGRAPH Asia 2025.
- Desert tiles: OpenDroneMap COPR dataset (Coal Oil Point Reserve; Tobias and Mandel 2009), CC BY-SA 4.0.
- River tiles: Lütschine L2 drone survey (Zenodo 6415047) and Uruguayan native grassland UAV images (Zenodo 14771664), CC BY 4.0.
- See also: [Capture](#capture), [Wang tiles](#tiles), [The material rule](#rule), [River and water](#water)

<a id="tileset"></a>
## Tilesets

*A .splat of packed Gaussians and a .json manifest: tile size, edge codes, materials, levels of detail.*

`scripts/build.py <capture.ply> --preset ground -o web/data/<name>` runs the export, the decals and the far-field atlas. Every decision is measured from the capture and printed with its reason: number of materials, patch choice, hole checks, centre pool, turns, height order. A manifest can name its own view and terrain (the river opens on its valley and river view) and its story.

A .splat without a manifest loads as one plain scene. The starter tileset is generated in the browser with the same construction and procedural patches.

- In the code: `scripts/build.py`, `scripts/export_wang.py`, `web/starter.js`
- See also: [Wang tiles](#tiles), [Frames and centres](#frames)

<a id="tiles"></a>
## Wang tiles

*Edge-coloured tiles: neighbours share the colour of their common edge, so any valid layout is continuous.*

Each tile is four triangles, one per edge. The triangle on an edge comes from a patch of that edge's colour, placed with its centre on the edge (Cohen et al. 2003), so the ground continues across it. Seams between the triangles follow a minimum-error graph cut.

Two tiles per (west, south) colour pair: with 3 colours, 18 tiles per material instead of 81. Materials share the edge codes, so a cell can take any material without consulting its neighbours.

A feature in a patch would appear at the same place in every tile built from it; the frames and centre pools below are how this project keeps it from repeating.

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

*Each material placed from the terrain: the one that collects in hollows and channels, the exposed one on steep, convex ground; along a river, gravel where the river leaves it.*

Inputs, sampled at 4× the cell grid: drainage, max-pooled so a channel narrower than a cell survives (or the erosion's sediment), topographic position index, slope, altitude. Weights are in `web/rule.json`, shared by the viewer and `bozkir/landform.py`. A median filter makes regions without moving their boundaries; a rank threshold gives each material exactly its share.

With three or more materials the height at which one gives way to the next wanders across the world by fractal noise. The capture sets the amount: how often the materials are out of height order over its ground (Mann-Whitney U over local relative height).

With a river drawn as water (the river rule), the gravel bar is where a river leaves one: low beside the channel (within about a channel depth above the water) and within reach of it (about half the river's width, wider on flat floors and inner bends, thin under steep banks), broken up along the river by noise. Its share then follows the river and the valley, not the capture's.

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

*Standard puts every control back to the tileset's own settings; Research adds the comparisons used in the write-up.*

A view sets the controls it names; controls another view set and this one does not go back to their defaults. Other views in web/views.json open by link (?preset=name). The story steps are views with a camera and a caption, and a tileset can have its own story.

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

- See also: [Baked light](#lighting), [Land variation](#macro), [Atmosphere](#atmosphere)

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

A normalised Jacobian thins material on slopes by √(1 + |∇h|²). One frame per tile leaves gaps on slopes; off, the slider sets frames per tile.

<a id="lod"></a>
## Level of detail

*Tiles at 100, 25, 6 and 2% of their splats; each level starts twice as far out as the one before.*

- See also: [Far field](#farfield)

<a id="ordering"></a>
## Draw order

*Back to front, within each tile and between tiles.*

Within a tile: a counting sort along the view direction in the tile's own frame. A single world-space order per tile left 47% of adjacent pairs wrong on warped terrain; this leaves none.

Between tiles: nearest-depth keys tie along whole rows at axis-aligned views (81 cells, 9 keys) and reverse a row in one frame. A topological order over shared-edge constraints removes this.

- In the code: `web/order.js`, `web/sort-worker.js`, `scripts/probe_order.py`
- See also: [Selective merging](#merging)

<a id="merging"></a>
## Selective merging

*Where no order between two tiles is right, both are drawn as one interleaved stream.*

Counting sort over the union: 3.8 ms against 9.5 ms for a k-way merge at 300k splats, 25.6 against 76.8 ms at 1.2M.

Off by default on ground tilesets: a merged group is drawn where its first tile falls in the order, so the cells around it can land in front of or behind it wrongly, and while the camera moves its merged stream arrives late. With frames and blended borders the popping it removes is small, so it costs more than it fixes there.

- In the code: `web/merge.js`
- See also: [Draw order](#ordering)

<a id="measure"></a>
## Measurement

*Popping while the camera turns, the seam gap between tiles, recordings for scripts/pop_metric.py, and a scaling sweep.*

Popping: the colour change in the middle of the picture per degree turned; a smooth turn changes it steadily, a spike is splats or tiles switching order. Seam gap: every shared edge placed from both of its tiles, in splat widths; 0 means they meet.

A recording saves frames along an orbit; scripts/pop_metric.py registers them (phase correlation, block matching) and compares two draw orders on the same path. GTX 1650 Ti: 60 fps to a world of 40 cells (5.5M splats); with far field and auto quality, 256 cells above 30 fps.

- In the code: `scripts/pop_metric.py`, `bozkir/popping.py`

<a id="repetition"></a>
## Repetition and busyness

*Two measures of tiled ground, compared with visual judgement.*

Repetition: excess brightness correlation at whole-tile shifts over half-tile shifts. Busyness: the share of a tile's brightness variance explained by which triangle a pixel is in. On two tilesets of the same capture, the one that repeated less had 3.6× busier scrub and was judged worse; repetition alone did not match the judgement. Two tilesets are not a proof; a perceptual study is the test. The script lays out exported tiles only; centres from the pool are not modelled.

- In the code: `scripts/repetition.py`
- See also: [Wang tiles](#tiles), [Land variation](#macro)

<a id="atmosphere"></a>
## Atmosphere

*Valley mist, sea haze, the sun's glow and a colour grade: how the air and the picture look, not the ground.*

Valley mist is cold-air pooling: densest over the channel each spot drains to and thinning by e every 0.3 relief above it (valley fog fills a valley's bottom, not its sides), measured with the same HAND (height above the nearest drainage, Renno et al. 2008) the river rule places gravel and grass by, and integrated along each ray (Quilez, Better fog). Sea haze is a coast's marine layer: even under a lid a quarter of the relief above the middle (the inversion that caps it), so high ground stands out; its logistic profile integrates in closed form. Both are depth cues scaled to the world's width, not a physical visibility: a 36 m world would show no real haze. The sun's glow lightens the sky and the haze towards it; the scans' light is baked, so the ground is not relit. The grade is a final tone curve. All are off at 0 and off in the debug views.

- See also: [Look](#look), [Baked light](#lighting)

<a id="capture"></a>
## Capture

*Real ground trained as Gaussian splats: straight-down drone photos or a walk around, a few square metres per material.*

Drone photos go through scripts/tools/groundsplat.py: COLMAP poses, a block of ground cut out where enough photos see it, and gsplat training. The block's ground is found from below, levelled and set in metres.

What a capture can show is set when it is taken: ground seen at 3-4 mm per pixel holds up from about a metre away, and grass that moves in the wind between photos stays soft however long it trains. More splats only add slivers that read as needles up close.

- In the code: `scripts/tools/groundsplat.py`, `bozkir/scene.py`
- See also: [Tilesets](#tileset), [bozkır](#about)

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

<a id="horizon"></a>
## Distant land

*Ground past the world's edge, at the world's own scale, thinning into the haze: a backdrop, not part of the world.*

One ring mesh out to four world widths, drawn like the far field: each point shows the tile of the world's cell it mirrors, as the height field and the river mirror, faded to the tile's mean colour with distance. Its hills are rounded ridged noise a few times the world's relief and the size of its landforms, a little taller further out; extra haze with the distance from the edge dissolves it, so from any height it reads as a horizon. Slopes darken by how much sky they see, as hollows do. No splats are drawn out there and the rule does not place it; 0 is off.

- See also: [Look](#look), [Atmosphere](#atmosphere)

<a id="water"></a>
## River and water

*A river cut into the terrain in metres where it drains, with water drawn from layers measured in drone photos.*

Course: the terrain's largest drainage systems, smoothed and meandered. Width in metres, narrower up the branches (width grows with the square root of the area drained; Leopold and Maddock 1953). Section: a bed a twentieth as deep as wide (at most 0.6 m), its deepest line towards the outside of bends, a cut bank there and a gentle point bar inside; pools and riffles about six widths apart; banks rising gently until they meet the ground.

Water: a level surface that never rises downstream. Each point under it is drawn on that surface and coloured by its own depth: the bed shows through by a share measured from the fading stone contrast in 29 L2 photos (T = 0.93 exp(-d / 0.53 m) from the waterline), under the milky shallow water near the banks and the dark channel water further out, both photographed. Stones above the level stay dry, so the waterline follows them. No sky reflection: the photos show none.

Flow: the water's ripples move along the river, faster where it is deeper (Manning); its broad pattern stays where it lies.

- Leopold, Maddock. The Hydraulic Geometry of Stream Channels. USGS 1953.
- Leopold, Wolman. River Meanders. GSA Bulletin 1960.
- Vlachos. Water Flow in Portal 2. SIGGRAPH 2010 course.
- In the code: `web/landform.js`, `web/shaders.js`, `scripts/water_texture.py`
- See also: [The material rule](#rule), [Terrain](#terrain)
