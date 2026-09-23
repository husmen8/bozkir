# Terrain: what the generator does, and where each part comes from

`bozkir/erosion.py` and `web/terrain.js` are one generator in two languages.
A profile and a seed name the same ground in both, bit for bit
(`tests/test_pipeline.py`, section *terrain parity*). This page maps each
step to the work it comes from, so the code can be explained in the terms
the field uses.

## The pipeline

| Step | Function (Python / JS) | Source |
|---|---|---|
| Gradient noise, quintic fade | `_gradient_noise` / `octave` | Perlin, *Improving Noise*, SIGGRAPH 2002 |
| Fractal sum of octaves | `fbm` | the standard fBm construction of procedural terrain |
| Ridged octaves | `ridged` | Musgrave's ridged multifractal (Ebert et al., *Texturing and Modeling*), simplified |
| Terraces | `terrace` | a shaping function; no single source |
| Fill closed hollows | `fill_depressions` / `fillDepressions` | Barnes, Lehman and Mulla, *Priority-Flood*, Computers & Geosciences 2014 |
| Steepest-descent receivers | `_receivers` | D8 flow routing, O'Callaghan and Mark 1984 |
| Receivers-first ordering, drainage area | `_levels` | the stack order of Braun and Willett 2013 |
| Implicit stream power incision | `erode` | Braun and Willett, *A very efficient O(n), implicit and parallel method to solve the stream power equation*, Geomorphology 2013 (FastScape) |
| Hillslope diffusion | `erode` | the linear diffusion term FastScape uses for creep |
| Uplift map from the noise | `generate`, `uplift=` | the uplift-domain approach of Cordonnier et al., Eurographics 2016, and Schott et al., ACM TOG 2023 |
| Sediment map | `erode`, `track=` | **a proxy**; see below |

## The equation

    dh/dt  =  U  -  K · A^m · S  +  D · ∇²h

Uplift, minus fluvial incision (drainage area `A`, slope `S`), plus creep.
With `m = 0.5` and slope exponent 1, the incision step is solved directly:
visiting cells receivers-first,

    h_i  ←  (h_i + F · h_r) / (1 + F),        F = K · dt · √A / distance

which is stable for any step size and costs one pass over the cells
(Braun and Willett 2013, the case n = 1). Drainage area is measured as a
fraction of the field, so the same `K` means the same thing at 256 and 512.

## Where this is simpler than the literature

**Pits are filled, not routed.** Priority-Flood runs on the starting surface
and on the result. Routing water through depressions at every step
(Cordonnier, Bovy and Braun, Earth Surface Dynamics 2019) is the thorough
version; FastScapeLib implements it.

**Sediment is a proxy.** Material eroded at each step is passed downstream
and settles in proportion to how flat the ground is. It produces a branching
deposit map that follows the drainage, which is what the material rule
needs, but it does not conserve mass the way a deposition model does. The
principled version is Yuan et al., *A new efficient method to solve the
stream power law model taking into account sediment deposition*, JGR Earth
Surface 2019 - an extension of the same FastScape solver, so it would slot
into `erode` rather than replace it.

**The map is the middle of a larger simulation.** The simulation's border
is base level - everything drains to it - so a rule that puts material in
low, collected ground would draw a frame around every map. The generator
simulates 15% extra on each side and crops it off, and the gentle rise that
lets closed hollows drain (`dome`) sits entirely in that cropped band.

**Closed basins become flat floors.** Filled depressions are flat, and
without a dome tilting the visible ground they stay as floors between the
hills. In dry profiles that is realistic - playas, alluvial flats - and it
is where collected material belongs; in green hills it is a
simplification. Routing water through depressions during the run
(Cordonnier, Bovy and Braun 2019) is the fix.

**Uplift profiles are short runs, not steady state.** The papers run to
equilibrium between uplift and erosion. The `foothills`, `ridges` and
`alpine` profiles stop after 120-180 steps, which is enough for valleys to
organise and short enough to generate in a browser in a few seconds.

## Browser generation

`web/terrain-worker.js` runs the generator off the main thread; results are
kept in IndexedDB by profile, seed and size. A 512 alpine terrain takes
about 7 s on first generation, instant afterwards. For comparison, TerrainX
(`github.com/GPU-Gang/WebGPU-Erosion-Simulation`) implements Schott et al.
2023 in WebGPU, using that paper's parallel approximation of drainage area -
the route to interactive rates at larger sizes.

## Questions this should let you answer

*Which erosion model?* Stream power with linear diffusion, solved with
FastScape's implicit scheme. The same equation Cordonnier et al. 2016 and
Schott et al. 2023 use in graphics.

*Why not droplet erosion?* Droplets depend on random walks finding the same
route; on noise most die in the nearest pit and the result is smoothing.
Stream power drives erosion from the drainage network directly.

*Is the sediment map physical?* No - a proxy that follows drainage and
slope. Yuan et al. 2019 is how to make it physical.

*Why do both languages give identical terrain?* Integer hashing instead of
a random number generator, fixed gradient vectors instead of trigonometry,
explicit rounding, and every floating-point sum in the same order.

## Further reading

Galin et al., *A Review of Digital Terrain Modeling*, Computer Graphics
Forum 2019 (Eurographics STAR) - the survey that places noise, erosion and
example-based methods side by side.
