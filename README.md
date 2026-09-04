# bozkır

Procedural terrain from captured Gaussian splats. Working toward a
composite-time treatment of tile boundary artifacts.

Portfolio piece for Japanese master's applications (kenkyuusei route).

---

## The research position

GSWT (Zeng, Ma, Sander — SIGGRAPH Asia 2025) tiles captured 3D Gaussian
fields using Wang Tiles to generate infinite terrain. Its stated limitations
were volumetric tiling, multi-class terrain, and boundary artifacts at
extreme viewing angles.

**Multi-class is closed.** The same authors published *Hybrid Gaussian Wang
Tiles for Class-aware Authoring and Rendering* at SIGGRAPH 2026. That paper
also attributes residual flicker to pre-sort direction changes and
tile-level ordering inherited from GSWT — the authors point at depth
ordering as the cause.

**The gap.** Every published seam method solves boundaries by
*optimization*: StitchGS refines appearance after freezing geometry,
Graph-GSReg runs test-time optimization, 3D Gaussian Stitching does
sample-guided synthesis, SeamlessNeRF propagated gradients. All assume a
finite merge you can afford to train over.

Tiling breaks that. A tile set is instanced thousands of times at
boundaries not known until the tiling is generated. There is no merged
artifact to optimize — the seam must resolve during rasterization.

So: the seam in a tiled Gaussian field is a **rendering-time depth-ordering
problem**, and the literature treats seams as a **reconstruction-time
optimization problem**. Nobody has connected the two.

Three properties that matter: needs no training (4 GB VRAM is enough), is
measurable with StopThePop's published metric, and is the thing GSWT's
authors said they did not solve.

Proposal title: *Composite-Time Boundary Resolution for Tiled Gaussian
Splat Fields*.

---

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python tests/test_all.py          # 51 tests, ~30s
```

Python 3.11 or 3.12. No CUDA, no compiler, no PyTorch. Everything is numpy.

Scenes: HuggingFace `alexmkwizu/gaussian_training_datasets` under
`tested_outputs/`, or INRIA's pre-trained models under
`point_cloud/iteration_30000/`. Not `input.ply` — that is a raw SfM cloud
with none of the required fields.

---

## Layout

```
bozkir/
  ply.py         load/save PLY, Splats container, SH evaluation, covariance
  scene.py       prepare(): the one load -> align -> clean path, cached
  render.py      CPU reference renderer, orthographic
  camera.py      perspective camera, projection (3DGS Eq. 5), Jacobian
  transform.py   quaternion algebra, ground detection, alignment
  select.py      cropping, floater removal
  tile.py        patch extraction, instancing, global vs tiled rendering
scripts/
  inspect_ply.py    9 checks that a PLY parsed correctly, plus 9 probes
  align.py          straighten a scene, write data/aligned/
  render_ortho.py   orthographic views
  render_view.py    perspective views, orbit camera
  seam_baseline.py  the boundary artifact measurement
  export_splat.py   pack a scene for the browser renderer
web/
  index.html      viewer page, control panel, orientation gizmo
  viewer.js       WebGL2 splat renderer
  sort-worker.js  depth sorting, isolated so it can be replaced
tests/
  test_all.py     51 tests
data/raw/         source PLYs          (gitignored)
data/aligned/     straightened PLYs    (gitignored)
data/cache/       prepared scenes      (gitignored)
out/              images               (gitignored)
```

## Usage

Every script shares the same preparation flags: `--clean`, `--flip`,
`--sh`, `--radius-pct`, `--floater-std`, `--no-cache`. Prepared scenes are
cached under `data/cache/`, keyed by those settings — pass `--no-cache`
while editing the pipeline, or the cache will hand back a stale scene.

```
python scripts/inspect_ply.py   data/raw/garden.ply
python scripts/render_view.py   data/raw/garden.ply --clean --elev 25
python scripts/seam_baseline.py data/raw/garden.ply --tiles 6 1 --elev 4
```

Browser renderer:

```
python scripts/export_splat.py data/raw/garden.ply --clean
python -m http.server 8000 --directory web
# http://localhost:8000/?scene=garden
```

Roughly 420k splats at 120 fps with a 10 ms sort. The panel has a
depth-sorting toggle — the seam question in one click — and prints camera
flags that reproduce the same view in `render_view.py`.

---

## First result

**Tile-local sorting versus global sorting on identical geometry**
(garden, 1.5 m patch, 6x1 tiles, elevation 4 degrees). Both renders see the
same splats from the same camera. The only difference is whether splats
from different tiles may sort against each other — the difference between a
single-scene renderer and GSWT's per-tile pre-sorting.

| angle from seam    | affected pixels | max per-pixel error |
| ------------------ | --------------- | ------------------- |
| 0 (along the seam) | 3.78%           | 0.42                |
| 15                 | 0.42%           | 0.10                |
| 30 and beyond      | 0.00%           | 0.00                |

Both coverage and strength scale with tile count: two tiles give 0.14% and
0.19. The sharp angular cutoff is the diagnostic. A content or geometry
discontinuity would show from every angle; appearing only when the view runs
along the boundary, and vanishing within 30 degrees, is the signature of
depth ordering.

---

## Measured findings (garden, 730,850 splats, SH degree 3)

These drove the design decisions. Re-measure for any new scene.

| Finding                         | Value                            | Consequence                     |
| ------------------------------- | -------------------------------- | ------------------------------- |
| Ground tilt                     | 25° off any axis                 | Alignment is mandatory          |
| Splats semi-transparent         | 57% between 0.1 and 0.9          | Blend order decides the pixel   |
| Depth complexity                | 19 deep median, 331 at p90       | Order matters, quantified       |
| Size vs spacing                 | 3.09 (overlapping)               | Seams will be overlap, not gaps |
| Anisotropy                      | median 4.5, p99 104, max 129,425 | Splats are slivers              |
| Past StopThePop's 1/scale clamp | 31.5%                            | That clamp is load-bearing      |
| SH truncation error             | deg 1: 3.3/255, deg 0: 3.8/255   | Degree 0 nearly free            |
| Storage by SH degree            | 140 / 35 / 8.8 MB                | Degree 0 chosen                 |
| CPU render speed                | ~30s at 400x300                  | A GPU renderer is not optional  |

Depth complexity and semi-transparency together are the empirical basis of
the project: 19 stacked semi-transparent layers means blend order decides
the image, which is why seams appear where two differently-sorted regions
meet.

---

## Decisions

- **Package name `bozkir`, ASCII.** The repo folder keeps `bozkır`; the
  importable package cannot risk the dotless ı across editors, consoles and
  git.
- **Align to +z.** GSWT §3.1 tiles on the XY plane.
- **SH degree 0 for rendering.** 3.8/255 mean error, 8.8 MB instead of 140.
  Degree 1 costs 35 MB if terrain looks lifeless.
- **Orthographic first, then perspective.** Orthographic projection is
  linear so its 2D covariance is exact — bugs there are unambiguous. It is
  also what GSWT §3.2 needs for tile construction.
- **Back-to-front blending.** 3DGS goes front-to-back for early termination;
  back-to-front gives the same result and reads clearly.
- **Keep the CPU renderer.** Slow and correct, so a fast GPU renderer can be
  diffed against it.
- **WebGL2 now, WebGPU later.** The first milestone is plain rasterisation,
  identical in both, and WebGL2 has readable references to check against.
  Sorting lives alone in `sort-worker.js` because that is the piece the
  research replaces, and the piece WebGPU compute shaders would transform.
- **`.splat`, 32 bytes per splat.** Positions and scales exact, colour
  within 1/255, rotation within 0.8 degrees. 13.5 MB instead of 181.

---

## Bugs found, worth not reintroducing

- **Half-pixel sampling.** Sampling at pixel corners rather than centres
  shifts every splat 0.5 px. Invisible in one image, very visible at a tile
  boundary.
- **`f_rest` is channel-major.** `reshape(n, 3, k).transpose(0, 2, 1)`. The
  intuitive `reshape(n, k, 3)` gives plausible images with wrong
  view-dependent colour.
- **Field names must sort numerically.** `f_rest_10` precedes `f_rest_2`
  lexically, silently permuting the SH coefficients.
- **`eigh` eigenvector signs are arbitrary.** Cannot decide which end of a
  scene is the ground; fit both ends, keep the flatter.
- **Which side of a plane is up is not in the geometry.** A plane looks the
  same from both sides, so half of all scenes came out inverted. Skew of the
  height distribution plus a mass test recovers it for most; `--flip` exists
  for the rest.
- **Rotating positions without orientations.** Looks plausible, is
  permanently wrong. `test_forgetting_to_rotate_orientations_is_caught`
  exists for this.
- **Eigenvalue ratios are ill-conditioned here.** At 1e5:1 anisotropy the
  smallest eigenvalues round below zero in float64. Compare covariances
  against `R Σ Rᵀ` directly instead.
- **View directions must be computed on the visible subset.** Numpy
  broadcasts silently when counts happen to match.
- **A GLSL constant is not a JavaScript constant.** `DILATION` was used
  inside a shader string where it was never declared. The shader failed to
  compile, the module threw, and the page sat there looking idle.
- **Two copies of a file drift.** Functions ended up defined in two modules
  each; tests passed against one copy and failed to import against the other.
- **A failing test is not always a failing program.** The first
  perspective-vs-orthographic check reported 0.51 correlation; the real
  disagreement was 0.027 px of framing offset.

---

## Next

The deliverable is a browser demo: terrain built from captured material,
flown through, looking good. The measurement work serves that, not the other
way round.

Two tracks, in parallel. Content has latency — weather, travel, training
queues — so it should not wait on engine work.

**Content**

1. **Capture an exemplar.** 2-3 m of gravel, dry ground or rocky slope.
   ~100 phone photos, slow spiral, twenty minutes on site.
2. **Train on Colab's free T4** (16 GB). The 4 GB local card was never going
   to do this and does not need to.
3. **Run it through the existing pipeline.** A scene it has never seen is
   the real test of everything built so far. Garden is a table, not terrain,
   and its ground is not flat; real exemplars should align to ~0.04 degrees
   rather than 1.78, and should not need `--flip`.

**Engine**

1. ~~WebGL renderer~~ — done.
2. **Instancing.** One patch in memory, drawn many times. This is where it
   stops being a scene viewer and becomes a terrain engine.
3. **Wang tiling.** Edge-coloured tile set, matching rule, aperiodic layout.
   `grid()` currently repeats one patch, which the eye catches immediately.
4. **Relief.** Height field so the ground has silhouette. GSWT §3.6
   Eqs. 4-5.
5. **Distance handling.** LOD, so far terrain does not cost what near
   terrain costs.
6. **The seam.** Last, deliberately: it is the research contribution and
   also the part that can fail without destroying the project. Overlap
   bands, local order stabilisation, stochastic transparency.

**Also:** StopThePop's popping metric. RAFT optical flow warps frame `i` to
`i+t`; FLIP measures the difference against the actually-rendered frame,
averaged over the video, `t ∈ {1, 7}`. Operates on rendered PNGs only — no
CUDA rasterizer, no trained scene, none of their code required. Their
`PoppingDetection` module is self-contained.

---

## Papers

Read in this order.

1. **3DGS** (Kerbl et al. 2023) — §4 for the representation (Eqs. 5, 6), §6
   and Appendix C Algorithm 2 for the rasterizer. The 64-bit sort key
   packing tile index and depth is the core trick.
2. **StopThePop** (Radl, Steiner, Kerbl, Steinberger, TOG 2024) — §3.1 (3DGS
   sorts on view-space z of the mean: consistent under translation, not
   rotation), §3.2 (Eq. 4 for `t_opt`, and the 1e3 clamp on `S⁻¹`), §6 and
   Appendix C for the popping metric. Table 1 calibrates cost: full per-ray
   sort is 100×, a resort window of 16–24 removes most popping at 2–6×.
3. **GSWT** (SIGGRAPH Asia 2025) — §3.4 above all: 9 pre-sort views per
   tile, tile-level topological sorting from boundary normals, selective
   merging when the camera nears a boundary plane. Fig. 6 is the artifact.
   §3.5 for LOD, §3.6 Eqs. 4–5 for placing splats on a height field, §5 for
   the limitations.
4. **Hybrid Gaussian Wang Tiles** (SIGGRAPH 2026) — the sorting section.
5. **Graph-GSReg** (arXiv 2606.29782) — naive-merge failure modes.

## Also worth tracking

- Zeng, Ma, Sander (HKUST / Eyeline) — two papers in eight months; the
  people most likely to do this next.
- Radl, Steiner, Steinberger (Graz), Kerbl (TU Wien) — sorting, popping.
- Lin Gao, Jie Yang — MesoSplats, competing splat texture synthesis.
- **Nobuyuki Umetani** (U. Tokyo), with Watanabe and Tojo — *3D Gabor
  Splatting*, Eurographics 2025. Procedural noise as a splat primitive, in
  Japan, and not on the original professor list.
- radiancefields.com (Michael Rubloff) — the field's aggregator.
- kesen.realtimerendering.com — SIGGRAPH paper lists; earliest warning of
  the next paper in this line.