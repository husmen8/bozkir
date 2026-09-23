# Session Notes — Preset Configuration & Scene Tuning (2026-09-23)

## What was done

Created and tuned `presets.json` entries for all viable datasets in `data/raw/`. Resolved performance, orientation, file size, and seam artifact issues on dense 10M-splat scenes (`bigsur`).

---

## Dataset analysis results

| Dataset | Splats | Planarity | Tilt | Aniso med | Up axis (raw) | Verdict |
|---|---|---|---|---|---|---|
| **desert** | 1.8M | good | low | 12.8 | Z(2) | ✅ Best — reference preset, untouched |
| **beachrock** | 3M | 0.013 ✅ | 4.4° | 12.8 | Y(1) | ✅ Good — flat, healthy geometry |
| **bigsur** | 10M | 0.168 ⚠️ | 81° raw → 3° aligned | 3.8 | Y(1) → Z(2) after align | ✅ Works with tuning (`flip`, `cut`, `max_per_tile`) |
| **bicycle** | 1M | — | — | — | Z(2) | ✅ Already had preset |
| **garden** | 730k | — | — | — | Z(2) | ✅ Already had preset |
| **italy** | 5M | 0.078 ⚠️ | 18.9° | 9.7 / 667 | Y(1) | ⚠️ Risky — steep terrain, needs loose thresholds |
| **minecraft** | 3M | 0.060 ⚠️ | 6.9° | 14.6 / 5083 | Y(1) | ⚠️ Synthetic — tight extent clipping needed |
| **mitsukejima** | 2.9M | 0.092 ❌ | 2.8° | **483.8** ❌ | Y(1) | ❌ Not viable — vertical sea stack, not tileable ground |

---

## New presets added

- **beachrock**: Modeled after desert. Flat coastal rock.
  - Key settings: `up_axis: 1`, `clean: true`, `min_cover: 0.65`, `radius_pct: 55.0`, `thickness: 0.4`.
- **italy**: Loose settings for steep hillside terrain.
  - Key settings: `up_axis: 1`, `max_tilt: 25.0`, `min_cover: 0.6`, `size: 2.0`, `radius_pct: 50.0`, `thickness: 0.5`.
- **minecraft**: Synthetic voxel scan.
  - Key settings: `up_axis: 1`, `floater_std: 1.5`, `max_extent_pct: 95.0`, `min_separation: 0.75`.
- **mitsukejima**: Excluded intentionally — README records anisotropy median of 484 (failed reconstruction / vertical monolithic rock).

---

## Bigsur fixes & deep dive (iterative)

The `bigsur` dataset presented multiple challenges due to its 10M-splat density and slanted capture angle:

1. **Search performance bottleneck**:
   - `clean: false` on 10M splats forced `extract_patch` to do boolean array indexing over 10M elements across 300+ candidate grid positions.
   - Enabling `floater_std > 0` on 10M splats built a massive KD-tree.
   - *Fix*: Set `floater_std: 0` during scene clean or rely on pre-aligned caching.

2. **Inverted terrain (Flip issue)**:
   - The raw scan has an 81.5° camera tilt. Ground normal estimation picked the opposite sign, causing the export to render upside down.
   - *Fix*: Added `"flip": true`.

3. **Viewer FPS / File size bloat**:
   - Extracted tiles contained 270k–440k splats each → 6.27M splat total tileset (200MB file), tanking browser FPS.
   - *Fix*: Added `"max_per_tile": 50000`, reducing total export size to ~650k splats (21MB) with high frame rates.

4. **Diagonal seam / hard boundary artifacts**:
   - Using simple linear feathering (`blend: 0.05`) created visible X-shaped cut seams across tiles when adjoining patches had differing textures or heights.
   - *Fix*: Enabled graph-cut boundary synthesis (`"cut": true`, `"cut_band": 0.14`, `"cut_res": 160`). The graph cut algorithm finds the minimum-energy path through agreement areas rather than cutting straight through mismatching geometry.

---

## Seam & boundary artifact detection (code metrics)

When evaluating boundary compatibility before visual inspection, the pipeline reports two key metrics in `preview_patches.py` / `export_wang.py`:

- **Appearance spread** (`spread` in `bozkir/wang.py`):
  - Measures the standard deviation of mean RGB and normal vectors among selected patches.
  - Values `< 0.05` indicate high homogeneity (clean seams); values `> 0.10` warn that tile boundaries will exhibit noticeable color/texture breaks.
- **Axis balance** (`axis balance` metric):
  - Measures divergence between horizontal (H) and vertical (V) edge patch sets.
  - Large values indicate asymmetric lighting or directional terrain slope that will show recurring pattern seams.
- **Mitigation rule**: If `appearance spread > 0.05` or terrain has distinct features, **always enable graph cuts** (`"cut": true`).

---

## Current `bigsur` preset configuration

```json
"bigsur": {
  "auto": true,
  "clean": false,
  "cover_margin": 0.1,
  "cut": true,
  "cut_band": 0.14,
  "cut_res": 160,
  "extract_margin": 0.35,
  "flip": true,
  "floater_std": 2.0,
  "max_below": 0.65,
  "max_extent_pct": 99.0,
  "max_per_tile": 50000,
  "max_tilt": 25.0,
  "min_cover": 0.6,
  "min_separation": 0.5,
  "radius_pct": 60.0,
  "raw": false,
  "size": 1.5,
  "stride": 0.4,
  "thickness": 0.375,
  "up_axis": 2
}
```

---

## Summary of modified files

- `presets.json` — Added entries for `beachrock`, `italy`, `minecraft`; optimized `bigsur` (flip, cut, max_per_tile).
- `web/data/bigsur.splat` + `web/data/bigsur.json` — Re-exported compact 21MB graph-cut tileset for web viewing.
- `session_notes.md` — Updated session documentation.
