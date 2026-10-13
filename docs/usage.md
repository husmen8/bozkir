# Using bozkır

The details behind the README: running the viewer, making a tileset from
your own capture, generating terrain, and the settings used for the figures.
What each control in the viewer does is in [viewer-guide.md](viewer-guide.md)
(the same text as the viewer's info drawer).

## The viewer

```bash
cd web && python -m http.server 8000      # ES modules will not load from file://
```

Open `http://localhost:8000`. It opens on the desert tileset over its
terrain; with no data at all it opens on the starter scene. Drop a tileset
zip, or its `.splat` and `.json` together, on the page to view another.

URL options:

| option | effect |
|---|---|
| `?scene=desert` / `?scene=starter` | which tileset |
| `?gen=canyon&seed=3&size=256` | generate this terrain in the browser |
| `?height=<name>` | a height field from `web/data` under the current tiles |
| `?preset=readme` | apply a named preset from `web/views.json` |

**Explore** shows the scene, world, materials, camera and look; **Research**
(the tab, or M) adds the renderer's switches, ordering, measurement, debug
views and live numbers. **story** steps through capture, tiles, world and
rule. **walk on it** puts you on the ground in first person (the terrain is
the collider). **share** copies a link that reopens the exact view.

**show the decision** paints each cell with the class the rule chose, grey
where it was nearly a tie. That is the decision, not the material drawn:
tiles are sorted into classes by average appearance, so a tile can hold
some of the other material, and cells in the blend band draw some of their
runner-up.

**Presets.** `web/views.json` holds named settings: the README's own, the
matched pair for the rule on/off figure, a low vista, height alone, and the
story chapters. A preset sets each control as a hand on the slider would,
so it can only do what the panel can; a test checks every preset names real
controls with values in range.

**Figures.** Figure mode (F) hides the panel; **save PNG** (P) writes the
canvas at full resolution, named after the scene, grid, rule and camera.
Settings used for the README views: size 24, relief scale = size, relief
1.2, follows height 40%, patch size 3, share 30%, decisiveness 3, blend on
at 0.25 (the `readme` preset).

Relief scale follows the grid until you move it, warns when it is below the
grid (the height field then repeats), and follows again once dragged back.

## From your own capture

One command, for a first capture:

```bash
python scripts/make_tileset.py my_ground.ply
```

It tries two materials, then one, with defaults meant for an untuned
capture, prints everything the exporter says, and ends with a verdict -
*good*, *marginal* (and why) or *no* (and why) - and the address to open.

The same steps with every control:

```bash
python scripts/inspect_ply.py data/raw/scene.ply          # can it become tiles?
python scripts/preview_patches.py data/raw/scene.ply --save-preset scene
python scripts/export_wang.py data/raw/scene.ply --preset scene --patches 0,2,4,5
python scripts/bake_atlas.py scene                         # far field
```

The shipped tileset: `export_wang.py data/raw/desert.ply --preset desert3`,
then `bake_atlas.py desert`. Re-bake the atlas after every export; an atlas
with a different tile count is ignored.

Exporter defaults are set for a capture nobody has tuned: graph-cut seams
(`--no-cut` for speed), at most 60,000 splats per patch (`--max-per-tile 0`
keeps all), purity on, and three warnings that name their fix - a capture
that looks upside down (`--flip`), a `--size` far from the capture's scale
(with a suggested size), and patches of one material that overlap (that
material will repeat). `--classes 2` splits a capture into two materials
and refuses if it only holds one; `--colours 3` and `--rotations 3` fight
repetition (see the README). Settings that worked are saved with
`--save-preset name` into `capture_presets.json`.

## Terrain

In the viewer, **make terrain…** opens a preview of all nineteen profiles at
the current seed; step the seed and the previews follow, double-click one to
make it. Generated terrains are cached in the browser; nothing touches the
server. From Python:

```bash
python scripts/terrain_gen.py --name canyon --profile canyon --seed 3 --size 512
python scripts/heightmap.py C:/odm/copr/odm_dem/dsm.tif --name desert   # from a drone DSM
```

Both write a 16-bit height PNG (two 8-bit channels; browsers decode 16-bit
greyscale to 8 bits) and a JSON sidecar; the generator also writes its
sediment map. A profile and a seed name the same ground in Python and in the
browser, bit for bit. [terrain.md](terrain.md) maps each step to its source.

Locally, `web/data/index.json` lists the tilesets and height fields in the
folder, so tileset and terrain can be chosen separately
(`?scene=desert&height=mesa`). The exporters rebuild it;
`python scripts/index_data.py` does it by hand. It is not published - the
public demo has the desert tileset and the starter only.

## Measurement and figures

```bash
python scripts/validate_rule.py dtm.tif odm_orthophoto.tif --dsm dsm.tif
python scripts/repetition.py desert
python scripts/figures.py              # every README figure, from code
python tests/run_all.py                # ~300 tests
```

The viewer's **Research** mode also has a scaling sweep (where the renderer
stops keeping up) and a capture sweep for `scripts/pop_metric.py`.
