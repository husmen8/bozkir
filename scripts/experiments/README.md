# Experiments

Scripts kept for the record rather than for use. Nothing in the pipeline,
the viewer or the README depends on them; the test suite still checks that
each one imports and builds its parser, so they keep working.

- `export_tileset.py` - the first exporter: a plain grid of patches, before
  Wang tiles. Superseded by `scripts/export_wang.py`.
- `seam_baseline.py` - measures the seam artifact of drawing tiles
  separately against drawing the whole scene at once.
- `render_view.py`, `render_ortho.py` - CPU reference renders of a PLY,
  perspective and top-down, from before the WebGL viewer.
