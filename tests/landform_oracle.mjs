// Runs web/landform.js classify() on cases read from stdin and prints the
// results as JSON. Used by tests/test_pipeline.py to hold the Python port
// in bozkir/landform.py to the code the viewer actually runs, and by
// scripts/validate_rule.py --engine js.
//
//   stdin:  {"cases": [{"z": [...], "n": 16, "classes": 2, "opts": {...}}]}
//   stdout: [{"cls": [...], "strength": [...], "source": "..."}]
import { classify } from '../web/landform.js';

let raw = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (d) => { raw += d; });
process.stdin.on('end', () => {
  const { cases } = JSON.parse(raw);
  const out = cases.map(({ z, n, classes, opts }) => {
    const o = { ...opts };
    if (o.sediment) o.sediment = Float64Array.from(o.sediment);
    const r = classify(Float64Array.from(z), n, classes, o);
    return { cls: Array.from(r.cls), strength: Array.from(r.strength),
             source: r.source };
  });
  process.stdout.write(JSON.stringify(out));
});
