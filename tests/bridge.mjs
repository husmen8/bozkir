// Python/JS bridge for the parity tests. The material rule and the terrain
// generator exist in both languages; this runs the JS side on cases sent
// from tests/test_pipeline.py so both can be required to agree cell for cell.
//
//   stdin:  {"fn": "classify" | "generate", "cases": [...]}
//   stdout: [result, ...]
//
// A new shared module needs one entry in RUN and one Python test.
import { classify } from '../web/landform.js';
import { generate } from '../web/terrain.js';

const RUN = {
  classify: ({ z, n, classes, opts }) => {
    const o = { ...opts };
    if (o.sediment) o.sediment = Float64Array.from(o.sediment);
    const r = classify(Float64Array.from(z), n, classes, o);
    return { cls: Array.from(r.cls), strength: Array.from(r.strength),
             runnerUp: Array.from(r.runnerUp), source: r.source };
  },
  generate: (c) => {
    const r = generate(c);
    return { z: Array.from(r.z), sediment: Array.from(r.sediment) };
  },
};

let raw = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (d) => { raw += d; });
process.stdin.on('end', () => {
  const { fn, cases } = JSON.parse(raw);
  if (!RUN[fn]) {
    process.stderr.write(`bridge: no function '${fn}'; have ${Object.keys(RUN).join(', ')}\n`);
    process.exit(2);
  }
  process.stdout.write(JSON.stringify(cases.map(RUN[fn])));
});
