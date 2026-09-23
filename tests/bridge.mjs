// The bridge between the two languages, for testing.
//
// Some of this project exists twice - the material rule and the terrain
// generator are written in Python and in JavaScript, because figures and
// validation run in one and the viewer in the other. Two copies drift
// unless something holds them together, and this is that something: it
// runs a JavaScript function on cases sent from Python and sends the
// results back, so tests/test_pipeline.py can require the same answer from
// both, cell for cell.
//
//   stdin:  {"fn": "classify" | "generate", "cases": [...]}
//   stdout: [result, ...]
//
// Adding a third shared module means adding one entry to RUN below and one
// test on the Python side - nothing else.
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
