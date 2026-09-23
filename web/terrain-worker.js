// Runs the terrain generator off the main thread. A 512 alpine terrain is
// several seconds of arithmetic; on the page itself that is several
// seconds of a frozen viewer.
//
//   in:   { id, spec }                 spec as generate() in terrain.js
//   out:  { id, progress }             once per erosion step, 0..1
//         { id, z, sediment, size, settings }   Float32Arrays, transferred
//         { id, error }
import { generate } from './terrain.js';

self.onmessage = (e) => {
  const { id, spec } = e.data;
  try {
    const r = generate(spec, (done, total) => {
      self.postMessage({ id, progress: done / total });
    });
    const z = Float32Array.from(r.z);
    const sediment = Float32Array.from(r.sediment);
    self.postMessage({ id, z, sediment, size: r.size, settings: r.settings },
                     [z.buffer, sediment.buffer]);
  } catch (err) {
    self.postMessage({ id, error: String(err && err.message || err) });
  }
};