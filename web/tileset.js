// Opening a tileset in the browser, from a zip or from loose files, so a
// tileset exported on any machine can be dropped on the page (as GSWT's
// renderer takes a zip of tiles). Python writes tilesets, the page reads
// them; a file joins the two, not a server.
//
// The zip reader is hand-written: a tileset is two stored or deflated files
// with no encryption or zip64, and DecompressionStream does the inflating.

const SIG_EOCD = 0x06054b50;
const SIG_CENTRAL = 0x02014b50;
const SIG_LOCAL = 0x04034b50;

/** Read a zip into a map of name to bytes. Sizes come from the central
 *  directory: a local header may hold zeroes with the real sizes after the
 *  data. */
export async function readZip(buffer) {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  // The end record is last unless there is a comment: scan back over the
  // largest comment allowed.
  let eocd = -1;
  const from = Math.max(0, bytes.length - 22 - 0xffff);
  for (let i = bytes.length - 22; i >= from; i--) {
    if (view.getUint32(i, true) === SIG_EOCD) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error('not a zip file');

  const count = view.getUint16(eocd + 10, true);
  let p = view.getUint32(eocd + 16, true);
  if (p === 0xffffffff) throw new Error('zip64 archives are not supported');

  const out = new Map();
  for (let k = 0; k < count; k++) {
    if (view.getUint32(p, true) !== SIG_CENTRAL) break;
    const method = view.getUint16(p + 10, true);
    const compressed = view.getUint32(p + 20, true);
    const nameLen = view.getUint16(p + 28, true);
    const extraLen = view.getUint16(p + 30, true);
    const commentLen = view.getUint16(p + 32, true);
    const local = view.getUint32(p + 42, true);
    const name = new TextDecoder().decode(
      bytes.subarray(p + 46, p + 46 + nameLen));
    p += 46 + nameLen + extraLen + commentLen;

    if (name.endsWith('/')) continue;                 // a directory entry
    if (view.getUint32(local, true) !== SIG_LOCAL) continue;
    // The local header's own name/extra lengths, which may differ from the
    // central ones.
    const dataAt = local + 30 + view.getUint16(local + 26, true)
                            + view.getUint16(local + 28, true);
    const raw = bytes.subarray(dataAt, dataAt + compressed);

    if (method === 0) {
      out.set(name, raw.slice());
    } else if (method === 8) {
      const s = new Blob([raw]).stream()
        .pipeThrough(new DecompressionStream('deflate-raw'));
      out.set(name, new Uint8Array(await new Response(s).arrayBuffer()));
    } else {
      throw new Error(`${name}: unsupported compression method ${method}`);
    }
  }
  return out;
}

/** The basename of a path inside a zip, ignoring any folder it sits in. */
function base(path) { return path.split('/').pop(); }

/** Pick the tileset out of a set of named blobs, by extension (zips made
 *  by hand put files in folders and carry __MACOSX or .DS_Store). The
 *  largest .splat wins; a .json with the same stem is preferred. */
export function pickTileset(files) {
  const usable = [...files.entries()]
    .filter(([n]) => !base(n).startsWith('.') && !n.startsWith('__MACOSX/'));

  const splats = usable.filter(([n]) => n.toLowerCase().endsWith('.splat'))
    .sort((a, b) => b[1].length - a[1].length);
  if (!splats.length) throw new Error('no .splat file in this tileset');

  const [splatName, splatBytes] = splats[0];
  const stem = base(splatName).replace(/\.splat$/i, '');
  const jsons = usable.filter(([n]) => n.toLowerCase().endsWith('.json'));
  const paired = jsons.find(([n]) => base(n).replace(/\.json$/i, '') === stem)
              || jsons[0];

  let meta = null;
  if (paired) {
    try {
      meta = JSON.parse(new TextDecoder().decode(paired[1]));
    } catch (e) {
      throw new Error(`${base(paired[0])} is not valid JSON: ${e.message}`);
    }
  }
  return { name: base(splatName), buffer: splatBytes.buffer.slice(
    splatBytes.byteOffset, splatBytes.byteOffset + splatBytes.length), meta };
}

/** The settings the metadata gives, with nulls (not defaults) for what it
 *  lacks, so the caller can warn: a .splat has no header, and a wrong tile
 *  size looks like a broken export. */
export function describe(meta) {
  if (!meta) return { size: null, wang: false, lod: 1, note: 'no metadata' };
  return {
    size: typeof meta.size === 'number' ? meta.size : null,
    wang: !!meta.wang,
    lod: meta.lod || 1,
    note: meta.size ? null : 'metadata has no tile size',
  };
}

/** Open whatever was dropped: one zip, or the .splat and .json themselves. */
export async function openDrop(fileList) {
  const files = [...fileList];
  if (!files.length) throw new Error('nothing to open');

  const zip = files.find((f) => /\.zip$/i.test(f.name));
  if (zip) return pickTileset(await readZip(await zip.arrayBuffer()));

  const map = new Map();
  for (const f of files) {
    map.set(f.name, new Uint8Array(await f.arrayBuffer()));
  }
  return pickTileset(map);
}

// ---------------------------------------------------------------- writing

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    t[i] = c >>> 0;
  }
  return t;
})();

function crc32(bytes) {
  let c = 0xFFFFFFFF;
  for (let i = 0; i < bytes.length; i++) {
    c = CRC_TABLE[(c ^ bytes[i]) & 0xFF] ^ (c >>> 8);
  }
  return (c ^ 0xFFFFFFFF) >>> 0;
}

/** Write a zip of [name, Uint8Array] entries, stored uncompressed (the
 *  contents are PNGs, already compressed). */
export function writeZip(entries) {
  const enc = new TextEncoder();
  const parts = [];
  const central = [];
  let offset = 0;

  for (const [name, data] of entries) {
    const nb = enc.encode(name);
    const body = data instanceof Uint8Array ? data : new Uint8Array(data);
    const sum = crc32(body);

    const lh = new Uint8Array(30 + nb.length);
    const lv = new DataView(lh.buffer);
    lv.setUint32(0, SIG_LOCAL, true);
    lv.setUint16(4, 20, true);            // version needed
    lv.setUint16(8, 0, true);             // stored
    lv.setUint32(14, sum, true);
    lv.setUint32(18, body.length, true);
    lv.setUint32(22, body.length, true);
    lv.setUint16(26, nb.length, true);
    lh.set(nb, 30);
    parts.push(lh, body);

    const ch = new Uint8Array(46 + nb.length);
    const cv = new DataView(ch.buffer);
    cv.setUint32(0, SIG_CENTRAL, true);
    cv.setUint16(4, 20, true);
    cv.setUint16(6, 20, true);
    cv.setUint16(10, 0, true);
    cv.setUint32(16, sum, true);
    cv.setUint32(20, body.length, true);
    cv.setUint32(24, body.length, true);
    cv.setUint16(28, nb.length, true);
    cv.setUint32(42, offset, true);
    ch.set(nb, 46);
    central.push(ch);
    offset += lh.length + body.length;
  }

  const cdSize = central.reduce((t, c) => t + c.length, 0);
  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  ev.setUint32(0, SIG_EOCD, true);
  ev.setUint16(8, entries.length, true);
  ev.setUint16(10, entries.length, true);
  ev.setUint32(12, cdSize, true);
  ev.setUint32(16, offset, true);

  const all = [...parts, ...central, eocd];
  const out = new Uint8Array(all.reduce((t, p) => t + p.length, 0));
  let w = 0;
  for (const p of all) { out.set(p, w); w += p.length; }
  return out;
}
