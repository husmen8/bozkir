// Opening a tileset in the browser, from a zip or from loose files.
//
// Until now a scene arrived one of two ways: a file sitting in web/data
// that the page fetches by name, or a .splat picked from the file dialog
// whose .json the page then looked for in web/data by the same name. Both
// assume the tileset is already on the server, which means the demo can
// only ever show what was exported into the repo. Somebody handed a link
// cannot bring their own.
//
// GSWT's own renderer solves this by taking a zip of tiles as its input and
// preprocessing it in the page, and the same shape fits here: the Python
// side stays a constructor that writes a tileset, the page stays a renderer
// that reads one, and the two are joined by a file rather than by a server.
//
// The zip reader is written out rather than pulled in. A tileset is two
// files stored with no encryption and no zip64, which is a small enough
// corner of the format to read directly, and DecompressionStream has done
// the inflating since Chrome 103. A library would be more code shipped for
// a case that is already covered.

const SIG_EOCD = 0x06054b50;
const SIG_CENTRAL = 0x02014b50;
const SIG_LOCAL = 0x04034b50;

/** Read a zip into a map of name to bytes.
 *
 *  The central directory is authoritative, not the local headers: a local
 *  header may carry zeroed sizes with the real ones trailing the data, and
 *  a zip written that way reads as empty if you trust the local copy.
 */
export async function readZip(buffer) {
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);

    // The end record sits at the very end unless there is a comment, so scan
    // back over the largest comment the format allows plus the record itself.
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
        // The local header's own name and extra lengths, not the central ones:
        // the two are allowed to differ and usually do.
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

/** Pick the tileset out of a set of named blobs.
 *
 *  A zip exported by hand usually has the two files inside a folder, and
 *  may carry a __MACOSX shadow tree or a .DS_Store beside them, so entries
 *  are matched by extension rather than by path. When several .splat files
 *  are present the largest wins, on the grounds that the others are
 *  previews or offcuts; when a .json shares a name with the chosen .splat
 *  that one is preferred over any other.
 */
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
    return {
        name: base(splatName), buffer: splatBytes.buffer.slice(
            splatBytes.byteOffset, splatBytes.byteOffset + splatBytes.length), meta
    };
}

/** What the settings mean, for anything the metadata does not say.
 *
 *  A .splat carries no header, so without metadata the page has to guess,
 *  and the one guess that matters is the tile size - get it wrong and the
 *  tiles either overlap or leave gaps, which reads as a broken export
 *  rather than a missing file. Returning nulls rather than defaults lets
 *  the caller say so.
 */
export function describe(meta) {
    if (!meta) return { size: null, wang: false, lod: 1, note: 'no metadata' };
    return {
        size: typeof meta.size === 'number' ? meta.size : null,
        wang: !!meta.wang,
        lod: meta.lod || 1,
        note: meta.size ? null : 'metadata has no tile size',
    };
}

/** Open whatever the person dropped: one zip, or the loose files themselves.
 *
 *  Accepting loose files matters more than it looks. Exporting produces a
 *  .splat and a .json side by side, and the obvious thing to do with two
 *  files is select both - being told to zip them first would be a step
 *  invented by the page for its own convenience.
 */
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

/** Write a zip. Entries are [name, Uint8Array], stored uncompressed.
 *
 *  Stored rather than deflated because everything this writes is already
 *  compressed - PNG frames - and deflating them again costs time to save
 *  nothing. It also keeps the writer to arithmetic, with no async in the
 *  middle of building a file.
 */
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