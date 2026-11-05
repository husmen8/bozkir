// The one-minute demo (?demo): both worlds played from a fixed plan, for
// recording the screen in one take. The plan is camera keys (as Blender's:
// a value per channel at a time, smooth in between, a repeated value a hold)
// and actions at times: captions, cards, photos, settings, the panel.
//
// Anything that rebuilds the world (the rule, a new tileset) happens while
// the camera holds or behind a card, where a stall cannot be seen. The clock
// stops while the second world loads and never steps more than 50 ms, so a
// slow frame delays the film instead of making it jump.
//
// ?demo plays it on its own; ?demo&rec waits on the title card for Space,
// for starting a screen recording first (F11 for full screen). Esc stops it.
// &slow=N plays it N times slower. ?demo&frames=60 renders it instead:
// nothing moves until scripts/record_demo.mjs asks for the next frame, and
// that waits for the frame to be drawn and sorted, so the video is smooth on
// any GPU.

/** Monotone cubic through (ts, vs) at t (Fritsch-Carlson): no overshoot,
 *  eased at the first and last key, and a repeated value holds still. */
export function monotone(ts, vs, t) {
  const n = ts.length;
  if (n === 1 || t <= ts[0]) return vs[0];
  if (t >= ts[n - 1]) return vs[n - 1];
  const d = [];
  for (let i = 0; i < n - 1; i++) d.push((vs[i + 1] - vs[i]) / (ts[i + 1] - ts[i]));
  const m = new Array(n).fill(0);
  for (let i = 1; i < n - 1; i++) m[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
  for (let i = 0; i < n - 1; i++) {
    if (d[i] === 0) { m[i] = 0; m[i + 1] = 0; continue; }
    const a = m[i] / d[i], b = m[i + 1] / d[i], s = a * a + b * b;
    if (s > 9) { const k = 3 / Math.sqrt(s); m[i] = k * a * d[i]; m[i + 1] = k * b * d[i]; }
  }
  let k = 0;
  while (t > ts[k + 1]) k++;
  const h = ts[k + 1] - ts[k], u = (t - ts[k]) / h, u2 = u * u, u3 = u2 * u;
  return (2 * u3 - 3 * u2 + 1) * vs[k] + (u3 - 2 * u2 + u) * h * m[k]
       + (-2 * u3 + 3 * u2) * vs[k + 1] + (u3 - u2) * h * m[k + 1];
}

// Camera keys: t in seconds, angles in degrees, distance in tiles, looking
// at an anchor ('tile' the first captured tile, 'water', 'water2' a water
// spot further along the river, 'mid' halfway between tile and water,
// 'centre', or [x, y] in tiles from the centre). The rise is the growing
// world: radius and band in tiles around the first tile; past its last key
// it is off.
export const SHOTS = [
  { world: 'river', preset: 'river',
    set: { classrule: false, merging: false, ranges: 0.5, mist: 0 },
    from: 0, to: 31,
    cam: [
      { t: 0, az: 30, el: 50, d: 2.3, at: 'tile' },
      { t: 6.7, az: 70, el: 46, d: 2.5, at: 'tile' },
      { t: 10.7, az: 92, el: 42, d: 5.5, at: 'tile' },
      { t: 15.7, az: 118, el: 36, d: 13, at: 'mid' },
      { t: 17.3, az: 124, el: 35, d: 14.5, at: 'mid' },
      { t: 18.9, az: 126, el: 35, d: 14.6, at: 'mid' },
      { t: 23.5, az: 160, el: 27, d: 12, at: 'mid' },
      { t: 27, az: 205, el: 12, d: 4.4, at: 'water' },
      { t: 31, az: 238, el: 8, d: 4.0, at: 'water2' },
    ],
    // Out past the world to the distant land, which arrives with it.
    rise: [
      { t: 0, r: 0.9, band: 0.8 },
      { t: 6.9, r: 0.9, band: 0.8 },
      { t: 8.3, r: 2.2, band: 1.2 },
      { t: 15.7, r: 40, band: 3 },
      { t: 17.2, r: 110, band: 6 },
    ] },
  { world: 'desert', preset: 'readme',
    set: { merging: false, relief: 1.4, seahaze: 0.5, ranges: 0.25, mist: 0 },
    from: 31, to: 66,
    cam: [
      { t: 31, az: 15, el: 28, d: 17, at: 'centre' },
      { t: 34.6, az: 32, el: 22, d: 13, at: 'centre' },
      // Low over the dunes, towards the haze.
      { t: 37.4, az: 52, el: 8, d: 5.5, at: [-5, 2] },
      { t: 41.4, az: 70, el: 6, d: 5, at: [5, -1] },
      { t: 44.2, az: 80, el: 32, d: 13, at: 'centre' },
      { t: 47.4, az: 88, el: 36, d: 14, at: 'centre' },
      // Back far enough to see the world grow by a ring and shrink again.
      { t: 49.4, az: 98, el: 62, d: 34, at: 'centre' },
      { t: 55.2, az: 108, el: 62, d: 34.5, at: 'centre' },
      { t: 57.4, az: 118, el: 36, d: 16, at: 'centre' },
      { t: 66, az: 142, el: 33, d: 15.5, at: 'centre' },
    ] },
];

export const CARDS = {
  title: { big: 'bozkır', line: 'Worlds grown from a few square metres of real ground.' },
  dunes: { small: 'a second capture', big: 'Coastal dunes, California',
           photos: ['dunes-1', 'dunes-2', 'dunes-3'],
           credit: 'drone photos: OpenDroneMap COPR (Tobias and Mandel 2009), CC BY-SA 4.0' },
  end: { big: 'bozkır', line: 'Real ground, captured. Worlds, placed by the land. In the browser.',
         small: 'github.com/husmen8/bozkir' },
};

export const PHOTOS = {
  capture: {
    left: { label: 'gravel bar · Lütschine, Switzerland', photos: ['gravel-1', 'gravel-2'] },
    right: { label: 'grassland · Uruguay', photos: ['grass-1', 'grass-2'] },
    credit: 'drone photos: Mair et al. 2022 and the Uruguayan grassland UAV set, CC BY 4.0',
  },
};

// Tables, drawn from the world on screen: its tiles as the far-field atlas
// holds them with their edge codes, the maps its rule read with the
// decision, and the water's measurement (web/data/water.json).
// Colour ramps for the rule's maps, low to high.
const RAMPS = {
  height: [[0.17, 0.23, 0.16], [0.37, 0.48, 0.27], [0.72, 0.66, 0.42], [0.93, 0.90, 0.82]],
  slope: [[0.0, 0.0, 0.02], [0.32, 0.07, 0.49], [0.72, 0.22, 0.47], [0.99, 0.54, 0.38], [0.99, 0.99, 0.75]],
  tpi: [[0.23, 0.30, 0.75], [0.87, 0.87, 0.87], [0.71, 0.02, 0.15]],
  water: [[0.04, 0.10, 0.16], [0.12, 0.31, 0.47], [0.31, 0.61, 0.85], [0.84, 0.93, 1.0]],
  sediment: [[0.20, 0.13, 0.07], [0.55, 0.30, 0.08], [0.92, 0.58, 0.18], [1.0, 0.92, 0.62]],
};
const MAPS = {
  height: ['height', 'height'], slope: ['slope', 'slope'], tpi: ['hollow to ridge', 'tpi'],
  flow: ['where water runs', 'water'], near: ['near the river', 'water'],
  sediment: ['sediment left', 'sediment'],
};

// Actions. set: controls by id, as a preset would set them; tween: a slider
// (or a list of them) moved over s seconds; hl: a control to light up while
// it changes; tab: a sidebar page; scroll: the panel from one fraction of
// its height to another; scrollTo: a control brought into view; grow: the
// world growing or shrinking by whole rings of cells (a square rise around
// the centre, radius in tiles; off: remove it when done); hold: keep the
// relief scale, so a new world size extends the ground instead of
// stretching it. table: one of the tables above, or null.
export const EVENTS = [
  { t: 0, photos: 'capture' },
  { t: 0.4, cap: '<b>1 · capture</b> Drone photos of a gravel bar in Switzerland and grassland in Uruguay, trained into 3D Gaussian splats.' },
  { t: 5.3, gather: true },
  { t: 6.3, photos: null, cap: null },
  { t: 7.1, cap: '<b>2 · tiles</b> Cut into tiles whose edges always match, so a few square metres grow into ground of any size.' },
  { t: 7.5, table: 'tiles' },
  { t: 12.6, table: null },
  { t: 12.9, cap: null },
  { t: 13.5, cap: '<b>3 · terrain</b> Laid over a generated valley with a river cut into it. Materials here are at random.' },
  { t: 17.6, cap: null },
  { t: 17.9, set: { classrule: true } },
  { t: 18.3, cap: '<b>4 · placed by the land</b> Gravel bars low beside the river, grass on the higher banks.' },
  { t: 18.5, table: 'rule' },
  { t: 22.6, cap: null },
  { t: 23.0, tween: { id: 'mist', to: 0.35, s: 2.5 } },
  { t: 23.2, table: null },
  { t: 23.4, cap: '<b>water</b> Colour, depth and ripples measured from the same drone photos; the mist lies where the land drains.' },
  { t: 23.6, table: 'water' },
  { t: 27.2, table: null },
  { t: 30.0, cap: null },
  { t: 30.3, card: 'dunes' },
  { t: 33.4, card: null },
  { t: 34.2, cap: '<b>same pipeline, another place</b> Three materials found in one capture, cut into tiles the same way.' },
  { t: 34.4, table: 'tiles' },
  { t: 38.6, table: null, cap: null },
  { t: 41.4, table: 'rule', cap: '<b>placed by the land</b> Scrub collects in the hollows, where erosion left sediment; sand stays on the ridges.' },
  { t: 44.0, cap: null },
  { t: 44.2, set: { classview: true } },
  { t: 44.4, cap: '<b>the rule\'s choice</b> A colour per material, grey where it was nearly a tie.' },
  { t: 46.8, set: { classview: false }, cap: null, table: null },
  // Distant land off with the panel: it continues the ground past the
  // edge, which would hide the world growing by a ring.
  { t: 47.3, panel: true, set: { ranges: 0 } },
  { t: 47.6, cap: '<b>live in the browser</b> Every step is a setting you can change.', scrollTo: { id: 'grid', s: 0.8 } },
  // Thinner haze while the camera is far back, or the world is lost in it.
  { t: 47.8, tween: [{ id: 'fog', to: 0.35, s: 1.4 }, { id: 'seahaze', to: 0.1, s: 1.4 }] },
  { t: 48.6, hold: true, grow: { from: 12.4, to: 12.4, s: 0, band: 0.9 }, set: { grid: 40 }, hl: 'grid' },
  { t: 48.9, grow: { from: 12.4, to: 21, s: 2.2, band: 0.9, off: true },
    cap: '<b>world size</b> From 24 to 40 tiles across: more of the same ground, placed by the same land.' },
  { t: 51.6, scrollTo: { id: 'classshare0', s: 0.6 }, cap: null },
  { t: 52.0, tween: { id: 'classshare0', to: 0.38, s: 1.4 }, hl: 'classshare0',
    cap: '<b>material shares</b> More scrub: the land decides where it goes, into the hollows first.' },
  { t: 54.0, grow: { from: 21, to: 12.4, s: 1.4, band: 0.9 }, cap: null },
  { t: 55.5, set: { grid: 24, classshare0: 0.2 }, grow: null, hl: 'grid' },
  { t: 55.6, tween: [{ id: 'fog', to: 1, s: 1.0 }, { id: 'seahaze', to: 0.5, s: 1.0 }] },
  { t: 56.0, tab: 'look', cap: '<b>look</b> Sky, haze, sun glow and a grade. None of it relights the captured ground.' },
  { t: 56.4, set: { sky: 'clear' } },
  { t: 56.7, tween: [{ id: 'sunglow', to: 0.5, s: 1.3 }, { id: 'grade', to: 0.4, s: 1.3 },
                     { id: 'saturation', to: 1.1, s: 1.3 }, { id: 'fog', to: 1.25, s: 1.3 }] },
  { t: 58.3, tween: [{ id: 'sunglow', to: 0, s: 0.5 }, { id: 'grade', to: 0, s: 0.5 },
                     { id: 'saturation', to: 1, s: 0.5 }, { id: 'fog', to: 1, s: 0.5 }], cap: null },
  { t: 58.8, tab: 'debug', set: { sky: 'overcast' } },
  { t: 59.1, set: { lodcolours: true }, hl: 'lodcolours',
    cap: '<b>levels of detail</b> Green near, then yellow, orange and red as detail drops; grey is the far field.' },
  { t: 61.0, set: { lodcolours: false, identity: true }, hl: 'identity',
    cap: '<b>tiles</b> A colour per tile: the same few tiles, never in a pattern.' },
  { t: 62.4, set: { identity: false }, scroll: [1, 0, 0.3], cap: null },
  { t: 62.7, card: 'end' },
];

export const END = 66;


// Camera path: sample step (s) and the eye's least height over the ground
// (tiles).
const PATH_DT = 0.05, CLEAR = 0.3;

/** Gaussian smoothing of a series, sigma in samples, ends held. */
export function smooth(v, sigma) {
  const r = Math.ceil(3 * sigma), w = [];
  for (let k = -r; k <= r; k++) w.push(Math.exp(-0.5 * (k / sigma) ** 2));
  return v.map((_, i) => {
    let a = 0, b = 0;
    for (let k = -r; k <= r; k++) {
      const j = Math.min(v.length - 1, Math.max(0, i + k));
      a += w[k + r] * v[j]; b += w[k + r];
    }
    return a / b;
  });
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const photoUrl = (n) => `./media/demo/${n}.jpg`;

/** The player. `api` is the viewer's side (viewer.js): camera, controls,
 *  the world's anchors and the second world's load. */
export function createFilm(api, { rec = false, slow = 1, fps = 0 } = {}) {
  let state = 'loading';          // loading, ready, playing, waiting, done, off
  let clock = 0, last = 0, shot = -1, next = 0;
  let anchors = null, path = null;
  const tweens = [];
  let scrollAnim = null;
  let growAnim = null;            // {from, to, t0, s, band, off}: the square rise

  // ------------------------------------------------------------ overlay
  // Every fade and move below runs on the film's clock, not the page's, so
  // a frame-by-frame render (fps > 0) sees the same film as a live one.
  const root = document.createElement('div');
  root.id = 'film';
  root.innerHTML = '<div class="card"></div><div class="photos"></div><div class="table"></div><div class="cap"></div>';
  document.body.appendChild(root);
  document.body.classList.add('film');
  const cardEl = root.querySelector('.card'), photosEl = root.querySelector('.photos'),
        capEl = root.querySelector('.cap'), tableEl = root.querySelector('.table');
  let water = null;
  fetch('./data/water.json').then((r) => (r.ok ? r.json() : null)).then((w) => { water = w; }).catch(() => {});
  for (const n of ['water', 'water_deep']) new Image().src = `./data/${n}.png`;
  for (const n of [...Object.values(PHOTOS).flatMap((p) => [...p.left.photos, ...p.right.photos]),
                   ...CARDS.dunes.photos]) {
    new Image().src = photoUrl(n);
  }
  const ease = (u) => u * u * (3 - 2 * u);
  const clamp01 = (u) => Math.min(1, Math.max(0, u));
  /** A value easing to its target over `dur` seconds of film. */
  function fader(dur, v = 0) {
    let v0 = v, to = v, t0 = 0;
    const at = () => v0 + (to - v0) * ease(clamp01((clock - t0) / dur));
    return { at, set(x) { v0 = at(); to = x; t0 = clock; }, jump(x) { v0 = to = x; } };
  }
  const card = fader(0.7, 1), photosIn = fader(1.0), gather = fader(0.9), cap = fader(0.5),
        table = fader(0.6);
  let tableT0 = 0, tableWanted = null;
  const lit = [];                 // [{el, t0}]: controls lit up as they change

  function showCard(name, hint = '') {
    const c = name && CARDS[name];
    if (!c) { card.set(0); return; }
    cardEl.innerHTML = (c.small && name !== 'end' ? `<div class="small">${esc(c.small)}</div>` : '')
      + `<div class="big">${esc(c.big)}</div>`
      + (c.line ? `<div class="line">${esc(c.line)}</div>` : '')
      + (c.photos ? `<div class="pics">${c.photos.map((p) => `<img src="${photoUrl(p)}" alt="">`).join('')}</div>` : '')
      + (c.small && name === 'end' ? `<div class="small">${esc(c.small)}</div>` : '')
      + (c.credit ? `<div class="credit">${esc(c.credit)}</div>` : '')
      + `<div class="hint">${esc(hint)}</div>`;
    card.set(1);
  }
  function setHint(text) {
    const h = cardEl.querySelector('.hint');
    if (h) h.textContent = text;
  }
  function showPhotos(name) {
    const p = name && PHOTOS[name];
    if (!p) { photosIn.jump(0); gather.jump(0); return; }
    const side = (s, cls) => `<div class="side ${cls}"><div class="label">${esc(s.label)}</div>${
      s.photos.map((n) => `<img src="${photoUrl(n)}" alt="">`).join('')}</div>`;
    photosEl.innerHTML = side(p.left, 'l') + side(p.right, 'r')
      + `<div class="credit">${esc(p.credit)}</div>`;
    gather.jump(0); photosIn.jump(0); photosIn.set(1);
  }
  // ------------------------------------------------------------ tables
  const css = (c) => `rgb(${c.map((v) => Math.round(255 * Math.min(1, Math.max(0, v)))).join(',')})`;
  const ramp = (stops, v) => {
    const u = Math.min(1, Math.max(0, v)) * (stops.length - 1);
    const k = Math.min(stops.length - 2, Math.floor(u)), f = u - k;
    return stops[k].map((a, q) => a + (stops[k + 1][q] - a) * f);
  };
  const fmt = (v) => String(+v.toFixed(2));
  /** A canvas of CSS size w x h drawn at the screen's pixel density. */
  function canvas(w, h) {
    const c = document.createElement('canvas'), d = Math.min(devicePixelRatio || 1, 2);
    c.width = Math.round(w * d); c.height = Math.round(h * d);
    c.style.width = `${w}px`; c.style.height = `${h}px`;
    const g = c.getContext('2d');
    g.scale(d, d);
    return [c, g];
  }
  /** One part of a table, faded in after the one before it. */
  function part(html, node = null) {
    const el = document.createElement('div');
    el.className = 'it';
    el.innerHTML = html;
    if (node) el.querySelector('.c').appendChild(node);
    return el;
  }
  /** Each material's tiles from the atlas (north up), edges in their Wang
   *  code colours as the viewer's edge view paints them. */
  function tilesTable() {
    const tt = api.tileTable();
    if (!tt || !tt.img) return null;
    const per = 9, S = 50, G = 6, b = 5, R = tt.res;
    // Atlas pixels, and each tile with a centre of its material in it, as
    // the viewer joins them: the centre a rounded square (p-norm ball) of
    // the frame radius, faded over the ring (figures.py _centre_mask).
    const src = document.createElement('canvas');
    src.width = tt.img.width; src.height = tt.img.height;
    const sg = src.getContext('2d', { willReadFrequently: true });
    sg.drawImage(tt.img, 0, 0);
    const entry = (k) => sg.getImageData((k % tt.cols) * R, Math.floor(k / tt.cols) * R, R, R);
    const fr = tt.frames, mask = new Float32Array(R * R);
    if (fr) {
      const p = fr.shape || 2, ring = Math.max(fr.ring || 0.15, 1e-6);
      for (let y = 0; y < R; y++) {
        for (let x = 0; x < R; x++) {
          const u = Math.abs(((x + 0.5) / R - 0.5) * tt.size), v = Math.abs(((y + 0.5) / R - 0.5) * tt.size);
          const d = (u ** p + v ** p) ** (1 / p);
          mask[y * R + x] = Math.min(1, Math.max(0, (fr.radius - d) / ring + 0.5));
        }
      }
    }
    const joined = (i, k, q) => {
      const im = entry(i), pool = tt.centres.map((c, j) => [c, j]).filter(([c]) => c.cls === k && c.pool);
      if (fr && pool.length) {
        const ce = entry(tt.tiles.length + pool[q % pool.length][1]).data;
        for (let o = 0; o < R * R; o++) {
          for (let c = 0; c < 3; c++) im.data[4 * o + c] += (ce[4 * o + c] - im.data[4 * o + c]) * mask[o];
        }
      }
      const t = document.createElement('canvas');
      t.width = t.height = R;
      t.getContext('2d').putImageData(im, 0, 0);
      return t;
    };
    const head = `<b>tiles</b>${tt.tiles.length} tiles of ${fmt(tt.size)} m, cut from the capture. `
      + 'Edge colours are their codes: where two tiles meet, both have the same colour.';
    const parts = tt.names.map((name, k) => {
      const own = tt.tiles.map((t, i) => [t, i]).filter(([t]) => t.cls === k);
      const nc = tt.centres.filter((c) => c.cls === k && c.pool).length;
      const [c, g] = canvas(per * (S + G) - G, S);
      own.slice(0, per).forEach(([t, i], q) => {
        const x = q * (S + G);
        g.drawImage(joined(i, k, q), 0, 0, R, R, x, 0, S, S);
        const [n, e, so, w] = t.codes;
        g.fillStyle = css(tt.edgeRGB.h[n % 4]); g.fillRect(x, 0, S, b);
        g.fillStyle = css(tt.edgeRGB.h[so % 4]); g.fillRect(x, S - b, S, b);
        g.fillStyle = css(tt.edgeRGB.v[e % 4]); g.fillRect(x + S - b, 0, b, S);
        g.fillStyle = css(tt.edgeRGB.v[w % 4]); g.fillRect(x, 0, b, S);
      });
      return part(`<div class="lbl">${esc(name)} · ${own.length} tiles`
        + (nc ? `, ${nc} centre piece${nc > 1 ? 's' : ''}` : '') + '</div><div class="c"></div>', c);
    });
    return { head, parts, rows: true };
  }
  /** The maps the rule read for this world, one value per tile, then its
   *  decision in the colours of the viewer's decision view. */
  function ruleTable() {
    const r = api.ruleMaps();
    if (!r) return null;
    const n = r.n, S = 84, m = r.maps;
    const keys = ['height', 'slope', 'tpi', m.sediment ? 'sediment' : 'flow'].concat(m.near ? ['near'] : []);
    const draw = (val) => {
      const [c, g] = canvas(S, S);
      const tmp = document.createElement('canvas');
      tmp.width = tmp.height = n;
      const tg = tmp.getContext('2d'), im = tg.createImageData(n, n);
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
          const rgb = val(j * n + i), o = 4 * ((n - 1 - j) * n + i);
          im.data[o] = 255 * rgb[0]; im.data[o + 1] = 255 * rgb[1];
          im.data[o + 2] = 255 * rgb[2]; im.data[o + 3] = 255;
        }
      }
      tg.putImageData(im, 0, 0);
      g.imageSmoothingEnabled = false;
      g.drawImage(tmp, 0, 0, S, S);
      return c;
    };
    const parts = keys.map((k) => part(`<div class="c"></div><div class="lbl">${MAPS[k][0]}</div>`,
                                       draw((i) => ramp(RAMPS[MAPS[k][1]], m[k][i]))));
    parts.push(part('<div class="arrow">→</div>'));
    const grey = 0.62;
    const dec = draw((i) => r.colours[i].map((v) => v * r.strength[i] + grey * (1 - r.strength[i])));
    const classes = [...new Set(r.cls)].sort();
    const legend = classes.map((k) => `<span><i style="background:${css(r.colours[r.cls.indexOf(k)])}"></i>`
      + `${esc(r.names[r.classOf(k)])}</span>`).join('');
    parts.push(part(`<div class="c"></div><div class="lbl">decision<div class="legend">${legend}</div></div>`, dec));
    const head = '<b>the rule</b>Maps of this world, one value per tile. Each material scores every tile '
      + 'from them and the highest score wins; grey is a near tie.';
    return { head, parts };
  }
  /** The two water tiles and the bed's visible share against distance
   *  from the waterline, as water.json measured them. */
  function waterTable() {
    if (!water || !water.layers) return null;
    const L = water.layers, W = 250, H = 120, pad = [22, 8, 8, 22];
    const [c, g] = canvas(W, H);
    const X = (d) => pad[0] + (W - pad[0] - pad[1]) * d;
    const Y = (v) => H - pad[3] - (H - pad[2] - pad[3]) * v;
    g.strokeStyle = 'rgba(216, 210, 198, .35)'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(X(0), Y(1)); g.lineTo(X(0), Y(0)); g.lineTo(X(1), Y(0)); g.stroke();
    g.fillStyle = 'rgba(150, 180, 176, .75)';
    for (const bd of L.bands) {
      g.fillRect(X(bd.from_m) + 1, Y(bd.t), X(bd.to_m) - X(bd.from_m) - 2, Y(0) - Y(bd.t));
    }
    g.strokeStyle = '#d9a441'; g.lineWidth = 2; g.beginPath();
    for (let k = 0; k <= 50; k++) {
      const d = k / 50, v = L.t0 * Math.exp(-d / L.length_m);
      if (k) g.lineTo(X(d), Y(v)); else g.moveTo(X(d), Y(v));
    }
    g.stroke();
    g.fillStyle = '#b9b3a8'; g.font = '11px system-ui, sans-serif';
    g.fillText('1', X(0) - 12, Y(1) + 4); g.fillText('0', X(0) - 12, Y(0) + 4);
    g.fillText('0 m', X(0) - 4, H - 8); g.fillText('1 m', X(1) - 18, H - 8);
    g.fillStyle = '#d9a441';
    g.fillText(`T = ${fmt(L.t0)} exp(−d / ${fmt(L.length_m)} m)`, X(0.3), Y(0.97));
    const tile = (src, lbl) => part(`<img src="${src}" alt=""><div class="lbl">${lbl}</div>`);
    const head = `<b>water</b>Measured in ${L.photos.length} drone photos of the river: its shallow and deep `
      + 'colour, and how much of the bed shows through with distance from the waterline.';
    return { head, parts: [tile('./data/water.png', `shallow, ${fmt(water.metres)} m across`),
                           tile('./data/water_deep.png', `deep, ${fmt(water.deep.metres)} m across`),
                           part('<div class="c"></div><div class="lbl">share of the bed seen</div>', c)] };
  }
  /** Build a table; false while its data is not there yet. */
  function buildTable(name) {
    const got = name === 'tiles' ? tilesTable() : name === 'rule' ? ruleTable() : waterTable();
    if (!got) return false;
    tableEl.innerHTML = `<div class="t">${got.head}</div>`;
    const box = document.createElement('div');
    box.className = got.rows ? 'rows' : 'maps';
    got.parts.forEach((el, i) => { el.dataset.i = i; box.appendChild(el); });
    tableEl.appendChild(box);
    tableT0 = clock; table.jump(0); table.set(1);
    return true;
  }
  function showTable(name) {
    tableWanted = null;
    if (!name) { table.set(0); return; }
    if (!buildTable(name)) tableWanted = name;
  }

  function showCap(html) {
    if (!html) { cap.set(0); return; }
    capEl.innerHTML = `<div>${html}</div>`;
    cap.jump(0); cap.set(1);
  }
  /** Paint the overlay for the clock as it is now. */
  function paint() {
    const c = card.at();
    cardEl.style.opacity = c.toFixed(3);
    cardEl.style.visibility = c > 0.001 ? 'visible' : 'hidden';
    const p = photosIn.at(), g = gather.at();
    for (const s of photosEl.querySelectorAll('.side')) {
      const dir = s.classList.contains('l') ? -1 : 1;
      // In from the screen's edge, then shrinking into the tile in the middle.
      const x = dir * (30 * (1 - p) - 18 * g);
      s.style.transform = `translate(${x.toFixed(2)}vw, -50%) scale(${(1 - 0.85 * g).toFixed(3)})`;
      s.style.opacity = (p * (1 - g)).toFixed(3);
    }
    const cr = photosEl.querySelector('.credit');
    if (cr) cr.style.opacity = (0.8 * p * (1 - g)).toFixed(3);
    if (tableWanted && buildTable(tableWanted)) tableWanted = null;
    const tb = table.at();
    tableEl.style.opacity = tb.toFixed(3);
    tableEl.style.visibility = tb > 0.001 ? 'visible' : 'hidden';
    tableEl.style.transform = `translateY(${(-10 * (1 - tb)).toFixed(1)}px)`;
    // Its parts one after another, as a figure is read.
    for (const el of tableEl.querySelectorAll('.it')) {
      el.style.opacity = clamp01((clock - tableT0 - 0.25 - 0.28 * el.dataset.i) / 0.4).toFixed(3);
    }
    const k = cap.at();
    capEl.style.opacity = k.toFixed(3);
    capEl.style.transform = `translate(-50%, ${(12 * (1 - k)).toFixed(1)}px)`;
    for (let i = lit.length - 1; i >= 0; i--) {
      const u = (clock - lit[i].t0) / 1.6;
      const a = u < 0.6 ? 1 : Math.max(0, 1 - (u - 0.6) / 0.4);
      lit[i].el.style.boxShadow = a > 0
        ? `0 0 0 2px rgba(217, 164, 65, ${a.toFixed(2)}), 0 0 18px rgba(217, 164, 65, ${(0.45 * a).toFixed(2)})` : '';
      if (u >= 1) lit.splice(i, 1);
    }
  }

  // ------------------------------------------------------------ actions
  function light(el) {
    if (!el) return;
    el.style.borderRadius = '3px';
    lit.push({ el, t0: clock });
  }
  function rowOf(id) {
    const el = document.getElementById(id);
    return el && (el.closest('.s, label.c') || el);
  }
  function run(e) {
    if ('card' in e) showCard(e.card);
    if ('photos' in e) showPhotos(e.photos);
    if (e.gather) gather.set(1);
    if ('cap' in e) showCap(e.cap);
    if ('table' in e) showTable(e.table);
    if (e.set) for (const [k, v] of Object.entries(e.set)) api.setControl(k, v);
    for (const w of e.tween ? [].concat(e.tween) : []) {
      const el = document.getElementById(w.id);
      if (el) tweens.push({ id: w.id, from: +el.value, to: w.to, t0: clock, s: w.s });
      else console.warn(`demo: no control ${w.id}`);
    }
    if (e.hold) api.holdReliefScale();
    if ('grow' in e) growAnim = e.grow ? { ...e.grow, t0: clock } : null;
    if (e.hl) light(rowOf(e.hl));
    if (e.panel) { api.setFigure(false); api.setResearch(true); }
    if (e.tab) light(api.openTab(e.tab));
    if (e.scroll) scrollAnim = { from: e.scroll[0], to: e.scroll[1], t0: clock, s: e.scroll[2] };
    if (e.scrollTo) {
      // The control a third of the way down the panel, as fractions.
      const p = api.panelEl(), row = rowOf(e.scrollTo.id);
      if (p && row) {
        const room = Math.max(1, p.scrollHeight - p.clientHeight);
        const top = row.getBoundingClientRect().top - p.getBoundingClientRect().top + p.scrollTop;
        const want = Math.min(1, Math.max(0, (top - 0.33 * p.clientHeight) / room));
        scrollAnim = { from: p.scrollTop / room, to: want, t0: clock, s: e.scrollTo.s };
      }
    }
  }
  function animate() {
    for (let i = tweens.length - 1; i >= 0; i--) {
      const w = tweens[i], u = Math.min(1, (clock - w.t0) / w.s);
      api.setControl(w.id, w.from + (w.to - w.from) * ease(u));
      if (u >= 1) tweens.splice(i, 1);
    }
    if (scrollAnim) {
      const p = api.panelEl(), a = scrollAnim;
      const u = a.s > 0 ? Math.min(1, (clock - a.t0) / a.s) : 1;
      if (p) p.scrollTop = (a.from + (a.to - a.from) * ease(u)) * (p.scrollHeight - p.clientHeight);
      if (u >= 1) scrollAnim = null;
    }
  }

  // ------------------------------------------------------------ worlds
  /** Set a shot's world up: its preset and settings, then the anchors the
   *  camera keys name, once the rebuild has settled. */
  async function setUp(k) {
    const s = SHOTS[k], t0 = performance.now();
    if (api.scene() !== s.world) await api.switchScene(s.world);
    const t1 = performance.now();
    api.applyPreset(s.preset);
    for (const [id, v] of Object.entries(s.set || {})) api.setControl(id, v);
    // A render has the time a live film does not: full resolution throughout.
    if (fps) api.setControl('autoquality', false);
    await api.settled();
    console.log(`demo: shot ${k} (${s.world}) ready in ${((performance.now() - t0) / 1000).toFixed(1)} s, `
      + `${((performance.now() - t1) / 1000).toFixed(1)} s of it settling`);
    const T = api.tileSize();
    const tile = s.rise ? api.captureCell() : [0, 0];
    const water = api.waterSpot() || tile;
    const water2 = api.waterAlong(water[0], water[1], 6 * T) || water;
    const at = { tile, water, water2, centre: [0, 0], mid: [(tile[0] + water[0]) / 2, (tile[1] + water[1]) / 2] };
    // [x, y] in tiles from the centre.
    for (const q of s.cam) if (Array.isArray(q.at)) at[q.at.join(',')] = [q.at[0] * T, q.at[1] * T];
    anchors = { T, at };
    path = cameraPath(s, at, T);
    shot = k;
    pose(s.from);
  }
  /** The shot's camera sampled every PATH_DT seconds: the keys, then the
   *  orbit point's height from the ground under it, smoothed, and the
   *  elevation raised where the eye would come closer to the ground than
   *  CLEAR tiles, also smoothed. Set from the ground on every frame, the
   *  camera followed each dune and jiggled. */
  function cameraPath(s, at, T) {
    const ks = s.cam, ts = ks.map((q) => q.t);
    const A = (q) => at[Array.isArray(q.at) ? q.at.join(',') : q.at];
    const n = Math.ceil((s.to - s.from) / PATH_DT) + 1;
    const P = { x: [], y: [], z: [], az: [], el: [], d: [] };
    for (let i = 0; i < n; i++) {
      const t = s.from + i * PATH_DT, ch = (f) => monotone(ts, ks.map(f), t);
      P.x.push(ch((q) => A(q)[0])); P.y.push(ch((q) => A(q)[1]));
      P.az.push(ch((q) => q.az)); P.el.push(ch((q) => q.el));
      P.d.push(Math.exp(ch((q) => Math.log(q.d))) * T);
    }
    P.z = smooth(P.x.map((x, i) => api.surfaceAt(x, P.y[i])), 0.5 / PATH_DT);
    const lift = P.x.map((x, i) => {
      for (let el = P.el[i]; el < 85; el += 0.25) {
        const a = P.az[i] * Math.PI / 180, e = el * Math.PI / 180, d = P.d[i];
        const ex = x + d * Math.cos(e) * Math.cos(a), ey = P.y[i] + d * Math.cos(e) * Math.sin(a);
        if (P.z[i] + d * Math.sin(e) >= api.surfaceAt(ex, ey) + CLEAR * T) return el - P.el[i];
      }
      return 85 - P.el[i];
    });
    const r = Math.round(1.2 / PATH_DT);
    const held = lift.map((_, i) => Math.max(...lift.slice(Math.max(0, i - r), i + r + 1)));
    P.el = smooth(held, 0.4 / PATH_DT).map((v, i) => P.el[i] + v);
    return P;
  }
  function pose(t) {
    const s = SHOTS[shot], P = path, T = anchors.T;
    const u = Math.min(P.x.length - 1, Math.max(0, (t - s.from) / PATH_DT));
    const i = Math.min(P.x.length - 2, Math.floor(u)), f = u - i;
    const at = (a) => (a.length > 1 ? a[i] + (a[i + 1] - a[i]) * f : a[0]);
    api.setCamera([at(P.x), at(P.y), at(P.z)], at(P.az), at(P.el), at(P.d));
    const r = s.rise;
    if (r && t <= r[r.length - 1].t) {
      const rt = r.map((q) => q.t), c = anchors.at.tile;
      api.setRise(c[0], c[1], monotone(rt, r.map((q) => q.r), t) * T,
                  monotone(rt, r.map((q) => q.band), t) * T, 1.2 * T);
    } else if (growAnim) {
      const g = growAnim, u = g.s > 0 ? clamp01((clock - g.t0) / g.s) : 1;
      api.setRise(0, 0, (g.from + (g.to - g.from) * ease(u)) * T, g.band * T, 1.2 * T, true);
      if (u >= 1 && g.off) { growAnim = null; api.setRise(0, 0, 0, 0, 0); }
    } else api.setRise(0, 0, 0, 0, 0);
  }

  // ------------------------------------------------------------ control
  async function prepare() {
    showCard('title', 'loading…');
    card.jump(1);
    paint();
    api.prefetch(SHOTS.map((s) => s.world));
    await api.ready();
    api.setFigure(true);
    await setUp(0);
    card.jump(1);
    state = 'ready';
    if (rec) setHint('Space starts · F11 full screen · Esc stops');
    else if (!fps) start();
  }
  function start() {
    if (state !== 'ready') return false;
    setHint('');
    // No title card: the film opens on the capture itself.
    card.jump(0);
    state = 'playing';
    last = performance.now();
    clock = 0; next = 0;
    return true;
  }
  function stop() {
    if (state === 'off') return;
    state = 'off';
    api.setRise(0, 0, 0, 0, 0);
    api.setFlowClock(null);
    root.remove();
    document.body.classList.remove('film');
    api.setFigure(false);
  }
  /** Move the film on by dt seconds: actions, settings, camera. */
  function advance(dt) {
    clock += dt;
    while (next < EVENTS.length && EVENTS[next].t <= clock) run(EVENTS[next++]);
    animate();
    const k = SHOTS.findIndex((s) => clock >= s.from && clock < s.to);
    if (k > shot) {
      state = 'waiting';
      setUp(k).then(() => { state = 'playing'; last = performance.now(); });
      return;
    }
    if (shot >= 0) pose(Math.min(clock, SHOTS[shot].to));
    if (fps) api.setFlowClock(clock);
    if (clock >= END) {
      state = 'done';
      // Shown to a visitor, the viewer is theirs once the film is over.
      if (!rec && !fps) setTimeout(() => { card.set(0); setTimeout(stop, 900); }, 1500);
    }
  }
  /** Called every animation frame by the viewer. */
  function step(now) {
    if (state === 'off') return;
    if (state === 'playing' && !fps) {
      const dt = Math.min(0.05, Math.max(0, (now - last) / 1000)) / slow;
      last = now;
      advance(dt);
    } else if (state === 'done' && !rec && !fps) {
      clock += Math.min(0.05, Math.max(0, (now - last) / 1000)); last = now;
    }
    paint();
  }
  /** One frame of a render (fps > 0): false once the film is over. While
   *  the next world loads it does nothing and says so with 'waiting'. */
  function tick() {
    if (state === 'ready') start();
    if (state === 'waiting') return 'waiting';
    if (state !== 'playing') return false;
    advance(1 / fps);
    paint();
    return state !== 'done';
  }

  addEventListener('keydown', (e) => {
    if (state === 'off') return;
    if (e.key === 'Escape') { stop(); e.preventDefault(); e.stopImmediatePropagation(); return; }
    if (e.key === 'F11') return;
    if (e.key === ' ' && state === 'ready' && !fps) start();
    // The film has the camera: no other key reaches the viewer while it plays.
    e.preventDefault(); e.stopImmediatePropagation();
  }, { capture: true });

  return { prepare, start, stop, step, tick,
           state: () => state, clock: () => clock,
           playing: () => state === 'playing' || state === 'waiting',
           active: () => state !== 'off' };
}
