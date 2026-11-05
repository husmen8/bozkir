// WebGL2 Gaussian splat renderer with Wang tiling.
//
// Projection follows bozkir/camera.py (camera space +z forward, x right,
// y down; 2D covariance from the perspective Jacobian, 3DGS Eq. 5), so the
// slow Python renderer can be ground truth when this one looks wrong.

import { drawOrder } from './order.js';
import { mergeGroups, MAX_GROUP } from './merge.js';
import { openDrop, describe } from './tileset.js';
import { Capture } from './capture.js';
import { generatedField, listCached, loadHeightField, openness, roughnessMask } from './heightfield.js';
import { Benchmark, report } from './benchmark.js';
import { classify, sampleGrid, wanderField, riverChannel, drainBase, NO_WATER, RIVER } from './landform.js';
import { GROUPS as TERRAIN_GROUPS } from './terrain.js';
import { starterTileset } from './starter.js';
import { decalVariants, placeDecals } from './decals.js';
import { groundLevel, pickCentre, pickTurn, poolsByClass, turnCosSin, cellHash,
         ringProfile, toneTerms, partnerClass, spotDraws, SPOT_WARP, spotSharpness,
         gradedPool } from './frames.js';
import { initUI, store, loadProgress, loadDone, loadFailed, updateCard, setChapters,
         endChapters, takeView } from './ui.js';
import { SPLAT_VERT, SPLAT_FRAG, FAR_VERT, FAR_FRAG, LINE_VERT, LINE_FRAG,
         SKY_VERT, SKY_FRAG } from './shaders.js';
import { glTools } from './gl.js';
import { unpack, meanClassColours } from './splat.js';
import { Orbit } from './camera.js';
import { createFilm } from './film.js';

const BUILD = 'bozkir viewer 1.0';
console.log('%c' + BUILD, 'color:#c8a05a');


// =============================================================== helpers

const $ = (id) => document.getElementById(id);
const canvas = $('gl');
// Controls with a fixed id; ids made at run time go through $().
const ui = {};
for (const id of ['n', 'drawn', 'fps', 'sortms', 'azim', 'elev', 'dist', 'riverrule',
                  'cmd', 'gz', 'grid', 'gridn', 'used', 'usedn', 'wangnote',
                  'relief', 'reliefn', 'reliefscale', 'reliefscalen', 'bandn',
                  'subdiv', 'subdivn', 'seam', 'viewsn', 'sorterr', 'pop',
                  'gridanglen', 'lodbasen', 'lodinfo', 'fogn', 'freefly',
                  'flynote', 'shaden', 'exposuren', 'figuremode', 'figexit',
                  'scenepick', 'heightpick', 'expert', 'busy', 'subdivrow',
                  'gpuname', 'detailamtn', 'genseed', 'gensize', 'genbtn',
                  'genstatus', 'terrainwin', 'gencards', 'genrecent',
                  'presetchips', 'saturationn', 'macron', 'sunglown', 'sunazn', 'mistn', 'graden', 'rangesn', 'seahazen', 'resn', 'fovn',
                  'gainn', 'viewWalk', 'speedn', 'tileorder', 'mergethrn',
                  'mergestat', 'capture', 'capframes', 'capframesn',
                  'capcentre', 'caparc', 'caparcn', 'capstepn', 'fieldnote',
                  'bench', 'benchout', 'classsharpn', 'classnote',
                  'classblend', 'blendwidthn', 'driftn', 'classbalance', 'classbalancen',
                  'decalspacingn', 'decalnote', 'classcoherencen',
                  'classaltituden', 'classwander', 'classwandern',
                  'classshares', 'classbalancelbl', 'grip', 'overlay', 'riverwidthn']) {
  ui[id] = $(id);
}

// --- panel size ---------------------------------------------------------
// Width by dragging the panel's left edge, text size by the header buttons.
// Both only set a CSS variable; resize() runs every frame anyway.
const PANEL_MIN = 180, PANEL_MAX = 560;
const FS_MIN = 9, FS_MAX = 17;

function setPanelWidth(px) {
  const w = Math.round(Math.min(PANEL_MAX, Math.max(PANEL_MIN, px)));
  document.documentElement.style.setProperty('--panel-w', w + 'px');
}

function setFontSize(px) {
  const v = Math.min(FS_MAX, Math.max(FS_MIN, px));
  document.documentElement.style.setProperty('--fs', v + 'px');
  return v;
}

{
  // Text size is remembered, since it is set for the reader's screen.
  let fs = 12;
  fs = +store.get('bozkir.fs') || 12;
  fs = setFontSize(fs);
  const keep = () => store.set('bozkir.fs', String(fs));
  on('fsup', 'click', () => { fs = setFontSize(fs + 1); keep(); });
  on('fsdown', 'click', () => { fs = setFontSize(fs - 1); keep(); });

  const grip = ui.grip;
  if (grip) {
    grip.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      // Capture, so a fast drag keeps sending events here.
      grip.setPointerCapture(e.pointerId);
      document.body.classList.add('sizing');
    });
    grip.addEventListener('pointermove', (e) => {
      if (!grip.hasPointerCapture(e.pointerId)) return;
      setPanelWidth(window.innerWidth - e.clientX);
    });
    const stop = (e) => {
      if (grip.hasPointerCapture(e.pointerId)) {
        grip.releasePointerCapture(e.pointerId);
      }
      document.body.classList.remove('sizing');
    };
    grip.addEventListener('pointerup', stop);
    grip.addEventListener('pointercancel', stop);
    // Double-click resets the width.
    grip.addEventListener('dblclick', () => setPanelWidth(288));
  }
}

// Each slider paints its filled part from a CSS variable, set here for all
// sliders at once.
function paintSliders() {
  for (const el of document.querySelectorAll('.s input[type=range]')) {
    const lo = +el.min, hi = +el.max;
    const f = hi > lo ? ((+el.value - lo) / (hi - lo)) * 100 : 0;
    el.parentElement.style.setProperty('--f', f.toFixed(2));
  }
}
document.addEventListener('input', (e) => {
  if (e.target.type === 'range') paintSliders();
});

/** Attach a handler, or warn and carry on: a control missing from
 *  index.html should cost that control, not the whole viewer. */
function on(id, event, fn) {
  const el = $(id);
  if (!el) {
    console.warn(`no #${id} in the page; that control is inactive`);
    return null;
  }
  el.addEventListener(event, fn);
  return el;
}

function fail(err) {
  console.error(err);
  loadFailed('The renderer failed to start.', String(err && err.message || err));
  throw err;
}

// Ask for the fast GPU: laptops often give the browser the integrated chip.
const gl = canvas.getContext('webgl2',
  { antialias: false, alpha: false, premultipliedAlpha: false,
    powerPreference: 'high-performance' });
if (!gl) {
  loadFailed('This browser has no WebGL2, which the renderer needs.',
             'A current Chrome, Edge, Firefox or Safari has it; check that hardware acceleration is on.');
  throw new Error('webgl2 unavailable');
}

const { program, uniforms, makeTexture } = glTools(gl);

// Which GPU is drawing, shown in the console and the scene section.
let gpuName = 'unknown';
if (gl) {
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  gpuName = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)
                : gl.getParameter(gl.RENDERER);
  console.log(`GPU: ${gpuName}`);
}




const splatTex = [null, null];



// ================================================================= setup

let splatProg, lineProg, skyProg, farProg;
try {
  splatProg = program(SPLAT_VERT, SPLAT_FRAG);
  farProg = program(FAR_VERT, FAR_FRAG);
  lineProg = program(LINE_VERT, LINE_FRAG);
  skyProg = program(SKY_VERT, SKY_FRAG);
} catch (e) { fail(e); }

const splatU = uniforms(splatProg,
  ['view', 'eye', 'cellXY', 'relief', 'wave', 'subdiv',
   'focal', 'viewport', 'gain', 'near',
   'data', 'colour', 'tileSize', 'edgeMark', 'fade', 'decal',
   'tint', 'tintAmount', 'fogColour', 'fogDensity', 'gridRot', 'mix',
   'mixCorners', 'mixNoise', 'spots', 'mixField', 'mixWarp', 'spotSharp',
   'field', 'fieldSize', 'fieldExtent', 'hasField',
   'open', 'hasOpen', 'shade', 'exposure', 'saturation', 'waterView',
   'farOn', 'farStart', 'farBand', 'detail', 'detailFreq', 'macro',
   'edgeN', 'edgeE', 'edgeS', 'edgeW',
   'slotTurn', 'slotRole', 'slotZ0', 'slotTone', 'ring', 'prom',
   'slotDrift', 'drift', 'water', 'waterTex', 'waterMetres', 'waterMean', 'flow', 'time',
   'waterDeep', 'deepMetres', 'deepMean', 'layers', 'river', 'hasRiver',
   'rise', 'riseDrop', 'riseSquare', 'sun', 'sunRGB', 'mist', 'marine', 'grade']);
const farU = uniforms(farProg,
  ['view', 'eye', 'focal', 'viewport', 'relief', 'wave', 'hasField', 'field',
   'fieldSize', 'fieldExtent', 'atlas', 'cells', 'atlasGrid', 'tileSize',
   'gridN', 'rot', 'farStart', 'farBand', 'fogColour', 'fogDensity',
   'exposure', 'shade', 'saturation', 'open', 'hasOpen', 'debug', 'classRGB',
   'tileInfo', 'hasTileInfo', 'detail', 'detailFreq', 'macro', 'blend',
   'spots', 'mixWarp', 'centres', 'centreTone', 'hasCentres', 'ringFar',
   'water', 'waterTex', 'waterMetres', 'waterMean', 'flow', 'time',
   'waterDeep', 'deepMetres', 'deepMean', 'layers', 'river', 'hasRiver', 'rise',
   'sun', 'sunRGB', 'mist', 'marine', 'grade', 'horizon', 'riseSquare']);
// The demo's growing world (film.js): centre x, y, radius, band in world
// units; band 0 is off. riseDrop is the height a cell falls from.
let riseCircle = new Float32Array(4), riseDrop = 0, riseSquare = false;
/** How far a cell is from the rise's centre: round, or square on the grid. */
function riseDist(x, y) {
  const dx = x - riseCircle[0], dy = y - riseCircle[1];
  if (!riseSquare) return Math.hypot(dx, dy);
  const a = gridAngle * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
  return Math.max(Math.abs(c * dx + s * dy), Math.abs(-s * dx + c * dy));
}
let flowClock = null;     // seconds; null: the water flows on the page's clock
let drawCount = 0;        // frames drawn since the page opened
const lineU = uniforms(lineProg,
  ['view', 'eye', 'focal', 'viewport', 'near', 'alpha']);

// One VAO per program, or divisors leak between passes (the splat pass's
// divisor on attribute 1 broke the line colours).
const splatVAO = gl.createVertexArray();
const lineVAO = gl.createVertexArray();
const skyVAO = gl.createVertexArray();

const quadBuf = gl.createBuffer();
const indexBuf = gl.createBuffer();
// Whole-scene orders for RING headings around the vertical (one per 45
// degrees, at the view's elevation), sorted when idle. A fast turn ends before
// any sort can arrive, and an order more than 90 degrees off draws the ground's
// underside on top; the nearest of these is at most 22.5 degrees off, which
// does not show (an order 24 degrees off changed 0.1% of pixels).
const RING = 8;
const ringBufs = Array.from({ length: RING }, () => gl.createBuffer());
const ringFwd = new Array(RING).fill(null), ringParts = new Array(RING).fill(0);
let ringSize = 0, ringJob = -1, ringJobFwd = null, ringSeq = 0;
let liveFwd = null, liveFwdPending = null;      // the heading the live order was sorted for
let orderBuf = null;                             // the live-path order bound this frame
const cacheIndexBuf = gl.createBuffer();
gl.bindVertexArray(splatVAO);
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
gl.bufferData(gl.ARRAY_BUFFER,
  new Float32Array([-2, -2, 2, -2, -2, 2, 2, 2]), gl.STATIC_DRAW);
const aCorner = gl.getAttribLocation(splatProg, 'aCorner');
gl.enableVertexAttribArray(aCorner);
gl.vertexAttribPointer(aCorner, 2, gl.FLOAT, false, 0, 0);
const aIndex = gl.getAttribLocation(splatProg, 'aIndex');
gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
gl.enableVertexAttribArray(aIndex);
gl.vertexAttribIPointer(aIndex, 1, gl.UNSIGNED_INT, 0, 0);
gl.vertexAttribDivisor(aIndex, 1);

const linePosBuf = gl.createBuffer();
const lineRGBBuf = gl.createBuffer();
gl.bindVertexArray(lineVAO);
const aPos = gl.getAttribLocation(lineProg, 'aPos');
gl.bindBuffer(gl.ARRAY_BUFFER, linePosBuf);
gl.enableVertexAttribArray(aPos);
gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
const aRGB = gl.getAttribLocation(lineProg, 'aRGB');
gl.bindBuffer(gl.ARRAY_BUFFER, lineRGBBuf);
gl.enableVertexAttribArray(aRGB);
gl.vertexAttribPointer(aRGB, 3, gl.FLOAT, false, 0, 0);
const skyU = uniforms(skyProg,
  ['view', 'focal', 'viewport', 'skyTop', 'skyHorizon', 'ground',
   'eye', 'sun', 'sunRGB', 'mist', 'marine', 'grade']);
const skyBuf = gl.createBuffer();
gl.bindVertexArray(skyVAO);
gl.bindBuffer(gl.ARRAY_BUFFER, skyBuf);
gl.bufferData(gl.ARRAY_BUFFER,
  new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
const aNDC = gl.getAttribLocation(skyProg, 'aNDC');
gl.enableVertexAttribArray(aNDC);
gl.vertexAttribPointer(aNDC, 2, gl.FLOAT, false, 0, 0);
gl.bindVertexArray(null);

gl.disable(gl.DEPTH_TEST);
gl.enable(gl.BLEND);
gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA,
                     gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
gl.clearColor(0, 0, 0, 1);

// ================================================================= state

const cam = new Orbit();
let splatCount = 0;
let patches = [{ start: 0, count: 0 }];
let identityOrder = null;
let wangCodes = null;
// Decal copies (scripts/decals.py): the plain tile each copy was made from,
// -1 for plain tiles. Copies are never laid directly, only swapped in.
let variantOf = null;
let waterView = false;     // tint where the river's water lies, a preview before water is drawn
// Rivers drawn as water (waterShade): two tiles of real river surface from
// the L2 drone photos (scripts/water_texture.py), the milky shallow water
// near the banks in data/water.png on unit 10 and the dark channel water in
// data/water_deep.png on unit 12; water.json holds their sizes and how the
// bed fades under the water (its "layers").
let waterOn = false;
let waterTex = null, waterDeepTex = null, waterMeta = null;
const WATER_UNIT = 10, WATER_DEEP_UNIT = 12;
// Flowing water: the surface moves down the river at up to FLOW_SPEED
// (metres per second) and the view redraws at most FLOW_FPS times a second
// for it while nothing else moves. The shader's cycle is 2 s, so time wraps
// at a whole number of cycles.
let flowOn = false;
const FLOW_SPEED = 0.6, FLOW_FPS = 30, FLOW_WRAP_S = 3600;
function flowing() {
  return flowOn && waterOn && waterTex && field && field.carve && field.riverAny;
}
function waterTexture(unit, img) {
  const tex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.generateMipmap(gl.TEXTURE_2D);
  return tex;
}
async function loadWater() {
  try {
    const meta = await (await fetch('./data/water.json', { cache: 'no-cache' })).json();
    const load = async (src) => { const img = new Image(); img.src = src; await img.decode(); return img; };
    const shallow = await load('./data/water.png');
    const deep = meta.deep ? await load('./data/water_deep.png') : shallow;
    waterTex = waterTexture(WATER_UNIT, shallow);
    waterDeepTex = waterTexture(WATER_DEEP_UNIT, deep);
    waterMeta = meta;
  } catch (e) { /* no water texture: the option stays off */ }
}
// The river (riverChannel): its channel cut into the ground (field.carve,
// which height() and the shaders add) and its water level, built while
// water or the water map is on and redone when width, relief or the
// field's size in the world moves. RGBA32F on unit 11: carve and level in
// metres, the direction the water runs.
let riverWidth = 8;
let riverTex = null, riverKey = '', riverOf = null;
const RIVER_UNIT = 11;
// The channel's waterline slope (depth over distance from the edge): its
// parabola is width_depth times wider than deep, so 4 / width_depth.
const EDGE_SLOPE = 4 / RIVER.width_depth;
function riverOn() {
  return !!(field && field.carve && riverTex);
}
function buildRiver() {
  const want = (waterOn || waterView) && field && field.w === field.h;
  if (!want) {
    if (field) { field.carve = null; field.barZone = null; field.level = null; }
    riverKey = '';
    return;
  }
  const n = field.w;
  const key = [n, field.extent, amplitude(), riverWidth].join();
  if (key === riverKey && riverOf === field && field.carve) return;
  riverKey = key; riverOf = field;
  const r = riverChannel(field.z, n, { cellM: field.extent / (n - 1), metresPerZ: 2 * amplitude(),
                                       width: riverWidth });
  const carve = new Float32Array(n * n);
  let any = false;
  for (let c = 0; c < n * n; c++) {
    carve[c] = r[4 * c];
    if (r[4 * c + 1] > NO_WATER / 2) any = true;
  }
  field.carve = carve;
  field.riverAny = any;
  field.level = new Float32Array(n * n);
  for (let c = 0; c < n * n; c++) field.level[c] = r[4 * c + 1];
  field.barZone = barZone(r, carve, n, field.extent / (n - 1));
  if (!riverTex) riverTex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0 + RIVER_UNIT);
  gl.bindTexture(gl.TEXTURE_2D, riverTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, n, n, 0, gl.RGBA, gl.FLOAT, r);
}
/** Where the river leaves gravel, 0..1 per field cell (over 0.5: bar).
 *  A bar is low beside the channel: within about a channel depth above the
 *  water (RIVER.bar_height depths) and a few widths from it
 *  (RIVER.bar_reach widths), so a flat valley floor gets a wide one and a
 *  steep wall almost none; noise over 1.5 widths widens and breaks it up
 *  along the river. Distance and the nearest water's level by chamfer. */
function barZone(r, carve, n, cellM) {
  const amp = amplitude(), INF = 1e9;
  const d = new Float32Array(n * n), lev = new Float32Array(n * n);
  const ground = (c) => 2 * amp * (field.z[c] - 0.5) + carve[c];
  for (let c = 0; c < n * n; c++) {
    const wet = r[4 * c + 1] > NO_WATER / 2 && r[4 * c + 1] > ground(c);
    d[c] = wet ? 0 : INF;
    lev[c] = wet ? r[4 * c + 1] : 0;
  }
  const D1 = cellM, D2 = cellM * Math.SQRT2;
  const take = (c, q, step) => { if (d[q] + step < d[c]) { d[c] = d[q] + step; lev[c] = lev[q]; } };
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const c = j * n + i;
    if (i > 0) take(c, c - 1, D1);
    if (j > 0) {
      take(c, c - n, D1);
      if (i > 0) take(c, c - n - 1, D2);
      if (i < n - 1) take(c, c - n + 1, D2);
    }
  }
  for (let j = n - 1; j >= 0; j--) for (let i = n - 1; i >= 0; i--) {
    const c = j * n + i;
    if (i < n - 1) take(c, c + 1, D1);
    if (j < n - 1) {
      take(c, c + n, D1);
      if (i < n - 1) take(c, c + n + 1, D2);
      if (i > 0) take(c, c + n - 1, D2);
    }
  }
  const depth = Math.min(riverWidth / RIVER.width_depth, RIVER.max_depth);
  const hB = Math.max(0.1, RIVER.bar_height * depth), reach = RIVER.bar_reach * riverWidth;
  const out = new Float32Array(n * n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const c = j * n + i;
    if (d[c] >= INF) continue;
    const above = Math.max(0, ground(c) - lev[c]);
    const vary = 0.6 + 0.8 * dValue(i * cellM / (1.5 * riverWidth) + 3.1, j * cellM / (1.5 * riverWidth) + 7.7);
    out[c] = Math.min(1, Math.exp(-above / hB) * Math.exp(-d[c] / reach) * vary);
  }
  return out;
}
/** River and water uniforms for a program's uniform table `u`. */
function setWater(u) {
  const river = riverOn();
  gl.uniform1i(u.river, RIVER_UNIT);
  gl.uniform1f(u.hasRiver, river ? 1.0 : 0.0);
  const on = waterOn && waterTex && river;
  gl.uniform1f(u.water, on ? 1.0 : 0.0);
  gl.uniform1i(u.waterTex, WATER_UNIT);
  gl.uniform1i(u.waterDeep, WATER_DEEP_UNIT);
  gl.uniform1f(u.flow, on && flowing() ? FLOW_SPEED : 0.0);
  // The demo's frame-by-frame render gives the water its own clock.
  const now = flowClock != null ? flowClock : performance.now() * TIME_SCALE / 1000;
  gl.uniform1f(u.time, now % FLOW_WRAP_S);
  if (!on) return;
  const deep = waterMeta.deep || waterMeta;
  const layers = waterMeta.layers || { t0: 0.93, length_m: 0.53 };
  gl.uniform1f(u.waterMetres, waterMeta.metres || 1);
  gl.uniform1f(u.deepMetres, deep.metres || 1);
  gl.uniform3fv(u.waterMean, new Float32Array(waterMeta.mean || [0.5, 0.55, 0.55]));
  gl.uniform3fv(u.deepMean, new Float32Array(deep.mean || [0.35, 0.37, 0.39]));
  // The photos measure the bed's share against distance from the waterline;
  // on this channel that distance is depth / EDGE_SLOPE.
  gl.uniform2f(u.layers, layers.t0, layers.length_m * EDGE_SLOPE);
}
let riverRule = false;     // rule.json's river classes (HAND, channels); a tileset can ask for it
let barClass = null;       // the tile class a river's bars are made of (manifest bar)
let decalCopies = null;
let patchOverlay = null;   // per tile: a decal overlay ({ piece, at, rIn, r }) or null
let decalsOn = false;
let decalSpacing = 5;
let decalTerrain = false;
let decalCount = 0;
// Frames (export_wang.py --frames): the manifest's ring and turn rules, the
// centres ({start, count, levels, class, z0}, decal copies with `base`),
// each class's pool of centre indices, and each tile's ground height.
let frameInfo = null;
let centres = null;
let centrePools = null;
let centreCopies = null;
let tileZ0 = null;
let tileRing = null, centreRing = null;     // ground colour around the ring
// Live orders of turned centres: one buffer, `spunAt` maps a centre's
// level range and turn to where its order starts.
const spunBuf = gl.createBuffer();
let spunAt = new Map();
// Per sorted range, how many of its splats come before its middle: a frame
// is drawn in two parts with its centre between. `rangeIndex` maps a
// range's start to its place in the worker's list.
let rangeIndex = new Map(), liveBehind = null, cacheBehind = null;
// Class of each tile and how many classes; all zeros with one class.
let tileClass = null;
let classCount = 1;
let classOn = true;
let classSharp = 2.0;
// Share of the grid for the first class. Left to itself the rule gave ~27%
// on every scene; balance decides how much, the rule decides where.
let classBalance = 0.5;
// The capture's own share of each material, when the tileset measured it
// (export_wang --min-share); the share slider and presets start from it.
let captureShares = null;
const snapShare = (x) => Math.min(0.95, Math.max(0.05, Math.round(x * 20) / 20));
// Patch size in tiles, so the result is regions, not a scatter of single tiles.
let classCoherence = 2;
// How much altitude counts against shape (without it a hollow on a summit
// read the same as one in a valley).
let classAltitude = 0.4;
// How far the height line between materials wanders (wanderField): 0 one
// contour across the world, as before; a tileset that measured its capture
// (export --height-order) starts at what the capture shows.
let classWander = 0;
// Off by default: it doubles draw calls near class boundaries.
let classBlend = false;
// Blend by spot (uSpots): each spot one material. ?blend=splat brings back
// the earlier blend, dissolved splat by splat from the corners within
// blendWidth of a tie; it is kept for comparison and has no control.
const blendWidth = 0.35;
// How wide materials dissolve into each other by spot (0 the narrow strip).
let spotWidth = 0;
const blendSpots = new URLSearchParams(location.search).get('blend') !== 'splat';
let toneGrade = true;     // near another material, centres of the nearest tones
let cellMix = null;
let cellAlt = null;        // the class each cell would take instead
// Swap which tile class plays which role in the rule. Class order from the
// exporter is arbitrary (in the desert set class 0 is sand), so which
// material collects is a choice. Swapped, rule class k draws tile class
// count-1-k; the rule, shares and class view are unchanged.
let classSwap = false;
const tileClassFor = (k) => (classSwap ? classCount - 1 - k : k);
let blendedCells = 0;
let classStats = null;
// The maps and decision of the last rule run, for the demo's tables.
let lastRule = null;
// Mean colour of each class (linear 0..1), for the terrain window previews.
let classColours = [[0.62, 0.55, 0.45]];
let tileSize = 0;
let gridN = 1;
let cells = [];
// The grid `cells` was built for. gridN moves with the slider at once; the
// cells follow a rebuild later, and frames in between must index by this.
let cellsN = 1;
let seed = 1;
let usedPatches = 0;
let sortedReady = false, sortPending = false, lastSortKey = '', sortMs = 0;
let sortingEnabled = true, gain = 1;

// Cached per-direction orders, as in GSWT: `cacheDirs` the directions,
// `cacheBuf` every order back to back, `cacheStride` splats per order.
let sortMode = 'live';        // 'live' | 'cached'
let cacheViews = 9;
let cacheDirs = null, cacheBuf = null, cacheStride = 0, cacheMs = 0;
let cacheReady = false;
let showEdges = false, showDiagonals = false, showTints = false;
let relief = 0, reliefScale = 6, edgeBand = 0.12, subdiv = 0;
// Once the scale slider is dragged, the grid stops adjusting it.
let reliefScaleTouched = false;
// Height field and openness on the GPU (see the shader's fieldAt).
let fieldTex = null;
let openTex = null;
let shotWanted = false;
// Far field (see FAR_VERT): tile atlas, a texture of which tile each cell
// holds, and one mesh for all distant ground.
let farOn = true;
let farStartTiles = 14;          // splats out to here, atlas beyond
const FAR_OCTAVES = 3;           // the far field starts at most lodBase * 2^3 tiles out
let atlas = null;                // { tex, cols, rows }
let atlasTex = null;             // the GPU copy, freed when replaced
let cellTex = null;
let centreTex = null;     // frames: per cell, its centre and turn
let centreToneTex = null; // and its mean tint
let tileInfoTex = null;   // per tile: colour gain, mean colour
let farVAO = null, farCount = 0;
// Distant land past the world's edge (rangesAt in FAR_VERT): hill height in
// multiples of the world's relief (x RANGES_RELIEF), landforms the size of
// the world's own (relief scale); 0 is off and draws nothing past the edge.
// Hills several world widths tall dwarfed the world and read as a toy.
let ranges = 0;
let horizonVAO = null, horizonCount = 0, horizonKey = '';
const HORIZON_OUT = 4;      // outer radius in world widths; the edge haze has it by then
const RANGES_RELIEF = 2.5;
/** A ring mesh from inside the world's edge to HORIZON_OUT widths, rings
 *  spaced geometrically: fine near the edge, coarse where the haze is thick.
 *  It starts at 0.4 of the half width, so it still meets the world while
 *  the film shrinks it by rings (FAR_FRAG drops what lies inside). */
function buildHorizonMesh() {
  const half = gridN * (tileSize || 1) / 2;
  const key = `${half}`;
  if (key === horizonKey && horizonVAO) return;
  horizonKey = key;
  const K = 112, M = 512, R = HORIZON_OUT * 2 * half, r0 = 0.4 * half;
  const xy = new Float32Array(K * M * 2);
  for (let k = 0; k < K; k++) {
    const r = r0 * Math.pow(R / r0, k / (K - 1));
    for (let m = 0; m < M; m++) {
      const a = 2 * Math.PI * m / M;
      xy[2 * (k * M + m)] = r * Math.cos(a);
      xy[2 * (k * M + m) + 1] = r * Math.sin(a);
    }
  }
  const ix = new Uint32Array((K - 1) * M * 6);
  let t = 0;
  for (let k = 0; k < K - 1; k++) {
    for (let m = 0; m < M; m++) {
      const v = k * M + m, w = k * M + (m + 1) % M;
      ix[t++] = v; ix[t++] = w; ix[t++] = v + M;
      ix[t++] = w; ix[t++] = w + M; ix[t++] = v + M;
    }
  }
  if (!horizonVAO) {
    horizonVAO = gl.createVertexArray();
    buildHorizonMesh.vb = gl.createBuffer(); buildHorizonMesh.ib = gl.createBuffer();
  }
  gl.bindVertexArray(horizonVAO);
  gl.bindBuffer(gl.ARRAY_BUFFER, buildHorizonMesh.vb);
  gl.bufferData(gl.ARRAY_BUFFER, xy, gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(farProg, 'aXY');
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buildHorizonMesh.ib);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, ix, gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  horizonCount = ix.length;
}
// A scene that is not a Wang tileset is shown as the capture itself: no
// warping onto terrain, no material rule.
let plainScene = false;
let tileMeans = null;   // mean splat colour per tile, linear 0..1
let loadedManifest = null;
// Cells of each tile class in the current layout, for the scene card.
let tileClassCounts = null;
// Names and notes for the public tilesets (web/tilesets.json); a manifest's
// own "names" wins.
let tilesetMeta = {};
fetch('./tilesets.json').then((r) => (r.ok ? r.json() : {}))
  .then((m) => { tilesetMeta = m || {}; updateShareLabel(); })
  .catch(() => {});

/** A name per tile class: the manifest's, the catalogue's, or a number. */
function materialNames() {
  const meta = tilesetMeta[loadedScene] || {};
  const given = (loadedManifest && loadedManifest.names) || meta.names || [];
  return Array.from({ length: classCount }, (_, k) => given[k] || `material ${k + 1}`);
}

/** The share slider names the material it sets: the one that collects. */
function updateShareLabel() {
  const names = materialNames();
  names.forEach((name, m) => {
    const lbl = $(`classshare${m}lbl`);
    if (lbl) lbl.textContent = `${name} share`;
  });
  const el = ui.classbalancelbl;
  if (!el) return;
  el.textContent = classCount > 1 ? `${names[tileClassFor(0)]} share` : 'share';
}

// Three or more materials: one share slider each (the one slider sets the
// first material's and splits the rest as the capture does). Moving one
// keeps the others' proportions and the total at 100%. Per material, so
// the shares follow their materials through a swap. Two: as before.
let shareByMat = null;
function buildShareSliders() {
  const box = ui.classshares;
  const one = ui.classbalance;
  const oneRow = one && one.closest('.s');
  shareByMat = null;
  if (box) box.textContent = '';
  if (oneRow) oneRow.style.display = classCount >= 3 ? 'none' : '';
  if (!box || classCount < 3) return;
  const byPos = classShares(classCount);           // where the one slider would start
  shareByMat = new Array(classCount).fill(0);
  byPos.forEach((v, k) => { shareByMat[tileClassFor(k)] = v; });
  for (let m = 0; m < classCount; m++) {
    const row = document.createElement('div');
    row.className = 's';
    row.dataset.info = 'share';
    row.innerHTML = `<input type="range" id="classshare${m}" min="0.02" max="0.96" step="0.01">`
      + `<label><span id="classshare${m}lbl"></span><b id="classshare${m}n"></b></label>`;
    box.appendChild(row);
    const input = row.querySelector('input');
    input.defaultValue = String(shareByMat[m]);     // the capture's own share; links leave it out
    input.addEventListener('input', (e) => setShare(m, +e.target.value));
  }
  showShares();
  updateShareLabel();
}
function showShares() {
  shareByMat.forEach((v, m) => {
    const input = $(`classshare${m}`);
    if (input && Math.abs(+input.value - v) > 0.005) input.value = String(v);
    const n = $(`classshare${m}n`);
    if (n) n.textContent = `${Math.round(v * 100)}%`;
  });
}
function setShare(m, v) {
  const rest = shareByMat.reduce((a, s, i) => (i === m ? a : a + s), 0);
  const n = shareByMat.length;
  shareByMat = shareByMat.map((s, i) => (i === m ? v
    : rest > 0 ? s * (1 - v) / rest : (1 - v) / (n - 1)));
  showShares();
  scheduleRegenerate();
}
/** Several shares at once (a shared link): set them, then scale to 100% once.
 *  One by one through setShare, each would rescale the ones set before it. */
function setShares(given) {
  if (!shareByMat) return;
  for (const [m, v] of Object.entries(given)) if (m < shareByMat.length) shareByMat[m] = +v;
  const sum = shareByMat.reduce((a, s) => a + s, 0);
  if (sum > 0) shareByMat = shareByMat.map((s) => s / sum);
  showShares();
  scheduleRegenerate();
}

// Auto quality, the way games keep a frame budget.
// First lever: render resolution (50-100%, 33.3 ms budget, Unreal's dynamic
// resolution defaults), the canvas scaled up by the browser. Second, only
// when that is not enough: the LOD distance, capped by the slider.
// Proportional control (Intel's dynamic-resolution sample) plus a panic drop
// on several frames far over budget; recovery slower than the drop.
let autoQuality = true;
let lodBaseUser = 8;            // what the slider says
let lodBaseKept = null;         // the slider's value while a tileset's lod_base stands in
// Phones start lower; auto quality raises it if there is room.
let renderScale = matchMedia('(pointer: coarse)').matches ? 0.6 : 1;
let frameMs = 16;               // smoothed frame time
let overRun = 0, qualityTick = 0, lastNote = '', lastNoteAt = 0;
const BUDGET_MS = 33.3;
const SCALE_MIN = 0.5;
const QUALITY_MIN_BASE = 3;

/** The note over the canvas, hidden after `ms` (0: until hidden or
 *  replaced). A newer note cancels the older one's timer. */
let busyTimer = 0;
function showBusy(text, ms = 0) {
  if (!ui.busy) return;
  ui.busy.textContent = text;
  ui.busy.style.display = 'block';
  clearTimeout(busyTimer);
  if (ms) busyTimer = setTimeout(hideBusy, ms);
}
function hideBusy() {
  if (ui.busy && !regenPending) ui.busy.style.display = 'none';
}

function qualityNote(text) {
  // At most one note every 1.5 s, and the same one not within 2 s.
  const now = performance.now();
  if (now - lastNoteAt < 1500 || (text === lastNote && now - lastNoteAt < 2000)) return;
  // A running commentary is for measuring; Explore shows the result only.
  if (!document.body.classList.contains('expert')) return;
  lastNoteAt = now;
  lastNote = text;
  showBusy(text, 2000);
}

// Two levers on two time scales. Resolution answers within 250 ms. A LOD
// step re-sorts cells and is a spike itself; moving it back at the first
// quiet moment made the two chase each other (20 and 45 fps in turn). So
// LOD moves only on sustained need, restores with a wide margin, and holds
// still while a change settles.
let lodHold = 0, slowFor = 0, quickFor = 0;

function tuneQuality(dt) {
  if (!autoQuality || !splatCount || !(dt > 0) || dt > 500) return;
  // Benchmark and capture sweeps are measurements: full resolution and the
  // user's LOD distance, or the numbers are skewed.
  if (bench.active || capture.active) {
    if (renderScale !== 1 || lodBase !== lodBaseUser) {
      renderScale = 1;
      lodBase = lodBaseUser;
      if (ui.lodbasen) ui.lodbasen.textContent = `${lodBase} tiles`;
    }
    return;
  }
  if (lodHold > 0) {             // settling after a detail change
    lodHold -= dt;
    overRun = 0;
    return;
  }
  frameMs += (dt - frameMs) * 0.1;
  overRun = dt > BUDGET_MS * 1.5 ? overRun + 1 : 0;

  // Panic: five frames in a row far over budget.
  if (overRun >= 5 && renderScale > SCALE_MIN) {
    renderScale = Math.max(SCALE_MIN, renderScale * 0.8);
    overRun = 0;
    qualityTick = 0;
    qualityNote(`resolution ${Math.round(renderScale * 100)}% to keep up`);
    return;
  }
  qualityTick += dt;
  slowFor = frameMs > BUDGET_MS * 1.05 ? slowFor + dt : 0;
  quickFor = frameMs < BUDGET_MS * 0.6 ? quickFor + dt : 0;
  if (qualityTick < 250) return;
  qualityTick = 0;

  // Pixel cost goes with scale squared: the scale that lands at 90% of budget.
  const fit = Math.max(SCALE_MIN, Math.min(1,
    renderScale * Math.sqrt((BUDGET_MS * 0.9) / Math.max(frameMs, 1))));
  let detail = 0;
  if (frameMs > BUDGET_MS * 1.05) {
    if (renderScale > SCALE_MIN + 1e-3) {
      renderScale = Math.max(SCALE_MIN, renderScale + 0.8 * (fit - renderScale));
      qualityNote(`resolution ${Math.round(renderScale * 100)}% to keep up`);
    } else if (slowFor > 1500 && lodBase > QUALITY_MIN_BASE) {
      detail = -1;
    }
  } else if (frameMs < BUDGET_MS * 0.85) {
    if (renderScale < 1) {
      renderScale = Math.min(1, renderScale + Math.max(0.02, 0.5 * (fit - renderScale)));
      qualityNote(renderScale >= 1 ? 'full resolution'
        : `resolution back to ${Math.round(renderScale * 100)}%`);
    } else if (quickFor > 4000 && lodBase < lodBaseUser) {
      detail = 1;
    }
  }
  if (detail) {
    lodBase = Math.max(Math.min(QUALITY_MIN_BASE, lodBaseUser), Math.min(lodBaseUser, lodBase + detail));
    lodHold = 3000;
    slowFor = quickFor = 0;
    qualityNote(detail < 0 ? `detail from ${lodBase} tiles to keep up`
                           : `detail restored to ${lodBase} tiles`);
  }
  if (ui.lodbasen) {
    ui.lodbasen.textContent = lodBase === lodBaseUser
      ? `${lodBase} tiles` : `${lodBase} tiles (auto, set ${lodBaseUser})`;
  }
}

/** The height the terrain is actually raised by.
 *
 *  `relief` is tuned for a terrain ~24 tiles across. Relief scale follows
 *  the grid, so on a 256-tile world the same height was spread ten times
 *  wider and the hills went flat; this grows the height with the width. */
const RELIEF_REFERENCE = 24;
function amplitude() {
  return relief * Math.max(1, reliefScale / RELIEF_REFERENCE);
}
let loadedScene = '';
let figureMode = false;
let figExitTimer = 0;
let groundShade = 0.4;
let exposure = 1;
let saturation = 1;
let macro = 0.5;

// Turning the layout off the world axes. It moves the axis-alignment effect
// rather than removing it (a grid at 22 degrees aligns at 22, 112, ...).
let gridAngle = 0;
const FLAT = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

// A measured or generated height field; null means the analytic surface.
let field = null;

// Detail relief, the JS twin of detailAt() in the shaders: same integer hash
// (Math.imul is GLSL's uint multiply), so every part of the viewer agrees
// on where the ground is.
let detailOn = true, detailStrength = 1;
function detailAmp() { return detailOn ? 0.05 * detailStrength * (tileSize || 1) : 0; }
function detailFreq() { return 1 / (0.7 * (tileSize || 1)); }
function dHash(cx, cy) {
  let h = (Math.imul(cx, 0x27D4EB2D) ^ Math.imul(cy, 0x165667B1)) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x85EBCA6B) >>> 0;
  h = (h ^ (h >>> 13)) >>> 0;
  return (h & 0xFFFFFF) / 16777215;
}
function dValue(qx, qy) {
  const ix = Math.floor(qx), iy = Math.floor(qy);
  const fx = qx - ix, fy = qy - iy;
  const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  const a = dHash(ix, iy), b = dHash(ix + 1, iy);
  const d = dHash(ix, iy + 1), e = dHash(ix + 1, iy + 1);
  return (a + (b - a) * sx) * (1 - sy) + (d + (e - d) * sx) * sy;
}
function detailAt(x, y) {
  const amp = detailAmp();
  if (amp <= 0) return 0;
  const f = detailFreq();
  const r1 = 1 - Math.abs(2 * dValue(x * f, y * f) - 1);
  const r2 = 1 - Math.abs(2 * dValue(x * f * 2.03 + 17.3, y * f * 2.03 + 5.1) - 1);
  const mask = field ? field.sampleMask(x, y) : 1;
  return amp * mask * ((r1 * r1 + 0.5 * r2 * r2) / 1.5 - 0.5);
}

/** What the camera stands on: the ground, or a river's water over it while
 *  water is drawn (the channel is cut below the level; under it the eye
 *  looked up at the water from beneath). */
function surfaceAt(x, y) {
  const g = height(x, y);
  return waterOn && field && field.level ? Math.max(g, field.sampleLevel(x, y)) : g;
}

/** Ground height at a point, the CPU copy of terrainHeight() in the shader. */
function height(x, y) {
  const relief = amplitude();
  // A river's channel (buildRiver), metres, with or without relief.
  const carve = field ? field.sampleCarve(x, y) : 0;
  if (relief <= 0) return carve;
  if (field) {
    // Centred on zero. The field's extent is set by relief scale (tiles per
    // landform), and it mirrors past its edge, so a small survey still reads
    // at any grid size.
    return relief * (field.sample(x, y) - 0.5) * 2.0 + detailAt(x, y) + carve;
  }
  // Wavelength in tiles, so the shape stays the same relative to the tiling.
  const f = 1 / Math.max(reliefScale * (tileSize || 1), 0.01);
  return relief * (Math.sin(f * x) * Math.cos(f * y)
    + 0.5 * Math.sin(2.3 * f * x + 1.7) * Math.cos(1.9 * f * y + 0.4))
    + detailAt(x, y);
}

/** Tangent frame at a point (two tangents and the normal), column-major
 *  for uniformMatrix3fv. Not `frame` - that is the render loop. */
function tangentFrame(x, y) {
  if (amplitude() <= 0) return FLAT;
  // Same step as the shader's terrainGrad: one texel on a field.
  const e = field
    ? Math.max(field.extent, 1e-6) / Math.max(field.w, field.h, 2)
    : Math.max(reliefScale * (tileSize || 1), 0.01) * 0.01;
  const dx = (height(x + e, y) - height(x - e, y)) / (2 * e);
  const dy = (height(x, y + e) - height(x, y - e)) / (2 * e);
  const la = Math.hypot(1, dx), lb = Math.hypot(1, dy);
  const a = [1 / la, 0, dx / la];
  const b = [0, 1 / lb, dy / lb];
  let c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
           a[0] * b[1] - a[1] * b[0]];
  const lc = Math.hypot(...c) || 1;
  c = c.map(v => v / lc);
  return new Float32Array([a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]]);
}

let lineCount = 0;
let splatWidth = 0;        // median splat long axis, the unit gaps are judged in
let positionsRef = null;   // splat positions, kept for measuring sort error
let lastOrder = null;      // the live order most recently returned
let yawRate = 0, lastYaw = null, lastYawT = 0;   // turn rate about the vertical, deg/s
let lodLevels = 1;         // levels available per tile
let lodBase = 8;           // distance in tiles at which level 1 takes over
let lodOn = true;
let lodCounts = [];        // cells drawn at each level, for the readout
let showLodColours = false;
let showIdentity = false;
let showClasses = false;
// One colour per rule class: collected first, then exposed, then the rest.
const CLASS_RGB = [[0.30, 0.62, 0.35], [0.88, 0.72, 0.30],
                   [0.36, 0.54, 0.86], [0.80, 0.42, 0.62]];

// Named skies; the fog has to match the horizon colour.
const SKIES = {
  none:     { top: [0, 0, 0], horizon: [0, 0, 0], ground: [0, 0, 0], fog: 0 },
  overcast: { top: [0.52, 0.57, 0.62], horizon: [0.78, 0.80, 0.82],
              ground: [0.10, 0.10, 0.11], fog: 0.020, sun: [0.30, 0.26, 0.20] },
  dusk:     { top: [0.10, 0.13, 0.24], horizon: [0.72, 0.47, 0.35],
              ground: [0.05, 0.05, 0.07], fog: 0.030, sun: [0.55, 0.32, 0.16] },
  clear:    { top: [0.22, 0.45, 0.78], horizon: [0.70, 0.80, 0.90],
              ground: [0.08, 0.09, 0.10], fog: 0.012, sun: [0.32, 0.28, 0.20] },
};
// Atmosphere (ATMOS_GLSL), all off at 0: the sun's glow and its direction
// (degrees; the light in the scans is baked, so it only lights the sky and
// haze), mist that lies in the low ground, and a colour grade.
let sunGlow = 0, sunAzimuth = 225, sunElevation = 6;
let mist = 0, gradeAmount = 0, seaHaze = 0;
const SEA_BASE = 0.06;    // sea haze density per world unit under its top, 1, 24 cells
const MIST_BASE = 0.3;    // density per world unit over the channels, mist 1, 24 cells
function atmosUniforms(U, S, debug) {
  const az = sunAzimuth * Math.PI / 180, el = sunElevation * Math.PI / 180;
  gl.uniform4f(U.sun, Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el),
               debug ? 0 : sunGlow);
  gl.uniform3fv(U.sunRGB, new Float32Array(S.sun || [0, 0, 0]));
  // Densest over the channel each spot drains to (drainBaseAt), falling by
  // e every 0.3 relief above it: valley fog fills the bottom of a valley,
  // not its sides. Measured from the lowest ground instead, it filled
  // basins to a level shore that read as a lake.
  const amp = Math.abs(amplitude()) || 0.5;
  gl.uniform4f(U.mist, debug || S === SKIES.none ? 0
    : mist * MIST_BASE * Math.min(1, FOG_CELLS / Math.max(cellsN, 1)),
  1 / Math.max(0.1, 0.3 * amp), -amp, 0);
  // Sea haze: a layer whose top lies a quarter of the relief above the
  // middle, so the higher ground stands out of it.
  gl.uniform4f(U.marine, debug || S === SKIES.none ? 0
    : seaHaze * SEA_BASE * Math.min(1, FOG_CELLS / Math.max(cellsN, 1)),
  0.25 * amp, Math.max(0.05, 0.15 * amp), 0);
  gl.uniform4f(U.grade, debug ? 0 : gradeAmount, 1, 0.35, 0.12);
}
let sky = 'overcast';
let fogScale = 1.0;
// Haze per metre is set for a 24-cell world. A larger world keeps the haze
// it has at the same fraction of its width: at a fixed density per metre,
// a few hundred metres of ground read as haze alone.
const FOG_CELLS = 24;
function fogDensity(sky) {
  return sky.fog * fogScale * Math.min(1, FOG_CELLS / Math.max(cellsN, 1));
}
let orbiting = false;

// Green to red as detail drops, the usual engine LOD debug colours.
const LOD_RGB = [[0.30, 0.85, 0.35], [0.95, 0.85, 0.25],
                 [0.98, 0.58, 0.20], [0.92, 0.30, 0.30],
                 [0.75, 0.35, 0.85], [0.40, 0.60, 0.95]];

/** A fixed hue per tile index (golden angle). The tile of a cell is chosen
 *  once when the grid is built, so if a cell changes colour while turning,
 *  cells are swapping tiles and something is wrong. */
function tileRGB(i) {
  const h = ((i * 137.508) % 360) / 360;
  const k = (n) => {
    const v = (n + h * 6) % 6;
    return Math.max(0, Math.min(1, Math.min(v, 4 - v, 1)));
  };
  return [0.30 + 0.70 * k(5), 0.30 + 0.70 * k(3), 0.30 + 0.70 * k(1)];
}

/** Which level a cell uses, and how far through the cross-fade it is.
 *  Level i starts at lodBase * 2^i tiles; neighbouring levels blend over
 *  the last 20% (GSWT 3.5 uses ~5%; ours differ more between levels). */
function lodFor(distance) {
  if (!lodOn || lodLevels < 2) return [0, 1, 0];
  const d = distance / Math.max(lodBase * (tileSize || 1), 1e-3);
  const f = Math.log2(Math.max(d, 1e-6));
  const lvl = Math.max(0, Math.min(Math.floor(f), lodLevels - 1));
  const frac = f - lvl;
  const band = 0.2;
  if (frac > 1 - band && lvl + 1 < lodLevels) {
    const w = (frac - (1 - band)) / band;
    return [lvl, 1 - w, w];
  }
  return [lvl, 1, 0];
}
let drawnSplats = 0, drawCalls = 0;
let frames = 0, fpsTime = performance.now();
// Frame pacing: ?fps=30 lowers the cap (default 60); idle after WAKE_FRAMES
// frames with nothing changed.
const MAX_FPS = Math.max(10, Math.min(240,
  +new URLSearchParams(location.search).get('fps') || 60));
const WAKE_FRAMES = 90;
let wakeFrames = WAKE_FRAMES, lastFrameKey = '', lastDrawAt = 0;

const worker = new Worker('./sort-worker.js');
// Merged streams sort on their own worker, so the live sort never queues
// behind them; it holds its own copy of the positions.
const mergeWorker = new Worker('./sort-worker.js');
// The live sort split over helpers, each holding only its share of the
// positions: one sort over everything took 120-210 ms, and a fast turn swept
// past 90 degrees in that time, where the stale order is nearly reversed
// (the ground's underside drawn on top). `worker` keeps the cached orders and
// turned centres, and sorts alone when the split does not apply.
const SORT_HELPERS = 3;
const sortHelpers = Array.from({ length: SORT_HELPERS }, () => new Worker('./sort-worker.js'));
let helperChunks = null;                // per helper: { from, to, ks } (splat range, range indices)
let helperSeq = 0, helperPending = 0, helperMs = 0;
worker.onmessage = (e) => {
  const m = e.data;
  wakeFrames = WAKE_FRAMES;

  if (m.type === 'cached') {
    cacheDirs = m.dirs;
    cacheBuf = new Uint32Array(m.order);
    cacheStride = m.stride;
    cacheMs = m.ms;
    cacheBehind = new Uint32Array(m.behind);
    cacheReady = true;
    // One buffer holding every order; a cell binds the slice it needs.
    gl.bindVertexArray(splatVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, cacheIndexBuf);
    gl.bufferData(gl.ARRAY_BUFFER, cacheBuf, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    console.log(`cached ${m.views} orders in ${m.ms.toFixed(0)} ms`);
    return;
  }

  if (m.type === 'mergedGroups') {
    if (m.seq !== mergeSeq || !mergePendingSigs) return;   // sent before a reset
    mergePending = false;
    const f = mergePendingFwd;
    const dot = (a) => a[0] * f[0] + a[1] * f[1] + a[2] * f[2];
    m.orders.forEach((buf, k) => {
      const sig = mergePendingSigs[k];
      const list = mergeCache.get(sig) || [];
      mergeCache.delete(sig);                   // newest last, so the oldest go first
      // A new heading replaces one it nearly repeats, else the oldest when full.
      let at = list.findIndex((e) => dot(e.fwd) > MERGE_NEAR);
      if (at < 0 && list.length >= MERGE_HEADINGS) at = 0;
      if (at >= 0) { mergeBytes -= list[at].stream.byteLength; list.splice(at, 1); }
      const stream = new Uint32Array(buf);
      list.push({ fwd: f, stream });
      mergeBytes += stream.byteLength;
      mergeCache.set(sig, list);
    });
    while (mergeBytes > MERGE_BYTES && mergeCache.size) {
      const [oldest, list] = mergeCache.entries().next().value;
      for (const e of list) mergeBytes -= e.stream.byteLength;
      mergeCache.delete(oldest);
    }
    mergeStats.ms = m.ms;
    mergeStats.splats = m.sizes.reduce((t, n) => t + n, 0);
    return;
  }

  if (m.type !== 'sorted') return;
  if (!sortingEnabled) { sortPending = false; return; }
  gl.bindVertexArray(splatVAO);
  gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
  lastOrder = new Uint32Array(m.order);
  gl.bufferData(gl.ARRAY_BUFFER, lastOrder, gl.DYNAMIC_DRAW);
  liveFwd = liveFwdPending;
  liveBehind = m.behind ? new Uint32Array(m.behind) : null;
  spunAt = new Map();
  if (m.keys && m.keys.length) {
    gl.bindBuffer(gl.ARRAY_BUFFER, spunBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Uint32Array(m.spun), gl.DYNAMIC_DRAW);
    m.keys.forEach((k, i) => spunAt.set(k, m.offsets[i]));
  }
  gl.bindVertexArray(null);
  sortMs = m.ms;
  sortedReady = true;
  sortPending = false;
};
mergeWorker.onmessage = (e) => { if (e.data && e.data.type === 'mergedGroups') worker.onmessage(e); };
sortHelpers.forEach((h) => {
  h.onmessage = (e) => {
    const m = e.data;
    if (!m || m.type !== 'sortedPart') return;
    wakeFrames = WAKE_FRAMES;
    if (m.ring != null) {
      if (m.seq !== ringSeq || m.ring !== ringJob || !helperChunks) return;
      gl.bindVertexArray(splatVAO);
      gl.bindBuffer(gl.ARRAY_BUFFER, ringBufs[m.ring]);
      gl.bufferSubData(gl.ARRAY_BUFFER, helperChunks[m.part].from * 4, new Uint32Array(m.order));
      gl.bindVertexArray(null);
      if (++ringParts[m.ring] >= SORT_HELPERS) { ringFwd[m.ring] = ringJobFwd; ringJob = -1; }
      return;
    }
    if (m.seq !== helperSeq || !helperChunks || !sortingEnabled) {
      if (m.seq === helperSeq && --helperPending <= 0) sortPending = false;
      return;
    }
    const ch = helperChunks[m.part];
    const part = new Uint32Array(m.order);
    if (!lastOrder || lastOrder.length !== splatCount) lastOrder = identityOrder ? identityOrder.slice() : new Uint32Array(splatCount);
    lastOrder.set(part, ch.from);
    gl.bindVertexArray(splatVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, ch.from * 4, part);
    gl.bindVertexArray(null);
    const bh = new Uint32Array(m.behind);
    if (!liveBehind || liveBehind.length !== rangeIndex.size) liveBehind = new Uint32Array(rangeIndex.size);
    ch.ks.forEach((k, j) => { liveBehind[k] = bh[j]; });
    helperMs = Math.max(helperMs, m.ms);
    if (--helperPending <= 0) {
      sortPending = false;
      liveFwd = liveFwdPending;
      sortMs = helperMs;
      sortedReady = true;
      spunAt = new Map();
    }
  };
});

// ================================================================ layout

/** Share of the grid per class: the slider sets the first class, the rest
 *  split what is left evenly. */
function classShares(count) {
  // Three or more: each material's own slider (shareByMat).
  if (shareByMat && shareByMat.length === count) {
    return Array.from({ length: count }, (_, k) => shareByMat[tileClassFor(k)]);
  }
  const first = Math.min(0.95, Math.max(0.05, classBalance));
  // The rest keep the capture's proportions between them when known.
  // Shares belong to materials, so they follow them through a swap.
  const cs = captureShares && captureShares.length === count
    ? Array.from({ length: count }, (_, k) => captureShares[tileClassFor(k)]) : null;
  const others = cs ? cs.slice(1).reduce((a, b) => a + b, 0) : 0;
  return Array.from({ length: count }, (_, k) => {
    if (k === 0) return first;
    return cs && others > 0 ? (1 - first) * cs[k] / others
      : (1 - first) / Math.max(1, count - 1);
  });
}

/** A tile with the same edge code in another class, to blend against.
 *  Every class carries every code, so swapping cannot break the matching. */
/** The tile of material `cls` with the same edges as tile `t` (each
 *  material has every edge code), -1 if none. */
function tileOfClass(t, cls) {
  if (!wangCodes || !tileClass) return -1;
  const code = wangCodes[t];
  for (let k = 0; k < wangCodes.length; k++) {
    if (tileClass[k] !== cls || (variantOf && variantOf[k] >= 0)) continue;
    const o = wangCodes[k];
    if (o[0] === code[0] && o[1] === code[1] && o[2] === code[2] && o[3] === code[3]) return k;
  }
  return -1;
}

function alternateTile(c, p) {
  if (!wangCodes || !tileClass) return -1;
  const code = wangCodes[c.patch];
  // Blend towards the class the rule ranked next (matters with 3+ classes).
  for (let k = 0; k < wangCodes.length; k++) {
    if (c.alt != null && tileClass[k] !== tileClassFor(c.alt)) continue;
    if (tileClass[k] === tileClass[c.patch]) continue;
    if (variantOf && variantOf[k] >= 0) continue;
    const o = wangCodes[k];
    if (o[0] === code[0] && o[1] === code[1]
        && o[2] === code[2] && o[3] === code[3]) return k;
  }
  return -1;
}

/** `g` from mergeGroups with the cells marked in `alone` made singletons. */
function splitOut(g, alone) {
  if (!alone.some(Boolean)) return g;
  const groups = [];
  for (const x of g.groups) {
    const keep = x.filter((k) => !alone[k]);
    if (keep.length) groups.push(keep);
    for (const k of x) if (alone[k]) groups.push([k]);
  }
  const groupOf = new Int32Array(alone.length);
  groups.forEach((x, gi) => { for (const k of x) groupOf[k] = gi; });
  return { ...g, groups, groupOf };
}

/** Lay out gridN x gridN cells.
 *
 *  With a Wang set, west is fixed by the left neighbour and south by the one
 *  below; north and east are free. A complete set always has a fit, so no
 *  backtracking, and the free choices stop the terrain repeating. */
function buildGrid() {
  cells = [];
  if (!tileSize) { cells = [{ x: 0, y: 0, z: 0, warp: FLAT, patch: 0 }]; cellsN = 1; return; }
  usedPatches = Math.max(1, Math.min(usedPatches || patches.length,
                                     patches.length));
  let s = seed;
  const rand = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const half = (gridN - 1) / 2;
  const chosen = new Int32Array(gridN * gridN).fill(-1);
  cellsN = gridN;

  // Material per cell from the ground under it, once per layout. Classes
  // share edge codes, so the neighbours pick the code and the terrain picks
  // the class, and neither overrules the other.
  let wantClass = null;
  lastRule = null;
  cellMix = null;
  cellAlt = null;
  classStats = null;
  if (classOn && classCount > 1 && tileSize) {
    // Sampled finer than one point per tile: the maps are derivatives, and
    // at one sample per cell they alias (classes looked scattered, channels
    // narrower than a cell vanished). 4x up to 64 tiles, fewer beyond.
    const over = gridN <= 64 ? 4 : gridN <= 160 ? 2 : 1;
    const z = sampleGrid(gridN, tileSize, height, gridAngle, over);
    // The erosion's sediment record, on the same grid. Generated terrain only.
    const sed = field && field.sediment
      ? sampleGrid(gridN, tileSize, (x, y) => field.sampleSediment(x, y),
                   gridAngle, over)
      : null;
    const r = classify(z, gridN, classCount,
                       { spacing: tileSize, sharpness: classSharp,
                         coherence: classCoherence,
                         altitude: classAltitude,
                         heightShift: classWander > 0
                           ? wanderField(gridN, classWander, seed,
                                         Math.max(4, Math.min(gridN, reliefScale / 3)))
                           : null,
                         sediment: sed,
                         river: riverRule,
                         // The tileset names its bar material (manifest bar);
                         // a joined capture's height order cannot say which.
                         barPos: barClass == null ? 0 : (classSwap ? classCount - 1 - barClass : barClass),
                         barZone: riverRule && field && field.barZone
                           ? sampleGrid(gridN, tileSize, (x, y) => field.sampleBar(x, y),
                                        gridAngle, over)
                           : null,
                         balance: classCount === 2 ? classBalance
                           : classShares(classCount) });
    wantClass = r.cls;
    // How strongly each cell chose; near 1/classCount is where blending pays.
    cellMix = r.strength;
    cellAlt = r.runnerUp;
    const counts = new Array(classCount).fill(0);
    for (const k of wantClass) counts[k]++;
    let sure = 0;
    for (const v of r.strength) sure += v;
    classStats = { counts, sure: sure / Math.max(r.strength.length, 1),
                   source: r.source, weights: r.weights };
    lastRule = { n: gridN, maps: r.maps, cls: r.cls, strength: r.strength };
  }

  for (let j = 0; j < gridN; j++) {
    for (let i = 0; i < gridN; i++) {
      let pick;
      if (wangCodes) {
        const west = i > 0 ? wangCodes[chosen[j * gridN + i - 1]][1] : -1;
        const south = j > 0 ? wangCodes[chosen[(j - 1) * gridN + i]][0] : -1;
        const fits = [];
        for (let k = 0; k < wangCodes.length; k++) {
          if (variantOf && variantOf[k] >= 0) continue;
          const c = wangCodes[k];
          if (west >= 0 && c[3] !== west) continue;
          if (south >= 0 && c[2] !== south) continue;
          fits.push(k);
        }
        // Prefer the class the rule asked for. The fallback should never
        // fire (every class has every code); it is counted so it would show.
        let pool = fits;
        if (wantClass && tileClass) {
          const want = tileClassFor(wantClass[j * gridN + i]);
          const ofClass = fits.filter(k => tileClass[k] === want);
          if (ofClass.length) pool = ofClass;
          else if (classStats) classStats.missed = (classStats.missed || 0) + 1;
        }
        // Not the tile beside or below again where the codes allow another:
        // with two tiles per edge pair a row of three alike was common.
        if (captureShares && pool.length > 1) {
          const left = i > 0 ? chosen[j * gridN + i - 1] : -1;
          const below = j > 0 ? chosen[(j - 1) * gridN + i] : -1;
          const fresh = pool.filter(k => k !== left && k !== below);
          if (fresh.length) pool = fresh;
        }
        pick = pool.length ? pool[Math.floor(rand() * pool.length) % pool.length] : 0;
      } else {
        pick = Math.floor(rand() * usedPatches) % usedPatches;
      }
      chosen[j * gridN + i] = pick;
      const lx = (i - half) * tileSize, ly = (j - half) * tileSize;
      const a = gridAngle * Math.PI / 180;
      const x = Math.cos(a) * lx - Math.sin(a) * ly;
      const y = Math.sin(a) * lx + Math.cos(a) * ly;
      const cellIdx = j * gridN + i;
      cells.push({ i, j, x, y, z: height(x, y),
                   warp: tangentFrame(x, y), patch: pick,
                   cls: wantClass ? wantClass[cellIdx] : 0,
                   alt: cellAlt ? cellAlt[cellIdx] : null,
                   mix: cellMix ? cellMix[cellIdx] : 1 });
    }
  }
  // Decals last, over the finished layout: a copy has its tile's edges and
  // class, so swapping it in changes neither the matching nor the rule.
  decalCount = 0;
  if (decalsOn && decalCopies && wangCodes) {
    // Following the terrain: more where the rule was undecided (the edge
    // of the other material, where plants thin out into sand), fewer out
    // in the open; neighbours alike.
    const undecided = decalTerrain && cellMix && classCount > 1
      ? (k) => (1 - cellMix[k]) / (1 - 1 / classCount) : null;
    const placed = placeDecals((k) => cells[k].patch, gridN, decalCopies,
                               decalSpacing, seed,
                               { weight: undecided, groups: decalTerrain });
    for (const p of placed) cells[p.cell].patch = p.tile;
    decalCount = placed.length;
  }
  // Frames: each cell's centre from its material's pool (not the one left
  // of it or below where the pool has others) and its turn within the
  // material's rule, on streams of their own so the tiles do not move.
  if (frameInfo && centrePools) {
    // Tones that meet (the tone-ordered pool): near another material a cell
    // takes the centres of its own whose colour lies nearest that one's.
    const classAt = (i, j) => (i < 0 || j < 0 || i >= gridN || j >= gridN ? -1
      : tileClass ? tileClass[cells[j * gridN + i].patch] || 0 : 0);
    const graded = (cls, c) => {
      const pool = centrePools[cls];
      if (!toneGrade || classCount < 2 || !tileClass) return pool;
      for (const step of [1, 2]) {
        const other = partnerClass(classAt, c.i, c.j, step);
        if (other < 0) continue;
        const own = classColours[cls], axis = classColours[other].map((v, q) => v - own[q]);
        const tone = (k) => centres[k].mean.reduce((t, v, q) => t + (v - own[q]) * axis[q], 0);
        return gradedPool(pool, tone, step);
      }
      return pool;
    };
    for (const c of cells) {
      const cls = Math.min(tileClass ? tileClass[c.patch] || 0 : 0, centrePools.length - 1);
      const avoid = [];
      if (c.i > 0) avoid.push(cells[c.j * gridN + c.i - 1].centre);
      if (c.j > 0) avoid.push(cells[(c.j - 1) * gridN + c.i].centre);
      c.centre = pickCentre(graded(cls, c), c.i, c.j, seed, avoid);
      c.turn = pickTurn(frameInfo.turns && frameInfo.turns[cls], c.i, c.j, seed, 2,
                        frameInfo.shape > 2);
    }
    if (decalsOn && centreCopies) {
      // Decals live in centres here: a copy of a pool centre with a decal.
      const undecided = decalTerrain && cellMix && classCount > 1
        ? (k) => (1 - cellMix[k]) / (1 - 1 / classCount) : null;
      const placed = placeDecals((k) => cells[k].centre, gridN, centreCopies,
                                 decalSpacing, seed,
                                 { weight: undecided, groups: decalTerrain });
      for (const p of placed) cells[p.cell].centre = p.tile;
      decalCount = placed.length;
    }
    for (const c of cells) c.tone = toneFor(c.patch, c.centre, c.turn);
  }
  if (ui.decalnote) {
    const copies = decalCopies || centreCopies;
    ui.decalnote.textContent = !copies ? 'this tileset has no decals'
      : `${decalCount} placed, ${new Set([...copies.values()].flat()
          .map(v => v.decal)).size} kinds`;
  }
  tileClassCounts = new Array(classCount).fill(0);
  if (tileClass) for (const c of cells) tileClassCounts[Math.min(classCount - 1, tileClass[c.patch] || 0)]++;
}

// Edge colours as in the GSWT figures: warm for north/south, cool for east/west.
const EDGE_RGB = {
  h: [[0.92, 0.30, 0.30], [0.35, 0.85, 0.40], [0.98, 0.62, 0.18], [0.88, 0.40, 0.85]],
  v: [[0.35, 0.65, 0.95], [0.95, 0.82, 0.30], [0.40, 0.90, 0.88], [0.72, 0.55, 0.98]],
};
const DIAGONAL_RGB = [0.75, 0.75, 0.75];

/** Overlay lines: each cell's four edges in their code colour, and
 *  optionally its diagonals (where the four source patches meet). Shared
 *  edges are drawn by both neighbours, slightly inset; a mismatch shows as
 *  two colours side by side. */
function buildOverlay() {
  const pos = [], rgb = [];
  if (tileSize && cells.length && (showEdges || showDiagonals)) {
    const h = tileSize / 2;
    // Lifted clear of the tile, or the geometry hides the lines.
    const lift = tileSize * 0.2;
    // Placed through the cell's tangent frame, so the lines follow the tile.
    let W = FLAT, O = [0, 0, 0];
    const ga = gridAngle * Math.PI / 180;
    const ca = Math.cos(ga), sa = Math.sin(ga);
    const seg = (u0, v0, u1, v1, c) => {
      for (const [lu, lv] of [[u0, v0], [u1, v1]]) {
        const u = ca * lu - sa * lv, v = sa * lu + ca * lv;
        pos.push(O[0] + W[0] * u + W[3] * v + W[6] * lift,
                 O[1] + W[1] * u + W[4] * v + W[7] * lift,
                 O[2] + W[2] * u + W[5] * v + W[8] * lift);
        rgb.push(c[0], c[1], c[2]);
      }
    };
    for (const cell of cells) {
      W = cell.warp;
      O = [cell.x, cell.y, cell.z];
      if (showEdges) {
        const code = wangCodes ? wangCodes[cell.patch] : [0, 0, 0, 0];
        const k = h * 0.93;
        seg(-k, k, k, k, EDGE_RGB.h[code[0] % 4]);
        seg(k, -k, k, k, EDGE_RGB.v[code[1] % 4]);
        seg(-k, -k, k, -k, EDGE_RGB.h[code[2] % 4]);
        seg(-k, -k, -k, k, EDGE_RGB.v[code[3] % 4]);
      }
      if (showDiagonals) {
        seg(-h, -h, h, h, DIAGONAL_RGB);
        seg(-h, h, h, -h, DIAGONAL_RGB);
      }
    }
  }
  lineCount = pos.length / 3;
  gl.bindVertexArray(lineVAO);
  gl.bindBuffer(gl.ARRAY_BUFFER, linePosBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pos), gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, lineRGBBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(rgb), gl.DYNAMIC_DRAW);
  gl.bindVertexArray(null);
}

/** Where the shader puts a splat (a CPU copy of the vertex shader), so the
 *  seam measurement describes what is on screen. */
function placeSplat(cellX, cellY, local) {
  let ax, ay;
  if (subdiv <= 0) {
    ax = local[0]; ay = local[1];
  } else {
    const sub = tileSize / subdiv;
    const ix = Math.min(Math.max(Math.floor((local[0] + tileSize / 2) / sub), 0),
                        subdiv - 1);
    const iy = Math.min(Math.max(Math.floor((local[1] + tileSize / 2) / sub), 0),
                        subdiv - 1);
    ax = (ix + 0.5) * sub - tileSize / 2;
    ay = (iy + 0.5) * sub - tileSize / 2;
  }
  const wx = cellX + ax, wy = cellY + ay;
  const W = tangentFrame(wx, wy);
  const d = [local[0] - ax, local[1] - ay, local[2]];
  return [wx + W[0] * d[0] + W[3] * d[1] + W[6] * d[2],
          wy + W[1] * d[0] + W[4] * d[1] + W[7] * d[2],
          height(wx, wy) + W[2] * d[0] + W[5] * d[1] + W[8] * d[2]];
}

/** Seam gap: the same point on a shared edge placed from both tiles, over
 *  every interior boundary. Reported in splat widths. */
function measureSeams(samples = 9) {
  if (!tileSize || cells.length < 2) return null;
  const h = tileSize / 2;
  const byKey = new Map();
  for (const c of cells) byKey.set(`${Math.round(c.x / tileSize)},${Math.round(c.y / tileSize)}`, c);

  let worst = 0, total = 0, count = 0;
  for (const c of cells) {
    const gx = Math.round(c.x / tileSize), gy = Math.round(c.y / tileSize);
    for (const [dx, dy] of [[1, 0], [0, 1]]) {
      const nb = byKey.get(`${gx + dx},${gy + dy}`);
      if (!nb) continue;
      for (let i = 0; i < samples; i++) {
        const s = (i / (samples - 1) - 0.5) * 2 * h * 0.98;
        const a = dx ? placeSplat(c.x, c.y, [h, s, 0])
                     : placeSplat(c.x, c.y, [s, h, 0]);
        const b = dx ? placeSplat(nb.x, nb.y, [-h, s, 0])
                     : placeSplat(nb.x, nb.y, [s, -h, 0]);
        const d = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        worst = Math.max(worst, d);
        total += d; count++;
      }
    }
  }
  if (!count) return null;
  return { worst, mean: total / count, count };
}

/** How out of order one cell's splats are for the current view: share of
 *  adjacent pairs whose depth runs the wrong way, and the worst inversion
 *  in world units. The pair share is over-sensitive (neighbouring depths
 *  are nearly equal); read the depth. */
function measureSortError(cell, sample = 4000) {
  if (!splatCount || !cell) return null;
  const p = patches[cell.patch];
  if (!p || !p.count) return null;

  const b = cam.basis();
  const W = cell.warp;
  const idx = sortMode === 'cached' && cacheReady
    ? (() => {
        const f = b.forward;
        const local = [W[0] * f[0] + W[1] * f[1] + W[2] * f[2],
                       W[3] * f[0] + W[4] * f[1] + W[5] * f[2],
                       W[6] * f[0] + W[7] * f[1] + W[8] * f[2]];
        let best = 0, bd = -2;
        for (let k = 0; k < cacheDirs.length; k++) {
          const d = cacheDirs[k][0] * local[0] + cacheDirs[k][1] * local[1]
                  + cacheDirs[k][2] * local[2];
          if (d > bd) { bd = d; best = k; }
        }
        return cacheBuf.subarray(best * cacheStride,
                                 (best + 1) * cacheStride);
      })()
    : lastOrder;
  if (!idx) return null;

  const step = Math.max(1, Math.floor(p.count / sample));
  let inv = 0, pairs = 0, worst = 0, prev = null;
  for (let i = p.start; i < p.start + p.count; i += step) {
    const s = idx[i];
    const x = positionsRef[3 * s], y = positionsRef[3 * s + 1],
          z = positionsRef[3 * s + 2];
    const wx = cell.x + W[0] * x + W[3] * y + W[6] * z;
    const wy = cell.y + W[1] * x + W[4] * y + W[7] * z;
    const wz = cell.z + W[2] * x + W[5] * y + W[8] * z;
    const d = (wx - b.eye[0]) * b.forward[0]
            + (wy - b.eye[1]) * b.forward[1]
            + (wz - b.eye[2]) * b.forward[2];
    if (prev !== null) {
      // Far to near, so depth should fall along the order.
      pairs++;
      if (d > prev) { inv++; worst = Math.max(worst, d - prev); }
    }
    prev = d;
  }
  return pairs ? { frac: inv / pairs, worst, pairs } : null;
}

function reportSortError() {
  if (!ui.sorterr) return;
  const cell = cells.length ? cells[Math.floor(cells.length / 2)] : null;
  const m = measureSortError(cell);
  if (!m) { ui.sorterr.textContent = '—'; return; }
  const unit = splatWidth > 0 ? splatWidth : 1;
  ui.sorterr.textContent =
    `${(m.frac * 100).toFixed(0)}% of pairs, worst ` +
    `${(m.worst / unit).toFixed(1)} splat widths (${m.worst.toFixed(4)})`;
}

function reportSeams() {
  const m = measureSeams();
  if (!m || !ui.seam) return;
  const unit = splatWidth > 0 ? splatWidth : 1;
  ui.seam.textContent =
    `largest ${(m.worst / unit).toFixed(2)}, mean ${(m.mean / unit).toFixed(2)} splat widths ` +
    `(${m.count.toLocaleString()} points)`;
}

/** Load <scene>.atlas.png/.json if there is one; without it the far field
 *  stays off. */
async function loadAtlas(name) {
  atlas = null;
  try {
    const r = await fetch(`./data/${name}.atlas.json`);
    if (!r.ok) { adoptAtlas(bakedAtlas); return; }
    const lay = await r.json();
    const img = new Image();
    img.src = `./data/${name}.atlas.png`;
    await img.decode();
    adoptAtlas({ img, lay });
  } catch (e) { adoptAtlas(bakedAtlas); }
}

/** A top-down atlas made here from the splats, for tilesets that ship
 *  without one (the starter, a dropped file). Rougher than bake_atlas.py:
 *  each splat is a round footprint of its larger axis, averaged by opacity,
 *  from a coarse level of detail. At far-field distances that is enough. */
let bakedAtlas = null;
function bakeAtlas(positions, data, colour) {
  if (!wangCodes || !tileSize || patches.length < 2) return null;
  // With frames, the centres after the tiles (the far field turns them).
  const entries = patches.concat(centres || []);
  const count = entries.length, res = 48;
  const cols = Math.ceil(Math.sqrt(count)), rows = Math.ceil(count / cols);
  const cv = document.createElement('canvas');
  cv.width = cols * res; cv.height = rows * res;
  const acc = new Float32Array(res * res * 4);
  const img = new ImageData(cv.width, cv.height);
  const px = res / tileSize;
  for (let k = 0; k < count; k++) {
    acc.fill(0);
    const p = entries[k];
    // The coarsest level that still has a few thousand splats.
    let [start, n] = p.levels[0];
    for (const [s2, n2] of p.levels) if (n2 >= 3000) { start = s2; n = n2; }
    for (let i = start; i < start + n; i++) {
      const x = (positions[3 * i] / tileSize + 0.5) * res;
      const y = (0.5 - positions[3 * i + 1] / tileSize) * res;
      const d = i * 12;
      const sig = Math.min(4, Math.max(0.6, Math.max(data[d + 4], data[d + 5]) * px));
      const r = Math.ceil(2 * sig), op = data[d + 3];
      const x0 = Math.max(0, Math.floor(x - r)), x1 = Math.min(res - 1, Math.ceil(x + r));
      const y0 = Math.max(0, Math.floor(y - r)), y1 = Math.min(res - 1, Math.ceil(y + r));
      for (let yy = y0; yy <= y1; yy++) {
        for (let xx = x0; xx <= x1; xx++) {
          const dx = xx + 0.5 - x, dy = yy + 0.5 - y;
          const w = op * Math.exp(-(dx * dx + dy * dy) / (2 * sig * sig));
          const o = 4 * (yy * res + xx);
          acc[o] += colour[4 * i] * w; acc[o + 1] += colour[4 * i + 1] * w;
          acc[o + 2] += colour[4 * i + 2] * w; acc[o + 3] += w;
        }
      }
    }
    const q0 = (k % cols) * res, r0 = Math.floor(k / cols) * res;
    const m = k < patches.length ? tileMeans && tileMeans[k] : p.mean;
    const mean = m ? m.map((v) => v * 255) : [128, 128, 128];
    for (let yy = 0; yy < res; yy++) {
      for (let xx = 0; xx < res; xx++) {
        const o = 4 * (yy * res + xx), w = acc[o + 3];
        const t = 4 * ((r0 + yy) * cv.width + q0 + xx);
        for (let c = 0; c < 3; c++) img.data[t + c] = w > 1e-3 ? acc[o + c] / w : mean[c];
        img.data[t + 3] = 255;
      }
    }
  }
  cv.getContext('2d').putImageData(img, 0, 0);
  return { img: cv, lay: { cols, rows, count, res, tiles: patches.length }, baked: true };
}

/** Upload an atlas (a file's or one baked here) and turn the far field on. */
function adoptAtlas(source) {
  if (!source) { atlas = null; return; }
  try {
    const { img, lay } = source;
    // Per entry, a colour gain and the colour far ground fades to. The atlas
    // is seen from straight above, the splats from low down; bake_atlas.py
    // measures each entry from 20 degrees (`seen`), and the gain matches the
    // atlas to it, which removes most of the colour step at the hand-over.
    // Without it, none: the splats' plain mean (every layer, hidden or not)
    // washed far ground out, bush the most.
    // Row 0: gain (0..2 as 0..1). Row 1: what far ground fades to.
    const cv = document.createElement('canvas');
    cv.width = img.width; cv.height = img.height;
    const c2 = cv.getContext('2d', { willReadFrequently: true });
    c2.drawImage(img, 0, 0);
    const px = c2.getImageData(0, 0, img.width, img.height).data;
    const info = new Uint8Array(lay.count * 2 * 4);
    for (let k = 0; k < lay.count; k++) {
      const r0 = Math.floor(k / lay.cols) * lay.res;
      const q0 = (k % lay.cols) * lay.res;
      let sr = 0, sg = 0, sb = 0, sw = 0;
      for (let y = r0; y < r0 + lay.res; y++) {
        for (let x = q0; x < q0 + lay.res; x++) {
          const o = 4 * (y * img.width + x);
          const w = px[o + 3] / 255;
          sr += px[o] * w; sg += px[o + 1] * w; sb += px[o + 2] * w; sw += w;
        }
      }
      const am = sw > 0 ? [sr / sw / 255, sg / sw / 255, sb / sw / 255] : [0.5, 0.5, 0.5];
      const sm = lay.seen && lay.seen[k] ? lay.seen[k] : am;
      for (let c = 0; c < 3; c++) {
        const gain = Math.max(0.5, Math.min(2.0, sm[c] / Math.max(am[c], 1e-3)));
        info[4 * k + c] = Math.round(127.5 * gain);
        info[4 * (lay.count + k) + c] = Math.round(255 * Math.min(1, sm[c]));
      }
      info[4 * k + 3] = 255;
      info[4 * (lay.count + k) + 3] = 255;
    }
    if (!tileInfoTex) tileInfoTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE7);
    gl.bindTexture(gl.TEXTURE_2D, tileInfoTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, lay.count, 2, 0, gl.RGBA,
                  gl.UNSIGNED_BYTE, info);
    if (atlasTex) gl.deleteTexture(atlasTex);
    const tex = atlasTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.generateMipmap(gl.TEXTURE_2D);
    // Stop the mip chain while a tile is still several texels wide, or
    // coarse levels mix unrelated neighbours in the atlas.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL,
                     Math.max(0, Math.floor(Math.log2(lay.res)) - 3));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    atlas = { tex, cols: lay.cols, rows: lay.rows, count: lay.count, res: lay.res,
              img: source.img, baked: !!source.baked };
    farMeshKey = '';
    console.log(`far field: atlas of ${lay.count} tiles`
                + (source.baked ? ', baked in the browser' : ''));
    buildFarField();
  } catch (e) { console.warn('far field off:', e); atlas = null; }
}

/** Upload which tile each cell holds, and build the far mesh over the grid. */
function buildFarField() {
  if (!atlas || !cells.length) return;
  // An atlas baked from an older export would paint the wrong tiles.
  const want = wangCodes ? wangCodes.length + (centres ? centres.length : 0) : atlas.count;
  if (atlas.count !== want) {
    console.warn(`far field off: atlas has ${atlas.count} entries, the tileset `
                 + `${want}. Re-run scripts/bake_atlas.py.`);
    atlas = null;
    return;
  }
  const n = gridN;
  const idx = new Uint8Array(n * n * 4);
  for (const c of cells) {
    const k = 4 * (c.j * n + c.i);
    idx[k] = c.patch & 255;
    idx[k + 1] = (c.patch >> 8) & 255;
    // Rule class, tile class and the tile class it would blend towards, two
    // bits each (the far field's fade and the class view both read these).
    const tcls = tileClass ? tileClass[c.patch] & 3 : 0;
    const acls = c.alt != null ? tileClassFor(c.alt) & 3
      : (classCount === 2 ? 1 - tcls : tcls);
    idx[k + 2] = ((c.cls || 0) & 3) | (tcls << 2) | (acls << 4);
    idx[k + 3] = Math.round(255 * Math.max(0, Math.min(1, c.mix == null ? 1 : c.mix)));
  }
  if (!cellTex) cellTex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE6);
  gl.bindTexture(gl.TEXTURE_2D, cellTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, n, n, 0, gl.RGBA, gl.UNSIGNED_BYTE, idx);
  if (frameInfo && centres) {
    // Each cell's centre (its atlas entry, after the tiles) and turn.
    const cen = new Uint8Array(n * n * 4);
    for (const c of cells) {
      if (c.centre == null || c.centre < 0) continue;
      const k = 4 * (c.j * n + c.i), e = patches.length + c.centre;
      cen[k] = e & 255; cen[k + 1] = (e >> 8) & 255;
      cen[k + 2] = (c.turn || 0) & 255; cen[k + 3] = 255;
    }
    const tone = new Uint8Array(n * n * 4).fill(128);
    for (const c of cells) {
      if (!c.tone) continue;
      const k = 4 * (c.j * n + c.i);
      for (let q = 0; q < 3; q++) {
        tone[k + q] = Math.round(255 * Math.min(1, Math.max(0, 0.5 + 2 * c.tone[q])));
      }
    }
    if (!centreTex) centreTex = gl.createTexture();
    if (!centreToneTex) centreToneTex = gl.createTexture();
    for (const [unit, tex, px] of [[8, centreTex, cen], [9, centreToneTex, tone]]) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, n, n, 0, gl.RGBA, gl.UNSIGNED_BYTE, px);
    }
  }
  buildFarMesh();
}

/** The far mesh: ~2 vertices per tile over the grid. */
let farMeshKey = '';
function buildFarMesh() {
  if (!atlas || !cells.length) return;
  const n = gridN;
  // With water on the mesh is also the water's flat base near the eye
  // (FAR_FRAG), so it follows a river's edge more finely.
  const m = Math.max(waterOn ? 192 : 32, Math.min(512, 2 * n + 1));
  const extent = n * tileSize;
  const key = `${n},${tileSize},${gridAngle},${m}`;
  if (key === farMeshKey && farVAO) return;
  farMeshKey = key;
  const axis = [];
  for (let i = 0; i < m; i++) axis.push((i / (m - 1) - 0.5) * extent);
  const q = axis.length;
  const a = gridAngle * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a);
  const xy = new Float32Array(q * q * 2);
  for (let j = 0; j < q; j++) {
    for (let i = 0; i < q; i++) {
      const lx = axis[i], ly = axis[j];
      xy[2 * (j * q + i)] = ca * lx - sa * ly;
      xy[2 * (j * q + i) + 1] = sa * lx + ca * ly;
    }
  }
  const mm = q;
  const ix = new Uint32Array((mm - 1) * (mm - 1) * 6);
  let t = 0;
  for (let j = 0; j < mm - 1; j++) {
    for (let i = 0; i < mm - 1; i++) {
      const v = j * mm + i;
      ix[t++] = v; ix[t++] = v + 1; ix[t++] = v + mm;
      ix[t++] = v + 1; ix[t++] = v + mm + 1; ix[t++] = v + mm;
    }
  }
  if (!farVAO) farVAO = gl.createVertexArray();
  gl.bindVertexArray(farVAO);
  // Reuse the buffers: this runs again whenever the haze changes.
  if (!buildFarMesh.vb) { buildFarMesh.vb = gl.createBuffer(); buildFarMesh.ib = gl.createBuffer(); }
  const vb = buildFarMesh.vb;
  gl.bindBuffer(gl.ARRAY_BUFFER, vb);
  gl.bufferData(gl.ARRAY_BUFFER, xy, gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(farProg, 'aXY');
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const ib = buildFarMesh.ib;
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, ix, gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  farCount = ix.length;
}

function regenerate() {
  buildRiver();
  buildGrid(); buildOverlay(); reportSeams(); buildFarField();
  // New centres and turns need their own live orders.
  if (frameInfo) lastSortKey = '';
}

/** Rebuild after a control moves. The rebuild blocks the main thread (a
 *  second or more at hundreds of tiles), so the note is painted first, two
 *  frames ahead; moves in the meantime fold into one rebuild. */
let regenPending = false;
function scheduleRegenerate() {
  if (regenPending) return;
  regenPending = true;
  const big = gridN * gridN > 4096;
  if (big) showBusy(`rebuilding ${(gridN * gridN).toLocaleString()} cells…`);
  requestAnimationFrame(() => requestAnimationFrame(() => {
    regenPending = false;
    regenerate();
    hideBusy();
  }));
}

// ================================================================ loading

function load(buffer, manifest) {
  const { n, positions, data, colour } = unpack(buffer, (manifest && manifest.stride) || 32);
  splatCount = n;
  loadedManifest = manifest || null;

  if (manifest && manifest.tiles && manifest.tiles.length) {
    mergeCache.clear(); mergeBytes = 0;   // tile numbers mean other splats now
    patches = manifest.tiles.map(t => ({
      start: t.start, count: t.count,
      levels: t.levels || [[t.start, t.count]],
    }));
    lodLevels = Math.max(1, manifest.lod || 1);
    // A tileset with merged levels says from how far they start (by its
    // splat density); others keep the panel's value.
    if (manifest.lod_base) {
      if (lodBaseKept == null) lodBaseKept = lodBaseUser;
      lodBaseUser = lodBase = manifest.lod_base;
      if (ui.lodbasen) ui.lodbasen.textContent = `${lodBase} tiles`;
    } else if (lodBaseKept != null) {
      lodBaseUser = lodBase = lodBaseKept;
      lodBaseKept = null;
      if (ui.lodbasen) ui.lodbasen.textContent = `${lodBase} tiles`;
    }
    tileSize = manifest.size || 0;
    wangCodes = manifest.wang ? manifest.tiles.map(t => [t.n, t.e, t.s, t.w]) : null;
    plainScene = !manifest.wang;
    tileClass = manifest.tiles.map(t => t.class || 0);
    variantOf = manifest.tiles.map(t => (t.base != null ? t.base : -1));
    decalCopies = decalVariants(manifest.tiles);
    riverRule = manifest.rule === 'river';
    barClass = typeof manifest.bar === 'number' ? manifest.bar : null;
    if (ui.riverrule) ui.riverrule.checked = riverRule;
    patchOverlay = manifest.tiles.some((t) => t.overlay)
      ? manifest.tiles.map((t) => (t.overlay ? { piece: t.overlay.piece, at: t.at || [0, 0],
                                                 rIn: t.overlay.r_in, r: t.overlay.radius } : null))
      : null;
    classCount = Math.max(1, manifest.classes || 1);
    classColours = meanClassColours(colour, manifest.tiles, classCount);
    captureShares = Array.isArray(manifest.shares) ? manifest.shares : null;
    // Out of height order in the capture: 0 strict, 0.5 none (export
    // --height-order); twice that is how far the line wanders. 0 without.
    classWander = typeof manifest.height_order === 'number'
      ? Math.min(1, Math.max(0, 2 * manifest.height_order)) : 0;
    {
      const el = ui.classwander;
      if (el) el.value = String(classWander);
      if (ui.classwandern) ui.classwandern.textContent = `${Math.round(classWander * 100)}%`;
    }
    if (captureShares && classCount >= 2) {
      classBalance = snapShare(captureShares[tileClassFor(0)]);
      const el = ui.classbalance;
      if (el) el.value = String(classBalance);
      if (ui.classbalancen) ui.classbalancen.textContent = `${Math.round(classBalance * 100)}%`;
      // Measured materials meet in the capture's own way: dissolved, not
      // along tile edges.
      classBlend = true;
      if (ui.classblend) ui.classblend.checked = true;
    } else if (classCount >= 2) {
      // Captures put side by side carry no shares (their splat sizes
      // differ too much to measure them), but meet the same way.
      classBlend = true;
      if (ui.classblend) ui.classblend.checked = true;
    }
    // Each tile's own mean, for matching the far field to it.
    tileMeans = manifest.tiles.map((t) =>
      meanClassColours(colour, [{ start: t.start, count: t.count }], 1)[0]);
    classDrift = driftGains(tileMeans, manifest.tiles.map((t) => t.class || 0), classCount);
    frameInfo = manifest.wang && manifest.frames && Array.isArray(manifest.centres)
      && manifest.centres.length ? manifest.frames : null;
    centres = frameInfo ? manifest.centres.map((c) => ({
      start: c.start, count: c.count, levels: c.levels || [[c.start, c.count]],
      class: c.class || 0, base: c.base, decal: c.decal,
      mean: meanClassColours(colour, [{ start: c.start, count: c.count }], 1)[0],
    })) : null;
  } else {
    patches = [{ start: 0, count: n, levels: [[0, n]] }];
    lodLevels = 1;
    tileSize = 0;
    wangCodes = null;
    plainScene = true;
    tileClass = null;
    variantOf = null;
    decalCopies = null;
    patchOverlay = null;
    classColours = meanClassColours(colour, [{ start: 0, count: n }], 1);
    classCount = 1;
    captureShares = null;
    frameInfo = null;
    centres = null;
    classDrift = [];
  }
  // Frames: ground heights (a splat above its ground wins the ring), each
  // material's pool, and the centres' decal copies.
  tileZ0 = patches.map((p) => groundLevel(positions, p.start, p.count));
  if (centres) for (const c of centres) c.z0 = groundLevel(positions, c.start, c.count);
  centrePools = centres ? poolsByClass(centres, classCount) : null;
  centreCopies = centres ? decalVariants(centres) : null;
  tileRing = centreRing = null;
  if (centres) {
    const ring = (p, z0) => ringProfile(positions, colour, p.start, p.count, frameInfo.radius,
                                        frameInfo.ring / 2, z0, frameInfo.prominence[0], 16,
                                        frameInfo.shape || 2);
    tileRing = patches.map((p, k) => ring(p, tileZ0[k]));
    centreRing = centres.map((c) => ring(c, c.z0));
  }
  spunAt = new Map();
  usedPatches = patches.length;
  atlas = null;
  bakedAtlas = null;
  try { bakedAtlas = bakeAtlas(positions, data, colour); }
  catch (e) { console.warn('could not bake an atlas:', e); }

  const w = 2048;
  const h = Math.ceil(n * 3 / w);
  const padded = new Float32Array(w * h * 4);
  padded.set(data.subarray(0, Math.min(data.length, padded.length)));
  // The last tileset's textures go first: each load left ~50 MB of GPU
  // memory behind, and switching tilesets filled the card.
  for (const t of splatTex) if (t) gl.deleteTexture(t);
  splatTex[0] = makeTexture(0, gl.RGBA32F, w, h, gl.RGBA, gl.FLOAT, padded);
  const ch = Math.ceil(n / w);
  const cpad = new Uint8Array(w * ch * 4);
  cpad.set(colour);
  splatTex[1] = makeTexture(1, gl.RGBA8, w, ch, gl.RGBA, gl.UNSIGNED_BYTE, cpad);

  // Median long axis, so seam gaps can be quoted in splat widths.
  const sizes = [];
  for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 20000))) {
    const d = i * 12;
    sizes.push(Math.max(data[d + 4], data[d + 5], data[d + 6]));
  }
  sizes.sort((a, b) => a - b);
  splatWidth = sizes[Math.floor(sizes.length / 2)] || 0;

  identityOrder = new Uint32Array(n);
  for (let i = 0; i < n; i++) identityOrder[i] = i;
  gl.bindVertexArray(splatVAO);
  gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
  gl.bufferData(gl.ARRAY_BUFFER, identityOrder, gl.DYNAMIC_DRAW);
  gl.bindVertexArray(null);
  sortedReady = true;

  // Frame on one tile, or on the interquartile spread for a plain scene.
  const xs = [], ys = [], zs = [];
  const step = Math.max(1, Math.floor(n / 20000));
  for (let i = 0; i < n; i += step) {
    xs.push(positions[3 * i]); ys.push(positions[3 * i + 1]);
    zs.push(positions[3 * i + 2]);
  }
  const q = (arr, p) => {
    const a = arr.slice().sort((u, v) => u - v);
    return a[Math.floor(p * (a.length - 1))];
  };
  cam.target = [(q(xs, .25) + q(xs, .75)) / 2, (q(ys, .25) + q(ys, .75)) / 2,
                (q(zs, .25) + q(zs, .75)) / 2];
  cam.distance = Math.max(
    2.5 * Math.max(q(xs, .75) - q(xs, .25), q(ys, .75) - q(ys, .25)), 0.5);
  if (tileSize) { cam.target = [0, 0, cam.target[2]]; cam.elevation = 20; }

  positionsRef = positions;
  const pos = positions.slice();
  // Each LOD level holds different splats, so each is sorted on its own.
  const ranges = [];
  // Decal overlays share their plain tile's ranges and add their pieces:
  // each range once.
  const seenRange = new Set();
  const addRange = (start, count) => {
    if (seenRange.has(start)) return;
    seenRange.add(start);
    ranges.push({ start, count });
  };
  for (const p of patches.concat(centres || [])) {
    for (const [start, count] of p.levels) addRange(start, count);
  }
  for (const o of patchOverlay || []) {
    if (o) for (const [start, count] of o.piece) addRange(start, count);
  }
  rangeIndex = new Map(ranges.map((r, k) => [r.start, k]));
  liveBehind = cacheBehind = null;
  const posCopy = pos.slice();
  mergeWorker.postMessage({ type: 'init', positions: posCopy.buffer, patches: ranges }, [posCopy.buffer]);
  // Helper shares: consecutive ranges, split by splat count. Only when the
  // ranges lie end to end from 0 (each share is then one block of the order).
  helperChunks = null;
  helperSeq++; helperPending = 0;
  ringSeq++; ringJob = -1; ringFwd.fill(null); ringParts.fill(0); ringSize = 0;
  liveFwd = null;
  const byStart = ranges.map((r, k) => ({ ...r, k })).sort((p, q) => p.start - q.start);
  const tight = byStart.every((r, j) => r.start === (j ? byStart[j - 1].start + byStart[j - 1].count : 0));
  if (tight && byStart.length >= SORT_HELPERS) {
    const total = byStart.reduce((t, r) => t + r.count, 0);
    helperChunks = [];
    let j = 0;
    for (let h = 0; h < SORT_HELPERS; h++) {
      const goal = total * (h + 1) / SORT_HELPERS;
      const share = [];
      while (j < byStart.length && (h === SORT_HELPERS - 1 || !share.length
             || byStart[j].start + byStart[j].count / 2 < goal)) share.push(byStart[j++]);
      const from = share.length ? share[0].start : 0;
      const to = share.length ? share[share.length - 1].start + share[share.length - 1].count : 0;
      helperChunks.push({ from, to, ks: share.map((r) => r.k) });
      const own = pos.slice(3 * from, 3 * to);
      sortHelpers[h].postMessage({ type: 'init', positions: own.buffer,
                                   patches: share.map((r) => ({ start: r.start, count: r.count, local: r.start - from })) },
                                 [own.buffer]);
    }
    ringSize = total;
    gl.bindVertexArray(splatVAO);
    for (const rb of ringBufs) {
      gl.bindBuffer(gl.ARRAY_BUFFER, rb);
      gl.bufferData(gl.ARRAY_BUFFER, ringSize * 4, gl.DYNAMIC_DRAW);
    }
    gl.bindVertexArray(null);
  }
  worker.postMessage({ type: 'init', positions: pos.buffer, patches: ranges },
                     [pos.buffer]);
  cacheReady = false;
  worker.postMessage({ type: 'cache', views: cacheViews });

  if (ui.n) ui.n.textContent = n.toLocaleString() +
    (patches.length > 1 ? ` in ${patches.length}` : '');
  if (ui.grid) ui.grid.disabled = !tileSize;
  if (ui.used) ui.used.disabled = !tileSize || patches.length < 2 || !!wangCodes;
  if (ui.used) ui.used.max = patches.length;
  if (ui.used) ui.used.value = patches.length;
  if (ui.usedn) ui.usedn.textContent = `${patches.length} of ${patches.length}`;
  const copies = variantOf ? variantOf.filter(b => b >= 0).length : 0;
  if (ui.wangnote) ui.wangnote.textContent = wangCodes
    ? `wang, ${patches.length - copies} ${frameInfo ? 'frames' : 'tiles'}, `
      + `${manifest.colours || 2} colours/axis`
      + (copies ? ` + ${copies} with decals` : '')
      + (centres ? ` + ${centres.length} centres` : '')
    : (tileSize ? 'random placement, edges do not match' : 'single scene');

  buildShareSliders();
  regenerate();
  updateShareLabel();
  console.log(`loaded ${n} splats, ${patches.length} patches, ` +
              `tile size ${tileSize}, wang ${!!wangCodes}`);
}

// ================================================================= gizmo

const AXES = [
  { v: [1, 0, 0], label: 'X', colour: '#d9534f' },
  { v: [0, 1, 0], label: 'Y', colour: '#5cb85c' },
  { v: [0, 0, 1], label: 'Z', colour: '#4a90d9' },
];

function drawGizmo(b) {
  const R = 30, cx = 46, cy = 46, arms = [];
  for (const a of AXES) {
    for (const s of [1, -1]) {
      const v = [a.v[0] * s, a.v[1] * s, a.v[2] * s];
      arms.push({
        x: cx + R * (v[0] * b.right[0] + v[1] * b.right[1] + v[2] * b.right[2]),
        y: cy + R * (v[0] * b.down[0] + v[1] * b.down[1] + v[2] * b.down[2]),
        z: v[0] * b.forward[0] + v[1] * b.forward[1] + v[2] * b.forward[2],
        label: s > 0 ? a.label : '', colour: a.colour, positive: s > 0,
      });
    }
  }
  arms.sort((p, q) => q.z - p.z);
  if (!ui.gz) return;
  ui.gz.innerHTML = arms.map(a => {
    const dim = a.positive ? 1 : 0.35;
    return `<line x1="${cx}" y1="${cy}" x2="${a.x.toFixed(1)}" y2="${a.y.toFixed(1)}"` +
      ` stroke="${a.colour}" stroke-width="1.6" opacity="${dim}"/>` +
      `<circle cx="${a.x.toFixed(1)}" cy="${a.y.toFixed(1)}" r="7"` +
      ` fill="${a.positive ? a.colour : '#16191b'}" stroke="${a.colour}"` +
      ` stroke-width="1.4" opacity="${dim}"/>` +
      (a.label ? `<text x="${a.x.toFixed(1)}" y="${(a.y + 3.5).toFixed(1)}"` +
        ` text-anchor="middle" fill="#0b0d0e">${a.label}</text>` : '');
  }).join('');
}

// ============================================================== main loop

// --- pop meter -----------------------------------------------------------
// A pop is a big image change for a small camera move. Read the frame back
// small, diff it with the last one and divide by the angle turned; steady
// while dragging is good, spikes mean something switched. StopThePop's
// metric in miniature (they warp with optical flow; pure rotation needs none).
const POP_W = 192, POP_H = 108;
// From the topological sort each frame: constraints, pairs skipped because
// the eye is in their boundary plane (merging's job), and cycles (cells
// that fell back to the depth key).
let orderStats = { cycles: 0, weak: 0, constrained: 0 };
let cycleWarned = false;

// Selective merging, GSWT 3.4. Where the eye is in the plane of a shared
// boundary no draw order is right, so the pair is drawn as one interleaved
// stream built by the worker. The result arrives a frame late; `mergeKey`
// is what it must match, and if it doesn't the cells are drawn separately.
// Set from the panel (ordering).
let topoOrder = true;
let mergeOn = true;
let groupedOrder = false;      // merged groups ordered as one node (order.js groupedOrder)
let mergeThreshold = 0.5;   // in tiles
let mergeSeq = 0, mergeKey = '', mergeHaveKey = '';
// Merged streams by group (its cells, tiles and slots) with the heading each
// was sorted for. A group draws from its stream while the heading is within
// MERGE_REUSE of it: a stale order costs little (a stream 45 degrees off left
// 0.5% of pixels changed), a missing one draws the group's tiles apart, and
// along the grid line under the eye each tile's fringe then shows over its
// neighbour. Each group keeps streams for up to MERGE_HEADINGS headings, so a
// fast turn finds one near the view at once (GSWT keeps cached orders per
// direction the same way); idle jobs fill headings MERGE_SECTOR apart around
// the view. A best stream past MERGE_FRESH is re-sorted for the view.
const mergeCache = new Map();                   // signature -> [{ fwd, stream }]
const MERGE_REUSE = 0.5, MERGE_FRESH = Math.cos(8 * Math.PI / 180);
const MERGE_HEADINGS = 7, MERGE_SECTOR = 60, MERGE_NEAR = Math.cos(25 * Math.PI / 180);
const MERGE_BYTES = 160e6;                      // cache cap; the oldest groups go first
let mergeBytes = 0;
let mergePending = false, mergePendingSigs = null, mergePendingFwd = null;
let mergeStats = { groups: 0, cells: 0, ms: 0, splats: 0, drawn: 0, missing: 0, turned: 0 };
const mergeBuf = gl.createBuffer();
// Room for eight tile centres, refilled per draw.
const cellXYArr = new Float32Array(2 * MAX_GROUP);

let popPrev = null, popPixels = null;
let popRate = 0, popPeak = 0, popPeakAge = 0;
let popLastAz = 0, popLastEl = 0, popOn = false;

function measurePop() {
  if (!popOn || !splatCount) return;
  if (!popPixels) {
    popPixels = new Uint8Array(POP_W * POP_H * 4);
    popPrev = new Uint8Array(POP_W * POP_H * 4);
  }
  // A small window in the middle: cheap, and skips cells entering at the edges.
  const x = Math.max(0, (canvas.width - POP_W) >> 1);
  const y = Math.max(0, (canvas.height - POP_H) >> 1);
  gl.readPixels(x, y, POP_W, POP_H, gl.RGBA, gl.UNSIGNED_BYTE, popPixels);

  let sum = 0;
  for (let i = 0; i < popPixels.length; i += 4) {
    sum += Math.abs(popPixels[i] - popPrev[i])
         + Math.abs(popPixels[i + 1] - popPrev[i + 1])
         + Math.abs(popPixels[i + 2] - popPrev[i + 2]);
  }
  const diff = sum / (POP_W * POP_H * 3 * 255);

  const moved = Math.hypot(cam.azimuth - popLastAz, cam.elevation - popLastEl);
  popLastAz = cam.azimuth;
  popLastEl = cam.elevation;
  popPrev.set(popPixels);

  // Under 0.05 degrees the camera is still; dividing would make false spikes.
  if (moved < 0.05) return;
  popRate = diff / moved;
  if (popRate > popPeak) { popPeak = popRate; popPeakAge = 0; }
}

// ---------------------------------------------------------------- frames
// Per draw, slot 0 says how its patch sits in the cell; a merged group
// fills one slot per frame or centre.
const slotTurn = new Float32Array(2 * MAX_GROUP);
const slotRole = new Float32Array(MAX_GROUP);
const slotZ0 = new Float32Array(MAX_GROUP);
const slotTone = new Float32Array(21 * MAX_GROUP);
const NO_TONE = new Float32Array(21);
const slotDrift = new Float32Array(6 * MAX_GROUP).fill(1);
// Per material, the gains from its mean tile colour to the two ends of its
// tiles' colour range (along the direction they vary most); a slow field
// mixes between them. Only colours its own tiles have; 0 is off.
let classDrift = [];
let colourDrift = 0;
const ROLE_TILE = 0, ROLE_FRAME = 1, ROLE_CENTRE = 2;

/** Per class, [lo r, g, b, hi r, g, b]: the gains from the class's mean
 *  tile colour to the ends of its tiles' spread along the direction they
 *  differ most (the first principal axis of their means). */
function driftGains(means, classOf, count) {
  const out = [];
  for (let k = 0; k < count; k++) {
    const ms = means.filter((m, i) => m && classOf[i] === k);
    if (ms.length < 2) { out.push(null); continue; }
    const mu = [0, 1, 2].map((c) => ms.reduce((s, m) => s + m[c], 0) / ms.length);
    const d = ms.map((m) => m.map((v, c) => v - mu[c]));
    // Power iteration on the 3 x 3 covariance.
    const cov = [0, 1, 2].map((a) => [0, 1, 2].map((b) => d.reduce((s, x) => s + x[a] * x[b], 0)));
    let ax = [1, 1, 1];
    for (let it = 0; it < 30; it++) {
      const nx = cov.map((row) => row[0] * ax[0] + row[1] * ax[1] + row[2] * ax[2]);
      const n = Math.hypot(...nx) || 1;
      ax = nx.map((v) => v / n);
    }
    const proj = d.map((x) => x[0] * ax[0] + x[1] * ax[1] + x[2] * ax[2]);
    const lo = Math.min(...proj), hi = Math.max(...proj);
    const end = (p) => mu.map((v, c) => Math.max(0.02, v + p * ax[c]) / Math.max(v, 0.02));
    out.push([...end(lo), ...end(hi)]);
  }
  return out;
}

function setSlot(i, turnQ, role, z0, tone, cls = -1) {
  const d = classDrift[cls];
  slotDrift.set(d ? d : [1, 1, 1, 1, 1, 1], 6 * i);
  const [cs, sn] = turnQ ? turnCosSin(turnQ) : [1, 0];
  slotTurn[2 * i] = cs; slotTurn[2 * i + 1] = sn;
  slotRole[i] = role; slotZ0[i] = z0 || 0;
  slotTone.set(tone || NO_TONE, 21 * i);
}
/** Upload the first `n` slots (a single cell's draw uses only slot 0). */
function flushSlots(n = MAX_GROUP) {
  gl.uniform2fv(splatU.slotTurn, slotTurn.subarray(0, 2 * n));
  gl.uniform1fv(splatU.slotRole, slotRole.subarray(0, n));
  gl.uniform1fv(splatU.slotZ0, slotZ0.subarray(0, n));
  gl.uniform3fv(splatU.slotTone, slotTone.subarray(0, 21 * n));
  gl.uniform3fv(splatU.slotDrift, slotDrift.subarray(0, 6 * n));
}

/** The tint for centre `k` turned by `turnQ` in a cell holding tile `t`. */
function toneFor(t, k, turnQ) {
  return tileRing && centreRing && k >= 0 ? toneTerms(tileRing[t], centreRing[k], turnQ) : null;
}

/** A centre for a cell drawn in another class (the class blend), and its
 *  turn: seeded per cell, like its own. */
function partnerCentre(c, cls) {
  const pool = centrePools[Math.min(cls, centrePools.length - 1)];
  if (!pool || !pool.length) return [-1, 0];
  const k = pool[Math.floor(cellHash(c.i, c.j, seed, 5) * pool.length) % pool.length];
  return [k, pickTurn(frameInfo.turns && frameInfo.turns[cls], c.i, c.j, seed, 9,
                      frameInfo.shape > 2)];
}

/** The cached order nearest the view direction as a cell's patch sees it:
 *  in the cell's frame, and turned back by the patch's own turn. */
function nearestCached(c, forward, turnQ) {
  const W = c.warp, f = forward;
  let l = [W[0] * f[0] + W[1] * f[1] + W[2] * f[2],
           W[3] * f[0] + W[4] * f[1] + W[5] * f[2],
           W[6] * f[0] + W[7] * f[1] + W[8] * f[2]];
  if (turnQ) {
    const [cs, sn] = turnCosSin(turnQ);
    l = [cs * l[0] + sn * l[1], -sn * l[0] + cs * l[1], l[2]];
  }
  let best = 0, bestDot = -2;
  for (let k = 0; k < cacheDirs.length; k++) {
    const d = cacheDirs[k][0] * l[0] + cacheDirs[k][1] * l[1] + cacheDirs[k][2] * l[2];
    if (d > bestDot) { bestDot = d; best = k; }
  }
  return best;
}

/** Bind the sorted order for one level range of a cell's patch: returns
 *  its byte offset and how many of its splats come before the patch's
 *  middle (null when not known). A turned centre takes its own live order
 *  when the worker made one, else the cached order nearest its turned view. */
function bindOrder(range, turnQ, c, forward) {
  const start = range[0], k = rangeIndex.get(start);
  if (turnQ) {
    const at = sortMode === 'live' ? spunAt.get(`${start},${turnQ}`) : undefined;
    if (at != null) { gl.bindBuffer(gl.ARRAY_BUFFER, spunBuf); return { at: at * 4, behind: null }; }
  }
  if ((turnQ || sortMode === 'cached') && cacheReady) {
    gl.bindBuffer(gl.ARRAY_BUFFER, cacheIndexBuf);
    const d = nearestCached(c, forward, turnQ);
    return { at: (d * cacheStride + start) * 4,
             behind: cacheBehind && k != null ? cacheBehind[d * rangeIndex.size + k] : null };
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, orderBuf || indexBuf);
  return { at: start * 4, behind: liveBehind && k != null ? liveBehind[k] : null };
}

/** Draw one level range of a cell's patch in the order bound for it:
 *  whole, or with `span` 'back' the splats before its middle in that order
 *  and 'front' the rest (a frame's two parts; without a split known, the
 *  back part is all of it). Returns how many were drawn. */
function drawSpan(c, range, turnQ, span, forward) {
  const o = bindOrder(range, turnQ, c, forward);
  let from = 0, n = range[1];
  if (span) {
    const k = o.behind == null ? (span === 'back' ? n : 0) : Math.min(o.behind, n);
    if (span === 'back') n = k; else { from = k; n -= k; }
  }
  if (n <= 0) return 0;
  gl.vertexAttribIPointer(aIndex, 1, gl.UNSIGNED_INT, 0, o.at + 4 * from);
  gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n);
  drawnSplats += n; drawCalls++;
  return n;
}

/** The turned centres near enough for their order to show (the first two
 *  levels), as one live sort each: [{start, count, c, s, key}]. */
function turnedNear(b) {
  if (!frameInfo || !centres || !cells.length) return [];
  const out = new Map();
  const reach = lodBase * 2 * (tileSize || 1);
  const ga = gridAngle * Math.PI / 180;
  for (const c of cells) {
    if (c.centre == null || c.centre < 0 || !c.turn) continue;
    const dx = c.x - b.eye[0], dy = c.y - b.eye[1], dz = c.z - b.eye[2];
    if (dx * b.forward[0] + dy * b.forward[1] + dz * b.forward[2] < -2 * tileSize) continue;
    if (Math.hypot(dx, dy, dz) > reach) continue;
    const [cs, sn] = turnCosSin(c.turn);
    // The live sort runs along the world direction; the grid turns too.
    const ca = Math.cos(ga) * cs - Math.sin(ga) * sn, sa = Math.sin(ga) * cs + Math.cos(ga) * sn;
    for (const [start, count] of centres[c.centre].levels.slice(0, 2)) {
      const key = `${start},${c.turn}`;
      if (count && !out.has(key)) out.set(key, { start, count, c: ca, s: sa, key });
    }
    if (out.size >= 256) break;
  }
  return [...out.values()];
}

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  // Rendered smaller under auto quality; the browser scales it up.
  const k = autoQuality ? renderScale : 1;
  const w = Math.max(1, Math.floor(canvas.clientWidth * dpr * k));
  const h = Math.max(1, Math.floor(canvas.clientHeight * dpr * k));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
    gl.viewport(0, 0, w, h);
  }
}

/** The render loop. An error in one frame is logged once and the loop goes
 *  on, so a state that is briefly inconsistent cannot stop the viewer. */
let frameError = '';
function frame() {
  try {
    drawFrame();
  } catch (e) {
    if (String(e) !== frameError) { frameError = String(e); console.error(e); }
    requestAnimationFrame(frame);
  }
}

function drawFrame() {
  // Draw only when something changed, and at most MAX_FPS times a second.
  // Drawn flat out (120 fps, millions of splats), a large world took the
  // whole GPU and a video in another window stuttered and stopped.
  {
    const t = performance.now();
    const bb = cam.basis();
    const key = [...bb.forward, ...bb.eye, canvas.clientWidth, canvas.clientHeight, cam.fov]
      .map((v) => v.toFixed(4)).join(',');
    {
      // Turn rate about the vertical, degrees per second, smoothed.
      const h = Math.atan2(bb.forward[1], bb.forward[0]) * 180 / Math.PI;
      if (lastYaw != null && t > lastYawT) {
        let d = h - lastYaw;
        if (d > 180) d -= 360;
        if (d < -180) d += 360;
        const rate = d * 1000 / (t - lastYawT);
        yawRate = 0.5 * yawRate + 0.5 * rate;
      }
      lastYaw = h; lastYawT = t;
    }
    const busy = orbiting || (film && film.playing()) || held.size || zoomGoal || camVel[0] || camVel[1] || camVel[2]
      || capture.active || bench.active || sortPending || regenPending || shotWanted
      || (mergeOn && mergeKey !== '' && mergeHaveKey !== mergeKey);
    if (busy || key !== lastFrameKey) { lastFrameKey = key; wakeFrames = WAKE_FRAMES; }
    if (wakeFrames <= 0 && flowing() && t - lastDrawAt >= 1000 / FLOW_FPS - 1) wakeFrames = 1;
    const measuring = capture.active || bench.active;
    const due = measuring || t - lastDrawAt >= 1000 / MAX_FPS - 1;
    const heartbeat = t - lastDrawAt > 1000;         // anything a poke missed
    if (!due || (wakeFrames <= 0 && !heartbeat)) { requestAnimationFrame(frame); return; }
    wakeFrames--;
    lastDrawAt = t;
    // Frame time for the benchmark and auto quality (performance.now, not the
    // rAF timestamp, so throttling reads the same).
    if (benchPrev) benchDt = t - benchPrev;
    benchPrev = t;
  }
  // Nothing is seen under the intro card, so draw nothing: its first
  // seconds are the download and the first sorts, and a laptop needs them.
  if (introOpen()) { requestAnimationFrame(frame); return; }
  resize();
  const b = cam.basis();
  const fy = (canvas.height / 2) / Math.tan(cam.fov * Math.PI / 360);
  const viewMat = new Float32Array([
    b.right[0], b.down[0], b.forward[0],
    b.right[1], b.down[1], b.forward[1],
    b.right[2], b.down[2], b.forward[2],
  ]);

  if (splatCount && sortingEnabled && !sortPending) {
    const key = [b.forward, b.eye].flat().map(v => v.toFixed(2)).join(',');
    if (key !== lastSortKey) {
      lastSortKey = key;
      sortPending = true;
      // Sort for the heading the view will have when the order arrives: the
      // turn rate times the last sort's time, at most 40 degrees ahead.
      const ahead = Math.max(-40, Math.min(40, yawRate * (sortMs + 30) / 1000)) * Math.PI / 180;
      const ca = Math.cos(ahead), sa = Math.sin(ahead);
      const f = b.forward;
      const fwd = [ca * f[0] - sa * f[1], sa * f[0] + ca * f[1], f[2]];
      const spins = sortMode === 'live' ? turnedNear(b) : [];
      liveFwdPending = fwd;
      if (helperChunks && !spins.length) {
        helperSeq++;
        helperPending = SORT_HELPERS;
        helperMs = 0;
        sortHelpers.forEach((h, part) => h.postMessage({ type: 'sort', forward: fwd, eye: b.eye, part, seq: helperSeq }));
      } else {
        worker.postMessage({ type: 'sort', forward: fwd, eye: b.eye, spins });
      }
    }
  }
  if (splatCount && sortingEnabled && helperChunks && ringSize && !sortPending && ringJob < 0
      && Math.abs(yawRate) < 30) {
    // Idle: refresh the ring slot nearest the view whose heading (fixed
    // azimuth, the view's elevation) has drifted past 8 degrees.
    const f = b.forward, hm = Math.hypot(f[0], f[1]);
    const yaw = Math.atan2(f[1], f[0]);
    let pick = -1, pickGap = Infinity, pickFwd = null;
    for (let k = 0; k < RING; k++) {
      const az = k * 2 * Math.PI / RING;
      const want = [hm * Math.cos(az), hm * Math.sin(az), f[2]];
      const have = ringFwd[k];
      if (have && have[0] * want[0] + have[1] * want[1] + have[2] * want[2] > Math.cos(8 * Math.PI / 180)) continue;
      let gap = Math.abs(az - yaw) % (2 * Math.PI);
      if (gap > Math.PI) gap = 2 * Math.PI - gap;
      if (gap < pickGap) { pickGap = gap; pick = k; pickFwd = want; }
    }
    if (pick >= 0) {
      ringJob = pick; ringJobFwd = pickFwd; ringParts[pick] = 0; ringFwd[pick] = null; ringSeq++;
      sortHelpers.forEach((h, part) => h.postMessage({ type: 'sort', forward: pickFwd, eye: b.eye, part,
                                                       seq: ringSeq, ring: pick }));
    }
  }
  {
    // The live-path order for this frame: the live one or the nearest ring
    // slot, whichever was sorted nearer the view.
    const f = b.forward;
    const dot = (a) => (a ? a[0] * f[0] + a[1] * f[1] + a[2] * f[2] : -2);
    orderBuf = indexBuf;
    let best = liveFwd ? dot(liveFwd) : -2;
    for (let k = 0; k < RING; k++) {
      const d = dot(ringFwd[k]);
      if (d > best + 1e-6) { best = d; orderBuf = ringBufs[k]; }
    }
  }
  {
  }

  const S = SKIES[sky] || SKIES.none;
  // Debug views show the data as it is: no glow, mist or grade over them.
  const debugView = showLodColours || showIdentity || showClasses || showTints || showEdges;
  gl.clear(gl.COLOR_BUFFER_BIT);
  if (sky !== 'none') {
    gl.useProgram(skyProg);
    gl.bindVertexArray(skyVAO);
    gl.uniformMatrix3fv(skyU.view, false, viewMat);
    gl.uniform2f(skyU.focal, fy, fy);
    gl.uniform2f(skyU.viewport, canvas.width, canvas.height);
    gl.uniform3fv(skyU.skyTop, new Float32Array(S.top));
    gl.uniform3fv(skyU.skyHorizon, new Float32Array(S.horizon));
    gl.uniform3fv(skyU.ground, new Float32Array(S.ground));
    gl.uniform3fv(skyU.eye, new Float32Array(b.eye));
    atmosUniforms(skyU, S, debugView);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  drawnSplats = 0; drawCalls = 0;
  // The far field is the next LOD level, fading in over the same last fifth
  // of an octave as lodFor(). It starts one level before the coarsest (16
  // tiles at the default instead of 32): the coarsest splat level added
  // little over the atlas and cost most of the frame. At most three octaves
  // out: a tileset with more levels (seven, 64 splats a tile at the last)
  // kept splats to 150 m and drew every cell of a large world.
  const farStart = (lodOn && lodLevels > 1
    ? lodBase * Math.pow(2, Math.min(FAR_OCTAVES, Math.max(0, lodLevels - 2)))
    : farStartTiles) * (tileSize || 1);
  const farBand = farStart * (1 - Math.pow(2, -0.2));
  const farLive = farOn && atlas && farCount && cellTex && tileSize;
  if (farLive) {
    // Far ground first, with depth; splats on top without depth as before.
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.useProgram(farProg);
    gl.bindVertexArray(farVAO);
    gl.uniformMatrix3fv(farU.view, false, viewMat);
    gl.uniform3fv(farU.eye, new Float32Array(b.eye));
    gl.uniform2f(farU.focal, fy, fy);
    gl.uniform2f(farU.viewport, canvas.width, canvas.height);
    gl.uniform1f(farU.relief, amplitude());
    gl.uniform1f(farU.wave, Math.max(reliefScale * (tileSize || 1), 0.01));
    gl.uniform1i(farU.field, 3);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, field ? fieldTex : null);
    gl.uniform1f(farU.hasField, field ? 1.0 : 0.0);
    if (field) {
      gl.uniform2f(farU.fieldSize, field.w, field.h);
      gl.uniform1f(farU.fieldExtent, field.extent);
    }
    gl.uniform1i(farU.atlas, 5);
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, atlas.tex);
    gl.uniform1i(farU.cells, 6);
    gl.activeTexture(gl.TEXTURE6);
    gl.bindTexture(gl.TEXTURE_2D, cellTex);
    gl.uniform2f(farU.atlasGrid, atlas.cols, atlas.rows);
    gl.uniform1f(farU.tileSize, tileSize);
    gl.uniform1f(farU.gridN, cellsN);
    const ga = gridAngle * Math.PI / 180;
    gl.uniform2f(farU.rot, Math.cos(ga), Math.sin(ga));
    gl.uniform1f(farU.farStart, farStart);
    gl.uniform1f(farU.farBand, farBand);
    gl.uniform3fv(farU.fogColour, new Float32Array(S.horizon));
    gl.uniform1f(farU.fogDensity, fogDensity(S));
    gl.uniform1f(farU.exposure, exposure);
    gl.uniform1f(farU.shade, groundShade);
    gl.uniform1f(farU.saturation, saturation);
    gl.uniform1f(farU.macro, macro);
    gl.uniform1f(farU.blend, classBlend && classCount > 1 ? 1.0 : 0.0);
    gl.uniform1f(farU.spots, blendSpots ? 1.0 : 0.0);
    gl.uniform1f(farU.mixWarp, SPOT_WARP);
    gl.uniform1i(farU.open, 4);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, field && openTex ? openTex : null);
    gl.uniform1f(farU.hasOpen, field && openTex ? 1.0 : 0.0);
    setWater(farU);
    gl.uniform1i(farU.debug, showClasses && classCount > 1 ? 2
      : showIdentity ? 1 : showLodColours ? 3 : 0);
    gl.uniform3fv(farU.classRGB, new Float32Array(CLASS_RGB.flat()));
    gl.uniform1f(farU.detail, amplitude() > 0 ? detailAmp() : 0);
    gl.uniform1f(farU.detailFreq, detailFreq());
    gl.uniform1i(farU.tileInfo, 7);
    gl.activeTexture(gl.TEXTURE7);
    gl.bindTexture(gl.TEXTURE_2D, tileInfoTex);
    gl.uniform1f(farU.hasTileInfo, tileInfoTex ? 1.0 : 0.0);
    const ringed = !!(frameInfo && centres && centreTex);
    gl.uniform1i(farU.centres, 8);
    gl.activeTexture(gl.TEXTURE8);
    gl.bindTexture(gl.TEXTURE_2D, ringed ? centreTex : null);
    gl.uniform1i(farU.centreTone, 9);
    gl.activeTexture(gl.TEXTURE9);
    gl.bindTexture(gl.TEXTURE_2D, ringed ? centreToneTex : null);
    gl.uniform1f(farU.hasCentres, ringed ? 1.0 : 0.0);
    gl.uniform3f(farU.ringFar, ringed ? frameInfo.radius : 0, ringed ? frameInfo.wobble : 0,
                 ringed ? frameInfo.shape || 2 : 2);
    gl.uniform4fv(farU.rise, riseCircle);
    gl.uniform1f(farU.riseSquare, riseSquare ? 1 : 0);
    atmosUniforms(farU, S, debugView);
    // While the world grows or shrinks by rings (the film's square rise),
    // the distant land starts where the landed cells end.
    const half = riseSquare && riseCircle[3] > 0
      ? Math.min(gridN * tileSize / 2, Math.max(tileSize, riseCircle[2] - 0.5 * riseCircle[3]))
      : gridN * tileSize / 2;
    const rangeH = ranges * RANGES_RELIEF * Math.max(Math.abs(amplitude()), 0.2 * tileSize);
    const rangeW = 0.5 * Math.max(reliefScale, 4) * tileSize;
    gl.uniform4f(farU.horizon, half, rangeH, rangeW, 0);
    gl.drawElements(gl.TRIANGLES, farCount, gl.UNSIGNED_INT, 0);
    drawCalls++;
    if (ranges > 0) {
      buildHorizonMesh();
      gl.bindVertexArray(horizonVAO);
      gl.uniform4f(farU.horizon, half, rangeH, rangeW, 1);
      gl.drawElements(gl.TRIANGLES, horizonCount, gl.UNSIGNED_INT, 0);
      drawCalls++;
    }
    gl.bindVertexArray(null);
    gl.disable(gl.DEPTH_TEST);
  }

  if (splatCount && sortedReady) {
    gl.useProgram(splatProg);
    gl.bindVertexArray(splatVAO);
    gl.uniform1i(splatU.data, 0);
    gl.uniform1i(splatU.colour, 1);
    // The height field, so geometry is displaced by the surface the rule read.
    gl.uniform1i(splatU.field, 3);
    if (field && fieldTex) {
      gl.activeTexture(gl.TEXTURE3);
      gl.bindTexture(gl.TEXTURE_2D, fieldTex);
      gl.uniform2f(splatU.fieldSize, field.w, field.h);
      gl.uniform1f(splatU.fieldExtent, field.extent);
      gl.uniform1f(splatU.hasField, 1.0);
    } else {
      gl.uniform1f(splatU.hasField, 0.0);
    }
    gl.uniform1f(splatU.waterView, waterView && riverOn() ? 1.0 : 0.0);
    setWater(splatU);
    gl.uniform1i(splatU.open, 4);
    if (field && openTex) {
      gl.activeTexture(gl.TEXTURE4);
      gl.bindTexture(gl.TEXTURE_2D, openTex);
      gl.uniform1f(splatU.hasOpen, 1.0);
    } else {
      gl.uniform1f(splatU.hasOpen, 0.0);
    }
    gl.uniform1f(splatU.shade, groundShade);
    gl.uniform1f(splatU.detail, amplitude() > 0 ? detailAmp() : 0);
    gl.uniform1f(splatU.detailFreq, detailFreq());
    gl.uniform1f(splatU.farOn, farLive ? 1.0 : 0.0);
    gl.uniform1f(splatU.farStart, farStart);
    gl.uniform1f(splatU.farBand, farBand);
    gl.uniform1f(splatU.exposure, exposure);
    gl.uniform1f(splatU.saturation, saturation);
    gl.uniform1f(splatU.macro, tileSize ? macro : 0);
    gl.uniformMatrix3fv(splatU.view, false, viewMat);
    gl.uniform3fv(splatU.eye, new Float32Array(b.eye));
    gl.uniform2f(splatU.focal, fy, fy);
    gl.uniform2f(splatU.viewport, canvas.width, canvas.height);
    gl.uniform1f(splatU.gain, gain);
    gl.uniform1f(splatU.near, 0.05);
    gl.uniform1f(splatU.relief, amplitude());
    gl.uniform1f(splatU.wave, Math.max(reliefScale * (tileSize || 1), 0.01));
    gl.uniform1i(splatU.subdiv, subdiv);
    gl.uniform1f(splatU.tileSize, tileSize);
    gl.uniform2f(splatU.gridRot, Math.cos(gridAngle * Math.PI / 180),
                 Math.sin(gridAngle * Math.PI / 180));
    gl.uniform3fv(splatU.fogColour, new Float32Array(S.horizon));
    gl.uniform1f(splatU.fogDensity, fogDensity(S));

    // Cells drawn far to near and composited 'over'. Splats are sorted
    // within a patch, never across cells - the source of boundary artifacts.
    const visible = [];
    const half = (tileSize || 1) / 2;
    const tanHalf = Math.tan(cam.fov * Math.PI / 360);
    const aspect = canvas.width / canvas.height;
    const corners = [[-half, -half], [half, -half], [-half, half], [half, half]];
    // A corner lies at most this much nearer than the centre along the view.
    const cornerReach = half * (Math.abs(b.forward[0]) + Math.abs(b.forward[1]));

    for (let ci = 0; ci < cells.length; ci++) {
      const c = cells[ci];
      const dx = c.x - b.eye[0], dy = c.y - b.eye[1], dz = c.z - b.eye[2];
      const z = dx * b.forward[0] + dy * b.forward[1] + dz * b.forward[2];
      if (z < -tileSize * 2.0) continue;
      // Not yet reached by the demo's growing world (the shader hides it too).
      if (riseCircle[3] > 0 && riseDist(c.x, c.y) >= riseCircle[2]) continue;
      // Past the far field even at its nearest corner: atlas, as below.
      if (farLive && z - cornerReach > farStart) continue;
      const dist = Math.hypot(dx, dy, dz);
      const sx = dx * b.right[0] + dy * b.right[1] + dz * b.right[2];
      const sy = dx * b.down[0] + dy * b.down[1] + dz * b.down[2];
      // Culling by the centre, so the margin covers half a tile, the
      // overhang and the relief; too tight and a half-visible cell drops
      // out and its neighbour's overhang fills in (reads as a tile change).
      // Separate x and y margins, since the frustum is wider than tall.
      const reach = tileSize * 2.0 + Math.abs(amplitude());
      const padY = reach + Math.max(z, 0.01) * tanHalf;
      const padX = reach + Math.max(z, 0.01) * tanHalf * aspect;
      if (Math.abs(sx) > padX || Math.abs(sy) > padY) continue;

      // Depth key from the nearest corner, not the centre: cells overlap at
      // their shared edge. Only a tiebreak now (see order.js below).
      let near = Infinity;
      for (const [ox, oy] of corners) {
        const kx = dx + ox, ky = dy + oy;
        const kz = kx * b.forward[0] + ky * b.forward[1] + dz * b.forward[2];
        if (kz < near) near = kz;
      }
      // Past the far field's band this cell is atlas.
      if (farLive && near > farStart) continue;
      visible.push({ c, near, dist, side: Math.abs(sx) + Math.abs(sy) });
    }

    // Far to near. Ranking by `near` alone is degenerate when the view lines
    // up with a grid axis: a whole row ties and flips at once (27/54/27
    // reversed pairs at 0/89/90 degrees, probe_order.py). order.js sorts by
    // which side of each shared boundary plane the eye is on instead, and
    // `near` only breaks ties between cells that cannot overlap. What is
    // left is a boundary the camera really crosses - merging handles that.
    // 'depth' in the panel's tile order restores the old ranking for comparison.
    if (!topoOrder) {
      visible.sort((p, q) => (q.near - p.near) || (q.side - p.side));
    } else {
      const vc = visible.map((v) => v.c);
      const r = drawOrder(vc, b.eye, { key: visible.map((v) => v.near) });
      orderStats = { cycles: r.cycles, weak: r.weak.length,
                     constrained: r.constrained };
      const reordered = r.order.map((k) => visible[k]);
      visible.length = 0;
      for (const v of reordered) visible.push(v);
    }

    // --- selective merging ------------------------------------------
    // Group cells by how near the eye is to their shared boundary plane
    // (threshold in tiles) and ask the worker for one stream per group.
    // Merged cells draw at level 0 without cross-fade: merging only happens
    // right at the camera, where level 0 is used anyway.
    // Each cell's own-class weight, for the blend's corners. A corner is
    // shared by four cells, so its weight is theirs averaged: neighbours
    // agree along the edge and the boundary does not step by tiles.
    let blendGrid = null;
    const spotMode = classBlend && blendSpots && classCount > 1 && !!tileClass && cells.length > 1;
    if (classBlend && !blendSpots && classCount > 1 && tileClass && cells.length) {
      blendGrid = new Map();
      for (const c of cells) {
        const w = c.mix == null || c.mix >= blendWidth ? 1
          : 0.5 + 0.5 * (c.mix / Math.max(blendWidth, 1e-6));
        blendGrid.set(c.j * cellsN + c.i, { cls: c.cls, alt: c.alt, w });
      }
    }
    const weightOf = (cls, n) => (n.cls === cls ? n.w : (n.alt === cls || n.alt == null ? 1 - n.w : 0));
    const blendCorners = (c) => [[0, 0], [1, 0], [0, 1], [1, 1]].map(([di, dj]) => {
      let sum = 0, k = 0;
      for (const [ox, oy] of [[-1, -1], [0, -1], [-1, 0], [0, 0]]) {
        const i = c.i + di + ox, j = c.j + dj + oy;
        if (i < 0 || j < 0 || i >= cellsN || j >= cellsN) continue;
        const n = blendGrid.get(j * cellsN + i);
        if (n) { sum += weightOf(c.cls, n); k++; }
      }
      return k ? sum / k : 1;
    });
    // By spot: a cell next to other materials draws those whose field
    // reaches into it (spotDraws in frames.js), each with its own field; kept
    // on the cell until the grid is rebuilt.
    const classAt = (i, j) => (i < 0 || j < 0 || i >= cellsN || j >= cellsN ? -1
      : tileClass[cells[j * cellsN + i].patch]);
    const padded = (f) => { const a = new Float32Array(52); a.set(f); return a; };
    const spotOf = (c) => {
      if (c.spot === undefined) {
        const draws = spotDraws(classAt, c.i, c.j, SPOT_WARP, spotSharpness(spotWidth));
        const mates = draws.slice(1).map((d) => ({ tile: tileOfClass(c.patch, d.cls),
                                                    field: padded(d.field) }))
          .filter((m) => m.tile >= 0);
        c.spot = mates.length ? { field: padded(draws[0].field), mates } : null;
      }
      return c.spot;
    };
    // A cell dissolving into the other class draws twice with its own
    // uniforms, so it stays out of merged streams.
    const blends = (c) => (patchOverlay && patchOverlay[c.patch]) || (spotMode ? !!spotOf(c)
      : !!blendGrid && c.mix != null && Math.min(...blendCorners(c)) < 0.999);

    // With frames each cell draws a frame and a centre; a merged group gives
    // each its own slot, so it holds half as many cells.
    const framed = !!(frameInfo && centres);
    const hasCentre = (c) => framed && c.centre != null && c.centre >= 0;
    const cellParts = (c) => (hasCentre(c)
      ? [{ range: patches[c.patch].levels[0], turn: 0, role: ROLE_FRAME, z0: tileZ0[c.patch] },
         { range: centres[c.centre].levels[0], turn: c.turn, role: ROLE_CENTRE,
           z0: centres[c.centre].z0, tone: c.tone }]
      : [{ range: patches[c.patch].levels[0], turn: 0, role: ROLE_TILE, z0: tileZ0[c.patch] }]);
    for (let i = 0; i < MAX_GROUP; i++) setSlot(i, 0, ROLE_TILE, 0);
    gl.uniform4f(splatU.ring, framed ? frameInfo.radius : 0, framed ? frameInfo.ring / 2 : 0,
                 framed ? frameInfo.wobble : 0, framed ? frameInfo.shape || 2 : 2);
    const prom = framed && frameInfo.prominence ? frameInfo.prominence
      : [0.02 * (tileSize || 1.5), 0.08 * (tileSize || 1.5)];
    gl.uniform2f(splatU.prom, prom[0], prom[1]);
    gl.uniform1f(splatU.spots, blendSpots ? 1.0 : 0.0);
    gl.uniform1f(splatU.mixWarp, SPOT_WARP);
    gl.uniform1f(splatU.spotSharp, spotSharpness(spotWidth));
    gl.uniform1f(splatU.drift, colourDrift);
    gl.uniform4fv(splatU.rise, riseCircle);
    gl.uniform1f(splatU.riseSquare, riseSquare ? 1 : 0);
    atmosUniforms(splatU, S, debugView);
    gl.uniform1f(splatU.riseDrop, riseDrop);

    let groupOf = null, groups = null, groupParts = null, sigs = null;
    const bestStream = (sig, dir = b.forward) => {
      const list = mergeCache.get(sig);
      if (!list) return null;
      let best = null, bd = -2;
      for (const e of list) {
        const d = e.fwd[0] * dir[0] + e.fwd[1] * dir[1] + e.fwd[2] * dir[2];
        if (d > bd) { bd = d; best = e; }
      }
      return best && { stream: best.stream, fwd: best.fwd, dot: bd };
    };
    if (mergeOn && tileSize && visible.length > 1) {
      let vc = visible.map((v) => v.c);
      const ar = gridAngle * Math.PI / 180;
      const ga = { c: Math.cos(ar), s: Math.sin(ar) };
      // Groups over the whole grid, not just what is on screen: merging
      // depends on where the eye is against the grid lines, so turning the
      // view then leaves the groups (and their cached streams) as they are,
      // and only stepping across a line makes new ones. Turning used to bring
      // new cells into view every frame, each a new group without a stream.
      const mergeOpts = { threshold: tileSize * mergeThreshold, cap: framed ? MAX_GROUP / 2 : MAX_GROUP };
      const blendAll = cells.map(blends);
      const world = splitOut(mergeGroups(cells, b.eye, mergeOpts), blendAll);
      const where = new Map();
      cells.forEach((c, w) => where.set(c, w));
      const localOf = new Map();                  // world group -> local group
      const localGroups = [], localWorld = [];
      const localOfCell = new Int32Array(vc.length);
      vc.forEach((c, k) => {
        const wg = world.groupOf[where.get(c)];
        const many = world.groups[wg].length > 1;
        let lg = many ? localOf.get(wg) : undefined;
        if (lg === undefined) {
          lg = localGroups.length;
          localGroups.push([]);
          localWorld.push(many ? wg : -1);
          if (many) localOf.set(wg, lg);
        }
        localGroups[lg].push(k);
        localOfCell[k] = lg;
      });
      let g = { groupOf: localOfCell, groups: localGroups };
      // A group is drawn whole at its first cell, so with groupedOrder its
      // cells are made adjacent in the order: ordered cell by cell, a cell
      // that belongs between two members is drawn after the whole group and
      // covers the nearer member (a stripe along the grid line under the eye).
      if (groupedOrder && topoOrder && g.groups.some((x) => x.length > 1)) {
        const r = drawOrder(vc, b.eye, { key: visible.map((v) => v.near), groupOf: g.groupOf });
        const pos = new Int32Array(r.order.length);
        r.order.forEach((k, p) => { pos[k] = p; });
        const re = r.order.map((k) => visible[k]);
        visible.length = 0;
        for (const v of re) visible.push(v);
        g = { ...g, groupOf: Int32Array.from(r.order, (k) => g.groupOf[k]),
              groups: g.groups.map((x) => x.map((k) => pos[k]).sort((a, c) => a - c)) };
        vc = visible.map((v) => v.c);
        orderStats = { ...orderStats, cycles: r.cycles };
      }
      const merged = (gi) => localWorld[gi] >= 0;
      const multi = g.groups.filter((x, gi) => merged(gi));
      if (multi.length) {
        groupOf = g.groupOf; groups = g.groups;
        // One entry per slot: its cell (on screen or not) and what it draws.
        // Slots go by grid position, not draw order: a merged pair is the one
        // whose order is undecided and swaps from frame to frame, and the
        // stream (already sorted by depth) does not care which slot is which.
        const byIJ = (a, c) => (a.j - c.j) || (a.i - c.i);
        const partsOf = (cs) => cs.slice().sort(byIJ).flatMap((c) => cellParts(c).map((e) => ({ c, ...e })));
        // A group's signature: its cells, what they hold and in which slot,
        // and the grid angle the tile-frame heading was turned by.
        const sigOf = (parts) => gridAngle + '|' + parts.map((e) => {
          const c = e.c;
          return `${c.i}.${c.j}.${c.patch}.${e.role}.${e.turn || 0}.${e.range[0]}.${e.range[1]}`;
        }).join('+');
        groupParts = groups.map((x, gi) => partsOf(merged(gi)
          ? world.groups[localWorld[gi]].map((w) => cells[w]) : x.map((k) => vc[k])));
        mergeStats.groups = multi.length;
        mergeStats.cells = multi.reduce((t, x) => t + x.length, 0);

        sigs = groups.map((x, gi) => (merged(gi) ? sigOf(groupParts[gi]) : null));
        // Groups with no stream first (they draw apart until one comes), then
        // streams turned more than MERGE_FRESH from the view.
        const missing = [], stale = [];
        sigs.forEach((sig, gi) => {
          if (!sig) return;
          const e = bestStream(sig);
          if (!e) missing.push(gi);
          else if (e.dot < MERGE_FRESH) stale.push(gi);
        });
        const want = missing.length ? missing : stale;
        mergeKey = sigs.filter(Boolean).join('/');
        mergeHaveKey = want.length ? '' : mergeKey;
        const reqParts = want.map((gi) => groupParts[gi]), reqSigs = want.map((gi) => sigs[gi]);
        let jf = b.forward;                       // the heading this job sorts for
        const reach = 8 * tileSize;
        if (!reqParts.length && !mergePending && !cam.fly) {
          // Idle, first: orbiting swings the eye round the target (a 10 degree
          // turn moves it most of a tile at a few metres), so it crosses grid
          // lines and makes new groups; sort the ones it would make 10-40
          // degrees either way, at the heading it would have there.
          const t = cam.target;
          const rx = b.eye[0] - t[0], ry = b.eye[1] - t[1];
          for (const deg of [10, -10, 20, -20, 30, -30, 40, -40]) {
            const a = deg * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a);
            const e2 = [t[0] + ca * rx - sa * ry, t[1] + sa * rx + ca * ry, b.eye[2]];
            const h = [ca * b.forward[0] - sa * b.forward[1], sa * b.forward[0] + ca * b.forward[1], b.forward[2]];
            const gw = splitOut(mergeGroups(cells, e2, mergeOpts), blendAll);
            for (const x of gw.groups) {
              if (x.length < 2 || reqParts.length >= 32) continue;
              const cs = x.map((w) => cells[w]);
              if (!cs.some((c) => Math.hypot(c.x - e2[0], c.y - e2[1]) < reach)) continue;
              const parts = partsOf(cs), sig = sigOf(parts);
              const e = bestStream(sig, h);
              if ((e && e.dot >= MERGE_NEAR) || reqSigs.includes(sig)) continue;
              reqParts.push(parts); reqSigs.push(sig);
            }
            if (reqParts.length) { jf = h; break; }
          }
        }
        if (!reqParts.length && !mergePending) {
          // Then the groups on screen at headings around the view, so
          // a fast turn finds a stream near it at once.
          const [fx, fy, fz] = b.forward;
          for (let k = 1; k * MERGE_SECTOR < 360 && !reqParts.length; k++) {
            const a = (k * MERGE_SECTOR) * Math.PI / 180;
            const h = [Math.cos(a) * fx - Math.sin(a) * fy, Math.sin(a) * fx + Math.cos(a) * fy, fz];
            sigs.forEach((sig, gi) => {
              if (!sig) return;
              const e = bestStream(sig, h);
              if (!e || e.dot < MERGE_NEAR) { reqParts.push(groupParts[gi]); reqSigs.push(sig); }
            });
            if (reqParts.length) jf = h;
          }
        }
        if (!reqParts.length && !mergePending) {
          // Then the groups the eye would make a step or two across the
          // nearest grid lines (orbiting swings it over several).
          const seen = new Set(sigs.filter(Boolean));
          const steps = [];
          for (const r of [1, 2]) {
            for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]]) {
              steps.push([r * dx, r * dy]);
            }
          }
          for (const [dx, dy] of steps) {
            const e2 = [b.eye[0] + dx * tileSize, b.eye[1] + dy * tileSize, b.eye[2]];
            const gw = splitOut(mergeGroups(cells, e2, mergeOpts), blendAll);
            for (const x of gw.groups) {
              if (x.length < 2 || reqParts.length >= 32) continue;
              const cs = x.map((w) => cells[w]);
              if (!cs.some((c) => Math.hypot(c.x - b.eye[0], c.y - b.eye[1]) < reach)) continue;
              const parts = partsOf(cs), sig = sigOf(parts);
              if (seen.has(sig) || mergeCache.has(sig)) continue;
              seen.add(sig);
              reqParts.push(parts); reqSigs.push(sig);
            }
          }
        }
        // One job in flight: the live sort shares this worker, and a job per
        // frame while moving queued it behind seconds of merging.
        if (reqParts.length && !mergePending) {
          mergePending = true;
          mergeSeq++;
          mergePendingSigs = reqSigs;
          mergePendingFwd = jf.slice();
          mergeWorker.postMessage({
            type: 'mergeGroups', eye: [0, 0, 0], seq: mergeSeq,
            // View direction in the tile frame (positions are tile-local and
            // the grid may be rotated); world offsets go in `shift`, a
            // centre's turn in `spin`.
            forward: [ ga.c * jf[0] + ga.s * jf[1],
                      -ga.s * jf[0] + ga.c * jf[1],
                       jf[2] ],
            groups: reqParts.map((parts) => parts.map((e) => {
              const c = e.c;
              const shift = (c.x - b.eye[0]) * jf[0]
                          + (c.y - b.eye[1]) * jf[1]
                          + (c.z - b.eye[2]) * jf[2];
              return { start: e.range[0], count: e.range[1], shift,
                       ...(e.turn ? { spin: turnCosSin(e.turn) } : {}) };
            })),
          });
        }
      } else {
        // Clear the key: a stale one can never be matched, and the capture
        // waits on it and would hang.
        mergeKey = '';
        mergeStats.groups = 0; mergeStats.cells = 0;
      }
    } else {
      mergeKey = '';
      mergeStats.groups = 0; mergeStats.cells = 0;
    }
    mergeStats.drawn = 0; mergeStats.missing = 0; mergeStats.turned = 0;
    const streamFor = (gi) => {
      const e = sigs && sigs[gi] ? bestStream(sigs[gi]) : null;
      if (!e) { mergeStats.missing++; return null; }
      if (e.dot <= MERGE_REUSE || !e.stream.length) { mergeStats.turned++; return null; }
      mergeStats.drawn++;
      return e.stream;
    };
    const canMerge = !!(groups && sigs);
    const groupDrawn = canMerge ? new Uint8Array(groups.length) : null;

    lodCounts = new Array(lodLevels).fill(0);
    blendedCells = 0;
    for (let vi = 0; vi < visible.length; vi++) {
      const { c, dist } = visible[vi];
      const p = patches[c.patch];
      if (!p || !p.count) continue;

      // A merged group is drawn whole at its first cell in the order.
      if (canMerge) {
        const gi = groupOf[vi];
        if (sigs[gi]) {
          if (groupDrawn[gi]) continue;
          const stream = streamFor(gi);
          if (stream) {
            groupDrawn[gi] = 1;
            cellXYArr.fill(0);
            groupParts[gi].forEach((e, slot) => {
              if (slot >= MAX_GROUP) return;
              cellXYArr[2 * slot] = e.c.x;
              cellXYArr[2 * slot + 1] = e.c.y;
              setSlot(slot, e.turn, e.role, e.z0, e.tone, tileClass ? tileClass[e.c.patch] : -1);
            });
            flushSlots();
            gl.uniform2fv(splatU.cellXY, cellXYArr);
            gl.uniform1f(splatU.mix, -1.0);    // merged groups never blend
            gl.uniform1f(splatU.fade, 1.0);
            gl.uniform1f(splatU.edgeMark, 0.0);
            gl.uniform1f(splatU.tintAmount, 0.0);
            gl.bindBuffer(gl.ARRAY_BUFFER, mergeBuf);
            gl.bufferData(gl.ARRAY_BUFFER, stream, gl.DYNAMIC_DRAW);
            gl.vertexAttribIPointer(aIndex, 1, gl.UNSIGNED_INT, 0, 0);
            gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, stream.length);
            drawnSplats += stream.length; drawCalls++;
            lodCounts[0] += groups[gi].length;
            continue;
          }
          // Nothing usable arrived; fall through and draw this cell alone.
        }
      }

      // A weakly decided cell is drawn in both classes, dissolved per splat,
      // so the class boundary wanders at splat scale instead of stepping by
      // tiles. c.mix is the decision margin (0 = coin toss); below
      // blendWidth the second draw is worth it.
      let mix = -1, other = -1, corners = null, spot = null;
      if (spotMode) {
        spot = spotOf(c);
        if (spot) mix = 0.5;
      } else if (blendGrid && c.mix != null) {
        corners = blendCorners(c);
        if (Math.min(...corners) < 0.999) {
          other = alternateTile(c, p);
          if (other >= 0) mix = 0.5;
        }
      }
      if (mix >= 0) {
        if (corners) gl.uniform4f(splatU.mixCorners, corners[0], corners[1], corners[2], corners[3]);
        gl.uniform1f(splatU.mixNoise, 1.0);
      }

      cellXYArr[0] = c.x; cellXYArr[1] = c.y;
      gl.uniform2fv(splatU.cellXY, cellXYArr.subarray(0, 2));
      gl.uniform1f(splatU.mix, mix);
      if (showClasses && classCount > 1) {
        // The rule's decision painted over the tiles: hue is the class,
        // grey is low confidence. Separates "rule chose wrong" from "tile
        // is not what its class says" (a sand tile can hold a bush).
        const base = CLASS_RGB[c.cls % CLASS_RGB.length];
        const sure = c.mix == null ? 1 : c.mix;
        const g = 0.62;
        gl.uniform3f(splatU.tint,
                     base[0] * sure + g * (1 - sure),
                     base[1] * sure + g * (1 - sure),
                     base[2] * sure + g * (1 - sure));
        gl.uniform1f(splatU.tintAmount, 0.8);
        gl.uniform1f(splatU.edgeMark, 0.0);
      } else if (showIdentity) {
        const c2 = tileRGB(c.patch);
        gl.uniform3f(splatU.tint, c2[0], c2[1], c2[2]);
        gl.uniform1f(splatU.tintAmount, 0.75);
        gl.uniform1f(splatU.edgeMark, 0.0);
      } else if (showTints && wangCodes) {
        const code = wangCodes[c.patch];
        const set = (loc, rgb) => gl.uniform3f(loc, rgb[0], rgb[1], rgb[2]);
        set(splatU.edgeN, EDGE_RGB.h[code[0] % 4]);
        set(splatU.edgeE, EDGE_RGB.v[code[1] % 4]);
        set(splatU.edgeS, EDGE_RGB.h[code[2] % 4]);
        set(splatU.edgeW, EDGE_RGB.v[code[3] % 4]);
        gl.uniform1f(splatU.edgeMark, edgeBand);
        gl.uniform1f(splatU.tintAmount, 0.0);
      } else {
        gl.uniform1f(splatU.edgeMark, 0.0);
        gl.uniform1f(splatU.tintAmount, 0.0);
      }
      const [lvl, wA, wB] = lodFor(dist);
      lodCounts[lvl]++;

      // What the cell draws, in order: its tile; or with frames the part of
      // its frame before its middle in the sorted order, then its centre,
      // then the rest, so the centre lies between them. Each is sorted on
      // its own: live along the world view direction (right for unwarped
      // tiles), or cached for the view direction in the cell's frame (W^T v);
      // a turned centre along the view turned back by its turn. A cell in
      // two classes draws the other class's after all of its own, as with
      // whole tiles: the dissolve's other half on top everywhere (step by
      // step, its centre was buried between the two parts of a ripple tile
      // and showed as a disc). A class with whole tiles has no centre, and
      // its tile is drawn whole (in the same two parts).
      const own = { tile: c.patch, centre: hasCentre(c) ? c.centre : -1,
                    turn: c.turn || 0, tone: c.tone, mate: false };
      const mateOf = (tile, field) => {
        const [k, t] = framed ? partnerCentre(c, tileClass[tile]) : [-1, 0];
        return { tile, centre: k, turn: t, tone: k >= 0 ? toneFor(tile, k, t) : null,
                 mate: true, field };
      };
      if (spot) own.field = spot.field;
      const mates = spot ? spot.mates.map((m) => mateOf(m.tile, m.field))
        : other >= 0 ? [mateOf(other, null)] : [];
      const split = own.centre >= 0 || mates.some((m) => m.centre >= 0);
      for (const who of [own, ...mates]) {
        if (who.field) gl.uniform4fv(splatU.mixField, who.field);
        // A decal overlay: the plain tile with its disc cut, the decal's own
        // splats drawn between the tile's back and front parts.
        const ovl = !split && patchOverlay ? patchOverlay[who.tile] : null;
        for (const phase of split ? ['back', 'centre', 'front'] : ovl ? ['back', 'piece', 'front'] : [null]) {
          const isCentre = phase === 'centre';
          if (isCentre && who.centre < 0) continue;
          const isPiece = phase === 'piece';
          if (ovl) {
            if (isPiece) gl.uniform4f(splatU.decal, 0, 0, 0, 0);
            else gl.uniform4f(splatU.decal, ovl.at[0], ovl.at[1], ovl.rIn, ovl.r);
          }
          const src = isPiece ? { levels: ovl.piece } : isCentre ? centres[who.centre] : patches[who.tile];
          const role = isCentre ? ROLE_CENTRE : who.centre >= 0 ? ROLE_FRAME : ROLE_TILE;
          const z0 = isCentre ? src.z0 : tileZ0[who.tile];
          const turn = isCentre ? who.turn : 0;
          if (who.mate) {
            // The complementary dissolve, so each spot ends up with one
            // splat rather than two or none.
            if (corners) {
              gl.uniform4f(splatU.mixCorners, 1 - corners[0], 1 - corners[1],
                           1 - corners[2], 1 - corners[3]);
            }
            gl.uniform1f(splatU.mixNoise, -1.0);
          }
          if (showIdentity) {
            const c2 = tileRGB(isCentre ? patches.length + who.centre : c.patch);
            gl.uniform3f(splatU.tint, c2[0], c2[1], c2[2]);
          }
          for (const [li, w] of [[lvl, wA], [lvl + 1, wB]]) {
            if (w <= 0.001 || li >= src.levels.length) continue;
            const [start, count] = src.levels[li];
            if (!count) continue;
            gl.uniform1f(splatU.fade, w);
            // Only set the tint when this view owns it (zeroing it here once
            // broke the tile-identity view).
            if (showLodColours) {
              const c2 = LOD_RGB[li % LOD_RGB.length];
              gl.uniform3f(splatU.tint, c2[0], c2[1], c2[2]);
              gl.uniform1f(splatU.tintAmount, 0.6);
            }
            setSlot(0, turn, role, z0, isCentre ? who.tone : null,
                    tileClass ? tileClass[who.tile] : -1);
            flushSlots(1);
            const n = drawSpan(c, [start, count], turn, isCentre || isPiece ? null : phase, b.forward);
            if (who.mate && n) blendedCells++;
          }
          if (ovl) gl.uniform4f(splatU.decal, 0, 0, 0, 0);
          if (who.mate) {
            if (corners) gl.uniform4f(splatU.mixCorners, corners[0], corners[1], corners[2], corners[3]);
            gl.uniform1f(splatU.mixNoise, 1.0);
          }
        }
      }
    }
  }

  if (lineCount) {
    gl.useProgram(lineProg);
    gl.bindVertexArray(lineVAO);
    gl.uniformMatrix3fv(lineU.view, false, viewMat);
    gl.uniform3fv(lineU.eye, new Float32Array(b.eye));
    gl.uniform2f(lineU.focal, fy, fy);
    gl.uniform2f(lineU.viewport, canvas.width, canvas.height);
    gl.uniform1f(lineU.near, 0.05);
    gl.uniform1f(lineU.alpha, 0.9);
    gl.drawArrays(gl.LINES, 0, lineCount);
  }
  gl.bindVertexArray(null);

  measurePop();
  drawGizmo(b);

  frames++; drawCount++;
  const now = performance.now();
  if (now - fpsTime > 500) {
    if (ui.fps) ui.fps.textContent = (frames * 1000 / (now - fpsTime)).toFixed(0);
    if (ui.drawn) ui.drawn.textContent = drawnSplats.toLocaleString() +
      (drawCalls > 1 ? ` / ${drawCalls} cells` : '');
    if (ui.sortms) ui.sortms.textContent = sortMs ? sortMs.toFixed(0) + ' ms' : '—';
    if (ui.pop) {
      popPeakAge++;
      if (popPeakAge > 8) { popPeak *= 0.6; popPeakAge = 0; }
      ui.pop.textContent = popOn
        ? `${(popRate * 100).toFixed(2)}% colour change per degree now, ${(popPeak * 100).toFixed(2)}% peak`
        : 'off';
    }
    if (ui.lodinfo) {
      ui.lodinfo.textContent = lodLevels < 2 ? 'not in this file'
        : (lodOn ? lodCounts.map((n, i) => `L${i}:${n}`).join('  ')
                 : 'off, all cells at level 0');
    }
    if (ui.classnote) {
      if (classCount < 2) {
        ui.classnote.textContent = 'one class in this tileset';
      } else if (!classOn) {
        ui.classnote.textContent = `${classCount} classes, placed at random`;
      } else if (classStats) {
        const total = classStats.counts.reduce((a, b) => a + b, 0) || 1;
        const names = materialNames();
        const share = classStats.counts
          .map((c, k) => `${names[tileClassFor(k)]} ${(100 * c / total).toFixed(0)}%`).join(' · ');
        // A field squeezed into a few tiles repeats across the grid, and the
        // rule then looks like a chequerboard; say so.
        const repeats = field && reliefScale > 0
          ? Math.round(gridN / reliefScale) : 1;
        // What each cue actually weighs, so the sliders' effect is visible.
        const cues = (classStats.weights || [])
          .filter((c) => c.w > 0.005)
          .map((c) => `${c.cue} ${(c.w * 100).toFixed(0)}%`).join(' · ');
        const extra = [classBlend ? `${blendedCells} cells blended` : '',
                       classStats.source === 'sediment' ? 'drainage from sediment' : '']
          .filter(Boolean).join(' · ');
        ui.classnote.textContent = share
          + ` · confidence ${(classStats.sure * 100).toFixed(0)}%`
          + (cues ? `\n${cues}` : '')
          + (extra ? `\n${extra}` : '')
          + (repeats > 3 ? `\n${repeats} terrain repeats across the grid `
                           + `- raise relief scale to ${gridN}` : '')
          + (classStats.missed ? `   ${classStats.missed} unmatched` : '');
      }
    }
    // Order and merge stats have no panel row: console only. A cycle means
    // some cells fell back to the depth key; reported once.
    window.bozkirOrderStats = orderStats;
    window.bozkirMergeStats = mergeStats;
    if (ui.mergestat) {
      ui.mergestat.textContent = !mergeOn ? 'off'
        : (mergeStats.groups
            ? `${mergeStats.groups} group${mergeStats.groups > 1 ? 's' : ''}, `
              + `${mergeStats.cells} cells, ${mergeStats.ms.toFixed(0)} ms, `
              + `${mergeStats.drawn} drawn merged (${mergeStats.missing} waiting, ${mergeStats.turned} turned away)`
            : 'none here');
    }
    if (orderStats.cycles > 0 && !cycleWarned) {
      cycleWarned = true;
      console.warn(`tile order: ${orderStats.cycles} cycle(s) broken by depth `
                 + `key - relief has tilted boundary planes into disagreement`);
    }
    if (ui.resn) ui.resn.textContent = `${Math.round((autoQuality ? renderScale : 1) * 100)}%`;
    updateCard(viewerState());
    reportSortError();
    frames = 0; fpsTime = now;
  }
  if (ui.azim) ui.azim.textContent = cam.azimuth.toFixed(0) + '°';
  if (ui.elev) ui.elev.textContent = cam.elevation.toFixed(0) + '°';
  if (ui.dist) ui.dist.textContent = cam.fly
    ? `fly @ ${cam.eye().map(v => v.toFixed(1)).join(', ')}`
    : cam.distance.toFixed(2);
  if (ui.cmd) ui.cmd.textContent = `--azim ${cam.azimuth.toFixed(0)} ` +
    `--elev ${cam.elevation.toFixed(0)} --dist ${cam.distance.toFixed(2)} ` +
    `--fov ${cam.fov.toFixed(0)}`;

  if (orbiting) cam.azimuth = (cam.azimuth + 0.08) % 360;
  if (film) film.step(performance.now());

  // Here, inside the frame: without preserveDrawingBuffer the colour buffer
  // is only readable until the frame ends.
  if (capture.active) capture.step();
  if (bench.active) bench.step(benchDt);
  if (shotWanted) { shotWanted = false; saveFrame(); }
  tuneQuality(benchDt);
  moveCamera(benchDt);

  requestAnimationFrame(frame);
}

/** Save the frame just drawn as a PNG, at full canvas resolution and
 *  without the panel. Must run inside that frame (no preserveDrawingBuffer). */
function saveFrame() {
  canvas.toBlob((blob) => {
    if (!blob) return;
    const name = ['bozkir', loadedScene || 'scene',
                  `grid${gridN}`, classOn ? 'rule' : 'random',
                  `az${Math.round(cam.azimuth)}`,
                  `el${Math.round(cam.elevation)}`].join('-') + '.png';
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    console.log(`saved ${name} (${canvas.width}x${canvas.height})`);
  }, 'image/png');
}

// ============================================================ interaction

let dragging = false, lastX = 0, lastY = 0;
// Touch: one finger orbits, two pinch to zoom.
const touches = new Map();
let pinchFrom = 0;
const pinchSpan = () => {
  const [a, b] = [...touches.values()];
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
};
canvas.addEventListener('pointerdown', (e) => {
  touches.set(e.pointerId, [e.clientX, e.clientY]);
  if (touches.size === 2) { dragging = false; pinchFrom = pinchSpan(); }
  else { dragging = true; lastX = e.clientX; lastY = e.clientY; }
  canvas.setPointerCapture(e.pointerId);
});
const release = (e) => {
  touches.delete(e.pointerId);
  dragging = false;
  if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
};
canvas.addEventListener('pointerup', release);
canvas.addEventListener('pointercancel', release);
canvas.addEventListener('pointermove', (e) => {
  const locked = document.pointerLockElement === canvas;
  if (touches.has(e.pointerId)) touches.set(e.pointerId, [e.clientX, e.clientY]);
  if (touches.size === 2 && pinchFrom > 0) {
    const span = pinchSpan();
    if (!cam.fly && span > 0) {
      const near = Math.max(0.05, 0.5 * (tileSize || 0.1));
      zoomGoal = Math.max(near, (zoomGoal || cam.distance) * pinchFrom / span);
    }
    pinchFrom = span;
    return;
  }
  if (!dragging && !locked) return;
  // Orbit swings the eye around a point; fly turns the view about the eye.
  // With the pointer locked (walking) the movement comes as deltas.
  const dx = locked ? e.movementX * 0.7 : e.clientX - lastX;
  const dy = locked ? e.movementY * 0.7 : e.clientY - lastY;
  const s = cam.fly ? 0.18 : -0.3;
  cam.azimuth = ((cam.azimuth + dx * s) % 360 + 360) % 360;
  cam.elevation = Math.max(-89, Math.min(89,
    cam.elevation + dy * (cam.fly ? -0.18 : 0.3)));
  lastX = e.clientX; lastY = e.clientY;
});
// Walking: a click locks the pointer so the mouse looks around, as in games.
canvas.addEventListener('click', () => {
  if (cam.walk && document.pointerLockElement !== canvas && canvas.requestPointerLock) {
    try { canvas.requestPointerLock(); } catch (e) { /* drag still looks */ }
  }
});

// Walk: first person on the ground. Sizes assume the capture is in metres.
const EYE_HEIGHT = 1.7, WALK_SPEED = 1.4;
function setWalk(on) {
  if (on === cam.walk) return;
  if (on) {
    if (!cam.fly) cam.setFly(true);
    const t = cam.target;
    cam.pos = [t[0], t[1], surfaceAt(t[0], t[1]) + EYE_HEIGHT];
    cam.elevation = 4;
    cam.walk = true;
    camVel.fill(0);
  } else {
    cam.walk = false;
    if (document.pointerLockElement === canvas) document.exitPointerLock();
    cam.distance = 6 * (tileSize || 1);
    cam.setFly(false);
    cam.elevation = Math.max(cam.elevation, 12);
    if (ui.freefly) ui.freefly.checked = false;
  }
  if (ui.viewWalk) {
    ui.viewWalk.textContent = on ? 'leave walk (V)' : 'walk on it';
    ui.viewWalk.classList.toggle('on', on);
  }
  if (ui.flynote) {
    ui.flynote.textContent = on
      ? 'WASD walks · shift runs · click, then the mouse looks · Esc frees the mouse · V leaves'
      : 'drag orbits · WASD moves · scroll zooms';
  }
}
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (cam.fly) {
    cam.speed = Math.max(0.02, Math.min(200, cam.speed * Math.exp(-e.deltaY * 0.001)));
    if (ui.speedn) ui.speedn.textContent = cam.speed.toFixed(2);
  } else {
    // Zoom sets a goal that moveCamera eases towards; line-mode wheels are
    // scaled to pixels so trackpads and mice step the same.
    const dy = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaY;
    const near = Math.max(0.05, 0.5 * (tileSize || 0.1));
    zoomGoal = Math.max(near, (zoomGoal || cam.distance) * Math.exp(dy * 0.0012));
  }
}, { passive: false });
let zoomGoal = 0;

const held = new Set();
addEventListener('keydown', (e) => held.add(e.key.toLowerCase()));
addEventListener('keyup', (e) => held.delete(e.key.toLowerCase()));
addEventListener('blur', () => held.clear());
// Any input keeps the frames coming for a moment (see frame()).
for (const ev of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'keydown', 'keyup',
                  'input', 'change', 'resize', 'visibilitychange']) {
  addEventListener(ev, () => { wakeFrames = WAKE_FRAMES; }, { capture: true, passive: true });
}
// Movement runs in the frame loop with a velocity that eases in and out
// (~0.1 s). The old fixed step on its own timer drifted against the frames
// and read as judder.
const camVel = [0, 0, 0];
const EASE_S = 0.12;

/** Keep the walker on the laid-out world and at eye height over the same
 *  ground the splats are drawn on (detail relief included). */
function walkStep(sec) {
  const p = cam.pos;
  if (tileSize && gridN > 1) {
    const a = gridAngle * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a);
    const lim = 0.5 * gridN * tileSize - 0.25 * tileSize;
    const lx = Math.max(-lim, Math.min(lim, ca * p[0] + sa * p[1]));
    const ly = Math.max(-lim, Math.min(lim, -sa * p[0] + ca * p[1]));
    p[0] = ca * lx - sa * ly;
    p[1] = sa * lx + ca * ly;
  }
  const want = surfaceAt(p[0], p[1]) + EYE_HEIGHT;
  // A short ease, so the fine creases do not shake the view.
  p[2] += (want - p[2]) * (1 - Math.exp(-sec / 0.08));
  if (p[2] < want - 0.5) p[2] = want - 0.5;
}

function moveCamera(dt) {
  if (!splatCount || !(dt > 0)) return;
  const sec = Math.min(dt, 100) / 1000;
  if (zoomGoal && !cam.fly) {
    // Ease in log space so zooming in and out feel the same.
    const z = 1 - Math.exp(-sec / 0.1);
    cam.distance = Math.exp(Math.log(cam.distance)
      + (Math.log(zoomGoal) - Math.log(cam.distance)) * z);
    if (Math.abs(cam.distance - zoomGoal) < 1e-3 * zoomGoal) {
      cam.distance = zoomGoal;
      zoomGoal = 0;
    }
  }
  // The film places the camera itself, clear of the ground (film.js).
  if (!cam.fly && !(film && film.playing())) {
    // The orbit point rides on the ground and the eye stays above it, so
    // relief or a new terrain never leaves the view underground.
    const g = surfaceAt(cam.target[0], cam.target[1]);
    if (Math.abs(g - cam.target[2]) > 1e-4) {
      cam.target[2] += (g - cam.target[2]) * (1 - Math.exp(-sec / 0.15));
    }
    const e = cam.eye();
    if (e[2] < surfaceAt(e[0], e[1]) + 0.05 * (tileSize || 1) && cam.elevation < 85) {
      cam.elevation = Math.min(85, cam.elevation + 60 * sec);
    }
  }
  const b = cam.basis();
  const fast = held.has('shift') ? (cam.walk ? 3 : 4) : 1;
  const want = [0, 0, 0];
  let speed, fwd, rgt;
  if (cam.walk) {
    // Level movement at walking pace, whatever the view's pitch.
    speed = WALK_SPEED * fast;
    const fl = Math.hypot(b.forward[0], b.forward[1]) || 1;
    fwd = [b.forward[0] / fl, b.forward[1] / fl, 0];
    const rl = Math.hypot(b.right[0], b.right[1]) || 1;
    rgt = [b.right[0] / rl, b.right[1] / rl, 0];
  } else if (cam.fly) {
    speed = cam.speed * fast;                       // units per second
    fwd = b.forward;
    rgt = b.right;
  } else {
    // Scales with the view, with a floor so walking close up is not a crawl.
    speed = Math.max(cam.distance, 4 * (tileSize || 1)) * 1.25 * fast;
    const fl = Math.hypot(b.forward[0], b.forward[1]) || 1;
    fwd = [b.forward[0] / fl, b.forward[1] / fl, 0];
    const rl = Math.hypot(b.right[0], b.right[1]) || 1;
    rgt = [b.right[0] / rl, b.right[1] / rl, 0];
  }
  const add = (v, s) => { for (let i = 0; i < 3; i++) want[i] += v[i] * s; };
  if (held.has('w')) add(fwd, speed);
  if (held.has('s')) add(fwd, -speed);
  if (held.has('d')) add(rgt, speed);
  if (held.has('a')) add(rgt, -speed);
  if (!cam.walk && held.has('e')) want[2] += speed;
  if (!cam.walk && held.has('q')) want[2] -= speed;

  const k = 1 - Math.exp(-sec / EASE_S);
  let still = true;
  for (let i = 0; i < 3; i++) {
    camVel[i] += (want[i] - camVel[i]) * k;
    if (Math.abs(camVel[i]) < 1e-4 * Math.max(speed, 1e-3)) camVel[i] = 0;
    if (camVel[i]) still = false;
  }
  if (cam.walk) walkStep(sec);
  if (still) return;
  const p = cam.fly ? cam.pos : cam.target;
  for (let i = 0; i < 3; i++) p[i] += camVel[i] * sec;
  if (cam.walk) walkStep(sec);

  // Keep the orbit target on the map; past the grid there is nothing to see.
  if (!cam.fly && tileSize && gridN > 1) {
    const a = gridAngle * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a);
    const lim = 0.5 * gridN * tileSize;
    let lx = ca * p[0] + sa * p[1];
    let ly = -sa * p[0] + ca * p[1];
    lx = Math.max(-lim, Math.min(lim, lx));
    ly = Math.max(-lim, Math.min(lim, ly));
    p[0] = ca * lx - sa * ly;
    p[1] = sa * lx + ca * ly;
  }
}

// Opening a tileset from the file dialog or by dropping it on the page: a
// zip, or the .splat and .json together (tileset.js).
async function openTileset(fileList) {
  const list = [...(fileList || [])];
  if (!list.length) return;
  showBusy('opening ' + list[0].name + '…');
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  try {
    const t = await openDrop(list);
    const d = describe(t.meta);
    // Without metadata the tile size is a guess; say so.
    if (d.note) console.warn(`${t.name}: ${d.note}; tiling may be wrong`);
    loadedScene = t.name.replace(/\.(zip|splat|json)$/i, '');
    load(t.buffer, t.meta);
    adoptAtlas(bakedAtlas);
    showBusy(`opened ${t.name}` + (d.note ? ` (${d.note})` : ''), 4000);
  } catch (err) {
    console.error(err);
    showBusy('could not open: ' + err.message, 6000);
  }
}

on('file', 'change', (e) => openTileset(e.target.files));

// Drop anywhere. dragenter and dragover must both be cancelled, or the
// browser navigates to the file.
for (const ev of ['dragenter', 'dragover']) {
  window.addEventListener(ev, (e) => {
    e.preventDefault();
    document.body.classList.add('dropping');
  });
}
for (const ev of ['dragleave', 'drop']) {
  window.addEventListener(ev, (e) => {
    e.preventDefault();
    if (ev === 'dragleave' && e.relatedTarget) return;   // moved, not left
    document.body.classList.remove('dropping');
  });
}
window.addEventListener('drop', (e) => {
  if (e.dataTransfer && e.dataTransfer.files.length) {
    openTileset(e.dataTransfer.files);
  }
});

// The drop hint (styled in index.html).
{
  const hint = document.createElement('div');
  hint.id = 'drophint';
  hint.innerHTML = '<span>drop a tileset<br>'
    + '<small style="opacity:.65">a .zip, or the .splat and .json together'
    + '</small></span>';
  document.body.appendChild(hint);
}

on('fov', 'input', (e) => {
  cam.fov = +e.target.value;
  if (ui.fovn) ui.fovn.textContent = `${cam.fov}°`;
});
on('gain', 'input', (e) => {
  gain = +e.target.value;
  if (ui.gainn) ui.gainn.textContent = gain.toFixed(2);
});
on('sorting', 'change', (e) => {
  sortingEnabled = e.target.checked;
  lastSortKey = '';
  if (!sortingEnabled && identityOrder) {
    gl.bindVertexArray(splatVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
    gl.bufferData(gl.ARRAY_BUFFER, identityOrder, gl.DYNAMIC_DRAW);
    gl.bindVertexArray(null);
    sortMs = 0;
  }
});
/** Flag a relief scale below the grid: the field then mirrors to fill the
 *  rest and landforms repeat. */
function markReliefScale() {
  if (!ui.reliefscalen) return;
  const short = field && reliefScale < gridN;
  ui.reliefscalen.textContent = short
    ? `${reliefScale} < grid ${gridN}, repeats` : String(reliefScale);
  ui.reliefscalen.style.color = short ? '#d9a441' : '';
}

on('grid', 'input', (e) => {
  gridN = +e.target.value;
  // Relief scale follows the grid until the scale slider is touched; in
  // Explore, where that slider is hidden, it always follows.
  const explore = !document.body.classList.contains('expert');
  if (field && (!reliefScaleTouched || explore) && ui.reliefscale) {
    reliefScaleTouched = false;
    const want = Math.min(+ui.reliefscale.max, Math.max(4, gridN));
    ui.reliefscale.value = String(want);
    reliefScale = want;
    ui.reliefscalen.textContent = String(want);
    field.fitTo(reliefScale * (tileSize || 1));
    paintSliders();
  }
  ui.gridn.textContent = `${gridN} x ${gridN}`;
  markReliefScale();
  scheduleRegenerate();
});
on('used', 'input', (e) => {
  usedPatches = +e.target.value;
  if (ui.usedn) ui.usedn.textContent = `${usedPatches} of ${patches.length}`;
  scheduleRegenerate();
});
on('tints', 'change', (e) => { showTints = e.target.checked; });
on('classview', 'change', (e) => { showClasses = e.target.checked; });
on('identity', 'change', (e) => {
  showIdentity = e.target.checked;
});
on('lod', 'change', (e) => { lodOn = e.target.checked; });
on('sky', 'change', (e) => { sky = e.target.value; });
on('fog', 'input', (e) => {
  fogScale = +e.target.value;
  ui.fogn.textContent = fogScale.toFixed(2);
});
on('shade', 'input', (e) => {
  groundShade = +e.target.value;
  ui.shaden.textContent = groundShade.toFixed(2);
});
on('exposure', 'input', (e) => {
  exposure = +e.target.value;
  ui.exposuren.textContent = exposure.toFixed(2);
});
on('macro', 'input', (e) => {
  macro = +e.target.value;
  ui.macron.textContent = macro.toFixed(2);
});
on('sunglow', 'input', (e) => {
  sunGlow = +e.target.value;
  ui.sunglown.textContent = sunGlow.toFixed(2);
});
on('sunaz', 'input', (e) => {
  sunAzimuth = +e.target.value;
  ui.sunazn.textContent = `${sunAzimuth}°`;
});
on('ranges', 'input', (e) => {
  ranges = +e.target.value;
  ui.rangesn.textContent = ranges.toFixed(2);
});
on('seahaze', 'input', (e) => {
  seaHaze = +e.target.value;
  ui.seahazen.textContent = seaHaze.toFixed(2);
});
on('mist', 'input', (e) => {
  mist = +e.target.value;
  ui.mistn.textContent = mist.toFixed(2);
});
on('grade', 'input', (e) => {
  gradeAmount = +e.target.value;
  ui.graden.textContent = gradeAmount.toFixed(2);
});
on('saturation', 'input', (e) => {
  saturation = +e.target.value;
  ui.saturationn.textContent = saturation.toFixed(2);
});
/** Hide the panel and axis marker so the canvas is the figure; nothing else
 *  changes, so the figure is what was on screen. */
function setFigureMode(on) {
  figureMode = on;
  // One body class hides the panel, its drag handle and the gizmo together
  // (the handle used to stay behind as an invisible strip).
  document.body.classList.toggle('figure', on);
  if (ui.figuremode) ui.figuremode.checked = on;
  // The exit button fades after a moment so it is not in the shot.
  if (ui.figexit) {
    ui.figexit.classList.remove('faded');
    clearTimeout(figExitTimer);
    if (on) figExitTimer = setTimeout(() => ui.figexit.classList.add('faded'), 2500);
  }
}
on('expert', 'change', (e) => {
  // Remembered across visits.
  document.body.classList.toggle('expert', e.target.checked);
  // Back in Explore the relief scale slider is hidden, so a scale below the
  // world size (landforms mirrored and repeated) could not be undone there.
  if (!e.target.checked && field && tileSize && reliefScale < gridN && ui.reliefscale) {
    ui.reliefscale.value = String(Math.min(+ui.reliefscale.max, gridN));
    ui.reliefscale.dispatchEvent(new Event('input'));
  }
  store.set('bozkir.expert', e.target.checked ? '1' : '');
});
if (ui.expert) {
  const saved = store.get('bozkir.expert');
  ui.expert.checked = !!saved;
  document.body.classList.toggle('expert', !!saved);
}
on('figuremode', 'change', (e) => setFigureMode(e.target.checked));
on('savepng', 'click', () => { shotWanted = true; });
window.addEventListener('keydown', (e) => {
  if (e.target.matches('input, select, textarea')) return;
  const k = e.key.toLowerCase();
  // P saves a PNG; F toggles figure mode (the checkbox is hidden with the panel).
  if (k === 'p') shotWanted = true;
  else if (k === 'f') setFigureMode(!figureMode);
  else if (k === 'v' && splatCount) setWalk(!cam.walk);
  else if (k === 'escape' && figureMode) setFigureMode(false);
  else if (k === 'escape' && cam.walk && document.pointerLockElement !== canvas) setWalk(false);
});
on('figexit', 'click', () => setFigureMode(false));
on('farfield', 'change', (e) => { farOn = e.target.checked; });
if (ui.gpuname) {
  // Privacy-minded browsers (Brave) answer with their own name or a
  // made-up string; say so rather than print it as a GPU.
  const known = /nvidia|geforce|quadro|radeon|amd|intel|apple|mali|adreno|powervr|swiftshader|llvmpipe|angle|direct3d|vulkan|metal|opengl/i;
  ui.gpuname.textContent = known.test(gpuName)
    ? String(gpuName).replace(/^ANGLE \((.*)\)$/, '$1') : 'hidden by the browser';
  // Warn when it looks like an integrated or software GPU, once, where it
  // can be seen: it costs a laptop most of its frame rate.
  if (/intel|uhd|iris|integrated|swiftshader|llvmpipe/i.test(gpuName)) {
    const tip = 'This is an integrated or software GPU. If the machine has '
      + 'a dedicated graphics card, set the browser to "High performance" in '
      + 'Windows Settings > System > Display > Graphics.';
    ui.gpuname.style.color = '#d9a441';
    ui.gpuname.title = tip;
    setTimeout(() => showBusy('drawing on an integrated GPU: slower than it could be '
      + '(research mode shows which)', 7000), 2500);
  }
}
on('detailon', 'change', (e) => { detailOn = e.target.checked; scheduleRegenerate(); });
on('detailamt', 'input', (e) => {
  detailStrength = +e.target.value;
  ui.detailamtn.textContent = detailStrength.toFixed(2);
  scheduleRegenerate();
});
on('orbit', 'change', (e) => { orbiting = e.target.checked; });
on('freefly', 'change', (e) => {
  if (cam.walk) setWalk(false);
  cam.setFly(e.target.checked);
  ui.flynote.textContent = cam.fly
    ? 'drag looks around, WASD flies, Q/E down and up, shift is faster, '
      + 'scroll changes speed'
    : 'drag orbits, WASD slides the centre, scroll zooms';
  if (ui.speedn) ui.speedn.textContent = cam.speed.toFixed(2);
});
on('viewWalk', 'click', () => setWalk(!cam.walk));
for (const [id, elev, dist] of [['viewGround', 3, 8],
                                ['viewSurvey', 32, 14], ['viewTop', 85, 18]]) {
  on(id, 'click', () => {
    setWalk(false);
    if (cam.fly) { cam.setFly(false); ui.freefly.checked = false; }
    cam.elevation = elev;
    cam.distance = (tileSize || 1) * dist;
    cam.target = [0, 0, cam.target[2]];
  });
}
on('lodcolours', 'change', (e) => {
  showLodColours = e.target.checked;
});
on('lodbase', 'input', (e) => {
  lodBaseUser = lodBase = +e.target.value;
  ui.lodbasen.textContent = `${lodBase} tiles`;
  qualityTick = -1500;          // a hand on the slider gets a moment first
});
on('autoquality', 'change', (e) => {
  autoQuality = e.target.checked;
  if (!autoQuality) {
    lodBase = lodBaseUser;
    renderScale = 1;
    ui.lodbasen.textContent = `${lodBase} tiles`;
  }
});
on('gridangle', 'input', (e) => {
  gridAngle = +e.target.value;
  if (ui.gridanglen) ui.gridanglen.textContent = `${gridAngle}°`;
  scheduleRegenerate();
});
on('popmeter', 'change', (e) => {
  popOn = e.target.checked;
  popPeak = 0; popRate = 0;
  if (popPrev) popPrev.fill(0);
});
on('popreset', 'click', () => { popPeak = 0; });
on('sortmode', 'change', (e) => {
  sortMode = e.target.value;
  lastSortKey = '';
  setTimeout(reportSortError, 0);
});
on('views', 'input', (e) => {
  cacheViews = +e.target.value;
  ui.viewsn.textContent = `${cacheViews}`;
});
on('recache', 'click', () => {
  if (!splatCount) return;
  cacheReady = false;
  ui.sorterr.textContent = 'caching...';
  worker.postMessage({ type: 'cache', views: cacheViews });
  setTimeout(reportSortError, 300);
});
// Smoothing is a switch; the frames-per-tile count only shows when it is off.
on('smoothframes', 'change', (e) => {
  const smooth = e.target.checked;
  if (ui.subdivrow) ui.subdivrow.style.display = smooth ? 'none' : '';
  ui.subdiv.value = smooth ? '17' : '1';
  ui.subdiv.dispatchEvent(new Event('input'));
});
on('subdiv', 'input', (e) => {
  const v = +e.target.value;
  setTimeout(reportSeams, 0);
  // The top of the slider means one frame per splat.
  subdiv = v > 16 ? 0 : v;
  ui.subdivn.textContent =
    v > 16 ? 'smooth' : (v === 1 ? '1 (GSWT)' : `${v} x ${v}`);
});
on('band', 'input', (e) => {
  edgeBand = +e.target.value;
  ui.bandn.textContent = edgeBand.toFixed(2);
});
on('relief', 'input', (e) => {
  relief = +e.target.value;
  ui.reliefn.textContent = relief.toFixed(2);
  scheduleRegenerate();
});
on('reliefscale', 'input', (e) => {
  reliefScale = +e.target.value;
  // Moving it takes it over; dragging it back to the grid size hands it back.
  reliefScaleTouched = reliefScale !== gridN;
  ui.reliefscalen.textContent = reliefScale.toFixed(0);
  markReliefScale();
  if (field) field.fitTo(reliefScale * (tileSize || 1));
  scheduleRegenerate();
});
on('edges', 'change', (e) => {
  showEdges = e.target.checked;
  buildOverlay();
});
on('diagonals', 'change', (e) => {
  showDiagonals = e.target.checked;
  buildOverlay();
});
on('reseed', 'click', () => {
  seed = (Math.random() * 1e9) | 0;
  scheduleRegenerate();
});
on('reset', 'click', () => {
  setWalk(false);
  cam.azimuth = 45; cam.elevation = 25;
});

// --- ordering controls ------------------------------------------------

on('tileorder', 'change', (e) => { topoOrder = e.target.value !== 'depth'; });

on('classrule', 'change', (e) => { classOn = e.target.checked; scheduleRegenerate(); });
on('riverrule', 'change', (e) => { riverRule = e.target.checked; scheduleRegenerate(); });
// Water and the water map cut the river's channel into the ground, so the
// layout is rebuilt on top of it.
on('watermap', 'change', (e) => { waterView = e.target.checked; scheduleRegenerate(); });
on('water', 'change', (e) => {
  waterOn = e.target.checked;
  if (waterOn && !waterTex) loadWater().then(() => { wakeFrames = WAKE_FRAMES; });
  wakeFrames = WAKE_FRAMES;
  scheduleRegenerate();
});
on('riverwidth', 'input', (e) => {
  riverWidth = +e.target.value;
  ui.riverwidthn.textContent = `${riverWidth} m`;
  if (waterOn || waterView) scheduleRegenerate();
});
on('waterflow', 'change', (e) => { flowOn = e.target.checked; wakeFrames = WAKE_FRAMES; });
on('classswap', 'change', (e) => {
  classSwap = e.target.checked;
  // Swapping changes where each material goes, not how much of it there
  // is: a measured share stays with its material (with three, the bush
  // took the sand's 50%).
  if (captureShares && classCount >= 2) {
    setControl('classbalance', snapShare(captureShares[tileClassFor(0)]));
  }
  updateShareLabel();
  scheduleRegenerate();
});
on('classwander', 'input', (e) => {
  classWander = +e.target.value;
  ui.classwandern.textContent = `${Math.round(classWander * 100)}%`;
  scheduleRegenerate();
});
on('classaltitude', 'input', (e) => {
  classAltitude = +e.target.value;
  ui.classaltituden.textContent = `${Math.round(classAltitude * 100)}%`;
  scheduleRegenerate();
});
on('classcoherence', 'input', (e) => {
  classCoherence = +e.target.value;
  ui.classcoherencen.textContent = String(classCoherence);
  scheduleRegenerate();
});
on('classbalance', 'input', (e) => {
  classBalance = +e.target.value;
  ui.classbalancen.textContent = `${Math.round(classBalance * 100)}%`;
  scheduleRegenerate();
});
on('classblend', 'change', (e) => { classBlend = e.target.checked; });
on('drift', 'input', (e) => {
  colourDrift = +e.target.value;
  ui.driftn.textContent = `${Math.round(colourDrift * 100)}%`;
  wakeFrames = WAKE_FRAMES;
});
on('blendwidth', 'input', (e) => {
  spotWidth = +e.target.value;
  ui.blendwidthn.textContent = `${Math.round(spotWidth * 100)}%`;
  // Which cells draw a neighbour's material depends on the width.
  for (const c of cells) c.spot = undefined;
  wakeFrames = WAKE_FRAMES;
});
on('tonegrade', 'change', (e) => { toneGrade = e.target.checked; scheduleRegenerate(); });
on('decals', 'change', (e) => { decalsOn = e.target.checked; scheduleRegenerate(); });
on('decalterrain', 'change', (e) => { decalTerrain = e.target.checked; scheduleRegenerate(); });
on('decalspacing', 'input', (e) => {
  decalSpacing = +e.target.value;
  ui.decalspacingn.textContent = `${decalSpacing} tiles`;
  scheduleRegenerate();
});
on('classsharp', 'input', (e) => {
  classSharp = +e.target.value;
  ui.classsharpn.textContent = classSharp.toFixed(1);
  scheduleRegenerate();
});
on('merging', 'change', (e) => {
  mergeOn = e.target.checked;
  // Drop any stale stream, or it is drawn the moment merging comes back.
  mergeKey = ''; mergeHaveKey = ''; mergeCache.clear(); mergeBytes = 0; mergeSeq++; mergePending = false;
});
on('mergeorder', 'change', (e) => {
  groupedOrder = e.target.checked;
  mergeKey = ''; mergeHaveKey = ''; mergeCache.clear(); mergeBytes = 0; mergeSeq++; mergePending = false;
});
on('mergethr', 'input', (e) => {
  mergeThreshold = +e.target.value;
  ui.mergethrn.textContent = mergeThreshold.toFixed(2);
  mergeKey = ''; mergeCache.clear(); mergeBytes = 0; mergeSeq++; mergePending = false;
});
function showSweep() {
  const arc = +ui.caparc.value, frames = +ui.capframes.value;
  ui.caparcn.textContent = arc;
  ui.capframesn.textContent = frames;
  ui.capstepn.textContent = (arc / frames).toFixed(2);
}
on('capframes', 'input', showSweep);
on('caparc', 'input', showSweep);
showSweep();

// --- scaling sweep -----------------------------------------------------
// Measures where the renderer stops keeping up, and whether the frame, the
// sort or the draw calls give out first.
let benchDt = 16.7;
let benchPrev = 0;

const bench = new Benchmark({
  sizes: [4, 8, 12, 16, 20, 24, 32, 40, 48],
  apply: (g) => {
    gridN = g;
    if (ui.grid) ui.grid.value = String(Math.min(g, +ui.grid.max));
    if (ui.gridn) ui.gridn.textContent = `${g} x ${g}`;
    regenerate();
  },
  read: () => ({
    cells: cells.length,
    splats: drawnSplats,
    calls: drawCalls,
    sortMs,
  }),
  isBusy: () => sortPending || !sortedReady,
  onRow: (r) => {
    if (ui.bench) {
      ui.bench.textContent = `sweeping ${r.grid} x ${r.grid}...`;
    }
    console.log(`  grid ${r.grid}: ${r.fps.toFixed(0)} fps, `
      + `${r.splats.toLocaleString()} splats, ${r.sortMs.toFixed(1)} ms sort`);
  },
  onDone: (rows) => {
    if (ui.bench) ui.bench.textContent = 'scaling sweep';
    const text = report(rows);
    console.log('\n' + text);
    if (ui.benchout) ui.benchout.textContent = text;
    window.bozkirBench = rows;
  },
});

on('bench', 'click', () => {
  if (bench.active) { bench.cancel(); ui.bench.textContent = 'scaling sweep'; return; }
  if (!splatCount) { console.warn('nothing loaded to measure'); return; }
  if (ui.benchout) ui.benchout.textContent = 'measuring...';
  bench.start();
});

// --- capturing a sweep, for scripts/pop_metric.py ----------------------
// Each pose waits until no sort or merge is in flight, or the frame shows
// the previous pose's order and the capture measures worker latency.
const capture = new Capture({
  gl, canvas, cam,
  isBusy: () => sortPending || !sortedReady
    || (mergeOn && mergeKey !== '' && mergeHaveKey !== mergeKey),
  onProgress: (done, total) => {
    if (ui.capture) ui.capture.textContent = `capturing ${done} / ${total}`;
  },
  onDone: (count, file) => {
    if (ui.capture) ui.capture.textContent = 'capture sweep';
    console.log(`captured ${count} frames to ${file}`);
    if (queuedRun) { const q = queuedRun; queuedRun = null; q(); }
  },
});

/** The middle of the laid-out terrain, for the sweep to orbit. Not the
 *  camera target (WASD moves that), so a capture repeats across sessions;
 *  height from the cells, so a warped surface is orbited about its middle. */
function sceneCentre() {
  if (!cells.length) return cam.target.slice();
  let x = 0, y = 0, z = 0;
  for (const c of cells) { x += c.x; y += c.y; z += c.z; }
  return [x / cells.length, y / cells.length, z / cells.length];
}

/** Folder name for the zip, after the ordering that produced it. */
function runName() {
  return (topoOrder ? 'topological' : 'depth') + (mergeOn ? '' : '_unmerged');
}

let queuedRun = null;

function beginSweep() {
  if (!splatCount) { console.warn('nothing loaded to capture'); return false; }
  const frames = ui.capframes ? +ui.capframes.value : 48;
  const arc = ui.caparc ? +ui.caparc.value : 20;
  const centred = !ui.capcentre || ui.capcentre.checked;
  const target = centred ? sceneCentre() : cam.target.slice();
  if (!capture.start({ frames, arc, settle: 1, name: runName(), target })) {
    return false;
  }
  // Everything needed to repeat this run, on one line.
  console.log(`sweep ${arc}° in ${frames} frames `
    + `(${(arc / frames).toFixed(2)}° each) about `
    + `[${target.map((v) => v.toFixed(2)).join(', ')}], `
    + `az ${cam.azimuth.toFixed(1)}, el ${cam.elevation.toFixed(1)}, `
    + `dist ${cam.distance.toFixed(2)}`);
  if (ui.capture) ui.capture.textContent = `capturing 0 / ${frames}`;
  return true;
}

on('capture', 'click', () => {
  if (capture.active) {
    queuedRun = null;
    capture.cancel();
    ui.capture.textContent = 'capture sweep';
    return;
  }
  beginSweep();
});

// Both orderings back to back from the same camera, so the two runs travel
// the same path.
on('captureboth', 'click', () => {
  if (capture.active) return;
  if (!splatCount) { console.warn('nothing loaded to capture'); return; }
  const restore = topoOrder;
  topoOrder = true;
  if (ui.tileorder) ui.tileorder.value = 'topological';
  queuedRun = () => {
    topoOrder = false;
    if (ui.tileorder) ui.tileorder.value = 'depth';
    // Let the panel and the next sort settle before the second run.
    setTimeout(() => {
      queuedRun = () => {
        topoOrder = restore;
        if (ui.tileorder) ui.tileorder.value = restore ? 'topological' : 'depth';
        console.log('both runs captured; unzip each into its own folder, then\n'
          + '  python scripts/pop_metric.py frames/topological --against frames/depth');
      };
      beginSweep();
    }, 250);
  };
  beginSweep();
});

const params = new URLSearchParams(location.search);
// ?demo plays the one-minute demo (film.js), starting on the river.
// &slow=N plays it, and the water's flow, N times slower.
const wantFilm = params.has('demo');
const TIME_SCALE = 1 / Math.max(1, +params.get('slow') || 1);
const wanted = params.get('scene') || (wantFilm ? 'river' : null);
// Terrain is chosen separately: ?scene=desert&height=mesa. Without it a
// tileset looks for the height field with its own name.
const wantedHeight = params.get('height');
// Or generated by recipe, the same ground on every machine:
// ?gen=canyon&seed=3&size=256
const wantedGen = params.get('gen');
const wantedSeed = Math.max(0, parseInt(params.get('seed') || '0', 10) || 0);
const wantedSize = [128, 256, 512].includes(+params.get('size'))
  ? +params.get('size') : 256;

// Scenes tried when none is asked for: an export named 'scene' (your own
// capture), then the desert tileset that ships with the repo. The river
// ships too and is listed, but opens only when asked for.
const DEFAULT_SCENES = ['scene', 'desert', 'river'];

/** Fill the tileset and terrain menus from data/index.json (a static server
 *  cannot list a folder) plus the defaults, keeping only files that are
 *  really there (a HEAD request each), so no entry leads to a 404. */
async function loadCatalogue() {
  let index = {};
  try {
    const r = await fetch('./data/index.json', { cache: 'no-cache' });
    if (r.ok) index = await r.json();
  } catch (e) { /* no index: the defaults are still looked for */ }

  const withDefaults = (items) => {
    const seen = new Set(items.map((it) => it.name));
    return items.concat(DEFAULT_SCENES.filter((n) => !seen.has(n))
      .map((name) => ({ name })));
  };
  const present = async (items, file) => (await Promise.all(items.map(it =>
    fetch(`./data/${file(it.name)}`, { method: 'HEAD' })
      .then(r => (r.ok ? it : null)).catch(() => null)))).filter(Boolean);
  const scenes = await present(withDefaults(index.scenes || []), n => `${n}.splat`);
  const terrains = await present(withDefaults(index.terrains || []), n => `${n}.height.png`);
  // The starter needs no file, so it is always offered.
  scenes.push({ name: 'starter', starter: true });

  const fill = (el, items, label, current) => {
    if (!el) return;
    for (const it of items) {
      const o = document.createElement('option');
      o.value = it.name;
      o.textContent = label(it);
      if (it.name === current) o.selected = true;
      el.appendChild(o);
    }
  };
  fill(ui.scenepick, scenes,
       (s) => (s.starter ? 'starter · made in the browser'
         : s.classes == null ? s.name
         : `${s.name} · ${s.classes > 1 ? s.classes + ' materials' : '1 material'}`
           + ` · ${s.megabytes} MB`), loadedScene);
  fill(ui.heightpick, terrains,
       (t) => ((t.profile || t.source)
         ? `${t.name} · ${t.profile || t.source}` + (t.sediment ? ' · sediment' : '')
         : t.name),
       wantedHeight || loadedScene);
}

// A new tileset reloads the page with it in the URL (shareable); a new
// terrain is swapped in place under the same tiles.
on('scenepick', 'change', (e) => {
  if (!e.target.value) return;
  const p = new URLSearchParams(location.search);
  p.set('scene', e.target.value);
  location.search = p.toString();
});
// ------------------------------------------------------------ terrain window
// Every profile as a small preview at the current seed, generated at 64
// across through the same worker and cache. Changing the seed redraws them.

let genProfile = wantedGen || 'desert';
let previewToken = 0;

/** A terrain preview as it would look under the loaded tiles: shape as
 *  shading, each tile's worth of ground in the colour of the class the rule
 *  would give it. */
function drawPreview(canvas, z, sed, n) {
  const cells = 16, over = n / cells;
  let cls = null;
  if (classCount > 1 && classOn) {
    try {
      cls = classify(Float64Array.from(z), cells, classCount, {
        sediment: sed ? Float64Array.from(sed) : null,
        sharpness: classSharp, altitude: classAltitude,
        coherence: Math.min(classCoherence, 1),
        balance: classCount === 2 ? classBalance : null,
        river: riverRule,
      }).cls;
    } catch (e) { cls = null; }
  }
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(n, n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const k = j * n + i;
      const dx = z[j * n + Math.min(n - 1, i + 1)] - z[j * n + Math.max(0, i - 1)];
      const dy = z[Math.min(n - 1, j + 1) * n + i] - z[Math.max(0, j - 1) * n + i];
      const shade = Math.max(0.18, Math.min(1.25, 0.85 - 4.0 * (dx + dy)));
      let c;
      if (cls) c = classColours[tileClassFor(cls[Math.floor(j / over) * cells + Math.floor(i / over)])];
      else if (classCount > 1) {
        // Rule off: each cell takes a class at random, as the viewer does.
        const h = Math.imul((Math.floor(j / over) * cells + Math.floor(i / over)) + 1,
                            0x9E3779B1) >>> 0;
        c = classColours[h % classCount];
      } else c = classColours[0];
      const lift = 0.75 + 0.5 * z[k];
      for (let q = 0; q < 3; q++) {
        img.data[4 * k + q] = Math.max(0, Math.min(255, 255 * c[q] * shade * lift));
      }
      img.data[4 * k + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function buildCards() {
  if (!ui.gencards || ui.gencards.childElementCount) return;
  for (const group of TERRAIN_GROUPS) {
    const box = document.createElement('div');
    box.className = 'tw-group';
    const head = document.createElement('div');
    head.className = 'tw-group-head';
    head.textContent = group.name;
    const row = document.createElement('div');
    row.className = 'tw-group-row';
    box.append(head, row);
    ui.gencards.appendChild(box);
    for (const name of group.profiles) row.appendChild(profileCard(name));
  }
}

function profileCards() {
  return ui.gencards ? ui.gencards.querySelectorAll('.tw-card') : [];
}

function profileCard(name) {
  {
    const card = document.createElement('div');
    card.className = 'tw-card' + (name === genProfile ? ' on' : '');
    card.dataset.profile = name;
    const cv = document.createElement('canvas');
    cv.width = cv.height = 64;
    const label = document.createElement('span');
    label.textContent = name;
    card.append(cv, label);
    card.addEventListener('click', () => {
      genProfile = name;
      for (const c of profileCards()) c.classList.toggle('on', c === card);
    });
    card.addEventListener('dblclick', () => ui.genbtn.click());
    return card;
  }
}

async function refreshPreviews() {
  const token = ++previewToken;
  const seed = Math.max(0, parseInt(ui.genseed.value, 10) || 0);
  for (const card of profileCards()) {
    if (token !== previewToken) return;       // the seed moved on; stop
    try {
      const f = await generatedField({ profile: card.dataset.profile, seed, size: 64 });
      if (token !== previewToken) return;
      drawPreview(card.querySelector('canvas'), f.z, f.sediment, 64);
    } catch (e) { /* a preview that fails just stays blank */ }
  }
}

async function refreshRecent() {
  if (!ui.genrecent) return;
  ui.genrecent.textContent = '';
  const made = await listCached();
  if (!made.length) {
    ui.genrecent.textContent = 'nothing yet';
    return;
  }
  for (const t of made.slice(0, 16)) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = `${t.profile} ${t.seed} · ${t.size}`;
    b.addEventListener('click', () => {
      genProfile = t.profile;
      ui.genseed.value = String(t.seed);
      ui.gensize.value = String(t.size);
      ui.genbtn.click();
    });
    ui.genrecent.appendChild(b);
  }
}

function openTerrainWindow() {
  buildCards();
  for (const c of profileCards()) c.classList.toggle('on', c.dataset.profile === genProfile);
  ui.terrainwin.showModal();
  refreshPreviews();
  refreshRecent();
}

function stepSeed(d) {
  const v = Math.max(0, (parseInt(ui.genseed.value, 10) || 0) + d);
  ui.genseed.value = String(v);
  refreshPreviews();
}

on('terrainopen', 'click', openTerrainWindow);
on('terrainclose', 'click', () => ui.terrainwin.close());
on('seeddown', 'click', () => stepSeed(-1));
on('seedup', 'click', () => stepSeed(1));
on('seedrand', 'click', () => {
  ui.genseed.value = String(Math.floor(Math.random() * 100000));
  refreshPreviews();
});
on('genseed', 'change', () => refreshPreviews());
on('genseed', 'keydown', (e) => { if (e.key === 'Enter') ui.genbtn.click(); });
on('genbtn', 'click', () => {
  const profile = genProfile;
  const seed = Math.max(0, parseInt(ui.genseed.value, 10) || 0);
  const size = +ui.gensize.value || 256;
  const p = new URLSearchParams(location.search);
  p.delete('height');
  p.set('gen', profile); p.set('seed', String(seed)); p.set('size', String(size));
  history.replaceState(null, '', `${location.pathname}?${p}`);
  if (ui.terrainwin && ui.terrainwin.open) ui.terrainwin.close();
  useGeneratedField(profile, seed, size);
});
if (ui.genseed) {
  ui.genseed.value = String(wantedSeed);
  ui.gensize.value = String(wantedSize);
}

// ------------------------------------------------------------ presets
// Named settings from views.json. A preset sets each control and fires the
// same event a slider would, so it can only do what the panel can.

let presets = {};
let pendingPreset = params.get('preset');
// The loaded tileset's own view (manifest view) and story (manifest story:
// its chapters in views.json carry the same "story"; none is the default).
function ownPreset() {
  const v = loadedManifest && loadedManifest.view;
  return typeof v === 'string' && presets[v] ? v : 'readme';
}
function storyOf() {
  return (loadedManifest && loadedManifest.story) || null;
}

async function loadPresets() {
  try {
    const r = await fetch('./views.json');
    presets = r.ok ? await r.json() : {};
  } catch (e) { presets = {}; }
  if (!ui.presetchips) return;
  // One chip for the tileset's own settings (its manifest's view), then the
  // Research comparisons; other views stay reachable by link (?preset=).
  const standard = document.createElement('button');
  standard.type = 'button';
  standard.dataset.standard = '1';
  standard.textContent = 'standard';
  standard.dataset.tip = "This tileset's own settings: everything back to how it was set up to be seen.";
  standard.addEventListener('click', () => applyPreset(ownPreset()));
  ui.presetchips.appendChild(standard);
  for (const [name, p] of Object.entries(presets)) {
    if (name.startsWith('_') || p.chapter || !p.research) continue;
    const b = document.createElement('button');
    b.type = 'button';
    b.dataset.preset = name;
    b.classList.add('adv');
    b.textContent = p.short || name;
    b.dataset.tip = p.label || name;
    b.addEventListener('click', () => applyPreset(name));
    ui.presetchips.appendChild(b);
  }
  setChapters(presets, storyOf());
  // If the ground arrived before this file, the URL's preset is still waiting.
  if (pendingPreset && presets[pendingPreset] && field) {
    const name = pendingPreset;
    pendingPreset = null;
    applyPreset(name);
    applyView(takeView());
  }
}

function setControl(id, value) {
  const el = $(id);
  if (!el) { console.warn(`preset: no control '${id}'`); return; }
  if (el.type === 'checkbox') {
    el.checked = !!value;
    el.dispatchEvent(new Event('change'));
  } else if (el.tagName === 'SELECT') {
    el.value = String(value);
    el.dispatchEvent(new Event('change'));
  } else {
    el.value = String(value);
    el.dispatchEvent(new Event('input'));
  }
}

// Keys in views.json that are about the preset, not panel controls.
const PRESET_META = new Set(['label', 'short', 'view', 'research', 'chapter',
                             'caption', 'camera', 'story']);

/** A control back to the page's own value. */
function resetControl(id) {
  const el = $(id);
  if (!el) return;
  if (el.type === 'checkbox') setControl(id, el.defaultChecked);
  else if (el.tagName === 'SELECT') {
    const def = [...el.options].find((o) => o.defaultSelected) || el.options[0];
    if (def) setControl(id, def.value);
  } else setControl(id, el.defaultValue);
}

function applyPreset(name) {
  const p = presets[name];
  if (!p) { console.warn(`no preset '${name}'`); return; }
  // What another preset set and this one does not name goes back to its
  // default, or one preset's dusk stays on under the next. The world's size
  // (grid, relief scale) and the capture's own shares are left as they are;
  // controls no preset touches keep whatever the viewer set.
  const named = new Set(Object.keys(p));
  const touched = new Set();
  for (const [n, q] of Object.entries(presets)) {
    if (!n.startsWith('_')) for (const k of Object.keys(q)) touched.add(k);
  }
  for (const k of touched) {
    if (named.has(k) || PRESET_META.has(k) || k === 'grid' || k === 'reliefscale') continue;
    if (k === 'classbalance' && captureShares && classCount >= 2) continue;
    if (k === 'classblend' && classCount >= 2) continue;
    resetControl(k);
  }
  // Grid first, since 'reliefscale: grid' and the rule both depend on it.
  if ('grid' in p) setControl('grid', p.grid);
  for (const [k, v] of Object.entries(p)) {
    if (PRESET_META.has(k) || k === 'grid') continue;
    // A tileset that measured its capture keeps the capture's share.
    if (k === 'classbalance' && captureShares && classCount >= 2) {
      setControl(k, snapShare(captureShares[tileClassFor(0)]));
      continue;
    }
    if (k === 'classblend' && classCount >= 2) continue;
    setControl(k, k === 'reliefscale' && v === 'grid' ? gridN : v);
  }
  const viewBtn = p.view && $(p.view);
  if (viewBtn) viewBtn.click();
  // A chapter frames its own shot: angles in degrees, distance in tiles.
  if (p.camera) {
    setWalk(false);
    if (cam.fly) { cam.setFly(false); if (ui.freefly) ui.freefly.checked = false; }
    const t = tileSize || 1;
    if (p.camera.azimuth != null) cam.azimuth = p.camera.azimuth;
    if (p.camera.elevation != null) cam.elevation = p.camera.elevation;
    if (p.camera.distance != null) { cam.distance = p.camera.distance * t; zoomGoal = 0; }
    cam.target = [0, 0, height(0, 0)];
  }
  if (!p.chapter) endChapters();
  if (ui.presetchips) {
    for (const b of ui.presetchips.children) {
      b.classList.toggle('on', b.dataset.standard ? name === ownPreset() : b.dataset.preset === name);
    }
  }
  const q = new URLSearchParams(location.search);
  q.set('preset', name);
  history.replaceState(null, '', `${location.pathname}?${q}`);
}

// --- the demo (?demo, film.js) ------------------------------------------
/** The deepest-looking water nearest the world's centre, world x, y (null
 *  without a river). */
function waterSpot() {
  if (!field || !field.level || !field.carve) return null;
  const n = field.w, amp = amplitude(), scale = (n - 1) / (field.extent || 1);
  let best = null, bestD = Infinity;
  for (let j = 0; j < n; j += 2) for (let i = 0; i < n; i += 2) {
    const c = j * n + i;
    if (field.level[c] - (2 * amp * (field.z[c] - 0.5) + field.carve[c]) < 0.15) continue;
    const x = (i - (n - 1) / 2) / scale, y = ((n - 1) / 2 - j) / scale, d = x * x + y * y;
    if (d < bestD) { bestD = d; best = [x, y]; }
  }
  return best;
}
/** A water spot about `dist` from (x0, y0), for a camera pass along the
 *  river; null without water. */
function waterAlong(x0, y0, dist) {
  if (!field || !field.level || !field.carve) return null;
  const n = field.w, amp = amplitude(), scale = (n - 1) / (field.extent || 1);
  let best = null, bestE = Infinity;
  for (let j = 0; j < n; j += 2) for (let i = 0; i < n; i += 2) {
    const c = j * n + i;
    if (field.level[c] - (2 * amp * (field.z[c] - 0.5) + field.carve[c]) < 0.15) continue;
    const x = (i - (n - 1) / 2) / scale, y = ((n - 1) / 2 - j) / scale;
    const e = Math.abs(Math.hypot(x - x0, y - y0) - dist) + 0.3 * Math.hypot(x, y);
    if (e < bestE) { bestE = e; best = [x, y]; }
  }
  return best;
}
/** Water depth at world x, y in metres, negative where dry; null without
 *  a river or off the field. */
function waterDepthAt(x, y) {
  if (!field || !field.level || !field.carve) return null;
  const n = field.w, amp = amplitude(), scale = (n - 1) / (field.extent || 1);
  const i = Math.round(x * scale + (n - 1) / 2), j = Math.round((n - 1) / 2 - y * scale);
  if (i < 0 || j < 0 || i >= n || j >= n) return null;
  const c = j * n + i;
  return field.level[c] - (2 * amp * (field.z[c] - 0.5) + field.carve[c]);
}
/** The demo's first tile, world x, y: a dry cell of the material in `want`
 *  (gravel on the river), 1.5 to 4 tiles from the water so the river comes
 *  in early, nearest the world's centre. Else the cell nearest the centre. */
function captureCell(want = ['gravel']) {
  const T = tileSize || 1, k = materialNames().findIndex((n) => want.includes(n));
  const w = waterSpot();
  const dry = (x, y) => [[0, 0], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]
    .every(([u, v]) => { const d = waterDepthAt(x + u * T, y + v * T); return d == null || d < -0.05; });
  let best = null, bestD = Infinity, near = null, nearD = Infinity;
  for (const c of cells) {
    const d = Math.hypot(c.x, c.y);
    if (d < nearD) { nearD = d; near = [c.x, c.y]; }
    if (k >= 0 && tileClass && tileClass[c.patch] !== k) continue;
    const dw = w ? Math.hypot(c.x - w[0], c.y - w[1]) / T : 3;
    if (dw < 1.5 || dw > 4 || !dry(c.x, c.y)) continue;
    if (d < bestD) { bestD = d; best = [c.x, c.y]; }
  }
  return best || near || [0, 0];
}
const nextFrame = () => new Promise((r) => requestAnimationFrame(r));
async function waitFor(ok) { while (!ok()) await nextFrame(); }
/** Resolves once nothing has been rebuilt or waited on for 20 frames. */
async function settled() {
  let quiet = 0;
  while (quiet < 20) {
    await nextFrame();
    quiet = field && !regenPending && !sortPending && (!waterOn || waterTex) ? quiet + 1 : 0;
  }
}
const prefetched = {};
function fetchScene(name) {
  return prefetched[name] || (prefetched[name] = Promise.all([
    fetchBuffer(`./data/${name}.splat`, () => {}),
    fetch(`./data/${name}.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
  ]));
}
/** Open another tileset in place (no reload), with the ground it was set
 *  up on, as loadFirst does. */
async function switchScene(name) {
  const t0 = performance.now();
  const [b, m] = await fetchScene(name);
  if (!b) throw new Error(`demo: no tileset ${name}`);
  const t1 = performance.now();
  // Water is the river's; off, the new ground is not cut for it first.
  if (waterOn) { waterOn = false; if (ui.water) ui.water.checked = false; }
  loadedScene = name;
  if (ui.scenepick) ui.scenepick.value = name;
  load(b, m);
  loadAtlas(name);
  if (Object.keys(presets).length) setChapters(presets, storyOf());
  const t2 = performance.now();
  const own = m && m.terrain && m.terrain.gen ? m.terrain : null;
  if (own) await useGeneratedField(own.gen, own.seed || 0, own.size || 256);
  else await useHeightField(name);
  console.log(`demo: ${name} in ${((performance.now() - t0) / 1000).toFixed(1)} s `
    + `(download ${((t1 - t0) / 1000).toFixed(1)}, unpack ${((t2 - t1) / 1000).toFixed(1)}, `
    + `ground ${((performance.now() - t2) / 1000).toFixed(1)})`);
}
const film = wantFilm ? createFilm({
  scene: () => loadedScene,
  tileSize: () => tileSize || 1,
  surfaceAt, waterSpot, captureCell, settled, switchScene,
  // The tileset's own preset is applied a moment after its ground arrives;
  // the film's settings go over it, not under.
  ready: async () => {
    await waitFor(() => field && splatCount && Object.keys(presets).length && !pendingPreset);
    await settled();
  },
  prefetch: (names) => { for (const n of names) if (n !== loadedScene) fetchScene(n); },
  applyPreset: (name) => applyPreset(name),
  setControl: (id, v) => setControl(id, v),
  setFigure: (on) => { setWalk(false); orbiting = false; setFigureMode(on); },
  setResearch: (on) => { const b = $(on ? 'modeResearch' : 'modeExplore'); if (b) b.click(); },
  openTab: (name) => {
    const b = [...document.querySelectorAll('#side .tabs button')].find((x) => x.textContent === name);
    if (b && !b.classList.contains('on')) b.click();
    return b;
  },
  panelEl: () => $('panel'),
  setCamera: (t, az, el, d) => {
    if (cam.fly) cam.setFly(false);
    cam.target = t; cam.azimuth = az; cam.elevation = el; cam.distance = d;
    zoomGoal = 0; camVel.fill(0);
  },
  setRise: (x, y, r, band, drop, square = false) => {
    riseCircle.set([x, y, r, band]); riseDrop = drop; riseSquare = square;
  },
  // The world's size changes in the film extend the ground instead of
  // stretching it: the relief scale stays where it is.
  holdReliefScale: () => { reliefScaleTouched = true; },
  waterAlong,
  setFlowClock: (s) => { flowClock = s; },
  // What the film's tables draw: the rule's maps and decision for this
  // world, and the tileset's tiles as the far-field atlas shows them.
  ruleMaps: () => lastRule && { ...lastRule, cls: Array.from(lastRule.cls),
                                colours: Array.from(lastRule.cls, (k) => CLASS_RGB[k % CLASS_RGB.length]),
                                names: materialNames(), classOf: tileClassFor },
  tileTable: () => loadedManifest && atlas && {
    scene: loadedScene, img: atlas.img, cols: atlas.cols, res: atlas.res,
    tiles: loadedManifest.tiles.map((t) => ({ codes: [t.n, t.e, t.s, t.w], cls: t.class || 0 })),
    // Atlas entries after the tiles, in order; decal copies (with a base)
    // are not part of the pool.
    centres: (loadedManifest.centres || []).map((c) => ({ cls: c.class || 0, pool: !('base' in c) })),
    frames: loadedManifest.frames || null,
    names: materialNames(), edgeRGB: EDGE_RGB, size: tileSize,
  },
}, { rec: params.has('rec'), slow: 1 / TIME_SCALE, fps: Math.max(0, +params.get('frames') || 0) }) : null;
if (film) film.prepare().catch((e) => { console.error(e); film.stop(); });
// What scripts/record_demo.mjs drives a render with: the next frame, and
// whether the one asked for has been drawn with nothing left to sort.
if (film && params.has('frames')) {
  window.bozkirFilm = {
    state: () => film.state(), clock: () => film.clock(), tick: () => film.tick(),
    drawn: () => drawCount,
    // The eye and the ground under it, for checking a render's camera.
    eye: () => { const e = cam.eye(); return [...e, surfaceAt(e[0], e[1]), cam.elevation]; },
    idle: () => !sortPending && !regenPending && !(mergeOn && mergeKey !== '' && mergeHaveKey !== mergeKey),
  };
}

loadPresets();

on('heightpick', 'change', (e) => {
  if (!e.target.value) return;
  const p = new URLSearchParams(location.search);
  p.set('height', e.target.value);
  history.replaceState(null, '', `${location.pathname}?${p}`);
  useHeightField(e.target.value);
});

/** Nothing to load: build the starter tileset and lay it on generated
 *  ground, so the viewer never opens empty. */
function loadStarter(note) {
  if (splatCount) return;
  const { buffer, manifest } = starterTileset();
  loadedScene = 'starter';
  load(buffer, manifest);
  adoptAtlas(bakedAtlas);
  loadDone(note);
  loadCatalogue();
  // A shared link carries every setting itself; no default under it.
  if (!pendingPreset && !params.has('v')) pendingPreset = 'readme';
  useGeneratedField(wantedGen || 'rolling', wantedGen ? wantedSeed : 0,
                    wantedGen ? wantedSize : 256);
  console.log('no scene found: showing the starter tileset');
}

/** Load the first of `names` that exists, one after another (fetched in
 *  parallel, the smallest file won, not the first in the list). */
async function loadFirst(names) {
  for (const name of names) {
    if (splatCount) break;        // one loaded: the rest are not downloaded
    let b = null, m = null;
    try {
      [b, m] = await Promise.all([
        fetchBuffer(`./data/${name}.splat`, (got, total) => loadProgress(
          0.05 + 0.85 * got / total,
          `downloading ${name}: ${(got / 1e6).toFixed(1)} of ${(total / 1e6).toFixed(1)} MB`)),
        fetch(`./data/${name}.json`).then(r => (r.ok ? r.json() : null)).catch(() => null),
      ]);
    } catch (e) { continue; }
    if (!b || splatCount) continue;
    loadedScene = name;
    loadProgress(0.95, 'building the world…');
    // Let the note paint before the main thread is busy unpacking.
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    load(b, m);
    loadProgress(1, 'ready');
    loadDone();
    // A plain capture may get no height field, so no adoptField to apply
    // a shared view from; give it one here.
    if (plainScene) setTimeout(() => applyView(takeView()), 800);
    // A tileset opens on the default settings unless the link carries its
    // own (a view, a preset, a chapter); a plain capture keeps 'no rule'.
    // Choosing a tileset reloads with ?scene=, and that used to skip this
    // and open a 1 x 1 world.
    // The manifest can name its own (view: a preset in views.json).
    if (!pendingPreset && !plainScene && !params.has('v') && !params.has('chapter')) {
      pendingPreset = (m && typeof m.view === 'string' && m.view) || 'readme';
    }
    loadAtlas(name);
    loadCatalogue();
    if (Object.keys(presets).length) setChapters(presets, storyOf());
    // The height field arrives after the splats; the scene shows meanwhile.
    // A manifest's terrain ({gen, seed, size}) is the ground it was set up
    // on, used unless the link names one.
    const own = m && m.terrain && m.terrain.gen ? m.terrain : null;
    if (wantedGen) useGeneratedField(wantedGen, wantedSeed, wantedSize);
    else if (own && !wantedHeight) useGeneratedField(own.gen, own.seed || 0, own.size || 256);
    else useHeightField(wantedHeight || name);
    return true;
  }
  return false;
}

/** Fetch a file with progress; null if it is not there. */
async function fetchBuffer(url, onProgress) {
  const r = await fetch(url);
  if (!r.ok) return null;
  const total = +r.headers.get('content-length') || 0;
  if (!r.body || !total || r.headers.get('content-encoding')) return r.arrayBuffer();
  const reader = r.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    onProgress(got, total);
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const part of parts) { out.set(part, o); o += part.length; }
  return out.buffer;
}

loadProgress(0.03, 'looking for the tileset…');
if (wanted === 'starter') loadStarter();
else loadFirst(wanted ? [wanted] : DEFAULT_SCENES)
  .then((ok) => {
    if (!ok) loadStarter(wanted
      ? `There is no tileset called "${wanted}" here, so this is the starter scene: synthetic tiles made in the browser.`
      : 'The desert tileset did not load, so this is the starter scene: synthetic tiles made in the browser.');
  });

/** Look for a height field beside a tileset and adopt it if there is one. */
async function useHeightField(name) {
  const f = await loadHeightField('./data', name);
  if (f) { adoptField(f); return; }
  // No ground of its own: the desert's, else a generated one. Without any,
  // the settings that wait for ground never came and the world stayed 1 x 1.
  if (name !== 'desert') useHeightField('desert');
  else useGeneratedField('rolling', 0, 256);
}

/** Generate a terrain in the browser (or take it from the browser's cache)
 *  and put the tiles on it. */
async function useGeneratedField(profile, seed, size) {
  const say = (t) => { if (ui.genstatus) ui.genstatus.textContent = t; };
  say(`${profile} ${seed}: starting`);
  if (ui.genbtn) ui.genbtn.disabled = true;
  try {
    const t0 = performance.now();
    const f = await generatedField({ profile, seed, size }, (frac, cached) => {
      if (cached) say(`${profile} ${seed}: from cache`);
      else if (frac < 1) say(`${profile} ${seed}: eroding ${(frac * 100).toFixed(0)}%`);
    });
    adoptField(f);
    // Show the generated terrain in the terrain menu too.
    if (ui.heightpick) {
      let o = ui.heightpick.querySelector('option[data-generated]');
      if (!o) {
        o = document.createElement('option');
        o.dataset.generated = '1';
        ui.heightpick.insertBefore(o, ui.heightpick.children[1] || null);
      }
      o.value = '';
      o.textContent = `made here: ${profile} ${seed} · ${size}`;
      o.selected = true;
    }
    say(`${profile} seed ${seed}, ${size}²`
        + (f.cached ? ' (cached)' : `, ${((performance.now() - t0) / 1000).toFixed(1)} s`));
  } catch (err) {
    say(`could not generate: ${err.message}`);
    console.error(err);
  } finally {
    if (ui.genbtn) ui.genbtn.disabled = false;
  }
}

/** Make a height field the ground under the tiles, however it arrived. */
function adoptField(f) {
  field = f;

  // R32F with NEAREST: the shader does its own bilinear (see fieldAt).
  if (!fieldTex) fieldTex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE3);
  gl.bindTexture(gl.TEXTURE_2D, fieldTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, f.w, f.h, 0, gl.RED, gl.FLOAT,
                f.z);

  // Openness (r), the detail mask (g) and the drainage base (b: the height
  // of the channel each sample drains to, for the valley mist), once per field.
  if (!openTex) openTex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE4);
  gl.bindTexture(gl.TEXTURE_2D, openTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const open = openness(f.z, f.w, f.h);
  f.mask = roughnessMask(f.z, f.w, f.h, open);
  let base = null;
  if (f.w === f.h) {
    try { base = drainBase(f.z, f.w); } catch (e) { console.warn('no drainage base:', e); }
  }
  let zmin = Infinity;
  for (const v of f.z) if (v < zmin) zmin = v;
  const rgb = new Float32Array(f.w * f.h * 3);
  for (let i = 0; i < f.w * f.h; i++) {
    rgb[3 * i] = open[i]; rgb[3 * i + 1] = f.mask[i];
    rgb[3 * i + 2] = base ? base[i] : zmin;
  }
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB32F, f.w, f.h, 0, gl.RGB, gl.FLOAT, rgb);
  // Cover the whole grid by default: a small scale shows a corner of the
  // landscape mirrored many times, which is repetition, not tiling.
  if (ui.reliefscale && !reliefScaleTouched) {
    const want = Math.min(+ui.reliefscale.max, Math.max(4, gridN));
    ui.reliefscale.value = String(want);
    reliefScale = want;
    ui.reliefscalen.textContent = String(want);
    paintSliders();
  }
  field.fitTo(reliefScale * (tileSize || 1));
  markReliefScale();
  if (ui.fieldnote) ui.fieldnote.textContent = f.describe();
  // Presets wait for the ground, since relief and rule settings need it.
  if (plainScene && !pendingPreset && presets.plain) {
    setTimeout(() => applyPreset('plain'), 0);
    console.log('not a tileset: showing the capture as it is (preset "no rule")');
  }
  if (pendingPreset && presets[pendingPreset]) {
    const name = pendingPreset;
    pendingPreset = null;
    setTimeout(() => applyPreset(name), 0);
  }
  // A shared link's view goes last, over whatever preset came first (if
  // the presets file is still on its way, loadPresets applies both).
  if (!pendingPreset) setTimeout(() => applyView(takeView()), 0);
  if (ui.relief && +ui.relief.value <= 0) {
    // With relief at zero a new field is invisible; lift it so it shows.
    ui.relief.value = Math.min(1.0, +ui.relief.max);
    relief = +ui.relief.value;
    ui.reliefn.textContent = relief.toFixed(2);
    paintSliders();
  }
  // New ground: the rule decides again and the cells are rebuilt on it.
  regenerate();
  console.log(`height field: ${f.describe()}`);
}

/** Apply a shared link's view: controls first (grid before the rest, as
 *  presets do), then the camera. */
function applyView(v) {
  if (!v) return;
  const c = v.c || {};
  if ('grid' in c) setControl('grid', c.grid);
  const shares = {};
  for (const [k, val] of Object.entries(c)) {
    const share = /^classshare(\d+)$/.exec(k);
    if (share) shares[share[1]] = val;
    else if (k !== 'grid') setControl(k, val);
  }
  if (Object.keys(shares).length) setShares(shares);
  const m = v.cam;
  if (m) {
    if (m.walk) {
      setWalk(true);
      cam.pos = m.walk.slice();
    } else {
      setWalk(false);
      if (cam.fly) { cam.setFly(false); if (ui.freefly) ui.freefly.checked = false; }
      if (m.t) cam.target = m.t.slice();
      if (m.d) { cam.distance = m.d; zoomGoal = 0; }
    }
    if (m.az != null) cam.azimuth = m.az;
    if (m.el != null) cam.elevation = m.el;
    if (m.fov) setControl('fov', m.fov);
  }
}

const overlayEl = ui.overlay;
function introOpen() { return overlayEl && !overlayEl.classList.contains('hidden'); }

/** What the page around the canvas needs to know (ui.js). */
function viewerState() {
  const meta = tilesetMeta[loadedScene] || {};
  const mpp = cam.fly ? 0
    : 2 * cam.distance * Math.tan(cam.fov * Math.PI / 360) / Math.max(canvas.clientHeight, 1);
  return {
    scene: loadedScene || 'scene', splats: splatCount, plain: plainScene,
    synthetic: !!(loadedManifest && loadedManifest.starter),
    names: materialNames(), colours: classColours, counts: tileClassCounts,
    ruleOn: classOn && classCount > 1,
    collects: tileClassFor(0), exposed: tileClassFor(classCount - 1),
    tiles: patches.length - (variantOf ? variantOf.filter(b => b >= 0).length : 0),
    tileSize, world: gridN * tileSize,
    river: waterOn && field && field.carve ? riverWidth : 0,
    note: meta.note || '', units: meta.units || (loadedManifest && loadedManifest.units) || '',
    metresPerPixel: mpp,
    // How many times the height field repeats across the world (mirrored).
    repeats: field && reliefScale > 0 && reliefScale < gridN ? gridN / reliefScale : 0,
    ground: field ? field.describe() : 'analytic surface',
    // What the tileset has, so the panel shows only the controls that apply.
    wang: !!wangCodes, frames: !!(frameInfo && centres && centres.length),
    decals: !!(decalCopies || centreCopies),
    centres: centres ? centres.length : 0,
  };
}

initUI({
  state: viewerState,
  camera: () => ({ target: cam.target, azimuth: cam.azimuth, elevation: cam.elevation,
                   distance: cam.distance, fov: cam.fov, walk: cam.walk, pos: cam.pos }),
  applyPreset: (name) => applyPreset(name),
});

paintSliders();
frame();
