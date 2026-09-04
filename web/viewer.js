// WebGL2 Gaussian splat renderer.
//
// The projection follows the same maths as bozkir/camera.py, deliberately:
// camera space has +z forward, x right, y down, and the 2D covariance comes
// from the Jacobian of the perspective divide (3DGS Eq. 5). Keeping the two
// implementations line-for-line comparable means the slow Python renderer
// can be used as ground truth when this one looks wrong.

// Printed on load so a stale cached copy is obvious at a glance.
const BUILD = 'bozkir viewer 0.7 (shared tile parts)';
console.log('%c' + BUILD, 'color:#c8a05a');

const STRIDE = 32;             // bytes per splat in the .splat format
const DILATION = 0.3;          // matches DILATION in bozkir/render.py

// ---------------------------------------------------------------- shaders

const VERT = `#version 300 es
precision highp float;
precision highp int;

const float DILATION = ${DILATION.toFixed(4)};   // injected from JS

uniform sampler2D uData;       // RGBA32F, 3 texels per splat
uniform sampler2D uColour;     // RGBA8,   1 texel per splat
uniform mat3 uView;            // world -> camera rotation (rows are axes)
uniform vec3 uEye;
uniform vec3 uOffset;    // world position of the tile being drawn
uniform vec2 uFocal;           // pixels
uniform vec2 uViewport;        // pixels
uniform float uGain;           // splat size multiplier
uniform float uNear;

in vec2 aCorner;               // quad corner in sigma-ish units, -2..2
in uint aIndex;                // splat index, from the sorted order buffer

out vec2 vCorner;
out vec4 vColour;

vec4 fetch(uint i, int slot) {
  int t = int(i) * 3 + slot;
  return texelFetch(uData, ivec2(t & 2047, t >> 11), 0);
}

void main() {
  vec4 a = fetch(aIndex, 0);   // position.xyz, opacity
  vec4 b = fetch(aIndex, 1);   // scale.xyz
  vec4 q = fetch(aIndex, 2);   // rotation w,x,y,z

  vec3 cam = uView * (a.xyz + uOffset - uEye);
  if (cam.z < uNear) {         // behind the camera: collapse the quad
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }

  // Rotation matrix from the quaternion, same expansion as quat_to_matrix.
  float w = q.x, x = q.y, y = q.z, z = q.w;
  mat3 R = mat3(
    1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z),       2.0*(x*z - w*y),
    2.0*(x*y - w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x),
    2.0*(x*z + w*y),       2.0*(y*z - w*x),       1.0 - 2.0*(x*x + y*y)
  );
  mat3 M = R * mat3(b.x, 0.0, 0.0, 0.0, b.y, 0.0, 0.0, 0.0, b.z);
  mat3 sigma = M * transpose(M);

  // The Taylor expansion is only accurate near the optical axis, so clamp
  // the projected position to a slightly enlarged frustum before taking
  // the derivative. Same guard band as project_perspective.
  float invz = 1.0 / cam.z;
  float limx = 1.3 * uViewport.x * 0.5 / uFocal.x;
  float limy = 1.3 * uViewport.y * 0.5 / uFocal.y;
  float cx = clamp(cam.x * invz, -limx, limx) * cam.z;
  float cy = clamp(cam.y * invz, -limy, limy) * cam.z;

  mat3 J = mat3(
    uFocal.x * invz, 0.0, 0.0,
    0.0, uFocal.y * invz, 0.0,
    -uFocal.x * cx * invz * invz, -uFocal.y * cy * invz * invz, 0.0
  );
  mat3 T = J * uView;
  mat3 c3 = T * sigma * transpose(T);

  float ca = c3[0][0] + DILATION;
  float cb = c3[1][0];
  float cc = c3[1][1] + DILATION;

  // Eigen-decomposition of the 2x2 covariance gives the ellipse axes.
  float mid = 0.5 * (ca + cc);
  float rad = sqrt(max(mid * mid - (ca * cc - cb * cb), 0.1));
  float l1 = mid + rad;
  float l2 = max(mid - rad, 0.1);
  if (l1 < 0.15) {             // smaller than a pixel: not worth a quad
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }

  vec2 dir = normalize(vec2(cb, l1 - ca));
  // Axis length sqrt(2*lambda) pairs with alpha = exp(-dot(corner,corner)):
  // at |corner| = 1 that is one sqrt(2)-sigma step, where a true Gaussian
  // has fallen to exp(-1). The two conventions agree.
  vec2 major = min(sqrt(2.0 * l1), 1024.0) * dir * uGain;
  vec2 minor = min(sqrt(2.0 * l2), 1024.0) * vec2(dir.y, -dir.x) * uGain;

  vec2 centre = vec2(uFocal.x * cam.x * invz, uFocal.y * cam.y * invz);
  vec2 offset = aCorner.x * major + aCorner.y * minor;
  vec2 px = centre + offset;

  // Pixels from the centre of the frame to clip space. Screen y points
  // down, clip y points up.
  gl_Position = vec4(
    2.0 * px.x / uViewport.x,
    -2.0 * px.y / uViewport.y,
    0.0, 1.0);

  vCorner = aCorner;
  int ci = int(aIndex);
  vColour = texelFetch(uColour, ivec2(ci & 2047, ci >> 11), 0);
}
`;

const FRAG = `#version 300 es
precision highp float;

in vec2 vCorner;
in vec4 vColour;
out vec4 oColour;

void main() {
  float p = -dot(vCorner, vCorner);
  if (p < -4.0) discard;                  // beyond ~2.8 sigma
  float alpha = exp(p) * vColour.a;
  if (alpha < 0.004) discard;             // below one 8-bit level
  oColour = vec4(vColour.rgb * alpha, alpha);   // premultiplied
}
`;

// ------------------------------------------------------------------ setup

function compile(gl, src, type) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(s) || '(no log)';
        const kind = type === gl.VERTEX_SHADER ? 'vertex' : 'fragment';
        // Drivers report "ERROR: 0:37: ...". Pull the line out and show it,
        // because a line number without the line is nearly useless.
        const lines = src.split('\n');
        const context = [...log.matchAll(/\d+:(\d+)/g)]
            .map(m => Number(m[1]))
            .filter(n => n > 0 && n <= lines.length)
            .map(n => `  ${n}: ${lines[n - 1].trim()}`)
            .join('\n');
        throw new Error(`${kind} shader failed to compile\n\n${log}` +
            (context ? `\n${context}` : ''));
    }
    return s;
}

function program(gl, vs, fs) {
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl, vs, gl.VERTEX_SHADER));
    gl.attachShader(p, compile(gl, fs, gl.FRAGMENT_SHADER));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(p));
    }
    return p;
}

/** Unpack a .splat buffer into GPU-ready arrays. Mirrors scripts/export_splat.py. */
function unpack(buffer) {
    const bytes = new Uint8Array(buffer);
    const n = Math.floor(bytes.length / STRIDE);
    const f32 = new Float32Array(buffer);

    const positions = new Float32Array(n * 3);
    const data = new Float32Array(n * 3 * 4);     // 3 RGBA32F texels per splat
    const colour = new Uint8Array(n * 4);

    for (let i = 0; i < n; i++) {
        const f = i * 8;                            // 32 bytes = 8 floats
        const b = i * STRIDE;

        const px = f32[f + 0], py = f32[f + 1], pz = f32[f + 2];
        positions[3 * i] = px;
        positions[3 * i + 1] = py;
        positions[3 * i + 2] = pz;

        const d = i * 12;
        data[d + 0] = px;
        data[d + 1] = py;
        data[d + 2] = pz;
        data[d + 3] = bytes[b + 27] / 255;          // opacity

        data[d + 4] = f32[f + 3];                   // scale
        data[d + 5] = f32[f + 4];
        data[d + 6] = f32[f + 5];
        data[d + 7] = 0;

        // uint8 -> [-1, 1], then normalise: quantisation leaves it slightly off.
        let qw = (bytes[b + 28] - 128) / 128;
        let qx = (bytes[b + 29] - 128) / 128;
        let qy = (bytes[b + 30] - 128) / 128;
        let qz = (bytes[b + 31] - 128) / 128;
        const len = Math.hypot(qw, qx, qy, qz) || 1;
        data[d + 8] = qw / len;
        data[d + 9] = qx / len;
        data[d + 10] = qy / len;
        data[d + 11] = qz / len;

        colour[4 * i] = bytes[b + 24];
        colour[4 * i + 1] = bytes[b + 25];
        colour[4 * i + 2] = bytes[b + 26];
        colour[4 * i + 3] = bytes[b + 27];
    }
    return { n, positions, data, colour };
}

function makeTexture(gl, unit, internal, w, h, format, type, pixels) {
    const t = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, pixels);
    return t;
}

// ----------------------------------------------------------------- camera

/** Orbit camera. Matches orbit_camera() in bozkir/camera.py: z is up,
 *  elevation 0 looks along the ground, 90 looks straight down. */
class Orbit {
    constructor() {
        this.target = [0, 0, 0];
        this.distance = 5;
        this.azimuth = 45;
        this.elevation = 25;
        this.fov = 60;
    }

    eye() {
        const az = this.azimuth * Math.PI / 180;
        const el = this.elevation * Math.PI / 180;
        return [
            this.target[0] + this.distance * Math.cos(el) * Math.cos(az),
            this.target[1] + this.distance * Math.cos(el) * Math.sin(az),
            this.target[2] + this.distance * Math.sin(el),
        ];
    }

    /** World -> camera rotation, rows [right, down, forward]. */
    basis() {
        const e = this.eye();
        let f = [this.target[0] - e[0], this.target[1] - e[1], this.target[2] - e[2]];
        const fl = Math.hypot(...f) || 1;
        f = f.map(v => v / fl);

        let up = [0, 0, 1];
        if (Math.abs(f[2]) > 0.999) up = [1, 0, 0];

        let r = [f[1] * up[2] - f[2] * up[1],
        f[2] * up[0] - f[0] * up[2],
        f[0] * up[1] - f[1] * up[0]];
        const rl = Math.hypot(...r) || 1;
        r = r.map(v => v / rl);

        const d = [f[1] * r[2] - f[2] * r[1],
        f[2] * r[0] - f[0] * r[2],
        f[0] * r[1] - f[1] * r[0]];
        return { right: r, down: d, forward: f, eye: e };
    }
}

// ------------------------------------------------------------------- main

const canvas = document.getElementById('gl');
const overlay = document.getElementById('overlay');
const bar = document.querySelector('#bar i');
const ui = {
    n: document.getElementById('n'),
    drawn: document.getElementById('drawn'),
    fps: document.getElementById('fps'),
    sortms: document.getElementById('sortms'),
    azim: document.getElementById('azim'),
    elev: document.getElementById('elev'),
    dist: document.getElementById('dist'),
    cmd: document.getElementById('cmd'),
    gz: document.getElementById('gz'),
    grid: document.getElementById('grid'),
    gridn: document.getElementById('gridn'),
    used: document.getElementById('used'),
    usedn: document.getElementById('usedn'),
    wang: document.getElementById('wangnote'),
};

function fail(title, detail) {
    overlay.classList.remove('hidden');
    overlay.querySelector('.msg').innerHTML =
        `<b>${title}</b><pre style="text-align:left;white-space:pre-wrap;` +
        `font-size:11px;color:#c07a6a;max-height:60vh;overflow:auto">` +
        `${String(detail).replace(/[<>&]/g, c => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' }[c]))}</pre>`;
    console.error(title, detail);
}

const gl = canvas.getContext('webgl2', {
    antialias: false, alpha: false, premultipliedAlpha: false,
});
if (!gl) {
    fail('No WebGL2', 'This browser cannot run the renderer.');
    throw new Error('webgl2 unavailable');
}
if (!gl.getExtension('EXT_color_buffer_float') &&
    !gl.getExtension('OES_texture_float_linear')) {
    // Not fatal: we only sample float textures with NEAREST, which core
    // WebGL2 supports. Noted in case a driver disagrees.
    console.warn('float texture extensions unavailable; NEAREST sampling only');
}

let prog;
try {
    prog = program(gl, VERT, FRAG);
} catch (err) {
    fail('Shader error', err.message);
    throw err;
}
console.log('shaders compiled');
gl.useProgram(prog);

const loc = {
    view: gl.getUniformLocation(prog, 'uView'),
    eye: gl.getUniformLocation(prog, 'uEye'),
    offset: gl.getUniformLocation(prog, 'uOffset'),
    focal: gl.getUniformLocation(prog, 'uFocal'),
    viewport: gl.getUniformLocation(prog, 'uViewport'),
    gain: gl.getUniformLocation(prog, 'uGain'),
    near: gl.getUniformLocation(prog, 'uNear'),
    data: gl.getUniformLocation(prog, 'uData'),
    colour: gl.getUniformLocation(prog, 'uColour'),
};
gl.uniform1i(loc.data, 0);
gl.uniform1i(loc.colour, 1);
gl.uniform1f(loc.near, 0.05);

// One quad, drawn once per splat.
const quad = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quad);
gl.bufferData(gl.ARRAY_BUFFER,
    new Float32Array([-2, -2, 2, -2, -2, 2, 2, 2]), gl.STATIC_DRAW);
const aCorner = gl.getAttribLocation(prog, 'aCorner');
if (aCorner < 0) fail('Shader error', 'attribute aCorner was optimised away');
gl.enableVertexAttribArray(aCorner);
gl.vertexAttribPointer(aCorner, 2, gl.FLOAT, false, 0, 0);

// Per-instance splat index, rewritten whenever the sort finishes.
const indexBuf = gl.createBuffer();
const aIndex = gl.getAttribLocation(prog, 'aIndex');
if (aIndex < 0) fail('Shader error', 'attribute aIndex was optimised away');
gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
gl.enableVertexAttribArray(aIndex);
gl.vertexAttribIPointer(aIndex, 1, gl.UNSIGNED_INT, 0, 0);
gl.vertexAttribDivisor(aIndex, 1);

gl.disable(gl.DEPTH_TEST);
gl.enable(gl.BLEND);
// Premultiplied 'over', back to front. Matches render.py's compositing.
gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA,
    gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
gl.clearColor(0, 0, 0, 1);

let lineProg, lineLoc, linePosBuf, lineRGBBuf, lineCount = 0;
try {
    lineProg = program(gl, LINE_VERT, LINE_FRAG);
} catch (e) {
    fail(e);
}
lineLoc = {
    view: gl.getUniformLocation(lineProg, 'uView'),
    eye: gl.getUniformLocation(lineProg, 'uEye'),
    focal: gl.getUniformLocation(lineProg, 'uFocal'),
    viewport: gl.getUniformLocation(lineProg, 'uViewport'),
    near: gl.getUniformLocation(lineProg, 'uNear'),
    alpha: gl.getUniformLocation(lineProg, 'uAlpha'),
    pos: gl.getAttribLocation(lineProg, 'aPos'),
    rgb: gl.getAttribLocation(lineProg, 'aRGB'),
};
linePosBuf = gl.createBuffer();
lineRGBBuf = gl.createBuffer();

// Two colours per axis, as in the GSWT figures: warm for north/south,
// cool for east/west, so a glance tells you which constraint you are
// looking at.
const EDGE_RGB = {
    h: [[0.88, 0.32, 0.32], [0.35, 0.78, 0.35], [0.95, 0.60, 0.20],
    [0.85, 0.40, 0.80]],
    v: [[0.35, 0.63, 0.88], [0.88, 0.75, 0.35], [0.45, 0.85, 0.82],
    [0.70, 0.55, 0.95]],
};
const DIAGONAL_RGB = [0.55, 0.55, 0.55];

let showEdges = false, showDiagonals = false;

/** Rebuild the overlay geometry for the current grid.
 *
 *  Each cell contributes its four boundary segments, coloured by that
 *  edge's colour code, and optionally its two diagonals, which are where
 *  the four source patches meet inside the tile. Shared boundaries get
 *  drawn twice, by both neighbours - if the two disagree the line shows
 *  two colours, which is the failure this view exists to reveal. */
function buildOverlay() {
    if (!tileSize || !cells.length) { lineCount = 0; return; }
    const h = tileSize / 2;
    const lift = tileSize * 0.02;      // sit just above the ground
    const pos = [], rgb = [];

    const seg = (x0, y0, x1, y1, c) => {
        pos.push(x0, y0, lift, x1, y1, lift);
        rgb.push(c[0], c[1], c[2], c[0], c[1], c[2]);
    };

    for (const cell of cells) {
        const { x, y } = cell;
        if (showEdges) {
            const code = wangCodes ? wangCodes[cell.patch] : [0, 0, 0, 0];
            const cn = EDGE_RGB.h[code[0] % 4], ce = EDGE_RGB.v[code[1] % 4];
            const cs = EDGE_RGB.h[code[2] % 4], cw = EDGE_RGB.v[code[3] % 4];
            // Inset slightly so the two tiles sharing a boundary draw side by
            // side instead of on top of each other.
            const k = h * 0.94;
            seg(x - k, y + k, x + k, y + k, cn);
            seg(x + k, y - k, x + k, y + k, ce);
            seg(x - k, y - k, x + k, y - k, cs);
            seg(x - k, y - k, x - k, y + k, cw);
        }
        if (showDiagonals) {
            seg(x - h, y - h, x + h, y + h, DIAGONAL_RGB);
            seg(x - h, y + h, x + h, y - h, DIAGONAL_RGB);
        }
    }

    lineCount = pos.length / 3;
    gl.bindBuffer(gl.ARRAY_BUFFER, linePosBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pos), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, lineRGBBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(rgb), gl.DYNAMIC_DRAW);
}

const cam = new Orbit();
console.log('starting sort worker');
const worker = new Worker('./sort-worker.js');

let splatCount = 0;
let sortedReady = false;
let sortPending = false;
let lastSortKey = '';
let sortMs = 0;
let frames = 0, fpsTime = performance.now();
let sortingEnabled = true;
let gain = 1;
let identityOrder = null;   // file order, for the sorting-off comparison
let patches = [{ start: 0, count: 0 }];   // slices of the splat buffer
let tileSize = 0;           // world units; 0 means "not a tile set"
let gridN = 1;              // grid is gridN x gridN cells
let cells = [];             // {x, y, patch}
let seed = 1;
let wangCodes = null;       // [n, e, s, w] per tile, when the set is a Wang set
let tileParts = null;       // which parts each tile is assembled from
let usedPatches = 0;        // how many of the exported patches to draw from
let drawnSplats = 0, drawCalls = 0;

worker.onmessage = (e) => {
    if (e.data.type === 'sorted') {
        if (!sortingEnabled) { sortPending = false; return; }   // arrived too late
        gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
        gl.bufferData(gl.ARRAY_BUFFER, new Uint32Array(e.data.order), gl.DYNAMIC_DRAW);
        sortMs = e.data.ms;
        sortedReady = true;
        sortPending = false;
    }
};

const AXES = [
    { v: [1, 0, 0], label: 'X', colour: '#d9534f' },
    { v: [0, 1, 0], label: 'Y', colour: '#5cb85c' },
    { v: [0, 0, 1], label: 'Z', colour: '#4a90d9' },
];

/** Blender-style orientation gizmo: the world axes seen from the camera.
 *  Orthographic on purpose - it shows direction, not position. */
function drawGizmo(b) {
    const R = 30, cx = 46, cy = 46;
    // Sort back to front so axes pointing away are drawn under the others.
    const arms = [];
    for (const a of AXES) {
        for (const s of [1, -1]) {
            const v = [a.v[0] * s, a.v[1] * s, a.v[2] * s];
            arms.push({
                x: cx + R * (v[0] * b.right[0] + v[1] * b.right[1] + v[2] * b.right[2]),
                y: cy + R * (v[0] * b.down[0] + v[1] * b.down[1] + v[2] * b.down[2]),
                z: v[0] * b.forward[0] + v[1] * b.forward[1] + v[2] * b.forward[2],
                label: s > 0 ? a.label : '',
                colour: a.colour,
                positive: s > 0,
            });
        }
    }
    arms.sort((p, q) => q.z - p.z);

    ui.gz.innerHTML = arms.map(a => {
        const dim = a.positive ? 1 : 0.35;
        const line = `<line x1="${cx}" y1="${cy}" x2="${a.x.toFixed(1)}" ` +
            `y2="${a.y.toFixed(1)}" stroke="${a.colour}" stroke-width="1.6" ` +
            `opacity="${dim}"/>`;
        const dot = `<circle cx="${a.x.toFixed(1)}" cy="${a.y.toFixed(1)}" r="7" ` +
            `fill="${a.positive ? a.colour : '#16191b'}" stroke="${a.colour}" ` +
            `stroke-width="1.4" opacity="${dim}"/>`;
        const txt = a.label
            ? `<text x="${a.x.toFixed(1)}" y="${(a.y + 3.5).toFixed(1)}" ` +
            `text-anchor="middle" fill="#0b0d0e">${a.label}</text>` : '';
        return line + dot + txt;
    }).join('');
}

/** Lay out gridN x gridN cells.
 *
 *  With a Wang tile set, each cell's west colour is fixed by the cell to
 *  its left and its south colour by the cell below; north and east stay
 *  free. A complete set always has a tile that fits, so this never
 *  backtracks, and the free choices are what stop the terrain repeating.
 *
 *  Without edge codes it falls back to picking a patch at random, which
 *  is an array of copies rather than a tiling. */
function buildGrid() {
    cells = [];
    if (!tileSize) { cells = [{ x: 0, y: 0, patch: 0 }]; return; }
    usedPatches = Math.max(1, Math.min(usedPatches || patches.length,
        patches.length));
    let s = seed;
    const rand = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    const half = (gridN - 1) / 2;

    const chosen = new Int32Array(gridN * gridN).fill(-1);
    for (let j = 0; j < gridN; j++) {
        for (let i = 0; i < gridN; i++) {
            let pick;
            if (wangCodes) {
                const west = i > 0 ? wangCodes[chosen[j * gridN + i - 1]][1] : -1;
                const south = j > 0 ? wangCodes[chosen[(j - 1) * gridN + i]][0] : -1;
                const fits = [];
                for (let k = 0; k < wangCodes.length; k++) {
                    const c = wangCodes[k];
                    if (west >= 0 && c[3] !== west) continue;
                    if (south >= 0 && c[2] !== south) continue;
                    fits.push(k);
                }
                pick = fits.length
                    ? fits[Math.floor(rand() * fits.length) % fits.length] : 0;
            } else {
                pick = Math.floor(rand() * usedPatches) % usedPatches;
            }
            chosen[j * gridN + i] = pick;
            cells.push({
                x: (i - half) * tileSize, y: (j - half) * tileSize,
                patch: pick
            });
        }
    }
}

function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.floor(canvas.clientWidth * dpr);
    const h = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
        gl.viewport(0, 0, w, h);
    }
}

function load(buffer, manifest) {
    const { n, positions, data, colour } = unpack(buffer);
    splatCount = n;

    if (manifest && manifest.parts && manifest.tiles) {
        // A Wang set stores the distinct triangles once; each tile is four
        // references into them. Sixteen tiles share eight parts, so this is
        // eight times less data to hold, sort and upload.
        patches = manifest.parts.map(p => ({ start: p.start, count: p.count }));
        tileParts = manifest.tiles.map(t => t.parts);
        wangCodes = manifest.tiles.map(t => [t.n, t.e, t.s, t.w]);
        tileSize = manifest.size || 0;
    } else if (manifest && manifest.tiles && manifest.tiles.length) {
        patches = manifest.tiles.map(t => ({ start: t.start, count: t.count }));
        tileSize = manifest.size || 0;
        tileParts = null;
        wangCodes = manifest.wang
            ? manifest.tiles.map(t => [t.n, t.e, t.s, t.w]) : null;
    } else {
        patches = [{ start: 0, count: n }];
        tileSize = 0;
        wangCodes = null;
        tileParts = null;
    }
    buildGrid();

    const texels = n * 3;
    const w = 2048;
    const h = Math.ceil(texels / w);
    const padded = new Float32Array(w * h * 4);
    padded.set(data.subarray(0, Math.min(data.length, padded.length)));
    makeTexture(gl, 0, gl.RGBA32F, w, h, gl.RGBA, gl.FLOAT, padded);

    const ch = Math.ceil(n / w);
    const cpad = new Uint8Array(w * ch * 4);
    cpad.set(colour);
    makeTexture(gl, 1, gl.RGBA8, w, ch, gl.RGBA, gl.UNSIGNED_BYTE, cpad);

    // File order, used before the first sort returns and whenever depth
    // sorting is switched off.
    identityOrder = new Uint32Array(n);
    for (let i = 0; i < n; i++) identityOrder[i] = i;
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
    gl.bufferData(gl.ARRAY_BUFFER, identityOrder, gl.DYNAMIC_DRAW);
    sortedReady = true;

    // Frame the scene: median centre, distance from the interquartile spread.
    const xs = [], ys = [], zs = [];
    const step = Math.max(1, Math.floor(n / 20000));
    for (let i = 0; i < n; i += step) {
        xs.push(positions[3 * i]);
        ys.push(positions[3 * i + 1]);
        zs.push(positions[3 * i + 2]);
    }
    const q = (arr, p) => {
        const a = arr.slice().sort((u, v) => u - v);
        return a[Math.floor(p * (a.length - 1))];
    };
    cam.target = [(q(xs, .25) + q(xs, .75)) / 2,
    (q(ys, .25) + q(ys, .75)) / 2,
    (q(zs, .25) + q(zs, .75)) / 2];
    cam.distance = Math.max(
        2.5 * Math.max(q(xs, .75) - q(xs, .25), q(ys, .75) - q(ys, .25)), 0.5);
    if (tileSize) { cam.target = [0, 0, cam.target[2]]; cam.elevation = 12; }

    const pos = positions.slice();
    worker.postMessage({ type: 'init', positions: pos.buffer, patches },
        [pos.buffer]);

    console.log(`loaded ${n} splats`);
    ui.n.textContent = n.toLocaleString() +
        (patches.length > 1 ? ` in ${patches.length}` : '');
    const nTiles = wangCodes ? wangCodes.length : patches.length;
    ui.grid.disabled = !tileSize;
    // With a Wang set the arrangement is decided by the matching rule, so
    // restricting how many tiles may appear would break it.
    ui.used.disabled = !tileSize || patches.length < 2 || !!wangCodes;
    ui.wang.textContent = wangCodes
        ? `wang: ${nTiles} tiles from ${patches.length} shared parts, ` +
        `${manifest.colours || 2} colours per axis`
        : 'random placement, edges do not match';
    ui.used.max = patches.length;
    ui.used.value = usedPatches = patches.length;
    ui.usedn.textContent = `${patches.length} of ${patches.length}`;
    overlay.classList.add('hidden');
}

function frame() {
    resize();
    const b = cam.basis();

    if (splatCount) {
        // Re-sort only when the view has actually moved enough to matter.
        const key = [b.forward, b.eye].flat().map(v => v.toFixed(2)).join(',');
        if (sortingEnabled && key !== lastSortKey && !sortPending) {
            lastSortKey = key;
            sortPending = true;
            worker.postMessage({ type: 'sort', forward: b.forward, eye: b.eye });
        }

        const fy = (canvas.height / 2) / Math.tan(cam.fov * Math.PI / 360);
        gl.uniformMatrix3fv(loc.view, false, new Float32Array([
            b.right[0], b.down[0], b.forward[0],
            b.right[1], b.down[1], b.forward[1],
            b.right[2], b.down[2], b.forward[2],
        ]));
        gl.uniform3fv(loc.eye, new Float32Array(b.eye));
        gl.uniform2f(loc.focal, fy, fy);
        gl.uniform2f(loc.viewport, canvas.width, canvas.height);
        gl.uniform1f(loc.gain, gain);
    }

    drawGizmo(b);

    gl.clear(gl.COLOR_BUFFER_BIT);
    drawnSplats = 0;
    drawCalls = 0;
    if (splatCount && sortedReady) {
        // Cells are drawn far to near and composited with 'over'. Splats are
        // only sorted within a patch, never across cells - which is precisely
        // the approximation that produces the boundary artifact.
        const visible = [];
        for (const c of cells) {
            const dx = c.x - b.eye[0], dy = c.y - b.eye[1], dz = -b.eye[2];
            const z = dx * b.forward[0] + dy * b.forward[1] + dz * b.forward[2];
            if (z < -tileSize) continue;                       // fully behind
            // Cheap frustum test: how far off-axis the cell centre sits.
            const sx = dx * b.right[0] + dy * b.right[1] + dz * b.right[2];
            const sy = dx * b.down[0] + dy * b.down[1] + dz * b.down[2];
            const reach = tileSize * 1.5 + Math.max(z, 0.01) *
                Math.tan(cam.fov * Math.PI / 360) * (canvas.width / canvas.height);
            if (Math.abs(sx) > reach || Math.abs(sy) > reach) continue;
            visible.push({ c, z });
        }
        visible.sort((p, q) => q.z - p.z);

        for (const { c } of visible) {
            // A Wang tile is four shared parts drawn at the same offset; a plain
            // tile set is one patch. Either way the splats within a part are
            // sorted, and parts are not sorted against each other.
            const ids = tileParts ? tileParts[c.patch] : [c.patch];
            gl.uniform3f(loc.offset, c.x, c.y, 0);
            gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
            for (const id of ids) {
                const p = patches[id];
                if (!p || !p.count) continue;
                gl.vertexAttribIPointer(aIndex, 1, gl.UNSIGNED_INT, 0, p.start * 4);
                gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, p.count);
                drawnSplats += p.count;
                drawCalls++;
            }
        }
    }

    if (lineCount) {
        gl.useProgram(lineProg);
        gl.uniformMatrix3fv(lineLoc.view, false, new Float32Array([
            b.right[0], b.down[0], b.forward[0],
            b.right[1], b.down[1], b.forward[1],
            b.right[2], b.down[2], b.forward[2],
        ]));
        gl.uniform3fv(lineLoc.eye, new Float32Array(b.eye));
        const fy = (canvas.height / 2) / Math.tan(cam.fov * Math.PI / 360);
        gl.uniform2f(lineLoc.focal, fy, fy);
        gl.uniform2f(lineLoc.viewport, canvas.width, canvas.height);
        gl.uniform1f(lineLoc.near, 0.05);
        gl.uniform1f(lineLoc.alpha, 0.85);

        gl.bindBuffer(gl.ARRAY_BUFFER, linePosBuf);
        gl.enableVertexAttribArray(lineLoc.pos);
        gl.vertexAttribPointer(lineLoc.pos, 3, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(lineLoc.pos, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, lineRGBBuf);
        gl.enableVertexAttribArray(lineLoc.rgb);
        gl.vertexAttribPointer(lineLoc.rgb, 3, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(lineLoc.rgb, 0);

        gl.drawArrays(gl.LINES, 0, lineCount);

        gl.disableVertexAttribArray(lineLoc.pos);
        gl.disableVertexAttribArray(lineLoc.rgb);
        gl.useProgram(prog);
        gl.bindBuffer(gl.ARRAY_BUFFER, quad);
        gl.enableVertexAttribArray(aCorner);
        gl.vertexAttribPointer(aCorner, 2, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
        gl.enableVertexAttribArray(aIndex);
        gl.vertexAttribIPointer(aIndex, 1, gl.UNSIGNED_INT, 0, 0);
        gl.vertexAttribDivisor(aIndex, 1);
    }

    frames++;
    const now = performance.now();
    if (now - fpsTime > 500) {
        ui.fps.textContent = (frames * 1000 / (now - fpsTime)).toFixed(0);
        ui.drawn.textContent = drawnSplats.toLocaleString() +
            (drawCalls > 1 ? ` / ${drawCalls} cells` : '');
        ui.sortms.textContent = sortMs ? sortMs.toFixed(0) + ' ms' : '—';
        frames = 0;
        fpsTime = now;
    }
    ui.azim.textContent = cam.azimuth.toFixed(0) + '\u00b0';
    ui.elev.textContent = cam.elevation.toFixed(0) + '\u00b0';
    ui.dist.textContent = cam.distance.toFixed(2);
    ui.cmd.textContent =
        `--azim ${cam.azimuth.toFixed(0)} --elev ${cam.elevation.toFixed(0)} ` +
        `--dist ${cam.distance.toFixed(2)} --fov ${cam.fov.toFixed(0)}`;

    requestAnimationFrame(frame);
}

// -------------------------------------------------------------- interaction

let dragging = false, lastX = 0, lastY = 0;
canvas.addEventListener('pointerdown', (e) => {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointerup', (e) => {
    dragging = false; canvas.releasePointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    cam.azimuth = ((cam.azimuth - (e.clientX - lastX) * 0.3) % 360 + 360) % 360;
    cam.elevation = Math.max(-89, Math.min(89,
        cam.elevation + (e.clientY - lastY) * 0.3));
    lastX = e.clientX; lastY = e.clientY;
});
canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    cam.distance = Math.max(0.05, cam.distance * Math.exp(e.deltaY * 0.001));
}, { passive: false });

const held = new Set();
addEventListener('keydown', (e) => held.add(e.key.toLowerCase()));
addEventListener('keyup', (e) => held.delete(e.key.toLowerCase()));
setInterval(() => {
    if (!held.size || !splatCount) return;
    const step = cam.distance * 0.02;
    const b = cam.basis();
    const flat = [b.forward[0], b.forward[1], 0];
    const fl = Math.hypot(flat[0], flat[1]) || 1;
    const fwd = [flat[0] / fl, flat[1] / fl, 0];
    const right = [b.right[0], b.right[1], 0];
    const rl = Math.hypot(right[0], right[1]) || 1;
    const rgt = [right[0] / rl, right[1] / rl, 0];

    const move = (v, s) => { for (let i = 0; i < 3; i++) cam.target[i] += v[i] * s; };
    if (held.has('w')) move(fwd, step);
    if (held.has('s')) move(fwd, -step);
    if (held.has('d')) move(rgt, step);
    if (held.has('a')) move(rgt, -step);
    if (held.has('e')) cam.target[2] += step;
    if (held.has('q')) cam.target[2] -= step;
}, 16);

document.getElementById('file').addEventListener('change', async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    overlay.classList.remove('hidden');
    overlay.querySelector('.msg b').textContent = 'loading ' + f.name;
    bar.style.width = '30%';
    try {
        const buf = await f.arrayBuffer();
        bar.style.width = '70%';
        load(buf);
        bar.style.width = '100%';
    } catch (err) {
        fail('Could not load ' + f.name, err.stack || err.message);
    }
});

document.getElementById('fov').addEventListener('input', (e) => {
    cam.fov = +e.target.value;
});
document.getElementById('gain').addEventListener('input', (e) => {
    gain = +e.target.value;
});
document.getElementById('sorting').addEventListener('change', (e) => {
    sortingEnabled = e.target.checked;
    lastSortKey = '';                       // force a re-sort when switched on
    if (!sortingEnabled && identityOrder) {
        // Show the unsorted result straight away rather than leaving the last
        // sorted order on screen until the camera happens to move.
        gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
        gl.bufferData(gl.ARRAY_BUFFER, identityOrder, gl.DYNAMIC_DRAW);
        sortMs = 0;
    }
});
ui.grid.addEventListener('input', (e) => {
    gridN = +e.target.value;
    ui.gridn.textContent = `${gridN} x ${gridN}`;
    buildGrid();
    buildOverlay();
    buildOverlay();
});
ui.used.addEventListener('input', (e) => {
    usedPatches = +e.target.value;
    ui.usedn.textContent = `${usedPatches} of ${patches.length}`;
    buildGrid();
    buildOverlay();
    buildOverlay();
});
document.getElementById('reseed').addEventListener('click', () => {
    seed = (Math.random() * 1e9) | 0;
    buildGrid();
    buildOverlay();
    buildOverlay();
});
document.getElementById('edges').addEventListener('change', (e) => {
    showEdges = e.target.checked;
    buildOverlay();
});
document.getElementById('diagonals').addEventListener('change', (e) => {
    showDiagonals = e.target.checked;
    buildOverlay();
});
document.getElementById('reset').addEventListener('click', () => {
    cam.azimuth = 45; cam.elevation = 25;
});

// Open a file from web/data/ without a click. Query string wins:
//   index.html?scene=garden   ->  ./data/garden.splat
const wanted = new URLSearchParams(location.search).get('scene');
for (const name of (wanted ? [wanted] : ['scene', 'garden'])) {
    Promise.all([
        fetch(`./data/${name}.splat`).then(r => (r.ok ? r.arrayBuffer() : null)),
        fetch(`./data/${name}.json`).then(r => (r.ok ? r.json() : null))
            .catch(() => null),
    ]).then(([b, m]) => { if (b && !splatCount) load(b, m); })
        .catch(() => { });
}

addEventListener('error', (e) => fail('Uncaught error', e.message + '\n' + (e.filename || '') + ':' + (e.lineno || '')));

frame();