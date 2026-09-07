// WebGL2 Gaussian splat renderer with Wang tiling.
//
// The projection follows bozkir/camera.py deliberately: camera space has
// +z forward, x right, y down, and the 2D covariance comes from the
// Jacobian of the perspective divide (3DGS Eq. 5). Keeping the two
// implementations comparable means the slow Python renderer can serve as
// ground truth when this one looks wrong.

const BUILD = 'bozkir viewer 1.7 (sky and fog)';
console.log('%c' + BUILD, 'color:#c8a05a');

const STRIDE = 32;        // bytes per splat in the .splat format
const DILATION = 0.3;     // matches DILATION in bozkir/render.py

// ============================================================== shaders

const SPLAT_VERT = `#version 300 es
precision highp float;
precision highp int;

const float DILATION = ${DILATION.toFixed(4)};

uniform sampler2D uData;      // RGBA32F, 3 texels per splat
uniform sampler2D uColour;    // RGBA8,   1 texel per splat
uniform mat3 uView;           // world -> camera, rows are the camera axes
uniform vec3 uEye;
uniform vec2 uCellXY;         // this tile's centre on the ground plane
uniform float uRelief;        // height field amplitude
uniform float uWave;          // height field wavelength, world units
uniform int uSubdiv;          // tangent frames per tile edge; 1 is GSWT,
                              // 0 means one frame per splat
uniform float uTileSize;
uniform vec3 uEdgeN, uEdgeE, uEdgeS, uEdgeW;
uniform float uEdgeMark;      // band width as a fraction of the tile; 0 is off
uniform float uFade;          // level-of-detail cross-fade weight
uniform vec3 uTint;
uniform float uTintAmount;
uniform vec3 uFogColour;
uniform float uFogDensity;
uniform vec2 uFocal;
uniform vec2 uViewport;
uniform float uGain;
uniform float uNear;

in vec2 aCorner;              // quad corner, -2..2
in uint aIndex;               // splat index, from the sorted order buffer

out vec2 vCorner;
out vec4 vColour;

// The height field, duplicated from height() in the JavaScript. The two
// must agree exactly: the overlay lines are placed from the JS copy and
// the geometry from this one.
float terrainHeight(vec2 p) {
  if (uRelief <= 0.0) return 0.0;
  float f = 1.0 / max(uWave, 0.01);
  return uRelief * (sin(f * p.x) * cos(f * p.y)
    + 0.5 * sin(2.3 * f * p.x + 1.7) * cos(1.9 * f * p.y + 0.4));
}

// The surface gradient at a point, by central differences.
vec2 terrainGrad(vec2 p) {
  if (uRelief <= 0.0) return vec2(0.0);
  float e = max(uWave, 0.01) * 0.01;
  return vec2(
    (terrainHeight(p + vec2(e, 0.0)) - terrainHeight(p - vec2(e, 0.0))),
    (terrainHeight(p + vec2(0.0, e)) - terrainHeight(p - vec2(0.0, e)))
  ) / (2.0 * e);
}

// Two frames, and the difference between them matters.
//
// The Jacobian of the surface map has tangents (1, 0, dh/dx) and
// (0, 1, dh/dy). Those are longer than one unit on a slope, and that extra
// length is real: a step of one unit in x lands sqrt(1 + dx*dx) further
// along the surface. Normalising them throws that away and turns the
// Jacobian into a pure rotation.
//
// Positions use the orthonormal version, because the splats keep their
// ground-plane spacing. Covariance uses the true Jacobian, because the
// spacing measured along the surface has stretched and the splats have to
// stretch with it. Using the rotation for both leaves them the same size
// while their neighbours move apart, which thins the material out on
// steep ground.
mat3 terrainFrame(vec2 g) {
  if (uRelief <= 0.0) return mat3(1.0);
  vec3 a = normalize(vec3(1.0, 0.0, g.x));
  vec3 b = normalize(vec3(0.0, 1.0, g.y));
  return mat3(a, b, normalize(cross(a, b)));
}

mat3 terrainJacobian(vec2 g) {
  if (uRelief <= 0.0) return mat3(1.0);
  vec3 a = vec3(1.0, 0.0, g.x);
  vec3 b = vec3(0.0, 1.0, g.y);
  return mat3(a, b, normalize(cross(a, b)));
}

vec4 fetch(uint i, int slot) {
  int t = int(i) * 3 + slot;
  return texelFetch(uData, ivec2(t & 2047, t >> 11), 0);
}

void main() {
  vec4 a = fetch(aIndex, 0);  // position.xyz, opacity
  vec4 b = fetch(aIndex, 1);  // scale.xyz
  vec4 q = fetch(aIndex, 2);  // rotation w,x,y,z

  // GSWT Eq. 4-5 places a tile on a surface using one tangent frame taken
  // at its centre. That warp is a linearisation: exact in the middle and
  // increasingly wrong toward the edges, so neighbouring tiles warped by
  // different frames disagree along their shared boundary. The gap grows
  // with curvature times the distance the linearisation has to span.
  //
  // uSubdiv shortens that distance without touching the tile. The tile is
  // divided into uSubdiv x uSubdiv sub-cells and each takes its own frame,
  // so the splats keep their size, their material scale and their edge
  // bands - only the frame they are placed by changes. At uSubdiv = 1 this
  // is exactly GSWT. Raising it moves the discontinuity from the tile
  // boundary to the sub-cell boundaries, where it is uSubdiv times smaller.
  // Sub-cells shrink the span but replace one break at the tile boundary
  // with many smaller ones inside it. Taking the limit removes both: each
  // splat is anchored at its own position, so the frame is a continuous
  // function of where you are and no two neighbours can disagree.
  vec2 anchorLocal;
  if (uSubdiv <= 0) {
    anchorLocal = a.xy;
  } else {
    float sub = uTileSize / float(uSubdiv);
    vec2 idx = clamp(floor((a.xy + uTileSize * 0.5) / sub),
                     0.0, float(uSubdiv) - 1.0);
    anchorLocal = (idx + 0.5) * sub - uTileSize * 0.5;
  }
  vec2 anchorWorld = uCellXY + anchorLocal;

  vec2 grad = terrainGrad(anchorWorld);
  mat3 warp = terrainFrame(grad);
  vec3 origin = vec3(anchorWorld, terrainHeight(anchorWorld));
  vec3 world = origin + warp * (a.xyz - vec3(anchorLocal, 0.0));

  vec3 cam = uView * (world - uEye);
  if (cam.z < uNear) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }

  float w = q.x, x = q.y, y = q.z, z = q.w;
  mat3 R = mat3(
    1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z),       2.0*(x*z - w*y),
    2.0*(x*y - w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x),
    2.0*(x*z + w*y),       2.0*(y*z - w*x),       1.0 - 2.0*(x*x + y*y)
  );
  // Sigma' = W Sigma W^T with W the tangent frame, which folds into the
  // same product: (W R S)(W R S)^T.
  mat3 M = terrainJacobian(grad) * R
         * mat3(b.x, 0.0, 0.0, 0.0, b.y, 0.0, 0.0, 0.0, b.z);
  mat3 sigma = M * transpose(M);

  // The Taylor expansion is only accurate near the optical axis, so clamp
  // the projected position to a slightly enlarged frustum first.
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

  float mid = 0.5 * (ca + cc);
  float rad = sqrt(max(mid * mid - (ca * cc - cb * cb), 0.1));
  float l1 = mid + rad;
  float l2 = max(mid - rad, 0.1);
  if (l1 < 0.15) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }

  vec2 dir = normalize(vec2(cb, l1 - ca));
  // Axis length sqrt(2*lambda) pairs with alpha = exp(-dot(corner,corner)):
  // at |corner| = 1 that is one sqrt(2)-sigma step, where a true Gaussian
  // has fallen to exp(-1). The two conventions agree.
  vec2 major = min(sqrt(2.0 * l1), 1024.0) * dir * uGain;
  vec2 minor = min(sqrt(2.0 * l2), 1024.0) * vec2(dir.y, -dir.x) * uGain;

  vec2 centre = vec2(uFocal.x * cam.x * invz, uFocal.y * cam.y * invz);
  vec2 px = centre + aCorner.x * major + aCorner.y * minor;

  gl_Position = vec4(2.0 * px.x / uViewport.x,
                     -2.0 * px.y / uViewport.y, 0.0, 1.0);
  vCorner = aCorner;
  int ci = int(aIndex);
  vec4 col = texelFetch(uColour, ivec2(ci & 2047, ci >> 11), 0);
  vec3 rgb = col.rgb;

  if (uEdgeMark > 0.0 && uTileSize > 0.0) {
    // Paint each edge band with its own colour and wash the interior out,
    // the way the GSWT figures do. The constraint then reads off the
    // geometry itself: two tiles meet correctly when the band running
    // along their shared boundary is one colour on both sides.
    vec2 uv = a.xy / uTileSize;             // tile-local, -0.5 .. 0.5
    float d;
    vec3 ec;
    if (abs(uv.y) >= abs(uv.x)) {
      d = 0.5 - abs(uv.y);
      ec = uv.y > 0.0 ? uEdgeN : uEdgeS;
    } else {
      d = 0.5 - abs(uv.x);
      ec = uv.x > 0.0 ? uEdgeE : uEdgeW;
    }
    float w = 1.0 - smoothstep(0.0, uEdgeMark, d);
    float grey = dot(rgb, vec3(0.299, 0.587, 0.114));
    rgb = mix(vec3(0.55 + 0.45 * grey), ec, w);
  }
  // Distance haze, toward the same colour the sky has at the horizon, so
  // far ground dissolves into it instead of ending at a hard edge.
  float fog = 1.0 - exp(-uFogDensity * max(cam.z, 0.0));
  rgb = mix(rgb, uFogColour, fog);
  vColour = vec4(mix(rgb, uTint, uTintAmount), col.a * uFade);
}
`;

const SPLAT_FRAG = `#version 300 es
precision highp float;
in vec2 vCorner;
in vec4 vColour;
out vec4 oColour;
void main() {
  float p = -dot(vCorner, vCorner);
  if (p < -4.0) discard;
  float alpha = exp(p) * vColour.a;
  if (alpha < 0.004) discard;
  oColour = vec4(vColour.rgb * alpha, alpha);   // premultiplied
}
`;

// Overlay lines use the same projection so the two views agree exactly.
const LINE_VERT = `#version 300 es
precision highp float;
uniform mat3 uView;
uniform vec3 uEye;
uniform vec2 uFocal;
uniform vec2 uViewport;
uniform float uNear;
in vec3 aPos;
in vec3 aRGB;
out vec3 vRGB;
void main() {
  vec3 cam = uView * (aPos - uEye);
  if (cam.z < uNear) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  float invz = 1.0 / cam.z;
  vec2 px = vec2(uFocal.x * cam.x * invz, uFocal.y * cam.y * invz);
  gl_Position = vec4(2.0 * px.x / uViewport.x,
                     -2.0 * px.y / uViewport.y, 0.0, 1.0);
  vRGB = aRGB;
}
`;

const LINE_FRAG = `#version 300 es
precision highp float;
uniform float uAlpha;
in vec3 vRGB;
out vec4 oColour;
void main() { oColour = vec4(vRGB * uAlpha, uAlpha); }
`;

// A gradient standing in for a sky. Cheap, and it does more for how the
// terrain reads than anything else of this size: ground against black has
// no depth, ground against a horizon does.
const SKY_VERT = `#version 300 es
precision highp float;
in vec2 aNDC;
out vec2 vNDC;
void main() { vNDC = aNDC; gl_Position = vec4(aNDC, 0.0, 1.0); }
`;

const SKY_FRAG = `#version 300 es
precision highp float;
uniform mat3 uView;          // world -> camera; its transpose undoes that
uniform vec2 uFocal;
uniform vec2 uViewport;
uniform vec3 uSkyTop;
uniform vec3 uSkyHorizon;
uniform vec3 uGround;
in vec2 vNDC;
out vec4 oColour;
void main() {
  // Reconstruct the world direction this pixel looks along.
  vec2 px = vec2(vNDC.x * uViewport.x * 0.5, -vNDC.y * uViewport.y * 0.5);
  vec3 dir = transpose(uView) * normalize(vec3(px.x / uFocal.x,
                                               px.y / uFocal.y, 1.0));
  float t = dir.z;
  vec3 c = t > 0.0
    ? mix(uSkyHorizon, uSkyTop, pow(clamp(t, 0.0, 1.0), 0.55))
    : mix(uSkyHorizon, uGround, pow(clamp(-t, 0.0, 1.0), 0.5));
  oColour = vec4(c, 1.0);
}
`;

// =============================================================== helpers

const canvas = document.getElementById('gl');
const overlay = document.getElementById('overlay');
const bar = document.querySelector('#bar i');
const ui = {};
for (const id of ['n', 'drawn', 'fps', 'sortms', 'azim', 'elev', 'dist',
    'cmd', 'gz', 'grid', 'gridn', 'used', 'usedn',
    'wangnote', 'edges', 'diagonals', 'tints',
    'relief', 'reliefn', 'reliefscale', 'reliefscalen',
    'band', 'bandn', 'subdiv', 'subdivn', 'seam',
    'lod', 'lodbase', 'lodbasen', 'lodinfo', 'lodcolours',
    'sky', 'fog', 'fogn', 'orbit']) {
    ui[id] = document.getElementById(id);
}

function fail(err) {
    console.error(err);
    overlay.classList.remove('hidden');
    overlay.querySelector('.msg').innerHTML =
        '<b>renderer failed to start</b><pre style="text-align:left;' +
        'white-space:pre-wrap;font-size:11px;color:#c07a5a">' +
        String(err && err.message || err) + '</pre>';
    throw err;
}

const gl = canvas.getContext('webgl2',
    { antialias: false, alpha: false, premultipliedAlpha: false });
if (!gl) {
    overlay.querySelector('.msg').innerHTML =
        '<b>No WebGL2</b>This browser cannot run the renderer.';
    throw new Error('webgl2 unavailable');
}

function compile(src, type) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(s) || 'unknown error';
        const m = log.match(/ERROR:\s*\d+:(\d+)/);
        const lines = src.split('\n');
        const ctx = m ? lines.slice(Math.max(0, m[1] - 3), +m[1] + 1)
            .map((l, i) => `${Math.max(1, m[1] - 2) + i}: ${l}`)
            .join('\n') : '';
        throw new Error(`${type === gl.VERTEX_SHADER ? 'vertex' : 'fragment'} ` +
            `shader failed\n${log}\n${ctx}`);
    }
    return s;
}

function program(vs, fs) {
    const p = gl.createProgram();
    gl.attachShader(p, compile(vs, gl.VERTEX_SHADER));
    gl.attachShader(p, compile(fs, gl.FRAGMENT_SHADER));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(p));
    }
    return p;
}

function uniforms(p, names) {
    const out = {};
    for (const n of names) out[n] = gl.getUniformLocation(p, 'u' + n[0].toUpperCase() + n.slice(1));
    return out;
}

/** Unpack a .splat buffer. Mirrors scripts/export_splat.py. */
function unpack(buffer) {
    const bytes = new Uint8Array(buffer);
    const n = Math.floor(bytes.length / STRIDE);
    const f32 = new Float32Array(buffer);
    const positions = new Float32Array(n * 3);
    const data = new Float32Array(n * 3 * 4);
    const colour = new Uint8Array(n * 4);

    for (let i = 0; i < n; i++) {
        const f = i * 8, b = i * STRIDE, d = i * 12;
        const px = f32[f], py = f32[f + 1], pz = f32[f + 2];
        positions[3 * i] = px; positions[3 * i + 1] = py; positions[3 * i + 2] = pz;
        data[d] = px; data[d + 1] = py; data[d + 2] = pz;
        data[d + 3] = bytes[b + 27] / 255;
        data[d + 4] = f32[f + 3]; data[d + 5] = f32[f + 4]; data[d + 6] = f32[f + 5];
        data[d + 7] = 0;
        let qw = (bytes[b + 28] - 128) / 128, qx = (bytes[b + 29] - 128) / 128;
        let qy = (bytes[b + 30] - 128) / 128, qz = (bytes[b + 31] - 128) / 128;
        const len = Math.hypot(qw, qx, qy, qz) || 1;
        data[d + 8] = qw / len; data[d + 9] = qx / len;
        data[d + 10] = qy / len; data[d + 11] = qz / len;
        colour[4 * i] = bytes[b + 24]; colour[4 * i + 1] = bytes[b + 25];
        colour[4 * i + 2] = bytes[b + 26]; colour[4 * i + 3] = bytes[b + 27];
    }
    return { n, positions, data, colour };
}

function makeTexture(unit, internal, w, h, format, type, pixels) {
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

/** Orbit camera. Matches orbit_camera() in bozkir/camera.py: z is up,
 *  elevation 0 looks along the ground, 90 straight down. */
class Orbit {
    constructor() {
        this.target = [0, 0, 0];
        this.distance = 5;
        this.azimuth = 45;
        this.elevation = 25;
        this.fov = 60;
    }
    eye() {
        const az = this.azimuth * Math.PI / 180, el = this.elevation * Math.PI / 180;
        return [this.target[0] + this.distance * Math.cos(el) * Math.cos(az),
        this.target[1] + this.distance * Math.cos(el) * Math.sin(az),
        this.target[2] + this.distance * Math.sin(el)];
    }
    basis() {
        const e = this.eye();
        let f = [this.target[0] - e[0], this.target[1] - e[1], this.target[2] - e[2]];
        const fl = Math.hypot(...f) || 1;
        f = f.map(v => v / fl);
        let up = Math.abs(f[2]) > 0.999 ? [1, 0, 0] : [0, 0, 1];
        let r = [f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2],
        f[0] * up[1] - f[1] * up[0]];
        const rl = Math.hypot(...r) || 1;
        r = r.map(v => v / rl);
        const d = [f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2],
        f[0] * r[1] - f[1] * r[0]];
        return { right: r, down: d, forward: f, eye: e };
    }
}

// ================================================================= setup

let splatProg, lineProg, skyProg;
try {
    splatProg = program(SPLAT_VERT, SPLAT_FRAG);
    lineProg = program(LINE_VERT, LINE_FRAG);
    skyProg = program(SKY_VERT, SKY_FRAG);
} catch (e) { fail(e); }
console.log('shaders compiled');

const splatU = uniforms(splatProg,
    ['view', 'eye', 'cellXY', 'relief', 'wave', 'subdiv',
        'focal', 'viewport', 'gain', 'near',
        'data', 'colour', 'tileSize', 'edgeMark', 'fade',
        'tint', 'tintAmount', 'fogColour', 'fogDensity',
        'edgeN', 'edgeE', 'edgeS', 'edgeW']);
const lineU = uniforms(lineProg,
    ['view', 'eye', 'focal', 'viewport', 'near', 'alpha']);

// One VAO per program. Without them, attribute enables and divisors leak
// between draws: the splat pass sets divisor 1 on attribute 1, and the
// line pass then reads its colours from a single vertex, or nothing at all.
const splatVAO = gl.createVertexArray();
const lineVAO = gl.createVertexArray();
const skyVAO = gl.createVertexArray();

const quadBuf = gl.createBuffer();
const indexBuf = gl.createBuffer();
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
    ['view', 'focal', 'viewport', 'skyTop', 'skyHorizon', 'ground']);
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
let tileSize = 0;
let gridN = 1;
let cells = [];
let seed = 1;
let usedPatches = 0;
let sortedReady = false, sortPending = false, lastSortKey = '', sortMs = 0;
let sortingEnabled = true, gain = 1;
let showEdges = false, showDiagonals = false, showTints = false;
let relief = 0, reliefScale = 6, edgeBand = 0.12, subdiv = 1;
const FLAT = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

/** Ground height at a point. Two octaves is enough to bend tiles without
 *  turning the terrain into noise. */
function height(x, y) {
    if (relief <= 0) return 0;
    // Wavelength is measured in tiles, so the terrain keeps the same shape
    // relative to the tiling whatever the tile size happens to be. In world
    // units a fixed number puts a 3x3 grid inside a single hill, which reads
    // as a plane tilt rather than terrain.
    const f = 1 / Math.max(reliefScale * (tileSize || 1), 0.01);
    return relief * (Math.sin(f * x) * Math.cos(f * y)
        + 0.5 * Math.sin(2.3 * f * x + 1.7) * Math.cos(1.9 * f * y + 0.4));
}

/** The tangent frame at a point: two surface tangents and the normal.
 *  Returned column-major, which is what uniformMatrix3fv expects.
 *  Not `frame` - that name is the render loop. */
function tangentFrame(x, y) {
    if (relief <= 0) return FLAT;
    const e = Math.max(reliefScale * (tileSize || 1), 0.01) * 0.01;
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
let lodLevels = 1;         // levels available per tile
let lodBase = 8;           // distance in tiles at which level 1 takes over
let lodOn = true;
let lodCounts = [];        // cells drawn at each level, for the readout
let showLodColours = false;

// Three skies rather than one slider: choosing a mood is easier than
// choosing six numbers, and the fog has to match the horizon or the
// terrain ends against a colour the sky never has.
const SKIES = {
    none: { top: [0, 0, 0], horizon: [0, 0, 0], ground: [0, 0, 0], fog: 0 },
    overcast: {
        top: [0.52, 0.57, 0.62], horizon: [0.78, 0.80, 0.82],
        ground: [0.10, 0.10, 0.11], fog: 0.020
    },
    dusk: {
        top: [0.10, 0.13, 0.24], horizon: [0.72, 0.47, 0.35],
        ground: [0.05, 0.05, 0.07], fog: 0.030
    },
    clear: {
        top: [0.22, 0.45, 0.78], horizon: [0.70, 0.80, 0.90],
        ground: [0.08, 0.09, 0.10], fog: 0.012
    },
};
let sky = 'overcast';
let fogScale = 1.0;
let orbiting = false;

// Green through red as detail drops, the convention every engine's LOD
// debug view uses.
const LOD_RGB = [[0.30, 0.85, 0.35], [0.95, 0.85, 0.25],
[0.98, 0.58, 0.20], [0.92, 0.30, 0.30],
[0.75, 0.35, 0.85], [0.40, 0.60, 0.95]];

/** Which level a cell should use, and how far through the cross-fade it is.
 *
 *  Level i takes over at lodBase * 2^i tiles away, doubling each time, and
 *  the two neighbouring levels are blended across the last 20% of that
 *  range so the switch does not pop. GSWT does the same (Section 3.5) with
 *  a band of about 5%; a wider one is more forgiving when the levels differ
 *  as much as ours do. */
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

console.log('starting sort worker');
const worker = new Worker('./sort-worker.js');
worker.onmessage = (e) => {
    if (e.data.type !== 'sorted') return;
    if (!sortingEnabled) { sortPending = false; return; }
    gl.bindVertexArray(splatVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Uint32Array(e.data.order), gl.DYNAMIC_DRAW);
    gl.bindVertexArray(null);
    sortMs = e.data.ms;
    sortedReady = true;
    sortPending = false;
};

// ================================================================ layout

/** Lay out gridN x gridN cells.
 *
 *  With a Wang tile set, each cell's west colour is fixed by the cell to
 *  its left and its south colour by the cell below; north and east stay
 *  free. A complete set always has a tile that fits, so this never
 *  backtracks, and the free choices are what stop the terrain repeating. */
function buildGrid() {
    cells = [];
    if (!tileSize) { cells = [{ x: 0, y: 0, z: 0, warp: FLAT, patch: 0 }]; return; }
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
                pick = fits.length ? fits[Math.floor(rand() * fits.length) % fits.length] : 0;
            } else {
                pick = Math.floor(rand() * usedPatches) % usedPatches;
            }
            chosen[j * gridN + i] = pick;
            const x = (i - half) * tileSize, y = (j - half) * tileSize;
            cells.push({ x, y, z: height(x, y), warp: tangentFrame(x, y), patch: pick });
        }
    }
}

// Two colours per axis, as in the GSWT figures: warm for north/south, cool
// for east/west, so a glance tells you which constraint you are looking at.
const EDGE_RGB = {
    h: [[0.92, 0.30, 0.30], [0.35, 0.85, 0.40], [0.98, 0.62, 0.18], [0.88, 0.40, 0.85]],
    v: [[0.35, 0.65, 0.95], [0.95, 0.82, 0.30], [0.40, 0.90, 0.88], [0.72, 0.55, 0.98]],
};
const DIAGONAL_RGB = [0.75, 0.75, 0.75];

/** Overlay geometry for the current grid.
 *
 *  Each cell contributes its four boundary segments, coloured by that
 *  edge's colour, and optionally its two diagonals, where the four source
 *  patches meet inside the tile. Shared boundaries are drawn twice, once
 *  by each neighbour, inset slightly so both show. If the two disagree the
 *  line reads as two colours, which is the failure this view exists for. */
function buildOverlay() {
    const pos = [], rgb = [];
    if (tileSize && cells.length && (showEdges || showDiagonals)) {
        const h = tileSize / 2;
        // Lift above the tallest thing in the tile, not by a fixed fraction:
        // sitting the lines inside the geometry hides them.
        const lift = tileSize * 0.2;
        // Corners are placed through the cell's own tangent frame, so the lines
        // sit on the tile rather than on an imaginary flat plane above it.
        let W = FLAT, O = [0, 0, 0];
        const seg = (u0, v0, u1, v1, c) => {
            for (const [u, v] of [[u0, v0], [u1, v1]]) {
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
    console.log(`overlay: ${lineCount / 2} segments`);
}

/** Where the shader puts a splat. Mirrors the vertex shader exactly, so
 *  the measurement below describes what is actually on screen. */
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

/** How far apart two neighbouring tiles put the same point on their shared
 *  edge. This is the artifact, measured rather than eyeballed: sample along
 *  every interior boundary, place each sample from both sides, and take the
 *  distance. Reported in splat widths, because a gap much smaller than a
 *  splat cannot be seen. */
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

function reportSeams() {
    const m = measureSeams();
    if (!m || !ui.seam) return;
    const unit = splatWidth > 0 ? splatWidth : 1;
    ui.seam.textContent =
        `max ${(m.worst / unit).toFixed(2)} splat widths (${m.worst.toFixed(4)}), ` +
        `mean ${(m.mean / unit).toFixed(2)}, over ${m.count} samples`;
}

function regenerate() { buildGrid(); buildOverlay(); reportSeams(); }

// ================================================================ loading

function load(buffer, manifest) {
    const { n, positions, data, colour } = unpack(buffer);
    splatCount = n;

    if (manifest && manifest.tiles && manifest.tiles.length) {
        patches = manifest.tiles.map(t => ({
            start: t.start, count: t.count,
            levels: t.levels || [[t.start, t.count]],
        }));
        lodLevels = Math.max(1, manifest.lod || 1);
        tileSize = manifest.size || 0;
        wangCodes = manifest.wang ? manifest.tiles.map(t => [t.n, t.e, t.s, t.w]) : null;
    } else {
        patches = [{ start: 0, count: n, levels: [[0, n]] }];
        lodLevels = 1;
        tileSize = 0;
        wangCodes = null;
    }
    usedPatches = patches.length;

    const w = 2048;
    const h = Math.ceil(n * 3 / w);
    const padded = new Float32Array(w * h * 4);
    padded.set(data.subarray(0, Math.min(data.length, padded.length)));
    makeTexture(0, gl.RGBA32F, w, h, gl.RGBA, gl.FLOAT, padded);
    const ch = Math.ceil(n / w);
    const cpad = new Uint8Array(w * ch * 4);
    cpad.set(colour);
    makeTexture(1, gl.RGBA8, w, ch, gl.RGBA, gl.UNSIGNED_BYTE, cpad);

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

    const pos = positions.slice();
    // Every level is sorted separately: it holds a different set of splats,
    // so it needs its own order.
    const ranges = [];
    for (const p of patches) {
        for (const [start, count] of p.levels) ranges.push({ start, count });
    }
    worker.postMessage({ type: 'init', positions: pos.buffer, patches: ranges },
        [pos.buffer]);

    ui.n.textContent = n.toLocaleString() +
        (patches.length > 1 ? ` in ${patches.length}` : '');
    ui.grid.disabled = !tileSize;
    ui.used.disabled = !tileSize || patches.length < 2 || !!wangCodes;
    ui.used.max = patches.length;
    ui.used.value = patches.length;
    ui.usedn.textContent = `${patches.length} of ${patches.length}`;
    ui.wangnote.textContent = wangCodes
        ? `wang, ${patches.length} tiles, ${manifest.colours || 2} colours/axis`
        : (tileSize ? 'random placement, edges do not match' : 'single scene');

    regenerate();
    overlay.classList.add('hidden');
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

function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.floor(canvas.clientWidth * dpr);
    const h = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w; canvas.height = h;
        gl.viewport(0, 0, w, h);
    }
}

function frame() {
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
            worker.postMessage({ type: 'sort', forward: b.forward, eye: b.eye });
        }
    }

    const S = SKIES[sky] || SKIES.none;
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
        gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    drawnSplats = 0; drawCalls = 0;

    if (splatCount && sortedReady) {
        gl.useProgram(splatProg);
        gl.bindVertexArray(splatVAO);
        gl.uniform1i(splatU.data, 0);
        gl.uniform1i(splatU.colour, 1);
        gl.uniformMatrix3fv(splatU.view, false, viewMat);
        gl.uniform3fv(splatU.eye, new Float32Array(b.eye));
        gl.uniform2f(splatU.focal, fy, fy);
        gl.uniform2f(splatU.viewport, canvas.width, canvas.height);
        gl.uniform1f(splatU.gain, gain);
        gl.uniform1f(splatU.near, 0.05);
        gl.uniform1f(splatU.relief, relief);
        gl.uniform1f(splatU.wave, Math.max(reliefScale * (tileSize || 1), 0.01));
        gl.uniform1i(splatU.subdiv, subdiv);
        gl.uniform1f(splatU.tileSize, tileSize);
        gl.uniform3fv(splatU.fogColour, new Float32Array(S.horizon));
        gl.uniform1f(splatU.fogDensity, S.fog * fogScale);

        // Cells far to near, composited with 'over'. Splats are sorted within
        // a patch and never across cells, which is the approximation that
        // produces the boundary artifact.
        const visible = [];
        for (const c of cells) {
            const dx = c.x - b.eye[0], dy = c.y - b.eye[1], dz = c.z - b.eye[2];
            const z = dx * b.forward[0] + dy * b.forward[1] + dz * b.forward[2];
            if (z < -tileSize) continue;
            const dist = Math.hypot(dx, dy, c.z - b.eye[2]);
            const sx = dx * b.right[0] + dy * b.right[1] + dz * b.right[2];
            const sy = dx * b.down[0] + dy * b.down[1] + dz * b.down[2];
            const reach = tileSize * 1.5 + Math.max(z, 0.01) *
                Math.tan(cam.fov * Math.PI / 360) * (canvas.width / canvas.height);
            if (Math.abs(sx) > reach || Math.abs(sy) > reach) continue;
            visible.push({ c, z, dist });
        }
        visible.sort((p, q) => q.z - p.z);

        lodCounts = new Array(lodLevels).fill(0);
        for (const { c, dist } of visible) {
            const p = patches[c.patch];
            if (!p || !p.count) continue;
            gl.uniform2f(splatU.cellXY, c.x, c.y);
            if (showTints && wangCodes) {
                const code = wangCodes[c.patch];
                const set = (loc, rgb) => gl.uniform3f(loc, rgb[0], rgb[1], rgb[2]);
                set(splatU.edgeN, EDGE_RGB.h[code[0] % 4]);
                set(splatU.edgeE, EDGE_RGB.v[code[1] % 4]);
                set(splatU.edgeS, EDGE_RGB.h[code[2] % 4]);
                set(splatU.edgeW, EDGE_RGB.v[code[3] % 4]);
                gl.uniform1f(splatU.edgeMark, edgeBand);
            } else {
                gl.uniform1f(splatU.edgeMark, 0.0);
            }
            const [lvl, wA, wB] = lodFor(dist);
            lodCounts[lvl]++;
            gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
            for (const [li, w] of [[lvl, wA], [lvl + 1, wB]]) {
                if (w <= 0.001 || li >= p.levels.length) continue;
                const [start, count] = p.levels[li];
                if (!count) continue;
                gl.uniform1f(splatU.fade, w);
                if (showLodColours) {
                    const c2 = LOD_RGB[li % LOD_RGB.length];
                    gl.uniform3f(splatU.tint, c2[0], c2[1], c2[2]);
                    gl.uniform1f(splatU.tintAmount, 0.6);
                } else {
                    gl.uniform1f(splatU.tintAmount, 0.0);
                }
                gl.vertexAttribIPointer(aIndex, 1, gl.UNSIGNED_INT, 0, start * 4);
                gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
                drawnSplats += count; drawCalls++;
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

    drawGizmo(b);

    frames++;
    const now = performance.now();
    if (now - fpsTime > 500) {
        ui.fps.textContent = (frames * 1000 / (now - fpsTime)).toFixed(0);
        ui.drawn.textContent = drawnSplats.toLocaleString() +
            (drawCalls > 1 ? ` / ${drawCalls} cells` : '');
        ui.sortms.textContent = sortMs ? sortMs.toFixed(0) + ' ms' : '—';
        if (ui.lodinfo) {
            ui.lodinfo.textContent = lodLevels < 2 ? 'not in this file'
                : (lodOn ? lodCounts.map((n, i) => `L${i}:${n}`).join('  ')
                    : 'off, all cells at level 0');
        }
        frames = 0; fpsTime = now;
    }
    ui.azim.textContent = cam.azimuth.toFixed(0) + '\u00b0';
    ui.elev.textContent = cam.elevation.toFixed(0) + '\u00b0';
    ui.dist.textContent = cam.distance.toFixed(2);
    ui.cmd.textContent = `--azim ${cam.azimuth.toFixed(0)} ` +
        `--elev ${cam.elevation.toFixed(0)} --dist ${cam.distance.toFixed(2)} ` +
        `--fov ${cam.fov.toFixed(0)}`;

    if (orbiting) cam.azimuth = (cam.azimuth + 0.08) % 360;

    requestAnimationFrame(frame);
}

// ============================================================ interaction

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
    const fl = Math.hypot(b.forward[0], b.forward[1]) || 1;
    const fwd = [b.forward[0] / fl, b.forward[1] / fl, 0];
    const rl = Math.hypot(b.right[0], b.right[1]) || 1;
    const rgt = [b.right[0] / rl, b.right[1] / rl, 0];
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
    const buf = await f.arrayBuffer();
    bar.style.width = '70%';
    const m = await fetch(`./data/${f.name.replace(/\.splat$/, '.json')}`)
        .then(r => (r.ok ? r.json() : null)).catch(() => null);
    load(buf, m);
    bar.style.width = '100%';
});

document.getElementById('fov').addEventListener('input',
    (e) => { cam.fov = +e.target.value; });
document.getElementById('gain').addEventListener('input',
    (e) => { gain = +e.target.value; });
document.getElementById('sorting').addEventListener('change', (e) => {
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
ui.grid.addEventListener('input', (e) => {
    gridN = +e.target.value;
    ui.gridn.textContent = `${gridN} x ${gridN}`;
    regenerate();
});
ui.used.addEventListener('input', (e) => {
    usedPatches = +e.target.value;
    ui.usedn.textContent = `${usedPatches} of ${patches.length}`;
    regenerate();
});
ui.tints.addEventListener('change', (e) => { showTints = e.target.checked; });
ui.lod.addEventListener('change', (e) => { lodOn = e.target.checked; });
ui.sky.addEventListener('change', (e) => { sky = e.target.value; });
ui.fog.addEventListener('input', (e) => {
    fogScale = +e.target.value;
    ui.fogn.textContent = fogScale.toFixed(2);
});
ui.orbit.addEventListener('change', (e) => { orbiting = e.target.checked; });
for (const [id, elev, dist] of [['viewGround', 3, 8], ['viewWalk', 12, 6],
['viewSurvey', 32, 14], ['viewTop', 85, 18]]) {
    document.getElementById(id).addEventListener('click', () => {
        cam.elevation = elev;
        cam.distance = (tileSize || 1) * dist;
        cam.target = [0, 0, cam.target[2]];
    });
}
ui.lodcolours.addEventListener('change', (e) => {
    showLodColours = e.target.checked;
});
ui.lodbase.addEventListener('input', (e) => {
    lodBase = +e.target.value;
    ui.lodbasen.textContent = `${lodBase} tiles`;
});
ui.subdiv.addEventListener('input', (e) => {
    const v = +e.target.value;
    setTimeout(reportSeams, 0);
    // The top of the slider is the limit of subdividing: one frame per splat.
    subdiv = v > 16 ? 0 : v;
    ui.subdivn.textContent =
        v > 16 ? 'smooth' : (v === 1 ? '1 (GSWT)' : `${v} x ${v}`);
});
ui.band.addEventListener('input', (e) => {
    edgeBand = +e.target.value;
    ui.bandn.textContent = edgeBand.toFixed(2);
});
ui.relief.addEventListener('input', (e) => {
    relief = +e.target.value;
    ui.reliefn.textContent = relief.toFixed(2);
    regenerate();
});
ui.reliefscale.addEventListener('input', (e) => {
    reliefScale = +e.target.value;
    ui.reliefscalen.textContent = reliefScale.toFixed(0);
    regenerate();
});
ui.edges.addEventListener('change', (e) => {
    showEdges = e.target.checked;
    buildOverlay();
});
ui.diagonals.addEventListener('change', (e) => {
    showDiagonals = e.target.checked;
    buildOverlay();
});
document.getElementById('reseed').addEventListener('click', () => {
    seed = (Math.random() * 1e9) | 0;
    regenerate();
});
document.getElementById('reset').addEventListener('click', () => {
    cam.azimuth = 45; cam.elevation = 25;
});

const wanted = new URLSearchParams(location.search).get('scene');
for (const name of (wanted ? [wanted] : ['scene', 'bigsur', 'garden'])) {
    Promise.all([
        fetch(`./data/${name}.splat`).then(r => (r.ok ? r.arrayBuffer() : null)),
        fetch(`./data/${name}.json`).then(r => (r.ok ? r.json() : null)).catch(() => null),
    ]).then(([b, m]) => { if (b && !splatCount) load(b, m); }).catch(() => { });
}

frame();