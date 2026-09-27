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
import { classify, sampleGrid } from './landform.js';
import { GROUPS as TERRAIN_GROUPS, PROFILES as TERRAIN_PROFILES } from './terrain.js';
import { starterTileset } from './starter.js';

const BUILD = 'bozkir viewer 5.0 (far field, auto quality, detail relief, starter)';
console.log('%c' + BUILD, 'color:#c8a05a');

const STRIDE = 32;        // bytes per splat in the .splat format
const DILATION = 0.3;     // matches DILATION in bozkir/render.py

// ============================================================== shaders

// The ground as both vertex shaders see it: height field, detail relief and
// the analytic fallback. Shared so splats and far field always agree.
// Needs uField, uFieldSize, uFieldExtent, uHasField, uOpen, uHasOpen,
// uRelief and uWave declared first.
const TERRAIN_GLSL = `
// Mirror into 0..m like HeightField does, so a small field repeats
// without a cliff at the join.
float mirrorCoord(float t, float m) {
  if (m <= 0.0) return 0.0;
  float p = 2.0 * m;
  float v = mod(t, p);
  if (v < 0.0) v += p;
  return v <= m ? v : p - v;
}

float fieldAt(vec2 world) {
  float longEdge = max(uFieldSize.x, uFieldSize.y) - 1.0;
  float scale = longEdge / max(uFieldExtent, 1e-6);
  float u = world.x * scale + (uFieldSize.x - 1.0) * 0.5;
  float v = (uFieldSize.y - 1.0) * 0.5 - world.y * scale;
  u = mirrorCoord(u, uFieldSize.x - 1.0);
  v = mirrorCoord(v, uFieldSize.y - 1.0);

  ivec2 p0 = ivec2(floor(u), floor(v));
  ivec2 p1 = min(p0 + 1, ivec2(uFieldSize) - 1);
  vec2 f = vec2(u, v) - vec2(p0);
  float a = texelFetch(uField, ivec2(p0.x, p0.y), 0).r;
  float b = texelFetch(uField, ivec2(p1.x, p0.y), 0).r;
  float c = texelFetch(uField, ivec2(p0.x, p1.y), 0).r;
  float d = texelFetch(uField, ivec2(p1.x, p1.y), 0).r;
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Detail relief: tile-scale creases so close ground is not a smooth curve
// (a big world has ~2 height samples per tile and erosion rounds crests).
// Ridged value noise on an integer hash, bit-identical to detailAt() in
// the JS, so camera, cells, splats and far field agree.
uniform float uDetail;       // amplitude in world units; 0 is off
uniform float uDetailFreq;   // lattice cells per world unit
float dHash(ivec2 c) {
  uint h = uint(c.x) * 0x27D4EB2Du ^ uint(c.y) * 0x165667B1u;
  h = (h ^ (h >> 15u)) * 0x85EBCA6Bu;
  h ^= h >> 13u;
  return float(h & 0xFFFFFFu) / 16777215.0;
}
float dValue(vec2 q) {
  vec2 i = floor(q), f = q - i;
  vec2 s = f * f * (3.0 - 2.0 * f);
  ivec2 c = ivec2(i);
  float a = dHash(c), b = dHash(c + ivec2(1, 0));
  float d = dHash(c + ivec2(0, 1)), e = dHash(c + ivec2(1, 1));
  return mix(mix(a, b, s.x), mix(d, e, s.x), s.y);
}
float maskAt(vec2 world) {
  if (uHasOpen < 0.5) return 1.0;
  float longEdge = max(uFieldSize.x, uFieldSize.y) - 1.0;
  float scale = longEdge / max(uFieldExtent, 1e-6);
  float u = mirrorCoord(world.x * scale + (uFieldSize.x - 1.0) * 0.5, uFieldSize.x - 1.0);
  float v = mirrorCoord((uFieldSize.y - 1.0) * 0.5 - world.y * scale, uFieldSize.y - 1.0);
  ivec2 p0 = ivec2(floor(u), floor(v));
  ivec2 p1 = min(p0 + 1, ivec2(uFieldSize) - 1);
  vec2 f = vec2(u, v) - vec2(p0);
  float a = texelFetch(uOpen, p0, 0).g, b = texelFetch(uOpen, ivec2(p1.x, p0.y), 0).g;
  float c = texelFetch(uOpen, ivec2(p0.x, p1.y), 0).g, d = texelFetch(uOpen, p1, 0).g;
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float detailAt(vec2 p) {
  if (uDetail <= 0.0) return 0.0;
  vec2 q = p * uDetailFreq;
  float r1 = 1.0 - abs(2.0 * dValue(q) - 1.0);
  float r2 = 1.0 - abs(2.0 * dValue(q * 2.03 + vec2(17.3, 5.1)) - 1.0);
  return uDetail * maskAt(p) * ((r1 * r1 + 0.5 * r2 * r2) / 1.5 - 0.5);
}
float terrainHeight(vec2 p) {
  if (uRelief <= 0.0) return 0.0;
  if (uHasField > 0.5) {
    // Centred on zero: relief raises and lowers about the middle.
    return uRelief * (fieldAt(p) - 0.5) * 2.0 + detailAt(p);
  }
  float f = 1.0 / max(uWave, 0.01);
  return uRelief * (sin(f * p.x) * cos(f * p.y)
    + 0.5 * sin(2.3 * f * p.x + 1.7) * cos(1.9 * f * p.y + 0.4)) + detailAt(p);
}
`;

const SPLAT_VERT = `#version 300 es
precision highp float;
precision highp int;

const float DILATION = ${DILATION.toFixed(4)};

uniform sampler2D uData;      // RGBA32F, 3 texels per splat
uniform sampler2D uColour;    // RGBA8,   1 texel per splat
uniform mat3 uView;           // world -> camera, rows are the camera axes
uniform vec3 uEye;
// Tile centres, one per slot of a merged group (unmerged draws use slot 0).
uniform vec2 uCellXY[8];

// Per-splat class blending. Hybrid GSWT picks the class per Gaussian, which
// is why their boundaries look organic; a per-tile choice is a staircase.
// Near a boundary both classes are drawn and each splat survives only if a
// fixed hash of its index falls on its class's side of the mix - stable
// frame to frame, and repeated tiles at different mixes keep different
// splats. uMix < 0 turns it off.
uniform float uMix;
uniform float uRelief;        // height field amplitude
uniform float uWave;          // height field wavelength, world units
uniform vec2 uGridRot;        // cos, sin of the grid's rotation
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
// Splat index from the sorted order buffer; the top 3 bits are the cell's
// slot in a merged group.
in uint aIndex;

out vec2 vCorner;
out vec4 vColour;
out float vFar;                // 0 before the far field, 1 past its start
uniform float uFarOn;
uniform float uFarStart;
uniform float uFarBand;

// The height field. Must match height() in the JS exactly: the rule and the
// overlay use the JS copy, the geometry this one. (It once had only the
// sine branch, so classes came from real drainage and were drawn on sine
// waves.) texelFetch + manual bilinear because R32F is not filterable
// everywhere, and so it matches heightfield.js exactly.
uniform sampler2D uField;
uniform vec2 uFieldSize;      // texels
uniform float uFieldExtent;   // world units across the long edge
uniform float uHasField;
uniform sampler2D uOpen;      // sky openness, built from the height field
uniform float uHasOpen;
uniform float uShade;         // how much of it to apply
uniform float uExposure;
uniform float uSaturation;

${TERRAIN_GLSL}

// Sky openness, 0 in a hollow and 1 in the open. From a texture: it needs
// the ground around the point and does not change with the camera.
float openAt(vec2 world) {
  float longEdge = max(uFieldSize.x, uFieldSize.y) - 1.0;
  float scale = longEdge / max(uFieldExtent, 1e-6);
  float u = world.x * scale + (uFieldSize.x - 1.0) * 0.5;
  float v = (uFieldSize.y - 1.0) * 0.5 - world.y * scale;
  u = mirrorCoord(u, uFieldSize.x - 1.0);
  v = mirrorCoord(v, uFieldSize.y - 1.0);
  ivec2 p0 = ivec2(floor(u), floor(v));
  ivec2 p1 = min(p0 + 1, ivec2(uFieldSize) - 1);
  vec2 f = vec2(u, v) - vec2(p0);
  float a = texelFetch(uOpen, ivec2(p0.x, p0.y), 0).r;
  float b = texelFetch(uOpen, ivec2(p1.x, p0.y), 0).r;
  float c = texelFetch(uOpen, ivec2(p0.x, p1.y), 0).r;
  float d = texelFetch(uOpen, ivec2(p1.x, p1.y), 0).r;
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Surface gradient by central differences.
vec2 terrainGrad(vec2 p) {
  if (uRelief <= 0.0) return vec2(0.0);
  // Step of one texel on a field, a fraction of the wavelength otherwise.
  float e = uHasField > 0.5
    ? max(uFieldExtent, 1e-6) / max(max(uFieldSize.x, uFieldSize.y), 2.0)
    : max(uWave, 0.01) * 0.01;
  // With detail on, the step must be finer than the detail's own creases.
  if (uDetail > 0.0) e = min(e, 0.15 / max(uDetailFreq, 1e-6));
  return vec2(
    (terrainHeight(p + vec2(e, 0.0)) - terrainHeight(p - vec2(e, 0.0))),
    (terrainHeight(p + vec2(0.0, e)) - terrainHeight(p - vec2(0.0, e)))
  ) / (2.0 * e);
}

// Two frames. Positions use the orthonormal one (splats keep their ground
// spacing); covariance uses the true Jacobian, whose tangents are longer on
// a slope. Normalising both thinned material on steep ground by
// sqrt(1 + |grad h|^2).
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
  uint slot = aIndex >> 29u;
  uint sid = aIndex & 0x1FFFFFFFu;
  vec2 cellXY = uCellXY[int(slot)];

  if (uMix >= 0.0) {
    // Fixed hash of the index, flat enough that the threshold means what it says.
    uint h = sid * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    float r = float((h >> 22u) ^ h) / 4294967296.0;
    if (r > uMix) {
      // Behind the camera: cheaper than a discard, keeps the fragment
      // shader branch-free.
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      return;
    }
  }

  vec4 a = fetch(sid, 0);     // position.xyz, opacity
  vec4 b = fetch(sid, 1);     // scale.xyz
  vec4 q = fetch(sid, 2);     // rotation w,x,y,z

  // GSWT Eq. 4-5 warps a whole tile by the frame at its centre, which is
  // wrong towards the edges, so neighbours disagree along their border.
  // uSubdiv gives each sub-cell its own frame (1 = GSWT); 0 anchors every
  // splat at its own position, so the frame is continuous and no seam is left.
  // Turn the tile with the grid first.
  vec3 local = vec3(uGridRot.x * a.x - uGridRot.y * a.y,
                    uGridRot.y * a.x + uGridRot.x * a.y,
                    a.z);
  mat3 spin = mat3(uGridRot.x, uGridRot.y, 0.0,
                   -uGridRot.y, uGridRot.x, 0.0,
                   0.0, 0.0, 1.0);

  vec2 anchorLocal;
  if (uSubdiv <= 0) {
    anchorLocal = local.xy;
  } else {
    float sub = uTileSize / float(uSubdiv);
    vec2 idx = clamp(floor((local.xy + uTileSize * 0.5) / sub),
                     0.0, float(uSubdiv) - 1.0);
    anchorLocal = (idx + 0.5) * sub - uTileSize * 0.5;
  }
  vec2 anchorWorld = cellXY + anchorLocal;

  vec2 grad = terrainGrad(anchorWorld);
  mat3 warp = terrainFrame(grad);
  vec3 origin = vec3(anchorWorld, terrainHeight(anchorWorld));
  vec3 world = origin + warp * (local - vec3(anchorLocal, 0.0));

  vec3 cam = uView * (world - uEye);
  vFar = uFarOn > 0.5 ? smoothstep(uFarStart - uFarBand, uFarStart, cam.z) : 0.0;
  if (cam.z < uNear) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }

  float w = q.x, x = q.y, y = q.z, z = q.w;
  mat3 R = mat3(
    1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z),       2.0*(x*z - w*y),
    2.0*(x*y - w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x),
    2.0*(x*z + w*y),       2.0*(y*z - w*x),       1.0 - 2.0*(x*x + y*y)
  );
  // Sigma' = W Sigma W^T = (W R S)(W R S)^T.
  mat3 M = terrainJacobian(grad) * spin * R
         * mat3(b.x, 0.0, 0.0, 0.0, b.y, 0.0, 0.0, 0.0, b.z);
  mat3 sigma = M * transpose(M);

  // The Taylor expansion only holds near the axis: clamp to a slightly
  // enlarged frustum first.
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
  // Axis sqrt(2*lambda) with alpha = exp(-|corner|^2): at |corner| = 1 both
  // are at exp(-1).
  vec2 major = min(sqrt(2.0 * l1), 1024.0) * dir * uGain;
  vec2 minor = min(sqrt(2.0 * l2), 1024.0) * vec2(dir.y, -dir.x) * uGain;

  vec2 centre = vec2(uFocal.x * cam.x * invz, uFocal.y * cam.y * invz);
  vec2 px = centre + aCorner.x * major + aCorner.y * minor;

  gl_Position = vec4(2.0 * px.x / uViewport.x,
                     -2.0 * px.y / uViewport.y, 0.0, 1.0);
  vCorner = aCorner;
  int ci = int(sid);
  vec4 col = texelFetch(uColour, ivec2(ci & 2047, ci >> 11), 0);
  vec3 rgb = col.rgb;

  if (uEdgeMark > 0.0 && uTileSize > 0.0) {
    // Edge bands in their code colour, interior washed out, as in the GSWT
    // figures: a correct join is one colour across the shared border.
    vec2 uv = local.xy / uTileSize;         // tile-local, -0.5 .. 0.5
    float d;
    vec3 ec;
    if (abs(uv.y) >= abs(uv.x)) {
      d = 0.5 - abs(uv.y);
      ec = uv.y > 0.0 ? uEdgeN : uEdgeS;
    } else {
      d = 0.5 - abs(uv.x);
      ec = uv.x > 0.0 ? uEdgeE : uEdgeW;
    }
    // Splats past the tile's square (d < 0) stay uncoloured, or the band
    // looks too wide and neighbours paint the same strip twice.
    float w = d < 0.0 ? 0.0 : 1.0 - smoothstep(0.0, uEdgeMark, d);
    float grey = dot(rgb, vec3(0.299, 0.587, 0.114));
    rgb = mix(vec3(0.55 + 0.45 * grey), ec, w);
  }
  // Ground shading: only darkens where the terrain would (a hollow sees
  // less sky). Nothing directional - the capture's light is baked in.
  if (uHasOpen > 0.5 && uShade > 0.0) {
    rgb *= mix(1.0, 0.45 + 0.55 * openAt(anchorWorld), uShade);
  }
  // .splat keeps only the constant colour term, so it reads flat and a bit
  // washed; exposure and saturation compensate by eye.
  rgb = clamp(rgb * uExposure, 0.0, 1.0);
  float lum = dot(rgb, vec3(0.299, 0.587, 0.114));
  rgb = clamp(mix(vec3(lum), rgb, uSaturation), 0.0, 1.0);

  // Haze towards the horizon colour so far ground has no hard edge.
  float fog = 1.0 - exp(-uFogDensity * max(cam.z, 0.0));
  rgb = mix(rgb, uFogColour, fog);
  vColour = vec4(mix(rgb, uTint, uTintAmount), col.a * uFade);
}
`;

const SPLAT_FRAG = `#version 300 es
precision highp float;
in vec2 vCorner;
in vec4 vColour;
in float vFar;
out vec4 oColour;
// Screen-door threshold per pixel, shared with the far field: in the
// hand-over band each pixel shows one or the other, never a mix.
float doorway(vec2 p) {
  return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}
void main() {
  if (vFar > 0.0 && vFar >= doorway(gl_FragCoord.xy)) discard;
  float p = -dot(vCorner, vCorner);
  if (p < -4.0) discard;
  float alpha = exp(p) * vColour.a;
  if (alpha < 0.004) discard;
  oColour = vec4(vColour.rgb * alpha, alpha);   // premultiplied
}
`;

// The far field: distant ground as one mesh instead of thousands of cells.
// Past a few dozen tiles a splat is under a pixel, so beyond uFarStart the
// ground is a height-field mesh (same terrainHeight) coloured per cell from
// an atlas of the tiles seen from above (scripts/bake_atlas.py), with the
// same layout and rule. Standard game terrain practice; hierarchical splat
// LOD (Kerbl et al. 2024) solves splat count, and the limit here is draw
// calls. Writes depth and uses a real perspective w so the atlas does not swim.
const FAR_VERT = `#version 300 es
precision highp float;
uniform mat3 uView;
uniform vec3 uEye;
uniform vec2 uFocal;
uniform vec2 uViewport;
uniform float uRelief;
uniform float uWave;
uniform float uHasField;
uniform sampler2D uField;
uniform vec2 uFieldSize;
uniform float uFieldExtent;
in vec2 aXY;
out vec2 vXY;
out float vDepth;
out float vOpen;
uniform sampler2D uOpen;
uniform float uHasOpen;
${TERRAIN_GLSL}

void main() {
  vec3 world = vec3(aXY, terrainHeight(aXY));
  vec3 cam = uView * (world - uEye);
  vXY = aXY;
  vDepth = cam.z;
  // Openness read the same way as the splats, so shading matches.
  vOpen = 1.0;
  if (uHasOpen > 0.5) {
    float longEdge = max(uFieldSize.x, uFieldSize.y) - 1.0;
    float scale = longEdge / max(uFieldExtent, 1e-6);
    float u = mirrorCoord(aXY.x * scale + (uFieldSize.x - 1.0) * 0.5, uFieldSize.x - 1.0);
    float v = mirrorCoord((uFieldSize.y - 1.0) * 0.5 - aXY.y * scale, uFieldSize.y - 1.0);
    vOpen = texelFetch(uOpen, ivec2(floor(u + 0.5), floor(v + 0.5)), 0).r;
  }
  const float n = 0.05, f = 50000.0;
  gl_Position = vec4(2.0 * uFocal.x * cam.x / uViewport.x,
                     -2.0 * uFocal.y * cam.y / uViewport.y,
                     (f + n) / (f - n) * cam.z - 2.0 * f * n / (f - n),
                     cam.z);
}
`;

const FAR_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uAtlas;
uniform sampler2D uCells;       // per cell: tile index in r + 256 g
uniform vec2 uAtlasGrid;        // columns, rows
uniform float uTileSize;
uniform float uGridN;
uniform vec2 uRot;              // cos, sin of the grid angle
uniform float uFarStart;
uniform float uFarBand;
uniform vec3 uFogColour;
uniform float uFogDensity;
uniform float uExposure;
uniform float uShade;
uniform float uSaturation;
uniform int uDebug;            // 0 none, 1 tile identity, 2 class, 3 level
uniform sampler2D uTileInfo;   // row 0 gain, row 1 mean colour, per tile
uniform float uHasTileInfo;
uniform vec3 uClassRGB[4];
in vec2 vXY;
in float vDepth;
in float vOpen;
out vec4 oColour;
// Same screen-door threshold as SPLAT_FRAG.
float doorway(vec2 p) {
  return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}
// Same palette as tileRGB() in the JS, so a tile keeps its colour across
// the hand-over.
vec3 tileRGB(float i) {
  float h = mod(i * 137.508, 360.0) / 360.0;
  vec3 n = vec3(5.0, 3.0, 1.0);
  vec3 v = mod(n + h * 6.0, 6.0);
  vec3 k = clamp(min(min(v, 4.0 - v), vec3(1.0)), 0.0, 1.0);
  return 0.30 + 0.70 * k;
}
void main() {
  float a = smoothstep(uFarStart - uFarBand, uFarStart, vDepth);
  if (a < doorway(gl_FragCoord.xy)) discard;
  a = 1.0;
  vec2 l = vec2(uRot.x * vXY.x + uRot.y * vXY.y,
                -uRot.y * vXY.x + uRot.x * vXY.y);
  vec2 g = l / uTileSize + (uGridN - 1.0) * 0.5 + 0.5;
  vec2 ij = floor(g);
  if (ij.x < 0.0 || ij.y < 0.0 || ij.x >= uGridN || ij.y >= uGridN) discard;
  vec2 fr = clamp(g - ij, 0.01, 0.99);   // east, north fraction in the cell
  vec4 t = texelFetch(uCells, ivec2(ij), 0);
  float idx = floor(t.r * 255.0 + 0.5) + 256.0 * floor(t.g * 255.0 + 0.5);
  float col = mod(idx, uAtlasGrid.x);
  float row = floor(idx / uAtlasGrid.x);
  // Atlas tiles are rendered north up.
  vec2 uv = vec2((col + fr.x) / uAtlasGrid.x,
                 (row + 1.0 - fr.y) / uAtlasGrid.y);
  // Derivatives from the continuous cell coordinate: uv jumps at cell
  // borders, which the GPU reads as extreme minification (a blurry line
  // along every border).
  vec2 gx = dFdx(g), gy = dFdy(g);
  vec2 k = vec2(1.0 / uAtlasGrid.x, -1.0 / uAtlasGrid.y);
  vec4 tex = textureGrad(uAtlas, uv, gx * k, gy * k);
  vec3 rgb = tex.rgb;
  if (uHasTileInfo > 0.5) {
    int ti = int(idx);
    rgb *= texelFetch(uTileInfo, ivec2(ti, 0), 0).rgb * 2.0;
    // Fade to the tile mean with distance, as terrain renderers do; far
    // away a tile's own pattern only reads as a lattice.
    vec3 mean = texelFetch(uTileInfo, ivec2(ti, 1), 0).rgb;
    float far = smoothstep(uFarStart * 1.5, uFarStart * 5.0, vDepth);
    rgb = mix(rgb, mean, 0.75 * far);
  }
  // Opaque on purpose: drawing with atlas coverage washed the sky into all
  // distant ground.
  rgb *= mix(1.0, 0.45 + 0.55 * vOpen, uShade);
  rgb = clamp(rgb * uExposure, 0.0, 1.0);
  float lum = dot(rgb, vec3(0.299, 0.587, 0.114));
  rgb = clamp(mix(vec3(lum), rgb, uSaturation), 0.0, 1.0);
  // Debug views reach the far field too.
  if (uDebug == 1) {
    rgb = mix(rgb, tileRGB(idx), 0.75);
  } else if (uDebug == 2) {
    int cls = int(floor(t.b * 255.0 + 0.5));
    float sure = t.a;
    vec3 c = uClassRGB[min(cls, 3)] * sure + vec3(0.62) * (1.0 - sure);
    rgb = mix(rgb, c, 0.8);
  } else if (uDebug == 3) {
    rgb = mix(rgb, vec3(0.92, 0.92, 0.95), 0.6);   // the far field's own level
  }
  float fog = 1.0 - exp(-uFogDensity * max(vDepth, 0.0));
  rgb = mix(rgb, uFogColour, fog);
  oColour = vec4(rgb, 1.0);
}
`;

// Overlay lines, same projection as the splats.
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

// Gradient sky: ground against a horizon reads with depth, against black it doesn't.
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
  // World direction of this pixel.
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
                  'wangnote', 'edges', 'diagonals', 'tints', 'identity',
                  'relief', 'reliefn', 'reliefscale', 'reliefscalen',
                  'band', 'bandn', 'subdiv', 'subdivn', 'seam',
                  'sortmode', 'views', 'viewsn', 'recache', 'sorterr',
                  'popmeter', 'popreset', 'pop', 'gridangle', 'gridanglen',
                  'lod', 'lodbase', 'lodbasen', 'lodinfo', 'lodcolours',
                  'sky', 'fog', 'fogn', 'orbit', 'freefly', 'flynote',
                  'shade', 'shaden', 'exposure', 'exposuren',
                  'figuremode', 'savepng', 'panel', 'gizmo', 'figexit',
                  'scenepick', 'heightpick', 'expert', 'classview',
                  'farfield', 'busy', 'smoothframes', 'subdivrow', 'autoquality',
                  'gpuname',
                  'detailon', 'detailamt', 'detailamtn',
                  'genseed', 'gensize', 'genbtn', 'genstatus',
                  'terrainopen', 'terrainwin', 'terrainclose', 'gencards',
                  'seeddown', 'seedup', 'seedrand', 'genrecent', 'presetchips',
                  'saturation', 'saturationn',
                  'speedn', 'tileorder', 'merging', 'mergethr', 'mergethrn',
                  'mergestat', 'capture', 'captureboth', 'capframes',
                  'capframesn', 'capcentre', 'caparc', 'caparcn',
                  'capstepn', 'fieldnote', 'bench', 'benchout',
                  'classrule', 'classsharp', 'classsharpn',
                  'classnote', 'classblend', 'blendwidth',
                  'blendwidthn', 'classbalance', 'classbalancen',
                  'classcoherence', 'classcoherencen',
                  'classaltitude', 'classaltituden']) {
  ui[id] = document.getElementById(id);
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
  let fs = 11;
  on('fsup', 'click', () => { fs = setFontSize(fs + 1); });
  on('fsdown', 'click', () => { fs = setFontSize(fs - 1); });

  const grip = document.getElementById('grip');
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
    grip.addEventListener('dblclick', () => setPanelWidth(232));
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
  const el = document.getElementById(id);
  if (!el) {
    console.warn(`no #${id} in the page; that control is inactive`);
    return null;
  }
  el.addEventListener(event, fn);
  return el;
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

// Ask for the fast GPU: laptops often give the browser the integrated chip.
const gl = canvas.getContext('webgl2',
  { antialias: false, alpha: false, premultipliedAlpha: false,
    powerPreference: 'high-performance' });
if (!gl) {
  overlay.querySelector('.msg').innerHTML =
    '<b>No WebGL2</b>This browser cannot run the renderer.';
  throw new Error('webgl2 unavailable');
}

// Which GPU is drawing, shown in the console and the scene section.
let gpuName = 'unknown';
if (gl) {
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  gpuName = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)
                : gl.getParameter(gl.RENDERER);
  console.log(`GPU: ${gpuName}`);
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
  for (const n of names) {
    const u = 'u' + n[0].toUpperCase() + n.slice(1);
    // Array uniforms: some drivers only answer to 'name[0]'.
    out[n] = gl.getUniformLocation(p, u) || gl.getUniformLocation(p, u + '[0]');
  }
  return out;
}

/** Mean colour of each class's splats, weighted by opacity. */
function meanClassColours(colour, tiles, count) {
  const acc = Array.from({ length: count }, () => [0, 0, 0, 0]);
  for (const t of tiles) {
    const a = acc[Math.min(count - 1, t.class || 0)];
    for (let i = t.start; i < t.start + t.count; i++) {
      const w = colour[4 * i + 3] / 255;
      a[0] += colour[4 * i] * w; a[1] += colour[4 * i + 1] * w;
      a[2] += colour[4 * i + 2] * w; a[3] += w;
    }
  }
  return acc.map(([r, g, b, w]) => (w > 0
    ? [r / w / 255, g / w / 255, b / w / 255] : [0.6, 0.6, 0.6]));
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

/** The camera, in two modes.
 *
 *  Orbit matches orbit_camera() in bozkir/camera.py: z up, elevation 0
 *  along the ground, 90 straight down, eye on a sphere around a target.
 *  Fly holds the eye and turns the view. Both share azimuth and elevation,
 *  so switching keeps the view. Elevation stops just short of +-90, where
 *  forward and up are parallel and the view would spin. */
class Orbit {
  constructor() {
    this.target = [0, 0, 0];
    this.distance = 5;
    this.azimuth = 45;
    this.elevation = 25;
    this.fov = 60;
    this.fly = false;
    this.pos = [0, 0, 0];        // eye, when flying
    this.speed = 1;              // world units per second, when flying
  }

  /** Unit vector the camera looks along. */
  dir() {
    const az = this.azimuth * Math.PI / 180, el = this.elevation * Math.PI / 180;
    return [-Math.cos(el) * Math.cos(az), -Math.cos(el) * Math.sin(az),
            -Math.sin(el)];
  }

  eye() {
    if (this.fly) return this.pos.slice();
    const az = this.azimuth * Math.PI / 180, el = this.elevation * Math.PI / 180;
    return [this.target[0] + this.distance * Math.cos(el) * Math.cos(az),
            this.target[1] + this.distance * Math.cos(el) * Math.sin(az),
            this.target[2] + this.distance * Math.sin(el)];
  }

  /** Keep the view unchanged when the mode changes. */
  setFly(on) {
    if (on === this.fly) return;
    if (on) this.pos = this.eye();
    else {
      const e = this.pos, f = this.dir();
      this.target = [e[0] + f[0] * this.distance, e[1] + f[1] * this.distance,
                     e[2] + f[2] * this.distance];
    }
    this.fly = on;
  }

  basis() {
    const e = this.eye();
    const f = this.dir();
    const up = Math.abs(f[2]) > 0.999 ? [1, 0, 0] : [0, 0, 1];
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

let splatProg, lineProg, skyProg, farProg;
try {
  splatProg = program(SPLAT_VERT, SPLAT_FRAG);
  farProg = program(FAR_VERT, FAR_FRAG);
  lineProg = program(LINE_VERT, LINE_FRAG);
  skyProg = program(SKY_VERT, SKY_FRAG);
} catch (e) { fail(e); }
console.log('shaders compiled');

const splatU = uniforms(splatProg,
  ['view', 'eye', 'cellXY', 'relief', 'wave', 'subdiv',
   'focal', 'viewport', 'gain', 'near',
   'data', 'colour', 'tileSize', 'edgeMark', 'fade',
   'tint', 'tintAmount', 'fogColour', 'fogDensity', 'gridRot', 'mix',
   'field', 'fieldSize', 'fieldExtent', 'hasField',
   'open', 'hasOpen', 'shade', 'exposure', 'saturation',
   'farOn', 'farStart', 'farBand', 'detail', 'detailFreq',
   'edgeN', 'edgeE', 'edgeS', 'edgeW']);
const farU = uniforms(farProg,
  ['view', 'eye', 'focal', 'viewport', 'relief', 'wave', 'hasField', 'field',
   'fieldSize', 'fieldExtent', 'atlas', 'cells', 'atlasGrid', 'tileSize',
   'gridN', 'rot', 'farStart', 'farBand', 'fogColour', 'fogDensity',
   'exposure', 'shade', 'saturation', 'open', 'hasOpen', 'debug', 'classRGB',
   'tileInfo', 'hasTileInfo', 'detail', 'detailFreq']);
const lineU = uniforms(lineProg,
  ['view', 'eye', 'focal', 'viewport', 'near', 'alpha']);

// One VAO per program, or divisors leak between passes (the splat pass's
// divisor on attribute 1 broke the line colours).
const splatVAO = gl.createVertexArray();
const lineVAO = gl.createVertexArray();
const skyVAO = gl.createVertexArray();

const quadBuf = gl.createBuffer();
const indexBuf = gl.createBuffer();
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
// Class of each tile and how many classes; all zeros with one class.
let tileClass = null;
let classCount = 1;
let classOn = true;
let classSharp = 2.0;
// Share of the grid for the first class. Left to itself the rule gave ~27%
// on every scene; balance decides how much, the rule decides where.
let classBalance = 0.5;
// Patch size in tiles, so the result is regions, not a scatter of single tiles.
let classCoherence = 2;
// How much altitude counts against shape (without it a hollow on a summit
// read the same as one in a valley).
let classAltitude = 0.4;
// Off by default: it doubles draw calls near class boundaries.
let classBlend = false;
let blendWidth = 0.35;
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
// Mean colour of each class (linear 0..1), for the terrain window previews.
let classColours = [[0.62, 0.55, 0.45]];
let tileSize = 0;
let gridN = 1;
let cells = [];
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
let atlas = null;                // { tex, cols, rows }
let cellTex = null;
let tileInfoTex = null;   // per tile: colour gain, mean colour
let farVAO = null, farCount = 0;
// A scene that is not a Wang tileset is shown as the capture itself: no
// warping onto terrain, no material rule.
let plainScene = false;
let tileMeans = null;   // mean splat colour per tile, linear 0..1

// Auto quality, the way games keep a frame budget.
// First lever: render resolution (50-100%, 33.3 ms budget, Unreal's dynamic
// resolution defaults), the canvas scaled up by the browser. Second, only
// when that is not enough: the LOD distance, capped by the slider.
// Proportional control (Intel's dynamic-resolution sample) plus a panic drop
// on several frames far over budget; recovery slower than the drop.
let autoQuality = true;
let lodBaseUser = 8;            // what the slider says
let renderScale = 1;            // fraction of full resolution
let frameMs = 16;               // smoothed frame time
let overRun = 0, qualityTick = 0, lastNote = '';
const BUDGET_MS = 33.3;
const SCALE_MIN = 0.5;
const QUALITY_MIN_BASE = 3;

function qualityNote(text) {
  // At most one note every 1.5 s.
  const now = performance.now();
  if (!ui.busy || text === lastNote || now - (qualityNote.at || 0) < 1500) return;
  qualityNote.at = now;
  lastNote = text;
  ui.busy.textContent = text;
  ui.busy.style.display = 'block';
  clearTimeout(qualityNote.t);
  qualityNote.t = setTimeout(() => {
    lastNote = '';
    if (!regenPending && ui.busy) ui.busy.style.display = 'none';
  }, 2000);
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
    lodBase = Math.max(QUALITY_MIN_BASE, Math.min(lodBaseUser, lodBase + detail));
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

/** Ground height at a point, the CPU copy of terrainHeight() in the shader. */
function height(x, y) {
  const relief = amplitude();
  if (relief <= 0) return 0;
  if (field) {
    // Centred on zero. The field's extent is set by relief scale (tiles per
    // landform), and it mirrors past its edge, so a small survey still reads
    // at any grid size.
    return relief * (field.sample(x, y) - 0.5) * 2.0 + detailAt(x, y);
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
              ground: [0.10, 0.10, 0.11], fog: 0.020 },
  dusk:     { top: [0.10, 0.13, 0.24], horizon: [0.72, 0.47, 0.35],
              ground: [0.05, 0.05, 0.07], fog: 0.030 },
  clear:    { top: [0.22, 0.45, 0.78], horizon: [0.70, 0.80, 0.90],
              ground: [0.08, 0.09, 0.10], fog: 0.012 },
};
let sky = 'overcast';
let fogScale = 1.0;
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

console.log('starting sort worker');
const worker = new Worker('./sort-worker.js');
worker.onmessage = (e) => {
  const m = e.data;

  if (m.type === 'cached') {
    cacheDirs = m.dirs;
    cacheBuf = new Uint32Array(m.order);
    cacheStride = m.stride;
    cacheMs = m.ms;
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
    if (m.seq !== mergeSeq) return;          // a later frame overtook it
    mergeHave = m.orders.map((b) => new Uint32Array(b));
    mergeHaveKey = mergeKey;
    mergeStats.ms = m.ms;
    mergeStats.splats = mergeHave.reduce((t, o) => t + o.length, 0);
    return;
  }

  if (m.type !== 'sorted') return;
  if (!sortingEnabled) { sortPending = false; return; }
  gl.bindVertexArray(splatVAO);
  gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
  lastOrder = new Uint32Array(m.order);
  gl.bufferData(gl.ARRAY_BUFFER, lastOrder, gl.DYNAMIC_DRAW);
  gl.bindVertexArray(null);
  sortMs = m.ms;
  sortedReady = true;
  sortPending = false;
};

// ================================================================ layout

/** Share of the grid per class: the slider sets the first class, the rest
 *  split what is left evenly. */
function classShares(count) {
  const first = Math.min(0.95, Math.max(0.05, classBalance));
  const rest = (1 - first) / Math.max(1, count - 1);
  return Array.from({ length: count }, (_, k) => (k === 0 ? first : rest));
}

/** A tile with the same edge code in another class, to blend against.
 *  Every class carries every code, so swapping cannot break the matching. */
function alternateTile(c, p) {
  if (!wangCodes || !tileClass) return -1;
  const code = wangCodes[c.patch];
  // Blend towards the class the rule ranked next (matters with 3+ classes).
  for (let k = 0; k < wangCodes.length; k++) {
    if (c.alt != null && tileClass[k] !== tileClassFor(c.alt)) continue;
    if (tileClass[k] === tileClass[c.patch]) continue;
    const o = wangCodes[k];
    if (o[0] === code[0] && o[1] === code[1]
        && o[2] === code[2] && o[3] === code[3]) return k;
  }
  return -1;
}

/** Lay out gridN x gridN cells.
 *
 *  With a Wang set, west is fixed by the left neighbour and south by the one
 *  below; north and east are free. A complete set always has a fit, so no
 *  backtracking, and the free choices stop the terrain repeating. */
function buildGrid() {
  cells = [];
  if (!tileSize) { cells = [{ x: 0, y: 0, z: 0, warp: FLAT, patch: 0 }]; return; }
  usedPatches = Math.max(1, Math.min(usedPatches || patches.length,
                                     patches.length));
  let s = seed;
  const rand = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const half = (gridN - 1) / 2;
  const chosen = new Int32Array(gridN * gridN).fill(-1);

  // Material per cell from the ground under it, once per layout. Classes
  // share edge codes, so the neighbours pick the code and the terrain picks
  // the class, and neither overrules the other.
  let wantClass = null;
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
                         sediment: sed,
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
  }

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
        // Prefer the class the rule asked for. The fallback should never
        // fire (every class has every code); it is counted so it would show.
        let pool = fits;
        if (wantClass && tileClass) {
          const want = tileClassFor(wantClass[j * gridN + i]);
          const ofClass = fits.filter(k => tileClass[k] === want);
          if (ofClass.length) pool = ofClass;
          else if (classStats) classStats.missed = (classStats.missed || 0) + 1;
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
  console.log(`overlay: ${lineCount / 2} segments`);
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
    `max ${(m.worst / unit).toFixed(2)} splat widths (${m.worst.toFixed(4)}), ` +
    `mean ${(m.mean / unit).toFixed(2)}, over ${m.count} samples`;
}

/** Load <scene>.atlas.png/.json if there is one; without it the far field
 *  stays off. */
async function loadAtlas(name) {
  atlas = null;
  try {
    const r = await fetch(`./data/${name}.atlas.json`);
    if (!r.ok) return;
    const lay = await r.json();
    const img = new Image();
    img.src = `./data/${name}.atlas.png`;
    await img.decode();
    // Per-tile colour gain: the atlas is seen from straight above, the
    // splats at a grazing angle, so each atlas tile is scaled to its own
    // splats' mean (removes most of the colour step at the hand-over).
    // Row 0: gain (0..2 as 0..1). Row 1: tile mean, what far ground fades to.
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
      const sm = tileMeans && tileMeans[k] ? tileMeans[k] : am;
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
    const tex = gl.createTexture();
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
    atlas = { tex, cols: lay.cols, rows: lay.rows, count: lay.count };
    console.log(`far field: atlas of ${lay.count} tiles`);
    buildFarField();
  } catch (e) { atlas = null; }
}

/** Upload which tile each cell holds, and build the far mesh over the grid. */
function buildFarField() {
  if (!atlas || !cells.length) return;
  // An atlas baked from an older export would paint the wrong tiles.
  if (wangCodes && atlas.count !== wangCodes.length) {
    console.warn(`far field off: atlas has ${atlas.count} tiles, the tileset `
                 + `${wangCodes.length}. Re-run scripts/bake_atlas.py.`);
    atlas = null;
    return;
  }
  const n = gridN;
  const idx = new Uint8Array(n * n * 4);
  for (const c of cells) {
    const k = 4 * (c.j * n + c.i);
    idx[k] = c.patch & 255;
    idx[k + 1] = (c.patch >> 8) & 255;
    idx[k + 2] = (c.cls || 0) & 255;
    idx[k + 3] = Math.round(255 * Math.max(0, Math.min(1, c.mix == null ? 1 : c.mix)));
  }
  if (!cellTex) cellTex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE6);
  gl.bindTexture(gl.TEXTURE_2D, cellTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, n, n, 0, gl.RGBA, gl.UNSIGNED_BYTE, idx);

  // Regular mesh, ~2 vertices per tile: enough at the distances it is used.
  const m = Math.max(32, Math.min(512, 2 * n + 1));
  const extent = n * tileSize;
  const a = gridAngle * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a);
  const xy = new Float32Array(m * m * 2);
  for (let j = 0; j < m; j++) {
    for (let i = 0; i < m; i++) {
      const lx = (i / (m - 1) - 0.5) * extent, ly = (j / (m - 1) - 0.5) * extent;
      xy[2 * (j * m + i)] = ca * lx - sa * ly;
      xy[2 * (j * m + i) + 1] = sa * lx + ca * ly;
    }
  }
  const ix = new Uint32Array((m - 1) * (m - 1) * 6);
  let t = 0;
  for (let j = 0; j < m - 1; j++) {
    for (let i = 0; i < m - 1; i++) {
      const v = j * m + i;
      ix[t++] = v; ix[t++] = v + 1; ix[t++] = v + m;
      ix[t++] = v + 1; ix[t++] = v + m + 1; ix[t++] = v + m;
    }
  }
  if (!farVAO) farVAO = gl.createVertexArray();
  gl.bindVertexArray(farVAO);
  const vb = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vb);
  gl.bufferData(gl.ARRAY_BUFFER, xy, gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(farProg, 'aXY');
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const ib = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, ix, gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  farCount = ix.length;
}

function regenerate() { buildGrid(); buildOverlay(); reportSeams(); buildFarField(); }

/** Rebuild after a control moves. The rebuild blocks the main thread (a
 *  second or more at hundreds of tiles), so the note is painted first, two
 *  frames ahead; moves in the meantime fold into one rebuild. */
let regenPending = false;
function scheduleRegenerate() {
  if (regenPending) return;
  regenPending = true;
  const big = gridN * gridN > 4096;
  if (big && ui.busy) {
    ui.busy.textContent = `rebuilding ${(gridN * gridN).toLocaleString()} cells…`;
    ui.busy.style.display = 'block';
  }
  requestAnimationFrame(() => requestAnimationFrame(() => {
    regenPending = false;
    regenerate();
    if (ui.busy) ui.busy.style.display = 'none';
  }));
}

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
    plainScene = !manifest.wang;
    tileClass = manifest.tiles.map(t => t.class || 0);
    classCount = Math.max(1, manifest.classes || 1);
    classColours = meanClassColours(colour, manifest.tiles, classCount);
    // Each tile's own mean, for matching the far field to it.
    tileMeans = manifest.tiles.map((t) =>
      meanClassColours(colour, [{ start: t.start, count: t.count }], 1)[0]);
  } else {
    patches = [{ start: 0, count: n, levels: [[0, n]] }];
    lodLevels = 1;
    tileSize = 0;
    wangCodes = null;
    plainScene = true;
    tileClass = null;
    classColours = meanClassColours(colour, [{ start: 0, count: n }], 1);
    classCount = 1;
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

  positionsRef = positions;
  const pos = positions.slice();
  // Each LOD level holds different splats, so each is sorted on its own.
  const ranges = [];
  for (const p of patches) {
    for (const [start, count] of p.levels) ranges.push({ start, count });
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
  if (ui.wangnote) ui.wangnote.textContent = wangCodes
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
// Panel-driven; window.bozkirTopo / bozkirMerge mirror them for the console.
let topoOrder = true;
let mergeOn = true;
let mergeThreshold = 0.5;   // in tiles
let mergeSeq = 0, mergeKey = '', mergeHave = null, mergeHaveKey = '';
let mergeStats = { groups: 0, cells: 0, ms: 0, splats: 0 };
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

function frame() {
  // Frame time for the benchmark and auto quality (performance.now, not the
  // rAF timestamp, so throttling reads the same).
  {
    const t = performance.now();
    if (benchPrev) benchDt = t - benchPrev;
    benchPrev = t;
  }
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
  // The far field is the next LOD level, fading in over the same last fifth
  // of an octave as lodFor(). It starts one level before the coarsest (16
  // tiles at the default instead of 32): the coarsest splat level added
  // little over the atlas and cost most of the frame.
  const farStart = (lodOn && lodLevels > 1
    ? lodBase * Math.pow(2, Math.max(0, lodLevels - 2)) : farStartTiles) * (tileSize || 1);
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
    gl.uniform1f(farU.gridN, gridN);
    const ga = gridAngle * Math.PI / 180;
    gl.uniform2f(farU.rot, Math.cos(ga), Math.sin(ga));
    gl.uniform1f(farU.farStart, farStart);
    gl.uniform1f(farU.farBand, farBand);
    gl.uniform3fv(farU.fogColour, new Float32Array(S.horizon));
    gl.uniform1f(farU.fogDensity, S.fog * fogScale);
    gl.uniform1f(farU.exposure, exposure);
    gl.uniform1f(farU.shade, groundShade);
    gl.uniform1f(farU.saturation, saturation);
    gl.uniform1i(farU.open, 4);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, field && openTex ? openTex : null);
    gl.uniform1f(farU.hasOpen, field && openTex ? 1.0 : 0.0);
    gl.uniform1i(farU.debug, showClasses && classCount > 1 ? 2
      : showIdentity ? 1 : showLodColours ? 3 : 0);
    gl.uniform3fv(farU.classRGB, new Float32Array(CLASS_RGB.flat()));
    gl.uniform1f(farU.detail, amplitude() > 0 ? detailAmp() : 0);
    gl.uniform1f(farU.detailFreq, detailFreq());
    gl.uniform1i(farU.tileInfo, 7);
    gl.activeTexture(gl.TEXTURE7);
    gl.bindTexture(gl.TEXTURE_2D, tileInfoTex);
    gl.uniform1f(farU.hasTileInfo, tileInfoTex ? 1.0 : 0.0);
    gl.drawElements(gl.TRIANGLES, farCount, gl.UNSIGNED_INT, 0);
    gl.bindVertexArray(null);
    gl.disable(gl.DEPTH_TEST);
    drawCalls++;
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
    gl.uniform1f(splatU.fogDensity, S.fog * fogScale);

    // Cells drawn far to near and composited 'over'. Splats are sorted
    // within a patch, never across cells - the source of boundary artifacts.
    const visible = [];
    const half = (tileSize || 1) / 2;
    const tanHalf = Math.tan(cam.fov * Math.PI / 360);
    const aspect = canvas.width / canvas.height;

    for (let ci = 0; ci < cells.length; ci++) {
      const c = cells[ci];
      const dx = c.x - b.eye[0], dy = c.y - b.eye[1], dz = c.z - b.eye[2];
      const z = dx * b.forward[0] + dy * b.forward[1] + dz * b.forward[2];
      if (z < -tileSize * 2.0) continue;
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
      for (const [ox, oy] of [[-half, -half], [half, -half],
                              [-half, half], [half, half]]) {
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
    // window.bozkirTopo = false restores the old ranking for comparison.
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
    let groupOf = null, groups = null;
    if (mergeOn && tileSize && visible.length > 1) {
      const vc = visible.map((v) => v.c);
      const ar = gridAngle * Math.PI / 180;
      const ga = { c: Math.cos(ar), s: Math.sin(ar) };
      const g = mergeGroups(vc, b.eye, { threshold: tileSize * mergeThreshold });
      const multi = g.groups.filter((x) => x.length > 1);
      if (multi.length) {
        groupOf = g.groupOf; groups = g.groups;
        mergeStats.groups = multi.length;
        mergeStats.cells = multi.reduce((t, x) => t + x.length, 0);

        // What a returned stream must match: the cells, the groups and the
        // rough heading (or a stream for the opposite heading draws back
        // to front).
        const dir = b.forward.map((v) => Math.round(v * 24)).join(',');
        const key = dir + '|' + groups.map((x) => x.map(
          (k) => `${vc[k].i}.${vc[k].j}`).join('+')).join('/');
        if (key !== mergeKey) {
          mergeKey = key;
          mergeSeq++;
          worker.postMessage({
            type: 'mergeGroups', eye: [0, 0, 0], seq: mergeSeq,
            // View direction in the tile frame (positions are tile-local and
            // the grid may be rotated); world offsets go in `shift`.
            forward: [ ga.c * b.forward[0] + ga.s * b.forward[1],
                      -ga.s * b.forward[0] + ga.c * b.forward[1],
                       b.forward[2] ],
            groups: groups.map((x) => x.map((k) => {
              const c = vc[k], pp = patches[c.patch];
              const [start, count] = pp.levels[0];
              const shift = (c.x - b.eye[0]) * b.forward[0]
                          + (c.y - b.eye[1]) * b.forward[1]
                          + (c.z - b.eye[2]) * b.forward[2];
              return { patch: c.patch, start, count, shift };
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
    const canMerge = groups && mergeHave && mergeHaveKey === mergeKey
                  && mergeHave.length === groups.length;
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
        if (groups[gi].length > 1) {
          if (groupDrawn[gi]) continue;
          groupDrawn[gi] = 1;
          const stream = mergeHave[gi];
          if (stream && stream.length) {
            cellXYArr.fill(0);
            groups[gi].forEach((k, slot) => {
              if (slot >= MAX_GROUP) return;
              cellXYArr[2 * slot] = visible[k].c.x;
              cellXYArr[2 * slot + 1] = visible[k].c.y;
            });
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
      let mix = -1, other = -1;
      if (classBlend && classCount > 1 && tileClass && c.mix != null) {
        if (c.mix < blendWidth) {
          // Half and half at a tie, all its own class at the band's edge.
          mix = 0.5 + 0.5 * (c.mix / Math.max(blendWidth, 1e-6));
          other = alternateTile(c, p);
        }
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

      // Live: one order along the world view direction (right only for
      // unwarped tiles). Cached: the order for the view direction in the
      // cell's own frame (W^T v), which is what a warped tile sees.
      let buf = indexBuf, base = 0;
      if (sortMode === 'cached' && cacheReady) {
        const W = c.warp, f = b.forward;
        const local = [W[0] * f[0] + W[1] * f[1] + W[2] * f[2],
                       W[3] * f[0] + W[4] * f[1] + W[5] * f[2],
                       W[6] * f[0] + W[7] * f[1] + W[8] * f[2]];
        let best = 0, bestDot = -2;
        for (let k = 0; k < cacheDirs.length; k++) {
          const d2 = cacheDirs[k][0] * local[0] + cacheDirs[k][1] * local[1]
                   + cacheDirs[k][2] * local[2];
          if (d2 > bestDot) { bestDot = d2; best = k; }
        }
        buf = cacheIndexBuf;
        base = best * cacheStride;
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      for (const [li, w] of [[lvl, wA], [lvl + 1, wB]]) {
        if (w <= 0.001 || li >= p.levels.length) continue;
        const [start, count] = p.levels[li];
        if (!count) continue;
        gl.uniform1f(splatU.fade, w);
        // Only set the tint when this view owns it (zeroing it here once
        // broke the tile-identity view).
        if (showLodColours) {
          const c2 = LOD_RGB[li % LOD_RGB.length];
          gl.uniform3f(splatU.tint, c2[0], c2[1], c2[2]);
          gl.uniform1f(splatU.tintAmount, 0.6);
        }
        gl.vertexAttribIPointer(aIndex, 1, gl.UNSIGNED_INT, 0,
                                (base + start) * 4);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
        drawnSplats += count; drawCalls++;

        // The other class, with the complementary dissolve, so each spot
        // ends up with one splat rather than two or none.
        if (other >= 0 && patches[other]
            && li < patches[other].levels.length) {
          const [os, oc] = patches[other].levels[li];
          if (oc) {
            gl.uniform1f(splatU.mix, 1.0 - mix);
            gl.vertexAttribIPointer(aIndex, 1, gl.UNSIGNED_INT, 0,
                                    (base + os) * 4);
            gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, oc);
            drawnSplats += oc; drawCalls++;
            blendedCells++;
            gl.uniform1f(splatU.mix, mix);
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

  frames++;
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
        ? `${(popRate * 100).toFixed(2)} now, ${(popPeak * 100).toFixed(2)} peak`
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
        const share = classStats.counts
          .map((c, k) => `${k}: ${(100 * c / total).toFixed(0)}%`).join('  ');
        // A field squeezed into a few tiles repeats across the grid, and the
        // rule then looks like a chequerboard; say so.
        const repeats = field && reliefScale > 0
          ? Math.round(gridN / reliefScale) : 1;
        // What each cue actually weighs, so the sliders' effect is visible.
        const cues = (classStats.weights || [])
          .filter((c) => c.w > 0.005)
          .map((c) => `${c.cue} ${(c.w * 100).toFixed(0)}%`).join(' \u00b7 ');
        ui.classnote.textContent = share
          + `   confidence ${(classStats.sure * 100).toFixed(0)}%`
          + (cues ? `\n${cues}` : '')
          + (classBlend ? `   ${blendedCells} blended` : '')
          + (classStats.source === 'sediment' ? '   from sediment' : '')
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
              + `${mergeStats.cells} cells, ${mergeStats.ms.toFixed(0)} ms`
            : 'none here');
    }
    if (orderStats.cycles > 0 && !cycleWarned) {
      cycleWarned = true;
      console.warn(`tile order: ${orderStats.cycles} cycle(s) broken by depth `
                 + `key - relief has tilted boundary planes into disagreement`);
    }
    frames = 0; fpsTime = now;
  }
  if (ui.azim) ui.azim.textContent = cam.azimuth.toFixed(0) + '\u00b0';
  if (ui.elev) ui.elev.textContent = cam.elevation.toFixed(0) + '\u00b0';
  if (ui.dist) ui.dist.textContent = cam.fly
    ? `fly @ ${cam.eye().map(v => v.toFixed(1)).join(', ')}`
    : cam.distance.toFixed(2);
  if (now - fpsTime > 500) reportSortError();
  if (ui.cmd) ui.cmd.textContent = `--azim ${cam.azimuth.toFixed(0)} ` +
    `--elev ${cam.elevation.toFixed(0)} --dist ${cam.distance.toFixed(2)} ` +
    `--fov ${cam.fov.toFixed(0)}`;

  if (orbiting) cam.azimuth = (cam.azimuth + 0.08) % 360;

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
canvas.addEventListener('pointerdown', (e) => {
  dragging = true; lastX = e.clientX; lastY = e.clientY;
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointerup', (e) => {
  dragging = false; canvas.releasePointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  // Orbit swings the eye around a point; fly turns the view about the eye.
  const s = cam.fly ? 0.18 : -0.3;
  cam.azimuth = ((cam.azimuth + (e.clientX - lastX) * s) % 360 + 360) % 360;
  cam.elevation = Math.max(-89, Math.min(89,
    cam.elevation + (e.clientY - lastY) * (cam.fly ? -0.18 : 0.3)));
  lastX = e.clientX; lastY = e.clientY;
});
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
// Movement runs in the frame loop with a velocity that eases in and out
// (~0.1 s). The old fixed step on its own timer drifted against the frames
// and read as judder.
const camVel = [0, 0, 0];
const EASE_S = 0.12;

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
  const b = cam.basis();
  const fast = held.has('shift') ? 4 : 1;
  const want = [0, 0, 0];
  let speed, fwd, rgt;
  if (cam.fly) {
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
  if (held.has('e')) want[2] += speed;
  if (held.has('q')) want[2] -= speed;

  const k = 1 - Math.exp(-sec / EASE_S);
  let still = true;
  for (let i = 0; i < 3; i++) {
    camVel[i] += (want[i] - camVel[i]) * k;
    if (Math.abs(camVel[i]) < 1e-4 * Math.max(speed, 1e-3)) camVel[i] = 0;
    if (camVel[i]) still = false;
  }
  if (still) return;
  const p = cam.fly ? cam.pos : cam.target;
  for (let i = 0; i < 3; i++) p[i] += camVel[i] * sec;

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
  overlay.classList.remove('hidden');
  const msg = overlay.querySelector('.msg b');
  msg.textContent = 'opening ' + list[0].name;
  bar.style.width = '20%';
  try {
    const t = await openDrop(list);
    bar.style.width = '70%';
    const d = describe(t.meta);
    // Without metadata the tile size is a guess; say so.
    if (d.note) console.warn(`${t.name}: ${d.note}; tiling may be wrong`);
    msg.textContent = 'loading ' + t.name;
    load(t.buffer, t.meta);
    bar.style.width = '100%';
  } catch (err) {
    console.error(err);
    msg.textContent = 'could not open: ' + err.message;
    bar.style.width = '0%';
    setTimeout(() => overlay.classList.add('hidden'), 4000);
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

// The drop hint is built here, so index.html works without this module.
{
  const style = document.createElement('style');
  style.textContent = `
    #drophint { position: fixed; inset: 0; display: none; z-index: 50;
      align-items: center; justify-content: center; pointer-events: none;
      background: rgba(10,10,12,0.72);
      font: 500 15px/1.5 ui-monospace, Menlo, Consolas, monospace;
      color: #e8e2d8; letter-spacing: 0.02em; text-align: center; }
    body.dropping #drophint { display: flex; }
    #drophint span { border: 1px dashed #c8a05a; border-radius: 10px;
      padding: 28px 40px; }`;
  document.head.appendChild(style);
  const hint = document.createElement('div');
  hint.id = 'drophint';
  hint.innerHTML = '<span>drop a tileset<br>'
    + '<small style="opacity:.65">a .zip, or the .splat and .json together'
    + '</small></span>';
  document.body.appendChild(hint);
}

document.getElementById('fov').addEventListener('input',
  (e) => { cam.fov = +e.target.value; });
document.getElementById('gain').addEventListener('input',
  (e) => { gain = +e.target.value; });
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
  // Relief scale follows the grid until the scale slider is touched.
  if (field && !reliefScaleTouched && ui.reliefscale) {
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
  try { localStorage.setItem('bozkir.expert', e.target.checked ? '1' : ''); }
  catch (err) { /* private window: the choice just does not persist */ }
});
if (ui.expert) {
  let saved = '';
  try { saved = localStorage.getItem('bozkir.expert') || ''; } catch (e) { /**/ }
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
  else if (k === 'escape' && figureMode) setFigureMode(false);
});
on('figexit', 'click', () => setFigureMode(false));
on('farfield', 'change', (e) => { farOn = e.target.checked; });
if (ui.gpuname) {
  ui.gpuname.textContent = String(gpuName).replace(/^ANGLE \((.*)\)$/, '$1');
  // Warn when it looks like an integrated or software GPU.
  if (/intel|uhd|iris|integrated|swiftshader|llvmpipe/i.test(gpuName)) {
    ui.gpuname.style.color = '#d9a441';
    ui.gpuname.title = 'This is an integrated or software GPU. If the machine has '
      + 'a dedicated graphics card, set the browser to "High performance" in '
      + 'Windows Settings > System > Display > Graphics.';
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
  cam.setFly(e.target.checked);
  ui.flynote.textContent = cam.fly
    ? 'drag looks around, WASD flies, Q/E down and up, shift is faster, '
      + 'scroll changes speed'
    : 'drag orbits, WASD slides the centre, scroll zooms';
  if (ui.speedn) ui.speedn.textContent = cam.speed.toFixed(2);
});
for (const [id, elev, dist] of [['viewGround', 3, 8], ['viewWalk', 12, 6],
                                ['viewSurvey', 32, 14], ['viewTop', 85, 18]]) {
  on(id, 'click', () => {
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
  if (ui.gridanglen) ui.gridanglen.textContent = `${gridAngle}\u00b0`;
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
  cam.azimuth = 45; cam.elevation = 25;
});

// --- ordering controls ------------------------------------------------

on('tileorder', 'change', (e) => { topoOrder = e.target.value !== 'depth'; });

on('classrule', 'change', (e) => { classOn = e.target.checked; scheduleRegenerate(); });
on('classswap', 'change', (e) => { classSwap = e.target.checked; scheduleRegenerate(); });
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
on('blendwidth', 'input', (e) => {
  blendWidth = +e.target.value;
  ui.blendwidthn.textContent = blendWidth.toFixed(2);
});
on('classsharp', 'input', (e) => {
  classSharp = +e.target.value;
  ui.classsharpn.textContent = classSharp.toFixed(1);
  scheduleRegenerate();
});
on('merging', 'change', (e) => {
  mergeOn = e.target.checked;
  // Drop any stale stream, or it is drawn the moment merging comes back.
  mergeKey = ''; mergeHave = null; mergeHaveKey = '';
});
on('mergethr', 'input', (e) => {
  mergeThreshold = +e.target.value;
  ui.mergethrn.textContent = mergeThreshold.toFixed(2);
  mergeKey = '';
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

// Console shortcuts, from before the panel had these controls.
Object.defineProperty(window, 'bozkirTopo', {
  get: () => topoOrder,
  set: (v) => { topoOrder = v !== false; if (ui.tileorder) ui.tileorder.value = topoOrder ? 'topological' : 'depth'; },
});
Object.defineProperty(window, 'bozkirMerge', {
  get: () => mergeOn,
  set: (v) => { mergeOn = v !== false; if (ui.merging) ui.merging.checked = mergeOn; mergeKey = ''; },
});

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
  console.log(`sweep ${arc}\u00b0 in ${frames} frames `
    + `(${(arc / frames).toFixed(2)}\u00b0 each) about `
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
const wanted = params.get('scene');
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
// capture), then the desert tileset that ships with the repo.
const DEFAULT_SCENES = ['scene', 'desert'];

/** Fill the tileset and terrain menus from data/index.json (a static server
 *  cannot list a folder) plus the defaults, keeping only files that are
 *  really there (a HEAD request each), so no entry leads to a 404. */
async function loadCatalogue() {
  let index = {};
  try {
    const r = await fetch('./data/index.json');
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

async function loadPresets() {
  try {
    const r = await fetch('./views.json');
    presets = r.ok ? await r.json() : {};
  } catch (e) { presets = {}; }
  if (!ui.presetchips) return;
  for (const [name, p] of Object.entries(presets)) {
    if (name.startsWith('_')) continue;
    const b = document.createElement('button');
    b.type = 'button';
    b.dataset.preset = name;
    b.textContent = p.short || name;
    b.dataset.tip = p.label || name;
    b.title = p.label || name;
    b.addEventListener('click', () => applyPreset(name));
    ui.presetchips.appendChild(b);
  }
  // If the ground arrived before this file, the URL's preset is still waiting.
  if (pendingPreset && presets[pendingPreset] && field) {
    const name = pendingPreset;
    pendingPreset = null;
    applyPreset(name);
  }
}

function setControl(id, value) {
  const el = document.getElementById(id);
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

function applyPreset(name) {
  const p = presets[name];
  if (!p) { console.warn(`no preset '${name}'`); return; }
  // Grid first, since 'reliefscale: grid' and the rule both depend on it.
  if ('grid' in p) setControl('grid', p.grid);
  for (const [k, v] of Object.entries(p)) {
    if (k === 'label' || k === 'short' || k === 'grid' || k === 'view') continue;
    setControl(k, k === 'reliefscale' && v === 'grid' ? gridN : v);
  }
  if (p.view) document.getElementById(p.view)?.click();
  if (ui.presetchips) {
    for (const b of ui.presetchips.children) b.classList.toggle('on', b.dataset.preset === name);
  }
  const q = new URLSearchParams(location.search);
  q.set('preset', name);
  history.replaceState(null, '', `${location.pathname}?${q}`);
  console.log(`preset ${name} applied`);
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
function loadStarter() {
  if (splatCount) return;
  const { buffer, manifest } = starterTileset();
  loadedScene = 'starter';
  load(buffer, manifest);
  loadCatalogue();
  if (!pendingPreset) pendingPreset = 'readme';
  useGeneratedField(wantedGen || 'rolling', wantedGen ? wantedSeed : 0,
                    wantedGen ? wantedSize : 256);
  console.log('no scene found: showing the starter tileset');
}

/** Load the first of `names` that exists, one after another (fetched in
 *  parallel, the smallest file won, not the first in the list). */
async function loadFirst(names) {
  for (const name of names) {
    let b = null, m = null;
    try {
      [b, m] = await Promise.all([
        fetch(`./data/${name}.splat`).then(r => (r.ok ? r.arrayBuffer() : null)),
        fetch(`./data/${name}.json`).then(r => (r.ok ? r.json() : null)).catch(() => null),
      ]);
    } catch (e) { continue; }
    if (!b || splatCount) continue;
    loadedScene = name;
    load(b, m);
    // With nothing asked for, a tileset opens on the README settings; a
    // plain capture keeps its 'no rule' preset.
    if (!wanted && !pendingPreset && !plainScene) pendingPreset = 'readme';
    loadAtlas(name);
    loadCatalogue();
    // The height field arrives after the splats; the scene shows meanwhile.
    if (wantedGen) useGeneratedField(wantedGen, wantedSeed, wantedSize);
    else useHeightField(wantedHeight || name);
    return true;
  }
  return false;
}

if (wanted === 'starter') loadStarter();
else loadFirst(wanted ? [wanted] : DEFAULT_SCENES)
  .then((ok) => { if (!ok) loadStarter(); });

/** Look for a height field beside a tileset and adopt it if there is one. */
async function useHeightField(name) {
  const f = await loadHeightField('./data', name);
  if (!f) return;
  adoptField(f);
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

  // Openness (r) and the detail mask (g), once per field.
  if (!openTex) openTex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE4);
  gl.bindTexture(gl.TEXTURE_2D, openTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const open = openness(f.z, f.w, f.h);
  f.mask = roughnessMask(f.z, f.w, f.h, open);
  const rg = new Float32Array(f.w * f.h * 2);
  for (let i = 0; i < f.w * f.h; i++) { rg[2 * i] = open[i]; rg[2 * i + 1] = f.mask[i]; }
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32F, f.w, f.h, 0, gl.RG, gl.FLOAT, rg);
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

paintSliders();
frame();
