// GLSL for the viewer: splats, far field, overlay lines and sky. The
// splat projection follows bozkir/camera.py (3DGS Eq. 5); the ground is
// TERRAIN_GLSL, shared by both vertex shaders so they always agree.

const DILATION = 0.3;     // matches DILATION in bozkir/render.py

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

export const SPLAT_VERT = `#version 300 es
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
// splats. uMix < 0 turns it off. The weight varies across the tile from
// its corners (shared with the neighbours, so the boundary is continuous
// instead of one level per tile) and is broken up by noise where the rule
// is undecided, so the boundary wanders (height-blend terrain splatting).
uniform float uMix;
uniform vec4 uMixCorners;     // own-class weight at sw, se, nw, ne
uniform float uMixNoise;      // +1 for the cell's class, -1 for the other
// Blend by spot (spotShare, spotField in frames.js): each spot is one
// material or another. Which, from the drawn material's field over the
// grid around the cell (7 x 7 values at half-cell steps from -1 to 2 cells:
// 1 inside it, 0.5 where it ties, less where another leads), read where
// noise moves the point by up to uMixWarp cells, so boundaries wander in
// and out of both sides; sharpened, so only a narrow strip dissolves,
// where a splat standing above its ground keeps more of its chance (as in
// the frames' ring). 0: splat by splat from the corners, as before (two
// thinned layers, see-through).
uniform float uSpots;
uniform vec4 uMixField[13];
uniform float uMixWarp;
float mixField(int k) { return uMixField[k >> 2][k & 3]; }
// Frames (bozkir/frames.py): per slot, how the patch is turned in its cell
// (cos, sin), what it is (0 a whole tile, 1 a frame, 2 a centre) and the
// height of its ground; and the ring where frame and centre meet.
uniform vec2 uSlotTurn[8];
uniform float uSlotRole[8];
uniform float uSlotZ0[8];
// A centre's tint, so its ground meets its frame's along the ring: seven
// terms per slot (toneTerms in frames.js), carried in as r^k.
uniform vec3 uSlotTone[56];
uniform vec4 uRing;           // radius, half width, wobble (world units), shape
uniform vec2 uProm;           // height over the ground: starts to count, counts fully
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
uniform float uMacro;         // variation over the land; 0 is off

${TERRAIN_GLSL}

// Slow brightness variation over the land, at scales of several tiles.
// A tileset repeats at one tile; this does not, so it breaks the lattice the
// eye picks out (landscape 'macro variation'). The same function in the far
// field, so the hand-over keeps it. Render-time only; figures turn it off.
float macroAt(vec2 p) {
  vec2 w = p / uTileSize;
  float m = 0.6 * dValue(w / 11.0 + 3.7) + 0.4 * dValue(w / 4.3 + 17.1);
  return 1.0 + uMacro * 0.36 * (m - 0.5);
}

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

// Distance from the cell's middle as the centre's shape measures it
// (ringNorm in frames.js): a disc for 2 or less, the p-norm's rounded square.
float ringNorm(vec2 l, float p) {
  if (p <= 2.0) return length(l);
  float m = max(abs(l.x), abs(l.y));
  if (m <= 0.0) return 0.0;
  vec2 u = abs(l) / m;
  return m * pow(pow(u.x, p) + pow(u.y, p), 1.0 / p);
}

// Frames: whether a splat of a frame (role 1) or a centre (role 2) stays
// where the two meet. The ring's line wobbles along the ground, the same
// for both sides; across it each side's share falls off, sharpened so most
// of the ring is one side or the other (hex-tiling's gamma). A splat
// standing above its ground and opaque keeps more of its chance, so a tuft
// near the line is kept whole, not cut (height-based blending). Mirrors
// centreShare and keepChance in frames.js.
bool keepInRing(vec4 a, uint sid, float role, float z0, vec2 cellXY) {
  float r = ringNorm(a.xy, uRing.w);
  float reach = uRing.y + uRing.z;
  if (r < uRing.x - reach) return role > 1.5;
  if (r > uRing.x + reach) return role < 1.5;
  vec2 dir = a.xy / max(length(a.xy), 1e-6);
  vec2 g = cellXY + uRing.x * vec2(uGridRot.x * dir.x - uGridRot.y * dir.y,
                                   uGridRot.y * dir.x + uGridRot.x * dir.y);
  float t = max(uTileSize, 1e-6);
  float n = 0.65 * dValue(g / (0.23 * t)) + 0.35 * dValue(g / (0.09 * t) + 17.0) - 0.5;
  float rho = uRing.x + 2.0 * uRing.z * n;
  float s = clamp(0.5 - 0.5 * (r - rho) / max(uRing.y, 1e-6), 0.0, 1.0);
  float s3 = s * s * s, o3 = (1.0 - s) * (1.0 - s) * (1.0 - s);
  float share = s3 / max(s3 + o3, 1e-6);
  if (role < 1.5) share = 1.0 - share;
  float prom = clamp((a.z - z0 - uProm.x) / max(uProm.y - uProm.x, 1e-6), 0.0, 1.0)
             * smoothstep(0.15, 0.5, a.w);
  float keep = pow(share, 1.0 / (1.0 + 3.0 * prom));
  uint h = (sid ^ 0x68E31DA4u) * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  return float((h >> 22u) ^ h) / 4294967296.0 < keep;
}

void main() {
  uint slot = aIndex >> 29u;
  uint sid = aIndex & 0x1FFFFFFFu;
  int si = int(slot);
  vec2 cellXY = uCellXY[si];

  vec4 a = fetch(sid, 0);     // position.xyz, opacity
  // A centre is turned in its cell; everything else has turn (1, 0).
  vec2 turn = uSlotTurn[si];
  a.xy = vec2(turn.x * a.x - turn.y * a.y, turn.y * a.x + turn.x * a.y);
  float role = uSlotRole[si];
  if (role > 0.5 && !keepInRing(a, sid, role, uSlotZ0[si], cellXY)) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }
  if (uMix >= 0.0) {
    vec2 t = clamp(a.xy / max(uTileSize, 1e-6) + 0.5, 0.0, 1.0);
    float w = mix(mix(uMixCorners.x, uMixCorners.y, t.x),
                  mix(uMixCorners.z, uMixCorners.w, t.x), t.y);
    vec2 g = cellXY + vec2(uGridRot.x * a.x - uGridRot.y * a.y,
                           uGridRot.y * a.x + uGridRot.x * a.y);
    if (uSpots > 0.5) {
      // The same world point moves the same way from either cell.
      float T = max(uTileSize, 1e-6);
      // Three scales, down to a sixth of a tile, so the line is ragged.
      vec2 d = vec2(0.55 * dValue(g / (1.1 * T)) + 0.3 * dValue(g / (0.4 * T) + 5.0)
                      + 0.15 * dValue(g / (0.15 * T) + 9.0),
                    0.55 * dValue(g / (1.1 * T) + 41.0) + 0.3 * dValue(g / (0.4 * T) + 77.0)
                      + 0.15 * dValue(g / (0.15 * T) + 23.0))
             - 0.5;
      vec2 u = (clamp(a.xy / T + 0.5 + 2.0 * uMixWarp * d, -1.0, 2.0) + 1.0) * 2.0;
      vec2 f0 = min(floor(u), vec2(5.0)), fr = u - f0;
      int k = int(f0.y) * 7 + int(f0.x);
      w = mix(mix(mixField(k), mixField(k + 1), fr.x),
              mix(mixField(k + 7), mixField(k + 8), fr.x), fr.y);
    } else {
      float n = 0.65 * dValue(g * 1.3) + 0.35 * dValue(g * 3.1 + 17.0) - 0.5;
      // Complementary for the other class: 1 - w exactly, noise and all.
      w = clamp(w + uMixNoise * n * 4.0 * w * (1.0 - w), 0.0, 1.0);
    }
    // Fixed hash of the index, flat enough that the threshold means what it says.
    uint h = sid * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    float r = float((h >> 22u) ^ h) / 4294967296.0;
    if (uSpots > 0.5) {
      float s = clamp(0.5 + (w - 0.5) * 4.0, 0.0, 1.0);
      float s3 = s * s * s, o3 = (1.0 - s) * (1.0 - s) * (1.0 - s);
      float prom = clamp((a.z - uSlotZ0[si] - uProm.x) / max(uProm.y - uProm.x, 1e-6), 0.0, 1.0)
                 * smoothstep(0.15, 0.5, a.w);
      w = pow(s3 / max(s3 + o3, 1e-6), 1.0 / (1.0 + 3.0 * prom));
    }
    if (r > w) {
      // Behind the camera: cheaper than a discard, keeps the fragment
      // shader branch-free.
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      return;
    }
  }
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
                   0.0, 0.0, 1.0)
           * mat3(turn.x, turn.y, 0.0, -turn.y, turn.x, 0.0, 0.0, 0.0, 1.0);

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
  if (role > 1.5) {
    // The harmonic extension of the ring's tone difference (toneAt).
    int k = si * 7;
    float r = length(a.xy);
    float q = clamp(ringNorm(a.xy, uRing.w) / max(uRing.x, 1e-6), 0.0, 1.0);
    vec2 e1 = r > 1e-6 ? a.xy / r : vec2(1.0, 0.0);
    vec2 e2 = vec2(e1.x * e1.x - e1.y * e1.y, 2.0 * e1.x * e1.y);
    vec2 e3 = vec2(e2.x * e1.x - e2.y * e1.y, e2.x * e1.y + e2.y * e1.x);
    vec3 off = uSlotTone[k]
             + q * (uSlotTone[k + 1] * e1.x + uSlotTone[k + 2] * e1.y)
             + q * q * (uSlotTone[k + 3] * e2.x + uSlotTone[k + 4] * e2.y)
             + q * q * q * (uSlotTone[k + 5] * e3.x + uSlotTone[k + 6] * e3.y);
    rgb = clamp(rgb + clamp(off, -0.12, 0.12), 0.0, 1.0);
  }

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
  if (uMacro > 0.0 && uTileSize > 0.0) rgb *= macroAt(anchorWorld);
  rgb = clamp(rgb * uExposure, 0.0, 1.0);
  float lum = dot(rgb, vec3(0.299, 0.587, 0.114));
  rgb = clamp(mix(vec3(lum), rgb, uSaturation), 0.0, 1.0);

  // Haze towards the horizon colour so far ground has no hard edge.
  float fog = 1.0 - exp(-uFogDensity * max(cam.z, 0.0));
  rgb = mix(rgb, uFogColour, fog);
  vColour = vec4(mix(rgb, uTint, uTintAmount), col.a * uFade);
}
`;

export const SPLAT_FRAG = `#version 300 es
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
export const FAR_VERT = `#version 300 es
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

export const FAR_FRAG = `#version 300 es
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
uniform float uMacro;
uniform float uBlend;          // 1: dissolve material borders as near tiles do
uniform float uSpots;          // 1: by spot, as the splats (uSpots there)
uniform float uMixWarp;
// Frames: per cell, its centre's atlas entry (r + 256 g) and turn (b, in
// 256ths of a turn), a > 0 if it has one; and the ring: radius, wobble.
uniform sampler2D uCentres;
uniform sampler2D uCentreTone;  // per cell: the centre's mean tint, rgb 0.5 + 2 x
uniform float uHasCentres;
uniform vec3 uRingFar;        // radius, wobble, shape
float ringNorm(vec2 l, float p) {      // as in the splat shader
  if (p <= 2.0) return length(l);
  float m = max(abs(l.x), abs(l.y));
  if (m <= 0.0) return 0.0;
  vec2 u = abs(l) / m;
  return m * pow(pow(u.x, p) + pow(u.y, p), 1.0 / p);
}
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
// Value noise for the macro variation, on the same kind of integer hash as
// the detail relief (a separate copy: TERRAIN_GLSL is vertex-only here).
float mHash(ivec2 c) {
  uint h = uint(c.x) * 0x27D4EB2Du ^ uint(c.y) * 0x165667B1u;
  h = (h ^ (h >> 15u)) * 0x85EBCA6Bu;
  h ^= h >> 13u;
  return float(h & 0xFFFFFFu) / 16777215.0;
}
float mValue(vec2 q) {
  vec2 i = floor(q), f = q - i;
  vec2 s = f * f * (3.0 - 2.0 * f);
  ivec2 c = ivec2(i);
  return mix(mix(mHash(c), mHash(c + ivec2(1, 0)), s.x),
             mix(mHash(c + ivec2(0, 1)), mHash(c + ivec2(1, 1)), s.x), s.y);
}
// A cell's tile material (bits 2-3 of b), -1 off the grid.
int matOf(ivec2 q) {
  if (q.x < 0 || q.y < 0 || q.x >= int(uGridN) || q.y >= int(uGridN)) return -1;
  return (int(floor(texelFetch(uCells, q, 0).b * 255.0 + 0.5)) >> 2) & 3;
}
// Every material's field at a half-cell node X (grid units), as spotField in
// frames.js: 0.5 plus half its lead over the strongest other, among the
// cells the node touches (one in a cell's middle, two on an edge, four at
// a corner).
vec4 spotNode(vec2 X) {
  bool ex = abs(X.x - floor(X.x + 0.5)) < 0.25, ey = abs(X.y - floor(X.y + 0.5)) < 0.25;
  int x0 = ex ? int(floor(X.x + 0.5)) - 1 : int(floor(X.x));
  int y0 = ey ? int(floor(X.y + 0.5)) - 1 : int(floor(X.y));
  vec4 c = vec4(0.0);
  float n = 0.0;
  for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
    if ((dx == 1 && !ex) || (dy == 1 && !ey)) continue;
    int k = matOf(ivec2(x0 + dx, y0 + dy));
    if (k >= 0) { c[k] += 1.0; n += 1.0; }
  }
  if (n == 0.0) return vec4(0.5);
  vec4 f;
  for (int k = 0; k < 4; k++) {
    float best = 0.0;
    for (int o = 0; o < 4; o++) if (o != k) best = max(best, c[o]);
    f[k] = 0.5 + 0.5 * (c[k] - best) / n;
  }
  return f;
}
void main() {
  vec2 l = vec2(uRot.x * vXY.x + uRot.y * vXY.y,
                -uRot.y * vXY.x + uRot.x * vXY.y);
  vec2 g = l / uTileSize + uGridN * 0.5;
  vec2 ij = floor(g);
  if (ij.x < 0.0 || ij.y < 0.0 || ij.x >= uGridN || ij.y >= uGridN) discard;
  float a = smoothstep(uFarStart - uFarBand, uFarStart, vDepth);
  if (a < doorway(gl_FragCoord.xy)) discard;
  vec2 fr = clamp(g - ij, 0.01, 0.99);   // east, north fraction in the cell
  vec4 t = texelFetch(uCells, ivec2(ij), 0);
  ivec2 from = ivec2(ij);         // the cell whose tile this pixel shows
  if (uBlend > 0.5 && uSpots > 0.5) {
    // By spot, as the splats: the fields read where the same noise moves
    // the point, and the pixel shows the material that leads there, from
    // its nearest cell. Opaque, so the leader alone and no dissolving strip.
    int mine = matOf(ivec2(ij));
    bool mixed = false;
    for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
      int k = matOf(ivec2(ij) + ivec2(dx, dy));
      if (k >= 0 && k != mine) mixed = true;
    }
    if (mixed) {
      float T = max(uTileSize, 1e-6);
      vec2 p = vXY;
      vec2 d = vec2(0.55 * mValue(p / (1.1 * T)) + 0.3 * mValue(p / (0.4 * T) + 5.0)
                      + 0.15 * mValue(p / (0.15 * T) + 9.0),
                    0.55 * mValue(p / (1.1 * T) + 41.0) + 0.3 * mValue(p / (0.4 * T) + 77.0)
                      + 0.15 * mValue(p / (0.15 * T) + 23.0))
             - 0.5;
      vec2 q = clamp(g + 2.0 * uMixWarp * d, ij - 1.0, ij + 2.0);
      vec2 h = q * 2.0, n0 = floor(h), f = h - n0;
      n0 *= 0.5;
      vec4 w = mix(mix(spotNode(n0), spotNode(n0 + vec2(0.5, 0.0)), f.x),
                   mix(spotNode(n0 + vec2(0.0, 0.5)), spotNode(n0 + vec2(0.5)), f.x), f.y);
      int win = mine;
      for (int k = 0; k < 4; k++) if (w[k] > w[win] + 1e-4) win = k;
      if (win != mine) {
        ivec2 c0 = ivec2(floor(q));
        float rMin = 1e9;
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
          ivec2 o = c0 + ivec2(dx, dy);
          float r = length(vec2(o) + 0.5 - q);
          if (r < rMin && matOf(o) == win) { rMin = r; from = o; }
        }
        if (rMin < 1e9) t = texelFetch(uCells, from, 0);
      }
    }
  } else if (uBlend > 0.5) {
    // The near tiles' border blend, here too: this cell's material weight at
    // each corner is its share of the four cells there, bilinear inside,
    // with value noise where it is undecided. Below one half the pixel is
    // drawn from the nearest neighbour of the other material. Without it
    // the far ground stepped by whole tiles at every border.
    ivec2 c0 = ivec2(ij);
    int mine = (int(floor(t.b * 255.0 + 0.5)) >> 2) & 3;
    float same[9];
    for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
      ivec2 q = c0 + ivec2(dx, dy);
      bool inside = q.x >= 0 && q.y >= 0 && q.x < int(uGridN) && q.y < int(uGridN);
      int k = inside ? (int(floor(texelFetch(uCells, q, 0).b * 255.0 + 0.5)) >> 2) & 3 : mine;
      same[(dy + 1) * 3 + dx + 1] = k == mine ? 1.0 : 0.0;
    }
    float sw = 0.25 * (same[0] + same[1] + same[3] + same[4]);
    float se = 0.25 * (same[1] + same[2] + same[4] + same[5]);
    float nw = 0.25 * (same[3] + same[4] + same[6] + same[7]);
    float ne = 0.25 * (same[4] + same[5] + same[7] + same[8]);
    float w = mix(mix(sw, se, fr.x), mix(nw, ne, fr.x), fr.y);
    if (w < 0.999) {
      float n = 0.65 * mValue(g * 1.3) + 0.35 * mValue(g * 3.1 + 17.0) - 0.5;
      if (w + n * 4.0 * w * (1.0 - w) < 0.5) {
        ivec2 sx = ivec2(fr.x < 0.5 ? -1 : 1, 0), sy = ivec2(0, fr.y < 0.5 ? -1 : 1);
        bool xFirst = min(fr.x, 1.0 - fr.x) < min(fr.y, 1.0 - fr.y);
        ivec2 tries[3] = ivec2[3](xFirst ? sx : sy, xFirst ? sy : sx, sx + sy);
        for (int i = 0; i < 3; i++) {
          ivec2 q = c0 + tries[i];
          if (q.x < 0 || q.y < 0 || q.x >= int(uGridN) || q.y >= int(uGridN)) continue;
          vec4 o = texelFetch(uCells, q, 0);
          if (((int(floor(o.b * 255.0 + 0.5)) >> 2) & 3) != mine) { t = o; from = q; break; }
        }
      }
    }
  }
  float idx = floor(t.r * 255.0 + 0.5) + 256.0 * floor(t.g * 255.0 + 0.5);
  // Derivatives from the continuous cell coordinate: uv jumps at cell
  // borders, which the GPU reads as extreme minification (a blurry line
  // along every border).
  vec2 gx = dFdx(g), gy = dFdy(g);
  vec3 tint = vec3(0.0);
  float cellIdx = idx;              // the cell's tile, before a centre takes over
  if (uHasCentres > 0.5) {
    // Inside the ring the cell's centre, turned as the splats turn it.
    vec4 cc = texelFetch(uCentres, from, 0);
    vec2 l = (fr - 0.5) * uTileSize;
    float r = ringNorm(l, uRingFar.z);
    if (cc.a > 0.5 && r < uRingFar.x + uRingFar.y) {
      vec2 dir = l / max(length(l), 1e-6);
      vec2 mid = (ij + 0.5 - uGridN * 0.5) * uTileSize;
      vec2 ring = mid + uRingFar.x * dir;
      vec2 w = vec2(uRot.x * ring.x - uRot.y * ring.y, uRot.y * ring.x + uRot.x * ring.y);
      float n = 0.65 * mValue(w / (0.23 * uTileSize))
              + 0.35 * mValue(w / (0.09 * uTileSize) + 17.0) - 0.5;
      if (r < uRingFar.x + 2.0 * uRingFar.y * n) {
        float a = floor(cc.b * 255.0 + 0.5) * 6.28318530718 / 256.0;
        vec2 cs = vec2(cos(a), sin(a));
        vec2 p = vec2(cs.x * l.x + cs.y * l.y, -cs.y * l.x + cs.x * l.y);
        fr = clamp(p / uTileSize + 0.5, 0.01, 0.99);
        gx = vec2(cs.x * gx.x + cs.y * gx.y, -cs.y * gx.x + cs.x * gx.y);
        gy = vec2(cs.x * gy.x + cs.y * gy.y, -cs.y * gy.x + cs.x * gy.y);
        idx = floor(cc.r * 255.0 + 0.5) + 256.0 * floor(cc.g * 255.0 + 0.5);
        tint = (texelFetch(uCentreTone, from, 0).rgb - 0.5) * 0.5;
      }
    }
  }
  float col = mod(idx, uAtlasGrid.x);
  float row = floor(idx / uAtlasGrid.x);
  // Atlas tiles are rendered north up.
  vec2 uv = vec2((col + fr.x) / uAtlasGrid.x,
                 (row + 1.0 - fr.y) / uAtlasGrid.y);
  vec2 k = vec2(1.0 / uAtlasGrid.x, -1.0 / uAtlasGrid.y);
  vec4 tex = textureGrad(uAtlas, uv, gx * k, gy * k);
  vec3 rgb = tex.rgb;
  // b holds the rule class in its low two bits.
  int packed = int(floor(t.b * 255.0 + 0.5));
  if (uHasTileInfo > 0.5) {
    int ti = int(idx);
    rgb *= texelFetch(uTileInfo, ivec2(ti, 0), 0).rgb * 2.0;
    // Fade part-way to the tile mean with distance, as terrain renderers do.
    // Fading fully to the material mean made distant
    // ground a flat third look next to the splats and the near atlas.
    // The cell's tile's, centre or not: a centre fading to its own mean
    // showed as a dark square in every bush cell far off.
    vec3 mean = texelFetch(uTileInfo, ivec2(int(cellIdx), 1), 0).rgb;
    float far = smoothstep(uFarStart * 1.5, uFarStart * 5.0, vDepth);
    rgb = mix(rgb, mean, 0.75 * far);
  }
  rgb = clamp(rgb + clamp(tint, -0.12, 0.12), 0.0, 1.0);    // a centre's, as near
  // Same macro variation as the splats (macroAt there).
  if (uMacro > 0.0) {
    vec2 w = vXY / uTileSize;
    float m = 0.6 * mValue(w / 11.0 + 3.7) + 0.4 * mValue(w / 4.3 + 17.1);
    rgb *= 1.0 + uMacro * 0.36 * (m - 0.5);
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
    int cls = packed & 3;
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
export const LINE_VERT = `#version 300 es
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

export const LINE_FRAG = `#version 300 es
precision highp float;
uniform float uAlpha;
in vec3 vRGB;
out vec4 oColour;
void main() { oColour = vec4(vRGB * uAlpha, uAlpha); }
`;

// Gradient sky: ground against a horizon reads with depth, against black it doesn't.
export const SKY_VERT = `#version 300 es
precision highp float;
in vec2 aNDC;
out vec2 vNDC;
void main() { vNDC = aNDC; gl_Position = vec4(aNDC, 0.0, 1.0); }
`;

export const SKY_FRAG = `#version 300 es
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
  // Below the horizon it stays the haze colour for a while: darkening right
  // under the horizon drew a grey band wherever the ground ended.
  vec3 c = t > 0.0
    ? mix(uSkyHorizon, uSkyTop, pow(clamp(t, 0.0, 1.0), 0.55))
    : mix(uSkyHorizon, uGround, smoothstep(0.08, 0.7, -t));
  oColour = vec4(c, 1.0);
}
`;
