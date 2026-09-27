// WebGL2 Gaussian splat renderer with Wang tiling.
//
// The projection follows bozkir/camera.py deliberately: camera space has
// +z forward, x right, y down, and the 2D covariance comes from the
// Jacobian of the perspective divide (3DGS Eq. 5). Keeping the two
// implementations comparable means the slow Python renderer can serve as
// ground truth when this one looks wrong.

import { drawOrder } from './order.js';
import { mergeGroups, MAX_GROUP } from './merge.js';
import { openDrop, describe } from './tileset.js';
import { Capture } from './capture.js';
import { generatedField, listCached, loadHeightField, openness, roughnessMask } from './heightfield.js';
import { Benchmark, report } from './benchmark.js';
import { classify, sampleGrid } from './landform.js';
import { GROUPS as TERRAIN_GROUPS, PROFILES as TERRAIN_PROFILES } from './terrain.js';
import { starterTileset } from './starter.js';

const BUILD = 'bozkir viewer 4.1 (16-bit heights, sediment)';
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
// Tile centres on the ground plane, one per slot in a merged group.
// An unmerged draw uses slot 0 and ignores the rest, so both paths read
// the same way and the shader has no branch for it.
uniform vec2 uCellXY[8];

// Per-splat class blending. Hybrid GSWT's Equation 1 selects a class at a
// position using a priority that varies *within* the exemplar - per
// Gaussian, not per tile. That is why their transitions look organic and a
// per-tile choice cannot: a tile boundary is a straight line, so a class
// boundary made of tile choices is a staircase however good the rule is.
//
// Here the same effect is had by drawing both classes in a cell near the
// boundary and dissolving between them. Each splat gets a fixed pseudo
// random priority from its own index, and survives only if that priority
// falls on its class's side of the local mix. The dissolve is stable
// because the priority is a function of the index and nothing else - a
// splat does not flicker between frames, and two instances of the same
// tile at different mixes keep different splats, which breaks up the
// identical-twin look of repeated tiles as a side effect.
//
// uMix < 0 disables it: one class, every splat drawn, the old behaviour.
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
// Splat index, from the sorted order buffer. The top 3 bits hold the slot
// of the cell this splat belongs to; a merged group interleaves splats from
// several cells into one stream, so the index alone no longer says where
// the splat sits. 29 bits is 536 million splats, which is not the limit
// that will bite first.
in uint aIndex;

out vec2 vCorner;
out vec4 vColour;
out float vFar;                // 0 before the far field, 1 past its start
uniform float uFarOn;
uniform float uFarStart;
uniform float uFarBand;

// The height field, duplicated from height() in the JavaScript. The two
// must agree exactly: the overlay lines and the material rule are computed
// from the JS copy and the geometry from this one, so any divergence means
// material placed by one surface and drawn on another.
//
// That is not hypothetical. This function used to have only the analytic
// branch, so a loaded height field reached the rule but never the GPU: the
// classes were chosen from real drainage and then drawn on sine waves.
// Everything looked periodic because it was.
//
// Sampled with texelFetch and interpolated by hand rather than with a
// LINEAR sampler, because a single-channel float texture is not filterable
// everywhere, and because doing it explicitly is the only way to be sure
// it matches the bilinear in heightfield.js rather than merely resembling
// it.
uniform sampler2D uField;
uniform vec2 uFieldSize;      // texels
uniform float uFieldExtent;   // world units across the long edge
uniform float uHasField;
uniform sampler2D uOpen;      // sky openness, built from the height field
uniform float uHasOpen;
uniform float uShade;         // how much of it to apply
uniform float uExposure;
uniform float uSaturation;

// Fold a coordinate back into 0..n by reflection, as HeightField does.
// Mirroring rather than wrapping: the far edge meets itself, so a field
// smaller than the grid repeats with no cliff along the join.
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

// How much sky a point can see, 0 deep in a hollow and 1 out in the open.
// Read from a texture rather than worked out here: it needs the ground
// around the point, which would be another sixteen texel fetches per
// splat, and it does not change while the camera moves.
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

// Detail relief: creases at the scale of a tile, added to the terrain so
// close-up ground is not perfectly smooth. The height field has two or so
// samples per tile on a big world and erosion rounds every crest, so
// between samples everything was a curve. Ridged value noise on an integer
// hash - the same hash, bit for bit, as detailAt() in viewer.js, so the
// camera, the cells, the splats and the far field all agree on the ground.
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
    // Centred on zero so raising relief lifts and lowers about the middle
    // rather than pushing the whole terrain upwards.
    return uRelief * (fieldAt(p) - 0.5) * 2.0 + detailAt(p);
  }
  float f = 1.0 / max(uWave, 0.01);
  return uRelief * (sin(f * p.x) * cos(f * p.y)
    + 0.5 * sin(2.3 * f * p.x + 1.7) * cos(1.9 * f * p.y + 0.4)) + detailAt(p);
}

// The surface gradient at a point, by central differences.
vec2 terrainGrad(vec2 p) {
  if (uRelief <= 0.0) return vec2(0.0);
  // The step has to suit whichever surface is in use. For a sampled field
  // the meaningful scale is one texel of ground, not the analytic
  // wavelength: too large a step smooths real features away, too small and
  // the difference is quantisation noise.
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
  uint slot = aIndex >> 29u;
  uint sid = aIndex & 0x1FFFFFFFu;
  vec2 cellXY = uCellXY[int(slot)];

  if (uMix >= 0.0) {
    // A hash of the index: cheap, fixed per splat, and flat enough across
    // the range that the dissolve threshold means what it says.
    uint h = sid * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    float r = float((h >> 22u) ^ h) / 4294967296.0;
    if (r > uMix) {
      // Collapsed to nothing and placed behind the camera, which costs
      // less than a discard in the fragment stage and keeps the fragment
      // shader branch-free.
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      return;
    }
  }

  vec4 a = fetch(sid, 0);     // position.xyz, opacity
  vec4 b = fetch(sid, 1);     // scale.xyz
  vec4 q = fetch(sid, 2);     // rotation w,x,y,z

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
  // Turn the tile with the grid, so a tile still lines up with its
  // neighbours after the whole layout is rotated off the world axes.
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
  // Sigma' = W Sigma W^T with W the tangent frame, which folds into the
  // same product: (W R S)(W R S)^T.
  mat3 M = terrainJacobian(grad) * spin * R
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
  int ci = int(sid);
  vec4 col = texelFetch(uColour, ivec2(ci & 2047, ci >> 11), 0);
  vec3 rgb = col.rgb;

  if (uEdgeMark > 0.0 && uTileSize > 0.0) {
    // Paint each edge band with its own colour and wash the interior out,
    // the way the GSWT figures do. The constraint then reads off the
    // geometry itself: two tiles meet correctly when the band running
    // along their shared boundary is one colour on both sides.
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
    // Tiles carry Gaussians past their own square, so d goes negative out
    // there. Without this those all take full edge colour, the band looks
    // several times its width, and two neighbours paint the same strip in
    // different colours.
    float w = d < 0.0 ? 0.0 : 1.0 - smoothstep(0.0, uEdgeMark, d);
    float grey = dot(rgb, vec3(0.299, 0.587, 0.114));
    rgb = mix(vec3(0.55 + 0.45 * grey), ec, w);
  }
  // Ground shading. The capture was lit where it was taken and cannot be
  // relit, so this adds nothing directional - no second sun, no shadows
  // that would disagree with the ones baked into the splats. It only
  // takes light away where the terrain itself would: a hollow sees less
  // sky than open ground, so it sits darker. That is the part of lighting
  // that depends on the shape the tiles were laid on, and the part a flat
  // exemplar cannot know about.
  if (uHasOpen > 0.5 && uShade > 0.0) {
    rgb *= mix(1.0, 0.45 + 0.55 * openAt(anchorWorld), uShade);
  }
  // Exposure and saturation last, because .splat keeps only the constant
  // term of each Gaussian's colour: view-dependent response is gone, and
  // what is left reads flat and slightly washed next to the capture.
  // These put that back by eye rather than pretending to recover it.
  rgb = clamp(rgb * uExposure, 0.0, 1.0);
  float lum = dot(rgb, vec3(0.299, 0.587, 0.114));
  rgb = clamp(mix(vec3(lum), rgb, uSaturation), 0.0, 1.0);

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
in float vFar;
out vec4 oColour;
// Screen-door threshold: a fixed per-pixel value in 0..1. The splats and
// the far field compare their hand-over weight against the same value, so
// each pixel in the band shows one or the other, never a mix of both - the
// dithered cross-fade games use to swap levels without a moving edge.
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
//
// Past a few dozen tiles a splat is smaller than a pixel, and what the
// viewer was paying for there - a sort and a draw call per cell - bought
// nothing a texture could not show. So beyond uFarStart the ground is a
// height-field mesh, displaced by the same terrainHeight the splats use,
// and coloured per cell from an atlas of every tile seen from above
// (scripts/bake_atlas.py), chosen by the same Wang layout and class rule.
// It is the standard way games draw distant terrain; hierarchical splat
// LOD (Kerbl et al. 2024) answers too many splats, and the ceiling here is
// draw calls.
//
// Unlike the splats this pass writes depth, so near hills hide far ground
// correctly, and it uses a real perspective w so the atlas does not swim.
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
// Detail relief: creases at the scale of a tile, added to the terrain so
// close-up ground is not perfectly smooth. The height field has two or so
// samples per tile on a big world and erosion rounds every crest, so
// between samples everything was a curve. Ridged value noise on an integer
// hash - the same hash, bit for bit, as detailAt() in viewer.js, so the
// camera, the cells, the splats and the far field all agree on the ground.
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
  if (uHasField > 0.5) return uRelief * (fieldAt(p) - 0.5) * 2.0 + detailAt(p);
  float f = 1.0 / max(uWave, 0.01);
  return uRelief * (sin(f * p.x) * cos(f * p.y)
    + 0.5 * sin(2.3 * f * p.x + 1.7) * cos(1.9 * f * p.y + 0.4)) + detailAt(p);
}
void main() {
  vec3 world = vec3(aXY, terrainHeight(aXY));
  vec3 cam = uView * (world - uEye);
  vXY = aXY;
  vDepth = cam.z;
  // Openness, read from the same texture and the same way as fieldAt, so
  // distant ground is shaded like the splats nearer in.
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
// Screen-door threshold: a fixed per-pixel value in 0..1. The splats and
// the far field compare their hand-over weight against the same value, so
// each pixel in the band shows one or the other, never a mix of both - the
// dithered cross-fade games use to swap levels without a moving edge.
float doorway(vec2 p) {
  return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}
// Same palette as tileRGB() in viewer.js, so a tile keeps its colour when
// it crosses from splats into the far field.
vec3 tileRGB(float i) {
  float h = mod(i * 137.508, 360.0) / 360.0;
  vec3 n = vec3(5.0, 3.0, 1.0);
  vec3 v = mod(n + h * 6.0, 6.0);
  vec3 k = clamp(min(min(v, 4.0 - v), vec3(1.0)), 0.0, 1.0);
  return 0.30 + 0.70 * k;
}
void main() {
  // Fades in across the band, so the splats handing over and the atlas
  // taking over overlap instead of meeting at a seam.
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
  // Atlas tiles are rendered north up: north is the top of each square.
  vec2 uv = vec2((col + fr.x) / uAtlasGrid.x,
                 (row + 1.0 - fr.y) / uAtlasGrid.y);
  // Derivatives from the continuous cell coordinate, not from uv: at a cell
  // border uv jumps to another tile, the GPU reads the jump as extreme
  // minification and draws its blurriest level there - a line along every
  // border. The continuous coordinate has no jump.
  vec2 gx = dFdx(g), gy = dFdy(g);
  vec2 k = vec2(1.0 / uAtlasGrid.x, -1.0 / uAtlasGrid.y);
  vec4 tex = textureGrad(uAtlas, uv, gx * k, gy * k);
  vec3 rgb = tex.rgb;
  if (uHasTileInfo > 0.5) {
    int ti = int(idx);
    rgb *= texelFetch(uTileInfo, ivec2(ti, 0), 0).rgb * 2.0;
    // Detail fades to each tile's mean with distance, as terrain renderers
    // do: past a few hand-over distances a tile's own pattern is just the
    // same few shapes repeating, which the eye reads as a lattice.
    vec3 mean = texelFetch(uTileInfo, ivec2(ti, 1), 0).rgb;
    float far = smoothstep(uFarStart * 1.5, uFarStart * 5.0, vDepth);
    rgb = mix(rgb, mean, 0.75 * far);
  }
  // Opaque. Drawing with the atlas coverage was tried: behind the far mesh
  // there is only sky, so a fifth or more of the sky washed into all
  // distant ground. Gaps between near splats mostly show more ground, not
  // sky; distance tint belongs to the haze below, and the colour step at
  // the hand-over to the per-tile gain above.
  rgb *= mix(1.0, 0.45 + 0.55 * vOpen, uShade);
  rgb = clamp(rgb * uExposure, 0.0, 1.0);
  float lum = dot(rgb, vec3(0.299, 0.587, 0.114));
  rgb = clamp(mix(vec3(lum), rgb, uSaturation), 0.0, 1.0);
  // Debug views reach the far field too; otherwise they describe only the
  // nearest few dozen tiles and say nothing about the distance.
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

/** Attach a handler, or say so and carry on.
 *
 *  The panel and the script drift: a control gets added on one side only,
 *  and every listener after the missing one never runs - so a single stale
 *  index.html takes the whole viewer down at load with nothing on screen.
 *  A missing control should cost that control, not the renderer.
 */
// Every slider draws its own filled portion, from a CSS variable this sets
// on input. Done generically rather than per control: there are a dozen
// sliders and each already has a handler doing something else, so folding
// the paint into those would mean a dozen chances to forget one.
// --- panel size ---------------------------------------------------------
//
// Width by dragging the panel's left edge, text size by the two buttons in
// its header. Both write a CSS variable and nothing else: row height is
// derived from the text size in the stylesheet, so the sliders whose labels
// are written inside them grow to fit rather than clipping.
//
// resize() already runs every frame, so the canvas follows the new width by
// itself and there is nothing to notify.
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
      // Capture on the grip, so a fast drag that outruns the pointer keeps
      // sending events here instead of to whatever it passed over.
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
    // Back to the width it started at, for when a drag goes wrong.
    grip.addEventListener('dblclick', () => setPanelWidth(232));
  }
}

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

// Ask for the fast GPU. Laptops with a discrete card beside an integrated
// one often hand a browser the integrated chip unless asked - the splats
// are still drawn on a GPU, just the slow one, and it feels like CPU.
const gl = canvas.getContext('webgl2',
  { antialias: false, alpha: false, premultipliedAlpha: false,
    powerPreference: 'high-performance' });
if (!gl) {
  overlay.querySelector('.msg').innerHTML =
    '<b>No WebGL2</b>This browser cannot run the renderer.';
  throw new Error('webgl2 unavailable');
}

// Which GPU is drawing. Printed to the console and shown in the scene
// section, because "is it using my graphics card?" should never need a guess.
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
    // An array uniform is addressed by its first element. Asking for the
    // bare name works on most drivers and returns null on some, which
    // shows up as a tile stuck at the origin rather than as an error.
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
 *  Orbit matches orbit_camera() in bozkir/camera.py: z is up, elevation 0
 *  looks along the ground, 90 straight down, and the eye sits on a sphere
 *  around a target. Good for inspecting one thing from all sides.
 *
 *  Fly holds the eye and turns the view instead, so you can go anywhere -
 *  under the terrain, inside it, out to the horizon. Both modes share the
 *  same azimuth and elevation, so switching between them keeps the view.
 *
 *  Elevation still stops just short of straight up or down. That is not a
 *  restriction on where you can go: at exactly 90 the forward direction and
 *  the world up are parallel, there is no way to say which way is right,
 *  and the picture spins. Every camera of this shape has the same limit. */
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

// One VAO per program. Without them, attribute enables and divisors leak
// between draws: the splat pass sets divisor 1 on attribute 1, and the
// line pass then reads its colours from a single vertex, or nothing at all.
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
// The class each tile belongs to, and how many there are. With one class
// this is all zeros and every decision below collapses to the old
// behaviour.
let tileClass = null;
let classCount = 1;
// Off by default so a scene loads looking the way it always did, and the
// rule is something you turn on and see happen.
let classOn = true;
let classSharp = 2.0;
// How much of the grid the first class takes. The rule's own arithmetic
// gave whatever it gave - 27% in every scene tried, 6% on a synthetic
// valley - which is no use to somebody who wants a third of their terrain
// to be scrub. Balance decides how much; the rule still decides where.
let classBalance = 0.5;
// How large a patch of one material should be, in tiles. Without it,
// neighbouring cells either side of the threshold make opposite choices
// over a difference too small to mean anything, and the result is a
// scatter of single tiles. A patch of scrub is a patch.
let classCoherence = 2;
// How much altitude counts against shape. Every terrain shader places
// material by height first - snow above a line, grass below one - and it
// was the input missing here: a hollow at the top of a mountain read the
// same as a hollow at the bottom.
let classAltitude = 0.4;
// Off by default. It doubles the draw calls for cells near a class
// boundary, and a scene should look the way it always did until somebody
// asks for something else.
let classBlend = false;
let blendWidth = 0.35;
let cellMix = null;
let cellAlt = null;        // the class each cell would take instead
let blendedCells = 0;
let classStats = null;
// What each class of the loaded tileset looks like on average, as linear
// 0..1 RGB. The terrain window paints its previews with these, so a preview
// shows this capture's materials rather than a stock palette.
let classColours = [[0.62, 0.55, 0.45]];
let tileSize = 0;
let gridN = 1;
let cells = [];
let seed = 1;
let usedPatches = 0;
let sortedReady = false, sortPending = false, lastSortKey = '', sortMs = 0;
let sortingEnabled = true, gain = 1;

// Cached per-direction orders, the scheme GSWT uses. `cacheDirs` holds the
// directions, `cacheBuf` every order back to back, `cacheStride` the number
// of splats in one order.
let sortMode = 'live';        // 'live' | 'cached'
let cacheViews = 9;
let cacheDirs = null, cacheBuf = null, cacheStride = 0, cacheMs = 0;
let cacheReady = false;
let showEdges = false, showDiagonals = false, showTints = false;
let relief = 0, reliefScale = 6, edgeBand = 0.12, subdiv = 0;
// Set once somebody drags the scale slider, after which the grid stops
// adjusting it for them. Guessing on their behalf is helpful until they
// have said what they want, and rude afterwards.
let reliefScaleTouched = false;
// The height field on the GPU. Single channel float, fetched by texel and
// interpolated in the shader, so what the geometry is displaced by is the
// same arithmetic the rule read on the CPU.
let fieldTex = null;
let openTex = null;
let shotWanted = false;
// Far field (see FAR_VERT): an atlas of tiles seen from above, a texture of
// which tile each cell holds, and one mesh for all distant ground.
let farOn = true;
let farStartTiles = 14;          // splats out to here, atlas beyond
let atlas = null;                // { tex, cols, rows }
let cellTex = null;
let tileInfoTex = null;   // per tile: colour gain, mean colour
let farVAO = null, farCount = 0;
// A scene that is not a Wang tileset - garden, bicycle - is shown as the
// capture itself: warping it onto terrain or colouring it by a terrain rule
// only bends and turns something that was never ground.
let plainScene = false;
let tileMeans = null;   // mean splat colour per tile, linear 0..1

// Automatic quality, the way games keep a frame budget.
//
// First lever: render resolution. Splats are costly per pixel - many
// blended layers each - so drawing fewer pixels and letting the browser
// scale the canvas up is the fastest relief there is, and it can change
// every frame. Range and budget follow Unreal's dynamic resolution
// defaults: 50-100% of full size, 33.3 ms. Second lever, only when
// resolution alone cannot keep up: the level-of-detail distance, which also
// brings the far field closer. The person's slider is its ceiling.
//
// Control is proportional (Intel's dynamic-resolution sample): each
// correction is sized by how far the frame time is from the budget, so a
// big miss is fixed at once, not a notch at a time. A panic drop, as in
// Unreal, reacts to a few consecutive frames far over budget. Recovery is
// deliberately slower than the drop, so the picture does not bounce.
let autoQuality = true;
let lodBaseUser = 8;            // what the slider says
let renderScale = 1;            // fraction of full resolution
let frameMs = 16;               // smoothed frame time
let overRun = 0, qualityTick = 0, lastNote = '';
const BUDGET_MS = 33.3;
const SCALE_MIN = 0.5;
const QUALITY_MIN_BASE = 3;

function qualityNote(text) {
  // At most one note every 1.5 s, so a correction is announced, not chanted.
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

// Two levers on two time scales. Resolution is smooth and cheap to move,
// so it answers within a quarter second. The detail distance is not: a
// step moves cells between levels and re-sorts them, which is itself a
// spike, and moving it back at the first quiet moment made the two chase
// each other (20 and 45 fps in turn). So detail moves only on sustained
// need, restores only with a wide margin, and everything holds still while
// a detail change settles.
let lodHold = 0, slowFor = 0, quickFor = 0;

function tuneQuality(dt) {
  if (!autoQuality || !splatCount || !(dt > 0) || dt > 500) return;
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

  // Pixel cost goes with the square of the resolution, so the frame time
  // predicts the scale that would land at 90% of the budget.
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
 *  `relief` is set for a terrain spanning about 24 tiles. Relief scale
 *  follows the grid so the height field covers it without repeating, and
 *  without this a 256-tile world spread the same 1.2 of height over ten
 *  times the width: hills stretched flat. Growing the height with the
 *  width keeps landforms in proportion, so a bigger world has bigger hills
 *  rather than none. Below the reference scale nothing changes. */
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

// Turning the whole layout off the world axes. Worth having, but it does
// not remove the alignment effect - it moves it. A grid turned 22 degrees
// lines up with the screen at 22, 112, 202 and 292 instead of at the world
// axes. Default is zero so the behaviour is at least predictable.
let gridAngle = 0;
const FLAT = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

/** Ground height at a point. Two octaves is enough to bend tiles without
 *  turning the terrain into noise. */
// A measured height field, when the tileset came with one. Null otherwise,
// and then the analytic surface below is used.
let field = null;

// Detail relief, the JavaScript twin of detailAt() in the shaders: the
// same integer hash (Math.imul is GLSL's uint multiply), so every part of
// the viewer that asks where the ground is gets the same answer.
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

function height(x, y) {
  const relief = amplitude();
  if (relief <= 0) return 0;
  if (field) {
    // Centred on zero so raising relief lifts and lowers about the middle
    // rather than pushing the whole terrain upwards.
    //
    // `reliefScale` means the same thing here as for the analytic surface
    // below: how many tiles a landform spans. A sixteen-metre survey has no
    // hills in it, so laying it across the whole grid gives one gentle
    // swell and a control that does nothing. Letting it set the field's
    // extent instead makes the same terrain readable at any size, and the
    // field mirrors beyond its edge so there is no seam where it repeats.
    return relief * (field.sample(x, y) - 0.5) * 2.0 + detailAt(x, y);
  }
  // Wavelength is measured in tiles, so the terrain keeps the same shape
  // relative to the tiling whatever the tile size happens to be. In world
  // units a fixed number puts a 3x3 grid inside a single hill, which reads
  // as a plane tilt rather than terrain.
  const f = 1 / Math.max(reliefScale * (tileSize || 1), 0.01);
  return relief * (Math.sin(f * x) * Math.cos(f * y)
    + 0.5 * Math.sin(2.3 * f * x + 1.7) * Math.cos(1.9 * f * y + 0.4))
    + detailAt(x, y);
}

/** The tangent frame at a point: two surface tangents and the normal.
 *  Returned column-major, which is what uniformMatrix3fv expects.
 *  Not `frame` - that name is the render loop. */
function tangentFrame(x, y) {
  if (amplitude() <= 0) return FLAT;
  // The step has to suit whichever surface is being differenced. For a
  // sampled field the meaningful scale is one texel of ground: a step
  // sized for the analytic wavelength is either far larger than a texel,
  // which smooths real features out of the normal, or far smaller, which
  // reads quantisation noise. Matches the shader's terrainGrad.
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
// One colour per class, in the order the tileset numbers them: the class
// that collects material first, then exposed, then any further ones.
const CLASS_RGB = [[0.30, 0.62, 0.35], [0.88, 0.72, 0.30],
                   [0.36, 0.54, 0.86], [0.80, 0.42, 0.62]];

// Three skies rather than one slider: choosing a mood is easier than
// choosing six numbers, and the fog has to match the horizon or the
// terrain ends against a colour the sky never has.
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

// Green through red as detail drops, the convention every engine's LOD
// debug view uses.
const LOD_RGB = [[0.30, 0.85, 0.35], [0.95, 0.85, 0.25],
                 [0.98, 0.58, 0.20], [0.92, 0.30, 0.30],
                 [0.75, 0.35, 0.85], [0.40, 0.60, 0.95]];

/** A fixed hue per tile index, spread by the golden angle so neighbouring
 *  indices look different. Which tile a cell uses is decided once, when the
 *  grid is built, and nothing about the camera can change it - so if a cell
 *  changes colour while you turn, cells really are swapping tiles and
 *  something is wrong. If the colours hold, they are not. */
function tileRGB(i) {
  const h = ((i * 137.508) % 360) / 360;
  const k = (n) => {
    const v = (n + h * 6) % 6;
    return Math.max(0, Math.min(1, Math.min(v, 4 - v, 1)));
  };
  return [0.30 + 0.70 * k(5), 0.30 + 0.70 * k(3), 0.30 + 0.70 * k(1)];
}

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

/** Lay out gridN x gridN cells.
 *
 *  With a Wang tile set, each cell's west colour is fixed by the cell to
 *  its left and its south colour by the cell below; north and east stay
 *  free. A complete set always has a tile that fits, so this never
 *  backtracks, and the free choices are what stop the terrain repeating. */
/** A tile of the same code but a different class, for blending against.
 *
 *  The codes are what make this possible: every class carries the same
 *  set, so the alternative to any tile is the tile with its code in
 *  another class, and swapping between them cannot break the matching.
 */
/** The share of the grid each class gets. The slider names the first
 *  class's share directly, as it does with two; the rest divide what is
 *  left evenly, because a slider per class is a panel nobody reads and
 *  the interesting control is how much of the ground the main material
 *  covers. */
function classShares(count) {
  const first = Math.min(0.95, Math.max(0.05, classBalance));
  const rest = (1 - first) / Math.max(1, count - 1);
  return Array.from({ length: count }, (_, k) => (k === 0 ? first : rest));
}

function alternateTile(c, p) {
  if (!wangCodes || !tileClass) return -1;
  const code = wangCodes[c.patch];
  // Blend towards the class the terrain ranked next, not towards whichever
  // other class happens to come first in the tileset. With two classes
  // those are the same; with three they are not, and picking the wrong one
  // dissolves a cell into a material the rule never considered.
  for (let k = 0; k < wangCodes.length; k++) {
    if (c.alt != null && tileClass[k] !== c.alt) continue;
    if (tileClass[k] === tileClass[c.patch]) continue;
    const o = wangCodes[k];
    if (o[0] === code[0] && o[1] === code[1]
        && o[2] === code[2] && o[3] === code[3]) return k;
  }
  return -1;
}

function buildGrid() {
  cells = [];
  if (!tileSize) { cells = [{ x: 0, y: 0, z: 0, warp: FLAT, patch: 0 }]; return; }
  usedPatches = Math.max(1, Math.min(usedPatches || patches.length,
                                     patches.length));
  let s = seed;
  const rand = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const half = (gridN - 1) / 2;
  const chosen = new Int32Array(gridN * gridN).fill(-1);

  // Which material each cell should carry, from the shape of the ground
  // under it. Computed once here rather than per frame: the decision
  // changes when the terrain or the layout does, and neither changes while
  // the camera moves.
  //
  // Every class was built over the same edge codes, so the code and the
  // class are independent: the matching constraint picks the code from the
  // neighbours, and the terrain picks the class, and neither can overrule
  // the other. That separation is what makes this work at all.
  let wantClass = null;
  cellMix = null;
  cellAlt = null;
  classStats = null;
  if (classOn && classCount > 1 && tileSize) {
    // Sampled finer than one point per tile. Every map below is a
    // derivative of the surface, and a derivative taken from one sample
    // per cell only means anything when the landforms are several cells
    // across. Squeeze the field into a few tiles and one sample per cell
    // is at or past the Nyquist limit - what comes back is not the terrain
    // but an alias of it, which is why the classes looked scattered rather
    // than placed. Drainage suffers most: a channel narrower than a cell
    // cannot be traced at all.
    // 4 samples per tile up to 64 tiles; fewer beyond, or a kilometre
    // grid would ask the rule for tens of millions of samples.
    const over = gridN <= 64 ? 4 : gridN <= 160 ? 2 : 1;
    const z = sampleGrid(gridN, tileSize, height, gridAngle, over);
    // The erosion's own record of where material settled, sampled on the
    // same grid as the height so the two line up cell for cell. Only
    // generated terrain has one.
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
    // How strongly each cell chose. Near 1/classCount the rule is nearly
    // undecided, which is exactly where a hard choice shows as a staircase
    // and where drawing both classes is worth the second draw call.
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
        // Of the tiles that fit, prefer the ones in the class the terrain
        // asked for. Falling back to the whole set when that class has
        // none is a safety net that should never fire - every class
        // carries every code - and firing silently would look like the
        // rule not working, so it is counted.
        let pool = fits;
        if (wantClass && tileClass) {
          const want = wantClass[j * gridN + i];
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

/** How badly out of order the splats of one cell are, for the current view.
 *
 *  Walks the order actually being used and counts adjacent pairs whose true
 *  depths run the wrong way, plus the worst depth by which a pair is
 *  inverted. Pair counts are over-sensitive - with thousands of splats,
 *  neighbouring depths differ by almost nothing and any perturbation flips
 *  many of them - so the depth number is the one to read. It is in world
 *  units, and worth comparing against the splat width alongside it.
 */
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
    // Where this splat actually ends up, and how deep that is.
    const x = positionsRef[3 * s], y = positionsRef[3 * s + 1],
          z = positionsRef[3 * s + 2];
    const wx = cell.x + W[0] * x + W[3] * y + W[6] * z;
    const wy = cell.y + W[1] * x + W[4] * y + W[7] * z;
    const wz = cell.z + W[2] * x + W[5] * y + W[8] * z;
    const d = (wx - b.eye[0]) * b.forward[0]
            + (wy - b.eye[1]) * b.forward[1]
            + (wz - b.eye[2]) * b.forward[2];
    if (prev !== null) {
      // The order runs far to near, so depth should fall as it is walked.
      // A pair where it rises is a splat drawn in front of something that
      // is actually nearer.
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

/** Load <scene>.atlas.png/.json if the scene has one. Without it the far
 *  field stays off and every cell is splats, as before. */
async function loadAtlas(name) {
  atlas = null;
  try {
    const r = await fetch(`./data/${name}.atlas.json`);
    if (!r.ok) return;
    const lay = await r.json();
    const img = new Image();
    img.src = `./data/${name}.atlas.png`;
    await img.decode();
    // Per-tile correction. The atlas is painted from straight above, where
    // the bright tops of things dominate; up close the same tile is seen at
    // a grazing angle, splats and the ground between them. Scaling each
    // atlas tile so its mean matches the mean of its own splats removes
    // most of the step in colour at the hand-over. Row 0: gain (0..2 as
    // 0..1). Row 1: the tile's mean colour, which distant ground fades to.
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
    // Stop the mip chain while each tile is still several texels wide: the
    // coarser levels would average a tile with its unrelated neighbours in
    // the atlas.
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

  // A regular mesh over the whole grid, about two vertices per tile: the
  // height field has no detail finer than that at the distances it is used.
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

/** Rebuild after a control moves, with a note on screen first.
 *
 *  The rebuild runs on the main thread and freezes the page while it does
 *  - a second or more at hundreds of tiles - so the note is shown, the
 *  browser is given two frames to paint it, and only then does the work
 *  start. Moves that arrive meanwhile are folded into the one rebuild. */
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
    // And each tile's own, for matching the far field to it.
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
  // Every level is sorted separately: it holds a different set of splats,
  // so it needs its own order.
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
//
// A pop is a discontinuity: the picture changes a lot while the camera
// barely moves. Smooth motion changes the image in proportion to how far
// the camera turned; a pop does not.
//
// So: read the frame back small, compare with the last one, and divide by
// the angle turned since. Steady while dragging means the renderer is
// behaving. Spikes mean something switched. The running peak survives the
// moment so it can be read after the fact rather than caught live.
//
// This is StopThePop's metric in miniature - they warp frame i to i+t with
// optical flow and compare, which handles translation too. Rotating in
// place needs no flow, so a direct difference is enough.
const POP_W = 192, POP_H = 108;
// Set each frame by the topological sort: how many pairwise constraints it
// had, how many it declined to enforce because the eye sits in the boundary
// plane (merging's job), and whether relief tilted planes enough to make
// three cells disagree. A non-zero `cycles` is worth knowing about; it means
// the sort fell back to the depth key for some cells.
let orderStats = { cycles: 0, weak: 0, constrained: 0 };
let cycleWarned = false;

// Selective merging. GSWT Section 3.4, second half.
//
// Where the camera stands in the plane of a shared boundary neither cell is
// in front and no draw order is right, so the pair is drawn as one stream
// with their splats interleaved. The worker builds that stream; this side
// decides which cells to group, asks for it, and draws the answer.
//
// The request is asynchronous, so a merged stream always describes the
// previous frame's camera. That is fine - it is one frame of lag on a
// boundary the camera is barely moving across - but it means a group can
// arrive that no longer matches the grouping this frame. `mergeKey` is the
// signature the result has to match; when it does not, the cells are drawn
// separately, which is what the renderer did before merging existed. There
// is no failure mode here worse than the status quo.
// Both of these are driven from the panel. They stay plain variables with
// window mirrors rather than reading window every frame, so the controls
// are the source of truth and the console overrides still work for anyone
// used to them.
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
  // Read a small window from the middle of the canvas: cheap, and it skips
  // the edges where cells legitimately enter and leave.
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

  // Below a twentieth of a degree the camera is effectively still, and
  // dividing by it turns rounding into a false spike.
  if (moved < 0.05) return;
  popRate = diff / moved;
  if (popRate > popPeak) { popPeak = popRate; popPeakAge = 0; }
}

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  // Below full size while auto quality needs it; the canvas is still laid
  // out at full size, so the browser scales the smaller image up.
  const k = autoQuality ? renderScale : 1;
  const w = Math.max(1, Math.floor(canvas.clientWidth * dpr * k));
  const h = Math.max(1, Math.floor(canvas.clientHeight * dpr * k));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
    gl.viewport(0, 0, w, h);
  }
}

function frame() {
  // Frame-to-frame time, kept for the scaling sweep. performance.now rather
  // than the rAF timestamp so it reads the same whether or not the browser
  // is throttling the callback.
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
  // The far field is the next LOD level: it takes over where the coarsest
  // splat level would begin, fading in over the same last fifth of the
  // octave that lodFor() uses, so there is one sequence of transitions
  // rather than two bands moving at different distances.
  // One level earlier than the coarsest: at the default that is 16 tiles
  // rather than 32, a quarter of the splat cells to sort and draw. The
  // coarsest splat level added little that the atlas does not show, and
  // cost most of the frame (moving re-sorts every visible cell).
  const farStart = (lodOn && lodLevels > 1
    ? lodBase * Math.pow(2, Math.max(0, lodLevels - 2)) : farStartTiles) * (tileSize || 1);
  const farBand = farStart * (1 - Math.pow(2, -0.2));
  const farLive = farOn && atlas && farCount && cellTex && tileSize;
  if (farLive) {
    // Opaque-ish ground with depth, drawn first; the splats go on top
    // without depth, exactly as before, so nothing near the camera changes.
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
    // The height field, so the geometry is displaced by the same surface
    // the material rule read. Without this the shader falls back to the
    // analytic sine waves, which is what it did for every field ever
    // loaded until now.
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

    // Cells far to near, composited with 'over'. Splats are sorted within
    // a patch and never across cells, which is the approximation that
    // produces the boundary artifact.
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
      // A cell is tested by its centre, so the margin has to cover
      // everything that centre stands for: half a tile to a corner, the
      // overhang past it, and the slab's own thickness once the surface
      // tips it toward the camera. Being stingy here drops a cell that is
      // still partly on screen, and the gap it leaves fills with its
      // neighbour's overhang - which reads as the tile changing.
      //
      // The two axes also need their own margins. The frustum is wider
      // than it is tall by the aspect ratio, so sharing one number drops
      // cells above and below while keeping the ones to the side. Rotate,
      // and rows swing between the two, which is why it shows most when a
      // row lines up with the screen.
      const reach = tileSize * 2.0 + Math.abs(amplitude());
      const padY = reach + Math.max(z, 0.01) * tanHalf;
      const padX = reach + Math.max(z, 0.01) * tanHalf * aspect;
      if (Math.abs(sx) > padX || Math.abs(sy) > padY) continue;

      // Order by the corner nearest the camera rather than the centre.
      //
      // Two cells only need ordering where they overlap, which is at their
      // shared edge, and the centre says nothing about that. It does not
      // remove the swap: neighbouring cells still cross in depth as the
      // camera turns, and at the crossing the strip they share repaints
      // from the other tile. There is no ordering that fixes it, because
      // one cell covering the other is already the wrong answer - the two
      // sets of splats need to interleave. GSWT merges such pairs into one
      // sorted stream when the camera nears their shared boundary plane
      // (Section 3.4). That is the fix; this is only a better ordering.
      let near = Infinity;
      for (const [ox, oy] of [[-half, -half], [half, -half],
                              [-half, half], [half, half]]) {
        const kx = dx + ox, ky = dy + oy;
        const kz = kx * b.forward[0] + ky * b.forward[1] + dz * b.forward[2];
        if (kz < near) near = kz;
      }
      // Handed over to the far field: past its band this cell is atlas.
      if (farLive && near > farStart) continue;
      visible.push({ c, near, dist, side: Math.abs(sx) + Math.abs(sy) });
    }

    // Far to near.
    //
    // Ranking cells by `near` alone is degenerate: when the view direction
    // lines up with a grid axis every cell in a row projects to the same
    // value, bit for bit, and whatever breaks the tie decides the whole row
    // at once. Rotating through alignment then reverses nine cells in one
    // frame, which is the row-wide repaint the pop meter sees. Measured at
    // 27, 54 and 27 reversed pairs at 0, 89 and 90 degrees (probe_order.py).
    //
    // order.js replaces the ranking with one constraint per adjacent pair,
    // decided by which side of their shared boundary plane the eye is on -
    // a sign test, so two cells at identical depth are still separable -
    // then topologically sorts. `near` is demoted to a tiebreak among cells
    // that share no boundary and therefore cannot overlap, which is exactly
    // where its degeneracy is harmless. Those three counts go to zero.
    //
    // What remains is a boundary the camera genuinely crosses, where the
    // sign passes through zero and no order is right. That is what the
    // merging below is for.
    //
    // `window.bozkirTopo = false` in the console restores the old ranking,
    // so the pop meter can be read against both without a rebuild.
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
    //
    // Group the visible cells by how near the eye is to each shared
    // boundary plane, and ask the worker for one interleaved stream per
    // group. The threshold is a distance in world units; a tile is the
    // natural unit for it, since it is the tile that decides how far a
    // splat reaches past its own boundary.
    //
    // Merged cells are drawn at level 0 with no cross-fade. Merging only
    // fires where the eye is practically on a boundary, which is the
    // nearest ground there is, so level 0 is what those cells would have
    // been given anyway - and a per-slot fade would need eight more
    // uniforms to express something no camera will see.
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

        // The signature a returned stream has to match: which cells, in
        // which groups, and roughly which way the camera faces. Without
        // the direction a stream built for the opposite heading would be
        // accepted and drawn back to front.
        const dir = b.forward.map((v) => Math.round(v * 24)).join(',');
        const key = dir + '|' + groups.map((x) => x.map(
          (k) => `${vc[k].i}.${vc[k].j}`).join('+')).join('/');
        if (key !== mergeKey) {
          mergeKey = key;
          mergeSeq++;
          worker.postMessage({
            type: 'mergeGroups', eye: [0, 0, 0], seq: mergeSeq,
            // The view direction turned into the tile's own frame, because
            // stored splat positions are tile-local and the layout may be
            // rotated. Everything world-space - where the cell sits, where
            // the eye is - goes into the per-cell shift instead.
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
        // Nothing to merge from here. The key has to be cleared, not left
        // as it was: anything asking whether a merged stream is current
        // compares mergeHaveKey against it, and a key describing a
        // grouping that no longer exists can never be matched. The capture
        // waits on exactly that and would wait for ever.
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

      // A merged group is drawn whole, at the position of whichever of its
      // cells comes first in the order, and skipped thereafter.
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

      // A cell whose class the terrain chose only weakly is drawn twice:
      // once in its own class and once in the other, dissolved against
      // each other per splat. That turns a staircase of whole tiles into a
      // boundary that wanders at splat scale, which is what Hybrid GSWT's
      // per-Gaussian priority achieves and a per-tile choice cannot.
      //
      // `blendWidth` is how undecided a cell has to be to be worth the
      // second draw. Cells well inside one class cost nothing extra.
      let mix = -1, other = -1;
      if (classBlend && classCount > 1 && tileClass && c.mix != null) {
        // c.mix is the margin the cell decided by, 0 for a coin toss and 1
        // where the rule is as sure as it gets anywhere on this terrain.
        // A cell below the width is close enough to the boundary to be
        // worth drawing both ways.
        if (c.mix < blendWidth) {
          // Half and half at a dead tie, all of its own class at the edge
          // of the band.
          mix = 0.5 + 0.5 * (c.mix / Math.max(blendWidth, 1e-6));
          other = alternateTile(c, p);
        }
      }

      cellXYArr[0] = c.x; cellXYArr[1] = c.y;
      gl.uniform2fv(splatU.cellXY, cellXYArr.subarray(0, 2));
      gl.uniform1f(splatU.mix, mix);
      if (showClasses && classCount > 1) {
        // What the rule decided, painted over what the tiles look like.
        // Without this the only evidence of the decision is the material
        // drawn, which is a different thing: tiles are sorted into classes
        // by average appearance, so a sand tile can hold a bush, and a
        // cell in the blend band draws some of its runner-up as well. A
        // summit that looks green is then either a rule that went wrong or
        // a tile that is not what its class says, and there was no way to
        // tell the two apart from the picture.
        //
        // Hue is the class. Washed out towards grey is low confidence,
        // which is also where blending is mixing materials in.
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

      // Which order to use. Live means one order for everything, along the
      // world view direction - correct only while tiles are unrotated.
      // Cached means each cell asks for the order matching the view
      // direction in its own frame, which is what a warped tile actually
      // sees. W is orthonormal, so that direction is W transposed times v.
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
        // Only touch the tint if this view owns it. Zeroing it here
        // unconditionally cancels whatever the per-cell branch above set,
        // which is what made the tile-identity view do nothing at all.
        if (showLodColours) {
          const c2 = LOD_RGB[li % LOD_RGB.length];
          gl.uniform3f(splatU.tint, c2[0], c2[1], c2[2]);
          gl.uniform1f(splatU.tintAmount, 0.6);
        }
        gl.vertexAttribIPointer(aIndex, 1, gl.UNSIGNED_INT, 0,
                                (base + start) * 4);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
        drawnSplats += count; drawCalls++;

        // The other class, taking the splats this one dissolved away.
        // Same code, same position, same level - only the material
        // differs, and the two dissolves are complementary so the cell
        // ends up with one splat everywhere rather than two or none.
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
    // No DOM id for these, so they go to the console rather than the panel.
    // `window.bozkirOrderStats` holds the latest; a cycle is reported once
    // per run because it means the sort fell back to the depth key and the
    // guarantee above does not hold for those cells.
    if (ui.classnote) {
      if (classCount < 2) {
        ui.classnote.textContent = 'one class in this tileset';
      } else if (!classOn) {
        ui.classnote.textContent = `${classCount} classes, placed at random`;
      } else if (classStats) {
        const total = classStats.counts.reduce((a, b) => a + b, 0) || 1;
        const share = classStats.counts
          .map((c, k) => `${k}: ${(100 * c / total).toFixed(0)}%`).join('  ');
        // Aliasing is the one failure the rule cannot see for itself. If
        // the height field is squeezed into a few tiles it repeats across
        // the grid, and the landforms are then only a cell or two across -
        // at which point no amount of sampling helps, because the decision
        // is per cell and there is nothing left to resolve. It looks like
        // the rule producing a chequerboard, so it is worth saying plainly
        // rather than leaving it to be discovered.
        const repeats = field && reliefScale > 0
          ? Math.round(gridN / reliefScale) : 1;
        // `sure` is the mean margin now, so it reads as how strongly the
        // terrain is driving the choice rather than as a probability.
        // What the sliders are actually worth, rather than what they are
        // called. 'follows height' is one number in a weighted sum and
        // reads as if it were the whole rule; printing the shares makes
        // the trade visible while it is being dragged.
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

  // After the draw and inside the same animation frame: the context has
  // no preserveDrawingBuffer, so the colour buffer is only readable here.
  if (capture.active) capture.step();
  if (bench.active) bench.step(benchDt);
  if (shotWanted) { shotWanted = false; saveFrame(); }
  tuneQuality(benchDt);
  moveCamera(benchDt);

  requestAnimationFrame(frame);
}

/** Write the frame that has just been drawn to a PNG file.
 *
 *  Must run inside the animation frame that drew it: the context asks for
 *  no preserveDrawingBuffer, so the colour buffer is cleared the moment
 *  the frame ends. A figure taken this way is the full canvas at its real
 *  resolution, with none of the panel over it, which is what a screenshot
 *  of the window can never quite be.
 */
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
  // Orbiting, the eye swings around a fixed point; flying, the view turns
  // about a fixed eye. Opposite senses, same two numbers.
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
    // Zoom sets a goal the camera eases towards (moveCamera), in even steps
    // whatever the device reports: trackpads send pixels, some mice lines.
    const dy = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaY;
    const near = Math.max(0.05, 0.5 * (tileSize || 0.1));
    zoomGoal = Math.max(near, (zoomGoal || cam.distance) * Math.exp(dy * 0.0012));
  }
}, { passive: false });
let zoomGoal = 0;

const held = new Set();
addEventListener('keydown', (e) => held.add(e.key.toLowerCase()));
addEventListener('keyup', (e) => held.delete(e.key.toLowerCase()));
// Movement runs in the frame loop with the real time between frames, and
// through a velocity that eases towards what the keys ask for: pressing a
// key accelerates over about a tenth of a second, releasing it glides to a
// stop. A fixed step on its own timer started and stopped instantly and
// drifted against the frames, which read as judder.
const camVel = [0, 0, 0];
const EASE_S = 0.12;

function moveCamera(dt) {
  if (!splatCount || !(dt > 0)) return;
  const sec = Math.min(dt, 100) / 1000;
  if (zoomGoal && !cam.fly) {
    // Ease the distance towards the scroll's goal, in log space so zooming
    // in and out feel the same.
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
    // Scaled with the view, as before, but with a floor: zoomed in close,
    // walking speed tied to the distance fell to a crawl.
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

  // The orbited point stays on the map: past the grid there is no ground to
  // look at, which looked the same as being stuck.
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

// Opening a tileset: from the file dialog, or dropped anywhere on the page.
//
// The old path took a .splat and then went looking in web/data for a .json
// of the same name, which only ever worked for scenes already exported into
// the repo. tileset.js takes a zip, or the .splat and .json together, so a
// tileset built on someone else's machine opens here.
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
    // A .splat has no header, so without metadata the tile size is a guess,
    // and a wrong tile size reads as a broken export rather than a missing
    // file. Say which it is.
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

// Drop anywhere. Dragover has to be cancelled on both the enter and the
// over event or the browser navigates away to the file instead, which
// loses the page and looks like a crash.
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

// The drop hint and its styling are built here rather than added to
// index.html, so the page keeps working unchanged if this module is not
// loaded.
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
/** Flag a relief scale below the grid: the height field then covers less
 *  than the tiles do, mirrors to fill the rest, and landforms repeat. */
function markReliefScale() {
  if (!ui.reliefscalen) return;
  const short = field && reliefScale < gridN;
  ui.reliefscalen.textContent = short
    ? `${reliefScale} < grid ${gridN}, repeats` : String(reliefScale);
  ui.reliefscalen.style.color = short ? '#d9a441' : '';
}

on('grid', 'input', (e) => {
  gridN = +e.target.value;
  // A field covering the grid should keep covering it as the grid changes,
  // until somebody moves the scale slider themselves.
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
/** Panel and axis marker out of the way, so the canvas is the figure.
 *  Nothing else changes - not the camera, not the rule - because a figure
 *  that differs from what was on screen is not evidence of anything. */
function setFigureMode(on) {
  figureMode = on;
  // One class on the body rather than styles on each element: the panel,
  // its drag handle and the axis marker all go together, and the handle
  // was the one left behind before - an invisible strip that still grabbed
  // the pointer at the edge where the panel had been.
  document.body.classList.toggle('figure', on);
  if (ui.figuremode) ui.figuremode.checked = on;
  // The way back stays readable long enough to be noticed, then fades so
  // it does not sit in the middle of what is being framed.
  if (ui.figexit) {
    ui.figexit.classList.remove('faded');
    clearTimeout(figExitTimer);
    if (on) figExitTimer = setTimeout(() => ui.figexit.classList.add('faded'), 2500);
  }
}
on('expert', 'change', (e) => {
  // Remembered, because somebody who wants the measurement controls wants
  // them every time, and somebody who does not should never meet them.
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
  // P takes the picture, F leaves figure mode - with the panel hidden
  // there is no checkbox left to untick.
  if (k === 'p') shotWanted = true;
  else if (k === 'f') setFigureMode(!figureMode);
  else if (k === 'escape' && figureMode) setFigureMode(false);
});
on('figexit', 'click', () => setFigureMode(false));
on('farfield', 'change', (e) => { farOn = e.target.checked; });
if (ui.gpuname) {
  ui.gpuname.textContent = String(gpuName).replace(/^ANGLE \((.*)\)$/, '$1');
  // An integrated chip on a machine that may have a faster one: say so.
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
// Smoothing is a switch first, a number second: almost everyone wants it
// on, and the frames-per-tile count only means anything once it is off.
on('smoothframes', 'change', (e) => {
  const smooth = e.target.checked;
  if (ui.subdivrow) ui.subdivrow.style.display = smooth ? 'none' : '';
  ui.subdiv.value = smooth ? '17' : '1';
  ui.subdiv.dispatchEvent(new Event('input'));
});
on('subdiv', 'input', (e) => {
  const v = +e.target.value;
  setTimeout(reportSeams, 0);
  // The top of the slider is the limit of subdividing: one frame per splat.
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
  // Moving it takes it over; dragging it back to the grid size hands it
  // back, so it follows the grid again. There was no way back before, and
  // a scale left below the grid is how a terrain ends up squeezed and
  // repeating without anyone having chosen that.
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
  // A stale stream would otherwise be drawn the moment merging came back on.
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

// Kept because they were the only way to reach these before the panel
// existed, and because typing one is quicker than finding the control.
Object.defineProperty(window, 'bozkirTopo', {
  get: () => topoOrder,
  set: (v) => { topoOrder = v !== false; if (ui.tileorder) ui.tileorder.value = topoOrder ? 'topological' : 'depth'; },
});
Object.defineProperty(window, 'bozkirMerge', {
  get: () => mergeOn,
  set: (v) => { mergeOn = v !== false; if (ui.merging) ui.merging.checked = mergeOn; mergeKey = ''; },
});

// --- capturing a sweep, for scripts/pop_metric.py ----------------------
//
// Each pose is held until nothing is in flight: a sort takes a frame or
// more to come back, and a merged group another, so a frame grabbed the
// moment the camera moves shows the previous pose's ordering. That would
// measure worker latency rather than the ordering being compared, and it
// would do it to both runs, hiding the difference the capture exists to
// find.
// --- scaling sweep -----------------------------------------------------
//
// The grid slider's upper end was chosen by nobody. This measures where the
// renderer actually stops keeping up, and which of the frame, the sort or
// the draw calls gives out first - which are different problems with
// different fixes.
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

/** The middle of the laid-out terrain, for the sweep to orbit.
 *
 *  Not the camera's own target: WASD slides that, and a target nudged off
 *  the terrain sends the sweep around a point in mid air, through the
 *  ground on one side and into empty sky on the other. The grid's own
 *  centre is the same place every time, which is also what makes a capture
 *  repeatable across sessions.
 *
 *  Height is taken from the cells rather than assumed zero, so a warped
 *  surface is orbited about its own middle and not about the flat plane it
 *  was displaced from.
 */
function sceneCentre() {
  if (!cells.length) return cam.target.slice();
  let x = 0, y = 0, z = 0;
  for (const c of cells) { x += c.x; y += c.y; z += c.z; }
  return [x / cells.length, y / cells.length, z / cells.length];
}

/** The folder name the zip should unpack into, named for what produced it
 *  so two runs cannot be confused once they are in the downloads folder. */
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
  // Everything needed to repeat this run, in one line to paste into notes.
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

// Both orderings, back to back, from this same camera. The comparison is
// only meaningful if the two runs travel the same path, and doing it in one
// press is the difference between a measurement anyone can repeat and one
// that depends on not touching the mouse in between.
on('captureboth', 'click', () => {
  if (capture.active) return;
  if (!splatCount) { console.warn('nothing loaded to capture'); return; }
  const restore = topoOrder;
  topoOrder = true;
  if (ui.tileorder) ui.tileorder.value = 'topological';
  queuedRun = () => {
    topoOrder = false;
    if (ui.tileorder) ui.tileorder.value = 'depth';
    // A frame's grace so the panel and the next sort settle before the
    // second run starts posing the camera.
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
// The terrain is its own choice. `?scene=desert&height=mesa` puts the
// desert tiles on mesa ground; without it a tileset still looks for the
// height field named after itself, which is what every scene did before.
const wantedHeight = params.get('height');
// And a terrain can be asked for by recipe instead of by file:
// `?gen=canyon&seed=3&size=256` is generated here, the same ground on every
// machine, since the generator is deterministic.
const wantedGen = params.get('gen');
const wantedSeed = Math.max(0, parseInt(params.get('seed') || '0', 10) || 0);
const wantedSize = [128, 256, 512].includes(+params.get('size'))
  ? +params.get('size') : 256;

/** Fill the tileset and terrain menus from data/index.json.
 *
 *  A static server will not list a folder, so the folder lists itself.
 *  Without the index the menus stay empty and everything else works, which
 *  is why nothing here throws.
 */
async function loadCatalogue() {
  let index = null;
  try {
    const r = await fetch('./data/index.json');
    index = r.ok ? await r.json() : null;
  } catch (e) { /* no index: menus stay empty */ }
  if (!index) return;

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
  fill(ui.scenepick, index.scenes || [],
       (s) => `${s.name} · ${s.classes > 1 ? s.classes + ' materials' : '1 material'}`
              + ` · ${s.megabytes} MB`, loadedScene);
  fill(ui.heightpick, index.terrains || [],
       (t) => `${t.name} · ${t.profile || t.source}`
              + (t.sediment ? ' · sediment' : ''),
       wantedHeight || loadedScene);
}

// Changing tileset means a different .splat, so the page reloads with the
// choice in the URL - the same address that comes back from a figure's
// file name, and one that can be sent to somebody.
on('scenepick', 'change', (e) => {
  if (!e.target.value) return;
  const p = new URLSearchParams(location.search);
  p.set('scene', e.target.value);
  location.search = p.toString();
});
// Changing terrain does not: the tiles stay where they are and the ground
// under them is swapped, which is a second or two rather than a reload.
// ------------------------------------------------------------ terrain window
//
// Twelve profiles, each drawn as a small preview at the current seed, so a
// terrain is chosen by looking at it rather than by name. Previews are
// generated at 64 across through the same worker and cache as the real
// thing; changing the seed redraws them, which is how one actually browses
// for a good piece of ground.

let genProfile = wantedGen || 'desert';
let previewToken = 0;

/** A preview of a terrain as it would look under the loaded tiles: the
 *  shape as shading, and each tile's worth of ground coloured by the class
 *  the rule would give it, in that class's average colour. The same
 *  preview under a scrub-and-sand desert capture and under a single-
 *  material cobble capture should look different, because the result
 *  would. */
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
      if (cls) c = classColours[cls[Math.floor(j / over) * cells + Math.floor(i / over)]];
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
//
// Settings by name, from views.json. A preset sets each control it names
// and fires the same event a hand on the slider would, so it can only do
// what the panel can do, and what it did is visible afterwards.

let presets = {};
let pendingPreset = params.get('preset');

async function loadPresets() {
  try {
    const r = await fetch('./views.json');
    presets = r.ok ? await r.json() : {};
  } catch (e) { presets = {}; }
  if (!ui.presetchips) return;
  // Buttons rather than a menu: a first-time visitor can see every
  // starting point at once, and the tooltip on each says what it is for.
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
  // If the ground arrived before this file did, the URL's preset is still
  // waiting; apply it now rather than never.
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

/** Nothing to load: make the starter tileset and lay it on generated
 *  ground, so the viewer never opens onto an empty page. */
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

const attempts = [];
if (wanted === 'starter') loadStarter();
else for (const name of (wanted ? [wanted] : ['scene', 'bigsur', 'garden'])) {
  attempts.push(Promise.all([
    fetch(`./data/${name}.splat`).then(r => (r.ok ? r.arrayBuffer() : null)),
    fetch(`./data/${name}.json`).then(r => (r.ok ? r.json() : null)).catch(() => null),
  ]).then(([b, m]) => {
    if (!b || splatCount) return;
    loadedScene = name;
    load(b, m);
    loadAtlas(name);
    loadCatalogue();
    // The height field is optional and arrives after the splats, so the
    // scene is up either way and gains its terrain a moment later rather
    // than waiting on a file most tilesets do not have.
    if (wantedGen) useGeneratedField(wantedGen, wantedSeed, wantedSize);
    else useHeightField(wantedHeight || name);
  }).catch(() => {}));
}
Promise.allSettled(attempts).then(() => { if (!splatCount) loadStarter(); });

/** Look for a height field beside a tileset and adopt it if there is one. */
async function useHeightField(name) {
  const f = await loadHeightField('./data', name);
  if (!f) return;
  adoptField(f);
}

/** Generate a terrain in the browser (or fetch it from the browser's own
 *  cache) and put the tiles on it. No file, no server, no Python. */
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
    // The terrain menu should say what is under the tiles now, even when
    // it did not come from a file.
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

  // Upload it. R32F with NEAREST because the shader does its own bilinear:
  // a single-channel float texture is not filterable on every device, and
  // relying on the hardware would also mean trusting it to interpolate the
  // same way heightfield.js does.
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

  // And the openness built from it, on its own unit. Computed once here:
  // it depends on the field and nothing the camera or the sliders do.
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
  // Cover the whole grid by default rather than the six tiles the analytic
  // surface wanted. A generated field is 512 across and holds a landscape;
  // showing two tiles of it and mirroring that fourteen times across the
  // grid is the repetition, not the tiling. The analytic surface has no
  // scale of its own so it keeps its old default.
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
  // A preset in the URL waits for the ground: its relief scale and its
  // rule settings mean nothing until there is a field to apply them to.
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
    // A measured field with relief at zero shows nothing, and the person
    // has no way to know a field arrived. Lift it enough to be visible and
    // leave the control where they can see what happened.
    ui.relief.value = Math.min(1.0, +ui.relief.max);
    relief = +ui.relief.value;
    ui.reliefn.textContent = relief.toFixed(2);
    paintSliders();
  }
  // New ground means the rule has to decide again and the cells have to
  // be rebuilt on it. On the first load this is a repeat of the layout
  // that just happened; when the terrain is swapped from the menu it is
  // the whole point.
  regenerate();
  console.log(`height field: ${f.describe()}`);
}

paintSliders();
frame();