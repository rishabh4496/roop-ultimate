// ── GlRenderer: two picture layers, one draw call ─────────────────────────
//
// Shared by <FastCanvasPlayer> (pushed JPEG frames, usually inside a worker on
// an OffscreenCanvas) and <VideoCompareStage> (two <video> elements on the
// main thread). No DOM, no React: it takes anything WebGL's texImage2D takes
// (ImageBitmap, HTMLVideoElement, HTMLImageElement, a canvas) and a canvas-like
// object (HTMLCanvasElement or OffscreenCanvas).
//
// LAYERS AND MODES
//   layer 0 = A (the original / the only picture), layer 1 = B (the result)
//   'single' — A, fitted contain into the whole viewport
//   'wipe'   — A left of the split, B right of it; both fitted to the viewport,
//              so the cut shows the same pixels of two aligned pictures
//   'side'   — A fitted into the left half, B into the right half
//
// DOUBLE BUFFERING
// Each layer owns TWO textures. An upload always goes into the one NOT being
// displayed, and only then do the two swap, so a frame is never sampled while
// it is half-written and an upload never stalls on a texture the GPU is still
// reading for the previous draw. Same-size uploads use texSubImage2D, which
// reuses the allocation instead of reallocating per frame.
//
// FALLBACK & STAGE 18 WEBGL2 3D LUT COLOR PIPELINE
// WebGL2 -> WebGL1 -> Canvas 2D. In WebGL2 mode, the pipeline executes in linear
// float space with 3-way Lift/Gamma/Gain wheels, Kelvin Temperature/Tint,
// Vibrance with 20°-50° skin-tone hue corridor protection, trilinear 3D .cube
// LUT evaluation on a dedicated float sampler3D unit, and strict premultiplied
// alpha scaling (rgb * a, a).
// Context loss (the GPU process resetting, which Chromium does under memory pressure)
// is survived: the next frame re-creates the resources and is drawn normally.

import { uploadLut3DTexture } from '../faceswap/cubeLut.js';

const VERT = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main() {
  // Clip space -> 0..1 with the origin TOP-left, the way boxes are measured.
  v_uv = vec2(a_pos.x * 0.5 + 0.5, 0.5 - a_pos.y * 0.5);
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FRAG = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
varying vec2 v_uv;
uniform sampler2D u_texA;
uniform sampler2D u_texB;
uniform vec4 u_boxA;      // x, y, w, h as fractions of the viewport
uniform vec4 u_boxB;
uniform float u_hasA;
uniform float u_hasB;
uniform float u_mode;     // 0 single, 1 wipe, 2 side
uniform float u_split;    // wipe position, fraction of the viewport width

vec4 layer(sampler2D tex, vec4 box, float has) {
  vec2 q = (v_uv - box.xy) / box.zw;
  if (has < 0.5 || q.x < 0.0 || q.y < 0.0 || q.x > 1.0 || q.y > 1.0) return vec4(0.0);
  return vec4(texture2D(tex, q).rgb, 1.0);
}

void main() {
  if (u_mode < 0.5) {
    gl_FragColor = layer(u_texA, u_boxA, u_hasA);
  } else if (u_mode < 1.5) {
    gl_FragColor = v_uv.x < u_split ? layer(u_texA, u_boxA, u_hasA)
                                    : layer(u_texB, u_boxB, u_hasB);
  } else {
    vec4 a = layer(u_texA, u_boxA, u_hasA);
    gl_FragColor = a.a > 0.0 ? a : layer(u_texB, u_boxB, u_hasB);
  }
}`;

const VERT_300 = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = vec2(a_pos.x * 0.5 + 0.5, 0.5 - a_pos.y * 0.5);
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FRAG_300 = `#version 300 es
precision highp float;
precision highp sampler3D;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_texA;
uniform sampler2D u_texB;
uniform sampler3D u_lut3d;
uniform vec4 u_boxA;
uniform vec4 u_boxB;
uniform float u_hasA;
uniform float u_hasB;
uniform float u_mode;
uniform float u_split;

// Stage 18 WebGL2 Grading Uniforms
uniform float u_hasLut;
uniform float u_lutOpacity;
uniform float u_lutSize;
uniform vec3 u_domainMin;
uniform vec3 u_domainMax;
uniform float u_temperature;
uniform float u_tint;
uniform float u_vibrance;
uniform float u_skinProtect;
uniform vec3 u_lift;
uniform vec3 u_gamma;
uniform vec3 u_gain;
uniform float u_premultiply;

// Stage 21 WebGL2 3D / VR Equirectangular & Stereo Inspection Uniforms
uniform float u_vrMode;        // 0.0: 2D Flat, 1.0: VR180 Equirectangular, 2.0: VR360 Equirectangular
uniform float u_yaw;           // Camera yaw (radians)
uniform float u_pitch;         // Camera pitch (radians)
uniform float u_fov;           // Field of view (radians)
uniform float u_aspect;        // Viewport aspect ratio (W / H)
uniform float u_stereoMode;    // 0: Off, 1: Left Eye, 2: Right Eye, 3: Side-by-Side, 4: Red-Cyan Anaglyph, 5: Disparity Diff Map
uniform float u_stereoLayout;  // 0: Auto, 1: SBS, 2: TB, 3: DualTexture (texA=L, texB=R)

vec3 srgbToLinear(vec3 c) {
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}

vec3 linearToSrgb(vec3 c) {
  c = clamp(c, 0.0, 1.0);
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

vec3 rgbToHsv(vec3 c) {
  vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
  vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
  float d = q.x - min(q.w, q.y);
  float e = 1.0e-10;
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsvToRgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 applyGrade(vec3 rgb) {
  vec3 linear = srgbToLinear(rgb);
  vec3 shift = vec3(0.0);
  if (u_temperature > 0.0) { shift.r += u_temperature * 0.28; shift.b -= u_temperature * 0.22; }
  else if (u_temperature < 0.0) { shift.r += u_temperature * 0.20; shift.b -= u_temperature * 0.30; }
  shift.g -= u_tint * 0.25; shift.r += u_tint * 0.125; shift.b += u_tint * 0.125;
  linear = max(vec3(0.0), linear + shift);
  vec3 lifted = max(vec3(0.0), linear + u_lift * (vec3(1.0) - linear));
  vec3 gained = max(vec3(0.0), lifted * u_gain);
  linear = pow(gained, vec3(1.0) / max(u_gamma, vec3(0.01)));
  vec3 srgb = linearToSrgb(linear);
  if (abs(u_vibrance) > 0.001) {
    vec3 hsv = rgbToHsv(srgb);
    float hueDeg = hsv.x * 360.0;
    float skinDist = abs(hueDeg - 35.0);
    if (skinDist > 180.0) skinDist = 360.0 - skinDist;
    float skinWeight = 1.0 - smoothstep(12.0, 25.0, skinDist);
    float satInv = (1.0 - hsv.y);
    float effVib = u_vibrance * satInv * (1.0 - skinWeight * u_skinProtect);
    hsv.y = clamp(hsv.y * (1.0 + effVib), 0.0, 1.0);
    srgb = hsvToRgb(hsv);
  }
  if (u_hasLut > 0.5) {
    vec3 norm = (clamp(srgb, u_domainMin, u_domainMax) - u_domainMin) / max(u_domainMax - u_domainMin, vec3(0.0001));
    vec3 coord = (norm * (u_lutSize - 1.0) + 0.5) / u_lutSize;
    vec3 lutCol = texture(u_lut3d, coord).rgb;
    srgb = mix(srgb, lutCol, clamp(u_lutOpacity, 0.0, 1.0));
  }
  return srgb;
}

vec2 computeEquirectangularUV(vec2 uv, float vrMode, float yaw, float pitch, float fov, float aspect) {
  if (vrMode < 0.5) {
    return uv;
  }
  float tanHalfFov = tan(max(0.01, fov * 0.5));
  float rx = (uv.x - 0.5) * 2.0 * aspect * tanHalfFov;
  float ry = (0.5 - uv.y) * 2.0 * tanHalfFov;
  vec3 ray = normalize(vec3(rx, ry, 1.0));

  // Rotate pitch (X-axis) then yaw (Y-axis)
  float cp = cos(pitch); float sp = sin(pitch);
  vec3 rPitch = vec3(ray.x, ray.y * cp - ray.z * sp, ray.y * sp + ray.z * cp);
  float cy = cos(yaw); float sy = sin(yaw);
  vec3 dir = vec3(rPitch.x * cy + rPitch.z * sy, rPitch.y, -rPitch.x * sy + rPitch.z * cy);

  float lon = atan(dir.x, dir.z);
  float lat = asin(clamp(dir.y, -1.0, 1.0));

  float u = 0.5;
  if (vrMode > 1.5) {
    // 360 degree sphere
    u = lon / (2.0 * 3.141592653589793) + 0.5;
  } else {
    // 180 degree hemisphere (centered)
    u = lon / 3.141592653589793 + 0.5;
  }
  float v = 0.5 - lat / 3.141592653589793;
  return vec2(clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0));
}

vec4 sampleLayerPos(sampler2D tex, vec4 box, float has, bool grade, vec2 pos) {
  vec2 q = (pos - box.xy) / box.zw;
  if (has < 0.5 || q.x < 0.0 || q.y < 0.0 || q.x > 1.0 || q.y > 1.0) return vec4(0.0);
  vec4 s = texture(tex, q);
  vec3 c = grade ? applyGrade(s.rgb) : s.rgb;
  if (u_premultiply > 0.5) {
    return vec4(c * s.a, s.a);
  }
  return vec4(c, s.a);
}

vec4 layer(sampler2D tex, vec4 box, float has, bool grade) {
  return sampleLayerPos(tex, box, has, grade, v_uv);
}

void main() {
  if (u_vrMode > 0.5 || u_stereoMode > 0.5) {
    vec2 p = computeEquirectangularUV(v_uv, u_vrMode, u_yaw, u_pitch, u_fov, u_aspect);
    vec2 uvL = p;
    vec2 uvR = p;
    if (u_stereoLayout < 1.5) {
      // Side-by-Side (SBS)
      uvL = vec2(p.x * 0.5, p.y);
      uvR = vec2(0.5 + p.x * 0.5, p.y);
    } else if (u_stereoLayout < 2.5) {
      // Top-Bottom (TB)
      uvL = vec2(p.x, p.y * 0.5);
      uvR = vec2(p.x, 0.5 + p.y * 0.5);
    }

    vec4 colL;
    vec4 colR;
    if (u_stereoLayout > 2.5) {
      // Dual texture (Layer A = Left Eye, Layer B = Right Eye)
      colL = sampleLayerPos(u_texA, u_boxA, u_hasA, false, uvL);
      colR = sampleLayerPos(u_texB, u_boxB, u_hasB, true, uvR);
    } else {
      sampler2D mainTex = (u_hasB > 0.5) ? u_texB : u_texA;
      vec4 mainBox = (u_hasB > 0.5) ? u_boxB : u_boxA;
      float mainHas = (u_hasB > 0.5) ? u_hasB : u_hasA;
      bool mainGrade = (u_hasB > 0.5);
      colL = sampleLayerPos(mainTex, mainBox, mainHas, mainGrade, uvL);
      colR = sampleLayerPos(mainTex, mainBox, mainHas, mainGrade, uvR);
    }

    if (u_stereoMode < 0.5) {
      // Projection view (mono/active layer or left eye)
      fragColor = colL;
    } else if (u_stereoMode < 1.5) {
      // Left Eye
      fragColor = colL;
    } else if (u_stereoMode < 2.5) {
      // Right Eye
      fragColor = colR;
    } else if (u_stereoMode < 3.5) {
      // Side-by-Side
      fragColor = v_uv.x < 0.5 ? colL : colR;
    } else if (u_stereoMode < 4.5) {
      // Red-Cyan 3D Anaglyph
      vec3 anaglyph = vec3(colL.r, colR.g, colR.b);
      fragColor = vec4(anaglyph, max(colL.a, colR.a));
    } else {
      // Stereo Disparity Diff Map (amplified difference with color relief)
      vec3 diff = abs(colL.rgb - colR.rgb);
      float dMag = max(max(diff.r, diff.g), diff.b) * 4.0;
      vec3 turbo = vec3(clamp(dMag * 2.0, 0.0, 1.0), clamp(dMag * 0.9, 0.0, 1.0), clamp(1.0 - dMag * 1.4, 0.0, 1.0));
      fragColor = vec4(mix(diff * 3.5, turbo, 0.55), 1.0);
    }
    return;
  }

  // Standard non-VR execution
  if (u_mode < 0.5) {
    fragColor = layer(u_texA, u_boxA, u_hasA, false);
  } else if (u_mode < 1.5) {
    fragColor = v_uv.x < u_split ? layer(u_texA, u_boxA, u_hasA, false)
                                 : layer(u_texB, u_boxB, u_hasB, true);
  } else {
    vec4 a = layer(u_texA, u_boxA, u_hasA, false);
    fragColor = a.a > 0.0 ? a : layer(u_texB, u_boxB, u_hasB, true);
  }
}`;

const MODES = { single: 0, wipe: 1, side: 2 };

const sourceSize = (src) => {
  if (!src) return [0, 0];
  if (typeof src.videoWidth === 'number' && src.videoWidth) return [src.videoWidth, src.videoHeight];
  if (typeof src.naturalWidth === 'number' && src.naturalWidth) return [src.naturalWidth, src.naturalHeight];
  if (typeof src.displayWidth === 'number' && src.displayWidth) return [src.displayWidth, src.displayHeight];
  return [src.width || 0, src.height || 0];
};

/** \object-fit: contain\ of a (iw x ih) picture into a region, as fractions. */
export function fitContain(iw, ih, region, viewW, viewH) {
  const [rx, ry, rw, rh] = region;           // fractions of the viewport
  const pw = rw * viewW;
  const ph = rh * viewH;
  if (!iw || !ih || !pw || !ph) return [rx, ry, rw, rh];
  const s = Math.min(pw / iw, ph / ih);
  const w = (iw * s) / viewW;
  const h = (ih * s) / viewH;
  return [rx + (rw - w) / 2, ry + (rh - h) / 2, w, h];
}

export class GlRenderer {
  /**
   * @param {HTMLCanvasElement|OffscreenCanvas} canvas
   * @param {{ force2d?: boolean }} [opts]
   */
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.mode = 'single';
    this.split = 0.5;
    this.layers = [
      { w: 0, h: 0, has: false, img2d: null, owned2d: false },
      { w: 0, h: 0, has: false, img2d: null, owned2d: false },
    ];
    this.lost = false;
    this.kind = '2d';
    this.gl = null;
    this.ctx2d = null;
    this.lutTexture = null;
    this.lutInfo = null;
    this.gradingParams = {
      lut_opacity: 1.0,
      color_temperature: 0.0,
      color_tint: 0.0,
      color_vibrance: 0.0,
      color_skin_protect: 1.0,
      lift_r: 0.0, lift_g: 0.0, lift_b: 0.0, lift_m: 0.0,
      gamma_r: 1.0, gamma_g: 1.0, gamma_b: 1.0, gamma_m: 1.0,
      gain_r: 1.0, gain_g: 1.0, gain_b: 1.0, gain_m: 1.0,
    };
    this.vrParams = {
      vrMode: 'off',        // 'off' | 'vr180' | 'vr360'
      yaw: 0.0,             // radians
      pitch: 0.0,           // radians
      fov: 90.0,            // degrees
      stereoMode: 'off',    // 'off' | 'left' | 'right' | 'side_by_side' | 'anaglyph' | 'disparity_diff'
      stereoLayout: 'sbs',  // 'sbs' | 'tb' | 'dual'
    };
    if (!opts.force2d) this._initGl();
    if (!this.gl) {
      this.ctx2d = canvas.getContext('2d', { alpha: true });
      this.kind = '2d';
    }
    if (this.gl && typeof canvas.addEventListener === 'function') {
      this._onLost = (e) => { e.preventDefault(); this.lost = true; };
      this._onRestored = () => { this.lost = false; this._initGlResources(); };
      canvas.addEventListener('webglcontextlost', this._onLost);
      canvas.addEventListener('webglcontextrestored', this._onRestored);
    }
  }

  _initGl() {
    const attrs = {
      alpha: true, antialias: false, depth: false, stencil: false,
      premultipliedAlpha: true, preserveDrawingBuffer: false,
      powerPreference: 'low-power',
    };
    let gl = null;
    try { gl = this.canvas.getContext('webgl2', attrs); } catch { gl = null; }
    if (gl) this.kind = 'webgl2';
    else {
      try { gl = this.canvas.getContext('webgl', attrs); } catch { gl = null; }
      if (gl) this.kind = 'webgl';
    }
    if (!gl) return;
    this.gl = gl;
    if (!this._initGlResources()) {
      this.gl = null;
      this.kind = '2d';
    }
  }

  _initGlResources() {
    const gl = this.gl;
    const isW2 = this.kind === 'webgl2';
    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        gl.deleteShader(s);
        return null;
      }
      return s;
    };
    const vs = compile(gl.VERTEX_SHADER, isW2 ? VERT_300 : VERT);
    const fs = compile(gl.FRAGMENT_SHADER, isW2 ? FRAG_300 : FRAG);
    if (!vs || !fs) return false;
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return false;
    this.prog = prog;
    this.loc = {
      pos: gl.getAttribLocation(prog, 'a_pos'),
      texA: gl.getUniformLocation(prog, 'u_texA'),
      texB: gl.getUniformLocation(prog, 'u_texB'),
      boxA: gl.getUniformLocation(prog, 'u_boxA'),
      boxB: gl.getUniformLocation(prog, 'u_boxB'),
      hasA: gl.getUniformLocation(prog, 'u_hasA'),
      hasB: gl.getUniformLocation(prog, 'u_hasB'),
      mode: gl.getUniformLocation(prog, 'u_mode'),
      split: gl.getUniformLocation(prog, 'u_split'),
    };
    if (isW2) {
      this.loc.lut3d = gl.getUniformLocation(prog, 'u_lut3d');
      this.loc.hasLut = gl.getUniformLocation(prog, 'u_hasLut');
      this.loc.lutOpacity = gl.getUniformLocation(prog, 'u_lutOpacity');
      this.loc.lutSize = gl.getUniformLocation(prog, 'u_lutSize');
      this.loc.domainMin = gl.getUniformLocation(prog, 'u_domainMin');
      this.loc.domainMax = gl.getUniformLocation(prog, 'u_domainMax');
      this.loc.temperature = gl.getUniformLocation(prog, 'u_temperature');
      this.loc.tint = gl.getUniformLocation(prog, 'u_tint');
      this.loc.vibrance = gl.getUniformLocation(prog, 'u_vibrance');
      this.loc.skinProtect = gl.getUniformLocation(prog, 'u_skinProtect');
      this.loc.lift = gl.getUniformLocation(prog, 'u_lift');
      this.loc.gamma = gl.getUniformLocation(prog, 'u_gamma');
      this.loc.gain = gl.getUniformLocation(prog, 'u_gain');
      this.loc.premultiply = gl.getUniformLocation(prog, 'u_premultiply');
      this.loc.vrMode = gl.getUniformLocation(prog, 'u_vrMode');
      this.loc.yaw = gl.getUniformLocation(prog, 'u_yaw');
      this.loc.pitch = gl.getUniformLocation(prog, 'u_pitch');
      this.loc.fov = gl.getUniformLocation(prog, 'u_fov');
      this.loc.aspect = gl.getUniformLocation(prog, 'u_aspect');
      this.loc.stereoMode = gl.getUniformLocation(prog, 'u_stereoMode');
      this.loc.stereoLayout = gl.getUniformLocation(prog, 'u_stereoLayout');
    }

    // One triangle strip covering the viewport.
    this.vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const makeTex = () => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return { tex: t, w: 0, h: 0 };
    };
    this.tex = [[makeTex(), makeTex()], [makeTex(), makeTex()]];
    this.front = [0, 0];
    for (const l of this.layers) l.has = false;
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    return true;
  }

  setGradingParams(params) {
    this.gradingParams = { ...this.gradingParams, ...params };
  }

  setVrParams(params) {
    this.vrParams = { ...this.vrParams, ...params };
  }

  setLut(parsedLut) {
    if (!this.gl || this.kind !== 'webgl2') return;
    const gl = this.gl;
    if (this.lutTexture) {
      gl.deleteTexture(this.lutTexture);
      this.lutTexture = null;
      this.lutInfo = null;
    }
    if (!parsedLut) return;
    const res = uploadLut3DTexture(gl, parsedLut, 2);
    if (res) {
      this.lutTexture = res.texture;
      this.lutInfo = res;
    }
  }

  /** Backing-store size in device pixels. Returns true when it changed. */
  resize(w, h) {
    w = Math.max(1, Math.round(w));
    h = Math.max(1, Math.round(h));
    if (this.canvas.width === w && this.canvas.height === h) return false;
    this.canvas.width = w;
    this.canvas.height = h;
    return true;
  }

  setMode(mode) { this.mode = MODES[mode] !== undefined ? mode : 'single'; }

  /** Wipe position, 0..1 of the viewport width (A is left of it). */
  setSplit(frac) { this.split = Math.max(0, Math.min(1, Number(frac) || 0)); }

  setLayer(i, src, { owned = false } = {}) {
    const [w, h] = sourceSize(src);
    const layer = this.layers[i];
    if (!w || !h) {
      if (owned) src?.close?.();
      return null;
    }
    if (this.gl && !this.lost) {
      const gl = this.gl;
      const back = 1 - this.front[i];
      const slot = this.tex[i][back];
      gl.bindTexture(gl.TEXTURE_2D, slot.tex);
      try {
        if (slot.w === w && slot.h === h) {
          gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, src);
        } else {
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
          slot.w = w;
          slot.h = h;
        }
        this.front[i] = back;
        layer.has = true;
      } catch {
        // A video with no decoded frame yet, or a cross-origin taint.
      }
      if (owned) src.close?.();
    } else {
      if (layer.owned2d && layer.img2d && layer.img2d !== src) layer.img2d.close?.();
      layer.img2d = src;
      layer.owned2d = owned;
      layer.has = true;
    }
    layer.w = w;
    layer.h = h;
    return [w, h];
  }

  clearLayer(i) {
    const layer = this.layers[i];
    if (layer.owned2d && layer.img2d) layer.img2d.close?.();
    layer.img2d = null;
    layer.owned2d = false;
    layer.has = false;
  }

  _boxes() {
    const W = this.canvas.width;
    const H = this.canvas.height;
    const [a, b] = this.layers;
    if (this.mode === 'side') {
      return [fitContain(a.w, a.h, [0, 0, 0.5, 1], W, H), fitContain(b.w, b.h, [0.5, 0, 0.5, 1], W, H)];
    }
    return [fitContain(a.w, a.h, [0, 0, 1, 1], W, H), fitContain(b.w, b.h, [0, 0, 1, 1], W, H)];
  }

  draw() {
    const W = this.canvas.width;
    const H = this.canvas.height;
    const [boxA, boxB] = this._boxes();
    if (this.gl) {
      if (this.lost) return false;
      const gl = this.gl;
      gl.viewport(0, 0, W, H);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(this.prog);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
      gl.enableVertexAttribArray(this.loc.pos);
      gl.vertexAttribPointer(this.loc.pos, 2, gl.FLOAT, false, 0, 0);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.tex[0][this.front[0]].tex);
      gl.uniform1i(this.loc.texA, 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.tex[1][this.front[1]].tex);
      gl.uniform1i(this.loc.texB, 1);
      gl.uniform4f(this.loc.boxA, ...boxA);
      gl.uniform4f(this.loc.boxB, ...boxB);
      gl.uniform1f(this.loc.hasA, this.layers[0].has ? 1 : 0);
      gl.uniform1f(this.loc.hasB, this.layers[1].has ? 1 : 0);
      gl.uniform1f(this.loc.mode, MODES[this.mode]);
      gl.uniform1f(this.loc.split, this.split);

      if (this.kind === 'webgl2') {
        const p = this.gradingParams;
        if (this.lutTexture && this.lutInfo) {
          gl.activeTexture(gl.TEXTURE2);
          gl.bindTexture(gl.TEXTURE_3D, this.lutTexture);
          gl.uniform1i(this.loc.lut3d, 2);
          gl.uniform1f(this.loc.hasLut, 1.0);
          gl.uniform1f(this.loc.lutOpacity, Number(p.lut_opacity ?? 1.0));
          gl.uniform1f(this.loc.lutSize, this.lutInfo.size);
          gl.uniform3fv(this.loc.domainMin, this.lutInfo.domainMin);
          gl.uniform3fv(this.loc.domainMax, this.lutInfo.domainMax);
        } else {
          gl.uniform1f(this.loc.hasLut, 0.0);
        }
        gl.uniform1f(this.loc.temperature, Number(p.color_temperature || 0));
        gl.uniform1f(this.loc.tint, Number(p.color_tint || 0));
        gl.uniform1f(this.loc.vibrance, Number(p.color_vibrance || 0));
        gl.uniform1f(this.loc.skinProtect, Number(p.color_skin_protect ?? 1.0));
        const lm = Number(p.lift_m || 0);
        const gm = Number(p.gamma_m ?? 1.0);
        const gnm = Number(p.gain_m ?? 1.0);
        gl.uniform3f(this.loc.lift, Number(p.lift_r || 0) + lm, Number(p.lift_g || 0) + lm, Number(p.lift_b || 0) + lm);
        gl.uniform3f(this.loc.gamma, Math.max(0.01, Number(p.gamma_r ?? 1.0) * gm), Math.max(0.01, Number(p.gamma_g ?? 1.0) * gm), Math.max(0.01, Number(p.gamma_b ?? 1.0) * gm));
        gl.uniform3f(this.loc.gain, Math.max(0.0, Number(p.gain_r ?? 1.0) * gnm), Math.max(0.0, Number(p.gain_g ?? 1.0) * gnm), Math.max(0.0, Number(p.gain_b ?? 1.0) * gnm));
        gl.uniform1f(this.loc.premultiply, 1.0);

        // Stage 21 VR & Stereo Uniforms
        const vr = this.vrParams;
        const vrModeVal = vr.vrMode === 'vr360' ? 2.0 : (vr.vrMode === 'vr180' ? 1.0 : 0.0);
        const stereoModeMap = {
          off: 0.0,
          left: 1.0,
          right: 2.0,
          side_by_side: 3.0,
          anaglyph: 4.0,
          disparity_diff: 5.0,
        };
        const stereoModeVal = stereoModeMap[vr.stereoMode] ?? 0.0;
        const stereoLayoutMap = { auto: 0.0, sbs: 1.0, tb: 2.0, dual: 3.0 };
        const stereoLayoutVal = stereoLayoutMap[vr.stereoLayout] ?? 1.0;

        gl.uniform1f(this.loc.vrMode, vrModeVal);
        gl.uniform1f(this.loc.yaw, Number(vr.yaw || 0));
        gl.uniform1f(this.loc.pitch, Number(vr.pitch || 0));
        gl.uniform1f(this.loc.fov, Number((vr.fov || 90) * Math.PI / 180));
        gl.uniform1f(this.loc.aspect, W / Math.max(1, H));
        gl.uniform1f(this.loc.stereoMode, stereoModeVal);
        gl.uniform1f(this.loc.stereoLayout, stereoLayoutVal);
      }

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      return true;
    }
    const ctx = this.ctx2d;
    if (!ctx) return false;
    ctx.clearRect(0, 0, W, H);
    const put = (layer, box, clip) => {
      if (!layer.has || !layer.img2d) return;
      ctx.save();
      if (clip) { ctx.beginPath(); ctx.rect(clip[0], 0, clip[1], H); ctx.clip(); }
      ctx.drawImage(layer.img2d, box[0] * W, box[1] * H, box[2] * W, box[3] * H);
      ctx.restore();
    };
    const [a, b] = this.layers;
    if (this.mode === 'wipe') {
      const x = this.split * W;
      put(a, boxA, [0, x]);
      put(b, boxB, [x, W - x]);
    } else {
      put(a, boxA);
      if (this.mode === 'side') put(b, boxB);
    }
    return true;
  }

  destroy() {
    for (let i = 0; i < 2; i++) this.clearLayer(i);
    if (this.gl && !this.lost) {
      const gl = this.gl;
      if (this.lutTexture) gl.deleteTexture(this.lutTexture);
      for (const pair of this.tex || []) for (const t of pair) gl.deleteTexture(t.tex);
      if (this.vbo) gl.deleteBuffer(this.vbo);
      if (this.prog) gl.deleteProgram(this.prog);
    }
    if (this._onLost && typeof this.canvas.removeEventListener === 'function') {
      this.canvas.removeEventListener('webglcontextlost', this._onLost);
      this.canvas.removeEventListener('webglcontextrestored', this._onRestored);
    }
    this.gl = null;
    this.ctx2d = null;
  }
}
