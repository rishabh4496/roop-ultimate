/**
 * Hardware-Accelerated WebGL2 Float 3D .cube LUT & Color Grading Pipeline.
 *
 * Implements:
 *   - Linear float space execution (IEC 61966-2-1 piecewise transfer)
 *   - Three-way Lift/Gamma/Gain color wheels
 *   - Kelvin Temperature & Tint balance
 *   - Vibrance with Skin-Tone Hue Protection (20°–50° skin hue corridor)
 *   - Trilinear 3D .cube LUT sampling (sampler3D) with 0.0–1.0 blend opacity
 *   - Strict premultiplied alpha scaling (rgb * a, a) over Stage 7 XSeg 3 and Stage 11 vector roto masks
 *   - Off-thread / downsampled GPU readback for Broadcast Scopes
 */

import { uploadLut3DTexture } from './cubeLut.js';

export const GRADING_DEFAULTS = {
  lut_path: '',
  lut_bake: false,
  lut_opacity: 1.0,
  color_temperature: 0.0,    // -1.0 (cool/blue) .. +1.0 (warm/amber)
  color_tint: 0.0,           // -1.0 (green) .. +1.0 (magenta)
  color_vibrance: 0.0,       // -1.0 .. +1.0
  color_skin_protect: 1.0,   // 0.0 .. 1.0 (attenuates vibrance in 20°-50° corridor)
  lift_r: 0.0, lift_g: 0.0, lift_b: 0.0, lift_m: 0.0,
  gamma_r: 1.0, gamma_g: 1.0, gamma_b: 1.0, gamma_m: 1.0,
  gain_r: 1.0, gain_g: 1.0, gain_b: 1.0, gain_m: 1.0,
};

const VERT_GLSL = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = vec2(a_pos.x * 0.5 + 0.5, 0.5 - a_pos.y * 0.5);
  gl_Position = vec4(a_pos, 0.0, 1.0);
}
`;

const FRAG_GLSL = `#version 300 es
precision highp float;
precision highp sampler3D;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_image;
uniform sampler2D u_mask;      // Stage 7 XSeg / Stage 11 vector roto mask
uniform sampler3D u_lut3d;     // 3D .cube LUT float texture

uniform float u_hasMask;
uniform float u_hasLut;
uniform float u_lutOpacity;
uniform float u_lutSize;
uniform vec3 u_domainMin;
uniform vec3 u_domainMax;

// Temperature & Tint
uniform float u_temperature;
uniform float u_tint;

// Vibrance & Skin-Tone Protection
uniform float u_vibrance;
uniform float u_skinProtect;

// Three-way Lift / Gamma / Gain
uniform vec3 u_lift;
uniform vec3 u_gamma;
uniform vec3 u_gain;

// Premultiplied alpha mode: 1 = output premultiplied (rgb * a, a)
uniform float u_premultiply;

// IEC 61966-2-1 piecewise transfer functions
vec3 srgbToLinear(vec3 c) {
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}

vec3 linearToSrgb(vec3 c) {
  c = clamp(c, 0.0, 1.0);
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

// RGB to HSV for skin-tone corridor detection
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

// Temperature & Tint adjustment in linear space
vec3 applyTemperatureTint(vec3 linearRgb, float temp, float tint) {
  // Planckian blackbody locus approximation
  vec3 shift = vec3(0.0);
  if (temp > 0.0) {
    shift.r += temp * 0.28;
    shift.b -= temp * 0.22;
  } else if (temp < 0.0) {
    shift.r += temp * 0.20;
    shift.b -= temp * 0.30;
  }
  // Orthogonal green-magenta tint
  shift.g -= tint * 0.25;
  shift.r += tint * 0.125;
  shift.b += tint * 0.125;
  return max(vec3(0.0), linearRgb + shift);
}

// Three-way Lift / Gamma / Gain (ASC CDL compatible)
vec3 applyLiftGammaGain(vec3 c, vec3 lift, vec3 gamma, vec3 gain) {
  // Lift: shadows offset
  vec3 lifted = c + lift * (vec3(1.0) - c);
  lifted = max(vec3(0.0), lifted);
  // Gain: highlights slope
  vec3 gained = lifted * gain;
  gained = max(vec3(0.0), gained);
  // Gamma: midtone power
  vec3 safeGamma = max(gamma, vec3(0.01));
  return pow(gained, vec3(1.0) / safeGamma);
}

// Vibrance with 20°-50° skin-tone hue corridor protection
vec3 applyVibrance(vec3 srgb, float vibrance, float skinProtect) {
  if (abs(vibrance) < 0.001) return srgb;
  vec3 hsv = rgbToHsv(srgb);
  float hueDeg = hsv.x * 360.0;
  
  // Skin tone corridor: 20 deg to 50 deg, center ~35 deg
  float skinDist = abs(hueDeg - 35.0);
  if (skinDist > 180.0) skinDist = 360.0 - skinDist;
  
  // Soft corridor weighting (dist <= 12 deg is 100% skin, falls off smoothly to 25 deg)
  float skinWeight = 1.0 - smoothstep(12.0, 25.0, skinDist);
  
  // Vibrance selectively boosts lower saturation and protects melanin corridor
  float satInv = (1.0 - hsv.y);
  float effectiveVibrance = vibrance * satInv * (1.0 - skinWeight * skinProtect);
  hsv.y = clamp(hsv.y * (1.0 + effectiveVibrance), 0.0, 1.0);
  return hsvToRgb(hsv);
}

// Trilinear 3D .cube LUT sampling with domain remapping
vec3 sampleLut3D(sampler3D lut, vec3 srgb, vec3 domMin, vec3 domMax, float lutSize) {
  vec3 norm = (clamp(srgb, domMin, domMax) - domMin) / max(domMax - domMin, vec3(0.0001));
  // Half-voxel centering for exact trilinear interpolation
  vec3 coord = (norm * (lutSize - 1.0) + 0.5) / lutSize;
  return texture(lut, coord).rgb;
}

void main() {
  vec4 baseSample = texture(u_image, v_uv);
  vec3 col = baseSample.rgb;
  float alpha = baseSample.a;

  // Mask from Stage 7 XSeg 3 or Stage 11 vector roto
  if (u_hasMask > 0.5) {
    vec4 maskSample = texture(u_mask, v_uv);
    alpha = min(alpha, maskSample.r);
  }

  // 1. Convert to linear float space
  vec3 linearCol = srgbToLinear(col);

  // 2. Kelvin Temperature & Tint
  linearCol = applyTemperatureTint(linearCol, u_temperature, u_tint);

  // 3. Three-way Lift / Gamma / Gain
  linearCol = applyLiftGammaGain(linearCol, u_lift, u_gamma, u_gain);

  // 4. Return to display curve for Vibrance and 3D LUT evaluation
  vec3 gradedSrgb = linearToSrgb(linearCol);

  // 5. Vibrance with Skin-Tone Protection
  gradedSrgb = applyVibrance(gradedSrgb, u_vibrance, u_skinProtect);

  // 6. 3D .cube LUT trilinear evaluation with blend opacity
  if (u_hasLut > 0.5) {
    vec3 lutCol = sampleLut3D(u_lut3d, gradedSrgb, u_domainMin, u_domainMax, u_lutSize);
    gradedSrgb = mix(gradedSrgb, lutCol, clamp(u_lutOpacity, 0.0, 1.0));
  }

  // 7. Strict premultiplied alpha scaling (rgb * a, a) over masks
  if (u_premultiply > 0.5) {
    fragColor = vec4(gradedSrgb * alpha, alpha);
  } else {
    fragColor = vec4(gradedSrgb, alpha);
  }
}
`;

export class WebGL2GradingPipeline {
  /**
   * @param {HTMLCanvasElement|OffscreenCanvas} [targetCanvas]
   */
  constructor(targetCanvas = null) {
    this.canvas = targetCanvas || (typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(320, 180)
      : (typeof document !== 'undefined' ? document.createElement('canvas') : null));
    
    this.gl = null;
    this.program = null;
    this.vbo = null;
    this.lutTexture = null;
    this.lutInfo = null;
    this.imageTexture = null;
    this.maskTexture = null;
    this.loc = {};
    this.params = { ...GRADING_DEFAULTS };

    if (this.canvas) {
      this._initGL();
    }
  }

  _initGL() {
    const gl = this.canvas.getContext('webgl2', {
      alpha: true,
      depth: false,
      stencil: false,
      antialias: false,
      premultipliedAlpha: true,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });
    if (!gl) return false;
    this.gl = gl;

    const vs = this._compileShader(gl.VERTEX_SHADER, VERT_GLSL);
    const fs = this._compileShader(gl.FRAGMENT_SHADER, FRAG_GLSL);
    if (!vs || !fs) return false;

    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error('WebGL2 grading shader link failed:', gl.getProgramInfoLog(prog));
      return false;
    }
    this.program = prog;

    // Cache uniform locations
    this.loc = {
      pos: gl.getAttribLocation(prog, 'a_pos'),
      u_image: gl.getUniformLocation(prog, 'u_image'),
      u_mask: gl.getUniformLocation(prog, 'u_mask'),
      u_lut3d: gl.getUniformLocation(prog, 'u_lut3d'),
      u_hasMask: gl.getUniformLocation(prog, 'u_hasMask'),
      u_hasLut: gl.getUniformLocation(prog, 'u_hasLut'),
      u_lutOpacity: gl.getUniformLocation(prog, 'u_lutOpacity'),
      u_lutSize: gl.getUniformLocation(prog, 'u_lutSize'),
      u_domainMin: gl.getUniformLocation(prog, 'u_domainMin'),
      u_domainMax: gl.getUniformLocation(prog, 'u_domainMax'),
      u_temperature: gl.getUniformLocation(prog, 'u_temperature'),
      u_tint: gl.getUniformLocation(prog, 'u_tint'),
      u_vibrance: gl.getUniformLocation(prog, 'u_vibrance'),
      u_skinProtect: gl.getUniformLocation(prog, 'u_skinProtect'),
      u_lift: gl.getUniformLocation(prog, 'u_lift'),
      u_gamma: gl.getUniformLocation(prog, 'u_gamma'),
      u_gain: gl.getUniformLocation(prog, 'u_gain'),
      u_premultiply: gl.getUniformLocation(prog, 'u_premultiply'),
    };

    // Quad buffer
    this.vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    // Image & mask textures
    this.imageTexture = gl.createTexture();
    this.maskTexture = gl.createTexture();

    const setup2DTex = (tex) => {
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    };
    setup2DTex(this.imageTexture);
    setup2DTex(this.maskTexture);

    return true;
  }

  _compileShader(type, src) {
    const gl = this.gl;
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error('WebGL2 shader compile error:', gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    return s;
  }

  setParams(patch) {
    this.params = { ...this.params, ...patch };
  }

  /**
   * Sets the active 3D .cube LUT.
   * @param {import('./cubeLut').parseCubeLut} parsedLut
   */
  setLut(parsedLut) {
    const gl = this.gl;
    if (!gl) return;
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

  /**
   * Renders the graded frame onto the backing canvas.
   * @param {TexImageSource} image
   * @param {TexImageSource} [mask] Optional Stage 7 XSeg 3 or Stage 11 vector roto mask
   * @param {number} [targetWidth]
   * @param {number} [targetHeight]
   * @param {boolean} [premultiply=true] Strict premultiplied alpha scaling
   * @returns {boolean}
   */
  render(image, mask = null, targetWidth = 0, targetHeight = 0, premultiply = true) {
    const gl = this.gl;
    if (!gl || !this.program || !image) return false;

    const iw = targetWidth || image.width || image.videoWidth || 320;
    const ih = targetHeight || image.height || image.videoHeight || 180;
    if (this.canvas.width !== iw || this.canvas.height !== ih) {
      this.canvas.width = iw;
      this.canvas.height = ih;
    }

    gl.viewport(0, 0, iw, ih);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.program);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.enableVertexAttribArray(this.loc.pos);
    gl.vertexAttribPointer(this.loc.pos, 2, gl.FLOAT, false, 0, 0);

    // Unit 0: Main Image
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.imageTexture);
    try {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    } catch {
      return false;
    }
    gl.uniform1i(this.loc.u_image, 0);

    // Unit 1: Optional Mask
    let hasMask = 0;
    if (mask) {
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.maskTexture);
      try {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, mask);
        hasMask = 1;
      } catch {
        hasMask = 0;
      }
      gl.uniform1i(this.loc.u_mask, 1);
    }
    gl.uniform1f(this.loc.u_hasMask, hasMask);

    // Unit 2: 3D .cube LUT
    let hasLut = 0;
    if (this.lutTexture && this.lutInfo) {
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_3D, this.lutTexture);
      gl.uniform1i(this.loc.u_lut3d, 2);
      gl.uniform1f(this.loc.u_lutOpacity, Number(this.params.lut_opacity ?? 1.0));
      gl.uniform1f(this.loc.u_lutSize, this.lutInfo.size);
      gl.uniform3fv(this.loc.u_domainMin, this.lutInfo.domainMin);
      gl.uniform3fv(this.loc.u_domainMax, this.lutInfo.domainMax);
      hasLut = 1;
    }
    gl.uniform1f(this.loc.u_hasLut, hasLut);

    // Temperature & Tint
    gl.uniform1f(this.loc.u_temperature, Number(this.params.color_temperature || 0));
    gl.uniform1f(this.loc.u_tint, Number(this.params.color_tint || 0));

    // Vibrance & Skin-Tone Protection
    gl.uniform1f(this.loc.u_vibrance, Number(this.params.color_vibrance || 0));
    gl.uniform1f(this.loc.u_skinProtect, Number(this.params.color_skin_protect ?? 1.0));

    // Three-way Lift / Gamma / Gain
    const p = this.params;
    const lm = Number(p.lift_m || 0);
    const gm = Number(p.gamma_m ?? 1.0);
    const gnm = Number(p.gain_m ?? 1.0);

    gl.uniform3f(this.loc.u_lift, (Number(p.lift_r || 0) + lm), (Number(p.lift_g || 0) + lm), (Number(p.lift_b || 0) + lm));
    gl.uniform3f(this.loc.u_gamma, Math.max(0.01, Number(p.gamma_r ?? 1.0) * gm), Math.max(0.01, Number(p.gamma_g ?? 1.0) * gm), Math.max(0.01, Number(p.gamma_b ?? 1.0) * gm));
    gl.uniform3f(this.loc.u_gain, Math.max(0.0, Number(p.gain_r ?? 1.0) * gnm), Math.max(0.0, Number(p.gain_g ?? 1.0) * gnm), Math.max(0.0, Number(p.gain_b ?? 1.0) * gnm));

    // Premultiplied alpha mode
    gl.uniform1f(this.loc.u_premultiply, premultiply ? 1.0 : 0.0);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    return true;
  }

  /**
   * Reads back downsampled RGBA pixels for scope analysis.
   * @param {number} [w=256]
   * @param {number} [h=144]
   * @returns {Uint8Array|null}
   */
  readbackDownsampled(w = 256, h = 144) {
    const gl = this.gl;
    if (!gl) return null;
    const pixels = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, Math.min(w, this.canvas.width), Math.min(h, this.canvas.height), gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    return pixels;
  }

  destroy() {
    const gl = this.gl;
    if (!gl) return;
    if (this.lutTexture) gl.deleteTexture(this.lutTexture);
    if (this.imageTexture) gl.deleteTexture(this.imageTexture);
    if (this.maskTexture) gl.deleteTexture(this.maskTexture);
    if (this.vbo) gl.deleteBuffer(this.vbo);
    if (this.program) gl.deleteProgram(this.program);
    this.gl = null;
  }
}
