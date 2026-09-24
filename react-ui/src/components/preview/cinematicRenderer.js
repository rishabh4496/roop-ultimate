// ── CinematicRenderer: High-Performance WebGL 2.0 Engine ───────────────────
//
// Double-buffered GPU rendering engine built for roop-ultimate.
// Supports:
//   - WebGL 2.0 context with hardware context loss recovery
//   - Double-buffered textures per channel (Target, Swap, Mask) for zero-tear swaps
//   - Direct binary frame packet ingestion (raw RGBA buffers or WebP/JPEG blobs)
//   - GLSL 300 es fragment modes:
//       Mode 0: Standard (Target or Swap)
//       Mode 1: A/B Split Screen (vertical/horizontal with 2px boundary line)
//       Mode 2: Difference Heatmap (Turbo / Inferno colormaps)
//       Mode 3: Mask Channel Alpha Blend (BiSeNet skin/occluder overlay)
//   - Screen-space matrix transformations for pan/zoom in vertex shader
//   - gl.LINEAR (smooth) vs gl.NEAREST (pixel-peeping inspection)
//   - Color-under-cursor inspection with gl.readPixels

import { parseFrameMessage } from '../../transport/frameProtocol.js';

const VS_SOURCE = `#version 300 es
precision highp float;

layout(location = 0) in vec2 a_position; // Quad [-1, 1]
layout(location = 1) in vec2 a_texCoord; // UV [0, 1]

uniform mat3 u_transform;

out vec2 v_uv;

void main() {
    v_uv = a_texCoord;
    vec3 clipPos = u_transform * vec3(a_position, 1.0);
    gl_Position = vec4(clipPos.xy, 0.0, 1.0);
}
`;

const FS_SOURCE = `#version 300 es
precision highp float;

in vec2 v_uv;

uniform sampler2D u_texA;    // Target
uniform sampler2D u_texB;    // Swap
uniform sampler2D u_texMask; // BiSeNet Occlusion Mask

uniform int u_mode;             // 0: Standard, 1: A/B Split, 2: Difference, 3: Mask Blend
uniform int u_standardLayer;    // 0: Target (A), 1: Swap (B)
uniform float u_splitRatio;     // 0.0 to 1.0
uniform int u_splitOrientation; // 0: Vertical, 1: Horizontal
uniform float u_diffGain;       // Heatmap gain multiplier (e.g. 1.0 - 10.0)
uniform int u_colormap;         // 0: Turbo, 1: Inferno
uniform float u_maskOpacity;    // 0.0 to 1.0
uniform vec3 u_skinColor;       // Red tint for swapped skin
uniform vec3 u_occlusionColor;  // Emerald green tint for preserved hair/occluders
uniform float u_hasA;
uniform float u_hasB;
uniform float u_hasMask;

out vec4 fragColor;

// Google Turbo Colormap (7th-degree polynomial approximation)
vec3 colormapTurbo(float x) {
    x = clamp(x, 0.0, 1.0);
    const vec4 kRedVec4   = vec4(0.13572138,  4.61539260, -42.66032258,  132.13108234);
    const vec4 kGreenVec4 = vec4(0.09140261,  2.19418839,   4.84296658,  -14.18503333);
    const vec4 kBlueVec4  = vec4(0.10667330, 12.64194608, -60.58204836,  110.36276771);
    const vec2 kRedVec2   = vec2(-152.94239396,  59.28637943);
    const vec2 kGreenVec2 = vec2(   4.27729857,   2.82956604);
    const vec2 kBlueVec2  = vec2( -89.90310912,  27.34824973);

    vec4 v4 = vec4(1.0, x, x * x, x * x * x);
    vec2 v2 = v4.zw * v4.z;
    return clamp(vec3(
        dot(v4, kRedVec4)   + dot(v2, kRedVec2),
        dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
        dot(v4, kBlueVec4)  + dot(v2, kBlueVec2)
    ), 0.0, 1.0);
}

// Matplotlib Inferno Colormap (5th-degree polynomial approximation)
vec3 colormapInferno(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.001462, 0.000466, 0.013866);
    vec3 c1 = vec3(0.340348, 0.053896, 0.627764);
    vec3 c2 = vec3(3.948293, 0.252069, -0.669862);
    vec3 c3 = vec3(-8.825227, 4.437340, 2.128796);
    vec3 c4 = vec3(10.198945, -7.868770, -3.957262);
    vec3 c5 = vec3(-4.654817, 4.129379, 1.857738);
    return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))), 0.0, 1.0);
}

void main() {
    // Bounds guard: transparent outside [0, 1]
    if (v_uv.x < 0.0 || v_uv.x > 1.0 || v_uv.y < 0.0 || v_uv.y > 1.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    if (u_mode == 0) {
        // Mode 0: Standard
        if (u_standardLayer == 0) {
            fragColor = (u_hasA > 0.5) ? texture(u_texA, v_uv) : vec4(0.0, 0.0, 0.0, 1.0);
        } else {
            fragColor = (u_hasB > 0.5) ? texture(u_texB, v_uv) : vec4(0.0, 0.0, 0.0, 1.0);
        }
    } else if (u_mode == 1) {
        // Mode 1: A/B Split Screen
        float coord = (u_splitOrientation == 0) ? v_uv.x : v_uv.y;
        float dUv = fwidth(coord);
        float pxDist = abs(coord - u_splitRatio) / max(dUv, 1e-6);

        // 2px boundary line (pxDist <= 1.0 means 1px on either side of the exact split)
        if (pxDist <= 1.0) {
            fragColor = vec4(1.0, 1.0, 1.0, 1.0);
            return;
        }
        // Subtle 1px dark halo for contrast against light or white scenes
        if (pxDist <= 2.0) {
            vec4 base = (coord < u_splitRatio)
                ? ((u_hasA > 0.5) ? texture(u_texA, v_uv) : vec4(0.0, 0.0, 0.0, 1.0))
                : ((u_hasB > 0.5) ? texture(u_texB, v_uv) : vec4(0.0, 0.0, 0.0, 1.0));
            fragColor = mix(base, vec4(0.0, 0.0, 0.0, 0.75), (2.0 - pxDist) * 0.6);
            return;
        }

        if (coord < u_splitRatio) {
            fragColor = (u_hasA > 0.5) ? texture(u_texA, v_uv) : vec4(0.0, 0.0, 0.0, 1.0);
        } else {
            fragColor = (u_hasB > 0.5) ? texture(u_texB, v_uv) : vec4(0.0, 0.0, 0.0, 1.0);
        }
    } else if (u_mode == 2) {
        // Mode 2: Difference Heatmap
        if (u_hasA < 0.5 || u_hasB < 0.5) {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }
        vec4 colA = texture(u_texA, v_uv);
        vec4 colB = texture(u_texB, v_uv);
        vec3 diff = abs(colA.rgb - colB.rgb) * u_diffGain;
        float metric = clamp(max(max(diff.r, diff.g), diff.b), 0.0, 1.0);
        vec3 heatColor = (u_colormap == 0) ? colormapTurbo(metric) : colormapInferno(metric);
        fragColor = vec4(heatColor, 1.0);
    } else if (u_mode == 3) {
        // Mode 3: Mask Channel Alpha Blend (BiSeNet Occlusion Mask)
        vec4 base = (u_standardLayer == 0 && u_hasA > 0.5) ? texture(u_texA, v_uv) :
                    (u_hasB > 0.5) ? texture(u_texB, v_uv) :
                    (u_hasA > 0.5) ? texture(u_texA, v_uv) : vec4(0.08, 0.08, 0.1, 1.0);

        if (u_hasMask < 0.5) {
            fragColor = base;
            return;
        }

        vec4 mask = texture(u_texMask, v_uv);
        vec3 overlayColor = vec3(0.0);
        float overlayAlpha = 0.0;

        float channelDelta = abs(mask.r - mask.g) + abs(mask.g - mask.b);
        if (channelDelta > 0.05) {
            // Multi-channel BiSeNet mask:
            // mask.r = swapped skin region -> Red
            // mask.g = preserved occluders / hair -> Emerald Green
            vec3 skinContrib = u_skinColor * mask.r;
            vec3 occlContrib = u_occlusionColor * mask.g;
            float total = mask.r + mask.g;
            if (total > 0.001) {
                overlayColor = (skinContrib + occlContrib) / total;
                overlayAlpha = clamp(total, 0.0, 1.0) * u_maskOpacity;
            }
        } else {
            // Grayscale BiSeNet mask:
            // High values = swapped skin (Red), boundary / edge = occluder (Emerald Green)
            float val = mask.r;
            if (val > 0.35) {
                overlayColor = u_skinColor;
                overlayAlpha = val * u_maskOpacity;
            } else if (val > 0.04) {
                overlayColor = u_occlusionColor;
                overlayAlpha = (val / 0.35) * u_maskOpacity;
            }
        }

        fragColor = vec4(mix(base.rgb, overlayColor, overlayAlpha), 1.0);
    } else {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
`;

/** Double-buffered WebGL texture slot. Uploads go to back texture, then ping-pong. */
export class DoubleBufferedTexture {
  constructor(gl) {
    this.gl = gl;
    this.textures = [gl.createTexture(), gl.createTexture()];
    this.front = 0;
    this.width = 0;
    this.height = 0;
    this.hasData = false;
    this.filter = gl.LINEAR;
    this._initParams();
  }

  _initParams() {
    const gl = this.gl;
    for (const tex of this.textures) {
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, this.filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, this.filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }
  }

  setFilter(filter) {
    this.filter = filter;
    const gl = this.gl;
    for (const tex of this.textures) {
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    }
  }

  getFront() {
    return this.textures[this.front];
  }

  getBack() {
    return this.textures[1 - this.front];
  }

  uploadImageSource(source) {
    const gl = this.gl;
    const w = source.width || source.videoWidth || source.naturalWidth;
    const h = source.height || source.videoHeight || source.naturalHeight;
    if (!w || !h) return false;

    const back = this.getBack();
    gl.bindTexture(gl.TEXTURE_2D, back);
    if (this.width === w && this.height === h) {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, source);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
      this.width = w;
      this.height = h;
    }
    // Swap buffer
    this.front = 1 - this.front;
    this.hasData = true;
    return true;
  }

  uploadRgbaBuffer(buffer, width, height) {
    const gl = this.gl;
    if (!width || !height) return false;
    const data = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);

    const back = this.getBack();
    gl.bindTexture(gl.TEXTURE_2D, back);
    if (this.width === width && this.height === height) {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, data);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
      this.width = width;
      this.height = height;
    }
    // Swap buffer
    this.front = 1 - this.front;
    this.hasData = true;
    return true;
  }

  clear() {
    this.hasData = false;
    this.width = 0;
    this.height = 0;
  }

  destroy() {
    for (const tex of this.textures) {
      this.gl.deleteTexture(tex);
    }
    this.hasData = false;
  }
}

/**
 * Image MIME type from a payload's magic bytes, or null for anything else
 * (raw pixels). WebP = 'RIFF' .... 'WEBP', JPEG = FF D8, PNG = 89 'PNG'.
 */
export function sniffImageMime(u8) {
  if (u8.length >= 12 &&
      u8[0] === 0x52 && u8[1] === 0x49 && u8[2] === 0x46 && u8[3] === 0x46 &&
      u8[8] === 0x57 && u8[9] === 0x45 && u8[10] === 0x42 && u8[11] === 0x50) return 'image/webp';
  if (u8.length >= 4 && u8[0] === 0x89 && u8[1] === 0x50 && u8[2] === 0x4e && u8[3] === 0x47) return 'image/png';
  if (u8.length >= 2 && u8[0] === 0xff && u8[1] === 0xd8) return 'image/jpeg';
  return null;
}

/**
 * Pan that keeps the image point under the cursor fixed while zoom changes
 * from curZoom to nextZoom. (dx, dy) is the cursor offset from the viewport
 * centre, in the same pixel units as pan.
 */
export function anchoredPan(pan, dx, dy, curZoom, nextZoom) {
  const alpha = nextZoom / curZoom;
  return {
    x: dx - (dx - pan.x) * alpha,
    y: dy - (dy - pan.y) * alpha,
  };
}

/** Compute 3x3 affine transformation matrix mapping [-1, 1] quad to clip space. */
export function computeTransformMatrix({
  viewportWidth,
  viewportHeight,
  imageWidth,
  imageHeight,
  zoom = 1.0,
  panX = 0,
  panY = 0,
}) {
  const W = Math.max(1, viewportWidth);
  const H = Math.max(1, viewportHeight);
  const imgW = Math.max(1, imageWidth);
  const imgH = Math.max(1, imageHeight);

  const baseScale = Math.min(W / imgW, H / imgH);
  const dispW = imgW * baseScale * zoom;
  const dispH = imgH * baseScale * zoom;

  const scaleX = dispW / W;
  const scaleY = dispH / H;
  const transX = (2 * panX) / W;
  const transY = (-2 * panY) / H; // Invert Y for clip space

  // Column-major Float32Array for WebGL mat3
  return new Float32Array([
    scaleX, 0,      0,
    0,      scaleY, 0,
    transX, transY, 1,
  ]);
}

/** Pure math colormap helpers for inspection & CPU unit checks */
export function evalTurbo(x) {
  const t = Math.max(0, Math.min(1, x));
  const r = 0.13572138 + 4.6153926 * t - 42.66032258 * t * t + 132.13108234 * t ** 3 - 152.94239396 * t ** 4 + 59.28637943 * t ** 5;
  const g = 0.09140261 + 2.19418839 * t + 4.84296658 * t * t - 14.18503333 * t ** 3 + 4.27729857 * t ** 4 + 2.82956604 * t ** 5;
  const b = 0.1066733 + 12.64194608 * t - 60.58204836 * t * t + 110.36276771 * t ** 3 - 89.90310912 * t ** 4 + 27.34824973 * t ** 5;
  return [Math.max(0, Math.min(1, r)), Math.max(0, Math.min(1, g)), Math.max(0, Math.min(1, b))];
}

export function evalInferno(t) {
  const x = Math.max(0, Math.min(1, t));
  const c0 = [0.001462, 0.000466, 0.013866];
  const c1 = [0.340348, 0.053896, 0.627764];
  const c2 = [3.948293, 0.252069, -0.669862];
  const c3 = [-8.825227, 4.437340, 2.128796];
  const c4 = [10.198945, -7.868770, -3.957262];
  const c5 = [-4.654817, 4.129379, 1.857738];
  const r = c0[0] + x * (c1[0] + x * (c2[0] + x * (c3[0] + x * (c4[0] + x * c5[0]))));
  const g = c0[1] + x * (c1[1] + x * (c2[1] + x * (c3[1] + x * (c4[1] + x * c5[1]))));
  const b = c0[2] + x * (c1[2] + x * (c2[2] + x * (c3[2] + x * (c4[2] + x * c5[2]))));
  return [Math.max(0, Math.min(1, r)), Math.max(0, Math.min(1, g)), Math.max(0, Math.min(1, b))];
}

export class CinematicRenderer {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} [options]
   */
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.options = options;
    this.gl = null;
    this.program = null;
    this.vao = null;
    this.vbo = null;
    this.ibo = null;
    this.lost = false;

    // Double-buffered layers: 0 = Target (A), 1 = Swap (B), 2 = Mask (C)
    this.layers = [];

    // State uniforms
    this.mode = 0;              // 0: Standard, 1: Split, 2: Diff, 3: Mask
    this.standardLayer = 1;     // 0: Target, 1: Swap
    this.splitRatio = 0.5;      // 0.0 to 1.0
    this.splitOrientation = 0;  // 0: Vertical, 1: Horizontal
    this.diffGain = 3.0;        // 1.0 to 10.0
    this.colormap = 0;          // 0: Turbo, 1: Inferno
    this.maskOpacity = 0.55;    // 0.0 to 1.0
    this.skinColor = [0.95, 0.22, 0.25];      // Red
    this.occlusionColor = [0.05, 0.88, 0.45]; // Emerald Green
    this.filterMode = 'linear'; // 'linear' | 'nearest'

    // Pan & Zoom
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;

    // Locations
    this.loc = {};
    this.disposed = false;

    this._initGl();
  }

  _initGl() {
    const attrs = {
      alpha: true,
      antialias: false,
      depth: false,
      stencil: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: true, // Needed for accurate color-under-cursor readPixels
      powerPreference: 'high-performance',
    };

    let gl = null;
    try {
      gl = this.canvas.getContext('webgl2', attrs);
    } catch {
      gl = null;
    }

    if (!gl) {
      console.warn('[CinematicRenderer] WebGL 2.0 unavailable.');
      return false;
    }

    this.gl = gl;

    if (typeof this.canvas.addEventListener === 'function') {
      this._onContextLost = (e) => {
        e.preventDefault();
        this.lost = true;
      };
      this._onContextRestored = () => {
        this.lost = false;
        this._buildPipeline();
      };
      this.canvas.addEventListener('webglcontextlost', this._onContextLost);
      this.canvas.addEventListener('webglcontextrestored', this._onContextRestored);
    }

    return this._buildPipeline();
  }

  _buildPipeline() {
    const gl = this.gl;
    if (!gl) return false;

    // Compile Shaders
    const vs = this._compileShader(gl.VERTEX_SHADER, VS_SOURCE);
    const fs = this._compileShader(gl.FRAGMENT_SHADER, FS_SOURCE);
    if (!vs || !fs) return false;

    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error('[CinematicRenderer] Program link failed:', gl.getProgramInfoLog(prog));
      gl.deleteProgram(prog);
      return false;
    }

    this.program = prog;

    // Cache uniform and attribute locations
    this.loc = {
      transform: gl.getUniformLocation(prog, 'u_transform'),
      texA: gl.getUniformLocation(prog, 'u_texA'),
      texB: gl.getUniformLocation(prog, 'u_texB'),
      texMask: gl.getUniformLocation(prog, 'u_texMask'),
      mode: gl.getUniformLocation(prog, 'u_mode'),
      standardLayer: gl.getUniformLocation(prog, 'u_standardLayer'),
      splitRatio: gl.getUniformLocation(prog, 'u_splitRatio'),
      splitOrientation: gl.getUniformLocation(prog, 'u_splitOrientation'),
      diffGain: gl.getUniformLocation(prog, 'u_diffGain'),
      colormap: gl.getUniformLocation(prog, 'u_colormap'),
      maskOpacity: gl.getUniformLocation(prog, 'u_maskOpacity'),
      skinColor: gl.getUniformLocation(prog, 'u_skinColor'),
      occlusionColor: gl.getUniformLocation(prog, 'u_occlusionColor'),
      hasA: gl.getUniformLocation(prog, 'u_hasA'),
      hasB: gl.getUniformLocation(prog, 'u_hasB'),
      hasMask: gl.getUniformLocation(prog, 'u_hasMask'),
    };

    // Quad geometry: pos(x, y), uv(u, v)
    // Coords: pos [-1, 1], UV [0, 1] with origin at top-left
    const vertices = new Float32Array([
      // pos.x, pos.y, uv.u, uv.v
      -1.0,  1.0, 0.0, 0.0, // Top-Left
       1.0,  1.0, 1.0, 0.0, // Top-Right
      -1.0, -1.0, 0.0, 1.0, // Bottom-Left
       1.0, -1.0, 1.0, 1.0, // Bottom-Right
    ]);

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);

    this.vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    // a_position (loc 0): vec2
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 4 * 4, 0);

    // a_texCoord (loc 1): vec2
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 4 * 4, 2 * 4);

    gl.bindVertexArray(null);

    // Initialize 3 double-buffered texture slots
    this.layers = [
      new DoubleBufferedTexture(gl), // Layer 0: Target (A)
      new DoubleBufferedTexture(gl), // Layer 1: Swap (B)
      new DoubleBufferedTexture(gl), // Layer 2: Mask (C)
    ];

    this.setFilterMode(this.filterMode);
    return true;
  }

  _compileShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error('[CinematicRenderer] Shader compile failed:', gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  setFilterMode(mode) {
    this.filterMode = mode === 'nearest' ? 'nearest' : 'linear';
    if (!this.gl || this.lost) return;
    const filter = this.filterMode === 'nearest' ? this.gl.NEAREST : this.gl.LINEAR;
    for (const layer of this.layers) {
      layer.setFilter(filter);
    }
  }

  setMode(mode) {
    this.mode = Number(mode) || 0;
  }

  setStandardLayer(layer) {
    this.standardLayer = layer === 0 ? 0 : 1;
  }

  setSplitRatio(ratio) {
    this.splitRatio = Math.max(0.0, Math.min(1.0, Number(ratio) || 0.0));
  }

  setSplitOrientation(orientation) {
    this.splitOrientation = orientation === 'horizontal' || orientation === 1 ? 1 : 0;
  }

  setDiffGain(gain) {
    this.diffGain = Math.max(0.1, Math.min(20.0, Number(gain) || 3.0));
  }

  setColormap(colormap) {
    this.colormap = colormap === 'inferno' || colormap === 1 ? 1 : 0;
  }

  setMaskOpacity(opacity) {
    this.maskOpacity = Math.max(0.0, Math.min(1.0, Number(opacity) || 0.5));
  }

  setPanZoom(panX, panY, zoom) {
    this.panX = panX;
    this.panY = panY;
    this.zoom = Math.max(0.1, Math.min(16.0, zoom));
  }

  /** Current active image dimensions from layers */
  getImageDimensions() {
    for (const layer of this.layers) {
      if (layer.hasData && layer.width > 0 && layer.height > 0) {
        return [layer.width, layer.height];
      }
    }
    return [0, 0];
  }

  /**
   * Upload image source (ImageBitmap, HTMLImageElement, HTMLVideoElement, Canvas)
   * into double-buffered layer (0: Target, 1: Swap, 2: Mask).
   */
  uploadImage(layerIndex, source) {
    if (!this.gl || this.lost || !this.layers[layerIndex]) return false;
    return this.layers[layerIndex].uploadImageSource(source);
  }

  /**
   * Upload raw RGBA buffer into double-buffered layer.
   */
  uploadRgba(layerIndex, buffer, width, height) {
    if (!this.gl || this.lost || !this.layers[layerIndex]) return false;
    return this.layers[layerIndex].uploadRgbaBuffer(buffer, width, height);
  }

  /**
   * Ingest a binary frame packet (ArrayBuffer or Blob or wire format packet).
   * Asynchronously decodes WebP / JPEG / PNG or uploads raw RGBA directly to GPU texture.
   */
  async ingestBinaryPacket(layerIndex, data, meta = {}) {
    if (this.disposed || !this.layers[layerIndex]) return false;

    // Case 1: roop-ultimate wire format packet (parseFrameMessage)
    if (data instanceof ArrayBuffer && data.byteLength >= 20) {
      const msg = parseFrameMessage(data);
      if (msg && msg.bytes) {
        return this._ingestPayloadBytes(layerIndex, msg.bytes, msg.width, msg.height);
      }
    }

    // Case 2: Blob (WebP / JPEG / PNG)
    if (typeof Blob !== 'undefined' && data instanceof Blob) {
      try {
        const bitmap = await createImageBitmap(data, {
          imageOrientation: 'none',
          premultiplyAlpha: 'none',
        });
        if (this.disposed) {
          bitmap.close?.();
          return false;
        }
        const ok = this.uploadImage(layerIndex, bitmap);
        bitmap.close?.();
        return ok;
      } catch (err) {
        console.error('[CinematicRenderer] Failed to decode Blob frame:', err);
        return false;
      }
    }

    // Case 3: Raw ArrayBuffer or Uint8Array
    if (data instanceof ArrayBuffer || ArrayBuffer.isView(data)) {
      const bytes = data instanceof Uint8Array ? data : new Uint8Array(data.buffer || data);
      return this._ingestPayloadBytes(layerIndex, bytes, meta.width, meta.height);
    }

    return false;
  }

  async _ingestPayloadBytes(layerIndex, bytes, width, height) {
    if (!bytes || bytes.byteLength === 0) return false;
    const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);

    const mime = sniffImageMime(u8);
    if (mime) {
      try {
        const blob = new Blob([u8], { type: mime });
        const bitmap = await createImageBitmap(blob, {
          imageOrientation: 'none',
          premultiplyAlpha: 'none',
        });
        if (this.disposed) {
          bitmap.close?.();
          return false;
        }
        const ok = this.uploadImage(layerIndex, bitmap);
        bitmap.close?.();
        return ok;
      } catch (err) {
        console.error('[CinematicRenderer] createImageBitmap failed:', err);
        return false;
      }
    }

    // Raw RGBA buffer
    if (width > 0 && height > 0) {
      return this.uploadRgba(layerIndex, u8, width, height);
    }

    return false;
  }

  clearLayer(layerIndex) {
    if (this.layers[layerIndex]) {
      this.layers[layerIndex].clear();
    }
  }

  /**
   * Sample the exact rendered color under the cursor using gl.readPixels.
   * Returns { r, g, b, a, hex, rgb } or null.
   */
  readPixelAt(canvasX, canvasY, dpr = 1) {
    const gl = this.gl;
    if (!gl || this.lost) return null;

    const px = Math.round(canvasX * dpr);
    const py = Math.round((this.canvas.clientHeight - canvasY) * dpr); // WebGL Y is inverted

    if (px < 0 || px >= this.canvas.width || py < 0 || py >= this.canvas.height) {
      return null;
    }

    const pixel = new Uint8Array(4);
    gl.readPixels(px, py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);

    const [r, g, b, a] = pixel;
    const hex = `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1).toUpperCase()}`;
    return {
      r, g, b, a,
      hex,
      rgb: `rgb(${r}, ${g}, ${b})`,
    };
  }

  /** Main WebGL 2.0 draw call */
  draw() {
    const gl = this.gl;
    if (!gl || this.lost || !this.program) return false;

    const W = this.canvas.width;
    const H = this.canvas.height;
    if (W <= 0 || H <= 0) return false;

    gl.viewport(0, 0, W, H);
    gl.clearColor(0.04, 0.04, 0.05, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.program);

    // Compute coordinate transformation matrix for vertex shader
    const [imgW, imgH] = this.getImageDimensions();
    const transform = computeTransformMatrix({
      viewportWidth: W,
      viewportHeight: H,
      imageWidth: imgW || W,
      imageHeight: imgH || H,
      zoom: this.zoom,
      panX: this.panX,
      panY: this.panY,
    });
    gl.uniformMatrix3fv(this.loc.transform, false, transform);

    // Bind double-buffered textures to texture units 0, 1, 2
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.layers[0].getFront());
    gl.uniform1i(this.loc.texA, 0);
    gl.uniform1f(this.loc.hasA, this.layers[0].hasData ? 1.0 : 0.0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.layers[1].getFront());
    gl.uniform1i(this.loc.texB, 1);
    gl.uniform1f(this.loc.hasB, this.layers[1].hasData ? 1.0 : 0.0);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.layers[2].getFront());
    gl.uniform1i(this.loc.texMask, 2);
    gl.uniform1f(this.loc.hasMask, this.layers[2].hasData ? 1.0 : 0.0);

    // Uniforms
    gl.uniform1i(this.loc.mode, this.mode);
    gl.uniform1i(this.loc.standardLayer, this.standardLayer);
    gl.uniform1f(this.loc.splitRatio, this.splitRatio);
    gl.uniform1i(this.loc.splitOrientation, this.splitOrientation);
    gl.uniform1f(this.loc.diffGain, this.diffGain);
    gl.uniform1i(this.loc.colormap, this.colormap);
    gl.uniform1f(this.loc.maskOpacity, this.maskOpacity);
    gl.uniform3fv(this.loc.skinColor, this.skinColor);
    gl.uniform3fv(this.loc.occlusionColor, this.occlusionColor);

    // Render triangle strip
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);

    return true;
  }

  destroy() {
    this.disposed = true;
    for (const layer of this.layers) {
      layer.destroy();
    }
    this.layers = [];

    if (this.gl && !this.lost) {
      const gl = this.gl;
      if (this.vbo) gl.deleteBuffer(this.vbo);
      if (this.vao) gl.deleteVertexArray(this.vao);
      if (this.program) gl.deleteProgram(this.program);
    }

    if (this._onContextLost && typeof this.canvas.removeEventListener === 'function') {
      this.canvas.removeEventListener('webglcontextlost', this._onContextLost);
      this.canvas.removeEventListener('webglcontextrestored', this._onContextRestored);
    }

    this.gl = null;
  }
}
