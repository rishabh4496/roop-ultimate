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
//   'single' — A, fitted `contain` into the whole viewport
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
// FALLBACK
// WebGL2 -> WebGL1 -> Canvas 2D. The 2D path draws the same three modes with
// drawImage + a clip rect; it just cannot upload asynchronously. Context loss
// (the GPU process resetting, which Chromium does under memory pressure) is
// survived: the next frame re-creates the resources and is drawn normally.

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

const MODES = { single: 0, wipe: 1, side: 2 };

const sourceSize = (src) => {
  if (!src) return [0, 0];
  if (typeof src.videoWidth === 'number' && src.videoWidth) return [src.videoWidth, src.videoHeight];
  if (typeof src.naturalWidth === 'number' && src.naturalWidth) return [src.naturalWidth, src.naturalHeight];
  if (typeof src.displayWidth === 'number' && src.displayWidth) return [src.displayWidth, src.displayHeight];
  return [src.width || 0, src.height || 0];
};

/** `object-fit: contain` of a (iw x ih) picture into a region, as fractions. */
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
      // A preview must never be the reason the render's GPU gets clocked up.
      // (Not `desynchronized`: the stage has DOM overlays — face boxes, the
      // compare handle — stacked on this canvas, and a low-latency canvas
      // trades tearing for latency the preview does not need.)
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
    const vs = compile(gl.VERTEX_SHADER, VERT);
    const fs = compile(gl.FRAGMENT_SHADER, FRAG);
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
    // One triangle strip covering the viewport.
    this.vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const makeTex = () => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      // NPOT-safe on WebGL1: no mipmaps, clamp.
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

  /**
   * Put a picture on a layer. For an ImageBitmap passed with `owned`, the
   * renderer takes ownership and closes it as soon as it is no longer needed
   * (right after the upload on WebGL; on replacement for 2D, which draws from
   * it). A video or image element is never closed.
   * Returns [width, height] of the picture, or null if it had no size yet.
   */
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
        // A video with no decoded frame yet, or a cross-origin taint. Keep the
        // previous picture rather than showing a hole.
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
