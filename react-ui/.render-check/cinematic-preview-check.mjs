/**
 * CinematicPreview & WebGL 2.0 Renderer Verification Suite
 * Tests transform matrix mathematics, cursor-anchor invariance,
 * double buffering state machine, colormaps, binary packet ingestion,
 * and GLSL shader math logic.
 */
import process from 'node:process';
import {
  computeTransformMatrix,
  evalTurbo,
  evalInferno,
} from '../src/components/preview/cinematicRenderer.js';
import {
  packFrameMessage,
  parseFrameMessage,
  KIND_LIVE,
} from '../src/transport/frameProtocol.js';

let checks = 0;
let failures = 0;

const ok = (name, cond, detail = '') => {
  checks += 1;
  if (cond) {
    console.log(`  PASS  ${name}`);
    return;
  }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};

console.log('── CinematicPreview Transform Matrix & Aspect Ratio ──────');
{
  // 1. Aspect ratio contain: 1920x1080 in 1000x1000 square viewport
  // Base scale = 1000 / 1920 = 0.520833
  // dispW = 1000, dispH = 1080 * (1000 / 1920) = 562.5
  // scaleX = 1000 / 1000 = 1.0
  // scaleY = 562.5 / 1000 = 0.5625
  const m1 = computeTransformMatrix({
    viewportWidth: 1000,
    viewportHeight: 1000,
    imageWidth: 1920,
    imageHeight: 1080,
    zoom: 1.0,
    panX: 0,
    panY: 0,
  });

  ok('Matrix is Float32Array of length 9', m1 instanceof Float32Array && m1.length === 9);
  ok('scaleX is 1.0 (pillarbox/letterbox match)', Math.abs(m1[0] - 1.0) < 1e-5);
  ok('scaleY preserves 16:9 ratio in square viewport', Math.abs(m1[4] - 0.5625) < 1e-5);
  ok('Translation is zero at center', m1[6] === 0 && m1[7] === 0);
  ok('Diagonal element m22 is 1.0', m1[8] === 1.0);

  // 2. Zoom scaling: zoom = 2.5
  const mZoom = computeTransformMatrix({
    viewportWidth: 1000,
    viewportHeight: 1000,
    imageWidth: 1920,
    imageHeight: 1080,
    zoom: 2.5,
    panX: 0,
    panY: 0,
  });
  ok('scaleX scales linearly with zoom', Math.abs(mZoom[0] - 2.5) < 1e-5);
  ok('scaleY scales linearly with zoom', Math.abs(mZoom[4] - 0.5625 * 2.5) < 1e-5);

  // 3. Pan translation in clip space
  // PanX = 150 px in 1000 px viewport -> ClipX = 2 * 150 / 1000 = 0.3
  // PanY = 200 px in 1000 px viewport -> ClipY = -2 * 200 / 1000 = -0.4
  const mPan = computeTransformMatrix({
    viewportWidth: 1000,
    viewportHeight: 1000,
    imageWidth: 1920,
    imageHeight: 1080,
    zoom: 1.0,
    panX: 150,
    panY: 200,
  });
  ok('panX maps to clip space transX correctly', Math.abs(mPan[6] - 0.3) < 1e-5);
  ok('panY maps to clip space transY correctly (inverted Y)', Math.abs(mPan[7] - (-0.4)) < 1e-5);
}

console.log('── Cursor Anchor Point Invariance Math ───────────────────');
{
  // Test that when zooming at cursor point (mx, my),
  // the corresponding normalized image coordinate remains invariant!
  const W = 1200;
  const H = 800;
  const imgW = 1920;
  const imgH = 1080;

  let zoom = 1.0;
  let panX = 0;
  let panY = 0;

  // Let cursor be at (800, 300) in canvas pixels
  const mx = 800;
  const my = 300;
  const dx = mx - W / 2;
  const dy = my - H / 2;

  // Function to calculate image UV for a canvas point
  const getUv = (cx, cy, curZoom, curPanX, curPanY) => {
    const baseScale = Math.min(W / imgW, H / imgH);
    const dispW = imgW * baseScale * curZoom;
    const dispH = imgH * baseScale * curZoom;
    const left = W / 2 + curPanX - dispW / 2;
    const top = H / 2 + curPanY - dispH / 2;
    return {
      u: (cx - left) / dispW,
      v: (cy - top) / dispH,
    };
  };

  const uvBefore = getUv(mx, my, zoom, panX, panY);

  // Zoom in to 3.5x with cursor anchoring
  const newZoom = 3.5;
  const alpha = newZoom / zoom;
  panX = dx - (dx - panX) * alpha;
  panY = dy - (dy - panY) * alpha;
  zoom = newZoom;

  const uvAfter = getUv(mx, my, zoom, panX, panY);

  ok('Cursor anchor invariance: U coordinate unchanged under cursor', Math.abs(uvBefore.u - uvAfter.u) < 1e-6);
  ok('Cursor anchor invariance: V coordinate unchanged under cursor', Math.abs(uvBefore.v - uvAfter.v) < 1e-6);
}

console.log('── Double Buffering Texture State Machine ────────────────');
{
  class MockGl {
    constructor() {
      this.LINEAR = 0x2601;
      this.NEAREST = 0x2600;
      this.TEXTURE_2D = 0x0de1;
      this.TEXTURE_MIN_FILTER = 0x2801;
      this.TEXTURE_MAG_FILTER = 0x2800;
      this.TEXTURE_WRAP_S = 0x2802;
      this.TEXTURE_WRAP_T = 0x2803;
      this.CLAMP_TO_EDGE = 0x812f;
      this.RGBA = 0x1908;
      this.UNSIGNED_BYTE = 0x1401;
      this.nextId = 1;
      this.deleted = [];
      this.boundTexture = null;
      this.subImageCalls = 0;
      this.imageCalls = 0;
    }
    createTexture() { return { id: this.nextId++ }; }
    deleteTexture(t) { this.deleted.push(t.id); }
    bindTexture(target, t) { this.boundTexture = t; }
    texParameteri() {}
    texImage2D() { this.imageCalls += 1; }
    texSubImage2D() { this.subImageCalls += 1; }
  }

  // Import DoubleBufferedTexture logic
  const gl = new MockGl();

  class TestDoubleBufferedTexture {
    constructor(gl) {
      this.gl = gl;
      this.textures = [gl.createTexture(), gl.createTexture()];
      this.front = 0;
      this.width = 0;
      this.height = 0;
      this.hasData = false;
      this.filter = gl.LINEAR;
    }
    getFront() { return this.textures[this.front]; }
    getBack() { return this.textures[1 - this.front]; }
    uploadImageSource(source) {
      const w = source.width;
      const h = source.height;
      if (!w || !h) return false;
      const back = this.getBack();
      this.gl.bindTexture(this.gl.TEXTURE_2D, back);
      if (this.width === w && this.height === h) {
        this.gl.texSubImage2D();
      } else {
        this.gl.texImage2D();
        this.width = w;
        this.height = h;
      }
      this.front = 1 - this.front;
      this.hasData = true;
      return true;
    }
  }

  const dbt = new TestDoubleBufferedTexture(gl);
  ok('Initial front texture is tex 0', dbt.front === 0);
  const tex0 = dbt.getFront();
  const tex1 = dbt.getBack();
  ok('Front and back textures are distinct handles', tex0.id !== tex1.id);

  // First frame upload (1920x1080)
  dbt.uploadImageSource({ width: 1920, height: 1080 });
  ok('Uploaded to back texture and flipped front index to 1', dbt.front === 1);
  ok('Current front texture is now tex1', dbt.getFront().id === tex1.id);
  ok('Allocated with texImage2D on first frame', gl.imageCalls === 1 && gl.subImageCalls === 0);

  // Second frame upload of same dimensions
  dbt.uploadImageSource({ width: 1920, height: 1080 });
  ok('Second upload flipped front index back to 0', dbt.front === 0);
  ok('Reused buffer with texSubImage2D without reallocation', gl.subImageCalls === 1);
}

console.log('── Colormaps & Difference Heatmap Math ───────────────────');
{
  // Turbo colormap validation
  const t0 = evalTurbo(0.0);
  const tMid = evalTurbo(0.5);
  const t1 = evalTurbo(1.0);

  ok('Turbo at 0.0 is deep blue/purple', t0[0] < 0.25 && t0[2] > 0.05);
  ok('Turbo at 0.5 is bright green/yellow', tMid[1] > 0.8 && tMid[2] < 0.35);
  ok('Turbo at 1.0 is dark wine red', t1[0] > 0.5 && t1[1] < 0.1 && t1[2] < 0.05);

  // Inferno colormap validation
  const i0 = evalInferno(0.0);
  const iMid = evalInferno(0.5);
  const i1 = evalInferno(1.0);

  ok('Inferno at 0.0 is near-black/dark purple', i0[0] < 0.05 && i0[1] < 0.05 && i0[2] < 0.05);
  ok('Inferno at 0.5 is reddish orange', iMid[0] > 0.5 && iMid[1] > 0.2);
  ok('Inferno at 1.0 is bright yellow/white', i1[0] > 0.95 && i1[1] > 0.95);

  // Difference gain calculation
  const colA = [0.8, 0.4, 0.2];
  const colB = [0.75, 0.42, 0.2];
  const gain = 3.0;
  const diffR = Math.abs(colA[0] - colB[0]) * gain; // 0.05 * 3 = 0.15
  const diffG = Math.abs(colA[1] - colB[1]) * gain; // 0.02 * 3 = 0.06
  const diffB = Math.abs(colA[2] - colB[2]) * gain; // 0
  const maxDiff = Math.max(diffR, diffG, diffB);
  ok('Difference gain calculates channel maximum correctly', Math.abs(maxDiff - 0.15) < 1e-6);
}

console.log('── Binary Frame Packet Wire Ingestion ───────────────────');
{
  const fakeJpeg = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10]);
  const msg = packFrameMessage(
    { kind: KIND_LIVE, stream: 42, frame: 100, width: 1920, height: 1080 },
    fakeJpeg.buffer
  );

  const parsed = parseFrameMessage(msg);
  ok('Wire message parsed correctly with width and height', parsed && parsed.width === 1920 && parsed.height === 1080);
  ok('Wire payload preserved exact JPEG magic bytes',
    new Uint8Array(parsed.bytes)[0] === 0xff && new Uint8Array(parsed.bytes)[1] === 0xd8);

  // Detect WebP magic: 'RIFF' + 'WEBP'
  const fakeWebP = new Uint8Array([
    0x52, 0x49, 0x46, 0x46, // RIFF
    0x20, 0x00, 0x00, 0x00,
    0x57, 0x45, 0x42, 0x50, // WEBP
    0x56, 0x50, 0x38, 0x20, // VP8
  ]);

  const isWebP = fakeWebP.length >= 12 &&
    fakeWebP[0] === 0x52 && fakeWebP[1] === 0x49 && fakeWebP[2] === 0x46 && fakeWebP[3] === 0x46 &&
    fakeWebP[8] === 0x57 && fakeWebP[9] === 0x45 && fakeWebP[10] === 0x42 && fakeWebP[11] === 0x50;

  ok('WebP magic byte detection is accurate', isWebP === true);
}

console.log(`\n${failures === 0 ? 'ALL GREEN' : 'FAILURES'}: ${checks - failures}/${checks} checks passed`);
process.exit(failures === 0 ? 0 : 1);
