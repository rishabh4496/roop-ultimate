/**
 * CinematicPreview & WebGL 2.0 Renderer Verification Suite
 * Tests transform matrix mathematics, cursor-anchor invariance,
 * double buffering state machine, colormaps, binary packet ingestion,
 * and GLSL shader math logic.
 */
import process from 'node:process';
import {
  anchoredPan,
  computeTransformMatrix,
  DoubleBufferedTexture,
  sniffImageMime,
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
console.log('── Cursor Anchor Point Invariance (anchoredPan + real matrix) ──');
{
  // Map a canvas pixel back to quad coordinates through the matrix the
  // shader actually uses. anchoredPan must keep that point fixed.
  const W = 1200;
  const H = 800;
  const imgW = 1920;
  const imgH = 1080;
  const quadAt = (cx, cy, zoom, pan) => {
    const m = computeTransformMatrix({
      viewportWidth: W, viewportHeight: H, imageWidth: imgW, imageHeight: imgH,
      zoom, panX: pan.x, panY: pan.y,
    });
    const clipX = (cx / W) * 2 - 1;
    const clipY = 1 - (cy / H) * 2;
    return { qx: (clipX - m[6]) / m[0], qy: (clipY - m[7]) / m[4] };
  };

  const mx = 800;
  const my = 300;
  const dx = mx - W / 2;
  const dy = my - H / 2;
  let zoom = 1.0;
  let pan = { x: 0, y: 0 };
  const before = quadAt(mx, my, zoom, pan);
  // Several lerp steps, as the hook's animation loop takes them.
  for (const nextZoom of [1.4, 2.1, 3.5]) {
    pan = anchoredPan(pan, dx, dy, zoom, nextZoom);
    zoom = nextZoom;
  }
  const after = quadAt(mx, my, zoom, pan);
  ok('Cursor anchor invariance: X unchanged under cursor', Math.abs(before.qx - after.qx) < 1e-5,
    `${before.qx} -> ${after.qx}`);
  ok('Cursor anchor invariance: Y unchanged under cursor', Math.abs(before.qy - after.qy) < 1e-5,
    `${before.qy} -> ${after.qy}`);
  const off = quadAt(mx + 100, my, zoom, pan);
  ok('A point away from the cursor DOES move (test is not vacuous)', Math.abs(off.qx - before.qx) > 1e-3);
}

console.log('── DoubleBufferedTexture (real class, mock GL) ───────────');
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
      this.boundTexture = null;
      this.uploads = []; // { kind, texId }
    }
    createTexture() { return { id: this.nextId++ }; }
    deleteTexture() {}
    bindTexture(target, t) { this.boundTexture = t; }
    texParameteri() {}
    texImage2D() { this.uploads.push({ kind: 'image', texId: this.boundTexture.id }); }
    texSubImage2D() { this.uploads.push({ kind: 'sub', texId: this.boundTexture.id }); }
  }

  const gl = new MockGl();
  const dbt = new DoubleBufferedTexture(gl);
  const tex0 = dbt.getFront();
  const tex1 = dbt.getBack();
  ok('Front and back textures are distinct handles', tex0.id !== tex1.id);
  ok('Rejects a source with no dimensions', dbt.uploadImageSource({}) === false && gl.uploads.length === 0);

  dbt.uploadImageSource({ width: 1920, height: 1080 });
  ok('First upload writes the BACK texture', gl.uploads[0].texId === tex1.id);
  ok('First upload allocates (texImage2D)', gl.uploads[0].kind === 'image');
  ok('Front flips to the texture just written', dbt.getFront().id === tex1.id && dbt.hasData);

  dbt.uploadImageSource({ width: 1920, height: 1080 });
  ok('Second upload writes the other texture', gl.uploads[1].texId === tex0.id);
  ok('Same size reuses storage (texSubImage2D)', gl.uploads[1].kind === 'sub');

  dbt.uploadImageSource({ videoWidth: 1280, videoHeight: 720 });
  ok('Size change reallocates, reading videoWidth/videoHeight', gl.uploads[2].kind === 'image' && dbt.width === 1280);
}

console.log('── Colormaps ─────────────────────────────────────────────');
{
  const t0 = evalTurbo(0.0);
  const tMid = evalTurbo(0.5);
  const t1 = evalTurbo(1.0);
  ok('Turbo at 0.0 is deep blue/purple', t0[0] < 0.25 && t0[2] > 0.05);
  ok('Turbo at 0.5 is bright green/yellow', tMid[1] > 0.8 && tMid[2] < 0.35);
  ok('Turbo at 1.0 is dark wine red', t1[0] > 0.5 && t1[1] < 0.1 && t1[2] < 0.05);

  const i0 = evalInferno(0.0);
  const iMid = evalInferno(0.5);
  const i1 = evalInferno(1.0);
  ok('Inferno at 0.0 is near-black/dark purple', i0[0] < 0.05 && i0[1] < 0.05 && i0[2] < 0.05);
  ok('Inferno at 0.5 is reddish orange', iMid[0] > 0.5 && iMid[1] > 0.2);
  ok('Inferno at 1.0 is bright yellow/white', i1[0] > 0.95 && i1[1] > 0.95);
  ok('Colormaps clamp out-of-range input', evalTurbo(-1).join() === t0.join() && evalInferno(2).join() === i1.join());
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

  const webp = new Uint8Array([0x52, 0x49, 0x46, 0x46, 0x20, 0, 0, 0, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38, 0x20]);
  const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const riffWav = new Uint8Array([0x52, 0x49, 0x46, 0x46, 0x20, 0, 0, 0, 0x57, 0x41, 0x56, 0x45]);
  const riffAvi = new Uint8Array([0x52, 0x49, 0x46, 0x46, 0x20, 0, 0, 0, 0x41, 0x56, 0x49, 0x20]);
  ok('sniffImageMime: WebP', sniffImageMime(webp) === 'image/webp');
  ok('sniffImageMime: PNG', sniffImageMime(png) === 'image/png');
  ok('sniffImageMime: JPEG', sniffImageMime(new Uint8Array(parsed.bytes)) === 'image/jpeg');
  ok('sniffImageMime: RIFF WAVE is not an image', sniffImageMime(riffWav) === null);
  ok('sniffImageMime: RIFF AVI is not an image', sniffImageMime(riffAvi) === null);
  const nearWebp = webp.slice(); nearWebp[8] = 0x58; // 'XEBP': one byte off
  ok('sniffImageMime: one wrong fourcc byte is not WebP', sniffImageMime(nearWebp) === null);
  ok('sniffImageMime: raw pixels are not an image', sniffImageMime(new Uint8Array([1, 2, 3, 4])) === null);
}

console.log(`\n${failures === 0 ? 'ALL GREEN' : 'FAILURES'}: ${checks - failures}/${checks} checks passed`);
process.exit(failures === 0 ? 0 : 1);
