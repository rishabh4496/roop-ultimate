/**
 * Mask Brush pointer -> canvas mapping (plain node). Run: node .render-check/brush-mapping-check.mjs
 *
 * Pins: sub-pixel positions are kept (never snapped), the brush radius follows the on-screen
 * diameter, and an `object-contain` letterbox INSIDE the canvas box is accounted for (the
 * anamorphic case, where the box takes the display aspect and the pixels keep their own).
 */
import process from 'node:process';
import { containedRect, brushSample } from '../src/components/faceswap/maskBrush.js';

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const near = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;
const R = (left, top, width, height) => ({ left, top, width, height });

console.log('── box == picture (square pixels) ──────────────────────────');
{
  const rect = R(100, 50, 800, 450);                 // 16:9 box, 1920x1080 media
  const s = brushSample(rect, 1920, 1080, 100, 50, 30);
  ok('the top-left corner is canvas (0,0)', near(s.x, 0) && near(s.y, 0));
  const c = brushSample(rect, 1920, 1080, 500, 275, 30);
  ok('the centre is canvas (960,540)', near(c.x, 960) && near(c.y, 540), `${c.x},${c.y}`);
  ok('scale is canvas px per visual px (2.4)', near(c.scale, 2.4));
  ok('a 30 px brush is a 36 canvas-px radius (30/2 * 2.4)', near(c.r, 36), `${c.r}`);
  const f = brushSample(rect, 1920, 1080, 100.25, 50.125, 30);
  ok('sub-pixel input stays sub-pixel (no rounding)', near(f.x, 0.6) && near(f.y, 0.3), `${f.x},${f.y}`);
  ok('a zero-size box returns null, not NaN', brushSample(R(0, 0, 0, 0), 1920, 1080, 1, 1, 30) === null);
  ok('the radius never collapses to nothing', brushSample(rect, 1920, 1080, 500, 275, 0).r >= 0.5);
}

console.log('── stage zoom (the rect already carries it) ────────────────');
{
  const rect = R(-300, -170, 1600, 900);             // 2x zoom, panned: box larger than the viewport
  const s = brushSample(rect, 1920, 1080, 500, 280, 30);
  ok('the same visual point maps through the zoomed rect', near(s.x, (500 + 300) * 1.2) && near(s.y, (280 + 170) * 1.2));
  ok('the brush is still 30 visual px: radius = 15 * 1.2', near(s.r, 18), `${s.r}`);
}

console.log('── letterboxed pixels (anamorphic box) ─────────────────────');
{
  // 1440x1080 pixels shown at 16:9 (a 4:3 clip with SAR 4/3): the box is 800x450 but object-contain
  // fits the 4:3 picture to 600x450, centred.
  const rect = R(0, 0, 800, 450);
  const c = containedRect(rect, 1440, 1080);
  ok('the picture is 600x450 centred in the 800x450 box', near(c.width, 600) && near(c.height, 450) && near(c.left, 100) && near(c.top, 0));
  const left = brushSample(rect, 1440, 1080, 100, 0, 30);
  ok('the picture\'s own left edge is canvas x = 0 (not the box edge)', near(left.x, 0), `${left.x}`);
  const mid = brushSample(rect, 1440, 1080, 400, 225, 30);
  ok('the box centre is the canvas centre', near(mid.x, 720) && near(mid.y, 540), `${mid.x},${mid.y}`);
  ok('scale comes from the PICTURE width (2.4), not the box (1.8)', near(mid.scale, 2.4), `${mid.scale}`);
  const naive = 1440 / 800;
  ok('(the old formula would have been off)', !near(naive, mid.scale));
  ok('a click on the letterbox bar is reported as off the image', brushSample(rect, 1440, 1080, 40, 225, 30).onImage === false && mid.onImage === true);

  // Tall letterbox: a 1080x1920 portrait clip in a 16:9 box.
  const tall = brushSample(R(0, 0, 800, 450), 1080, 1920, 400, 225, 30);
  ok('pillarboxed portrait: centre is centre', near(tall.x, 540) && near(tall.y, 960), `${tall.x},${tall.y}`);
}

globalThis.__RENDER_CHECK_FAILURES__ = failures;
console.log(failures === 0 ? '\nbrush-mapping: all checks passed' : `\nbrush-mapping: ${failures} check(s) FAILED`);
process.exit(failures === 0 ? 0 : 1);
