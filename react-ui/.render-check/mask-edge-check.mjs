/**
 * The mask micro-profile's maths against the production feather.
 *
 *   node .render-check/run-module.mjs mask-edge-check.mjs
 *
 * `mask-edge.fixture.json` is what `MaskComposer.asymmetric_inward` (the real class) produced for straight-edged mattes
 * (gen-mask-edge-fixture.py). Every sample is compared with `maskEdgeModel.js` given the SAME inputs. What this cannot show:
 * that the sliders reach `face_mask_blend` / `mask_erode_dilate_radius` (e2e/stage5-studio.spec.js), or that a real face's matte
 * has the span assumed here (only the 18% cap reads it).
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const m = await import('../src/components/faceswap/maskEdgeModel.js');
const fixture = JSON.parse(readFileSync(join(here, 'mask-edge.fixture.json'), 'utf8'));

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const section = (t) => console.log(`\n── ${t} ${'─'.repeat(Math.max(2, 64 - t.length))}`);
const near = (a, b, eps = 1e-5) => Math.abs(a - b) <= eps;

section('The model reproduces MaskComposer.asymmetric_inward, sample for sample');
{
  let total = 0; let worst = 0; let worstCase = '';
  for (const c of fixture.cases) {
    const feather = m.pasteFeatherPx(c.blend, c.crop);
    ok(`feather px for blend ${c.blend}: ${feather.toFixed(2)}`, near(feather, c.feather, 1e-9), `${feather} vs ${c.feather}`);
    const s = m.yawStrength(c.yaw);
    for (const p of c.samples) {
      const { f, n } = m.lateralFactors(p.rel, c.yaw);
      // the fixture's `dist` is in PIXELS inside the matte; the production distance transform reads DT_STEP per pixel
      const alpha = m.edgeAlpha(p.dist * m.DT_STEP, m.rampWidthPx(feather, c.span, s, f, n), m.farInsetPx(c.span, s, f));
      const d = Math.abs(alpha - p.alpha);
      total += 1;
      if (d > worst) { worst = d; worstCase = `blend ${c.blend} yaw ${c.yaw} x ${p.x}: model ${alpha} vs real ${p.alpha}`; }
    }
  }
  ok(`${total} samples across ${fixture.cases.length} cases agree to 1e-5 (worst ${worst.toExponential(2)})`, worst < 1e-5, worstCase);
  ok('the fixture really exercised the yaw terms (far edge inset, soft far, tight near)',
    fixture.cases.some((c) => c.yaw !== 0 && c.samples.some((p) => p.alpha > 0 && p.alpha < 1)));
  ok('and the 18% cap (blend 100 / 200 ask for more than the cap allows)',
    m.rampWidthPx(m.pasteFeatherPx(100, 512), 430) < m.pasteFeatherPx(100, 512));
}

section('buildProfile: the curve the component draws');
{
  const a = m.buildProfile({ blend: 12, erode: 0 });
  ok('the frontal ramp is the feather in distance-transform units (12 -> 30.72) = 32.17 px', near(a.widthPx, 30.72 / 0.955, 1e-9), String(a.widthPx));
  ok('not capped at the default', a.capped === false);
  ok('alpha is 0 at and beyond the boundary, 1 deep inside',
    a.alpha[a.alpha.length - 1] === 0 && a.alpha[0] === 1, `${a.alpha[0]} .. ${a.alpha[a.alpha.length - 1]}`);
  ok('alpha never increases as x moves outward', a.alpha.every((v, i) => i === 0 || v <= a.alpha[i - 1] + 1e-12));
  ok('the 50% point is half a ramp inside the boundary', near(a.midPx, -(30.72 / 0.955) / 2, 1e-9), String(a.midPx));
  ok('10-90% width is 60.8% of the ramp (smoothstep: t = 0.196 .. 0.804)', near(a.width1090Px / a.widthPx, 0.6084, 1e-3), String(a.width1090Px / a.widthPx));
  ok('max slope is 1.5 per ramp, in alpha per pixel', near(a.maxSlopePerPx, 1.5 / a.widthPx, 1e-12));

  const grown = m.buildProfile({ blend: 12, erode: 10 });
  ok('dilate +10 shifts the whole ramp 10 px outward (alpha reaches 0 at x = +10)',
    near(grown.midPx - a.midPx, 10, 1e-9) && grown.alpha[grown.alpha.length - 1] === 0);
  const shrunk = m.buildProfile({ blend: 12, erode: -10 });
  ok('erode -10 shifts it 10 px inward', near(shrunk.midPx - a.midPx, -10, 1e-9));
  ok('the window always contains the whole ramp, whatever the radius',
    [-64, -10, 0, 10, 64].every((r) => { const p = m.buildProfile({ blend: 40, erode: r }); return p.lo <= p.midPx - p.widthPx / 2 && p.hi >= r; }));

  const capped = m.buildProfile({ blend: 100 });
  ok('blend 100 is capped at 18% of the span, and says so', capped.capped && near(capped.widthPx, (0.18 * 512 * 0.8) / 0.955, 1e-9), String(capped.widthPx));
  ok('blend 0 is a 1 px edge (never zero width, never NaN)', Number.isFinite(m.buildProfile({ blend: 0 }).maxSlopePerPx) && near(m.buildProfile({ blend: 0 }).widthPx, 1 / 0.955, 1e-12));
  ok('bad numbers do not poison the curve', m.buildProfile({ blend: 'x', erode: NaN }).alpha.every(Number.isFinite));

  const yawed = m.buildProfile({ blend: 12, yawDeg: 60 });
  const mid = (arr) => arr.findIndex((v) => v < 0.5);
  ok('a 60-degree face: the far cheek fades later, the near cheek sooner than the frontal one',
    yawed.far[mid(yawed.far)] !== undefined && mid(yawed.far) < mid(yawed.alpha) && mid(yawed.near) > mid(yawed.alpha),
    `far ${mid(yawed.far)} front ${mid(yawed.alpha)} near ${mid(yawed.near)}`);
}

section('isModelled: the legacy Gaussian is not drawn');
{
  ok('semantic paste on (the default) is modelled', m.isModelled({}) && m.isModelled({ semantic_paste_mask: true }) && m.isModelled(undefined));
  ok('semantic paste off is not', m.isModelled({ semantic_paste_mask: false }) === false);
}

globalThis.__RENDER_CHECK_FAILURES__ = failures;
console.log(failures === 0 ? '\nmask edge: all checks passed' : `\nmask edge: ${failures} FAILED`);
