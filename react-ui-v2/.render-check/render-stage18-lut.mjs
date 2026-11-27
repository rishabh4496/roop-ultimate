/**
 * Stage 18: WebGL2 3D .cube LUT Color Pipeline, Skin-Protected Vibrance, and Scopes Tests
 *
 * Exercises:
 *   - .cube parser with LUT_3D_SIZE, DOMAIN_MIN, DOMAIN_MAX
 *   - Explicit DOMAIN_MIN < DOMAIN_MAX validation refusing inverted/invalid domains
 *   - Premultiplied alpha scaling (rgb * a, a) and compositing
 *   - Color grading defaults and schema parity
 *
 * Run with:
 *   node .render-check/run-module.mjs render-stage18-lut.mjs
 */
import {
  parseCubeLut,
  validateCubeDomain,
  premultiplyAlpha,
  blendPremultiplied
} from '../src/components/faceswap/cubeLut.js';
import { GRADING_DEFAULTS } from '../src/components/faceswap/webgl2GradingPipeline.js';
import { FACESWAP_DEFAULTS } from '../src/components/faceswap/defaults.js';

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) {
    console.log(`  PASS  ${name}`);
    return;
  }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};

console.log('── Stage 18: .cube LUT parsing & domain validation ──────────────────');
{
  const validCube = `
# Sample 2x2x2 3D LUT
TITLE "Test Valid LUT"
LUT_3D_SIZE 2
DOMAIN_MIN 0.0 0.0 0.0
DOMAIN_MAX 1.0 1.0 1.0
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
`;
  const lut = parseCubeLut(validCube);
  ok('parses title', lut.title === 'Test Valid LUT');
  ok('parses size', lut.size === 2);
  ok('parses domain min', lut.domainMin[0] === 0.0 && lut.domainMin[1] === 0.0 && lut.domainMin[2] === 0.0);
  ok('parses domain max', lut.domainMax[0] === 1.0 && lut.domainMax[1] === 1.0 && lut.domainMax[2] === 1.0);
  ok('computes raw data array with size^3 * 3 floats', lut.data.length === 2 * 2 * 2 * 3);
  ok('computes rgbaData array with size^3 * 4 floats', lut.rgbaData.length === 2 * 2 * 2 * 4);
  ok('fills alpha channel in rgbaData with 1.0', lut.rgbaData[3] === 1.0 && lut.rgbaData[7] === 1.0);

  // Validation function checks
  const validDomain = validateCubeDomain([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
  ok('validateCubeDomain passes for min < max', validDomain === true);

  // Inverted domain min >= max
  let caughtInverted = false;
  try {
    validateCubeDomain([1.0, 0.0, 0.0], [0.5, 1.0, 1.0]);
  } catch (err) {
    caughtInverted = true;
    ok('rejection includes descriptive DOMAIN_MIN / DOMAIN_MAX error',
      err.message.includes('DOMAIN_MIN') && err.message.includes('DOMAIN_MAX'));
  }
  ok('throws on inverted domain min >= max', caughtInverted);

  // Equal domain min == max
  let caughtEqual = false;
  try {
    validateCubeDomain([0.5, 0.5, 0.5], [0.5, 1.0, 1.0]);
  } catch {
    caughtEqual = true;
  }
  ok('throws on equal domain min == max', caughtEqual);

  // Non-finite / NaN domain
  let caughtNaN = false;
  try {
    validateCubeDomain([NaN, 0.0, 0.0], [1.0, 1.0, 1.0]);
  } catch {
    caughtNaN = true;
  }
  ok('throws on NaN / non-finite domain', caughtNaN);

  // Parser itself rejects invalid domains
  const invalidCube = `
LUT_3D_SIZE 2
DOMAIN_MIN 1.0 0.0 0.0
DOMAIN_MAX 0.0 1.0 1.0
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
`;
  let parserRejected = false;
  try {
    parseCubeLut(invalidCube);
  } catch {
    parserRejected = true;
  }
  ok('parseCubeLut refuses invalid cube domain', parserRejected);
}

console.log('── Stage 18: Premultiplied alpha custody ───────────────────────────');
{
  const rgb = [1.0, 0.6, 0.2];
  const alpha = 0.5;
  const pm = premultiplyAlpha(rgb, alpha);
  ok('premultiplyAlpha scales RGB by alpha: rgb * a',
    Math.abs(pm[0] - 0.5) < 1e-6 &&
    Math.abs(pm[1] - 0.3) < 1e-6 &&
    Math.abs(pm[2] - 0.1) < 1e-6
  );
  ok('premultiplyAlpha preserves alpha in 4th component', Math.abs(pm[3] - 0.5) < 1e-6);

  // Zero alpha turns RGB to 0
  const pmZero = premultiplyAlpha(rgb, 0.0);
  ok('zero alpha produces strict black (0,0,0,0)',
    pmZero[0] === 0.0 && pmZero[1] === 0.0 && pmZero[2] === 0.0 && pmZero[3] === 0.0);

  // Compositing Porter-Duff Over: src over dst
  // src: [0.5, 0.25, 0.1, 0.5], dst: [0.5, 0.5, 0.5, 1.0]
  // result rgb = src_rgb + dst_rgb * (1 - src_a) = [0.75, 0.5, 0.35]
  // result a = src_a + dst_a * (1 - src_a) = 0.5 + 0.5 = 1.0
  const src = [0.5, 0.25, 0.1, 0.5];
  const dst = [0.5, 0.5, 0.5, 1.0];
  const blended = blendPremultiplied(src, dst);
  ok('blendPremultiplied executes standard Porter-Duff Over',
    Math.abs(blended[0] - 0.75) < 1e-6 &&
    Math.abs(blended[1] - 0.50) < 1e-6 &&
    Math.abs(blended[2] - 0.35) < 1e-6 &&
    Math.abs(blended[3] - 1.00) < 1e-6
  );
}

console.log('── Stage 18: Defaults and parameter invariants ────────────────────');
{
  ok('FACESWAP_DEFAULTS includes lut_path', 'lut_path' in FACESWAP_DEFAULTS);
  ok('FACESWAP_DEFAULTS includes lut_bake', 'lut_bake' in FACESWAP_DEFAULTS);
  ok('FACESWAP_DEFAULTS lut_bake defaults to false (display preview by default)',
    FACESWAP_DEFAULTS.lut_bake === false);
  ok('FACESWAP_DEFAULTS lut_opacity defaults to 1.0', FACESWAP_DEFAULTS.lut_opacity === 1.0);
  ok('FACESWAP_DEFAULTS color_vibrance defaults to 0.0', FACESWAP_DEFAULTS.color_vibrance === 0.0);
  ok('FACESWAP_DEFAULTS color_skin_protect defaults to 1.0', FACESWAP_DEFAULTS.color_skin_protect === 1.0);

  // Grading pipeline neutrals
  ok('GRADING_DEFAULTS color_temperature neutral is 0.0', GRADING_DEFAULTS.color_temperature === 0.0);
  ok('GRADING_DEFAULTS color_tint neutral is 0.0', GRADING_DEFAULTS.color_tint === 0.0);
  ok('GRADING_DEFAULTS lift neutral is 0.0',
    GRADING_DEFAULTS.lift_r === 0.0 && GRADING_DEFAULTS.lift_g === 0.0 && GRADING_DEFAULTS.lift_b === 0.0);
  ok('GRADING_DEFAULTS gamma neutral is 1.0',
    GRADING_DEFAULTS.gamma_r === 1.0 && GRADING_DEFAULTS.gamma_g === 1.0 && GRADING_DEFAULTS.gamma_b === 1.0);
  ok('GRADING_DEFAULTS gain neutral is 1.0',
    GRADING_DEFAULTS.gain_r === 1.0 && GRADING_DEFAULTS.gain_g === 1.0 && GRADING_DEFAULTS.gain_b === 1.0);
}

globalThis.__RENDER_CHECK_FAILURES__ = failures;
console.log(failures === 0 ? '\nStage 18 LUT & Scopes: all checks passed' : `\nStage 18: ${failures} check(s) FAILED`);
