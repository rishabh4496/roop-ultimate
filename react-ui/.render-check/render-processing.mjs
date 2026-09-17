/**
 * Render the Processing tab for real, in Node, against payloads captured from
 * the live backend.
 *
 * WHY THIS EXISTS, given the tab already had 162 passing tests.
 *
 * Those tests — and every other `test_ui_*.py` in this repo — are REGEXES OVER
 * SOURCE TEXT. They are good at what they are for (a prop that stopped being
 * passed, a panel that got hidden again, a one-off font size) and they are
 * structurally incapable of catching the class of bug that this change could
 * plausibly introduce, because they never execute a line of it:
 *
 *   - an interdependent derived value that is accidentally read before it is
 *     initialized, which ordinary source checks and a production build can miss;
 *   - a null-guard that is wrong for a shape the backend actually sends, e.g.
 *     `runtime.sections` present but `MODEL` absent, or `output.path` empty
 *     string rather than missing;
 *   - a hook called conditionally, which React only complains about at runtime.
 *
 * `vite build` catches none of those: they are all valid JavaScript. So this
 * mounts the real components with `react-dom/server` and asserts on the HTML
 * that comes out, across the run states the backend genuinely produces.
 *
 * Fixtures are RECORDED, not invented: `.render-check/fixtures.json` came from
 * /api/progress, /api/settings and /api/system/telemetry during a real
 * 88,483-frame render. Only machine-local workspace prefixes were normalized so
 * the fixture is portable. A hand-written fixture would only ever prove the
 * component agrees with my assumptions about the payload, which is the thing
 * that was wrong in the first place.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// ── Minimal browser surface ───────────────────────────────────────────────
// The app reads `window.location` at MODULE scope (src/api.js computes the API
// origin on import), and several hooks touch localStorage and matchMedia the
// same way. This is not a DOM implementation and is not pretending to be one:
// it is the smallest set of globals that lets the real modules be imported, so
// that the component code underneath can be executed unmodified. Anything the
// components do beyond this will throw, and throwing is the point — a render
// that needs more of the DOM than this is a render being tested honestly.
const store = new Map();
globalThis.window = globalThis;
globalThis.location = { protocol: 'http:', host: '127.0.0.1:42003', origin: 'http://127.0.0.1:42003', hash: '' };
globalThis.window.location = globalThis.location;
globalThis.localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
};
globalThis.window.localStorage = globalThis.localStorage;
globalThis.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
globalThis.window.matchMedia = globalThis.matchMedia;
globalThis.document = {
  documentElement: { style: { setProperty() {} }, dataset: {}, setAttribute() {}, removeAttribute() {} },
  addEventListener() {}, removeEventListener() {},
  visibilityState: 'visible',
};
globalThis.window.addEventListener = () => {};
globalThis.window.removeEventListener = () => {};
globalThis.requestAnimationFrame = (fn) => setTimeout(() => fn(Date.now()), 0);
globalThis.cancelAnimationFrame = (id) => clearTimeout(id);

const { renderToStaticMarkup } = await import('react-dom/server');
const React = (await import('react')).default;

const here = dirname(fileURLToPath(import.meta.url));
// PowerShell's Set-Content -Encoding UTF8 emits a BOM, which JSON.parse rejects.
const fixtures = JSON.parse(readFileSync(join(here, 'fixtures.json'), 'utf8').replace(/^\uFEFF/, ''));

const Processing = (await import('../src/components/Processing.jsx')).default;
const RunModelsPanel = (await import('../src/components/faceswap/RunModelsPanel.jsx')).default;
const DiagnosticsPanel = (await import('../src/components/faceswap/DiagnosticsPanel.jsx')).default;

const { settings, telemetry } = fixtures;
const base = fixtures.progress;

let failures = 0;
let checks = 0;
const ok = (name, cond, detail = '') => {
  checks += 1;
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};

/** Render, converting a throw into a reportable failure rather than a crash. */
const render = (label, element) => {
  try {
    return { html: renderToStaticMarkup(element) };
  } catch (err) {
    failures += 1;
    checks += 1;
    console.log(`  FAIL  ${label} threw during render\n          ${err && err.stack ? err.stack.split('\n').slice(0, 4).join('\n          ') : err}`);
    return { html: null, err };
  }
};

const tab = (progress) => React.createElement(Processing, {
  progress,
  settings,
  notify: () => {},
  setTab: () => {},
  desktopAlerts: false,
  onToggleDesktopAlerts: () => {},
  onPauseRun: () => {},
  onResumeRun: () => {},
  onStopRun: () => {},
  controlBusy: '',
});

// The states the backend actually goes through. `base` is a real finished run;
// the mid-render values are the ones this session observed live.
const MID = {
  ...base,
  processing: true,
  paused: false,
  progress: 0.5927692325079393,
  desc: 'Processing frame 52450 / 88483 (20.4 FPS)',
  status_line: 'Processing frame 52450 / 88483 (20.4 FPS)',
  eta_s: 1766.97,
  error: '',
  output: { path: '', kind: '' },
  current_frame: 52450,
  total_frames: 88483,
  fps: 20.4,
};

console.log('\n── mid-render ─────────────────────────────────────────────');
{
  const { html } = render('mid-render', tab(MID));
  if (html) {
    ok('renders', html.length > 1000);
    ok('progressbar role is present', html.includes('role="progressbar"'));
    ok('aria-valuenow is the real percentage', html.includes('aria-valuenow="59"'),
       (html.match(/aria-valuenow="[^"]*"/) || ['<none>'])[0]);
    ok('aria-valuetext names the frame', /aria-valuetext="[^"]*frame 52450 of 88483/.test(html));
    ok('frame counter is in the headline', html.includes('52,450') && html.includes('88,483'));
    ok('models panel rendered', html.includes('Models in use'));
    ok('swapper read from runtime, not settings', html.includes('hyperswap'));
    ok('enhancer shown', html.includes('Restore Ultra'));
    ok('provider shown', html.includes('tensorrt'));
    ok('panel claims live provenance', html.includes('live from pipeline'));
    ok('no literal "undefined" leaked into the DOM', !html.includes('>undefined<'));
    ok('no literal "NaN" leaked into the DOM', !html.includes('NaN'));
    // The bar widths must be in range. A negative or >100% width is the
    // original bug class, and it is visible right here in the style attribute.
    const widths = [...html.matchAll(/width:\s*([-\d.]+)%/g)].map((m) => parseFloat(m[1]));
    ok('every bar width is within 0..100',
       widths.length > 0 && widths.every((w) => w >= 0 && w <= 100),
       `widths=${JSON.stringify(widths)}`);
    const offsets = [...html.matchAll(/stroke-dashoffset:\s*([-\d.]+)/g)].map((m) => parseFloat(m[1]));
    ok('stroke-dashoffset is never negative',
       offsets.every((o) => o >= 0), `offsets=${JSON.stringify(offsets)}`);

    // On THIS fixture the job fraction and the stage fraction happen to be the
    // same number (the swap stage is the whole job), so it cannot tell a
    // stage-local rail from a whole-run one. The discriminating case is its own
    // block below; this only pins the value.
    const frameFrac = (52450 / 88483) * 100;
    ok('rail width matches the frame counter',
       widths.some((w) => Math.abs(w - frameFrac) < 0.5),
       `widths=${JSON.stringify(widths)} expected≈${frameFrac.toFixed(2)}`);
  }
}

console.log('\n── rail is stage-local, not whole-run ─────────────────────');
{
  // The discriminating case: the JOB is nearly done (97%) while the CURRENT
  // stage has only just started (10% of its own frames). A rail drawing the
  // job fraction shows ~97%; a stage-local one shows ~10%. Before the fix the
  // upscale segment opened at whatever the job was at and crawled, which is
  // precisely this divergence.
  const diverged = {
    ...MID,
    progress: 0.97,
    desc: 'Upscaling frame 500 / 5000',
    status_line: 'Upscaling frame 500 / 5000',
    current_frame: 500,
    total_frames: 5000,
  };
  const { html } = render('stage-local rail', tab(diverged));
  if (html) {
    const widths = [...html.matchAll(/width:\s*([-\d.]+)%/g)].map((m) => parseFloat(m[1]));
    // The full-width run bar at the bottom legitimately shows 97; the rail
    // segment must not.
    const railish = widths.filter((w) => w > 2 && w < 100 && Math.abs(w - 97) > 0.001);
    ok('rail shows the STAGE fraction (~10%), not the job fraction (97%)',
       railish.some((w) => Math.abs(w - 10) < 0.6),
       `widths=${JSON.stringify(widths)}`);
    ok('no rail segment is drawn at the job fraction',
       !railish.some((w) => Math.abs(w - 97) < 0.6),
       `widths=${JSON.stringify(widths)}`);
    ok('the run bar itself still reports the job fraction',
       widths.some((w) => Math.abs(w - 97) < 0.6),
       `widths=${JSON.stringify(widths)}`);
  }
}

console.log('\n── out-of-range progress (the full-ring bug) ──────────────');
for (const bad of [1.4, -0.2, Number.NaN, null, undefined, '0.5']) {
  const { html } = render(`progress=${String(bad)}`, tab({ ...MID, progress: bad }));
  if (!html) continue;
  const offsets = [...html.matchAll(/stroke-dashoffset:\s*([-\d.]+)/g)].map((m) => parseFloat(m[1]));
  const widths = [...html.matchAll(/width:\s*([-\d.]+)%/g)].map((m) => parseFloat(m[1]));
  ok(`progress=${String(bad)} clamps the ring`,
     offsets.every((o) => o >= 0 && Number.isFinite(o)),
     `offsets=${JSON.stringify(offsets)}`);
  ok(`progress=${String(bad)} clamps every bar`,
     widths.every((w) => w >= 0 && w <= 100 && Number.isFinite(w)),
     `widths=${JSON.stringify(widths)}`);
}

console.log('\n── paused ─────────────────────────────────────────────────');
{
  const { html } = render('paused', tab({ ...MID, paused: true }));
  if (html) {
    ok('says Paused', html.includes('Paused'));
    // The whole point of the ETA fix: a stopped render must not advertise a
    // countdown, and must say why rather than showing a dead clock.
    ok('ETA is suppressed, not shown as a stale countdown', html.includes('paused'));
    ok('no --:-- dead clock', !html.includes('--:--'));
    ok('Finishes falls back to a dash', html.includes('—'));
  }
}

console.log('\n── stopping ───────────────────────────────────────────────');
{
  const { html } = render('stopping', tab({ ...MID, stop_requested: true }));
  if (html) {
    ok('says Stopping', html.includes('Stopping'));
    ok('ETA says stopping', html.includes('stopping'));
  }
}

console.log('\n── encode tail: no counter in the status line ─────────────');
{
  // The stage that used to blank every figure and trip the stall watchdog.
  const encode = {
    ...MID,
    desc: 'Combining video…',
    status_line: 'Combining video…',
    current_frame: 88483,
    total_frames: 88483,
  };
  const { html } = render('encode tail', tab(encode));
  if (html) {
    ok('still renders', html.length > 1000);
    ok('frame counter survives a counterless status line', html.includes('88,483'));
    ok('does not claim to be stalled', !html.includes('Stalled'));
  }
}

console.log('\n── cold start: nothing known yet ──────────────────────────');
{
  const cold = {
    processing: true, paused: false, progress: 0, desc: 'Starting…',
    error: '', output: null,
  };
  const { html } = render('cold start', tab(cold));
  if (html) {
    ok('renders with a near-empty payload', html.length > 500);
    ok('no undefined leaked', !html.includes('>undefined<'));
    ok('no NaN leaked', !html.includes('NaN'));
    ok('ETA admits it is estimating', html.includes('estimating'));
  }
}

console.log('\n── finished run (real captured payload) ───────────────────');
{
  const { html } = render('finished', tab(base));
  if (html) {
    ok('renders the finished state', html.length > 500);
    ok('reports completion', html.includes('Run complete') || html.includes('Run stopped'));
    ok('models panel is shown beside the output', html.includes('Models in use'));
    ok('attributes the output to the swapper that made it', html.includes('hyperswap'));
    ok('no undefined leaked', !html.includes('>undefined<'));
  }
}

console.log('\n── RunModelsPanel in isolation ────────────────────────────');
{
  // Degenerate shapes the backend can legitimately send.
  const cases = [
    ['no runtime at all', { runtime: null, settings, telemetry }],
    ['runtime without sections', { runtime: { model: 'inswapper' }, settings, telemetry }],
    ['sections present but MODEL missing', { runtime: { sections: {} }, settings, telemetry }],
    ['nothing whatsoever', { runtime: null, settings: null, telemetry: null }],
    ['empty objects', { runtime: {}, settings: {}, telemetry: {} }],
  ];
  for (const [name, props] of cases) {
    const { html } = render(`models: ${name}`, React.createElement(RunModelsPanel, props));
    if (html) {
      ok(`models: ${name} renders`, html.includes('Models in use'));
      ok(`models: ${name} leaks no undefined`, !html.includes('>undefined<'));
    }
  }
  // Provenance must be honest: settings-sourced data has to say so.
  const { html: stale } = render('models: settings only',
    React.createElement(RunModelsPanel, { runtime: null, settings, telemetry }));
  if (stale) {
    ok('settings-only render is labelled "from settings"', stale.includes('from settings'));
  }
  const { html: live } = render('models: runtime present',
    React.createElement(RunModelsPanel, { runtime: base.runtime, settings, telemetry }));
  if (live) {
    ok('runtime-backed render is labelled live', live.includes('live from pipeline'));
  }

  // ── The discriminating case ────────────────────────────────────────────
  // This is the WHOLE POINT of the panel, and the reason it does not simply
  // read `settings`: a run holds the models it started with, while Settings is
  // free to drift the moment someone touches a control in another tab. The two
  // fixtures above agree with each other, so neither can tell a runtime-sourced
  // panel from a settings-sourced one. Here they deliberately disagree.
  const runtimeSays = {
    sections: { MODEL: { values: {
      swap_model: 'hyperswap',
      selected_enhancer: 'Restore Ultra',
      detector_engine: 'retinaface_r50',
      mask_engine: 'DFL XSeg',
    } } },
  };
  const settingsSay = {
    ...settings,
    swap_model: 'inswapper_DRIFTED',
    selected_enhancer: 'GFPGAN_DRIFTED',
    detector_engine: 'yunet_DRIFTED',
    mask_engine: 'XSeg3_DRIFTED',
  };
  const { html: split } = render('models: runtime and settings disagree',
    React.createElement(RunModelsPanel,
      { runtime: runtimeSays, settings: settingsSay, telemetry }));
  if (split) {
    ok('swapper comes from the RUNTIME when the two disagree',
       split.includes('hyperswap') && !split.includes('inswapper_DRIFTED'));
    ok('enhancer comes from the RUNTIME when the two disagree',
       split.includes('Restore Ultra') && !split.includes('GFPGAN_DRIFTED'));
    ok('detector comes from the RUNTIME when the two disagree',
       split.includes('retinaface_r50') && !split.includes('yunet_DRIFTED'));
    ok('mask comes from the RUNTIME when the two disagree',
       split.includes('DFL XSeg') && !split.includes('XSeg3_DRIFTED'));
  }

  // And the converse: with no runtime at all it MUST fall back to settings,
  // rather than showing dashes, and must admit that is what it did.
  const { html: fallback } = render('models: settings fallback is real',
    React.createElement(RunModelsPanel,
      { runtime: null, settings: settingsSay, telemetry }));
  if (fallback) {
    ok('falls back to settings when there is no runtime',
       fallback.includes('inswapper_DRIFTED'));
    ok('and says so', fallback.includes('from settings'));
  }
  // A silent TensorRT -> CUDA fallback is the thing worth shouting about.
  const fellBack = {
    sections: {
      MODEL: { values: { swap_model: 'inswapper' } },
      PROVIDER: { values: { requested: 'tensorrt', effective: 'cuda' } },
    },
  };
  const { html: fb } = render('models: provider fallback',
    React.createElement(RunModelsPanel, { runtime: fellBack, settings, telemetry }));
  if (fb) {
    ok('a provider fallback is called out', fb.includes('was requested but the run is on'));
  }
}

console.log('\n── DiagnosticsPanel in isolation ──────────────────────────');
{
  const { html } = render('diagnostics: pushed counters',
    React.createElement(DiagnosticsPanel, {
      desc: 'Combining video…', telemetry, processing: true, paused: false,
      config: [['threads', '10']], elapsedMs: 600000, etaMs: 120000, prog: 0.9,
      framesDone: 88000, framesTotal: 88483, fps: 21.7,
    }));
  if (html) {
    ok('counters render from props, not the status string', html.includes('88,000'));
    ok('remaining is derived', html.includes('483'));
    ok('backend fps fills in before the local series warms up', html.includes('21.7'));
    ok('not reported as stalled', !html.includes('Stalled'));
    ok('no NaN leaked', !html.includes('NaN'));
  }
  const { html: noCounters } = render('diagnostics: no counters anywhere',
    React.createElement(DiagnosticsPanel, {
      desc: 'Starting…', telemetry: null, processing: true, paused: false,
      config: [], elapsedMs: 0, etaMs: 0, prog: 0,
    }));
  if (noCounters) {
    ok('degrades without counters or telemetry', noCounters.length > 500);
    ok('no NaN leaked', !noCounters.includes('NaN'));
  }
}

console.log(`\n${failures === 0 ? 'ALL GREEN' : 'FAILURES'}: ${checks - failures}/${checks} checks passed\n`);
// Reported through a global rather than process.exit(): this module is executed
// by Vite's SSR runner, which swallows the exit and would report success for a
// failing run. Verified by deliberately reverting the progress clamp — the
// checks failed, and the process still exited 0 until this was fixed.
globalThis.__RENDER_CHECK_FAILURES__ = failures;
