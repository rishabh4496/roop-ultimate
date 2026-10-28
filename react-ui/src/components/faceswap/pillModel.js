// The 3-Step Pipeline Pill's model: a pure function of the telemetry store's `pipeline` and `run` slices.
//
//   1 Pre-Scan      Stage 1 (sparse keyframe scan: which frames can hold the target at all)
//   2 Range Lock    Stage 2 (tracking pre-pass over the active ranges: the frames that WILL be swapped are fixed)
//   3 Render & Mux  Stage 3 (the swap) and Stage 4 (stream-copy the source audio on and verify the file)
//
// Everything comes from /ws/telemetry (`stages`, `summary`, `fps_now`); the pill makes no REST call, so it cannot
// run into the 15 s API deadline (api.js) however long a stage takes. Stage 4 is not a progress stage in the
// backend's `stages` table: it shows as step 3 staying active ("Muxing") between Stage 3 finishing and the
// report arriving, and step 3 only reads done when `summary` carries the finished report.
//
// "Culled" is the share of the trim Stage 1 proved target-free and Stage 3 passes through untouched
// (`skipped_frames_pct`). It is null when Stage 1 did not run, and then it is not shown rather than shown as 0.

import { STAGE_1, STAGE_2, STAGE_3, fmtPct } from './reportFormat';

export const PILL_STEPS = Object.freeze([
  { n: 1, key: 'prescan', label: 'Pre-Scan', stage: STAGE_1 },
  { n: 2, key: 'rangelock', label: 'Range Lock', stage: STAGE_2 },
  { n: 3, key: 'render', label: 'Render & Mux', stage: STAGE_3 },
]);

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

/** Backend stage state -> the pill's step state. */
function stepState(backend) {
  switch (backend) {
    case 'RUNNING': return 'active';
    case 'DONE': return 'done';
    case 'RESTORED': return 'restored';
    case 'SKIPPED': return 'skipped';
    case 'FAILED': return 'failed';
    default: return 'pending';
  }
}

/** Words for a step state; the pill prints them, so state never rests on colour alone. */
export const STATE_WORD = Object.freeze({
  active: 'running', done: 'done', restored: 'resumed from checkpoint',
  skipped: 'not needed', failed: 'failed', pending: 'waiting',
});

/** The share of the trim the scan proved target-free, or null when no stage reports one. */
export function culledPct(pipeline) {
  const stages = pipeline?.stages || {};
  for (const name of [STAGE_1, STAGE_2, STAGE_3]) {
    const v = stages[name]?.skipped_frames_pct;
    if (isNum(v)) return Math.max(0, Math.min(100, v));
  }
  return null;
}

/** "46.7% culled" / ''. */
export const culledText = (pipeline) => {
  const v = culledPct(pipeline);
  return v == null ? '' : `${fmtPct(v)} culled`;
};

/** Which of the three steps is the active one (1-3), or 0 when none is. */
export function activeStep(model) {
  const hit = model?.steps?.find((s) => s.state === 'active');
  return hit ? hit.n : 0;
}

/**
 * The pill: `null` when the backend has reported no run at all (an idle app), else
 * `{ steps: [{ n, key, label, state, word, detail }], culled, activeN, announce }`.
 * `processing` is whether a job is running now (Stage 4 is invisible to `stages`, so it is what tells "Stage 3
 * finished, still muxing" from "the run is over").
 */
export function pillModel(pipeline, { processing = true } = {}) {
  const stages = pipeline?.stages || {};
  if (!Object.keys(stages).length) return null;
  const summary = pipeline.summary && pipeline.summary.stages ? pipeline.summary : null;

  const steps = PILL_STEPS.map((def) => {
    const backend = stages[def.stage]?.state || 'PENDING';
    let state = stepState(backend);
    let detail = '';
    const pct = stages[def.stage]?.stage_progress_pct;
    if (def.n === 3) {
      if (summary) {
        state = summary.ok === false || summary.stages?.STAGE_4_FINALIZE?.ok === false ? 'failed' : 'done';
      } else if (state === 'done' && processing) {
        state = 'active';           // Stage 3 is over; Stage 4 (mux + verify) has not reported yet
        detail = 'Muxing';
      }
    }
    if (state === 'active' && !detail && isNum(pct)) detail = fmtPct(pct);
    if (state === 'restored') detail = 'resumed';
    return { n: def.n, key: def.key, label: def.label, state, word: STATE_WORD[state], detail };
  });

  const model = { steps, culled: culledPct(pipeline), activeN: 0, announce: '' };
  model.activeN = activeStep(model);
  const failed = steps.find((s) => s.state === 'failed');
  if (failed) model.announce = `Step ${failed.n} of 3, ${failed.label}, failed`;
  else if (model.activeN) {
    const a = steps[model.activeN - 1];
    model.announce = `Step ${a.n} of 3, ${a.label}${a.detail === 'Muxing' ? ', muxing audio' : ''}`
      + (model.culled == null ? '' : `, ${fmtPct(model.culled)} of frames skipped`);
  } else if (steps.every((s) => s.state === 'done' || s.state === 'skipped' || s.state === 'restored')) {
    model.announce = 'Render finished';
  }
  return model;
}

/**
 * "Lip-sync 4.2 ms" beside step 3 while audio-driven sync is the job's mode and has been timed, else ''. The figure is the
 * Lip-Sync column of the dashboard's latency split (ms per delivered frame), so the pill, the console mirror and the
 * terminal block show one number.
 */
export function lipsyncText(run) {
  if (!run || run.mouth_mode !== 'audio_driven_sync' || run.mouth_fallback) return '';
  const v = run.lipsync_ms;
  return isNum(v) && v >= 0 ? `Lip-sync ${v.toFixed(1)} ms` : '';
}

/** The reason the mouth is not being driven by audio even though it was asked to, or ''. Stated, never implied. */
export function mouthFallbackText(run) {
  if (!run || run.mouth_mode !== 'audio_driven_sync' || !run.mouth_fallback) return '';
  return `Lip-sync off: ${run.mouth_fallback}. Using Source mouth restore.`;
}

/** Frames per second to show beside step 3: the current rate, else the run average; '' outside step 3. */
export function fpsText(run, model) {
  if (!model || model.activeN !== 3) return '';
  const v = isNum(run?.fps_now) && run.fps_now > 0 ? run.fps_now : run?.fps;
  return isNum(v) && v > 0 ? `${v.toFixed(1)} fps` : '';
}

// ── Stage 24: Modular Pipeline Flow Strip Model ────────────────────────────
export const MODULAR_FLOW_NODES = Object.freeze([
  {
    id: 'detect_align',
    n: 1,
    key: 'detect',
    label: 'Detect & Align',
    shortLabel: 'Detect',
    latencyKey: 'warp',
    fallbackMs: 3.4,
  },
  {
    id: 'hyperswap',
    n: 2,
    key: 'swap',
    label: 'HyperSwap',
    shortLabel: 'Swap',
    latencyKey: 'model',
    fallbackMs: 11.2,
  },
  {
    id: 'gaze_mouth',
    n: 3,
    key: 'gaze_mouth',
    label: 'Gaze & Mouth',
    shortLabel: 'Gaze/Mouth',
    latencyKey: 'lipsync',
    fallbackMs: 4.2,
  },
  {
    id: 'xseg_roto',
    n: 4,
    key: 'mask',
    label: 'XSeg3 + Roto',
    shortLabel: 'Mask',
    latencyKey: 'mask',
    fallbackMs: 6.8,
  },
  {
    id: 'restore_ultra',
    n: 5,
    key: 'restore',
    label: 'Restore Ultra',
    shortLabel: 'Restore',
    latencyKey: 'restore',
    fallbackMs: 14.5,
  },
  {
    id: 'relight_lut',
    n: 6,
    key: 'blend',
    label: 'Relight & LUT',
    shortLabel: 'Relight',
    latencyKey: 'blend',
    fallbackMs: 4.1,
  },
]);

/** Latency color code: green (<5ms), amber (5-15ms), red (>15ms). */
export function latencyTone(ms) {
  if (ms == null || !Number.isFinite(ms)) return 'text-muted border-white/10 bg-white/5';
  if (ms < 5.0) return 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10';
  if (ms <= 15.0) return 'text-amber-400 border-amber-500/30 bg-amber-500/10';
  return 'text-rose-400 border-rose-500/30 bg-rose-500/10';
}

/** Formatted latency label (e.g. '4.1ms'). */
export function formatLatency(ms) {
  if (ms == null || !Number.isFinite(ms) || ms < 0) return '– ms';
  return `${ms.toFixed(1)}ms`;
}

