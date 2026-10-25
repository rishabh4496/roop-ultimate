// The per-job 3-stage + finalization report, as rows a panel can print.
//
// The backend builds ONE report when a render ends (`roop/stage_telemetry.build_summary`) and sends it on
// /ws/telemetry (`summary`), in /api/runtime/state (PERFORMANCE.pipeline_report) and in each history entry
// (`report`). Everything here is a pure function of that object, so it can be checked without a render and
// so the Processing HUD, the Settings panel and the History row cannot disagree about what a number means.
//
// A metric the backend did not measure arrives as null and is shown as an em dash, never as 0: a stage
// that was skipped or restored from a checkpoint has no wall clock, and "0.0 s" would claim it was instant.

export const STAGE_1 = 'STAGE_1_SPARSE_PRESCAN';
export const STAGE_2 = 'STAGE_2_ACTIVE_PREPASS';
export const STAGE_3 = 'STAGE_3_MAIN_SWAP';
export const STAGE_4 = 'STAGE_4_FINALIZE';
export const STAGE_ORDER = [STAGE_1, STAGE_2, STAGE_3, STAGE_4];

export const STAGE_LABEL = {
  [STAGE_1]: 'Stage 1 · Sparse pre-scan',
  [STAGE_2]: 'Stage 2 · Active-range pre-pass',
  [STAGE_3]: 'Stage 3 · Main swap',
  [STAGE_4]: 'Stage 4 · Finalize',
};

const DASH = '—';
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

export const fmtSecs = (v) => {
  if (!isNum(v)) return DASH;
  if (v >= 60) return `${Math.floor(v / 60)}m ${String(Math.round(v % 60)).padStart(2, '0')}s`;
  return `${v.toFixed(v >= 10 ? 1 : 2)}s`;
};
export const fmtCount = (v) => (isNum(v) ? Math.round(v).toLocaleString('en-US') : DASH);
export const fmtPct = (v) => (isNum(v) ? `${v.toFixed(1)}%` : DASH);
export const fmtGb = (v) => (isNum(v) ? `${v.toFixed(2)} GB` : DASH);
export const fmtFps = (v) => (isNum(v) ? `${v.toFixed(1)} fps` : DASH);
export const fmtMs = (v) => (isNum(v) ? `${v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2)} ms` : DASH);
export const fmtRate = (v, unit) => (isNum(v) ? `${v.toFixed(1)} ${unit}` : DASH);

/** done / running / failed / skipped / restored / pending, for a state chip. */
export const stateTone = (state) => {
  switch (state) {
    case 'DONE': return 'text-emerald-400';
    case 'RUNNING': return 'text-[var(--accent)]';
    case 'FAILED': return 'text-red-300';
    case 'RESTORED': return 'text-sky-300';
    case 'SKIPPED': return 'text-muted';
    default: return 'text-muted';
  }
};

/** Speedup multiplier as the backend's own words allow: always marked as an estimate. */
export const speedupText = (speedup) => {
  if (!speedup || !isNum(speedup.multiplier)) return DASH;
  return `≈ ${speedup.multiplier.toFixed(2)}×${speedup.estimated ? ' (estimate)' : ''}`;
};

const row = (label, value, tone) => ({ label, value, tone: tone || '' });

/**
 * Stage 3's per-operation timings as HUD cells, in pipeline order. `ops` is `stage3.ops` from the backend: one
 * entry per column with `{ms, calls, backend}`, or null for a column that never ran (shown as a dash, never 0).
 * The labels follow the backend's own backend name: a TensorRT model column is "TRT", an *_nvenc encoder is
 * "NVENC"; anything else keeps a neutral name rather than claiming hardware it did not use.
 */
export const OPS_ORDER = ['decode', 'warp', 'model', 'restore', 'blend', 'encode'];
export function opsCells(ops) {
  if (!ops || typeof ops !== 'object') return [];
  return OPS_ORDER.map((key) => {
    const o = ops[key] || null;
    const backend = o && o.backend ? String(o.backend) : '';
    let label = { decode: 'Decode', warp: 'Warp', model: 'Model', restore: 'Restore', blend: 'Blend', encode: 'Encode' }[key];
    // The label follows the backend's own name for what ran, never the other way round: a CUDA-graph replay is "TRT
    // Graph", a fused Triton kernel is "Triton Blend", and anything else (today the blend is cv2 on the CPU) keeps
    // its neutral name rather than claiming hardware it did not use.
    if (key === 'model' && /^tensorrt|^trt/i.test(backend)) label = /graph/i.test(backend) ? 'TRT Graph' : 'TRT';
    if (key === 'blend' && /triton/i.test(backend)) label = 'Triton Blend';
    if (key === 'encode' && /nvenc/i.test(backend)) label = 'NVENC';
    return {
      key, label, backend,
      value: o && isNum(o.ms) ? fmtMs(o.ms) : DASH,
      calls: o && isNum(o.calls) ? o.calls : null,
      title: o ? `${label}: ${fmtMs(o.ms)} per call over ${fmtCount(o.calls)} calls${backend ? ` on ${backend}` : ''}` : `${label}: did not run`,
    };
  });
}

/**
 * Stage 2's optical-flow engine as the backend names it (`flow_backend`, roop/flow_backend.py): `nvof` is the NVIDIA
 * optical-flow ASIC, `cv` OpenCV on the CPU, `torch` a GPU fallback. '' for an unknown / absent name, so a caller can
 * fall back to the neutral word "flow" instead of asserting an engine nobody reported.
 */
export const FLOW_ENGINE_LABEL = { nvof: 'NVOFA', cv: 'CPU flow', torch: 'GPU flow' };
export const flowEngineLabel = (backend) => FLOW_ENGINE_LABEL[String(backend || '').toLowerCase()] || '';

/** "Decode 2.1 ms | Warp 3.4 ms | TRT 11 ms | ..." - the brief's one-line Stage 3 readout. Empty when no ops. */
export const opsLine = (ops) => opsCells(ops).map((c) => `${c.label} ${c.value}`).join(' | ');

/** The OpenVINO aux offload block (`report.aux_offload`) as one line, or '' when the runner never ran. */
export function auxOffloadText(aux) {
  if (!aux || !Array.isArray(aux.sessions) || !aux.sessions.length) return '';
  return aux.sessions
    .map((x) => `${x.model_key || 'model'} on ${x.device || '?'}${x.device_name ? ` (${x.device_name})` : ''}`
      + `${x.precision ? ` ${x.precision}` : ''}${isNum(x.ms_per_call) ? ` · ${fmtMs(x.ms_per_call)}/call` : ''}`)
    .join(' · ');
}

/**
 * Sections of label/value rows for a finished report. Returns [] for a missing / empty report so a caller
 * can render "no finished run yet" instead of a table of dashes.
 */
export function reportSections(report) {
  if (!report || typeof report !== 'object' || !report.stages) return [];
  const s = report.stages;
  const s1 = s[STAGE_1] || {};
  const s2 = s[STAGE_2] || {};
  const s3 = s[STAGE_3] || {};
  const s4 = s[STAGE_4] || null;
  const out = [];

  out.push({
    key: STAGE_1, label: STAGE_LABEL[STAGE_1], state: s1.state || 'PENDING',
    rows: [
      row('Wall clock', fmtSecs(s1.wall_s)),
      row('Sampled frames', isNum(s1.sampled_frames) && isNum(s1.samples_total)
        ? `${fmtCount(s1.sampled_frames)} / ${fmtCount(s1.samples_total)}` : fmtCount(s1.sampled_frames)),
      row('Worker pool utilization', isNum(s1.worker_utilization_pct) && isNum(s1.workers)
        ? `${fmtPct(s1.worker_utilization_pct)} of ${s1.workers} workers` : fmtPct(s1.worker_utilization_pct)),
      row('Keyframes sampled / s', fmtRate(s1.keyframes_per_sec, '/s')),
      row('Scene cuts found', fmtCount(s1.scene_cuts)),
      row('Active ranges', fmtCount(s1.active_ranges_count)),
      row('Frames skipped (no target)', fmtPct(s1.skipped_frames_pct)),
    ],
  });

  out.push({
    key: STAGE_2, label: STAGE_LABEL[STAGE_2], state: s2.state || 'PENDING',
    rows: [
      row('Wall clock', fmtSecs(s2.wall_s)),
      row('Coalesced active ranges', fmtCount(s2.active_ranges_count)),
      row('cap.set() seeks', fmtCount(s2.set_calls)),
      row('cap.grab() frames walked', fmtCount(s2.grab_calls)),
      row('Verified target frames', fmtCount(s2.verified_target_frames)),
      // What the pre-pass actually ran: absent (a dash) when Stage 2 was restored from a checkpoint.
      row('Detector calls', isNum(s2.detector_calls) && isNum(s2.scan_frames)
        ? `${fmtCount(s2.detector_calls)} of ${fmtCount(s2.scan_frames)} frames${isNum(s2.detect_interval) && s2.detect_interval > 1 ? ` (every ${s2.detect_interval}th)` : ''}`
        : '—'),
      row('Neural-detected frames', fmtCount(s2.neural_detected_frames)),
      row('Flow-tracked frames', isNum(s2.flow_tracked_frames) && isNum(s2.flow_ratio_pct)
        ? `${fmtCount(s2.flow_tracked_frames)} (${fmtPct(s2.flow_ratio_pct)})` : fmtCount(s2.flow_tracked_frames)),
      ...(flowEngineLabel(s2.flow_backend) ? [row('Flow engine', flowEngineLabel(s2.flow_backend))] : []),
      row('Pre-pass speed', fmtRate(s2.fps, 'fps')),
    ],
  });

  out.push({
    key: STAGE_3, label: STAGE_LABEL[STAGE_3], state: s3.state || 'PENDING',
    rows: [
      row('Wall clock', fmtSecs(s3.wall_s)),
      row('Zero-GPU passthrough frames', fmtCount(s3.passthrough_frames)),
      row('GPU frames (swap / enhance)', fmtCount(s3.gpu_frames)),
      row('Frames with a swapped face', fmtCount(s3.swapped_frames)),
      row('Faces swapped', fmtCount(s3.faces_swapped)),
      row('Peak VRAM', fmtGb(s3.peak_vram_gb)),
      ...opsCells(s3.ops).map((c) => row(`${c.label} per call`, c.value + (c.backend ? ` · ${c.backend}` : ''))),
    ],
  });

  if (s4) {
    const ok = !!s4.ok;
    const color = s4.color || {};
    const rows = [
      row('Verification', ok ? 'passed' : 'FAILED', ok ? 'text-emerald-400' : 'text-red-300'),
      row('Frames delivered', isNum(s4.expected_frames)
        ? `${fmtCount(s4.frames)} of ${fmtCount(s4.expected_frames)} expected` : fmtCount(s4.frames)),
      row('Audio start vs video', isNum(s4.audio_start_offset_ms)
        ? `${s4.audio_start_offset_ms > 0 ? '+' : ''}${s4.audio_start_offset_ms.toFixed(1)} ms`
        : (s4.audio_present === false ? 'no audio track' : DASH)),
      row('Colour tags', color.color_primaries
        ? `${color.color_primaries} · ${color.color_space || '?'}${s4.color_preserved === false ? ' (CHANGED)' : ''}`
        : DASH, s4.color_preserved === false ? 'text-red-300' : ''),
      row('Mux + verify', `${fmtSecs(s4.mux_s)} + ${fmtSecs(s4.wall_s)} total`),
      row('Temp files', s4.orphans && s4.orphans.length ? `${s4.orphans.length} left behind` : 'none left',
        s4.orphans && s4.orphans.length ? 'text-amber-300' : ''),
    ];
    (s4.problems || []).forEach((p) => rows.push(row('Problem', p, 'text-red-300')));
    out.push({ key: STAGE_4, label: STAGE_LABEL[STAGE_4], state: s4.state || (ok ? 'DONE' : 'FAILED'), rows });
  }
  return out;
}

/** The one-line summary a collapsed card or a history row shows. */
export function headline(report) {
  if (!report || typeof report !== 'object' || !report.stages) return null;
  return {
    ok: report.ok !== false,
    fps: fmtFps(report.effective_fps),
    wall: fmtSecs(report.total_wall_s),
    speedup: speedupText(report.speedup),
    vram: fmtGb(report.vram && report.vram.peak_gb),
    frames: fmtCount(report.total_frames),
  };
}

/**
 * Canary precision status: what the fidelity canaries (roop/aux_canary.py, swap_canary.py) decided about the engines
 * this machine runs. `precision` is `runtime.sections.PRECISION.values` (configured / effective precision plus
 * `canary_verdicts`, `{stem: {passed, reason}}`); `run` is the telemetry store's `run` slice, which says whether a canary
 * is running RIGHT NOW. Pure, so it can be checked without a render.
 *
 *   RUNNING    a canary is verifying an engine now (`engine_state === 'RUNNING_CANARY'`)
 *   FAILED     at least one verdict is `passed === false` - that engine was stepped down
 *   PASSED     every recorded verdict passed
 *   NONE       no verdict recorded: not "passed", the canaries simply have not run (or none is persisted yet)
 *
 * A verdict whose `passed` is neither true nor false (a skipped canary) is listed as UNKNOWN and counts toward neither
 * PASSED nor FAILED, so it can never make a run look verified.
 */
export function canaryView(precision, run) {
  const values = precision && typeof precision === 'object' ? precision : {};
  const verdicts = values.canary_verdicts && typeof values.canary_verdicts === 'object' ? values.canary_verdicts : {};
  const rows = Object.entries(verdicts)
    .map(([stem, v]) => ({
      stem,
      result: v && v.passed === true ? 'PASS' : v && v.passed === false ? 'FAIL' : 'UNKNOWN',
      reason: v && typeof v.reason === 'string' ? v.reason : '',
    }))
    .sort((a, b) => (a.result === b.result ? a.stem.localeCompare(b.stem) : a.result === 'FAIL' ? -1 : b.result === 'FAIL' ? 1 : 0));
  const failed = rows.filter((r) => r.result === 'FAIL').length;
  const passed = rows.filter((r) => r.result === 'PASS').length;
  const live = run && run.engine_state === 'RUNNING_CANARY'
    ? { model: run.engine_model || 'model', precision: run.engine_precision || '' } : null;
  let state = 'NONE';
  if (live) state = 'RUNNING';
  else if (failed) state = 'FAILED';
  else if (passed && passed === rows.length) state = 'PASSED';
  else if (rows.length) state = 'UNKNOWN';
  const pick = (k) => (values[k] === undefined || values[k] === null || values[k] === '' ? null : String(values[k]));
  return { state, rows, passed, failed, live, configured: pick('configured'), effective: pick('effective') };
}

export const canaryTone = (state) => (state === 'PASSED' ? 'text-emerald-400' : state === 'FAILED' ? 'text-red-300'
  : state === 'RUNNING' ? 'text-[var(--accent)]' : 'text-muted');
