import { create } from 'zustand';

// ── High-frequency telemetry, OUTSIDE React state ─────────────────────────
//
// Before this, every pushed telemetry frame (4 Hz, /ws/telemetry) went through
// `setProgress` in App — so App, and with it whichever tab was mounted,
// re-rendered four times a second for the whole length of a render. On Face
// Swap that is the 4,000-line panel with its thirty-odd hooks; on Processing it
// is the rolling 250-line terminal. All to move a percentage and an fps figure.
//
// Now the fast fields land HERE, and App's `progress` state only changes when
// something structural does (processing, paused, error, a stop/pause request)
// or when the slow poll brings the log. Consumers that show a fast number
// subscribe to exactly that number through the leaf components in
// components/LiveTelemetry.jsx, which re-render themselves — or write a DOM
// node directly — at no more than UI_HZ.
//
// The store is written by transport code (App.onTelemetry, App.mergeProgress,
// the system-telemetry poller) with setState. Nothing reads it during render
// except through the throttled hooks below.

/** The most any telemetry-driven piece of UI updates per second. */
export const UI_HZ = 10;

const EMPTY_RUN = {
  progress: 0,
  desc: '',
  status_line: '',
  fps: 0,
  fps_now: null,
  frame_ms: null,
  current_frame: 0,
  total_frames: 0,
  eta_s: null,
  live_seq: 0,
  started_at: 0,
  // The backend compiling a TensorRT engine / running a startup canary (pipeline_status.telemetry_fields).
  // Flat primitives on purpose: this store compares fields with `!==`, and the sampler only sends a
  // frame that changed, so a stage that takes 15 minutes is silent apart from the 15 s heartbeat.
  engine_state: 'IDLE',
  engine_label: '',
  engine_model: '',
  engine_precision: '',
  engine_since: null,
  // Video I/O (roop/video_io_hw.py): the decode / encode backend this job really got ('NVDEC-DLPack',
  // 'NVDEC-Pipe', 'QSV-oneVPL', 'NVENC-Direct', 'NVENC-Pipe', 'CPU-Fallback'), PCIe traffic avoided versus a
  // host-staged pipeline in whole MB (an ESTIMATE), and the share of decoder reads served without waiting (%).
  // Kept after the run ends, like the report; reset by the next job's first frame.
  io_decode: '',
  io_encode: '',
  io_pcie_saved_mb: 0,
  io_queue_sat: null,
};

/** The fields a telemetry frame carries that change every tick. */
export const FAST_FIELDS = Object.keys(EMPTY_RUN);

// The 3-stage pipeline view and the finished per-job report (roop/stage_telemetry.py): `stages` is the
// live state of Stage 1-3, `summary` the report the backend builds when the render ends ({} until then).
// Objects, not flat fields, so they are compared by value below instead of with `!==`.
const EMPTY_PIPELINE = { stage: null, stages: {}, summary: {} };

export const useTelemetryStore = create(() => ({
  run: EMPTY_RUN,
  pipeline: EMPTY_PIPELINE,
  /** /api/system/telemetry: gpu, vram_used/total, cpu_percent, ram_*, threads. */
  system: null,
}));

/** Merge fast run fields. Only keys present in `patch` are touched. */
export function setRunTelemetry(patch) {
  if (!patch) return;
  const cur = useTelemetryStore.getState().run;
  let changed = false;
  const next = { ...cur };
  for (const k of FAST_FIELDS) {
    if (Object.prototype.hasOwnProperty.call(patch, k) && patch[k] !== cur[k]) {
      next[k] = patch[k];
      changed = true;
    }
  }
  if (changed) useTelemetryStore.setState({ run: next });
}

/** Take the `stage` / `stages` / `summary` of a telemetry frame; a no-op when none of them changed. */
export function setPipelineTelemetry(frame) {
  if (!frame || typeof frame !== 'object') return;
  if (!('stages' in frame) && !('summary' in frame)) return;
  const next = {
    stage: frame.stage ?? null,
    stages: frame.stages || {},
    summary: frame.summary || {},
  };
  const cur = useTelemetryStore.getState().pipeline;
  if (cur && JSON.stringify(cur) === JSON.stringify(next)) return;
  useTelemetryStore.setState({ pipeline: next });
}

export function resetRunTelemetry() {
  useTelemetryStore.setState({ run: EMPTY_RUN });
}

/**
 * Forget the engine claim. "Compiling..." is a statement about a backend we can currently hear; once
 * the socket is down it is a statement nobody is confirming, and a backend that died mid-build would
 * otherwise show it forever. The next frame after a reconnect restores the truth.
 */
export function clearEngineTelemetry() {
  setRunTelemetry({
    engine_state: EMPTY_RUN.engine_state, engine_label: '', engine_model: '',
    engine_precision: '', engine_since: null,
  });
}

/**
 * The line to show while the backend is building / verifying an engine: "Compiling TensorRT Engine...
 * gpen_256_pro · FP16", or '' when it is not. Pure so it can be checked without a render.
 */
export const engineTextOf = (run) => {
  if (!run || !run.engine_label) return '';
  const parts = [run.engine_model, run.engine_precision].filter(Boolean).join(' · ');
  return parts ? `${run.engine_label} ${parts}` : run.engine_label;
};

/**
 * "Video I/O: NVDEC-DLPack -> NVENC-Direct · PCIe saved ~2480 MB (est.) · decode queue 74%", or '' before a job
 * has registered a backend. Pure so it can be checked without a render.
 */
export const videoIoTextOf = (run) => {
  if (!run || (!run.io_decode && !run.io_encode)) return '';
  const path = [run.io_decode || '—', run.io_encode || '—'].join(' → ');
  const parts = [`Video I/O: ${path}`];
  if (run.io_pcie_saved_mb > 0) parts.push(`PCIe saved ~${run.io_pcie_saved_mb} MB (est.)`);
  if (run.io_queue_sat != null) parts.push(`decode queue ${run.io_queue_sat}%`);
  return parts.join(' · ');
};

export function setSystemTelemetry(system) {
  useTelemetryStore.setState({ system });
}

export const getRun = () => useTelemetryStore.getState().run;

// ── Derived values every consumer used to compute for itself ─────────────

/** Progress as a clamped 0..1 (the backend can overshoot; see Processing). */
export const selectProg = (s) => {
  const p = Number(s.run.progress);
  return Number.isFinite(p) ? Math.min(1, Math.max(0, p)) : 0;
};

/** {done, total} from the pushed counters, else parsed from the status line. */
export const framesOf = (run) => {
  const done = Number(run.current_frame);
  const total = Number(run.total_frames);
  if (Number.isFinite(done) && Number.isFinite(total) && total > 0) return { done, total };
  const m = /(\d[\d,]*)\s*\/\s*(\d[\d,]*)/.exec(run.desc || '');
  if (!m) return null;
  const d = parseInt(m[1].replace(/,/g, ''), 10);
  const t = parseInt(m[2].replace(/,/g, ''), 10);
  if (!Number.isFinite(d) || !Number.isFinite(t) || t <= 0) return null;
  return { done: d, total: t };
};

/**
 * Milliseconds left: the terminal's own eta_s where a bar is counting, else
 * the elapsed x (1 - p) / p extrapolation (start-up, encode tail). 0 = unknown.
 */
export const etaMsOf = (run, elapsedMs) => {
  if (typeof run.eta_s === 'number' && run.eta_s > 0) return run.eta_s * 1000;
  const p = Math.min(1, Math.max(0, Number(run.progress) || 0));
  return p > 0.01 && elapsedMs > 0 ? (elapsedMs * (1 - p)) / p : 0;
};
