// The "Benchmark Pipeline Providers" result as rows a panel can print.
//
// The backend (`roop/bench.py`, profile `providers`) returns `provider_matrix`: per-stage ms/call on the CPU, on
// the primary provider and on native OpenVINO; the frame-level warp/blend and optical-flow backends; and a
// serial per-frame cost per ARM (CPU / primary / primary + OpenVINO aux / TRT + CV-CUDA + NVOFA).
//
// Everything is a pure function of that object. An arm or backend the machine cannot run arrives with a
// `reason` and no number and is shown WITH the reason, never as an estimate or a zero: "unavailable" is a
// result, a made-up figure is a defect.
import { fmtMs } from './reportFormat';

const DASH = '—';
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

/** Per-stage table rows: one per measured stage. A missing arm is a dash; an arm that errored says so. */
export function stageRows(matrix) {
  if (!matrix || !Array.isArray(matrix.stages)) return [];
  return matrix.stages.map((r) => ({
    key: r.key || r.stage,
    stage: r.stage,
    callsPerFrame: isNum(r.calls_per_frame) ? r.calls_per_frame : null,
    primaryProvider: r.primary_provider || 'primary',
    primary: fmtMs(r.primary_ms),
    cpu: r.cpu_error ? 'error' : fmtMs(r.cpu_ms),
    openvino: r.openvino_error ? 'error' : fmtMs(r.openvino_ms),
    openvinoDevice: r.openvino_device || '',
    note: r.cpu_error || r.openvino_error || '',
    // Fastest measured arm for this stage, so the table can mark it. null when fewer than two arms ran.
    fastest: (() => {
      const arms = [['primary', r.primary_ms], ['cpu', r.cpu_ms], ['openvino', r.openvino_ms]]
        .filter(([, v]) => isNum(v) && v > 0);
      if (arms.length < 2) return null;
      return arms.reduce((a, b) => (b[1] < a[1] ? b : a))[0];
    })(),
  }));
}

/** Stages the per-stage measurement could not time. Listed, never dropped: the serial figures need every stage. */
export function unmeasuredRows(matrix) {
  if (!matrix || !Array.isArray(matrix.unmeasured)) return [];
  return matrix.unmeasured.map((u) => ({ stage: u.stage, reason: u.reason || 'no timing was produced' }));
}

/** The four arms: a number (ms per frame, serial) or the reason there is none. */
export function armCards(matrix) {
  if (!matrix || !Array.isArray(matrix.arms)) return [];
  const fastest = matrix.arms.filter((a) => a.available && isNum(a.serial_ms_frame))
    .reduce((best, a) => (!best || a.serial_ms_frame < best.serial_ms_frame ? a : best), null);
  return matrix.arms.map((a) => ({
    key: a.key,
    label: a.label,
    available: !!a.available && isNum(a.serial_ms_frame),
    value: a.available && isNum(a.serial_ms_frame) ? `${fmtMs(a.serial_ms_frame)} / frame` : 'unavailable',
    sub: a.available && isNum(a.serial_fps) ? `${a.serial_fps.toFixed(1)} fps serial` : (a.reason || DASH),
    fastest: !!fastest && fastest.key === a.key,
  }));
}

/** Warp/blend and optical-flow backend lists (`{backends: [{label, available, ms, reason}]}`). */
export function backendRows(section) {
  if (!section || !Array.isArray(section.backends)) return [];
  return section.backends.map((b) => ({
    key: b.key,
    label: b.label,
    available: !!b.available && isNum(b.ms),
    value: b.available && isNum(b.ms) ? fmtMs(b.ms) : 'unavailable',
    reason: b.available && isNum(b.ms) ? '' : (b.reason || ''),
  }));
}

/** One honest line under the table: what a number here means and does not mean. */
export const matrixNote = (matrix) => (matrix && matrix.note) || '';

/**
 * The benchmark state the status endpoint returns -> what the panel shows.
 * `phase`: 'idle' (never run) | 'running' | 'failed' | 'cancelled' | 'done'.
 */
export function benchPhase(state) {
  if (!state || typeof state !== 'object') return 'idle';
  if (state.running) return 'running';
  if (state.error) return 'failed';
  if (state.result && state.result.provider_matrix) return 'done';
  if (typeof state.status_msg === 'string' && /cancel/i.test(state.status_msg)) return 'cancelled';
  return 'idle';
}
