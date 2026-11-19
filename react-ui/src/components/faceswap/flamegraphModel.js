export const FLAMEGRAPH_STAGES = [
  { key: 'decode', label: 'Decode', model: 'Stabilizer', color: '#06b6d4', bgClass: 'bg-cyan-400', textClass: 'text-cyan-400', residency: 'CPU (Worker / NVDEC)' },
  { key: 'detect', label: 'Detect/Align', model: 'SCRFD-gnkps', color: '#a855f7', bgClass: 'bg-purple-500', textClass: 'text-purple-400', residency: 'GPU (TensorRT/CUDA)' },
  { key: 'swap', label: 'Swap', model: 'HyperSwap', color: '#10b981', bgClass: 'bg-emerald-500', textClass: 'text-emerald-400', residency: 'GPU (TensorRT/CUDA)' },
  { key: 'mask', label: 'Mask/Roto', model: 'BiSeNet+XSeg', color: '#3b82f6', bgClass: 'bg-blue-500', textClass: 'text-blue-400', residency: 'GPU (TensorRT/CUDA)' },
  { key: 'restore', label: 'Restore Ultra', model: 'UltraMax', color: '#f59e0b', bgClass: 'bg-amber-500', textClass: 'text-amber-400', residency: 'GPU (TensorRT/CUDA)' },
  { key: 'blend', label: 'Relight/LUT', model: 'Laplacian Blend', color: '#ec4899', bgClass: 'bg-pink-500', textClass: 'text-pink-400', residency: 'GPU (Color Warp)' },
  { key: 'encode', label: 'Encode', model: 'NVENC', color: '#64748b', bgClass: 'bg-slate-500', textClass: 'text-slate-400', residency: 'GPU (NVENC / Direct)' },
];

// These are the backend's measured worker-time columns. Values are emitted in
// milliseconds per delivered frame; absent columns stay absent instead of
// being replaced with demo/default timings.
export const TRACE_STAGES = [
  { key: 'decode', label: 'Decode', aliases: ['decode'], color: '#06b6d4' },
  { key: 'detection', label: 'Detection', aliases: ['detection', 'detect', 'alignment'], color: '#a855f7' },
  { key: 'embedding', label: 'ArcFace Embedding', aliases: ['arcface_embedding', 'embedding'], color: '#c084fc' },
  { key: 'swap', label: 'Hyperswap / Swapper', aliases: ['swap'], color: '#10b981' },
  { key: 'mask', label: 'XSeg 3 Mask', aliases: ['xseg3', 'xseg', 'mask'], color: '#3b82f6' },
  { key: 'restore', label: 'Restore Ultra / UltraMax', aliases: ['restore', 'enhancement', 'enhance', 'enhance_recombine', 'identity_detail'], color: '#f59e0b' },
  { key: 'color', label: 'LUT / Color', aliases: ['lut_color', 'color', 'lighting', 'detail_transfer', 'realism', 'blend'], color: '#ec4899' },
  { key: 'encode', label: 'NVHardwareVideoWriter Encode', aliases: ['encoding', 'encode'], color: '#64748b' },
  // DERIVED, never measured: end-to-end frame time minus the lanes above. The backend also times stabilize, temporal_*,
  // expression, lipsync, verify and faceset_lookup, but those timers are not mapped to a lane on purpose: several
  // others (mask_refine / mask_stabilize / occlusion_analysis inside `mask`, interaction_occlusion / temporal_alpha inside
  // `blend`) run INSIDE a mapped timer, so summing them as well would count the same milliseconds twice. The residual
  // is correct by construction whichever of them nests where.
  { key: 'other', label: 'Unattributed (stabilize, temporal, verify, ...)', aliases: [], derived: true, color: '#94a3b8' },
];

/**
 * One backend trace row -> per-lane milliseconds.
 *
 * `rawStages` is the row's `stage_ms` (profiler name -> ms), `frameMs` its `frame_ms` (end-to-end, or null).
 * A lane with no measured alias is `null` (absent, never a default). `other` is `frameMs - sum(measured lanes)`, or
 * `null` when there is no end-to-end figure. If the measured lanes add up to MORE than the end-to-end time (a nested
 * timer was counted, or frames overlapped), `other` is 0 and the excess is returned as `overAttributedMs` so the
 * caller can say so instead of showing a stack taller than the frame.
 */
export function traceStageMs(rawStages, frameMs) {
  const stageMs = {};
  for (const stage of TRACE_STAGES) {
    if (!stage.derived) stageMs[stage.key] = sumMeasuredValues(rawStages, stage.aliases);
  }
  const detectionMs = stageMs.detection;
  const embeddingMs = stageMs.embedding;
  if (detectionMs != null && embeddingMs != null) {
    // The detector wrapper measures the full FaceAnalysis call. The recognizer timer is nested within it, so
    // subtract that exact call time to keep the stacked detection/embedding bars from counting ArcFace twice.
    stageMs.detection = Math.max(0, Number((detectionMs - embeddingMs).toFixed(3)));
  }
  const measured = Object.values(stageMs).filter(Number.isFinite).reduce((sum, v) => sum + v, 0);
  const end = frameMs != null && Number.isFinite(Number(frameMs)) ? Number(frameMs) : null;
  let overAttributedMs = 0;
  let other = null;
  if (end !== null) {
    const residual = end - measured;
    other = residual > 0 ? Number(residual.toFixed(3)) : 0;
    overAttributedMs = residual < 0 ? Number((-residual).toFixed(3)) : 0;
  }
  stageMs.other = other;
  return { stageMs, measuredMs: Number(measured.toFixed(3)), overAttributedMs };
}

export const firstMeasuredValue = (record, aliases) => {
  if (!record || typeof record !== 'object') return null;
  for (const key of aliases) {
    const value = Number(record[key]);
    if (Number.isFinite(value) && value >= 0) return value;
  }
  return null;
};

export const sumMeasuredValues = (record, aliases) => {
  if (!record || typeof record !== 'object') return null;
  let total = 0;
  let found = false;
  for (const key of aliases) {
    const value = Number(record[key]);
    if (Number.isFinite(value) && value >= 0) {
      total += value;
      found = true;
    }
  }
  return found ? total : null;
};

/** Canonical per-stage model timings (Tech Requirement 3d / Stage 27) */
export const MODEL_PASS_STAGES = [
  { key: 'detect_scrfd', modelKey: 'SCRFD-gnkps', label: 'SCRFD-gnkps', stageKey: 'detect', category: 'Detect / 5-Point Align', color: '#a855f7', bgClass: 'bg-purple-500', textClass: 'text-purple-400', residency: 'GPU (TensorRT FP16)', defaultMs: 7.8 },
  { key: 'decode_stabilizer', modelKey: 'Stabilizer', label: 'Stabilizer', stageKey: 'decode', category: 'Temporal Flow / Work-Steal', color: '#06b6d4', bgClass: 'bg-cyan-400', textClass: 'text-cyan-400', residency: 'CPU / NVDEC Work-Steal', defaultMs: 2.4 },
  { key: 'mask_bisenet_xseg', modelKey: 'BiSeNet+XSeg', label: 'BiSeNet+XSeg', stageKey: 'mask', category: 'Dual Occlusion Masking', color: '#3b82f6', bgClass: 'bg-blue-500', textClass: 'text-blue-400', residency: 'GPU (TensorRT FP16)', defaultMs: 4.2 },
  { key: 'swap_hyperswap', modelKey: 'HyperSwap', label: 'HyperSwap', stageKey: 'swap', category: 'High-Fidelity Swapper Base', color: '#10b981', bgClass: 'bg-emerald-500', textClass: 'text-emerald-400', residency: 'GPU (TensorRT FP16)', defaultMs: 13.5 },
  { key: 'swap_liveportrait', modelKey: 'LivePortrait', label: 'LivePortrait', stageKey: 'swap', category: 'Expression & Dynamics Warp', color: '#14b8a6', bgClass: 'bg-teal-500', textClass: 'text-teal-400', residency: 'GPU (TensorRT FP16)', defaultMs: 8.6 },
  { key: 'restore_ultramax', modelKey: 'UltraMax', label: 'UltraMax', stageKey: 'restore', category: 'GPEN-512 + CodeFormer Texture', color: '#f59e0b', bgClass: 'bg-amber-500', textClass: 'text-amber-400', residency: 'GPU (TensorRT Mixed)', defaultMs: 18.2 },
  { key: 'blend_laplacian', modelKey: 'Laplacian Blend', label: 'Laplacian Blend', stageKey: 'blend', category: 'Pyramid Multi-Band Blend', color: '#ec4899', bgClass: 'bg-pink-500', textClass: 'text-pink-400', residency: 'GPU (CUDA Kernel)', defaultMs: 3.1 },
];

export const STAGE_MODEL_MAPPING = Object.freeze({
  detect: 'SCRFD-gnkps',
  decode: 'Stabilizer',
  mask: 'BiSeNet+XSeg',
  swap: 'HyperSwap',
  liveportrait: 'LivePortrait',
  restore: 'UltraMax',
  blend: 'Laplacian Blend',
  encode: 'NVENC',
});

export const DEFAULT_LATENCY_MS = {
  decode: 2.4,
  detect: 7.8,
  swap: 13.5,
  mask: 4.2,
  restore: 18.2,
  blend: 3.1,
  encode: 1.6,
};

export const DEFAULT_MODEL_LATENCY_MS = {
  'SCRFD-gnkps': 7.8,
  'Stabilizer': 2.4,
  'BiSeNet+XSeg': 4.2,
  'HyperSwap': 13.5,
  'LivePortrait': 8.6,
  'UltraMax': 18.2,
  'Laplacian Blend': 3.1,
};

export const calcPercentile = (sortedArr, p) => {
  if (!sortedArr.length) return 0;
  const idx = Math.floor(p * (sortedArr.length - 1));
  return sortedArr[idx];
};

export const STAGE_COLORS = Object.fromEntries(
  FLAMEGRAPH_STAGES.map((s) => [s.key, { bar: s.bgClass, text: s.textClass, hex: s.color }])
);

export const STAGE_RESIDENCY = Object.fromEntries(
  FLAMEGRAPH_STAGES.map((s) => [s.key, { residency: s.residency.startsWith('GPU') ? 'GPU' : 'CPU (Worker / NVDEC)', full: s.residency }])
);

export const computeStageMetrics = (stageKey, historySamples = []) => {
  const vals = (historySamples || []).map((h) => (typeof h === 'number' ? h : h?.[stageKey])).filter((v) => typeof v === 'number' && Number.isFinite(v));
  if (!vals.length) {
    const d = DEFAULT_LATENCY_MS[stageKey] || 0;
    return { min: d, avg: d, p95: d, max: d };
  }
  const sorted = [...vals].sort((a, b) => a - b);
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const sum = sorted.reduce((a, b) => a + b, 0);
  const avg = Number((sum / sorted.length).toFixed(2));
  const p95 = calcPercentile(sorted, 0.95);
  return { min, avg, p95, max };
};

export const computeModelMetrics = (modelKey, historySamples = []) => {
  const vals = (historySamples || []).map((h) => (typeof h === 'number' ? h : h?.[modelKey])).filter((v) => typeof v === 'number' && Number.isFinite(v));
  if (!vals.length) {
    const d = DEFAULT_MODEL_LATENCY_MS[modelKey] || 0;
    return { min: d, avg: d, p95: d, max: d };
  }
  const sorted = [...vals].sort((a, b) => a - b);
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const sum = sorted.reduce((a, b) => a + b, 0);
  const avg = Number((sum / sorted.length).toFixed(2));
  const p95 = calcPercentile(sorted, 0.95);
  return { min, avg, p95, max };
};

export const computeFlamePercentages = (stageTimes = {}) => {
  const keys = FLAMEGRAPH_STAGES.map((s) => s.key);
  const total = keys.reduce((acc, k) => acc + (stageTimes[k] || 0), 0);
  if (total <= 0) {
    const uniform = Number((100 / keys.length).toFixed(1));
    return Object.fromEntries(keys.map((k) => [k, uniform]));
  }
  return Object.fromEntries(
    keys.map((k) => [k, Number((((stageTimes[k] || 0) / total) * 100).toFixed(1))])
  );
};

export const exportTelemetryReport = ({ jobId, recipe, history, summary }) => {
  return {
    app: 'roop-ultimate',
    job_id: jobId || 'unknown',
    exported_at: new Date().toISOString(),
    recipe: recipe || {},
    summary: summary || {},
    stage_metrics: Object.fromEntries(
      FLAMEGRAPH_STAGES.map((s) => [s.key, computeStageMetrics(s.key, history)])
    ),
    model_metrics: Object.fromEntries(
      MODEL_PASS_STAGES.map((m) => [m.modelKey, computeModelMetrics(m.modelKey, history)])
    ),
  };
};
