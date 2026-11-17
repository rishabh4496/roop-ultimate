export const FLAMEGRAPH_STAGES = [
  { key: 'decode', label: 'Decode', model: 'Stabilizer', color: '#06b6d4', bgClass: 'bg-cyan-400', textClass: 'text-cyan-400', residency: 'CPU (Worker / NVDEC)' },
  { key: 'detect', label: 'Detect/Align', model: 'SCRFD-gnkps', color: '#a855f7', bgClass: 'bg-purple-500', textClass: 'text-purple-400', residency: 'GPU (TensorRT/CUDA)' },
  { key: 'swap', label: 'Swap', model: 'HyperSwap', color: '#10b981', bgClass: 'bg-emerald-500', textClass: 'text-emerald-400', residency: 'GPU (TensorRT/CUDA)' },
  { key: 'mask', label: 'Mask/Roto', model: 'BiSeNet+XSeg', color: '#3b82f6', bgClass: 'bg-blue-500', textClass: 'text-blue-400', residency: 'GPU (TensorRT/CUDA)' },
  { key: 'restore', label: 'Restore Ultra', model: 'UltraMax', color: '#f59e0b', bgClass: 'bg-amber-500', textClass: 'text-amber-400', residency: 'GPU (TensorRT/CUDA)' },
  { key: 'blend', label: 'Relight/LUT', model: 'Laplacian Blend', color: '#ec4899', bgClass: 'bg-pink-500', textClass: 'text-pink-400', residency: 'GPU (Color Warp)' },
  { key: 'encode', label: 'Encode', model: 'NVENC', color: '#64748b', bgClass: 'bg-slate-500', textClass: 'text-slate-400', residency: 'GPU (NVENC / Direct)' },
];

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
