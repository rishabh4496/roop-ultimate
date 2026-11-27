export const FLAMEGRAPH_STAGES = [
  { key: 'decode', label: 'Decode', color: '#06b6d4', bgClass: 'bg-cyan-400', textClass: 'text-cyan-400', residency: 'CPU (Worker / NVDEC)' },
  { key: 'detect', label: 'Detect/Align', color: '#a855f7', bgClass: 'bg-purple-500', textClass: 'text-purple-400', residency: 'GPU (TensorRT/CUDA)' },
  { key: 'swap', label: 'Swap', color: '#10b981', bgClass: 'bg-emerald-500', textClass: 'text-emerald-400', residency: 'GPU (TensorRT/CUDA)' },
  { key: 'mask', label: 'Mask/Roto', color: '#3b82f6', bgClass: 'bg-blue-500', textClass: 'text-blue-400', residency: 'GPU (TensorRT/CUDA)' },
  { key: 'restore', label: 'Restore Ultra', color: '#f59e0b', bgClass: 'bg-amber-500', textClass: 'text-amber-400', residency: 'GPU (TensorRT/CUDA)' },
  { key: 'blend', label: 'Relight/LUT', color: '#ec4899', bgClass: 'bg-pink-500', textClass: 'text-pink-400', residency: 'GPU (Color Warp)' },
  { key: 'encode', label: 'Encode', color: '#64748b', bgClass: 'bg-slate-500', textClass: 'text-slate-400', residency: 'GPU (NVENC / Direct)' },
];

export const DEFAULT_LATENCY_MS = {
  decode: 2.4,
  detect: 7.8,
  swap: 13.5,
  mask: 4.2,
  restore: 18.2,
  blend: 3.1,
  encode: 1.6,
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
  };
};
