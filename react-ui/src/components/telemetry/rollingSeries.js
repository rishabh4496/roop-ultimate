// A rolling 60-second window over the hardware telemetry the app already polls.
//
// Pure on purpose: no React, no store, no clock of its own. The caller passes
// `now`, so the render-check can pin the window, the gap handling and the path
// geometry without mounting anything.
//
// The honesty rule from telemetryReading.js applies here too: a metric the
// machine did not report is `null`, never 0. A null sample BREAKS the sparkline
// (a gap), it is not drawn as a flat line at zero, because "GPU at 0%" and "GPU
// nobody can measure" are different facts.

export const WINDOW_MS = 60_000;

export const METRICS = [
  { key: 'gpu', label: 'GPU utilisation', unit: '%', fixed: [0, 100], tone: '#38bdf8' },
  { key: 'vram', label: 'VRAM allocation', unit: 'GB', fixed: null, tone: '#a78bfa' },
  { key: 'cpu', label: 'CPU load', unit: '%', fixed: [0, 100], tone: '#34d399' },
  { key: 'frame', label: 'Frame render time', unit: 'ms', fixed: null, tone: '#fbbf24' },
];

const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** One sample from a /api/system/telemetry reading plus the live run's frame time. */
export function sampleOf(system, run, now) {
  const vramGb = num(system?.vram_used);
  const frame = num(run?.frame_ms);
  return {
    t: now,
    gpu: num(system?.gpu_util),
    vram: vramGb,
    cpu: num(system?.cpu_percent),
    // A frame time only means something while frames are being produced.
    frame: frame !== null && frame > 0 ? frame : null,
  };
}

/** Keep only samples inside [now - windowMs, now]. Input order is preserved. */
export function trimWindow(samples, now, windowMs = WINDOW_MS) {
  const floor = now - windowMs;
  return samples.filter((s) => s.t >= floor && s.t <= now);
}

/** Append one sample and drop everything that has left the window. */
export function pushSample(samples, sample, windowMs = WINDOW_MS) {
  return trimWindow([...samples, sample], sample.t, windowMs);
}

/** The values for one metric, in time order, nulls included (so gaps survive). */
export function valuesOf(samples, key) {
  return samples.map((s) => ({ t: s.t, v: s[key] ?? null }));
}

/** Min / max / mean / last over the window's real readings. `reported` is false when none exist. */
export function summarize(samples, key) {
  const vals = samples.map((s) => s[key]).filter((v) => typeof v === 'number');
  if (!vals.length) return { reported: false, count: 0, last: null, min: null, max: null, mean: null };
  const sum = vals.reduce((a, b) => a + b, 0);
  return {
    reported: true,
    count: vals.length,
    last: vals[vals.length - 1],
    min: Math.min(...vals),
    max: Math.max(...vals),
    mean: sum / vals.length,
  };
}

/** Y-axis bounds: the metric's fixed range when it has one, else a nice ceiling over the data. */
export function yBounds(metric, samples) {
  if (metric.fixed) return { lo: metric.fixed[0], hi: metric.fixed[1] };
  const { max } = summarize(samples, metric.key);
  if (max === null) return { lo: 0, hi: 1 };
  const hi = max <= 0 ? 1 : max * 1.15;
  return { lo: 0, hi };
}

/**
 * SVG geometry for one sparkline: a line path that lifts its pen after every
 * gap, plus the last point for a dot. Coordinates are in the viewBox space
 * `width` x `height`; x runs from the window's left edge (now - windowMs) to now.
 */
export function sparkGeometry(samples, metric, { now, windowMs = WINDOW_MS, width = 240, height = 56, pad = 3 } = {}) {
  const { lo, hi } = yBounds(metric, samples);
  const span = hi - lo || 1;
  const x = (t) => ((t - (now - windowMs)) / windowMs) * width;
  const y = (v) => height - pad - ((v - lo) / span) * (height - pad * 2);
  let d = '';
  let pen = 'up';
  let last = null;
  for (const { t, v } of valuesOf(samples, metric.key)) {
    if (v === null) { pen = 'up'; continue; }
    const px = x(t).toFixed(2);
    const py = y(Math.min(hi, Math.max(lo, v))).toFixed(2);
    d += `${pen === 'up' ? 'M' : 'L'}${px} ${py} `;
    pen = 'down';
    last = { x: Number(px), y: Number(py), v };
  }
  return { d: d.trim(), last, lo, hi, width, height };
}

/** The tile label for a metric's current reading: "62 %", "7.4 GB", "—" when unreported. */
export function readingText(metric, value) {
  if (value === null || value === undefined) return '—';
  const digits = metric.unit === 'GB' ? 2 : metric.unit === 'ms' ? 1 : 0;
  return `${Number(value).toFixed(digits)} ${metric.unit}`;
}
