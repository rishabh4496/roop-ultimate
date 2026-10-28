// The terminal dashboard's snapshot, as the Processing screen shows it (roop/terminal_dashboard.py).
//
// The backend paints the console FROM this object and puts the same object on /ws/telemetry (`dashboard`), so nothing
// here is computed: every percentage, count, rate, ETA and millisecond is the backend's rounded value, formatted with
// the console's own formats (`01m 14s -> 03:24:10`, `[BOTTLENECK: Restore 64%]`). A figure the backend did not measure
// is an em dash, never 0. Pure, so it is checked without a render in .render-check/render-console-mirror.mjs.

const DASH = '—';
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const pad2 = (n) => String(n).padStart(2, '0');

export const STAGE_IDS = ['S1', 'S2', 'S3', 'S4'];

/** Same text as the console: `01m 14s`, `1h 02m 05s`. */
export function fmtDur(seconds) {
  if (!isNum(seconds) || seconds < 0) return DASH;
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${pad2(m)}m ${pad2(s % 60)}s` : `${pad2(m)}m ${pad2(s % 60)}s`;
}

/** Local `HH:MM:SS` of an epoch (seconds), `(+Nd)` when it is not on `refS`'s calendar day. */
export function fmtClock(epochS, refS) {
  if (!isNum(epochS)) return DASH;
  const d = new Date(epochS * 1000);
  const text = `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
  if (!isNum(refS)) return text;
  const day = (t) => { const x = new Date(t * 1000); return new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime(); };
  const days = Math.round((day(epochS) - day(refS)) / 86400000);
  return days >= 1 ? `${text} (+${days}d)` : text;
}

/** `01m 14s -> 03:24:10`, or an em dash while the backend has no rate yet. */
export function fmtEta(etaS, etaAt, updatedAt) {
  if (!isNum(etaS)) return DASH;
  return `${fmtDur(etaS)} -> ${fmtClock(etaAt, updatedAt)}`;
}

const num = (v, digits = 1) => (isNum(v) ? v.toFixed(digits) : DASH);
const count = (v) => (isNum(v) ? Math.round(v).toLocaleString('en-US') : DASH);

/** Tailwind classes per stage, matching the console's colours: cyan, magenta, emerald, blue. */
export const STAGE_TONE = {
  S1: { bar: 'bg-cyan-400', text: 'text-cyan-300' },
  S2: { bar: 'bg-fuchsia-400', text: 'text-fuchsia-300' },
  S3: { bar: 'bg-emerald-400', text: 'text-emerald-300' },
  S4: { bar: 'bg-blue-400', text: 'text-blue-300' },
};

const stageNote = (s) => {
  const d = s.detail || {};
  const bits = [];
  if (s.id === 'S2') {
    if (d.phase && s.state === 'RUNNING') bits.push(d.phase);
    if (isNum(d.prescan_fps)) bits.push(`pre-scan ${num(d.prescan_fps)} smp/s`);
    if (d.scene_cuts != null) bits.push(`cuts ${d.scene_cuts}`);
    if (isNum(d.face_free_bypass_pct)) bits.push(`face-free ${num(d.face_free_bypass_pct)}%`);
  } else if (s.id === 'S3') {
    if (isNum(s.rate)) bits.push(`${num(s.rate)} fps`);
    if (isNum(d.fps_avg) && s.state !== 'RUNNING') bits.push(`avg ${num(d.fps_avg)} fps`);
    if (d.short_by) bits.push(`SHORT by ${count(d.short_by)} frames`);
    if (d.fps_basis) bits.push(`basis ${d.fps_basis}`);
  } else if (Array.isArray(d.steps) && d.steps.length) {
    bits.push(d.steps.join(', '));
  }
  return bits.join(' · ');
};

/** One row per stage: everything the console prints on that stage's line. */
export function stageRows(dash) {
  if (!dash || !dash.stages) return [];
  return STAGE_IDS.filter((id) => dash.stages[id]).map((id) => {
    const s = dash.stages[id];
    const live = s.state === 'RUNNING';
    const counts = isNum(s.total) ? `${count(s.done)}/${count(s.total)} ${s.unit}` : `${count(s.done)} ${s.unit}`;
    let status;
    if (s.state === 'PENDING') status = 'pending';
    else if (s.state === 'SKIPPED') status = 'skipped (not needed for this job)';
    else if (live) status = s.paused ? 'PAUSED' : `ETA: ${fmtEta(s.eta_s, s.eta_at, dash.updated_at)}`;
    else status = s.state.toLowerCase();
    return {
      id, name: s.name, title: s.title, state: s.state,
      pct: isNum(s.pct) ? s.pct : null, pctText: isNum(s.pct) ? `${s.pct.toFixed(1)}%` : DASH,
      counts: s.state === 'PENDING' || s.state === 'SKIPPED' ? '' : counts,
      elapsed: isNum(s.elapsed_s) && s.state !== 'PENDING' && s.state !== 'SKIPPED' ? `elapsed ${fmtDur(s.elapsed_s)}` : '',
      status, note: s.state === 'PENDING' || s.state === 'SKIPPED' ? '' : stageNote(s),
    };
  });
}

export const LATENCY_COLUMNS = [
  ['decode', 'Decode'], ['detect', 'Detect'], ['swap', 'Swap'], ['mask', 'Mask/Parse'],
  ['lipsync', 'Lip-Sync'], ['restore', 'Restore'], ['blend', 'Color/Blend'], ['encode', 'Encode'],
];

/** The per-frame latency split (ms per delivered frame), the bottleneck column flagged. */
export function latencyCells(dash) {
  const t = dash && dash.telemetry;
  if (!t) return [];
  const bn = t.bottleneck;
  return LATENCY_COLUMNS.map(([key, label]) => ({
    key, label, ms: isNum(t.latency_ms?.[key]) ? t.latency_ms[key].toFixed(1) : DASH,
    bottleneck: !!bn && bn.stage === key, severity: bn && bn.stage === key ? bn.severity : null,
  }));
}

/** `[BOTTLENECK: Restore 64%]`, or `[bottleneck: measuring]` until a window of frames exists. */
export function bottleneckText(dash) {
  const bn = dash && dash.telemetry && dash.telemetry.bottleneck;
  return bn ? `[BOTTLENECK: ${bn.label} ${Math.round(bn.share_pct)}%]` : '[bottleneck: measuring]';
}

/**
 * Decode / GPU / Encode fill, as the console prints them (`Decode Queue: 8/8 | GPU Queue: 4/4 | Encode Queue: 2/8`).
 * `fill` is 0..1 for the bar (null when the backend has no capacity to divide by); the GPU cell is absent on a path
 * that cannot count its workers. A unit other than frames (stabilization chunks) is named, never silently reused.
 */
export function queueCells(dash) {
  const q = dash && dash.telemetry && dash.telemetry.queues;
  if (!q || !q.decode || !q.encode) return [];
  const fill = (n, cap) => (isNum(n) && isNum(cap) && cap > 0 ? Math.max(0, Math.min(1, n / cap)) : null);
  const cells = [{ key: 'decode', label: 'Decode Queue', value: `${count(q.decode.depth)}/${count(q.decode.cap)}`, fill: fill(q.decode.depth, q.decode.cap), unit: q.unit || '' }];
  if (q.gpu) cells.push({ key: 'gpu', label: 'GPU Queue', value: `${count(q.gpu.busy)}/${count(q.gpu.cap)}`, fill: fill(q.gpu.busy, q.gpu.cap), unit: 'workers busy' });
  cells.push({ key: 'encode', label: 'Encode Queue', value: `${count(q.encode.depth)}/${count(q.encode.cap)}`, fill: fill(q.encode.depth, q.encode.cap), unit: q.unit || '' });
  return cells;
}

/** `[GPU-BOUND]` etc. and its tone; `[bound: measuring]` until the monitor has a window of frames. */
export function queueBound(dash) {
  const q = dash && dash.telemetry && dash.telemetry.queues;
  if (!q || !q.bound) return { text: '[bound: measuring]', tone: 'muted', basis: '' };
  // GPU-bound is the shape this pipeline should have; a starved decoder or a backed-up encoder is the fault.
  return { text: `[${q.bound}]`, tone: q.bound === 'GPU-BOUND' ? 'ok' : 'warn', basis: q.bound_basis || '' };
}

/** Smart-Skip ROI reuse, Stage-2 face-free skip share, VRAM allocated / peak. */
export function hitRateCells(dash) {
  const h = dash && dash.telemetry && dash.telemetry.hit_rates;
  if (!h) return [];
  const roi = h.smart_skip_enabled === false && h.smart_skip_roi_reused_pct == null ? 'off' : `${num(h.smart_skip_roi_reused_pct)}%`;
  return [
    { key: 'roi', label: 'Smart-Skip ROI reused', value: roi },
    { key: 'facefree', label: 'Stage-2 face-free skipped', value: `${num(h.stage2_face_free_skipped_pct)}%` },
    { key: 'vram', label: 'VRAM allocated / peak', value: `${num(h.vram_allocated_gb, 2)} / ${num(h.vram_peak_gb, 2)} GB` },
  ];
}

/** True when the snapshot carries a job (running or just finished); false for the idle `{schema, active:false}`. */
export const hasJob = (dash) => !!dash && !!dash.stages && Object.keys(dash.stages).length > 0;
