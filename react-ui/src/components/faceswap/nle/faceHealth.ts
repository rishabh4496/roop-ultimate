/**
 * faceHealth.ts: the Face Tracking & Similarity Confidence heatmap lane's model.
 *
 * Three states the editor acts on, derived from data the pre-scan really produces:
 *   face  (emerald)  some ByteTrack range covers the frame
 *   risk  (amber)    a telemetry warning of type pose / occlusion sits on the frame (yaw / pitch past
 *                    the thresholds in runBarSync.extractWarningTicks, occlusion > 25%)
 *   lost  (rose)     no track covers the frame: the detector dropped out
 * plus `unscanned` (neutral) when there is no scan at all: a clip nobody scanned must not read as
 * one long rose "dropout".
 *
 * What this is NOT: a per-frame similarity score. The pre-scan publishes track RANGES and sparse
 * telemetry warnings, not a confidence per frame, so `risk` is a sampled signal: a warning tick marks
 * the frame it was measured on (the lane widens it to a visible minimum when drawing, it does not
 * invent neighbouring measurements). The lane's header says so.
 */

export type HealthState = 'unscanned' | 'face' | 'risk' | 'lost';

export interface HealthSegment {
  /** inclusive */
  start: number;
  /** inclusive */
  end: number;
  state: HealthState;
}

export interface HealthWarning {
  frame: number;
  type?: string;
  /** frames this sample stands for (a decimated series): painted half a span either side of `frame` */
  span?: number;
}

export interface HealthInput {
  maxFrames: number;
  trackRanges: ReadonlyArray<readonly [number, number]>;
  warnings?: ReadonlyArray<HealthWarning>;
  scanned: boolean;
}

export interface HealthSummary {
  face: number;
  risk: number;
  lost: number;
  unscanned: number;
  total: number;
}

/** Lane colours: emerald = tracked, amber = extreme pose, rose = dropout, faint = never scanned. */
export const HEALTH_COLORS: Record<HealthState, string> = {
  face: '#10b981',
  risk: '#f59e0b',
  lost: '#f43f5e',
  unscanned: 'rgba(255,255,255,0.10)',
};

export const HEALTH_LABELS: Record<HealthState, string> = {
  face: 'Face tracked',
  risk: 'Extreme head pose',
  lost: 'Face dropped out',
  unscanned: 'Not scanned',
};

const FACE = 1;
const RISK = 2;
const LOST = 3;
const NAMES: HealthState[] = ['unscanned', 'face', 'risk', 'lost'];

const isRiskType = (type: string | undefined) => type === 'pose' || type === 'occlusion';

/** Per-frame state codes (0 unscanned, 1 face, 2 risk, 3 lost). */
export function healthCodes({ maxFrames, trackRanges, warnings = [], scanned }: HealthInput): Uint8Array {
  const total = Math.max(0, Math.floor(maxFrames));
  const codes = new Uint8Array(total);
  if (!scanned || total === 0) return codes;
  codes.fill(LOST);
  for (const range of trackRanges) {
    const a = Math.max(0, Math.round(Math.min(range[0], range[1])));
    const b = Math.min(total - 1, Math.round(Math.max(range[0], range[1])));
    for (let f = a; f <= b; f++) codes[f] = FACE;
  }
  for (const warning of warnings) {
    if (!isRiskType(warning.type)) continue;
    const f = Math.round(warning.frame);
    const halo = Math.max(0, Math.floor(((warning.span ?? 1) - 1) / 2));
    // A warning is only meaningful where a face was detected: it describes that face.
    for (let g = Math.max(0, f - halo); g <= Math.min(total - 1, f + halo); g++) if (codes[g] === FACE) codes[g] = RISK;
  }
  return codes;
}

/** Run-length encode per-frame codes into segments. */
export function healthSegments(input: HealthInput): HealthSegment[] {
  const codes = healthCodes(input);
  const out: HealthSegment[] = [];
  let start = 0;
  for (let f = 1; f <= codes.length; f++) {
    if (f === codes.length || codes[f] !== codes[start]) {
      if (codes.length) out.push({ start, end: f - 1, state: NAMES[codes[start]] });
      start = f;
    }
  }
  return out;
}

export function summarizeHealth(segments: ReadonlyArray<HealthSegment>): HealthSummary {
  const sum: HealthSummary = { face: 0, risk: 0, lost: 0, unscanned: 0, total: 0 };
  for (const s of segments) {
    const n = s.end - s.start + 1;
    sum[s.state] += n;
    sum.total += n;
  }
  return sum;
}

export interface WarningRun {
  /** inclusive */
  start: number;
  /** inclusive */
  end: number;
  type: string;
  /** ticks merged into the run */
  count: number;
  /** the largest `value` among them (yaw degrees for pose, percent for occlusion) */
  peak: number;
  /** the message of the first tick (what a tooltip says for a single-event run) */
  message: string;
}

/**
 * Merge a track's warning ticks into runs: consecutive ticks of one type whose neighbourhoods touch become
 * ONE run. A decimated pose series can carry thousands of ticks per track and a sustained head turn is one
 * event, not one per sample: rendering a DOM node per tick buried the lane in dots (182 on a 180-frame mock
 * clip) and would put tens of thousands of nodes on a long one.
 */
export function groupWarningRuns(ticks: ReadonlyArray<{ frame: number; type?: string; value?: number; span?: number; message?: string }>): WarningRun[] {
  const sorted = [...ticks].sort((a, b) => a.frame - b.frame);
  const runs: WarningRun[] = [];
  const open = new Map<string, WarningRun>();
  for (const t of sorted) {
    const type = t.type || 'other';
    const halo = Math.max(0, Math.floor(((t.span ?? 1) - 1) / 2));
    const start = Math.round(t.frame) - halo;
    const end = Math.round(t.frame) + halo;
    const cur = open.get(type);
    if (cur && start <= cur.end + 1) {
      cur.end = Math.max(cur.end, end);
      cur.count += 1;
      cur.peak = Math.max(cur.peak, t.value ?? 0);
    } else {
      const run: WarningRun = { start: Math.max(0, start), end, type, count: 1, peak: t.value ?? 0, message: t.message || '' };
      runs.push(run);
      open.set(type, run);
    }
  }
  return runs.sort((a, b) => a.start - b.start);
}

/** First frame of every risk / lost run, ascending: the jump targets for "next problem". */
export function problemFrames(segments: ReadonlyArray<HealthSegment>): number[] {
  return segments.filter((s) => s.state === 'risk' || s.state === 'lost').map((s) => s.start);
}

/** The segment containing `frame`, or null. Binary search: segments are sorted and gap-free. */
export function segmentAt(segments: ReadonlyArray<HealthSegment>, frame: number): HealthSegment | null {
  let lo = 0;
  let hi = segments.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const s = segments[mid];
    if (frame < s.start) hi = mid - 1;
    else if (frame > s.end) lo = mid + 1;
    else return s;
  }
  return null;
}

/**
 * Click target for a problem segment: the first frame of the run `frame` falls in (or, within
 * `tolFrames`, of the nearest risk/lost run: a one-frame warning is thinner than a pixel at full
 * zoom-out, so the hit has to be forgiving).
 */
export function problemTargetNear(
  segments: ReadonlyArray<HealthSegment>,
  frame: number,
  tolFrames = 0,
): number | null {
  const hit = segmentAt(segments, frame);
  if (hit && (hit.state === 'risk' || hit.state === 'lost')) return hit.start;
  if (tolFrames <= 0) return null;
  let best: HealthSegment | null = null;
  let bestDist = Infinity;
  for (const s of segments) {
    if (s.state !== 'risk' && s.state !== 'lost') continue;
    const dist = frame < s.start ? s.start - frame : frame > s.end ? frame - s.end : 0;
    if (dist <= tolFrames && dist < bestDist) {
      best = s;
      bestDist = dist;
    }
  }
  return best ? best.start : null;
}
