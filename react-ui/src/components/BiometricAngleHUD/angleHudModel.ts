// Pure model for the Biometric Angle HUD: the wire types of /ws/angle-scan and
// /api/angle-scan/*, the 9-bin catalog, the reducer, and the derived views
// (quality tier, coverage warnings). No React here, so the render check can
// exercise every rule without a DOM.
//
// SIGN CONVENTION. The server reports angles in the render's pose convention:
// +yaw = face turned toward the VIEWER'S right, and +pitch = head tilted DOWN.
// Every entry also carries `pitch_up` (= -pitch), and that is the only pitch
// this UI shows, so "Looking up" and a positive pitch read the same way.

export type AngleBinName =
  | 'BIN_0_FRONTAL'
  | 'BIN_1_QUARTER_LEFT'
  | 'BIN_2_QUARTER_RIGHT'
  | 'BIN_3_HALF_PROFILE_LEFT'
  | 'BIN_4_HALF_PROFILE_RIGHT'
  | 'BIN_5_PROFILE_LEFT'
  | 'BIN_6_PROFILE_RIGHT'
  | 'BIN_7_PITCH_UP'
  | 'BIN_8_PITCH_DOWN';

export type BinStatus = 'selected' | 'nearest' | 'override' | 'missing';

export interface BinEntry {
  bin: AngleBinName;
  index: number;
  label: string;
  status: BinStatus;
  candidates: number;
  distance_deg: number;
  source_bin: AngleBinName | null;
  frame_idx?: number;          // 0-based decoder index
  track_id?: number | null;
  yaw?: number | null;
  pitch?: number | null;       // server convention (+ = down); not displayed
  pitch_up?: number | null;
  roll?: number | null;
  composite_score?: number;
  sharpness?: number;
  id_similarity?: number | null;
  ear?: number | null;
  file?: string | null;
  url?: string | null;
  time_s?: number;
  export_error?: string;
}

export type PortfolioMap = Partial<Record<AngleBinName, BinEntry>>;

export interface AngleSession {
  media_path: string;
  cache_key: string;
  person_id: string;
  media_id: string | null;
  frame_total: number;
  fps: number;
  coverage: { selected: number; nearest: number; missing: number; total: number };
  fused_embedding: {
    available: boolean; dim: number | null; sources: AngleBinName[];
    embed_failures: number; error: string | null;
  };
  bins: BinEntry[];
  thresholds: { min_iod: number; blur_frac: number };
  candidates: { evaluated: number; valid: number };
  scan: {
    frames_scanned?: number; frames_decoded?: number; faces_seen?: number; step_frames?: number;
    scan_fps?: number; elapsed_s?: number; similarity_threshold?: number; failed_frame_count?: number;
  };
  tracklets: { track_id: number; score: number | null; frames: number }[];
  best_rejected_score: number | null;
  overrides: AngleBinName[];
}

export type ScanPhase = 'scan' | 'quality' | 'export';

export interface ScanProgress {
  phase: ScanPhase;
  progress: number;
  frames_scanned?: number;
  frame_idx?: number;
  frame_total?: number;
  scan_fps?: number;
  faces_seen?: number;
  tracklet_size?: number;
  candidates?: number;
  done?: number;
}

export type ServerEvent =
  | ({ event: 'progress' } & ScanProgress)
  | { event: 'result'; session: AngleSession }
  | { event: 'cancelled' }
  | { event: 'error'; code: string; message: string }
  | { event: 'pong' };

export interface HudError { code: string; message: string }

export interface HudState {
  isScanning: boolean;
  phase: ScanPhase | null;
  progress: number;                 // 0..1 across the whole pipeline
  stats: ScanProgress | null;       // the latest progress frame, as sent
  session: AngleSession | null;
  portfolio: PortfolioMap;
  selectedReferenceBin: AngleBinName;
  overrideBin: AngleBinName | null; // the bin waiting for a scrubbed frame
  thresholds: { minIod: number; blurFrac: number };
  pending: boolean;                 // an HTTP request is in flight
  error: HudError | null;
  notice: string | null;            // non-error outcome worth saying (cancelled, warnings)
}

// ── Catalog ───────────────────────────────────────────────────────────────────
export interface BinSpec {
  name: AngleBinName;
  short: string;
  label: string;
  // Centre of the bin on the radar, in degrees (yaw, pitch_up).
  radar: [number, number];
  // Cell in the 3x3 matrix: the centre column is up / frontal / down; each
  // side column runs quarter -> half -> profile away from the centre row.
  grid: [number, number];           // [row, col]
}

export const BINS: BinSpec[] = [
  { name: 'BIN_0_FRONTAL', short: 'F', label: 'Frontal', radar: [0, 0], grid: [1, 1] },
  { name: 'BIN_1_QUARTER_LEFT', short: 'QL', label: 'Quarter left', radar: [-17.5, 0], grid: [0, 0] },
  { name: 'BIN_2_QUARTER_RIGHT', short: 'QR', label: 'Quarter right', radar: [17.5, 0], grid: [0, 2] },
  { name: 'BIN_3_HALF_PROFILE_LEFT', short: 'HL', label: 'Half profile left', radar: [-35, 0], grid: [1, 0] },
  { name: 'BIN_4_HALF_PROFILE_RIGHT', short: 'HR', label: 'Half profile right', radar: [35, 0], grid: [1, 2] },
  { name: 'BIN_5_PROFILE_LEFT', short: 'PL', label: 'Profile left', radar: [-62, 0], grid: [2, 0] },
  { name: 'BIN_6_PROFILE_RIGHT', short: 'PR', label: 'Profile right', radar: [62, 0], grid: [2, 2] },
  { name: 'BIN_7_PITCH_UP', short: 'UP', label: 'Looking up', radar: [0, 28], grid: [0, 1] },
  { name: 'BIN_8_PITCH_DOWN', short: 'DN', label: 'Looking down', radar: [0, -28], grid: [2, 1] },
];

export const BIN_BY_NAME: Record<AngleBinName, BinSpec> = Object.fromEntries(
  BINS.map((b) => [b.name, b]),
) as Record<AngleBinName, BinSpec>;

export const GRID_ORDER: AngleBinName[] = [...BINS]
  .sort((a, b) => (a.grid[0] - b.grid[0]) || (a.grid[1] - b.grid[1]))
  .map((b) => b.name);

// Bins whose absence the user is warned about: without a frontal and both
// quarter views the swap has nothing close to match a turning head against.
export const CRITICAL_BINS: AngleBinName[] = ['BIN_0_FRONTAL', 'BIN_1_QUARTER_LEFT', 'BIN_2_QUARTER_RIGHT'];

// Weighting of the three pipeline phases in the single progress bar. The scan
// is the long one; quality and export are bounded by the shortlist.
const PHASE_SPAN: Record<ScanPhase, [number, number]> = {
  scan: [0, 0.8],
  quality: [0.8, 0.97],
  export: [0.97, 1],
};

export const DEFAULT_THRESHOLDS = { minIod: 45, blurFrac: 0.5 };

// ── Derived views ─────────────────────────────────────────────────────────────
export type QualityTier = 'Optimal' | 'Acceptable' | 'Poor';

// Heuristic cut points on the composite score (0..1). Not fitted to footage:
// they sort the cards for the eye, they gate nothing.
export const TIER_OPTIMAL = 0.7;
export const TIER_ACCEPTABLE = 0.45;

export function qualityTier(entry: BinEntry | undefined | null): QualityTier | null {
  if (!entry || entry.status === 'missing' || entry.composite_score == null) return null;
  const s = entry.composite_score;
  if (s >= TIER_OPTIMAL) return 'Optimal';
  if (s >= TIER_ACCEPTABLE) return 'Acceptable';
  return 'Poor';
}

export function portfolioOf(session: AngleSession | null): PortfolioMap {
  const out: PortfolioMap = {};
  for (const entry of session?.bins || []) {
    if (entry.status !== 'missing') out[entry.bin] = entry;
  }
  return out;
}

export interface CoverageWarning { bins: AngleBinName[]; message: string }

export function coverageWarnings(session: AngleSession | null): CoverageWarning[] {
  if (!session) return [];
  const pf = portfolioOf(session);
  const out: CoverageWarning[] = [];
  if (!pf.BIN_0_FRONTAL) {
    out.push({
      bins: ['BIN_0_FRONTAL'],
      message: 'No usable frontal frame. Identity matching falls back to angled views, '
        + 'so expect a weaker swap even on straight-on shots.',
    });
  }
  const quarters = (['BIN_1_QUARTER_LEFT', 'BIN_2_QUARTER_RIGHT'] as AngleBinName[]).filter((b) => !pf[b]);
  if (quarters.length) {
    const sides = quarters.map((b) => (b === 'BIN_1_QUARTER_LEFT' ? 'left' : 'right')).join(' and ');
    out.push({
      bins: quarters,
      message: `Quarter-profile ${sides} missing. The swap may degrade or drop out while the head turns `
        + `${quarters.length === 2 ? 'either way' : `to the ${sides}`}.`,
    });
  }
  return out;
}

export function defaultReferenceBin(session: AngleSession | null): AngleBinName {
  const pf = portfolioOf(session);
  if (pf.BIN_0_FRONTAL) return 'BIN_0_FRONTAL';
  let best: BinEntry | null = null;
  for (const entry of Object.values(pf)) {
    if (entry && (!best || (entry.composite_score ?? 0) > (best.composite_score ?? 0))) best = entry;
  }
  return best ? best.bin : 'BIN_0_FRONTAL';
}

export function sessionMatches(
  session: AngleSession | null, personId: string | null | undefined, mediaId: string | null | undefined,
): boolean {
  return !!session && !!personId && session.person_id === personId
    && (mediaId == null || session.media_id == null || session.media_id === mediaId);
}

// The timeline is 1-based; the scanner's frame_idx is the 0-based decoder index
// (capturer.get_video_frame(path, n) reads index n - 1).
export const toTimelineFrame = (frameIdx: number): number => frameIdx + 1;
export const toFrameIdx = (timelineFrame: number): number => Math.max(0, timelineFrame - 1);

export function fmtAngle(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—';
  const r = Math.round(v * 10) / 10;
  return `${r > 0 ? '+' : ''}${r.toFixed(1)}°`;
}

export function overallProgress(p: ScanProgress): number {
  const [lo, hi] = PHASE_SPAN[p.phase] || [0, 1];
  const within = Math.max(0, Math.min(1, Number.isFinite(p.progress) ? p.progress : 0));
  return lo + (hi - lo) * within;
}

// ── Reducer ───────────────────────────────────────────────────────────────────
export type HudAction =
  | { type: 'scan/start' }
  | { type: 'server'; event: ServerEvent }
  | { type: 'scan/lost'; message: string }
  | { type: 'session/set'; session: AngleSession | null; notice?: string | null }
  | { type: 'reference/select'; bin: AngleBinName }
  | { type: 'override/begin'; bin: AngleBinName }
  | { type: 'override/end' }
  | { type: 'thresholds/set'; minIod?: number; blurFrac?: number }
  | { type: 'request/pending'; pending: boolean }
  | { type: 'error'; error: HudError | null }
  | { type: 'notice'; notice: string | null };

export const initialHudState: HudState = {
  isScanning: false,
  phase: null,
  progress: 0,
  stats: null,
  session: null,
  portfolio: {},
  selectedReferenceBin: 'BIN_0_FRONTAL',
  overrideBin: null,
  thresholds: { ...DEFAULT_THRESHOLDS },
  pending: false,
  error: null,
  notice: null,
};

function withSession(state: HudState, session: AngleSession | null): HudState {
  const portfolio = portfolioOf(session);
  const keep = portfolio[state.selectedReferenceBin] ? state.selectedReferenceBin : defaultReferenceBin(session);
  return {
    ...state,
    session,
    portfolio,
    selectedReferenceBin: keep,
    thresholds: session
      ? { minIod: session.thresholds.min_iod, blurFrac: session.thresholds.blur_frac }
      : state.thresholds,
  };
}

export function hudReducer(state: HudState, action: HudAction): HudState {
  switch (action.type) {
    case 'scan/start':
      return { ...state, isScanning: true, phase: 'scan', progress: 0, stats: null,
        error: null, notice: null, overrideBin: null };
    case 'server': {
      const ev = action.event;
      if (ev.event === 'progress') {
        if (!state.isScanning) return state;          // a straggler after the end
        return { ...state, phase: ev.phase, stats: ev,
          progress: Math.max(state.phase === ev.phase ? state.progress : 0, overallProgress(ev)) };
      }
      if (ev.event === 'result') {
        return { ...withSession(state, ev.session), isScanning: false, phase: null, progress: 1 };
      }
      if (ev.event === 'cancelled') {
        return { ...state, isScanning: false, phase: null, notice: 'Scan cancelled.' };
      }
      if (ev.event === 'error') {
        return { ...state, isScanning: false, phase: null, error: { code: ev.code, message: ev.message } };
      }
      return state;
    }
    case 'scan/lost':
      return state.isScanning
        ? { ...state, isScanning: false, phase: null, error: { code: 'connection', message: action.message } }
        : state;
    case 'session/set':
      return { ...withSession(state, action.session), notice: action.notice ?? state.notice };
    case 'reference/select':
      return { ...state, selectedReferenceBin: action.bin };
    case 'override/begin':
      return { ...state, overrideBin: action.bin, notice: null };
    case 'override/end':
      return { ...state, overrideBin: null };
    case 'thresholds/set':
      return { ...state, thresholds: {
        minIod: action.minIod ?? state.thresholds.minIod,
        blurFrac: action.blurFrac ?? state.thresholds.blurFrac,
      } };
    case 'request/pending':
      return { ...state, pending: action.pending };
    case 'error':
      return { ...state, error: action.error };
    case 'notice':
      return { ...state, notice: action.notice };
    default:
      return state;
  }
}

export function parseServerEvent(text: string): ServerEvent | null {
  try {
    const ev = JSON.parse(text);
    return ev && typeof ev === 'object' && typeof ev.event === 'string' ? (ev as ServerEvent) : null;
  } catch {
    return null;
  }
}

// Plain-language reasons for the server's rejection codes.
export const WARNING_TEXT: Record<string, string> = {
  low_identity_similarity: 'this face does not look much like the captured person',
  eyes_closed: 'eyes closed',
  too_small: 'face too small',
  too_dark: 'too dark',
  too_bright: 'too bright',
  clipped: 'blown highlights or crushed shadows',
  blurred: 'blurred',
  pose_unsolved: 'pose could not be measured',
};
