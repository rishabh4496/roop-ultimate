// Types and builders for the 3-stage pipeline telemetry the mock server serves: the live `stages` block of a
// /ws/telemetry frame (app/roop/stage_telemetry.py `snapshot`), and the PRECISION section of the structured runtime
// report (app/roop/runtime_state.py), which carries the canary verdicts.
//
// The shapes mirror the backend field for field, because the React UI reads them by name: a field renamed here but not
// in the UI (or the other way round) is a typecheck error in this file instead of a blank cell in the browser. Every
// metric the backend did not measure is `null`, never 0; the mock keeps that rule so the UI's "em dash, not zero"
// handling is exercised by it too.
//
// The NUMBERS are invented (this is a mock): the e2e suite checks what the SHELL does with a payload, not what a
// GPU would report.

export const STAGE_1 = 'STAGE_1_SPARSE_PRESCAN';
export const STAGE_2 = 'STAGE_2_ACTIVE_PREPASS';
export const STAGE_3 = 'STAGE_3_MAIN_SWAP';
export type StageName = typeof STAGE_1 | typeof STAGE_2 | typeof STAGE_3;
export type StageState = 'PENDING' | 'RUNNING' | 'DONE' | 'SKIPPED' | 'RESTORED' | 'FAILED';

/** roop/flow_backend.py `name`: the engine that carries faces between detector calls in Stage 2. */
export type FlowBackend = 'nvof' | 'cv' | 'torch';

export type OpName = 'decode' | 'warp' | 'model' | 'restore' | 'blend' | 'encode';
/** One Stage 3 column: per-call milliseconds, how many calls, and the backend that really ran. */
export interface OpCell { ms: number; calls: number; backend: string }
export type OpsReport = Record<OpName, OpCell | null>;

export interface StageTelemetry {
  state: StageState;
  stage_progress_pct: number | null;
  active_ranges_count: number | null;
  verified_target_frames: number | null;
  skipped_frames_pct: number | null;
  eta_seconds: number | null;
  wall_s: number | null;
  /** Stage-local items per second: keyframe samples (Stage 1), active-range frames (Stage 2), frames (Stage 3). */
  fps: number | null;
  scene_cuts?: number | null;
  neural_detected_frames?: number | null;
  flow_tracked_frames?: number | null;
  flow_ratio_pct?: number | null;
  flow_backend?: FlowBackend | null;
  ops?: OpsReport;
}

export interface PipelineTelemetry {
  stage: StageName | null;
  stages: Partial<Record<StageName, StageTelemetry>>;
  /** The finished-run report; `{}` until the run ends (the mock never builds one). */
  summary: Record<string, never>;
}

const IDLE: PipelineTelemetry = { stage: null, stages: {}, summary: {} };

/** A running job's three stages, placed by overall progress (0..1): 0-10% Stage 1, 10-25% Stage 2, the rest Stage 3. */
export function buildStages(progress: number, processing: boolean, fps: number): PipelineTelemetry {
  if (!processing) return IDLE;
  const p = Math.min(1, Math.max(0, progress));
  const edge1 = 0.1;
  const edge2 = 0.25;
  const stateOf = (from: number, to: number): StageState => (p >= to ? 'DONE' : p > from ? 'RUNNING' : 'PENDING');
  const pctOf = (from: number, to: number): number | null => {
    const s = stateOf(from, to);
    return s === 'DONE' ? 100 : s === 'RUNNING' ? Math.round(((p - from) / (to - from)) * 1000) / 10 : 0;
  };
  const common = { active_ranges_count: 2, verified_target_frames: 180, skipped_frames_pct: 46.7 };
  const s1 = stateOf(0, edge1);
  const s2 = stateOf(edge1, edge2);
  const s3 = stateOf(edge2, 1.01);
  const stage1: StageTelemetry = {
    ...common, state: s1, stage_progress_pct: pctOf(0, edge1), eta_seconds: 0, wall_s: 6.2, fps: 10, scene_cuts: 3,
  };
  const stage2: StageTelemetry = {
    ...common, state: s2, stage_progress_pct: pctOf(edge1, edge2), eta_seconds: 0, wall_s: s2 === 'PENDING' ? 0 : 4.5, fps: 20,
    neural_detected_frames: 25, flow_tracked_frames: 75, flow_ratio_pct: 75, flow_backend: 'nvof',
  };
  const stage3: StageTelemetry = {
    ...common, state: s3, stage_progress_pct: pctOf(edge2, 1.01), eta_seconds: 7, wall_s: s3 === 'PENDING' ? 0 : 5.5, fps,
    ...(s3 === 'PENDING' ? {} : {
      ops: {
        decode: { ms: 2.1, calls: 144, backend: 'NVDEC' },
        warp: { ms: 3.4, calls: 144, backend: 'cv2 (CPU)' },
        model: { ms: 11, calls: 144, backend: 'TensorrtExecutionProvider' },
        restore: null,
        blend: { ms: 4.1, calls: 144, backend: 'cv2 (CPU)' },
        encode: { ms: 1.2, calls: 144, backend: 'hevc_nvenc' },
      },
    }),
  };
  const current: StageName | null = s3 === 'RUNNING' ? STAGE_3 : s2 === 'RUNNING' ? STAGE_2 : s1 === 'RUNNING' ? STAGE_1 : null;
  return { stage: current, stages: { [STAGE_1]: stage1, [STAGE_2]: stage2, [STAGE_3]: stage3 }, summary: {} };
}

// ── Structured runtime report: the PRECISION section ───────────────────────────────────────────────────────────────

/** roop/aux_canary.py verdict as `runtime_state._surgeon_info` publishes it. `passed: null` is a skipped canary. */
export interface CanaryVerdict { passed: boolean | null; reason: string }

export interface PrecisionValues {
  configured: string;
  effective: string;
  surgeon_version: string;
  surgical_models: string[];
  verification_gates: { psnr_min_db: number; cosine_min: number; nan_inf_free: boolean };
  canary_verdicts: Record<string, CanaryVerdict>;
}

export interface RuntimeSection<V> {
  status: 'AVAILABLE' | 'UNKNOWN' | 'NOT_APPLICABLE';
  values: V;
  source: string;
}

export function precisionSection(configured = 'mixed', effective = 'mixed'): RuntimeSection<PrecisionValues> {
  return {
    status: 'AVAILABLE',
    source: 'runtime configuration/profile',
    values: {
      configured,
      effective,
      surgeon_version: 'v2',
      surgical_models: ['xseg', '2d106det', '1k3d68', 'restoreformer_pp', 'codeformer'],
      verification_gates: { psnr_min_db: 48, cosine_min: 0.9995, nan_inf_free: true },
      canary_verdicts: {
        xseg: { passed: true, reason: '' },
        '2d106det': { passed: true, reason: '' },
        '1k3d68': { passed: true, reason: '' },
        restoreformer_pp: { passed: true, reason: '' },
      },
    },
  };
}
