// Types and builders for the 3-stage pipeline telemetry the mock server serves.
// Types are imported canonically from react-ui/src/types/api-contract.ts to prevent schema drift.

import {
  STAGE_1,
  STAGE_2,
  STAGE_3,
  type StageName,
  type StageState,
  type FlowBackend,
  type OpName,
  type OpCell,
  type OpsReport,
  type StageTelemetry,
  type PipelineTelemetry,
  type CanaryVerdict,
  type PrecisionValues,
  type RuntimeSection,
  type SegmentStatus,
  type KeyframeOverride,
  type IdentityBlendLimits,
  type IdentityBlendDirections,
  type IdentityBlendResponse,
  type TrtCacheNamespace,
  type TrtCacheSnapshot,
  type AutotuneDefaults,
  type AutotuneSnapshot,
  type RecognitionSelection,
  type RecognitionHardwareAdvice,
  type RecognitionProviderOption,
  type RecognitionCurrent,
} from '../src/types/api-contract';

export {
  STAGE_1,
  STAGE_2,
  STAGE_3,
  type StageName,
  type StageState,
  type FlowBackend,
  type OpName,
  type OpCell,
  type OpsReport,
  type StageTelemetry,
  type PipelineTelemetry,
  type CanaryVerdict,
  type PrecisionValues,
  type RuntimeSection,
  type SegmentStatus,
  type KeyframeOverride,
  type IdentityBlendLimits,
  type IdentityBlendDirections,
  type IdentityBlendResponse,
  type TrtCacheNamespace,
  type TrtCacheSnapshot,
  type AutotuneDefaults,
  type AutotuneSnapshot,
  type RecognitionSelection,
  type RecognitionHardwareAdvice,
  type RecognitionProviderOption,
  type RecognitionCurrent,
};

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
      frames_retired: 144, effective_output_fps: fps, neural_inferences: 120, neural_inference_fps: Math.round(fps * 0.8 * 10) / 10,
      smart_skip_faces: 24, smart_skip_ratio_pct: 16.7, smart_skip_enabled: true,
      fps_basis: 'retired (frames accepted by the output writer)',
    }),
  };
  const current: StageName | null = s3 === 'RUNNING' ? STAGE_3 : s2 === 'RUNNING' ? STAGE_2 : s1 === 'RUNNING' ? STAGE_1 : null;
  return { stage: current, stages: { [STAGE_1]: stage1, [STAGE_2]: stage2, [STAGE_3]: stage3 }, summary: {} };
}

// ── Structured runtime report: the PRECISION section ───────────────────────────────────────────────

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
