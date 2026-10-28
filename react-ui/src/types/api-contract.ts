/**
 * Canonical API Contract Module (Stage 15).
 * 
 * Shared TypeScript request/response contracts and data schemas.
 * Imported by both the React UI (store/components) and the mock-server (server.ts / mockQueue.ts).
 * Any modification or schema drift enforces immediate compile-time errors in root typecheck.
 */

// ── 1. Normalized Swap Recipe Slices ────────────────────────────────────────

export interface DetectionSlice {
  detectorModel: string;
  confidence: number;
  maxFaces: number | 'all';
  order: 'left-right' | 'right-left' | 'top-bottom' | 'bottom-top' | 'small-large' | 'large-small' | 'best-worst';
  genderFilter: 'any' | 'male' | 'female';
  ageFilter: 'any' | 'child' | 'young' | 'adult' | 'senior';
  advanced: {
    iouThreshold: number;
    nmsThreshold: number;
    detSize: string | number;
    defaultDetSize: boolean;
    scalePyramid: string;
    refineLandmarks: boolean;
  };
}

export interface SourceFaceInfo {
  id: string;
  name?: string;
  url?: string;
  embedding?: number[];
  width?: number;
  height?: number;
  count?: number;
  poses?: string[];
  angles?: { pitch: number; yaw: number; roll: number };
  quality?: number;
  landmarks?: number;
  index?: number;
}

export interface TargetFaceBinding {
  sourceId: string;
  referenceFaceId?: string;
  similarity?: number;
}

export interface BlendComponent {
  source_id: string;
  weight: number;
}

export interface BlendWeights {
  enabled: boolean;
  jawline: number;
  min_cosine: number;
  components: BlendComponent[];
}

export interface IdentitySlice {
  sourceFaces: SourceFaceInfo[];
  targetBindings: Record<string, TargetFaceBinding>;
  blendWeights: BlendWeights;
  similarityThreshold: number;
  identityConfidenceThreshold: number;
}

export interface MaskTypes {
  engine1: string;
  engine2: string;
  clipText: string;
  sam2ModelSize: string;
}

export interface BoxPadding {
  top: number;
  bottom: number;
  left: number;
  right: number;
  linkedSides: boolean;
}

export interface ParserRegions {
  regions: string[];
  glassesFrameProtect: boolean;
  regionGrow: Record<string, number>;
}

export interface FaceCanvasMatte {
  edgeMode: 'gaussian' | 'distance';
  flowWarp: boolean;
  guidedFilter: boolean;
  imageMask: string | null;
}

export interface MaskingSlice {
  maskTypes: MaskTypes;
  boxPadding: BoxPadding;
  parserRegions: ParserRegions;
  erosion: number;
  feather: number;
  faceCanvasMatte: FaceCanvasMatte;
}

export interface OrderedEnhancer {
  name: string;
  enabled: boolean;
  order: number;
}

export interface RestorationSlice {
  orderedEnhancers: OrderedEnhancer[];
  selectedEnhancer: string;
  restorationMode: string;
  modelWeights: {
    codeformerFidelity: number;
    tileDiffusionStrength: number;
    tileDiffusionSteps: number;
    restoreUltraDetailWeight: number;
    ultramaxDetailWeight: number;
  };
  grainCloser: {
    grainMatch: number;
    clarity: number;
    degrade: number;
  };
  skinTextureBlend: {
    restoreUltraFrequencyBlend: boolean;
    ultramaxSwapLowband: boolean;
    skinTextureStrength: number;
    restoreUltraInnerOnly: boolean;
  };
  colorTransfer: {
    mode: string;
    matchAfterEnhance: boolean;
    skinToneWarmth: number;
    saturationMatch: number;
    histMatch: number;
  };
}

export interface SegmentKeyframeOverride {
  segmentId: string;
  frameStart: number;
  frameEnd: number;
  overrides: Record<string, any>;
}

export interface TemporalSlice {
  temporalPreset: 'off' | 'smooth' | 'cinematic' | 'action';
  smartSkipRoiReuse: boolean;
  prepassDetectInterval: number;
  emaLandmarkSmoothing: {
    enabled: boolean;
    stabilizeFace: boolean;
    stabilizeLandmarks: boolean;
    stabilizeMethod: 'one_euro' | 'ema';
    minCutoff: number;
    beta: number;
  };
  sceneCutSensitivity: number;
  keyframeOverrides: SegmentKeyframeOverride[];
}

export interface OutputSlice {
  codec: string;
  nvencQuality: string;
  crf: number;
  videoBitrate: string;
  segmentLength: number;
  keepTempSegments: boolean;
  audioRemux: boolean;
  audioSource: string;
  audioPath: string | null;
}

export interface ViewportSlice {
  zoom: number;
  panX: number;
  panY: number;
  fitMode: 'contain' | 'cover' | 'actual';
  lockToFaceId: string | number | null;
  compareMode: 'side-by-side' | 'split-wipe' | 'diff' | 'off';
  wipeAngle: number;
  wipePosition: number;
  activeScope: 'preview' | 'timeline' | 'source' | 'target';
  timelineZoom: number;
  timelineScroll: number;
}

export interface SwapRecipe {
  detection: DetectionSlice;
  identity: IdentitySlice;
  masking: MaskingSlice;
  restoration: RestorationSlice;
  temporal: TemporalSlice;
  output: OutputSlice;
  viewport: ViewportSlice;
}

// ── 2. Pipeline & Telemetry Contracts ───────────────────────────────────────

export const STAGE_1 = 'STAGE_1_SPARSE_PRESCAN';
export const STAGE_2 = 'STAGE_2_ACTIVE_PREPASS';
export const STAGE_3 = 'STAGE_3_MAIN_SWAP';
export type StageName = typeof STAGE_1 | typeof STAGE_2 | typeof STAGE_3;
export type StageState = 'PENDING' | 'RUNNING' | 'DONE' | 'SKIPPED' | 'RESTORED' | 'FAILED';
export type FlowBackend = 'nvof' | 'cv' | 'torch';
export type OpName = 'decode' | 'warp' | 'model' | 'restore' | 'blend' | 'encode';

export interface OpCell {
  ms: number;
  calls: number;
  backend: string;
  timing?: string;
}

export type OpsReport = Record<OpName, OpCell | null>;

export interface StageTelemetry {
  state: StageState;
  stage_progress_pct: number | null;
  active_ranges_count: number | null;
  verified_target_frames: number | null;
  skipped_frames_pct: number | null;
  eta_seconds: number | null;
  wall_s: number | null;
  fps: number | null;
  scene_cuts?: number | null;
  neural_detected_frames?: number | null;
  flow_tracked_frames?: number | null;
  flow_ratio_pct?: number | null;
  flow_backend?: FlowBackend | null;
  ops?: OpsReport;
  frames_retired?: number | null;
  effective_output_fps?: number | null;
  neural_inferences?: number | null;
  neural_inference_fps?: number | null;
  smart_skip_faces?: number | null;
  smart_skip_ratio_pct?: number | null;
  smart_skip_enabled?: boolean | null;
  fps_basis?: string;
}

export interface PipelineTelemetry {
  stage: StageName | null;
  stages: Partial<Record<StageName, StageTelemetry>>;
  summary: Record<string, never>;
}

export interface QueueLevelMetric {
  depth?: number;
  busy?: number;
  cap: number;
}

export interface PipelineQueueLevels {
  decode: QueueLevelMetric;
  gpu: QueueLevelMetric;
  encode: QueueLevelMetric;
  unit: string;
  bound: 'DISK/DECODE-BOUND' | 'GPU-BOUND' | 'ENCODE-BOUND' | 'BALANCED' | string | null;
  bound_basis: string;
  frames?: number;
  live?: boolean;
}

export interface DashboardSnapshot {
  updated_at: number;
  profile?: Record<string, any>;
  stages: Record<string, any>;
  telemetry: {
    frames: number;
    bottleneck: Record<string, any> | null;
    latency_ms: Record<string, number>;
    hit_rates: Record<string, any>;
    queues?: PipelineQueueLevels | null;
  };
}

// ── 3. Diagnostic & Hardware Contracts ──────────────────────────────────────

export interface CanaryVerdict {
  passed: boolean | null;
  reason: string;
}

export interface PrecisionValues {
  configured: string;
  effective: string;
  surgeon_version: string;
  surgical_models: string[];
  verification_gates: {
    psnr_min_db: number;
    cosine_min: number;
    nan_inf_free: boolean | null;
  };
  canary_verdicts: Record<string, CanaryVerdict>;
}

export interface RuntimeSection<V> {
  status: 'AVAILABLE' | 'UNKNOWN' | 'NOT_APPLICABLE';
  values: V;
  source: string;
}

export interface TrtCacheNamespace {
  name: string;
  kind: 'active' | 'stale' | 'other' | 'unknown';
  size_mb: number;
  engines: number;
  modified: string | null;
}

export interface TrtCacheSnapshot {
  root: string;
  provider_active: string;
  active_namespace: string | null;
  status: 'ready' | 'not_built' | 'not_loaded' | 'not_tensorrt';
  active_engines: number;
  total_mb: number;
  stale_mb: number;
  loose_mb: number;
  loose_engines: number;
  namespaces: TrtCacheNamespace[];
}

export interface AutotuneDefaults {
  screen_frames: number;
  confirm_frames: number;
  batches: number[];
  presets: string[];
}

export interface AutotuneSnapshot {
  ready: boolean;
  reason: string | null;
  target: string | null;
  available_frames: number;
  progress: Record<string, any> | null;
  result: Record<string, any> | null;
  defaults: AutotuneDefaults;
}

export interface RecognitionSelection {
  model: string;
  provider: string;
}

export interface RecognitionHardwareAdvice {
  tier: string;
  label: string;
  device_name: string;
  vram_gb: number;
  compute_capability: string;
  architecture: string;
  provider: string;
  strategy: string;
  model_hint: string | null;
  reason: string;
}

export interface RecognitionProviderOption {
  value: string;
  label: string;
  available: boolean;
}

export interface RecognitionCurrent {
  selection: RecognitionSelection;
  advice: RecognitionHardwareAdvice;
  providers: RecognitionProviderOption[];
  active: Record<string, any> | null;
  scope: string;
}

export interface IdentityBlendLimits {
  max_sources: number;
  age_years: number;
  dial_sigmas: number;
  dials: string[];
  default_min_cosine: number;
  /** Stage 20 */
  modes?: string[];
  curves?: string[];
  max_keyframes?: number;
  default_bezier?: number[];
  default_privacy_cosine?: number;
}

export interface IdentityBlendDirections {
  available: string[];
  fitted_names?: string[];
  heldout?: Record<string, any>;
  corpus?: Record<string, any>;
  fitted?: string | null;
  dials?: Record<string, any>;
  reason?: string;
}

export interface IdentityBlendResponse {
  recipe: Record<string, any>;
  active: boolean;
  limits: IdentityBlendLimits;
  directions: IdentityBlendDirections;
  diagnostics: Record<string, any>;
}

// ── 4. Target Media, Workstation & Batch Queue Contracts ────────────────────

export interface TargetMediaItem {
  id: number | string;
  media_id?: string;
  name: string;
  filename: string;
  preview_available: boolean;
  startframe: number;
  endframe: number;
  start_frame: number;
  end_frame: number;
  frames: number;
  fps: number;
  is_video: boolean;
}

export interface TargetFaceInfo {
  index: number;
  person: string;
  landmarks: number;
  gender: string;
  age: number;
  bbox: [number, number, number, number];
  pose: string;
}

export interface SegmentStatus {
  id: string;
  start: number;
  end: number;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress?: number;
  output_path?: string;
  job_id?: string;
}

export interface KeyframeOverride {
  id?: string;
  frame: number;
  segmentId?: string;
  settings?: Record<string, any>;
  prompt?: string;
  blend_ratio?: number;
}

export interface BatchQueueJob {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
  name?: string;
  target?: string;
  source?: string;
  progress: number;
  fps?: number;
  eta_seconds?: number;
  error?: string;
  segments?: SegmentStatus[];
}

export interface BatchQueueResponse {
  jobs: BatchQueueJob[];
  active_job_id: string | null;
  paused: boolean;
}

// ── 5. Full Server AppState & Response Shapes ───────────────────────────────

export interface AppStateResponse {
  source_faces: string[];
  source_faces_info: SourceFaceInfo[];
  target_faces: string[];
  target_groups: number[];
  target_faces_info: TargetFaceInfo[];
  target_names: string[];
  targets: TargetMediaItem[];
  selected_target_index: number;
  selected_source_index: number;
  selected_source_id?: string | null;
  faceset_count: number;
  desync?: Record<string, any>;
}

export interface ProgressResponse {
  processing: boolean;
  progress: number;
  fps: number;
  current_frame: number;
  total_frames: number;
  preview_image?: string | null;
  eta_seconds?: number | null;
  stage?: string | null;
}

export interface SwapRequestPayload {
  swap_model?: string;
  face_enhancer_name?: string;
  source_index?: number;
  target_index?: number;
  recipe?: Partial<SwapRecipe>;
  parameters?: Record<string, any>;
}

export interface SwapResponsePayload {
  ok: boolean;
  job_id?: string;
  status?: string;
  message?: string;
}

export interface PreviewRequestPayload {
  frame?: number;
  source_index?: number;
  target_face_index?: number;
  recipe?: Partial<SwapRecipe>;
}

export interface CrashAutopsyReport {
  job_id: string;
  timestamp: string;
  error_type: string;
  error_message: string;
  root_cause: string;
  fix_hint: string;
  failure_location: {
    frame_number: number;
    segment_index: number;
    file_line: string;
  };
  active_state: Record<string, any>;
  last_50_logs: string[];
}

export interface SegmentCustodyReceipt {
  ok: boolean;
  filename: string;
  expected_frames: number;
  verified_frames?: number;
  sha256?: string;
  bytes?: number;
  lossless?: boolean;
}

// ── Viseme Mouth Custody (Stage 17) ─────────────────────────────────────────────
// app/roop/mouth_custody.py is the rule set; react-ui/src/components/faceswap/mouthCustody.js the UI's model.

export type MouthCustodyMode = 'off' | 'source_mouth_restore' | 'audio_driven_sync';

/** The three request fields POST /api/swap, /api/preview and a project's settings accept (422 on a bad one). */
export interface MouthCustodyRequest {
  mouth_custody_mode?: MouthCustodyMode;
  /** The one place the remux reads its audio from; an existing audio file (or a video with audio). */
  override_audio_path?: string | null;
  /** -500..+500 ms; positive = the audio is heard later than the video. */
  audio_offset_ms?: number;
}

/** The flat fields of every /ws/telemetry frame, constant while idle. `lipsync_ms` is the dashboard's Lip-Sync latency column. */
export interface MouthCustodyTelemetry {
  lipsync_ms: number | null;
  mouth_mode: MouthCustodyMode | '';
  /** Why AUDIO_DRIVEN_SYNC ran as SOURCE_MOUTH_RESTORE for this clip (silent video, extraction failed), else ''. */
  mouth_fallback: string;
  audio_offset_ms: number;
}

/** GET /api/target/waveform and GET /api/audio/waveform. */
export interface AudioWaveformResponse {
  ready: boolean;
  state: 'ready' | 'building' | 'none' | 'error';
  duration: number;
  t0?: number;
  t1?: number;
  bins?: number;
  /** base64 of one uint8 peak (0-255, square-root scale) per bin. */
  peaks: string;
}

// ── WebGL2 3D .cube LUT Color Grading & Broadcast Scopes (Stage 18) ────────────

export interface LutGradingParams {
  lut_path?: string;
  lut_bake?: boolean;
  lut_opacity?: number;
  color_temperature?: number;
  color_tint?: number;
  color_vibrance?: number;
  color_skin_protect?: number;
  color_lift?: number;
  color_gamma?: number;
  color_gain?: number;
}


// ── Zero-Copy Desktop Window Capture & BroadcastChannel Monitor (Stage 19) ────

export type LiveCaptureSourceType = 'webcam' | 'monitor' | 'window' | 'region';

export interface DesktopCropRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DesktopWindow {
  hwnd: number;
  title: string;
  width: number;
  height: number;
  left: number;
  top: number;
  right: number;
  bottom: number;
  is_minimized: boolean;
}

export interface DesktopMonitor {
  index: number;
  name: string;
  width: number;
  height: number;
  left: number;
  top: number;
  right: number;
  bottom: number;
  is_primary: boolean;
}

export interface WebcamSource {
  id: string;
  kind: 'webcam';
  index: number;
  name: string;
}

export interface LiveSourcesResponse {
  webcams: WebcamSource[];
  monitors: DesktopMonitor[];
  windows: DesktopWindow[];
}

export interface LiveWindowsResponse {
  windows: DesktopWindow[];
}

export interface LiveMonitorsResponse {
  monitors: DesktopMonitor[];
}

export interface LiveCamStartPayload {
  source_type?: LiveCaptureSourceType;
  window_id?: number;
  monitor_index?: number;
  cam_number?: number;
  crop?: DesktopCropRect | null;
  resolution?: string;
  detector_interval?: number;
}


// ── Stereoscopic 3D & WebGL2 Equirectangular VR Viewport (Stage 21) ─────────

export type StereoMode = 'MONO_2D' | 'STEREO_SBS_HALF' | 'STEREO_SBS_FULL' | 'STEREO_TB' | 'VR180_SBS';

export type StereoInspectionMode = 'off' | 'left' | 'right' | 'side_by_side' | 'anaglyph' | 'disparity_diff';

export type VrProjectionMode = 'off' | 'vr180' | 'vr360';

export interface StereoVRViewportParams {
  vrMode: VrProjectionMode;
  yaw: number;
  pitch: number;
  fov: number;
  stereoMode: StereoInspectionMode;
  stereoLayout: 'sbs' | 'tb' | 'dual';
}
