/**
 * Wire types. Each mirrors a server-side model; the Python source is named on
 * every interface so a change on one side has an obvious counterpart.
 */

/** face_engine/server/processing.py: SwapperName */
export type SwapperModel =
  | "hyperswap_1a_256"
  | "hyperswap_1b_256"
  | "hyperswap_1c_256"
  | "alphaface_256"
  | "inswapper_128";

/** processing.py: PixelBoost */
export type PixelBoost = "none" | "256x256" | "512x512" | "1024x1024";

/** processing.py: MaskType */
export type MaskType = "box" | "occlusion" | "region";

/** processing.py: EnhancerName */
export type EnhancerModel = "none" | "gpen_bfr_512" | "gpen_bfr_1024" | "restoreformer_plus_plus";

/** processing.py: ProviderName */
export type ExecutionProvider = "cuda" | "tensorrt" | "cpu";

/** processing.py: RenderParams */
export interface RenderParams {
  swapper_model: SwapperModel;
  pixel_boost: PixelBoost;
  mask_types: MaskType[];
  enhancer_model: EnhancerModel;
  /** 0-100 */
  enhancer_blend: number;
  execution_provider: ExecutionProvider;
  /** Fractions of the crop side, 0-0.5 */
  mask_padding_top: number;
  mask_padding_bottom: number;
  mask_padding_left: number;
  mask_padding_right: number;
  /** Gaussian feather of the box layer, fraction 0-1 */
  mask_blur: number;
  match_threshold: number;
  /** Render only: full detection every N frames, optical flow between (1-8). */
  detection_stride: number;
  workers: number;
}

export const SWAPPER_MODELS: readonly SwapperModel[] = [
  "hyperswap_1a_256",
  "hyperswap_1b_256",
  "hyperswap_1c_256",
  "alphaface_256",
  "inswapper_128",
];
export const PIXEL_BOOSTS: readonly PixelBoost[] = ["none", "256x256", "512x512", "1024x1024"];
export const MASK_TYPES: readonly MaskType[] = ["box", "occlusion", "region"];
export const ENHANCER_MODELS: readonly EnhancerModel[] = [
  "none",
  "gpen_bfr_512",
  "gpen_bfr_1024",
  "restoreformer_plus_plus",
];
export const PROVIDERS: readonly ExecutionProvider[] = ["tensorrt", "cuda", "cpu"];

/** state.py: AppState.project_json()["sources"][i] */
export interface SourceFace {
  id: string;
  name: string;
  thumbnail_url: string;
}

/** state.py: project_json()["target"] */
export interface TargetInfo {
  kind: "image" | "video";
  width: number;
  height: number;
  frames: number;
  fps: number;
  duration: number;
  url: string;
}

/** state.py: Person, as returned by GET /api/detect/faces */
export interface DetectedFace {
  id: string;
  count: number;
  frame: number;
  bbox: [number, number, number, number];
  score: number;
  thumbnail_url: string;
}

export type JobState = "preparing" | "rendering" | "completed" | "failed" | "cancelled";

/** state.py: Job.snapshot() */
export interface JobSnapshot {
  id: string;
  state: JobState;
  message: string;
  frames_done: number;
  frames_total: number;
  elapsed_s: number;
  eta_s: number | null;
  fps: number;
  latency_ms: number;
  output: string | null;
  params: RenderParams;
}

/** state.py: project_json() */
export interface Project {
  sources: SourceFace[];
  target: TargetInfo | null;
  people: DetectedFace[];
  /** person id -> source id */
  assignments: Record<string, string>;
  job: JobSnapshot | null;
}

/** telemetry.py: GuardedSampler.sample() */
export interface GpuStats {
  name: string;
  temperature_c: number;
  utilization_pct: number;
  vram_used_mb: number;
  vram_total_mb: number;
  source: "nvml" | "nvidia-smi" | string;
  /** True when NVML stalled and this is the last good sample. */
  stale?: boolean;
  age_s?: number;
}

/** telemetry.py: TelemetryHub._render() */
export interface RenderTelemetry {
  state: JobState;
  fps: number;
  frames_done: number;
  frames_total: number;
  /** 0-1 */
  progress: number;
  elapsed_s: number;
  eta_s: number | null;
}

/** telemetry.py: TelemetryHub.snapshot() (4 Hz) */
export interface Telemetry {
  type: "telemetry";
  time: number;
  interval_s?: number;
  render?: RenderTelemetry | null;
  gpu: GpuStats | null;
  job: JobSnapshot | null;
}

/** processing.py: PRESETS */
export interface Preset {
  label: string;
  params: Partial<RenderParams>;
  /** fps the preset rendered on the reference machine; null = not measured */
  measured_fps: number | null;
}

/** api.py: GET /api/options */
export interface Options {
  defaults: RenderParams;
  /** model name -> reason it cannot be used */
  unavailable: Record<string, string>;
  providers: Record<ExecutionProvider, boolean>;
  presets?: Record<string, Preset>;
}

/** preview.py: POST /api/preview/frame response (JPEG + timing headers) */
export interface PreviewResult {
  url: string;
  frame: number;
  renderMs: number | null;
  decodeMs: number | null;
  processMs: number | null;
  encodeMs: number | null;
  cache: "hit" | "miss" | null;
  faces: number | null;
  swapped: number | null;
}

export const isActive = (job: JobSnapshot | null | undefined): boolean =>
  job != null && (job.state === "preparing" || job.state === "rendering");
