import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

// ── Types for Normalized Swap Recipe Slices ──────────────────────────────────

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

// ── Default State ────────────────────────────────────────────────────────────

export function createDefaultRecipe(): SwapRecipe {
  return {
    detection: {
      detectorModel: 'retinaface_r50',
      confidence: 0.5,
      maxFaces: 1,
      order: 'left-right',
      genderFilter: 'any',
      ageFilter: 'any',
      advanced: {
        iouThreshold: 0.3,
        nmsThreshold: 0.3,
        detSize: '640',
        defaultDetSize: true,
        scalePyramid: 'auto',
        refineLandmarks: true,
      },
    },
    identity: {
      sourceFaces: [],
      targetBindings: {},
      blendWeights: {
        enabled: false,
        jawline: 0,
        min_cosine: 0.8,
        components: [],
      },
      similarityThreshold: 0.75,
      identityConfidenceThreshold: 0.0,
    },
    masking: {
      maskTypes: {
        engine1: 'RealityUX',
        engine2: 'None',
        clipText: 'cup,hands,hair,banana',
        sam2ModelSize: 'tiny',
      },
      boxPadding: {
        top: 0,
        bottom: 0,
        left: 0,
        right: 0,
        linkedSides: true,
      },
      parserRegions: {
        regions: ['skin', 'brows', 'eyes', 'nose', 'mouth'],
        glassesFrameProtect: true,
        regionGrow: {},
      },
      erosion: 0,
      feather: 12,
      faceCanvasMatte: {
        edgeMode: 'gaussian',
        flowWarp: false,
        guidedFilter: false,
        imageMask: null,
      },
    },
    restoration: {
      orderedEnhancers: [
        { name: 'UltraMax', enabled: true, order: 0 },
      ],
      selectedEnhancer: 'UltraMax',
      restorationMode: 'balanced',
      modelWeights: {
        codeformerFidelity: 0.55,
        tileDiffusionStrength: 0.65,
        tileDiffusionSteps: 2,
        restoreUltraDetailWeight: 0.75,
        ultramaxDetailWeight: 0.75,
      },
      grainCloser: {
        grainMatch: 0.45,
        clarity: 1.0,
        degrade: 0,
      },
      skinTextureBlend: {
        restoreUltraFrequencyBlend: true,
        ultramaxSwapLowband: true,
        skinTextureStrength: 1.0,
        restoreUltraInnerOnly: false,
      },
      colorTransfer: {
        mode: 'lct',
        matchAfterEnhance: true,
        skinToneWarmth: 0,
        saturationMatch: 0,
        histMatch: 0.4,
      },
    },
    temporal: {
      temporalPreset: 'smooth',
      smartSkipRoiReuse: true,
      prepassDetectInterval: 1,
      emaLandmarkSmoothing: {
        enabled: true,
        stabilizeFace: true,
        stabilizeLandmarks: true,
        stabilizeMethod: 'one_euro',
        minCutoff: 0.1,
        beta: 0.1,
      },
      sceneCutSensitivity: 0.3,
      keyframeOverrides: [],
    },
    output: {
      codec: 'hevc_nvenc',
      nvencQuality: 'p6',
      crf: 14,
      videoBitrate: 'auto',
      segmentLength: 0,
      keepTempSegments: false,
      audioRemux: true,
      audioSource: 'original',
      audioPath: null,
    },
    viewport: {
      zoom: 1.0,
      panX: 0,
      panY: 0,
      fitMode: 'contain',
      lockToFaceId: null,
      compareMode: 'off',
      wipeAngle: 0,
      wipePosition: 50,
      activeScope: 'preview',
      timelineZoom: 1.0,
      timelineScroll: 0,
    },
  };
}

// ── Store Interface ──────────────────────────────────────────────────────────

export interface RecipeStoreState extends SwapRecipe {
  // Actions
  setDetection: (patch: Partial<DetectionSlice> | ((prev: DetectionSlice) => Partial<DetectionSlice>)) => void;
  setIdentity: (patch: Partial<IdentitySlice> | ((prev: IdentitySlice) => Partial<IdentitySlice>)) => void;
  setMasking: (patch: Partial<MaskingSlice> | ((prev: MaskingSlice) => Partial<MaskingSlice>)) => void;
  setBoxPadding: (side: 'top' | 'bottom' | 'left' | 'right', value: number) => void;
  setBoxPaddingLinked: (linked: boolean) => void;
  setRestoration: (patch: Partial<RestorationSlice> | ((prev: RestorationSlice) => Partial<RestorationSlice>)) => void;
  setTemporal: (patch: Partial<TemporalSlice> | ((prev: TemporalSlice) => Partial<TemporalSlice>)) => void;
  setOutput: (patch: Partial<OutputSlice> | ((prev: OutputSlice) => Partial<OutputSlice>)) => void;
  setViewport: (patch: Partial<ViewportSlice> | ((prev: ViewportSlice) => Partial<ViewportSlice>)) => void;
  resetViewport: () => void;
  resetToDefaults: () => void;

  // Flattened adapter for backward compatibility with payload builders
  toFlatSettings: () => Record<string, any>;
  fromFlatSettings: (flat: Record<string, any>) => void;
}

// ── Migration Helpers ────────────────────────────────────────────────────────

function migrateFlatToRecipe(flat: Record<string, any>, _target?: SwapRecipe): SwapRecipe {
  const r = createDefaultRecipe();
  if (!flat || typeof flat !== 'object') return r;

  // Detection
  if (flat.detector_engine !== undefined) r.detection.detectorModel = String(flat.detector_engine);
  if (flat.face_detector_threshold !== undefined) r.detection.confidence = Number(flat.face_detector_threshold) || 0.5;
  if (flat.face_detection_mode !== undefined) {
    r.detection.maxFaces = flat.face_detection_mode === 'All faces' ? 'all' : 1;
  }
  if (flat.face_detector_nms !== undefined) r.detection.advanced.nmsThreshold = Number(flat.face_detector_nms) || 0.3;
  if (flat.face_detector_size !== undefined) r.detection.advanced.detSize = flat.face_detector_size;
  if (flat.default_det_size !== undefined) r.detection.advanced.defaultDetSize = !!flat.default_det_size;
  if (flat.refine_landmarks !== undefined) r.detection.advanced.refineLandmarks = !!flat.refine_landmarks;

  // Identity
  if (flat.max_face_distance !== undefined) r.identity.similarityThreshold = Number(flat.max_face_distance) || 0.75;
  if (flat.identity_confidence_threshold !== undefined) {
    r.identity.identityConfidenceThreshold = Number(flat.identity_confidence_threshold) || 0;
  }
  if (flat.identity_blend && typeof flat.identity_blend === 'object') {
    r.identity.blendWeights = {
      enabled: !!flat.identity_blend.enabled,
      jawline: Number(flat.identity_blend.dials?.jawline || flat.identity_blend.jawline) || 0,
      min_cosine: Number(flat.identity_blend.min_cosine) || 0.8,
      components: Array.isArray(flat.identity_blend.components) ? flat.identity_blend.components : [],
    };
  }

  // Masking
  if (flat.mask_engine !== undefined) r.masking.maskTypes.engine1 = String(flat.mask_engine);
  if (flat.mask_engine_2 !== undefined) r.masking.maskTypes.engine2 = String(flat.mask_engine_2);
  if (flat.mask_clip_text !== undefined) r.masking.maskTypes.clipText = String(flat.mask_clip_text);
  if (flat.sam2_model_size !== undefined) r.masking.maskTypes.sam2ModelSize = String(flat.sam2_model_size);
  if (flat.mask_top !== undefined) r.masking.boxPadding.top = Number(flat.mask_top) || 0;
  if (flat.mask_bottom !== undefined) r.masking.boxPadding.bottom = Number(flat.mask_bottom) || 0;
  if (flat.mask_left !== undefined) r.masking.boxPadding.left = Number(flat.mask_left) || 0;
  if (flat.mask_right !== undefined) r.masking.boxPadding.right = Number(flat.mask_right) || 0;
  if (flat.face_mask_blend !== undefined) r.masking.feather = Number(flat.face_mask_blend) || 12;
  if (flat.mask_erode_dilate_radius !== undefined) r.masking.erosion = Number(flat.mask_erode_dilate_radius) || 0;
  if (flat.mask_edge_mode !== undefined) r.masking.faceCanvasMatte.edgeMode = flat.mask_edge_mode === 'distance' ? 'distance' : 'gaussian';
  if (flat.mask_flow_warp !== undefined) r.masking.faceCanvasMatte.flowWarp = !!flat.mask_flow_warp;
  if (flat.mask_guided_filter !== undefined) r.masking.faceCanvasMatte.guidedFilter = !!flat.mask_guided_filter;
  if (flat.imagemask !== undefined) r.masking.faceCanvasMatte.imageMask = flat.imagemask;
  if (Array.isArray(flat.parser_regions)) r.masking.parserRegions.regions = flat.parser_regions;
  if (flat.glasses_frame_protect !== undefined) r.masking.parserRegions.glassesFrameProtect = !!flat.glasses_frame_protect;

  // Restoration
  if (flat.selected_enhancer !== undefined) r.restoration.selectedEnhancer = String(flat.selected_enhancer);
  if (flat.restoration_mode !== undefined) r.restoration.restorationMode = String(flat.restoration_mode);
  if (flat.codeformer_fidelity !== undefined) r.restoration.modelWeights.codeformerFidelity = Number(flat.codeformer_fidelity) || 0.55;
  if (flat.tile_diffusion_strength !== undefined) r.restoration.modelWeights.tileDiffusionStrength = Number(flat.tile_diffusion_strength) || 0.65;
  if (flat.tile_diffusion_steps !== undefined) r.restoration.modelWeights.tileDiffusionSteps = Number(flat.tile_diffusion_steps) || 2;
  if (flat.restore_ultra_detail_weight !== undefined) r.restoration.modelWeights.restoreUltraDetailWeight = Number(flat.restore_ultra_detail_weight) || 0.75;
  if (flat.ultramax_detail_weight !== undefined) r.restoration.modelWeights.ultramaxDetailWeight = Number(flat.ultramax_detail_weight) || 0.75;
  if (flat.merger_grain_match !== undefined) r.restoration.grainCloser.grainMatch = Number(flat.merger_grain_match) || 0.45;
  if (flat.merger_clarity !== undefined) r.restoration.grainCloser.clarity = Number(flat.merger_clarity) || 1.0;
  if (flat.merger_degrade !== undefined) r.restoration.grainCloser.degrade = Number(flat.merger_degrade) || 0;
  if (flat.restore_ultra_frequency_blend !== undefined) r.restoration.skinTextureBlend.restoreUltraFrequencyBlend = !!flat.restore_ultra_frequency_blend;
  if (flat.ultramax_swap_lowband !== undefined) r.restoration.skinTextureBlend.ultramaxSwapLowband = !!flat.ultramax_swap_lowband;
  if (flat.restore_ultra_inner_only !== undefined) r.restoration.skinTextureBlend.restoreUltraInnerOnly = !!flat.restore_ultra_inner_only;
  if (flat.color_transfer_mode !== undefined) r.restoration.colorTransfer.mode = String(flat.color_transfer_mode);
  if (flat.color_match_after_enhance !== undefined) r.restoration.colorTransfer.matchAfterEnhance = !!flat.color_match_after_enhance;
  if (flat.skin_tone_warmth !== undefined) r.restoration.colorTransfer.skinToneWarmth = Number(flat.skin_tone_warmth) || 0;
  if (flat.saturation_match !== undefined) r.restoration.colorTransfer.saturationMatch = Number(flat.saturation_match) || 0;
  if (flat.merger_hist_match !== undefined) r.restoration.colorTransfer.histMatch = Number(flat.merger_hist_match) || 0.4;

  // Temporal
  if (flat.temporal_smart_skip !== undefined) r.temporal.smartSkipRoiReuse = !!flat.temporal_smart_skip;
  if (flat.prepass_detect_interval !== undefined) r.temporal.prepassDetectInterval = Number(flat.prepass_detect_interval) || 1;
  if (flat.stabilize_face !== undefined) r.temporal.emaLandmarkSmoothing.stabilizeFace = !!flat.stabilize_face;
  if (flat.stabilize_landmarks !== undefined) r.temporal.emaLandmarkSmoothing.stabilizeLandmarks = !!flat.stabilize_landmarks;
  if (flat.stabilize_method !== undefined) r.temporal.emaLandmarkSmoothing.stabilizeMethod = flat.stabilize_method === 'ema' ? 'ema' : 'one_euro';
  if (flat.stabilize_min_cutoff !== undefined) r.temporal.emaLandmarkSmoothing.minCutoff = Number(flat.stabilize_min_cutoff) || 0.1;
  if (flat.stabilize_beta !== undefined) r.temporal.emaLandmarkSmoothing.beta = Number(flat.stabilize_beta) || 0.1;

  // Output
  if (flat.output_video_codec !== undefined) r.output.codec = String(flat.output_video_codec);
  if (flat.keep_temp_segments !== undefined) r.output.keepTempSegments = !!flat.keep_temp_segments;

  return r;
}

// ── Creation & Storage Setup ─────────────────────────────────────────────────

const STORAGE_KEY = 'roop_swap_recipe';
const RECIPE_STORE_VERSION = 2;

function safeLocalStorage() {
  try {
    const probe = '__roop_recipe_probe__';
    window.localStorage.setItem(probe, '1');
    window.localStorage.removeItem(probe);
    return window.localStorage;
  } catch {
    const mem = new Map<string, string>();
    return {
      getItem: (k: string) => mem.get(k) ?? null,
      setItem: (k: string, v: string) => { mem.set(k, v); },
      removeItem: (k: string) => { mem.delete(k); },
    };
  }
}

export const useRecipeStore = create<RecipeStoreState>()(
  persist(
    (set, get) => ({
      ...createDefaultRecipe(),

      setDetection: (patch) => set((s) => ({
        detection: { ...s.detection, ...(typeof patch === 'function' ? patch(s.detection) : patch) },
      })),

      setIdentity: (patch) => set((s) => ({
        identity: { ...s.identity, ...(typeof patch === 'function' ? patch(s.identity) : patch) },
      })),

      setMasking: (patch) => set((s) => ({
        masking: { ...s.masking, ...(typeof patch === 'function' ? patch(s.masking) : patch) },
      })),

      setBoxPadding: (side, value) => set((s) => {
        const cur = s.masking.boxPadding;
        if (cur.linkedSides) {
          return {
            masking: {
              ...s.masking,
              boxPadding: { ...cur, top: value, bottom: value, left: value, right: value },
            },
          };
        }
        return {
          masking: {
            ...s.masking,
            boxPadding: { ...cur, [side]: value },
          },
        };
      }),

      setBoxPaddingLinked: (linkedSides) => set((s) => ({
        masking: {
          ...s.masking,
          boxPadding: { ...s.masking.boxPadding, linkedSides },
        },
      })),

      setRestoration: (patch) => set((s) => ({
        restoration: { ...s.restoration, ...(typeof patch === 'function' ? patch(s.restoration) : patch) },
      })),

      setTemporal: (patch) => set((s) => ({
        temporal: { ...s.temporal, ...(typeof patch === 'function' ? patch(s.temporal) : patch) },
      })),

      setOutput: (patch) => set((s) => ({
        output: { ...s.output, ...(typeof patch === 'function' ? patch(s.output) : patch) },
      })),

      setViewport: (patch) => set((s) => ({
        viewport: { ...s.viewport, ...(typeof patch === 'function' ? patch(s.viewport) : patch) },
      })),

      resetViewport: () => set({
        viewport: createDefaultRecipe().viewport,
      }),

      resetToDefaults: () => set(createDefaultRecipe()),

      toFlatSettings: () => {
        const s = get();
        return {
          detector_engine: s.detection.detectorModel,
          face_detector_threshold: s.detection.confidence,
          face_detection_mode: s.detection.maxFaces === 'all' ? 'All faces' : 'Selected face',
          face_detector_nms: s.detection.advanced.nmsThreshold,
          face_detector_size: s.detection.advanced.detSize,
          default_det_size: s.detection.advanced.defaultDetSize,
          refine_landmarks: s.detection.advanced.refineLandmarks,

          max_face_distance: s.identity.similarityThreshold,
          identity_confidence_threshold: s.identity.identityConfidenceThreshold,
          identity_blend: s.identity.blendWeights,

          mask_engine: s.masking.maskTypes.engine1,
          mask_engine_2: s.masking.maskTypes.engine2,
          mask_clip_text: s.masking.maskTypes.clipText,
          sam2_model_size: s.masking.maskTypes.sam2ModelSize,
          mask_top: s.masking.boxPadding.top,
          mask_bottom: s.masking.boxPadding.bottom,
          mask_left: s.masking.boxPadding.left,
          mask_right: s.masking.boxPadding.right,
          face_mask_blend: s.masking.feather,
          mask_erode_dilate_radius: s.masking.erosion,
          mask_edge_mode: s.masking.faceCanvasMatte.edgeMode,
          mask_flow_warp: s.masking.faceCanvasMatte.flowWarp,
          mask_guided_filter: s.masking.faceCanvasMatte.guidedFilter,
          imagemask: s.masking.faceCanvasMatte.imageMask,
          parser_regions: s.masking.parserRegions.regions,
          glasses_frame_protect: s.masking.parserRegions.glassesFrameProtect,
          parser_region_grow: s.masking.parserRegions.regionGrow,

          selected_enhancer: s.restoration.selectedEnhancer,
          restoration_mode: s.restoration.restorationMode,
          codeformer_fidelity: s.restoration.modelWeights.codeformerFidelity,
          tile_diffusion_strength: s.restoration.modelWeights.tileDiffusionStrength,
          tile_diffusion_steps: s.restoration.modelWeights.tileDiffusionSteps,
          restore_ultra_detail_weight: s.restoration.modelWeights.restoreUltraDetailWeight,
          ultramax_detail_weight: s.restoration.modelWeights.ultramaxDetailWeight,
          merger_grain_match: s.restoration.grainCloser.grainMatch,
          merger_clarity: s.restoration.grainCloser.clarity,
          merger_degrade: s.restoration.grainCloser.degrade,
          restore_ultra_frequency_blend: s.restoration.skinTextureBlend.restoreUltraFrequencyBlend,
          ultramax_swap_lowband: s.restoration.skinTextureBlend.ultramaxSwapLowband,
          restore_ultra_inner_only: s.restoration.skinTextureBlend.restoreUltraInnerOnly,
          color_transfer_mode: s.restoration.colorTransfer.mode,
          color_match_after_enhance: s.restoration.colorTransfer.matchAfterEnhance,
          skin_tone_warmth: s.restoration.colorTransfer.skinToneWarmth,
          saturation_match: s.restoration.colorTransfer.saturationMatch,
          merger_hist_match: s.restoration.colorTransfer.histMatch,

          temporal_smart_skip: s.temporal.smartSkipRoiReuse,
          prepass_detect_interval: s.temporal.prepassDetectInterval,
          stabilize_face: s.temporal.emaLandmarkSmoothing.stabilizeFace,
          stabilize_landmarks: s.temporal.emaLandmarkSmoothing.stabilizeLandmarks,
          stabilize_method: s.temporal.emaLandmarkSmoothing.stabilizeMethod,
          stabilize_min_cutoff: s.temporal.emaLandmarkSmoothing.minCutoff,
          stabilize_beta: s.temporal.emaLandmarkSmoothing.beta,

          output_video_codec: s.output.codec,
          keep_temp_segments: s.output.keepTempSegments,
        };
      },

      fromFlatSettings: (flat) => {
        if (!flat || typeof flat !== 'object') return;
        set((prev) => migrateFlatToRecipe(flat, prev));
      },
    }),
    {
      name: STORAGE_KEY,
      version: RECIPE_STORE_VERSION,
      storage: createJSONStorage(safeLocalStorage),
      migrate: (persistedState: any, version: number) => {
        const fallback = createDefaultRecipe();
        if (!persistedState || typeof persistedState !== 'object') {
          return fallback;
        }
        if (version === 2 && persistedState.detection && persistedState.masking && persistedState.restoration) {
          return {
            ...fallback,
            ...persistedState,
            detection: { ...fallback.detection, ...persistedState.detection },
            identity: { ...fallback.identity, ...persistedState.identity },
            masking: { ...fallback.masking, ...persistedState.masking },
            restoration: { ...fallback.restoration, ...persistedState.restoration },
            temporal: { ...fallback.temporal, ...persistedState.temporal },
            output: { ...fallback.output, ...persistedState.output },
            viewport: { ...fallback.viewport, ...persistedState.viewport },
          };
        }
        return migrateFlatToRecipe(persistedState, fallback);
      },
    }
  )
);
