/**
 * PreviewViewport.tsx: Studio Viewport for Roop Ultimate FaceSwap.
 *
 * Capabilities:
 *   1. Unified ViewportMatrix & Zero-Drift Preview Box:
 *      - Pure affine matrix (mediaToScreen and screenToMedia) eliminating coordinate drift
 *        between video playback, split comparison wipe, 3.5x magnifier loupe, landmark overlays,
 *        and paint/erase mask brush.
 *      - Driven by ResizeObserver and pan/zoom state.
 *   2. Synchronized Playback & In-Memory LRU Frame Cache:
 *      - requestVideoFrameCallback with requestAnimationFrame fallback.
 *      - Map<number, ImageBitmap> LRU cache with automatic .close() cleanup for flicker-free
 *        scrubbing and zero-lag A/B split-wipe comparison.
 *   3. Floating Studio Viewport HUD:
 *      - View mode switcher: 'Composite', 'Split Wipe', 'Diff Heatmap', 'BiSeNet Parsing', 'Occlusion Mask'.
 *      - 3.5x Magnifier Loupe: zero-drift sub-pixel inspection with crosshair reticle.
 *      - Interactive per-face Track Badges: click any face to view live Euler angles (Yaw/Pitch/Roll)
 *        + active pose bucket, assign/lock its .fsz faceset, or toggle swap on/off for that person.
 *      - Mask brush controls with adjustable radius, hardness/softness, invert, and full Undo/Redo stack.
 *   4. 100% Offline Compliance: zero external URLs, CDNs, or fonts.
 */

import { useHistoryStore } from '../../store/historyStore';
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  type MaskBrushStroke,
  type PtsFrameMap,
  type PtsTableEntry,
  type PtsTableRawInput,
  clampFrameIdx,
  createMaskBrushStroke,
  filterStrokesByFrame,
  formatSeconds,
  frameToTimecode,
  resolveFrameFromTime,
  resolveTimeFromFrame,
} from '../../utils/ptsFrameMap';
import {
  type PreviewExportParityState,
  bridgePreviewExportParity,
} from './runBarSync';
import {
  ViewportMatrix,
  createViewportMatrix,
  classifyPoseBucket,
  LruFrameCache,
  registerVideoFrameSync,
  type LoupeCrop,
  type PoseClassification,
} from './viewportMatrix';
// @ts-ignore
import { drawAmplifiedHeatmap } from './heatmap';

// ── Types & Interfaces ─────────────────────────────────────────────────────────

export type ViewportViewMode =
  | 'Composite'
  | 'Split Wipe'
  | 'Diff Heatmap'
  | 'BiSeNet Parsing'
  | 'Occlusion Mask';

export interface PreviewFacePose {
  yaw?: number;
  pitch?: number;
  roll?: number;
}

export interface PreviewFace {
  index?: number;
  face_index?: number;
  track_id?: number | string | null;
  trackId?: number | string | null;
  person_id?: number | string | null;
  personId?: number | string | null;
  bbox?: [number, number, number, number] | { x: number; y: number; width: number; height: number };
  kps?: Array<[number, number]>;
  landmarks?: Array<[number, number]>;
  pose?: PreviewFacePose;
  role?: 'target' | 'swapped' | 'bystander' | string;
  source_name?: string | null;
  sourceName?: string | null;
  source_index?: number | null;
  sourceIndex?: number | null;
  faceset_id?: string | null;
  facesetId?: string | null;
  faceset_path?: string | null;
  facesetPath?: string | null;
  excluded?: boolean;
  score?: number;
  occlusion_coverage_pct?: number;
  [key: string]: any;
}

export interface PreviewTelemetry {
  fps?: number;
  faces_per_sec?: number;
  render_mode?: string;
  detect_ms?: number;
  swap_ms?: number;
  enhance_ms?: number;
  total_ms?: number;
  tracks_source?: string;
  trajectory_plan_active?: boolean;
  model?: string;
  resolution?: string;
  provider?: string;
  [key: string]: any;
}

export interface PreviewViewportProps {
  // Backing Media Refs & Sources
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  canvasRef?: React.RefObject<HTMLCanvasElement | null>;
  maskCanvasRef?: React.RefObject<HTMLCanvasElement | null>;
  maskExportRef?: React.RefObject<HTMLCanvasElement | null>;
  containerRef?: React.RefObject<HTMLDivElement | null>;

  // Source URLs / Media
  videoSrc?: string;
  beforeSrc?: string;
  afterSrc?: string;
  rawUrl?: string;
  playbackSource?: 'video' | 'canvas' | 'auto';
  baseFrame?: ImageBitmap | null;
  topFrame?: ImageBitmap | null;

  // Frame Scrubber & Playback (Exact Integer frame_idx)
  frame?: number;
  setFrame?: (f: number | ((prev: number) => number)) => void;
  maxFrames?: number;
  isPlaying?: boolean;
  setIsPlaying?: (p: boolean) => void;
  fps?: number;
  ptsTable?: PtsTableRawInput | PtsFrameMap | PtsTableEntry[];

  // Faces & Track Selection
  faces?: PreviewFace[];
  kps?: any[];
  pose?: any;
  personIds?: Array<number | string | null>;
  selectedFaceIndex?: number | null;
  selectedTrackId?: number | string | null;
  onSelectFace?: (faceIndex: number, face?: PreviewFace) => void;
  onSelectTrack?: (trackId: number | string) => void;
  faceRoles?: Record<number, any> | any[];
  faceSources?: Record<number, any> | any[];
  pins?: any[];
  onPinIdentity?: (faceIndex: number) => void;
  onUnpinIdentity?: (pinId: any) => void;
  onToggleExclude?: (faceIndex: number) => void;
  onToggleSwap?: (faceIndex: number, trackId?: number | string | null) => void;
  onAssignFaceset?: (trackId: number | string, facesetPath: string) => void;

  // View Mode & Comparison
  viewMode?: ViewportViewMode | string;
  onViewModeChange?: (mode: ViewportViewMode) => void;
  showLoupe?: boolean;
  onToggleLoupe?: () => void;
  isPeekingOriginal?: boolean;
  isPeekingRaw?: boolean;

  // Mask Brush Overlay
  maskBrushActive?: boolean;
  onToggleMaskBrush?: () => void;
  brushMode?: 'paint' | 'erase';
  setBrushMode?: (mode: 'paint' | 'erase') => void;
  brushSize?: number;
  setBrushSize?: (size: number) => void;
  brushFeather?: number;
  setBrushFeather?: (feather: number) => void;
  maskApplied?: boolean;
  onApplyMask?: () => void;
  onClearMask?: () => void;
  onInvertMask?: () => void;
  maskStrokes?: MaskBrushStroke[];
  onMaskStrokesChange?: (strokes: MaskBrushStroke[]) => void;
  onMaskChange?: (maskDataUrl: string | null) => void;

  // Telemetry & Parity State
  telemetry?: PreviewTelemetry | null;
  parityState?: PreviewExportParityState | {
    tracksSource?: string;
    tracks_source?: string;
    trajectoryPlanActive?: boolean;
    trajectory_plan_active?: boolean;
    isParityVerified?: boolean;
    badgeText?: string;
    badgeTone?: 'verified' | 'unverified' | 'idle';
    reason?: string;
    [key: string]: any;
  } | null;
  tracksSource?: string;
  trajectoryPlanActive?: boolean;

  // Viewport / Zoom / Pan / Compare
  zoom?: number;
  setZoom?: (z: number | ((prev: number) => number)) => void;
  pan?: { x: number; y: number };
  setPan?: (p: { x: number; y: number } | ((prev: { x: number; y: number }) => { x: number; y: number })) => void;
  wipe?: number; // 0..100
  setWipe?: (w: number) => void;
  wipeAngle?: number;
  compare?: boolean;
  compareMode?: 'slider' | 'split' | 'diff' | 'heatmap';
  onToggleCompare?: () => void;

  // HUD & Display Flags
  showHUD?: boolean;
  showBoxes?: boolean;
  showLandmarks?: boolean;
  smallStage?: boolean;
  soloStage?: boolean;
  onSoloStage?: (solo: boolean) => void;
  className?: string;
  style?: React.CSSProperties;
}

// ── Constants & Helpers ───────────────────────────────────────────────────────

const ZOOM_MIN = 0.1;
const ZOOM_MAX = 16.0;
const LOUPE_DIAMETER = 180;
const LOUPE_MAGNIFICATION = 3.5;

function extractPoint(pt: any): { x: number; y: number } {
  if (Array.isArray(pt)) {
    return { x: Number(pt[0]), y: Number(pt[1]) };
  }
  return { x: Number(pt?.x ?? 0), y: Number(pt?.y ?? 0) };
}

function formatTrackLabel(trackId: number | string | null | undefined): string {
  if (trackId === null || trackId === undefined || trackId === '') return 'Track';
  const num = Number(trackId);
  return Number.isFinite(num) ? `Track #${num + 1}` : `Track ${trackId}`;
}

// ── PreviewViewport Component ─────────────────────────────────────────────────

export const PreviewViewport: React.FC<PreviewViewportProps> = ({
  // Refs
  videoRef: externalVideoRef,
  canvasRef: externalCanvasRef,
  maskCanvasRef: externalMaskCanvasRef,
  maskExportRef: externalMaskExportRef,
  containerRef: externalContainerRef,

  // Sources
  videoSrc,
  beforeSrc = '',
  afterSrc = '',
  rawUrl = '',
  playbackSource = 'auto',
  baseFrame = null,
  topFrame = null,

  // Scrubber & Playback
  frame: controlledFrame = 0,
  setFrame,
  maxFrames = 1000,
  isPlaying = false,
  setIsPlaying,
  fps = 30,
  ptsTable,

  // Faces
  faces = [],
  kps,
  pose,
  personIds = [],
  selectedFaceIndex = null,
  selectedTrackId = null,
  onSelectFace,
  onSelectTrack,
  faceRoles = {},
  faceSources = {},
  pins = [],
  onPinIdentity,
  onUnpinIdentity,
  onToggleExclude,
  onToggleSwap,
  onAssignFaceset,

  // View Mode & Comparison
  viewMode: controlledViewMode = 'Composite',
  onViewModeChange,
  showLoupe: controlledShowLoupe = false,
  onToggleLoupe,
  isPeekingOriginal = false,
  isPeekingRaw = false,

  // Mask Brush
  maskBrushActive = false,
  onToggleMaskBrush,
  brushMode: controlledBrushMode,
  setBrushMode: controlledSetBrushMode,
  brushSize: controlledBrushSize,
  setBrushSize: controlledSetBrushSize,
  brushFeather: controlledBrushFeather,
  setBrushFeather: controlledSetBrushFeather,
  maskApplied = false,
  onApplyMask,
  onClearMask,
  onInvertMask: controlledOnInvertMask,
  maskStrokes: controlledMaskStrokes,
  onMaskStrokesChange,
  onMaskChange,

  // Telemetry & Parity
  telemetry = null,
  parityState: controlledParityState = null,
  tracksSource: directTracksSource,
  trajectoryPlanActive: directPlanActive,

  // Zoom / Pan / Compare
  zoom: controlledZoom,
  setZoom: controlledSetZoom,
  pan: controlledPan,
  setPan: controlledSetPan,
  wipe: controlledWipe = 50,
  setWipe: controlledSetWipe,
  wipeAngle = 0,
  compare = false,
  compareMode = 'slider',
  onToggleCompare,

  // HUD
  showHUD = true,
  showBoxes = true,
  showLandmarks = true,
  smallStage = false,
  soloStage = false,
  onSoloStage,
  className = '',
  style,
}) => {
  // ── Internal State Fallbacks ────────────────────────────────────────────────
  const internalVideoRef = useRef<HTMLVideoElement | null>(null);
  const internalCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const internalMaskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const internalMaskExportRef = useRef<HTMLCanvasElement | null>(null);
  const internalContainerRef = useRef<HTMLDivElement | null>(null);
  const loupeCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const videoRef = externalVideoRef || internalVideoRef;
  const canvasRef = externalCanvasRef || internalCanvasRef;
  const maskCanvasRef = externalMaskCanvasRef || internalMaskCanvasRef;
  const maskExportRef = externalMaskExportRef || internalMaskExportRef;
  const containerRef = externalContainerRef || internalContainerRef;

  // View mode state
  const [internalViewMode, setInternalViewMode] = useState<ViewportViewMode>('Composite');
  const viewMode = (controlledViewMode as ViewportViewMode) || internalViewMode;
  const setViewMode = useCallback(
    (mode: ViewportViewMode) => {
      if (onViewModeChange) {
        onViewModeChange(mode);
      } else {
        setInternalViewMode(mode);
      }
    },
    [onViewModeChange]
  );

  // Loupe toggle state
  const [internalShowLoupe, setInternalShowLoupe] = useState<boolean>(false);
  const showLoupe = controlledShowLoupe !== undefined ? controlledShowLoupe : internalShowLoupe;
  const toggleLoupe = useCallback(() => {
    if (onToggleLoupe) {
      onToggleLoupe();
    } else {
      setInternalShowLoupe((v) => !v);
    }
  }, [onToggleLoupe]);

  // Zoom and Pan
  const [internalZoom, setInternalZoom] = useState<number>(1.0);
  const zoom = controlledZoom !== undefined ? controlledZoom : internalZoom;
  const setZoom = controlledSetZoom || setInternalZoom;

  const [internalPan, setInternalPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const pan = controlledPan !== undefined ? controlledPan : internalPan;
  const setPan = controlledSetPan || setInternalPan;

  // Split Wipe percentage (0..100)
  const [internalWipe, setInternalWipe] = useState<number>(50);
  const wipe = controlledWipe !== undefined ? controlledWipe : internalWipe;
  const setWipe = controlledSetWipe || setInternalWipe;
  const [isDraggingWipe, setIsDraggingWipe] = useState<boolean>(false);

  // Brush controls
  const [internalBrushMode, setInternalBrushMode] = useState<'paint' | 'erase'>('paint');
  const brushMode = controlledBrushMode || internalBrushMode;
  const setBrushMode = controlledSetBrushMode || setInternalBrushMode;

  const [internalBrushSize, setInternalBrushSize] = useState<number>(32);
  const brushSize = controlledBrushSize !== undefined ? controlledBrushSize : internalBrushSize;
  const setBrushSize = controlledSetBrushSize || setInternalBrushSize;

  const [internalBrushFeather, setInternalBrushFeather] = useState<number>(0.35);
  const brushFeather = controlledBrushFeather !== undefined ? controlledBrushFeather : internalBrushFeather;
  const setBrushFeather = controlledSetBrushFeather || setInternalBrushFeather;

  // Mask strokes and undo/redo history stack
  const [internalMaskStrokes, setInternalMaskStrokes] = useState<MaskBrushStroke[]>([]);
  const maskStrokes = controlledMaskStrokes || internalMaskStrokes;
  const setMaskStrokes = onMaskStrokesChange || setInternalMaskStrokes;

  const undoStackRef = useRef<MaskBrushStroke[][]>([]);
  const redoStackRef = useRef<MaskBrushStroke[][]>([]);
  const [, setHistoryTick] = useState<number>(0);

  // Faceset assignment popover state
  const [facesetAssignTrackId, setFacesetAssignTrackId] = useState<number | string | null>(null);
  const [facesetInput, setFacesetInput] = useState<string>('');

  // Media aspect & natural dimensions
  const [naturalDim, setNaturalDim] = useState<{ w: number; h: number }>({ w: 1920, h: 1080 });
  const [containerDim, setContainerDim] = useState<{ w: number; h: number }>({ w: 1280, h: 720 });
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const panStartRef = useRef<{ x: number; y: number; startPanX: number; startPanY: number }>({
    x: 0,
    y: 0,
    startPanX: 0,
    startPanY: 0,
  });

  // Cursor tracking for loupe and brush
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [cursorOnViewport, setCursorOnViewport] = useState<boolean>(false);
  const brushCursorRef = useRef<HTMLDivElement | null>(null);
  const isBrushingRef = useRef<boolean>(false);
  const currentStrokePointsRef = useRef<Array<[number, number]>>([]);

  // Telemetry drawer popover state
  const [telemetryOpen, setTelemetryOpen] = useState<boolean>(false);

  // ── In-Memory LRU Frame Cache ───────────────────────────────────────────────
  const lruCacheRef = useRef<LruFrameCache<number, ImageBitmap>>(new LruFrameCache<number, ImageBitmap>(64));

  useEffect(() => {
    const cache = lruCacheRef.current;
    return () => {
      cache.clear();
    };
  }, []);

  // ── ResizeObserver for Zero-Drift Viewport Transform ────────────────────────
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const updateDimensions = () => {
      const rect = container.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setContainerDim({ w: rect.width, h: rect.height });
      }
    };

    updateDimensions();

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const cr = entry.contentRect;
        if (cr.width > 0 && cr.height > 0) {
          setContainerDim({ w: cr.width, h: cr.height });
        }
      }
    });

    ro.observe(container);
    return () => ro.disconnect();
  }, [containerRef]);

  // ── ViewportMatrix Instance ─────────────────────────────────────────────────
  const viewportMatrix = useMemo(() => {
    return createViewportMatrix({
      containerWidth: containerDim.w,
      containerHeight: containerDim.h,
      mediaWidth: naturalDim.w,
      mediaHeight: naturalDim.h,
      zoom,
      pan,
      objectFit: 'contain',
    });
  }, [containerDim, naturalDim, zoom, pan]);

  // ── Exact Integer Frame Calculations (ptsFrameMap) ──────────────────────────
  const exactFrame = useMemo(() => {
    return clampFrameIdx(controlledFrame, 0, maxFrames);
  }, [controlledFrame, maxFrames]);

  const exactTimeSeconds = useMemo(() => {
    return resolveTimeFromFrame(exactFrame, ptsTable, fps);
  }, [exactFrame, ptsTable, fps]);

  const timecode = useMemo(() => {
    return frameToTimecode(exactFrame, fps);
  }, [exactFrame, fps]);

  // Synchronize video currentTime to exact PTS frame time when seeking / scrubbing
  useEffect(() => {
    const vid = videoRef.current;
    if (!vid || isPlaying) return;
    const targetSeconds = resolveTimeFromFrame(exactFrame, ptsTable, fps);
    if (Math.abs(vid.currentTime - targetSeconds) > 0.015) {
      vid.currentTime = targetSeconds;
    }
  }, [exactFrame, isPlaying, ptsTable, fps, videoRef]);

  // Video Frame Synchronization via requestVideoFrameCallback + rAF fallback
  useEffect(() => {
    const vid = videoRef.current;
    if (!vid) return;

    const unregister = registerVideoFrameSync(vid, (info) => {
      const resolvedFrame = resolveFrameFromTime(info.mediaTime, ptsTable, fps);
      const clamped = clampFrameIdx(resolvedFrame, 0, maxFrames);
      if (setFrame && isPlaying && clamped !== exactFrame) {
        setFrame(clamped);
      }
    });

    return () => unregister();
  }, [exactFrame, fps, isPlaying, maxFrames, ptsTable, setFrame, videoRef]);

  // Video time update handler fallback
  const handleVideoTimeUpdate = useCallback(() => {
    const vid = videoRef.current;
    if (!vid) return;
    const resolvedFrame = resolveFrameFromTime(vid.currentTime, ptsTable, fps);
    const clamped = clampFrameIdx(resolvedFrame, 0, maxFrames);
    if (setFrame && clamped !== exactFrame) {
      setFrame(clamped);
    }
  }, [exactFrame, fps, maxFrames, ptsTable, setFrame, videoRef]);

  // Store incoming topFrame or baseFrame into in-memory LRU cache
  useEffect(() => {
    if (topFrame) {
      lruCacheRef.current.set(exactFrame, topFrame);
    } else if (baseFrame) {
      lruCacheRef.current.set(exactFrame, baseFrame);
    }
  }, [topFrame, baseFrame, exactFrame]);

  // ── Preview Export Parity State ─────────────────────────────────────────────
  const parity: PreviewExportParityState = useMemo(() => {
    if (controlledParityState && 'badgeTone' in controlledParityState) {
      return controlledParityState as PreviewExportParityState;
    }
    const directSource = directTracksSource || controlledParityState?.tracks_source || controlledParityState?.tracksSource;
    const directPlan = directPlanActive ?? controlledParityState?.trajectory_plan_active ?? controlledParityState?.trajectoryPlanActive;
    const combinedTelemetry = {
      ...(telemetry || {}),
      tracks_source: directSource || telemetry?.tracks_source || 'none',
      trajectory_plan_active: directPlan ?? telemetry?.trajectory_plan_active,
    };
    return bridgePreviewExportParity(combinedTelemetry, null);
  }, [controlledParityState, directTracksSource, directPlanActive, telemetry]);

  const isParityActive = useMemo(() => {
    return Boolean(
      parity.isParityVerified ||
      parity.tracksSource === 'trajectory_plan' ||
      parity.trajectoryPlanActive ||
      (telemetry?.tracks_source === 'trajectory_plan') ||
      (telemetry?.trajectory_plan_active === true) ||
      directTracksSource === 'trajectory_plan' ||
      directPlanActive === true
    );
  }, [parity, telemetry, directTracksSource, directPlanActive]);

  // ── Canvas Frame Drawing & View Mode Rendering ──────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Check if cached frame is available for zero-flicker scrubbing
    const peeking = isPeekingOriginal || isPeekingRaw;
    const activeTop = !peeking ? (topFrame || lruCacheRef.current.get(exactFrame) || null) : null;
    const activeBase = baseFrame || null;

    if (activeTop) {
      if (canvas.width !== activeTop.width || canvas.height !== activeTop.height) {
        canvas.width = activeTop.width;
        canvas.height = activeTop.height;
        setNaturalDim({ w: activeTop.width, h: activeTop.height });
      }
    } else if (activeBase) {
      if (canvas.width !== activeBase.width || canvas.height !== activeBase.height) {
        canvas.width = activeBase.width;
        canvas.height = activeBase.height;
        setNaturalDim({ w: activeBase.width, h: activeBase.height });
      }
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (viewMode === 'Diff Heatmap') {
      // Amplified Thermal Heatmap Difference
      if (activeBase && activeTop) {
        try {
          drawAmplifiedHeatmap(ctx, activeBase, activeTop, canvas.width, canvas.height, 4.0, 'turbo');
        } catch {
          // Fallback simple difference
          ctx.drawImage(activeTop, 0, 0);
        }
      } else if (activeTop) {
        ctx.drawImage(activeTop, 0, 0);
      } else if (activeBase) {
        ctx.drawImage(activeBase, 0, 0);
      }
    } else if (viewMode === 'Split Wipe') {
      // Split Wipe between Base and Top
      const dividerX = (wipe / 100) * canvas.width;
      if (activeBase) {
        // Draw base frame on the left side
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, dividerX, canvas.height);
        ctx.clip();
        ctx.drawImage(activeBase, 0, 0);
        ctx.restore();
      }
      if (activeTop) {
        // Draw top frame on the right side
        ctx.save();
        ctx.beginPath();
        ctx.rect(dividerX, 0, canvas.width - dividerX, canvas.height);
        ctx.clip();
        ctx.drawImage(activeTop, 0, 0);
        ctx.restore();
      } else if (!activeBase && activeTop) {
        ctx.drawImage(activeTop, 0, 0);
      }
      // Divider line
      ctx.save();
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.shadowColor = 'rgba(56, 189, 248, 0.8)';
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.moveTo(dividerX, 0);
      ctx.lineTo(dividerX, canvas.height);
      ctx.stroke();
      ctx.restore();
    } else {
      // Composite or Normal render
      if (activeTop) {
        ctx.drawImage(activeTop, 0, 0);
      } else if (activeBase) {
        ctx.drawImage(activeBase, 0, 0);
      }
    }
  }, [topFrame, baseFrame, exactFrame, viewMode, wipe, isPeekingOriginal, isPeekingRaw, naturalDim, canvasRef]);

  // ── Loupe Rendering (3.5x Magnifier Lens) ───────────────────────────────────
  useEffect(() => {
    if (!showLoupe || !loupeCanvasRef.current) return;
    const loupeCanvas = loupeCanvasRef.current;
    const ctx = loupeCanvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, LOUPE_DIAMETER, LOUPE_DIAMETER);

    const crop: LoupeCrop = viewportMatrix.getLoupeCrop(
      cursorPos.x,
      cursorPos.y,
      LOUPE_DIAMETER,
      LOUPE_MAGNIFICATION
    );

    const activeTop = topFrame || lruCacheRef.current.get(exactFrame) || null;
    const activeBase = baseFrame || null;
    const sourceImage = activeTop || activeBase || canvasRef.current || videoRef.current;

    if (sourceImage) {
      ctx.save();
      // Clip to circular lens
      ctx.beginPath();
      ctx.arc(LOUPE_DIAMETER / 2, LOUPE_DIAMETER / 2, LOUPE_DIAMETER / 2 - 1, 0, Math.PI * 2);
      ctx.clip();

      // Render magnified crop with zero drift
      try {
        ctx.drawImage(
          sourceImage as any,
          crop.sourceX,
          crop.sourceY,
          crop.sourceWidth,
          crop.sourceHeight,
          0,
          0,
          LOUPE_DIAMETER,
          LOUPE_DIAMETER
        );
      } catch {
        // ignore cross-origin or video decoding frame delay
      }

      // Crosshair reticle
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.7)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(LOUPE_DIAMETER / 2 - 12, LOUPE_DIAMETER / 2);
      ctx.lineTo(LOUPE_DIAMETER / 2 + 12, LOUPE_DIAMETER / 2);
      ctx.moveTo(LOUPE_DIAMETER / 2, LOUPE_DIAMETER / 2 - 12);
      ctx.lineTo(LOUPE_DIAMETER / 2, LOUPE_DIAMETER / 2 + 12);
      ctx.stroke();

      ctx.restore();
    }
  }, [showLoupe, cursorPos, exactFrame, topFrame, baseFrame, viewportMatrix, canvasRef, videoRef]);

  // ── Mask Canvas Replay for Exact Frame ──────────────────────────────────────
  const activeFrameStrokes = useMemo(() => {
    return filterStrokesByFrame(maskStrokes, exactFrame);
  }, [maskStrokes, exactFrame]);

  const redrawMaskStrokes = useCallback(() => {
    const maskCanvas = maskCanvasRef.current;
    const exportCanvas = maskExportRef.current;
    if (!maskCanvas) return;

    const ctx = maskCanvas.getContext('2d');
    const expCtx = exportCanvas ? exportCanvas.getContext('2d') : null;
    if (!ctx) return;

    ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    if (expCtx && exportCanvas) {
      expCtx.clearRect(0, 0, exportCanvas.width, exportCanvas.height);
    }

    activeFrameStrokes.forEach((stroke) => {
      ctx.save();
      if (expCtx) expCtx.save();

      const isEraser = stroke.tool === 'eraser';
      ctx.globalCompositeOperation = isEraser ? 'destination-out' : 'source-over';
      ctx.strokeStyle = isEraser ? 'rgba(0,0,0,1)' : stroke.color || 'rgba(236,72,153,0.7)';
      ctx.lineWidth = stroke.size;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      if (expCtx) {
        expCtx.globalCompositeOperation = isEraser ? 'destination-out' : 'source-over';
        expCtx.strokeStyle = isEraser ? 'rgba(0,0,0,1)' : 'rgba(255,255,255,1)';
        expCtx.lineWidth = stroke.size;
        expCtx.lineCap = 'round';
        expCtx.lineJoin = 'round';
      }

      const pts = stroke.points;
      if (pts && pts.length > 0) {
        ctx.beginPath();
        if (expCtx) expCtx.beginPath();

        const first = extractPoint(pts[0]);
        ctx.moveTo(first.x, first.y);
        if (expCtx) expCtx.moveTo(first.x, first.y);

        for (let i = 1; i < pts.length; i++) {
          const pt = extractPoint(pts[i]);
          ctx.lineTo(pt.x, pt.y);
          if (expCtx) expCtx.lineTo(pt.x, pt.y);
        }

        ctx.stroke();
        if (expCtx) expCtx.stroke();
      }

      ctx.restore();
      if (expCtx) expCtx.restore();
    });
  }, [activeFrameStrokes, maskCanvasRef, maskExportRef]);

  useEffect(() => {
    redrawMaskStrokes();
  }, [redrawMaskStrokes]);

  // ── Undo / Redo Actions ─────────────────────────────────────────────────────
  const projectCanUndo = useHistoryStore((state) => state.canUndo);
  const projectCanRedo = useHistoryStore((state) => state.canRedo);
  const handleUndo = useCallback(() => {
    if (onMaskStrokesChange) { useHistoryStore.getState().undo(); return; }
    if (undoStackRef.current.length === 0) return;
    const previous = undoStackRef.current.pop()!;
    redoStackRef.current.push([...maskStrokes]);
    setMaskStrokes(previous);
    setHistoryTick((t) => t + 1);
  }, [maskStrokes, setMaskStrokes, onMaskStrokesChange]);

  const handleRedo = useCallback(() => {
    if (onMaskStrokesChange) { useHistoryStore.getState().redo(); return; }
    if (redoStackRef.current.length === 0) return;
    const next = redoStackRef.current.pop()!;
    undoStackRef.current.push([...maskStrokes]);
    setMaskStrokes(next);
    setHistoryTick((t) => t + 1);
  }, [maskStrokes, setMaskStrokes, onMaskStrokesChange]);

  const handleInvertMask = useCallback(() => {
    if (controlledOnInvertMask) {
      controlledOnInvertMask();
      return;
    }
    // Invert mask canvas pixels
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) return;
    const ctx = maskCanvas.getContext('2d');
    if (!ctx) return;
    const imgData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const d = imgData.data;
    for (let i = 3; i < d.length; i += 4) {
      d[i] = 255 - d[i];
    }
    ctx.putImageData(imgData, 0, 0);
  }, [controlledOnInvertMask, maskCanvasRef]);

  // Keyboard shortcut listener for Undo (Ctrl+Z) and Redo (Ctrl+Y / Ctrl+Shift+Z)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.defaultPrevented || onMaskStrokesChange) return;
      if ((e.target as HTMLElement)?.tagName === 'INPUT' || (e.target as HTMLElement)?.tagName === 'TEXTAREA') {
        return;
      }
      const isZ = e.key === 'z' || e.key === 'Z';
      const isY = e.key === 'y' || e.key === 'Y';
      const isMetaOrCtrl = e.ctrlKey || e.metaKey;

      if (isMetaOrCtrl && isZ && !e.shiftKey) {
        e.preventDefault();
        handleUndo();
      } else if (isMetaOrCtrl && (isY || (isZ && e.shiftKey))) {
        e.preventDefault();
        handleRedo();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleUndo, handleRedo, onMaskStrokesChange]);

  // ── Pointer Handlers with Zero-Drift ViewportMatrix Mapping ──────────────────
  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const container = containerRef.current;
    if (!container) return;
    const containerRect = container.getBoundingClientRect();
    const screenX = e.clientX - containerRect.left;
    const screenY = e.clientY - containerRect.top;

    // Split wipe divider drag interaction
    if (viewMode === 'Split Wipe') {
      const dividerScreenX = (wipe / 100) * containerRect.width;
      if (Math.abs(screenX - dividerScreenX) < 24) {
        setIsDraggingWipe(true);
        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        return;
      }
    }

    if (maskBrushActive) {
      // Map screen pointer directly through ViewportMatrix to media pixel coordinates
      const mediaPt = viewportMatrix.screenToMedia(screenX, screenY);
      isBrushingRef.current = true;
      currentStrokePointsRef.current = [[mediaPt.x, mediaPt.y]];

      const maskCanvas = maskCanvasRef.current;
      if (maskCanvas) {
        const ctx = maskCanvas.getContext('2d');
        if (ctx) {
          const rMedia = viewportMatrix.screenRadiusToMedia(brushSize / 2);
          ctx.save();
          ctx.globalCompositeOperation = brushMode === 'erase' ? 'destination-out' : 'source-over';
          ctx.fillStyle = brushMode === 'erase' ? 'rgba(0,0,0,1)' : 'rgba(236,72,153,0.7)';
          ctx.beginPath();
          ctx.arc(mediaPt.x, mediaPt.y, rMedia, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
        }
      }
      return;
    }

    // Panning with primary button (or middle button / space)
    if (e.button === 0 || e.button === 1) {
      setIsPanning(true);
      panStartRef.current = {
        x: e.clientX,
        y: e.clientY,
        startPanX: pan.x,
        startPanY: pan.y,
      };
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    }
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const container = containerRef.current;
    if (!container) return;
    const containerRect = container.getBoundingClientRect();
    const screenX = e.clientX - containerRect.left;
    const screenY = e.clientY - containerRect.top;

    setCursorPos({ x: screenX, y: screenY });

    // Handle Split Wipe dragging
    if (isDraggingWipe) {
      const newWipe = Math.max(0, Math.min(100, (screenX / containerRect.width) * 100));
      setWipe(newWipe);
      return;
    }

    // Update dual-concentric brush cursor
    if (maskBrushActive && brushCursorRef.current) {
      brushCursorRef.current.style.transform = `translate3d(${screenX}px, ${screenY}px, 0)`;
    }

    // Active brush stroke drawing
    if (maskBrushActive && isBrushingRef.current) {
      const mediaPt = viewportMatrix.screenToMedia(screenX, screenY);
      const pts = currentStrokePointsRef.current;
      const prev = pts[pts.length - 1];
      pts.push([mediaPt.x, mediaPt.y]);

      const maskCanvas = maskCanvasRef.current;
      if (maskCanvas && prev) {
        const ctx = maskCanvas.getContext('2d');
        if (ctx) {
          const rMedia = viewportMatrix.screenRadiusToMedia(brushSize / 2);
          ctx.save();
          ctx.globalCompositeOperation = brushMode === 'erase' ? 'destination-out' : 'source-over';
          ctx.strokeStyle = brushMode === 'erase' ? 'rgba(0,0,0,1)' : 'rgba(236,72,153,0.7)';
          ctx.lineWidth = rMedia * 2;
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';
          ctx.beginPath();
          ctx.moveTo(prev[0], prev[1]);
          ctx.lineTo(mediaPt.x, mediaPt.y);
          ctx.stroke();
          ctx.restore();
        }
      }
      return;
    }

    if (isPanning) {
      const dx = e.clientX - panStartRef.current.x;
      const dy = e.clientY - panStartRef.current.y;
      setPan({
        x: panStartRef.current.startPanX + dx,
        y: panStartRef.current.startPanY + dy,
      });
    }
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (isDraggingWipe) {
      setIsDraggingWipe(false);
      try {
        (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
      } catch {
        // ignore
      }
      return;
    }

    if (maskBrushActive && isBrushingRef.current) {
      isBrushingRef.current = false;
      const pts = currentStrokePointsRef.current;
      if (pts.length > 0) {
        const rMedia = viewportMatrix.screenRadiusToMedia(brushSize / 2);
        const newStroke = createMaskBrushStroke(
          {
            frame_idx: exactFrame,
            tool: brushMode === 'erase' ? 'eraser' : 'brush',
            size: rMedia * 2,
            points: pts,
            color: brushMode === 'erase' ? 'rgba(0,0,0,1)' : 'rgba(236,72,153,0.7)',
          },
          maxFrames
        );

        // Push to undo stack
        undoStackRef.current.push([...maskStrokes]);
        redoStackRef.current = []; // Clear redo stack on new stroke

        const updated = [...maskStrokes, newStroke];
        setMaskStrokes(updated);
        currentStrokePointsRef.current = [];

        if (onMaskChange && maskExportRef.current) {
          onMaskChange(maskExportRef.current.toDataURL('image/png'));
        }
      }
      return;
    }

    if (isPanning) {
      setIsPanning(false);
      try {
        (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
      } catch {
        // ignore
      }
    }
  };

  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    if (e.ctrlKey || e.metaKey || !maskBrushActive) {
      e.preventDefault();
      const delta = -e.deltaY;
      const factor = delta > 0 ? 1.15 : 1 / 1.15;
      setZoom((z) => Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, z * factor)));
    } else if (maskBrushActive) {
      e.preventDefault();
      const delta = e.deltaY < 0 ? 4 : -4;
      setBrushSize(Math.max(4, Math.min(200, brushSize + delta)));
    }
  };

  // Preset Zoom Controls
  const handlePresetFit = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  };
  const handlePresetFill = () => {
    setZoom(1.25);
    setPan({ x: 0, y: 0 });
  };
  const handlePreset1to1 = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  };
  const handlePreset200 = () => {
    setZoom(2.0);
  };
  const handlePreset400 = () => {
    setZoom(4.0);
  };

  // ── Face Detection Bounding Boxes & Interactive Studio Track Badges ─────────
  const faceOverlayElements = useMemo(() => {
    if (!showBoxes || !faces || faces.length === 0) return null;

    return faces.map((face, i) => {
      const faceIndex = face.index ?? face.face_index ?? i;
      const personId = face.personId ?? face.person_id ?? personIds[i] ?? null;
      const trackId = face.trackId ?? face.track_id ?? personId;

      const isSelectedByIndex = selectedFaceIndex != null && selectedFaceIndex === faceIndex;
      const isSelectedByTrack =
        selectedTrackId != null &&
        (trackId === selectedTrackId ||
          personId === selectedTrackId ||
          Number(trackId) === Number(selectedTrackId) ||
          Number(personId) === Number(selectedTrackId));
      const isSelected = isSelectedByIndex || isSelectedByTrack;

      // Extract bounding box percentages relative to naturalDim
      let leftPct = 0;
      let topPct = 0;
      let widthPct = 0;
      let heightPct = 0;

      if (Array.isArray(face.bbox) && face.bbox.length >= 4) {
        const [x0, y0, x1, y1] = face.bbox;
        if (x1 > 1 || y1 > 1) {
          leftPct = (x0 / naturalDim.w) * 100;
          topPct = (y0 / naturalDim.h) * 100;
          widthPct = ((x1 - x0) / naturalDim.w) * 100;
          heightPct = ((y1 - y0) / naturalDim.h) * 100;
        } else {
          leftPct = x0 * 100;
          topPct = y0 * 100;
          widthPct = (x1 - x0) * 100;
          heightPct = (y1 - y0) * 100;
        }
      } else if (typeof face.bbox === 'object' && face.bbox !== null) {
        const b = face.bbox as { x: number; y: number; width: number; height: number };
        if (b.width > 1 || b.height > 1) {
          leftPct = (b.x / naturalDim.w) * 100;
          topPct = (b.y / naturalDim.h) * 100;
          widthPct = (b.width / naturalDim.w) * 100;
          heightPct = (b.height / naturalDim.h) * 100;
        } else {
          leftPct = b.x * 100;
          topPct = b.y * 100;
          widthPct = b.width * 100;
          heightPct = b.height * 100;
        }
      }

      // Euler angle pose classification
      const yaw = face.pose?.yaw ?? face.yaw;
      const pitch = face.pose?.pitch ?? face.pitch;
      const roll = face.pose?.roll ?? face.roll;
      const poseClass: PoseClassification = classifyPoseBucket(yaw, pitch, roll);

      // Track metadata & roles
      const trackText = formatTrackLabel(trackId);
      const sourceName = face.sourceName || face.source_name || face.facesetPath || face.faceset_path || '';
      const role = face.role || (faceRoles && (faceRoles as any)[faceIndex]?.role);
      const isExcluded = Boolean(face.excluded || (faceRoles && (faceRoles as any)[faceIndex]?.excluded));

      // BiSeNet semantic parsing or landmark points
      const landmarks = face.landmarks || face.kps;

      return (
        <div
          key={`face_${faceIndex}_${trackId || i}`}
          data-testid="face-bounding-box"
          data-face-index={faceIndex}
          data-track-id={trackId ?? ''}
          data-selected={isSelected ? 'true' : 'false'}
          className={`absolute transition-all duration-200 pointer-events-auto cursor-pointer group ${
            isSelected
              ? 'border-2 border-cyan-400 bg-cyan-400/20 ring-4 ring-cyan-400/50 shadow-[0_0_24px_rgba(34,211,238,0.8)] z-30'
              : 'border border-[var(--accent,#38bdf8)] hover:border-white/90 hover:bg-white/10 z-20'
          }`}
          style={{
            left: `${Math.max(0, leftPct)}%`,
            top: `${Math.max(0, topPct)}%`,
            width: `${Math.min(100, Math.max(2, widthPct))}%`,
            height: `${Math.min(100, Math.max(2, heightPct))}%`,
          }}
          onClick={(e) => {
            e.stopPropagation();
            if (onSelectFace) onSelectFace(faceIndex, face);
            if (onSelectTrack && trackId != null) onSelectTrack(trackId);
          }}
          title={`Click to inspect ${trackText} (Face #${faceIndex + 1})`}
        >
          {/* ── Studio Track Badge Header ───────────────────────────────────────── */}
          <div
            data-testid={isSelected ? 'face-identity-badge-highlighted' : 'face-identity-badge'}
            className={`absolute -top-7 left-0 flex items-center gap-1.5 px-2 py-0.5 rounded shadow-lg text-micro font-bold tracking-tight whitespace-nowrap transition-all duration-200 ${
              isSelected
                ? 'bg-gradient-to-r from-cyan-500 via-sky-500 to-blue-600 text-white ring-2 ring-white shadow-[0_0_15px_rgba(6,182,212,0.9)] scale-105 z-40'
                : 'bg-black/85 backdrop-blur-md text-white/90 border border-white/20 group-hover:border-cyan-400 group-hover:text-white'
            }`}
          >
            {isSelected && (
              <span className="flex h-2 w-2 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-300 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-white"></span>
              </span>
            )}
            <span>
              {isSelected ? `★ ${trackText}` : trackText}
              {sourceName ? ` → ${sourceName}` : isExcluded ? ' (Excluded)' : ''}
            </span>
          </div>

          {/* ── Interactive Studio Face Inspector Popover (Active when Selected) ── */}
          {isSelected && (
            <div
              className="absolute top-full left-0 mt-2 z-50 flex flex-col gap-1.5 p-2.5 rounded-xl bg-black/95 backdrop-blur-xl border border-cyan-400/80 shadow-[0_0_30px_rgba(0,0,0,0.9)] min-w-[260px] text-white select-none pointer-events-auto"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header: Track & Euler Angles */}
              <div className="flex items-center justify-between border-b border-white/15 pb-1">
                <span className="text-micro font-bold text-cyan-300 uppercase tracking-wider">
                  {trackText} Inspector
                </span>
                <span
                  className={`px-1.5 py-0.2 rounded text-nano font-mono font-bold uppercase ${
                    poseClass.badgeTone === 'error'
                      ? 'bg-rose-950 text-rose-300 border border-rose-600/50'
                      : poseClass.badgeTone === 'warning'
                      ? 'bg-amber-950 text-amber-300 border border-amber-500/50'
                      : 'bg-emerald-950 text-emerald-300 border border-emerald-500/50'
                  }`}
                >
                  {poseClass.bucket}
                </span>
              </div>

              {/* Live Pose Readout: Yaw / Pitch / Roll */}
              <div className="grid grid-cols-3 gap-1 bg-zinc-900/80 p-1.5 rounded-lg border border-white/10 text-center font-mono">
                <div className="flex flex-col">
                  <span className="text-nano text-white/50">Yaw</span>
                  <span className={`text-micro font-bold ${Math.abs(poseClass.yawDeg) > 40 ? 'text-amber-400' : 'text-cyan-300'}`}>
                    {poseClass.yawDeg >= 0 ? `+${poseClass.yawDeg.toFixed(1)}°` : `${poseClass.yawDeg.toFixed(1)}°`}
                  </span>
                </div>
                <div className="flex flex-col border-x border-white/10">
                  <span className="text-nano text-white/50">Pitch</span>
                  <span className="text-micro font-bold text-sky-300">
                    {poseClass.pitchDeg >= 0 ? `+${poseClass.pitchDeg.toFixed(1)}°` : `${poseClass.pitchDeg.toFixed(1)}°`}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-nano text-white/50">Roll</span>
                  <span className="text-micro font-bold text-indigo-300">
                    {poseClass.rollDeg >= 0 ? `+${poseClass.rollDeg.toFixed(1)}°` : `${poseClass.rollDeg.toFixed(1)}°`}
                  </span>
                </div>
              </div>

              {/* Assign / Lock .fsz Faceset */}
              <div className="flex flex-col gap-1 pt-0.5">
                <div className="flex items-center justify-between text-nano text-white/70">
                  <span>Assigned Faceset:</span>
                  <span className="font-mono text-cyan-300 font-bold truncate max-w-[140px]">
                    {sourceName || 'None (Default)'}
                  </span>
                </div>
                {facesetAssignTrackId === trackId ? (
                  <div className="flex items-center gap-1">
                    <input
                      type="text"
                      placeholder="e.g. scarlett.fsz"
                      value={facesetInput}
                      onChange={(e) => setFacesetInput(e.target.value)}
                      className="flex-1 px-2 py-0.5 rounded bg-zinc-800 border border-cyan-500/50 text-micro text-white outline-none"
                      autoFocus
                    />
                    <button
                      type="button"
                      onClick={() => {
                        if (facesetInput.trim() && onAssignFaceset && trackId != null) {
                          onAssignFaceset(trackId, facesetInput.trim());
                        }
                        setFacesetAssignTrackId(null);
                        setFacesetInput('');
                      }}
                      className="px-2 py-0.5 rounded bg-cyan-600 hover:bg-cyan-500 text-micro font-bold text-white"
                    >
                      Lock
                    </button>
                    <button
                      type="button"
                      onClick={() => setFacesetAssignTrackId(null)}
                      className="px-1.5 py-0.5 rounded bg-white/10 text-micro text-white/70"
                    >
                      ✕
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={() => {
                      setFacesetAssignTrackId(trackId ?? faceIndex);
                      setFacesetInput(sourceName || '');
                    }}
                    className="w-full px-2 py-1 rounded bg-white/10 hover:bg-cyan-900/40 border border-white/15 hover:border-cyan-400 text-micro font-bold text-white flex items-center justify-center gap-1.5 transition-all"
                  >
                    <span>🎯 Assign / Lock .fsz</span>
                  </button>
                )}
              </div>

              {/* Toggle Swap On / Off Button */}
              <div className="flex items-center gap-1.5 pt-0.5">
                <button
                  type="button"
                  onClick={() => {
                    if (onToggleSwap) {
                      onToggleSwap(faceIndex, trackId);
                    } else if (onToggleExclude) {
                      onToggleExclude(faceIndex);
                    }
                  }}
                  className={`flex-1 px-2.5 py-1 rounded text-micro font-bold transition-all shadow ${
                    isExcluded
                      ? 'bg-amber-950/80 border border-amber-500/60 text-amber-300 hover:bg-amber-900/80'
                      : 'bg-emerald-950/80 border border-emerald-500/60 text-emerald-300 hover:bg-emerald-900/80'
                  }`}
                >
                  {isExcluded ? '⊘ Swap Excluded' : '✓ Swap Active'}
                </button>
                {onPinIdentity && (
                  <button
                    type="button"
                    onClick={() => onPinIdentity(faceIndex)}
                    className="px-2 py-1 rounded bg-white/10 hover:bg-white/20 text-micro font-bold text-white/90"
                    title="Pin identity"
                  >
                    📌 Pin
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Occlusion Warning Badge */}
          {typeof face.occlusion_coverage_pct === 'number' && face.occlusion_coverage_pct > 15 && (
            <div className="absolute -bottom-6 left-0 px-1.5 py-0.5 rounded bg-amber-950/90 border border-amber-500/50 text-amber-300 font-mono text-nano font-bold whitespace-nowrap shadow-md">
              ⚠ {face.occlusion_coverage_pct.toFixed(0)}% occluded
            </div>
          )}

          {/* BiSeNet Parsing / Landmark Points Overlay */}
          {showLandmarks && landmarks && Array.isArray(landmarks) && landmarks.length > 0 && (
            <div className="absolute inset-0 pointer-events-none">
              {landmarks.map((lm: any, lIdx: number) => {
                const pt = extractPoint(lm);
                const px = pt.x <= 1 ? pt.x * 100 : ((pt.x - (face.bbox as any)?.[0]) / ((face.bbox as any)?.[2] - (face.bbox as any)?.[0])) * 100;
                const py = pt.y <= 1 ? pt.y * 100 : ((pt.y - (face.bbox as any)?.[1]) / ((face.bbox as any)?.[3] - (face.bbox as any)?.[1])) * 100;
                return (
                  <span
                    key={`lm_${faceIndex}_${lIdx}`}
                    className="absolute w-1 h-1 rounded-full bg-cyan-300 -translate-x-1/2 -translate-y-1/2 shadow-[0_0_4px_rgba(34,211,238,0.9)]"
                    style={{ left: `${px}%`, top: `${py}%` }}
                  />
                );
              })}
            </div>
          )}
        </div>
      );
    });
  }, [
    showBoxes,
    showLandmarks,
    faces,
    naturalDim,
    selectedFaceIndex,
    selectedTrackId,
    personIds,
    faceRoles,
    facesetAssignTrackId,
    facesetInput,
    onSelectFace,
    onSelectTrack,
    onAssignFaceset,
    onToggleSwap,
    onToggleExclude,
    onPinIdentity,
  ]);

  // ── Render Component ────────────────────────────────────────────────────────
  const isVideoSource = playbackSource === 'video' || (playbackSource === 'auto' && Boolean(videoSrc));
  const { centerScreenX, centerScreenY, effectiveScale } = viewportMatrix.metrics;

  return (
    <div
      ref={containerRef}
      className={`relative w-full h-full min-h-[360px] bg-black/95 select-none overflow-hidden flex items-center justify-center font-sans ${className}`}
      style={style}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onWheel={handleWheel}
      onMouseEnter={() => setCursorOnViewport(true)}
      onMouseLeave={() => {
        setCursorOnViewport(false);
        if (isDraggingWipe) setIsDraggingWipe(false);
      }}
      data-testid="preview-viewport-container"
    >
      {/* ── Main Unified Media Transform Stage (Zero-Drift ViewportMatrix) ──── */}
      {(isPeekingOriginal || isPeekingRaw) && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full bg-amber-500/95 text-black font-mono font-bold text-micro shadow-[0_0_20px_rgba(245,158,11,0.8)] z-50 flex items-center gap-1.5 animate-pulse select-none">
          <span>👁️</span>
          <span>HOLD PEEK: ORIGINAL (UNSWAPPED)</span>
        </div>
      )}
      <div
        className="relative transition-transform duration-75 ease-out will-change-transform pointer-events-none"
        style={{
          width: `${naturalDim.w}px`,
          height: `${naturalDim.h}px`,
          transform: `translate3d(${centerScreenX - naturalDim.w / 2}px, ${centerScreenY - naturalDim.h / 2}px, 0) scale(${effectiveScale})`,
          transformOrigin: 'center center',
        }}
      >
        {/* Backing Video */}
        {isVideoSource && (
          <video
            ref={videoRef}
            src={videoSrc}
            playsInline
            muted
            className={`w-full h-full object-contain pointer-events-none rounded shadow-2xl ${
              viewMode === 'Split Wipe' || viewMode === 'Diff Heatmap' || topFrame || afterSrc ? 'hidden' : 'block'
            }`}
            onTimeUpdate={handleVideoTimeUpdate}
            onLoadedMetadata={(e) => {
              const v = e.currentTarget;
              if (v.videoWidth && v.videoHeight) {
                setNaturalDim({ w: v.videoWidth, h: v.videoHeight });
              }
            }}
          />
        )}

        {/* Backing Preview Canvas */}
        <canvas
          ref={canvasRef}
          width={naturalDim.w}
          height={naturalDim.h}
          className={`w-full h-full object-contain pointer-events-none rounded shadow-2xl ${
            viewMode === 'Split Wipe' || viewMode === 'Diff Heatmap' || topFrame || (!isVideoSource && (afterSrc || beforeSrc || baseFrame))
              ? 'block'
              : 'hidden'
          }`}
        />

        {/* Fallback Image Preview */}
        {!isVideoSource && !topFrame && !baseFrame && (afterSrc || beforeSrc || rawUrl) && (
          <img
            src={afterSrc || beforeSrc || rawUrl}
            alt="Preview Frame"
            className="w-full h-full object-contain pointer-events-none rounded shadow-2xl"
            onLoad={(e) => {
              const img = e.currentTarget;
              if (img.naturalWidth && img.naturalHeight) {
                setNaturalDim({ w: img.naturalWidth, h: img.naturalHeight });
              }
            }}
          />
        )}

        {/* ── Overlays: Face BBoxes & Studio Track Badges ──────────────────────── */}
        <div className="absolute inset-0 pointer-events-none z-20">
          {faceOverlayElements}
        </div>

        {/* ── Overlays: Mask Brush Canvas ──────────────────────────────────────── */}
        <canvas
          ref={maskCanvasRef}
          width={naturalDim.w}
          height={naturalDim.h}
          className={`absolute inset-0 w-full h-full object-contain z-25 ${
            maskBrushActive ? 'cursor-none pointer-events-auto' : 'pointer-events-none opacity-60'
          }`}
        />
        <canvas
          ref={maskExportRef}
          width={naturalDim.w}
          height={naturalDim.h}
          className="hidden"
          aria-hidden
        />
      </div>

      {/* ── Dual Concentric Circle Mask Brush Cursor ───────────────────────── */}
      {maskBrushActive && cursorOnViewport && (
        <div
          ref={brushCursorRef}
          className="absolute pointer-events-none z-50 -translate-x-1/2 -translate-y-1/2"
          style={{ width: brushSize, height: brushSize, willChange: 'transform' }}
          data-testid="mask-brush-cursor"
        >
          <div className="absolute inset-0 rounded-full border border-dashed border-pink-400 shadow-[0_0_8px_rgba(236,72,153,0.6)]" />
          <div
            className="absolute rounded-full border border-pink-300 left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"
            style={{
              width: Math.max(2, brushSize * (1 - brushFeather)),
              height: Math.max(2, brushSize * (1 - brushFeather)),
            }}
          />
          <span className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full bg-pink-200" />
        </div>
      )}

      {/* ── 3.5x Magnifier Loupe Floating Lens ─────────────────────────────── */}
      {showLoupe && cursorOnViewport && (
        <div
          className="absolute pointer-events-none z-50 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-cyan-400 bg-black/90 shadow-[0_0_25px_rgba(34,211,238,0.7)] overflow-hidden"
          style={{
            width: `${LOUPE_DIAMETER}px`,
            height: `${LOUPE_DIAMETER}px`,
            left: `${cursorPos.x}px`,
            top: `${cursorPos.y}px`,
          }}
          data-testid="magnifier-loupe"
        >
          <canvas
            ref={loupeCanvasRef}
            width={LOUPE_DIAMETER}
            height={LOUPE_DIAMETER}
            className="w-full h-full"
          />
          <div className="absolute bottom-2 inset-x-0 flex items-center justify-center pointer-events-none">
            <span className="px-2 py-0.5 rounded-full bg-black/80 border border-cyan-400/50 text-cyan-300 font-mono text-nano font-bold tracking-tight shadow">
              3.5x Loupe
            </span>
          </div>
        </div>
      )}

      {/* ── Floating Studio Viewport HUD ─────────────────────────────────────── */}
      {showHUD && (
        <div className="absolute inset-x-0 bottom-0 z-40 flex flex-col items-start gap-2 p-3 pointer-events-none">
          {/* ── Top Pill Row: Mode Switcher, Zoom, Frame Readout & Parity Badge ── */}
          <div className="flex flex-wrap items-center gap-2 max-w-full pointer-events-auto">
            {/* View Mode Switcher */}
            <div
              className="flex items-center gap-1 p-1 rounded-xl bg-black/90 backdrop-blur-md border border-white/20 shadow-2xl select-none text-white"
              data-testid="viewport-mode-switcher"
            >
              {(
                [
                  'Composite',
                  'Split Wipe',
                  'Diff Heatmap',
                  'BiSeNet Parsing',
                  'Occlusion Mask',
                ] as ViewportViewMode[]
              ).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setViewMode(mode)}
                  className={`px-2 py-0.5 rounded-lg text-micro font-bold transition-all ${
                    viewMode === mode
                      ? 'bg-cyan-500 text-black shadow-[0_0_10px_rgba(6,182,212,0.6)]'
                      : 'text-white/70 hover:text-white hover:bg-white/10'
                  }`}
                  title={`Switch to ${mode} mode`}
                >
                  {mode}
                </button>
              ))}
            </div>

            {/* 3.5x Loupe Toggle Button */}
            <button
              type="button"
              onClick={toggleLoupe}
              className={`px-2 py-1 rounded-xl border backdrop-blur-md text-micro font-bold flex items-center gap-1 shadow-2xl transition-all ${
                showLoupe
                  ? 'bg-cyan-950/90 border-cyan-400 text-cyan-300 shadow-[0_0_12px_rgba(34,211,238,0.5)]'
                  : 'bg-black/85 border-white/20 text-white/80 hover:border-cyan-400/50 hover:text-white'
              }`}
              title="Toggle 3.5x Magnifier Loupe"
            >
              <span>🔍 3.5x Loupe</span>
            </button>

            {/* Viewport Zoom & Preset Controls */}
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-xl bg-black/85 backdrop-blur-md border border-white/20 shadow-2xl select-none text-white">
              <span className="font-mono text-micro font-bold tabular-nums text-cyan-400 px-1.5 py-0.5 rounded bg-cyan-950/60 border border-cyan-500/30">
                {Math.round(zoom * 100)}%
              </span>
              <span className="w-px h-3.5 bg-white/20" />
              <button
                type="button"
                onClick={handlePresetFit}
                className="px-1.5 py-0.5 rounded text-micro font-mono font-bold bg-white/10 hover:bg-white/20 text-white/90 hover:text-white transition-colors"
                title="Fit frame to viewport (F)"
              >
                Fit
              </button>
              <button
                type="button"
                onClick={handlePresetFill}
                className="px-1.5 py-0.5 rounded text-micro font-mono font-bold bg-white/10 hover:bg-white/20 text-white/90 hover:text-white transition-colors"
                title="Fill viewport"
              >
                Fill
              </button>
              <button
                type="button"
                onClick={handlePreset1to1}
                className="px-1.5 py-0.5 rounded text-micro font-mono font-bold bg-white/10 hover:bg-white/20 text-white/90 hover:text-white transition-colors"
                title="1:1 Native Resolution"
              >
                1:1
              </button>
              <button
                type="button"
                onClick={handlePreset200}
                className="px-1.5 py-0.5 rounded text-micro font-mono font-bold bg-white/10 hover:bg-white/20 text-white/90 hover:text-white transition-colors"
                title="200% Zoom"
              >
                200%
              </button>
              <button
                type="button"
                onClick={handlePreset400}
                className="px-1.5 py-0.5 rounded text-micro font-mono font-bold bg-white/10 hover:bg-white/20 text-white/90 hover:text-white transition-colors"
                title="400% Zoom"
              >
                400%
              </button>
            </div>

            {/* Exact Integer Frame Readout Pill */}
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-xl bg-black/85 backdrop-blur-md border border-white/20 shadow-2xl select-none text-white">
              <span className="text-micro font-mono font-bold text-white/90">
                Frame <span className="text-cyan-400 font-extrabold">{exactFrame}</span> / {maxFrames}
              </span>
              <span className="w-px h-3.5 bg-white/20" />
              <span className="text-micro font-mono font-bold text-white/70 tabular-nums">
                {timecode}
              </span>
              <span className="text-nano font-mono text-white/50">
                ({formatSeconds(exactTimeSeconds)})
              </span>
            </div>

            {/* ── PREVIEW EXPORT PARITY INDICATOR BADGE ────────────────────────── */}
            <div
              data-testid="preview-export-parity-badge"
              data-parity-active={isParityActive ? 'true' : 'false'}
              className={`flex items-center gap-2 px-3 py-1 rounded-xl border backdrop-blur-md shadow-2xl select-none transition-all duration-300 ${
                isParityActive
                  ? 'bg-emerald-950/90 border-emerald-500/60 text-emerald-300 shadow-[0_0_15px_rgba(16,185,129,0.4)]'
                  : 'bg-zinc-900/90 border-white/20 text-zinc-400 hover:border-amber-500/40'
              }`}
              title={
                isParityActive
                  ? 'Preview Export Parity: 100% Locked. Current frame reads directly from active TrajectoryPlan cache.'
                  : 'Preview Export Parity: Unverified. Run "1 Pre-Scan" to seal TrajectoryPlan and lock export parity.'
              }
            >
              <span className="relative flex h-2.5 w-2.5">
                {isParityActive && (
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                )}
                <span
                  className={`relative inline-flex rounded-full h-2.5 w-2.5 ${
                    isParityActive ? 'bg-emerald-400' : 'bg-amber-400/70'
                  }`}
                />
              </span>
              <div className="flex flex-col">
                <div className="flex items-center gap-1">
                  <span className="text-micro font-bold tracking-tight uppercase">
                    {isParityActive ? '✓ Preview Export Parity' : 'Export Parity: Ad-Hoc'}
                  </span>
                  {isParityActive && (
                    <span className="px-1 py-0.2 rounded bg-emerald-500/20 text-emerald-300 text-nano font-mono font-bold">
                      TrajectoryPlan
                    </span>
                  )}
                </div>
                <span className="text-nano font-mono text-white/60 -mt-0.5">
                  {isParityActive ? '100% Frame-Identical' : 'Pre-Scan to lock parity'}
                </span>
              </div>
            </div>

            {/* Telemetry Toggle Button */}
            {telemetry && (
              <button
                type="button"
                onClick={() => setTelemetryOpen((v) => !v)}
                className="px-2 py-1 rounded-xl bg-black/85 backdrop-blur-md border border-white/20 hover:border-white/40 text-micro font-mono text-white/80 hover:text-white flex items-center gap-1 shadow-2xl transition-all"
                title="Toggle Telemetry Readout"
              >
                <span>⚡ Telemetry</span>
                {telemetry.fps && (
                  <span className="text-cyan-400 font-bold">{telemetry.fps.toFixed(1)} fps</span>
                )}
              </button>
            )}

            {/* Mask Brush Quick Toggle */}
            {onToggleMaskBrush && (
              <button
                type="button"
                onClick={onToggleMaskBrush}
                className={`px-2.5 py-1 rounded-xl border backdrop-blur-md text-micro font-bold flex items-center gap-1.5 shadow-2xl transition-all ${
                  maskBrushActive
                    ? 'bg-pink-950/90 border-pink-500/80 text-pink-300 shadow-[0_0_12px_rgba(236,72,153,0.5)]'
                    : 'bg-black/85 border-white/20 text-white/80 hover:border-pink-500/50 hover:text-white'
                }`}
                title="Toggle Sub-Pixel Mask Brush (B)"
              >
                <span>🖌️ Mask Brush</span>
                {maskApplied && <span className="text-emerald-400 text-nano">✓ Applied</span>}
              </button>
            )}
          </div>

          {/* ── Split Wipe Divider Control Slider (Visible in Split Wipe mode) ── */}
          {viewMode === 'Split Wipe' && (
            <div className="flex items-center gap-3 px-3 py-1.5 rounded-xl bg-black/90 backdrop-blur-md border border-cyan-500/40 shadow-2xl pointer-events-auto text-white select-none">
              <span className="text-micro font-bold text-cyan-300">Wipe Split:</span>
              <input
                type="range"
                min="0"
                max="100"
                value={wipe}
                onChange={(e) => setWipe(Number(e.target.value))}
                className="w-36 accent-cyan-400 cursor-pointer"
              />
              <span className="text-micro font-mono text-cyan-300 tabular-nums w-12">
                {Math.round(wipe)}%
              </span>
              <div className="flex items-center gap-1 text-nano font-mono text-white/60">
                <span className="px-1.5 py-0.5 rounded bg-white/10">RAW: 0..{Math.round(wipe)}%</span>
                <span className="px-1.5 py-0.5 rounded bg-white/10">SWAP: {Math.round(wipe)}..100%</span>
              </div>
            </div>
          )}

          {/* ── Mask Brush Control Drawer (Visible when brush is active) ────── */}
          {maskBrushActive && (
            <div className="flex flex-wrap items-center gap-2 px-3 py-1.5 rounded-xl bg-black/90 backdrop-blur-md border border-pink-500/40 shadow-2xl pointer-events-auto text-white select-none">
              <span className="text-micro font-bold text-pink-300 flex items-center gap-1">
                <span>🖌️ Mode:</span>
              </span>
              <button
                type="button"
                onClick={() => setBrushMode('paint')}
                className={`px-2 py-0.5 rounded text-micro font-bold transition-all ${
                  brushMode === 'paint'
                    ? 'bg-pink-600 text-white shadow-md'
                    : 'bg-white/10 text-white/70 hover:bg-white/20'
                }`}
              >
                Paint
              </button>
              <button
                type="button"
                onClick={() => setBrushMode('erase')}
                className={`px-2 py-0.5 rounded text-micro font-bold transition-all ${
                  brushMode === 'erase'
                    ? 'bg-amber-500 text-black shadow-md'
                    : 'bg-white/10 text-white/70 hover:bg-white/20'
                }`}
              >
                Erase
              </button>
              <span className="w-px h-3.5 bg-white/20" />
              {/* Radius / Size */}
              <div className="flex items-center gap-1.5">
                <span className="text-nano font-mono text-white/70">Radius:</span>
                <input
                  type="range"
                  min="4"
                  max="160"
                  value={brushSize}
                  onChange={(e) => setBrushSize(Number(e.target.value))}
                  className="w-20 accent-pink-500 cursor-pointer"
                />
                <span className="text-micro font-mono text-pink-300 tabular-nums w-8">
                  {brushSize}px
                </span>
              </div>
              <span className="w-px h-3.5 bg-white/20" />
              {/* Hardness / Softness (Feather) */}
              <div className="flex items-center gap-1.5">
                <span className="text-nano font-mono text-white/70">Feather:</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={brushFeather}
                  onChange={(e) => setBrushFeather(Number(e.target.value))}
                  className="w-16 accent-pink-500 cursor-pointer"
                />
                <span className="text-micro font-mono text-pink-300 tabular-nums w-8">
                  {Math.round(brushFeather * 100)}%
                </span>
              </div>
              <span className="w-px h-3.5 bg-white/20" />
              {/* Undo / Redo Buttons */}
              <button
                type="button"
                onClick={handleUndo}
                disabled={onMaskStrokesChange ? !projectCanUndo : undoStackRef.current.length === 0}
                className={`px-2 py-0.5 rounded text-micro font-bold transition-colors ${
                  undoStackRef.current.length > 0
                    ? 'bg-white/15 hover:bg-white/25 text-white'
                    : 'bg-white/5 text-white/30 cursor-not-allowed'
                }`}
                title="Undo mask stroke (Ctrl+Z)"
              >
                ↺ Undo
              </button>
              <button
                type="button"
                onClick={handleRedo}
                disabled={onMaskStrokesChange ? !projectCanRedo : redoStackRef.current.length === 0}
                className={`px-2 py-0.5 rounded text-micro font-bold transition-colors ${
                  redoStackRef.current.length > 0
                    ? 'bg-white/15 hover:bg-white/25 text-white'
                    : 'bg-white/5 text-white/30 cursor-not-allowed'
                }`}
                title="Redo mask stroke (Ctrl+Y)"
              >
                ↻ Redo
              </button>
              <span className="w-px h-3.5 bg-white/20" />
              {/* Invert Mask */}
              <button
                type="button"
                onClick={handleInvertMask}
                className="px-2 py-0.5 rounded text-micro font-bold bg-white/10 hover:bg-purple-900/60 text-white/80 hover:text-purple-200 transition-colors"
                title="Invert mask colors"
              >
                ◐ Invert
              </button>
              {onClearMask && (
                <button
                  type="button"
                  onClick={onClearMask}
                  className="px-2 py-0.5 rounded text-micro font-bold bg-white/10 hover:bg-rose-900/60 text-white/80 hover:text-rose-200 transition-colors"
                >
                  Clear
                </button>
              )}
              {onApplyMask && (
                <button
                  type="button"
                  onClick={onApplyMask}
                  className={`px-2.5 py-0.5 rounded text-micro font-bold shadow-md transition-all ${
                    maskApplied
                      ? 'bg-emerald-500 text-black'
                      : 'bg-pink-600 hover:bg-pink-500 text-white'
                  }`}
                >
                  {maskApplied ? '✓ Mask Applied' : 'Apply Mask'}
                </button>
              )}
            </div>
          )}

          {/* ── Telemetry Popover Drawer ─────────────────────────────────────── */}
          {telemetryOpen && telemetry && (
            <div className="flex flex-col gap-1 p-3 rounded-xl bg-black/95 backdrop-blur-md border border-cyan-500/40 shadow-2xl pointer-events-auto text-white select-none max-w-sm">
              <div className="flex items-center justify-between border-b border-white/15 pb-1">
                <span className="text-micro font-bold text-cyan-300 uppercase tracking-wider">
                  Live Viewport Telemetry
                </span>
                <button
                  type="button"
                  onClick={() => setTelemetryOpen(false)}
                  className="text-white/60 hover:text-white text-micro font-bold px-1"
                >
                  ✕
                </button>
              </div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-micro font-mono pt-1">
                <span className="text-white/60">Render FPS:</span>
                <span className="text-cyan-400 font-bold">{telemetry.fps ? telemetry.fps.toFixed(1) : '–'}</span>
                <span className="text-white/60">Faces / Sec:</span>
                <span className="text-white/90">{telemetry.faces_per_sec ? telemetry.faces_per_sec.toFixed(1) : '–'}</span>
                <span className="text-white/60">Tracks Source:</span>
                <span className={telemetry.tracks_source === 'trajectory_plan' ? 'text-emerald-400 font-bold' : 'text-amber-400'}>
                  {telemetry.tracks_source || 'none'}
                </span>
                <span className="text-white/60">Trajectory Plan:</span>
                <span className={telemetry.trajectory_plan_active ? 'text-emerald-400 font-bold' : 'text-zinc-500'}>
                  {telemetry.trajectory_plan_active ? 'Active' : 'Inactive'}
                </span>
                <span className="text-white/60">Detect Time:</span>
                <span className="text-white/90">{telemetry.detect_ms ? `${telemetry.detect_ms.toFixed(1)} ms` : '–'}</span>
                <span className="text-white/60">Swap Time:</span>
                <span className="text-white/90">{telemetry.swap_ms ? `${telemetry.swap_ms.toFixed(1)} ms` : '–'}</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PreviewViewport;
