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

// ── Types & Interfaces ─────────────────────────────────────────────────────────

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
  pose?: PreviewFacePose;
  role?: 'target' | 'swapped' | 'bystander' | string;
  source_name?: string | null;
  sourceName?: string | null;
  source_index?: number | null;
  sourceIndex?: number | null;
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

const ZOOM_MIN = 0.2;
const ZOOM_MAX = 5.0;

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
  wipe = 100,
  wipeAngle = 0,
  compare = false,
  compareMode = 'slider',
  onToggleCompare,

  // HUD
  showHUD = true,
  showBoxes = true,
  showLandmarks = false,
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

  const videoRef = externalVideoRef || internalVideoRef;
  const canvasRef = externalCanvasRef || internalCanvasRef;
  const maskCanvasRef = externalMaskCanvasRef || internalMaskCanvasRef;
  const maskExportRef = externalMaskExportRef || internalMaskExportRef;
  const containerRef = externalContainerRef || internalContainerRef;

  const [internalZoom, setInternalZoom] = useState<number>(1.0);
  const zoom = controlledZoom !== undefined ? controlledZoom : internalZoom;
  const setZoom = controlledSetZoom || setInternalZoom;

  const [internalPan, setInternalPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const pan = controlledPan !== undefined ? controlledPan : internalPan;
  const setPan = controlledSetPan || setInternalPan;

  const [internalBrushMode, setInternalBrushMode] = useState<'paint' | 'erase'>('paint');
  const brushMode = controlledBrushMode || internalBrushMode;
  const setBrushMode = controlledSetBrushMode || setInternalBrushMode;

  const [internalBrushSize, setInternalBrushSize] = useState<number>(32);
  const brushSize = controlledBrushSize !== undefined ? controlledBrushSize : internalBrushSize;
  const setBrushSize = controlledSetBrushSize || setInternalBrushSize;

  const [internalBrushFeather, setInternalBrushFeather] = useState<number>(0.35);
  const brushFeather = controlledBrushFeather !== undefined ? controlledBrushFeather : internalBrushFeather;
  const setBrushFeather = controlledSetBrushFeather || setInternalBrushFeather;

  const [internalMaskStrokes, setInternalMaskStrokes] = useState<MaskBrushStroke[]>([]);
  const maskStrokes = controlledMaskStrokes || internalMaskStrokes;
  const setMaskStrokes = onMaskStrokesChange || setInternalMaskStrokes;

  // Media aspect & natural dimensions
  const [naturalDim, setNaturalDim] = useState<{ w: number; h: number }>({ w: 1920, h: 1080 });
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const panStartRef = useRef<{ x: number; y: number; startPanX: number; startPanY: number }>({
    x: 0,
    y: 0,
    startPanX: 0,
    startPanY: 0,
  });

  // Brush cursor tracking & stroke in-flight
  const brushCursorRef = useRef<HTMLDivElement | null>(null);
  const isBrushingRef = useRef<boolean>(false);
  const currentStrokePointsRef = useRef<Array<[number, number]>>([]);

  // Telemetry drawer popover state
  const [telemetryOpen, setTelemetryOpen] = useState<boolean>(false);

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

  // Video time update handler: binary searches exact integer frame_idx
  const handleVideoTimeUpdate = useCallback(() => {
    const vid = videoRef.current;
    if (!vid) return;
    const resolvedFrame = resolveFrameFromTime(vid.currentTime, ptsTable, fps);
    const clamped = clampFrameIdx(resolvedFrame, 0, maxFrames);
    if (setFrame && clamped !== exactFrame) {
      setFrame(clamped);
    }
  }, [exactFrame, fps, maxFrames, ptsTable, setFrame, videoRef]);

  // ── Preview Export Parity State ─────────────────────────────────────────────
  const parity: PreviewExportParityState = useMemo(() => {
    if (controlledParityState && 'badgeTone' in controlledParityState) {
      return controlledParityState as PreviewExportParityState;
    }
    // Bridge from telemetry or props
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

  // ── Canvas Frame Drawing & Dimension Detection ──────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (topFrame) {
      if (canvas.width !== topFrame.width || canvas.height !== topFrame.height) {
        canvas.width = topFrame.width;
        canvas.height = topFrame.height;
        setNaturalDim({ w: topFrame.width, h: topFrame.height });
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(topFrame, 0, 0);
    } else if (baseFrame) {
      if (canvas.width !== baseFrame.width || canvas.height !== baseFrame.height) {
        canvas.width = baseFrame.width;
        canvas.height = baseFrame.height;
        setNaturalDim({ w: baseFrame.width, h: baseFrame.height });
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(baseFrame, 0, 0);
    }
  }, [topFrame, baseFrame, canvasRef]);

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

  // ── Pan and Zoom Handlers ───────────────────────────────────────────────────
  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (maskBrushActive) {
      // Start mask stroke
      isBrushingRef.current = true;
      const maskCanvas = maskCanvasRef.current;
      if (!maskCanvas) return;
      const rect = maskCanvas.getBoundingClientRect();
      const scaleX = maskCanvas.width / rect.width;
      const scaleY = maskCanvas.height / rect.height;
      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;
      currentStrokePointsRef.current = [[x, y]];

      const ctx = maskCanvas.getContext('2d');
      if (ctx) {
        ctx.save();
        ctx.globalCompositeOperation = brushMode === 'erase' ? 'destination-out' : 'source-over';
        ctx.fillStyle = brushMode === 'erase' ? 'rgba(0,0,0,1)' : 'rgba(236,72,153,0.7)';
        ctx.beginPath();
        ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
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
    // Update dual-concentric brush cursor
    if (maskBrushActive && brushCursorRef.current && containerRef.current) {
      const containerRect = containerRef.current.getBoundingClientRect();
      const curX = e.clientX - containerRect.left;
      const curY = e.clientY - containerRect.top;
      brushCursorRef.current.style.transform = `translate3d(${curX}px, ${curY}px, 0)`;
    }

    if (maskBrushActive && isBrushingRef.current) {
      const maskCanvas = maskCanvasRef.current;
      if (!maskCanvas) return;
      const rect = maskCanvas.getBoundingClientRect();
      const scaleX = maskCanvas.width / rect.width;
      const scaleY = maskCanvas.height / rect.height;
      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

      const pts = currentStrokePointsRef.current;
      const prev = pts[pts.length - 1];
      pts.push([x, y]);

      const ctx = maskCanvas.getContext('2d');
      if (ctx && prev) {
        ctx.save();
        ctx.globalCompositeOperation = brushMode === 'erase' ? 'destination-out' : 'source-over';
        ctx.strokeStyle = brushMode === 'erase' ? 'rgba(0,0,0,1)' : 'rgba(236,72,153,0.7)';
        ctx.lineWidth = brushSize;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        ctx.moveTo(prev[0], prev[1]);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.restore();
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
    if (maskBrushActive && isBrushingRef.current) {
      isBrushingRef.current = false;
      const pts = currentStrokePointsRef.current;
      if (pts.length > 0) {
        const newStroke = createMaskBrushStroke(
          {
            frame_idx: exactFrame, // integer frame!
            tool: brushMode === 'erase' ? 'eraser' : 'brush',
            size: brushSize,
            points: pts,
            color: brushMode === 'erase' ? 'rgba(0,0,0,1)' : 'rgba(236,72,153,0.7)',
          },
          maxFrames
        );
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
      // Wheel adjusts brush size
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

  // ── Face Detection Bounding Boxes & Badges with Selected Highlight ──────────
  const faceOverlayElements = useMemo(() => {
    if (!showBoxes || !faces || faces.length === 0) return null;

    return faces.map((face, i) => {
      const faceIndex = face.index ?? face.face_index ?? i;
      const personId = face.personId ?? face.person_id ?? personIds[i] ?? null;
      const trackId = face.trackId ?? face.track_id ?? personId;

      // Check whether this face matches the currently selected track lane or face index
      const isSelectedByIndex = selectedFaceIndex != null && selectedFaceIndex === faceIndex;
      const isSelectedByTrack =
        selectedTrackId != null &&
        (trackId === selectedTrackId ||
          personId === selectedTrackId ||
          Number(trackId) === Number(selectedTrackId) ||
          Number(personId) === Number(selectedTrackId));
      const isSelected = isSelectedByIndex || isSelectedByTrack;

      // Calculate BBox percentages
      let left = 0;
      let top = 0;
      let width = 0;
      let height = 0;

      if (Array.isArray(face.bbox) && face.bbox.length >= 4) {
        const [x0, y0, x1, y1] = face.bbox;
        if (x1 > 1 || y1 > 1) {
          // Pixel coordinates relative to natural dimensions
          left = (x0 / naturalDim.w) * 100;
          top = (y0 / naturalDim.h) * 100;
          width = ((x1 - x0) / naturalDim.w) * 100;
          height = ((y1 - y0) / naturalDim.h) * 100;
        } else {
          // Normalized 0..1
          left = x0 * 100;
          top = y0 * 100;
          width = (x1 - x0) * 100;
          height = (y1 - y0) * 100;
        }
      } else if (typeof face.bbox === 'object' && face.bbox !== null) {
        const b = face.bbox as { x: number; y: number; width: number; height: number };
        if (b.width > 1 || b.height > 1) {
          left = (b.x / naturalDim.w) * 100;
          top = (b.y / naturalDim.h) * 100;
          width = (b.width / naturalDim.w) * 100;
          height = (b.height / naturalDim.h) * 100;
        } else {
          left = b.x * 100;
          top = b.y * 100;
          width = b.width * 100;
          height = b.height * 100;
        }
      }

      // Track label
      const trackText = formatTrackLabel(trackId);
      const sourceName = face.sourceName || face.source_name || '';
      const role = face.role || (faceRoles && (faceRoles as any)[faceIndex]?.role);
      const isExcluded = Boolean(face.excluded || (faceRoles && (faceRoles as any)[faceIndex]?.excluded));

      // Badge display text
      let badgeContent = `${trackText}`;
      if (isExcluded) {
        badgeContent += ' → excluded (never swapped)';
      } else if (role === 'bystander') {
        badgeContent += ' → bystander';
      } else if (sourceName) {
        badgeContent += ` → ${sourceName}`;
      } else if (role === 'swapped') {
        badgeContent += ' → swapped';
      } else {
        badgeContent += ` (Face ${faceIndex + 1})`;
      }

      return (
        <div
          key={`face_${faceIndex}_${trackId || i}`}
          data-testid="face-bounding-box"
          data-face-index={faceIndex}
          data-track-id={trackId ?? ''}
          data-selected={isSelected ? 'true' : 'false'}
          className={`absolute transition-all duration-200 pointer-events-auto cursor-pointer group ${
            isSelected
              ? 'border-2 border-cyan-400 bg-cyan-400/20 ring-4 ring-cyan-400/50 shadow-[0_0_20px_rgba(34,211,238,0.7)] z-30'
              : 'border border-[var(--accent,#38bdf8)] hover:border-white/90 hover:bg-white/10 z-20'
          }`}
          style={{
            left: `${Math.max(0, left)}%`,
            top: `${Math.max(0, top)}%`,
            width: `${Math.min(100, Math.max(2, width))}%`,
            height: `${Math.min(100, Math.max(2, height))}%`,
          }}
          onClick={(e) => {
            e.stopPropagation();
            if (onSelectFace) onSelectFace(faceIndex, face);
            if (onSelectTrack && trackId != null) onSelectTrack(trackId);
          }}
          title={`Click to select ${trackText} (Face #${faceIndex + 1})`}
        >
          {/* Face Badge Header */}
          <div
            data-testid={isSelected ? 'face-identity-badge-highlighted' : 'face-identity-badge'}
            className={`absolute -top-7 left-0 flex items-center gap-1.5 px-2 py-0.5 rounded shadow-lg text-micro font-bold tracking-tight whitespace-nowrap transition-all duration-200 ${
              isSelected
                ? 'bg-gradient-to-r from-cyan-500 via-sky-500 to-blue-600 text-white ring-2 ring-white shadow-[0_0_15px_rgba(6,182,212,0.9)] scale-105 z-40'
                : 'bg-black/80 backdrop-blur-md text-white/90 border border-white/20 group-hover:border-cyan-400 group-hover:text-white'
            }`}
          >
            {isSelected && (
              <span className="flex h-2 w-2 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-300 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-white"></span>
              </span>
            )}
            <span>{isSelected ? `★ ${badgeContent}` : badgeContent}</span>
          </div>

          {/* Occlusion / Score Badge if present */}
          {typeof face.occlusion_coverage_pct === 'number' && face.occlusion_coverage_pct > 15 && (
            <div className="absolute -bottom-6 left-0 px-1.5 py-0.5 rounded bg-amber-950/90 border border-amber-500/50 text-amber-300 font-mono text-nano font-bold">
              {face.occlusion_coverage_pct.toFixed(0)}% occluded
            </div>
          )}
        </div>
      );
    });
  }, [
    showBoxes,
    faces,
    naturalDim,
    selectedFaceIndex,
    selectedTrackId,
    personIds,
    faceRoles,
    onSelectFace,
    onSelectTrack,
  ]);

  // ── Render Component ────────────────────────────────────────────────────────
  const isVideoSource = playbackSource === 'video' || (playbackSource === 'auto' && Boolean(videoSrc));

  return (
    <div
      ref={containerRef}
      className={`relative w-full h-full min-h-[360px] bg-black/95 select-none overflow-hidden flex items-center justify-center font-sans ${className}`}
      style={style}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onWheel={handleWheel}
      data-testid="preview-viewport-container"
    >
      {/* ── Main Media Transform Container (Zoom / Pan) ───────────────────────── */}
      <div
        className="relative flex items-center justify-center transition-transform duration-75 ease-out will-change-transform"
        style={{
          transform: `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${zoom})`,
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
            className={`max-w-none object-contain pointer-events-none rounded shadow-2xl ${
              topFrame || afterSrc ? 'hidden' : 'block'
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
          className={`max-w-none object-contain pointer-events-none rounded shadow-2xl ${
            topFrame || (!isVideoSource && (afterSrc || beforeSrc || baseFrame)) ? 'block' : 'hidden'
          }`}
        />

        {/* Fallback Image Preview if URL provided without bitmap */}
        {!isVideoSource && !topFrame && !baseFrame && (afterSrc || beforeSrc || rawUrl) && (
          <img
            src={afterSrc || beforeSrc || rawUrl}
            alt="Preview Frame"
            className="max-w-none object-contain pointer-events-none rounded shadow-2xl"
            onLoad={(e) => {
              const img = e.currentTarget;
              if (img.naturalWidth && img.naturalHeight) {
                setNaturalDim({ w: img.naturalWidth, h: img.naturalHeight });
              }
            }}
          />
        )}

        {/* ── Overlays: Face BBoxes & Badges ───────────────────────────────────── */}
        <div
          className="absolute inset-0 pointer-events-none z-20"
          style={{ width: '100%', height: '100%' }}
        >
          {faceOverlayElements}
        </div>

        {/* ── Overlays: Mask Brush Canvases ────────────────────────────────────── */}
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
      {maskBrushActive && (
        <div
          ref={brushCursorRef}
          className="absolute pointer-events-none z-50 -translate-x-1/2 -translate-y-1/2"
          style={{ width: brushSize, height: brushSize, willChange: 'transform' }}
          data-testid="mask-brush-cursor"
        >
          {/* Outer dashed ring (radius) */}
          <div className="absolute inset-0 rounded-full border border-dashed border-pink-400 shadow-[0_0_8px_rgba(236,72,153,0.6)]" />
          {/* Inner solid ring (feather core) */}
          <div
            className="absolute rounded-full border border-pink-300 left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"
            style={{
              width: Math.max(2, brushSize * (1 - brushFeather)),
              height: Math.max(2, brushSize * (1 - brushFeather)),
            }}
          />
          {/* Center pinpoint */}
          <span className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full bg-pink-200" />
        </div>
      )}

      {/* ── Bottom HUD Pill & Toolbar ────────────────────────────────────────── */}
      {showHUD && (
        <div className="absolute inset-x-0 bottom-0 z-40 flex flex-col items-start gap-2 p-3 pointer-events-none">
          {/* Top Pill Row: Zoom & Presets & Frame Readout & Parity Badge */}
          <div className="flex flex-wrap items-center gap-2 max-w-full pointer-events-auto">
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
              {/* Pulsing Status Dot */}
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

              {/* Parity Badge Text */}
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
                  {isParityActive
                    ? '100% Frame-Identical'
                    : 'Pre-Scan to lock parity'}
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

          {/* Mask Brush Control Drawer (Visible when brush is active) */}
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
              <div className="flex items-center gap-1.5">
                <span className="text-nano font-mono text-white/70">Size:</span>
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

          {/* Telemetry Popover Drawer */}
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
