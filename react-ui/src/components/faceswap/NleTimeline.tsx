/**
 * NleTimeline.tsx: Multi-Track Non-Linear Editing Timeline & Keyframe Editor.
 *
 * Capabilities:
 *   1. Ruler & Timecode Header: 'HH:MM:SS:FF' frame-accurate ruler with zoom-to-cursor (Ctrl + Wheel),
 *      magnetic snapping to playhead, markers, In/Out trim handles ('I' / 'O'), and chapter markers ('M').
 *   2. V1 Video Filmstrip Lane: Render cached thumbnail strips from /api/target/preview_grid.
 *   3. A1 Audio Waveform Lane: Render audio peak waveform from /api/media/waveform.
 *   4. Per-Target Face Track Lanes ('Track #1', 'Track #2', ...): Horizontal presence bars for every ByteTrack ID,
 *      color-coded by assigned .fsz faceset, with amber/red micro-indicators where hard yaw (>50) or
 *      foreign occlusion (>25%) occurs so users can jump straight to challenging frames.
 *   5. Keyframe Lane ('KF'): Add, drag, copy, and delete parameter keyframes ('Swap Opacity', 'Mask Feather',
 *      'Yaw Erosion', 'Color Match') with interpolation curve selector ('Linear', 'Ease In/Out', 'Hold').
 *   6. 100% Offline Compliance: zero external URLs, CDNs, or fonts.
 *   7. Zero Scrubber Stalls: Cached drag geometry + RAF coalesced dispatch + pointer capture.
 */

import React, {
  useState,
  useMemo,
  useCallback,
  useRef,
  useEffect,
  type ReactNode,
  type MouseEvent,
  type WheelEvent,
  type KeyboardEvent,
} from 'react';

import {
  type RangeLockInterval,
  type KeyframePoint,
  type PtsTableRawInput,
  type PtsTableEntry,
  PtsFrameMap,
  createPtsFrameMap,
  resolveFrameFromTime,
  resolveTimeFromFrame,
  clampFrameIdx,
  frameToTimecode,
  timecodeToFrame,
  formatSeconds,
  createRangeLockInterval,
  clampRangeLockInterval,
  createKeyframePoint,
  clampKeyframe,
  normalizeKeyframePoints,
} from '../../utils/ptsFrameMap';

import {
  type ChapterMarker,
  type SnapTarget,
  type SnapResult,
  snapToFrame,
  calculateZoomToCursor,
  generateRulerTicks,
} from './timeline/timelineSnap';

import {
  type TimelineParamType,
  type TimelineCurveType,
  type NleKeyframe,
  PARAMETER_CONFIGS,
  createNleKeyframe,
  sortAndNormalizeKeyframes,
  evaluateKeyframeValue,
  duplicateKeyframe,
} from './timeline/parameterKeyframes';

import { useAudioWaveform } from './timeline/useAudioWaveform';
import { useFilmstripThumbnails } from './timeline/useFilmstripThumbnails';

// Re-export core interval, keyframe, and marker interfaces
export type { RangeLockInterval, KeyframePoint, ChapterMarker, NleKeyframe, TimelineParamType, TimelineCurveType };

// ── Types & Interfaces ─────────────────────────────────────────────────────────

export interface ByteTrackLane {
  trackId: number;
  label: string;
  firstFrame: number;
  lastFrame: number;
  totalFrames: number;
  ranges: Array<[number, number]>;
  sourceIndex: number | null;
  facesetId: string | null;
  facesetPath: string | null;
  suggestedFsz: string | null;
  via: string;
  reason?: string;
  active: boolean;
  color?: string;
}

export interface WarningTick {
  frame: number; // exact integer frame_idx
  trackId: number;
  type: 'pose' | 'occlusion' | 'interpolated' | 'coasted' | 'gap' | string;
  severity: 'warning' | 'info' | 'error';
  message: string;
  value?: number;
}

export interface SuggestedFszAssignment {
  trackId: number;
  sourceIndex: number;
  facesetId: string | null;
  facesetPath: string | null;
  name: string;
  suggestedFileName: string;
  confidence?: number;
}

export interface NleTimelineProps {
  // Current playback position
  frame?: number;
  currentFrame?: number;
  currentTime?: number;
  maxFrames: number;
  fps?: number;
  isPlaying?: boolean;

  // Media targeting (for waveform & filmstrip)
  targetIndex?: number | null;
  mediaId?: string | null;
  targetKey?: string;

  // PTS table & VFR exact mapping
  ptsTable?: PtsTableRawInput | PtsFrameMap | PtsTableEntry[];
  ptsFrameMap?: PtsFrameMap;

  // Tracks, warnings, and suggested assignments from Step 1 Pre-Scan
  trackLanes?: ByteTrackLane[];
  warningTicks?: WarningTick[];
  suggestedAssignments?: SuggestedFszAssignment[];

  // Step 2 Range Lock intervals
  rangeLockIntervals?: RangeLockInterval[];

  // Selection state (linked to PreviewViewport)
  selectedTrackId?: number | string | null;
  selectedRangeId?: string | null;

  // Global clip trim in/out (exact integer frame_idx)
  trimIn?: number;
  trimOut?: number;

  // Chapter markers ('M')
  markers?: ChapterMarker[];

  // Identity / parameter keyframes (exact integer frame_idx)
  keyframes?: NleKeyframe[] | KeyframePoint[];
  activeKeyframeParam?: TimelineParamType;

  // Snapping toggle
  snapEnabled?: boolean;

  // Event handlers
  onSeek?: (frame: number, timeSeconds?: number) => void;
  onSelectTrack?: (trackId: number | string | null) => void;
  onSelectRange?: (interval: RangeLockInterval | null) => void;
  onUpdateRangeLockIntervals?: (intervals: RangeLockInterval[]) => void;
  onToggleTrackActive?: (trackId: number, active: boolean) => void;
  onUpdateTrim?: (trimIn: number, trimOut: number) => void;
  onSetTrimIn?: (frame: number) => void;
  onSetTrimOut?: (frame: number) => void;
  onAddMarker?: (marker: ChapterMarker) => void;
  onDeleteMarker?: (markerId: string) => void;
  onUpdateMarker?: (marker: ChapterMarker) => void;
  onUpdateKeyframes?: (keyframes: NleKeyframe[]) => void;
  onAssignFsz?: (trackId: number, fszFileName: string) => void;
  onPreScanClick?: () => void;

  // Step indicators & UI flags
  isPreScanActive?: boolean;
  isRangeLockActive?: boolean;
  isRenderActive?: boolean;
  showWarningTicks?: boolean;
  showSuggestedFsz?: boolean;
  showFilmstripLane?: boolean;
  showWaveformLane?: boolean;
  showKeyframeLane?: boolean;

  // Customization
  className?: string;
  style?: React.CSSProperties;
}

// ── Palette Constants ─────────────────────────────────────────────────────────

export const TRACK_PALETTE: readonly string[] = Object.freeze([
  '#38bdf8', // Sky 400
  '#34d399', // Emerald 400
  '#fbbf24', // Amber 400
  '#f472b6', // Pink 400
  '#a78bfa', // Violet 400
  '#2dd4bf', // Teal 400
  '#fb923c', // Orange 400
  '#e879f9', // Fuchsia 400
]);

export function getTrackColor(trackId: number, customColor?: string): string {
  if (customColor) return customColor;
  return TRACK_PALETTE[Math.abs(trackId) % TRACK_PALETTE.length];
}

// Derive a distinctive color from faceset name
export function getFacesetColor(facesetName?: string | null, fallback = '#38bdf8'): string {
  if (!facesetName) return fallback;
  let hash = 0;
  for (let i = 0; i < facesetName.length; i++) {
    hash = (hash << 5) - hash + facesetName.charCodeAt(i);
    hash |= 0;
  }
  return TRACK_PALETTE[Math.abs(hash) % TRACK_PALETTE.length];
}

// ── Component Implementation ──────────────────────────────────────────────────

export const NleTimeline: React.FC<NleTimelineProps> = ({
  frame: frameProp,
  currentFrame: currentFrameProp,
  currentTime,
  maxFrames: rawMaxFrames,
  fps = 30,
  isPlaying = false,
  targetIndex = null,
  mediaId = null,
  targetKey = '',
  ptsTable,
  ptsFrameMap: customPtsFrameMap,
  trackLanes = [],
  warningTicks = [],
  suggestedAssignments = [],
  rangeLockIntervals = [],
  selectedTrackId = null,
  selectedRangeId = null,
  trimIn: trimInProp,
  trimOut: trimOutProp,
  markers: markersProp = [],
  keyframes: keyframesProp = [],
  activeKeyframeParam: activeParamProp = 'Swap Opacity',
  snapEnabled: initialSnapEnabled = true,
  onSeek,
  onSelectTrack,
  onSelectRange,
  onUpdateRangeLockIntervals,
  onToggleTrackActive,
  onUpdateTrim,
  onSetTrimIn,
  onSetTrimOut,
  onAddMarker,
  onDeleteMarker,
  onUpdateMarker,
  onUpdateKeyframes,
  onAssignFsz,
  onPreScanClick,
  isPreScanActive = false,
  isRangeLockActive = false,
  isRenderActive = false,
  showWarningTicks = true,
  showSuggestedFsz = true,
  showFilmstripLane = true,
  showWaveformLane = true,
  showKeyframeLane = true,
  className = '',
  style,
}) => {
  const maxFrames = Math.max(1, Math.round(rawMaxFrames));

  // Initialize or memoize PTS Frame Map
  const frameMap = useMemo(() => {
    if (customPtsFrameMap instanceof PtsFrameMap) {
      return customPtsFrameMap;
    }
    if (ptsTable) {
      return createPtsFrameMap(ptsTable, fps);
    }
    return createPtsFrameMap(null, fps);
  }, [customPtsFrameMap, ptsTable, fps]);

  // Resolve current active integer frame index
  const activeFrame = useMemo(() => {
    if (typeof frameProp === 'number' && Number.isFinite(frameProp)) {
      return clampFrameIdx(frameProp, 0, maxFrames);
    }
    if (typeof currentFrameProp === 'number' && Number.isFinite(currentFrameProp)) {
      return clampFrameIdx(currentFrameProp, 0, maxFrames);
    }
    if (typeof currentTime === 'number' && Number.isFinite(currentTime)) {
      return frameMap.resolveFrameFromTime(currentTime);
    }
    return 0;
  }, [frameProp, currentFrameProp, currentTime, maxFrames, frameMap]);

  // Trim In / Out points (exact integer frames)
  const trimIn = useMemo(() => {
    const raw = typeof trimInProp === 'number' ? trimInProp : 0;
    return clampFrameIdx(raw, 0, maxFrames);
  }, [trimInProp, maxFrames]);

  const trimOut = useMemo(() => {
    const raw = typeof trimOutProp === 'number' ? trimOutProp : maxFrames - 1;
    return clampFrameIdx(raw, trimIn, maxFrames);
  }, [trimOutProp, trimIn, maxFrames]);

  // Chapter markers state (internal fallback if not controlled)
  const [internalMarkers, setInternalMarkers] = useState<ChapterMarker[]>([]);
  const markers = useMemo(() => {
    return markersProp.length > 0 ? markersProp : internalMarkers;
  }, [markersProp, internalMarkers]);

  // Active Keyframe Parameter Selection
  const [activeParam, setActiveParam] = useState<TimelineParamType>(activeParamProp);
  useEffect(() => {
    if (activeParamProp) {
      setActiveParam(activeParamProp);
    }
  }, [activeParamProp]);

  // Normalized parameter keyframes
  const nleKeyframes = useMemo<NleKeyframe[]>(() => {
    if (!keyframesProp || keyframesProp.length === 0) return [];
    return keyframesProp.map((k: any) => {
      if ('param' in k && 'curve' in k) {
        return k as NleKeyframe;
      }
      // Convert legacy KeyframePoint
      return {
        id: k.id || `kf_${k.frame_idx}`,
        frame_idx: clampFrameIdx(k.frame_idx, 0, maxFrames),
        param: (k.metadata?.param as TimelineParamType) || 'Swap Opacity',
        value: typeof k.value === 'number' ? k.value : 1.0,
        curve: (k.curve === 'hold' ? 'Hold' : k.curve === 'linear' ? 'Linear' : 'Ease In/Out') as TimelineCurveType,
        selected: false,
      };
    });
  }, [keyframesProp, maxFrames]);

  const [selectedKeyframeId, setSelectedKeyframeId] = useState<string | null>(null);

  // Snapping enabled state
  const [snapEnabled, setSnapEnabled] = useState(initialSnapEnabled);
  const [snapIndicatorFrame, setSnapIndicatorFrame] = useState<number | null>(null);

  // ── Viewport Window & Zoom State ─────────────────────────────────────────────
  // view: [viewStart, viewEnd] in integer frame numbers [0, maxFrames - 1]
  const [viewRange, setViewRange] = useState<[number, number]>([0, maxFrames - 1]);

  // Adjust view range when maxFrames changes significantly
  useEffect(() => {
    setViewRange((prev) => {
      const start = clampFrameIdx(prev[0], 0, maxFrames);
      const end = clampFrameIdx(prev[1], start + 1, maxFrames);
      return [start, end];
    });
  }, [maxFrames]);

  const [viewStart, viewEnd] = viewRange;
  const viewSpan = Math.max(1, viewEnd - viewStart);

  // Runway DOM Container Ref & Track Width
  const runwayRef = useRef<HTMLDivElement>(null);
  const [runwayWidth, setRunwayWidth] = useState(1000);

  useEffect(() => {
    if (!runwayRef.current) return;
    const updateWidth = () => {
      if (runwayRef.current) {
        setRunwayWidth(Math.max(200, runwayRef.current.clientWidth));
      }
    };
    updateWidth();
    const ro = new ResizeObserver(updateWidth);
    ro.observe(runwayRef.current);
    return () => ro.disconnect();
  }, []);

  // ── Audio Waveform & Video Filmstrip Hooks ──────────────────────────────────
  const waveform = useAudioWaveform({
    targetIndex,
    mediaId,
    targetKey,
    buckets: Math.min(2000, Math.max(200, Math.round(runwayWidth * 1.5))),
    enabled: showWaveformLane,
  });

  const filmstrip = useFilmstripThumbnails({
    targetIndex,
    targetKey,
    viewStart,
    viewEnd,
    maxFrames,
    trackWidth: runwayWidth,
    enabled: showFilmstripLane,
  });

  // ── Coordinate Conversions ───────────────────────────────────────────────────
  const frameToFraction = useCallback(
    (frameIdx: number) => {
      const clamped = clampFrameIdx(frameIdx, 0, maxFrames);
      if (viewSpan <= 0) return 0;
      return (clamped - viewStart) / viewSpan;
    },
    [viewStart, viewSpan, maxFrames]
  );

  // Cached drag rect for zero-stall scrubbing
  const dragRectRef = useRef<{ left: number; width: number } | null>(null);

  const xToFrame = useCallback(
    (clientX: number, cachedRect?: { left: number; width: number } | null) => {
      const rect = cachedRect || dragRectRef.current || runwayRef.current?.getBoundingClientRect();
      if (!rect || rect.width <= 0) return viewStart;
      const x = Math.max(0, Math.min(rect.width, clientX - rect.left));
      const ratio = x / rect.width;
      const rawFrame = viewStart + ratio * viewSpan;
      return clampFrameIdx(rawFrame, 0, maxFrames);
    },
    [viewStart, viewSpan, maxFrames]
  );

  // Snap Targets calculation
  const snapTargets = useMemo<SnapTarget[]>(() => {
    if (!snapEnabled) return [];
    const targets: SnapTarget[] = [
      { frame: activeFrame, type: 'playhead', label: 'Playhead' },
      { frame: trimIn, type: 'trim_in', label: 'Trim In' },
      { frame: trimOut, type: 'trim_out', label: 'Trim Out' },
    ];

    markers.forEach((m) => {
      targets.push({ frame: m.frame, type: 'marker', label: m.label });
    });

    nleKeyframes
      .filter((k) => k.param === activeParam)
      .forEach((k) => {
        targets.push({ frame: k.frame_idx, type: 'keyframe', label: `${k.param} Keyframe` });
      });

    trackLanes.forEach((t) => {
      t.ranges.forEach(([start, end]) => {
        targets.push({ frame: start, type: 'range_edge', label: `${t.label} Start` });
        targets.push({ frame: end, type: 'range_edge', label: `${t.label} End` });
      });
    });

    return targets;
  }, [snapEnabled, activeFrame, trimIn, trimOut, markers, nleKeyframes, activeParam, trackLanes]);

  // ── Seek & Navigation (Coalesced & Zero-Stall) ────────────────────────────────
  const rafSeekRef = useRef<number | null>(null);
  const pendingSeekFrameRef = useRef<number | null>(null);

  const handleSeekFrame = useCallback(
    (targetFrame: number, immediate = true) => {
      const exactFrame = clampFrameIdx(targetFrame, 0, maxFrames);

      if (immediate) {
        if (rafSeekRef.current !== null) {
          cancelAnimationFrame(rafSeekRef.current);
          rafSeekRef.current = null;
        }
        const timeSec = frameMap.resolveTimeFromFrame(exactFrame);
        onSeek?.(exactFrame, timeSec);
      } else {
        pendingSeekFrameRef.current = exactFrame;
        if (rafSeekRef.current === null) {
          rafSeekRef.current = requestAnimationFrame(() => {
            rafSeekRef.current = null;
            if (pendingSeekFrameRef.current !== null) {
              const f = pendingSeekFrameRef.current;
              pendingSeekFrameRef.current = null;
              const timeSec = frameMap.resolveTimeFromFrame(f);
              onSeek?.(f, timeSec);
            }
          });
        }
      }
    },
    [maxFrames, frameMap, onSeek]
  );

  // Scrubber Dragging with Pointer Capture & Magnetic Snapping
  const [isScrubbing, setIsScrubbing] = useState(false);

  const handleScrubberMouseDown = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsScrubbing(true);

      const rect = runwayRef.current?.getBoundingClientRect();
      if (rect) {
        dragRectRef.current = { left: rect.left, width: rect.width };
      }

      const rawFrame = xToFrame(e.clientX, dragRectRef.current);
      const snapRes = snapToFrame(rawFrame, snapTargets, Math.max(2, Math.round(viewSpan / 200)));

      if (snapRes.snapped) {
        setSnapIndicatorFrame(snapRes.frame);
      } else {
        setSnapIndicatorFrame(null);
      }

      handleSeekFrame(snapRes.frame, true);

      const onMouseMove = (ev: globalThis.MouseEvent) => {
        const nextRaw = xToFrame(ev.clientX, dragRectRef.current);
        const sRes = snapToFrame(nextRaw, snapTargets, Math.max(2, Math.round(viewSpan / 200)));

        if (sRes.snapped) {
          setSnapIndicatorFrame(sRes.frame);
        } else {
          setSnapIndicatorFrame(null);
        }

        handleSeekFrame(sRes.frame, false);
      };

      const onMouseUp = () => {
        setIsScrubbing(false);
        setSnapIndicatorFrame(null);
        dragRectRef.current = null;
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
      };

      window.addEventListener('mousemove', onMouseMove);
      window.addEventListener('mouseup', onMouseUp);
    },
    [xToFrame, snapTargets, viewSpan, handleSeekFrame]
  );

  // ── Zoom-to-Cursor & Pan ────────────────────────────────────────────────────
  const handleZoom = useCallback(
    (factor: number, anchorFraction = 0.5) => {
      setViewRange((prev) => calculateZoomToCursor(prev, factor, anchorFraction, maxFrames, 4));
    },
    [maxFrames]
  );

  const handleResetZoom = useCallback(() => {
    setViewRange([0, maxFrames - 1]);
  }, [maxFrames]);

  const handleWheel = useCallback(
    (e: WheelEvent<HTMLDivElement>) => {
      const rect = runwayRef.current?.getBoundingClientRect();
      const cursorX = rect ? e.clientX - rect.left : 0;
      const fraction = rect && rect.width > 0 ? Math.max(0, Math.min(1, cursorX / rect.width)) : 0.5;

      if (e.ctrlKey || e.metaKey) {
        // Zoom to exact cursor position
        e.preventDefault();
        const factor = e.deltaY > 0 ? 1.25 : 0.8;
        handleZoom(factor, fraction);
      } else if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        // Horizontal Pan
        e.preventDefault();
        const deltaFrames = Math.round((e.deltaX || e.deltaY) * (viewSpan / 1000));
        setViewRange(([s, end]) => {
          const span = end - s;
          let newS = s + deltaFrames;
          let newE = newS + span;
          if (newS < 0) {
            newS = 0;
            newE = Math.min(maxFrames - 1, span);
          } else if (newE >= maxFrames) {
            newE = maxFrames - 1;
            newS = Math.max(0, newE - span);
          }
          return [clampFrameIdx(newS, 0, maxFrames), clampFrameIdx(newE, 0, maxFrames)];
        });
      }
    },
    [handleZoom, viewSpan, maxFrames]
  );

  // ── In/Out Trim Handles Dragging ─────────────────────────────────────────────
  const handleGlobalTrimDrag = useCallback(
    (handle: 'in' | 'out', e: MouseEvent<HTMLDivElement>) => {
      e.stopPropagation();
      e.preventDefault();

      const rect = runwayRef.current?.getBoundingClientRect();
      if (rect) {
        dragRectRef.current = { left: rect.left, width: rect.width };
      }

      const onMouseMove = (ev: globalThis.MouseEvent) => {
        const rawF = xToFrame(ev.clientX, dragRectRef.current);
        const sRes = snapToFrame(rawF, snapTargets, Math.max(2, Math.round(viewSpan / 200)));
        const f = sRes.frame;

        if (handle === 'in') {
          const newIn = Math.min(f, trimOut);
          onUpdateTrim?.(clampFrameIdx(newIn, 0, maxFrames), trimOut);
          onSetTrimIn?.(clampFrameIdx(newIn, 0, maxFrames));
        } else {
          const newOut = Math.max(f, trimIn);
          onUpdateTrim?.(trimIn, clampFrameIdx(newOut, trimIn, maxFrames));
          onSetTrimOut?.(clampFrameIdx(newOut, trimIn, maxFrames));
        }
      };

      const onMouseUp = () => {
        dragRectRef.current = null;
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
      };

      window.addEventListener('mousemove', onMouseMove);
      window.addEventListener('mouseup', onMouseUp);
    },
    [xToFrame, snapTargets, viewSpan, trimIn, trimOut, maxFrames, onUpdateTrim, onSetTrimIn, onSetTrimOut]
  );

  // ── Chapter Markers ('M') ───────────────────────────────────────────────────
  const handleAddMarkerAtCurrentFrame = useCallback(() => {
    const existing = markers.find((m) => m.frame === activeFrame);
    if (existing) {
      return;
    }
    const newMarker: ChapterMarker = {
      id: `marker_${activeFrame}_${Date.now()}`,
      frame: activeFrame,
      label: `Marker #${markers.length + 1} (${frameToTimecode(activeFrame, fps)})`,
      color: '#38bdf8',
      createdAt: Date.now(),
    };

    if (onAddMarker) {
      onAddMarker(newMarker);
    } else {
      setInternalMarkers((prev) => [...prev, newMarker]);
    }
  }, [markers, activeFrame, fps, onAddMarker]);

  const handleDeleteMarker = useCallback(
    (markerId: string) => {
      if (onDeleteMarker) {
        onDeleteMarker(markerId);
      } else {
        setInternalMarkers((prev) => prev.filter((m) => m.id !== markerId));
      }
    },
    [onDeleteMarker]
  );

  // ── Keyboard Shortcuts ('I', 'O', 'M', 'Delete') ────────────────────────────
  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>) => {
      // Don't intercept if typing in an input
      const target = e.target as HTMLElement;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) {
        return;
      }

      if (e.key === 'i' || e.key === 'I') {
        e.preventDefault();
        onSetTrimIn?.(activeFrame);
        onUpdateTrim?.(activeFrame, Math.max(activeFrame, trimOut));
      } else if (e.key === 'o' || e.key === 'O') {
        e.preventDefault();
        onSetTrimOut?.(activeFrame);
        onUpdateTrim?.(Math.min(trimIn, activeFrame), activeFrame);
      } else if (e.key === 'm' || e.key === 'M') {
        e.preventDefault();
        handleAddMarkerAtCurrentFrame();
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedKeyframeId) {
          e.preventDefault();
          const nextKfs = nleKeyframes.filter((k) => k.id !== selectedKeyframeId);
          onUpdateKeyframes?.(nextKfs);
          setSelectedKeyframeId(null);
        }
      }
    },
    [
      activeFrame,
      trimIn,
      trimOut,
      selectedKeyframeId,
      nleKeyframes,
      onSetTrimIn,
      onSetTrimOut,
      onUpdateTrim,
      handleAddMarkerAtCurrentFrame,
      onUpdateKeyframes,
    ]
  );

  // ── Track & Range Lock Interactivity ─────────────────────────────────────────
  const handleTrackOrRangeClick = useCallback(
    (track: ByteTrackLane, range?: [number, number], e?: MouseEvent) => {
      if (e) {
        e.stopPropagation();
      }

      onSelectTrack?.(track.trackId);

      const targetStart = range ? range[0] : track.firstFrame;
      const targetEnd = range ? range[1] : track.lastFrame;

      const existingIdx = rangeLockIntervals.findIndex(
        (iv) =>
          iv.track_id === track.trackId &&
          Math.max(iv.frame_start, targetStart) <= Math.min(iv.frame_end, targetEnd)
      );

      let updated: RangeLockInterval[];

      if (existingIdx >= 0) {
        const existing = rangeLockIntervals[existingIdx];
        const toggled: RangeLockInterval = {
          ...existing,
          locked: existing.locked !== false ? false : true,
        };
        updated = [
          ...rangeLockIntervals.slice(0, existingIdx),
          toggled,
          ...rangeLockIntervals.slice(existingIdx + 1),
        ];
        onSelectRange?.(toggled);
      } else {
        const newInterval = createRangeLockInterval(
          {
            id: `rl_${track.trackId}_${targetStart}_${targetEnd}_${Date.now()}`,
            track_id: track.trackId,
            frame_start: clampFrameIdx(targetStart, 0, maxFrames),
            frame_end: clampFrameIdx(targetEnd, targetStart, maxFrames),
            label: `${track.label} [${targetStart}–${targetEnd}]`,
            faceset_id: track.facesetId,
            source_index: track.sourceIndex,
            color: getTrackColor(track.trackId, track.color),
            locked: true,
          },
          maxFrames
        );
        updated = [...rangeLockIntervals, newInterval];
        onSelectRange?.(newInterval);
      }

      onUpdateRangeLockIntervals?.(updated);
    },
    [rangeLockIntervals, maxFrames, onSelectTrack, onSelectRange, onUpdateRangeLockIntervals]
  );

  // Range Trim Dragging
  const handleRangeTrimDrag = useCallback(
    (intervalId: string, handle: 'start' | 'end', e: MouseEvent<HTMLDivElement>) => {
      e.stopPropagation();
      e.preventDefault();

      const rect = runwayRef.current?.getBoundingClientRect();
      if (rect) {
        dragRectRef.current = { left: rect.left, width: rect.width };
      }

      const onMouseMove = (ev: globalThis.MouseEvent) => {
        const currentFrame = xToFrame(ev.clientX, dragRectRef.current);
        const nextIntervals = rangeLockIntervals.map((item) => {
          if (item.id !== intervalId) return item;
          if (handle === 'start') {
            const newStart = Math.min(currentFrame, item.frame_end);
            return clampRangeLockInterval({ ...item, frame_start: newStart }, 0, maxFrames);
          } else {
            const newEnd = Math.max(currentFrame, item.frame_start);
            return clampRangeLockInterval({ ...item, frame_end: newEnd }, 0, maxFrames);
          }
        });
        onUpdateRangeLockIntervals?.(nextIntervals);
      };

      const onMouseUp = () => {
        dragRectRef.current = null;
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
      };

      window.addEventListener('mousemove', onMouseMove);
      window.addEventListener('mouseup', onMouseUp);
    },
    [rangeLockIntervals, xToFrame, maxFrames, onUpdateRangeLockIntervals]
  );

  // ── Keyframe Lane Operations (Add, Drag, Copy, Delete, Curves) ───────────────
  const handleAddKeyframeAtCurrent = useCallback(() => {
    const val = evaluateKeyframeValue(nleKeyframes, activeParam, activeFrame);
    const newKf = createNleKeyframe(activeParam, activeFrame, val, 'Ease In/Out', maxFrames);
    const updated = sortAndNormalizeKeyframes([...nleKeyframes, newKf], maxFrames);
    onUpdateKeyframes?.(updated);
    setSelectedKeyframeId(newKf.id);
  }, [activeParam, activeFrame, nleKeyframes, maxFrames, onUpdateKeyframes]);

  const handleKeyframeDrag = useCallback(
    (kfId: string, e: MouseEvent<HTMLDivElement>) => {
      e.stopPropagation();
      e.preventDefault();

      const isCopy = e.altKey; // Alt + Drag = Copy
      setSelectedKeyframeId(kfId);

      const targetKf = nleKeyframes.find((k) => k.id === kfId);
      if (!targetKf) return;

      const rect = runwayRef.current?.getBoundingClientRect();
      if (rect) {
        dragRectRef.current = { left: rect.left, width: rect.width };
      }

      const onMouseMove = (ev: globalThis.MouseEvent) => {
        const rawF = xToFrame(ev.clientX, dragRectRef.current);
        const sRes = snapToFrame(rawF, snapTargets, Math.max(2, Math.round(viewSpan / 200)));
        const f = sRes.frame;

        if (isCopy) {
          // Keep original, create duplicate
          const dup = duplicateKeyframe(targetKf, f - targetKf.frame_idx, maxFrames);
          const updated = sortAndNormalizeKeyframes([...nleKeyframes, dup], maxFrames);
          onUpdateKeyframes?.(updated);
        } else {
          const updatedKfs = nleKeyframes.map((k) =>
            k.id === kfId ? { ...k, frame_idx: clampFrameIdx(f, 0, maxFrames) } : k
          );
          onUpdateKeyframes?.(sortAndNormalizeKeyframes(updatedKfs, maxFrames));
        }
      };

      const onMouseUp = () => {
        dragRectRef.current = null;
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
      };

      window.addEventListener('mousemove', onMouseMove);
      window.addEventListener('mouseup', onMouseUp);
    },
    [nleKeyframes, snapTargets, viewSpan, xToFrame, maxFrames, onUpdateKeyframes]
  );

  const handleCurveChange = useCallback(
    (kfId: string, newCurve: TimelineCurveType) => {
      const updated = nleKeyframes.map((k) => (k.id === kfId ? { ...k, curve: newCurve } : k));
      onUpdateKeyframes?.(updated);
    },
    [nleKeyframes, onUpdateKeyframes]
  );

  const handleDeleteKeyframe = useCallback(
    (kfId: string) => {
      const updated = nleKeyframes.filter((k) => k.id !== kfId);
      onUpdateKeyframes?.(updated);
      if (selectedKeyframeId === kfId) {
        setSelectedKeyframeId(null);
      }
    },
    [nleKeyframes, selectedKeyframeId, onUpdateKeyframes]
  );

  // ── Minimap Panning ─────────────────────────────────────────────────────────
  const minimapRef = useRef<HTMLDivElement>(null);

  const handleMinimapMouseDown = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      e.preventDefault();
      if (!minimapRef.current) return;
      const rect = minimapRef.current.getBoundingClientRect();
      const clickRatio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const targetFrame = Math.round(clickRatio * (maxFrames - 1));

      const span = viewSpan;
      let newStart = Math.round(targetFrame - span / 2);
      let newEnd = newStart + span;
      if (newStart < 0) {
        newStart = 0;
        newEnd = Math.min(maxFrames - 1, span);
      } else if (newEnd >= maxFrames) {
        newEnd = maxFrames - 1;
        newStart = Math.max(0, newEnd - span);
      }
      setViewRange([clampFrameIdx(newStart, 0, maxFrames), clampFrameIdx(newEnd, 0, maxFrames)]);
    },
    [maxFrames, viewSpan]
  );

  // ── Adaptive Ruler Ticks ────────────────────────────────────────────────────
  const rulerTicks = useMemo(() => {
    return generateRulerTicks(viewStart, viewEnd, fps);
  }, [viewStart, viewEnd, fps]);

  // Suggested assignment map
  const assignmentMap = useMemo(() => {
    const map = new Map<number, SuggestedFszAssignment>();
    suggestedAssignments.forEach((item) => {
      map.set(item.trackId, item);
    });
    return map;
  }, [suggestedAssignments]);

  // Warning ticks by track
  const warningsByTrack = useMemo(() => {
    const map = new Map<number, WarningTick[]>();
    warningTicks.forEach((tick) => {
      const list = map.get(tick.trackId) || [];
      list.push(tick);
      map.set(tick.trackId, list);
    });
    return map;
  }, [warningTicks]);

  // Filter keyframes for active parameter
  const activeParamKeyframes = useMemo(() => {
    return nleKeyframes.filter((k) => k.param === activeParam);
  }, [nleKeyframes, activeParam]);

  const selectedKeyframe = useMemo(() => {
    return nleKeyframes.find((k) => k.id === selectedKeyframeId);
  }, [nleKeyframes, selectedKeyframeId]);

  // Active playhead time string
  const activeTimecode = useMemo(() => frameToTimecode(activeFrame, fps), [activeFrame, fps]);
  const activeSeconds = useMemo(
    () => formatSeconds(frameMap.resolveTimeFromFrame(activeFrame)),
    [activeFrame, frameMap]
  );

  const lockedIntervalsCount = useMemo(
    () => rangeLockIntervals.filter((iv) => iv.locked !== false).length,
    [rangeLockIntervals]
  );

  // Hover loupe state for filmstrip
  const [hoverFrame, setHoverFrame] = useState<number | null>(null);
  const [hoverClientX, setHoverClientX] = useState<number | null>(null);

  const handleRunwayMouseMove = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      if (isScrubbing) return;
      const rect = runwayRef.current?.getBoundingClientRect();
      if (!rect) return;
      const f = xToFrame(e.clientX, { left: rect.left, width: rect.width });
      setHoverFrame(f);
      setHoverClientX(e.clientX - rect.left);
    },
    [isScrubbing, xToFrame]
  );

  const handleRunwayMouseLeave = useCallback(() => {
    setHoverFrame(null);
    setHoverClientX(null);
  }, []);

  return (
    <div
      tabIndex={0}
      onKeyDown={handleKeyDown}
      className={`nle-timeline-root flex flex-col w-full bg-[var(--surface-1,#12141a)] border border-[var(--border-color,rgba(255,255,255,0.08))] rounded-xl overflow-hidden shadow-2xl select-none font-sans text-xs outline-none focus:ring-1 focus:ring-[var(--accent,#e94560)]/40 ${className}`}
      style={style}
      onWheel={handleWheel}
      data-testid="nle-timeline"
    >
      {/* ── Top Header Toolbar ──────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-2 bg-[var(--surface-2,#1a1d26)] border-b border-[var(--border-color,rgba(255,255,255,0.08))]">
        {/* Step Status Badges & Title */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 font-bold tracking-wide text-white text-sm">
            <svg
              className="w-4 h-4 text-[var(--accent,#e94560)]"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <rect x="2" y="4" width="20" height="16" rx="2" />
              <path d="M10 4v16" />
              <path d="M14 4v16" />
            </svg>
            <span>NLE Timeline</span>
          </div>

          {/* Step 1 Pre-Scan Badge */}
          <button
            type="button"
            onClick={onPreScanClick}
            disabled={isPreScanActive}
            title="Step 1: Run or inspect Pre-Scan ByteTrack spans"
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-mini font-medium border transition-colors ${
              trackLanes.length > 0
                ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
                : isPreScanActive
                ? 'border-sky-500/50 bg-sky-500/10 text-sky-300 animate-pulse'
                : 'border-white/10 bg-white/5 text-neutral-400 hover:text-white'
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                trackLanes.length > 0
                  ? 'bg-emerald-400'
                  : isPreScanActive
                  ? 'bg-sky-400 animate-ping'
                  : 'bg-neutral-500'
              }`}
            />
            <span>1 Pre-Scan</span>
            <span className="font-mono text-micro opacity-80">
              ({trackLanes.length} {trackLanes.length === 1 ? 'Track' : 'Tracks'})
            </span>
          </button>

          {/* Step 2 Range Lock Badge */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-mini font-medium border ${
              lockedIntervalsCount > 0
                ? 'border-[var(--accent,#e94560)]/50 bg-[var(--accent,#e94560)]/10 text-rose-300'
                : 'border-white/10 bg-white/5 text-neutral-400'
            }`}
            title="Step 2: Active Range Lock Intervals"
          >
            <span
              className={`w-2 h-2 rounded-full ${
                lockedIntervalsCount > 0 ? 'bg-[var(--accent,#e94560)]' : 'bg-neutral-500'
              }`}
            />
            <span>2 Range Lock</span>
            <span className="font-mono text-micro opacity-80">
              ({lockedIntervalsCount} {lockedIntervalsCount === 1 ? 'Interval' : 'Intervals'})
            </span>
          </div>

          {/* Snap Magnet Toggle */}
          <button
            type="button"
            onClick={() => setSnapEnabled((prev) => !prev)}
            className={`flex items-center gap-1 px-2 py-0.5 rounded border text-mini transition ${
              snapEnabled
                ? 'border-sky-500/50 bg-sky-500/15 text-sky-300'
                : 'border-white/10 bg-white/5 text-neutral-400 hover:text-white'
            }`}
            title={snapEnabled ? 'Magnetic Snapping ON (Playhead, In/Out, Markers, KF)' : 'Magnetic Snapping OFF'}
          >
            <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 14v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4" />
              <polyline points="7 9 12 4 17 9" />
              <line x1="12" y1="4" x2="12" y2="16" />
            </svg>
            <span>Snap</span>
          </button>
        </div>

        {/* Timecode & Frame Readout */}
        <div className="flex items-center gap-3">
          {/* Exact Timecode and Seconds Display */}
          <div className="flex items-center gap-2 px-3 py-1 bg-black/40 border border-white/10 rounded-lg font-mono">
            <span className="text-micro uppercase text-neutral-400 font-semibold tracking-wider">
              TC:
            </span>
            <span className="text-sm font-bold text-white tracking-wider tabular-nums">
              {activeTimecode}
            </span>
            <span className="text-neutral-500">|</span>
            <span className="text-xs text-neutral-300 tabular-nums">
              {activeSeconds}
            </span>
            <span className="text-neutral-500">|</span>
            <span className="text-xs text-[var(--accent,#e94560)] font-semibold tabular-nums">
              #{activeFrame}
              <span className="text-neutral-500 font-normal"> / {maxFrames - 1}</span>
            </span>
          </div>

          {/* In / Out Chips */}
          <div className="hidden sm:flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => handleSeekFrame(trimIn)}
              className="flex items-center gap-1 px-2 py-0.5 bg-white/5 border border-white/10 hover:border-amber-400/40 rounded text-mini font-mono text-neutral-300 transition"
              title="Click to jump to Trim In point ('I' sets at playhead)"
            >
              <span className="text-amber-400 font-bold">IN:</span>
              <span className="tabular-nums">#{trimIn}</span>
            </button>
            <button
              type="button"
              onClick={() => handleSeekFrame(trimOut)}
              className="flex items-center gap-1 px-2 py-0.5 bg-white/5 border border-white/10 hover:border-amber-400/40 rounded text-mini font-mono text-neutral-300 transition"
              title="Click to jump to Trim Out point ('O' sets at playhead)"
            >
              <span className="text-amber-400 font-bold">OUT:</span>
              <span className="tabular-nums">#{trimOut}</span>
            </button>
          </div>

          {/* Chapter Marker Button ('M') */}
          <button
            type="button"
            onClick={handleAddMarkerAtCurrentFrame}
            className="flex items-center gap-1 px-2 py-1 bg-white/5 border border-white/10 hover:border-sky-400/50 rounded text-mini text-neutral-300 hover:text-white transition"
            title="Add Chapter Marker at playhead ('M')"
          >
            <svg className="w-3.5 h-3.5 text-sky-400" viewBox="0 0 24 24" fill="currentColor">
              <path d="M17 3H7c-1.1 0-1.99.9-1.99 2L5 21l7-3 7 3V5c0-1.1-.9-2-2-2z" />
            </svg>
            <span>+ Marker</span>
            <span className="font-mono text-micro text-neutral-500">({markers.length})</span>
          </button>

          {/* Zoom Controls */}
          <div className="flex items-center gap-1 bg-black/30 border border-white/10 rounded-lg p-0.5">
            <button
              type="button"
              onClick={() => handleZoom(0.7, 0.5)}
              className="p-1 hover:bg-white/10 rounded text-neutral-300 hover:text-white transition"
              title="Zoom In (Ctrl + Wheel Up)"
              aria-label="Zoom In"
            >
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
                <line x1="11" y1="8" x2="11" y2="14" />
                <line x1="8" y1="11" x2="14" y2="11" />
              </svg>
            </button>
            <button
              type="button"
              onClick={() => handleZoom(1.4, 0.5)}
              className="p-1 hover:bg-white/10 rounded text-neutral-300 hover:text-white transition"
              title="Zoom Out (Ctrl + Wheel Down)"
              aria-label="Zoom Out"
            >
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
                <line x1="8" y1="11" x2="14" y2="11" />
              </svg>
            </button>
            <button
              type="button"
              onClick={handleResetZoom}
              className="px-2 py-0.5 text-micro font-medium hover:bg-white/10 rounded text-neutral-300 hover:text-white transition"
              title="Fit entire clip to window"
            >
              Fit
            </button>
          </div>
        </div>
      </div>

      {/* ── Minimap Overview Strip ──────────────────────────────────────────── */}
      <div
        ref={minimapRef}
        onClick={handleMinimapMouseDown}
        className="relative w-full h-4 bg-black/60 border-b border-white/10 cursor-pointer overflow-hidden"
        title="Full Clip Minimap (click to center view)"
      >
        {/* Track presence lines in minimap */}
        {trackLanes.map((track) => {
          const color = getTrackColor(track.trackId, track.color);
          return (
            <React.Fragment key={`mini_t_${track.trackId}`}>
              {track.ranges.map(([start, end], rIdx) => {
                const leftPct = (clampFrameIdx(start, 0, maxFrames) / maxFrames) * 100;
                const widthPct =
                  ((clampFrameIdx(end, start, maxFrames) - clampFrameIdx(start, 0, maxFrames)) /
                    maxFrames) *
                  100;
                return (
                  <div
                    key={`mini_r_${track.trackId}_${rIdx}`}
                    className="absolute top-0 bottom-0 opacity-40"
                    style={{
                      left: `${leftPct}%`,
                      width: `${Math.max(0.2, widthPct)}%`,
                      backgroundColor: color,
                    }}
                  />
                );
              })}
            </React.Fragment>
          );
        })}

        {/* Chapter markers in minimap */}
        {markers.map((m) => (
          <div
            key={`mini_m_${m.id}`}
            className="absolute top-0 bottom-0 w-1 bg-sky-400 z-10 opacity-70"
            style={{ left: `${(m.frame / maxFrames) * 100}%` }}
          />
        ))}

        {/* Shaded View Window Highlight in Minimap */}
        {maxFrames > 1 && (
          <div
            className="absolute top-0 bottom-0 border-x-2 border-white/70 bg-white/15 pointer-events-none transition-all duration-75"
            style={{
              left: `${(viewStart / maxFrames) * 100}%`,
              width: `${(viewSpan / maxFrames) * 100}%`,
            }}
          />
        )}

        {/* Minimap Playhead Indicator */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-[var(--accent,#e94560)] shadow-sm pointer-events-none"
          style={{ left: `${(activeFrame / maxFrames) * 100}%` }}
        />
      </div>

      {/* ── Main Runway Container (Ruler + Lanes Stack) ────────────────────── */}
      <div
        ref={runwayRef}
        onMouseMove={handleRunwayMouseMove}
        onMouseLeave={handleRunwayMouseLeave}
        className="relative flex flex-col w-full overflow-hidden bg-black/20"
      >
        {/* Global Warning Bar (if warning ticks exist) */}
        {showWarningTicks && warningTicks.length > 0 && (
          <div className="relative w-full h-5 bg-black/40 border-b border-white/5 flex items-center px-2">
            <span className="text-micro text-amber-400 font-semibold uppercase tracking-wider mr-2 shrink-0">
              Warnings ({warningTicks.length}):
            </span>
            <div className="relative flex-1 h-full overflow-hidden">
              {warningTicks.map((tick, idx) => {
                const tickX = frameToFraction(tick.frame) * 100;
                if (tickX < 0 || tickX > 100) return null;
                const isOcclusion = tick.type === 'occlusion' || (tick.value && tick.value > 25);
                const isPose = tick.type === 'pose' || (tick.value && tick.value > 50);
                const tickColor = isOcclusion ? '#f43f5e' : isPose ? '#f59e0b' : '#38bdf8';

                return (
                  <button
                    key={`warn_${tick.trackId}_${tick.frame}_${idx}`}
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleSeekFrame(tick.frame);
                      onSelectTrack?.(tick.trackId);
                    }}
                    title={`[Frame #${tick.frame}] ${tick.message}`}
                    className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2 h-3.5 rounded-sm hover:scale-125 transition-transform cursor-pointer"
                    style={{
                      left: `${tickX}%`,
                      backgroundColor: tickColor,
                    }}
                  />
                );
              })}
            </div>
          </div>
        )}

        {/* ── Ruler & Timecode Lane ─────────────────────────────────────────── */}
        <div className="flex w-full h-7 bg-[var(--surface-2,#1a1d26)] border-b border-white/10">
          {/* Header spacer column */}
          <div className="w-56 shrink-0 flex items-center justify-between px-3 border-r border-white/10 text-mini font-semibold text-neutral-400 uppercase tracking-wider bg-[var(--surface-2,#1a1d26)] z-10">
            <span>Ruler</span>
            <span className="text-micro text-neutral-500 font-mono">
              fps: {fps}
            </span>
          </div>

          {/* Time/Frame Ruler Track */}
          <div
            onClick={handleScrubberMouseDown}
            className="relative flex-1 h-full cursor-pointer overflow-hidden"
          >
            {/* Tick lines and labels */}
            {rulerTicks.map((tick) => {
              const leftPct = frameToFraction(tick.frame) * 100;
              if (leftPct < 0 || leftPct > 100) return null;
              return (
                <div
                  key={`tick_${tick.frame}`}
                  className="absolute top-0 bottom-0 pointer-events-none select-none flex flex-col justify-end pb-0.5"
                  style={{ left: `${leftPct}%` }}
                >
                  <div
                    className={`w-px ${
                      tick.major ? 'h-3 bg-white/40' : 'h-1.5 bg-white/20'
                    }`}
                  />
                  {tick.major && (
                    <span className="text-nano font-mono text-neutral-400 -translate-x-1/2 select-none">
                      {tick.label}
                    </span>
                  )}
                </div>
              );
            })}

            {/* Chapter Markers ('M') on Ruler */}
            {markers.map((m) => {
              const leftPct = frameToFraction(m.frame) * 100;
              if (leftPct < 0 || leftPct > 100) return null;
              return (
                <div
                  key={`marker_${m.id}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleSeekFrame(m.frame);
                  }}
                  className="absolute top-0 w-3 -translate-x-1/2 cursor-pointer z-30 group"
                  style={{ left: `${leftPct}%` }}
                  title={`[Frame #${m.frame}] ${m.label} (Click to jump)`}
                >
                  <div className="w-3 h-4 bg-sky-400 text-black flex items-center justify-center rounded-b-sm shadow-md group-hover:scale-125 transition-transform">
                    <span className="text-nano font-bold">M</span>
                  </div>
                </div>
              );
            })}

            {/* Global Trim In Bracket Marker */}
            {trimIn >= viewStart && trimIn <= viewEnd && (
              <div
                onMouseDown={(e) => handleGlobalTrimDrag('in', e)}
                className="absolute top-0 bottom-0 w-3 -translate-x-full cursor-ew-resize z-20 flex items-center justify-center group"
                style={{ left: `${frameToFraction(trimIn) * 100}%` }}
                title={`Trim In Handle: #${trimIn} (drag to trim, or press 'I')`}
              >
                <div className="w-1.5 h-5 bg-amber-400 rounded-l group-hover:scale-110 shadow-md flex items-center justify-center text-nano font-bold text-black">
                  [
                </div>
              </div>
            )}

            {/* Global Trim Out Bracket Marker */}
            {trimOut >= viewStart && trimOut <= viewEnd && (
              <div
                onMouseDown={(e) => handleGlobalTrimDrag('out', e)}
                className="absolute top-0 bottom-0 w-3 cursor-ew-resize z-20 flex items-center justify-center group"
                style={{ left: `${frameToFraction(trimOut) * 100}%` }}
                title={`Trim Out Handle: #${trimOut} (drag to trim, or press 'O')`}
              >
                <div className="w-1.5 h-5 bg-amber-400 rounded-r group-hover:scale-110 shadow-md flex items-center justify-center text-nano font-bold text-black">
                  ]
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ── V1 Video Filmstrip Lane ────────────────────────────────────────── */}
        {showFilmstripLane && (
          <div
            onClick={handleScrubberMouseDown}
            className="flex w-full h-12 bg-black/30 border-b border-white/5 group/film relative cursor-pointer"
            data-testid="lane-v1-filmstrip"
          >
            {/* Left Header */}
            <div className="w-56 shrink-0 flex items-center justify-between px-3 border-r border-white/10 bg-[var(--surface-2,#1a1d26)]/90 z-10">
              <div className="flex items-center gap-1.5 font-bold text-white text-mini">
                <span className="px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400 text-micro font-mono">
                  V1
                </span>
                <span>Video Filmstrip</span>
              </div>
              <span className="text-micro font-mono text-neutral-500">
                {filmstrip.slots.length} stills
              </span>
            </div>

            {/* Filmstrip Runway */}
            <div className="relative flex-1 h-full overflow-hidden bg-black/40">
              {filmstrip.slots.map((slot) => {
                const bmp = filmstrip.getThumbnail(slot.frame);
                return (
                  <div
                    key={`slot_${slot.slotIndex}_${slot.frame}`}
                    className="absolute top-0 bottom-0 border-r border-black/60 overflow-hidden flex items-center justify-center bg-black/50"
                    style={{
                      left: `${slot.leftPct}%`,
                      width: `${slot.widthPct}%`,
                    }}
                  >
                    {bmp ? (
                      <canvas
                        ref={(canvas) => {
                          if (canvas && bmp) {
                            const ctx = canvas.getContext('2d');
                            if (ctx) {
                              canvas.width = canvas.clientWidth || 100;
                              canvas.height = canvas.clientHeight || 48;
                              ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
                            }
                          }
                        }}
                        className="w-full h-full object-cover opacity-80 group-hover/film:opacity-100 transition-opacity"
                      />
                    ) : (
                      <div className="flex flex-col items-center justify-center text-nano font-mono text-neutral-600">
                        <span>#{slot.frame}</span>
                      </div>
                    )}
                    {/* Sprocket border tick */}
                    <div className="absolute bottom-0 left-0 right-0 h-1 bg-black/60 flex justify-around">
                      <span className="w-1 h-0.5 bg-white/20 rounded-full" />
                      <span className="w-1 h-0.5 bg-white/20 rounded-full" />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ── A1 Audio Waveform Lane ─────────────────────────────────────────── */}
        {showWaveformLane && (
          <div
            onClick={handleScrubberMouseDown}
            className="flex w-full h-9 bg-black/20 border-b border-white/5 relative cursor-pointer"
            data-testid="lane-a1-waveform"
          >
            {/* Left Header */}
            <div className="w-56 shrink-0 flex items-center justify-between px-3 border-r border-white/10 bg-[var(--surface-2,#1a1d26)]/90 z-10">
              <div className="flex items-center gap-1.5 font-bold text-white text-mini">
                <span className="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 text-micro font-mono">
                  A1
                </span>
                <span>Audio Peaks</span>
              </div>
              <span className="text-micro font-mono text-neutral-500">
                {waveform.state === 'ready'
                  ? 'ready'
                  : waveform.state === 'none'
                  ? 'no audio'
                  : waveform.state}
              </span>
            </div>

            {/* Audio Waveform Canvas Runway */}
            <div className="relative flex-1 h-full overflow-hidden flex items-center bg-black/30">
              {waveform.state === 'ready' && waveform.peaks.length > 0 ? (
                <canvas
                  ref={(canvas) => {
                    if (!canvas) return;
                    const ctx = canvas.getContext('2d');
                    if (!ctx) return;

                    const width = canvas.clientWidth || 800;
                    const height = canvas.clientHeight || 36;
                    const dpr = window.devicePixelRatio || 1;

                    canvas.width = width * dpr;
                    canvas.height = height * dpr;
                    ctx.scale(dpr, dpr);
                    ctx.clearRect(0, 0, width, height);

                    const midY = height / 2;
                    const peaks = waveform.peaks;
                    const numPeaks = peaks.length;

                    // Visible time window
                    const tStart = frameMap.resolveTimeFromFrame(viewStart);
                    const tEnd = frameMap.resolveTimeFromFrame(viewEnd);
                    const dur = Math.max(0.001, waveform.duration || 1);

                    const p0 = Math.max(0, Math.floor((tStart / dur) * numPeaks));
                    const p1 = Math.min(numPeaks, Math.ceil((tEnd / dur) * numPeaks));
                    const sliceLen = Math.max(1, p1 - p0);

                    // Draw center baseline
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(0, midY);
                    ctx.lineTo(width, midY);
                    ctx.stroke();

                    // Draw peak bars
                    const barWidth = Math.max(1, width / sliceLen);
                    for (let i = 0; i < sliceLen; i++) {
                      const peakVal = peaks[p0 + i] || 0; // 0..255
                      const norm = peakVal / 255;
                      const h = Math.max(1, norm * (midY - 2));

                      // Gradient color: Emerald/Teal to Amber/Rose on high amplitude
                      ctx.fillStyle =
                        norm > 0.8
                          ? '#f43f5e'
                          : norm > 0.5
                          ? '#fbbf24'
                          : '#10b981';

                      const x = (i / sliceLen) * width;
                      ctx.fillRect(x, midY - h, barWidth, h * 2);
                    }
                  }}
                  className="w-full h-full"
                />
              ) : waveform.state === 'none' ? (
                <div className="flex items-center px-4 text-micro font-mono text-neutral-500 italic">
                  (No audio track detected in source media)
                </div>
              ) : (
                <div className="flex items-center px-4 text-micro font-mono text-sky-400 animate-pulse">
                  Rendering waveform envelope from /api/media/waveform…
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Keyframe Lane ('KF') ───────────────────────────────────────────── */}
        {showKeyframeLane && (
          <div
            onClick={handleScrubberMouseDown}
            className="flex w-full h-8 bg-black/25 border-b border-white/5 relative cursor-pointer"
            data-testid="lane-kf-editor"
          >
            {/* Left Header with Parameter Selector */}
            <div
              onClick={(e) => e.stopPropagation()}
              className="w-56 shrink-0 flex items-center justify-between px-3 border-r border-white/10 bg-[var(--surface-2,#1a1d26)]/90 z-10"
            >
              <div className="flex items-center gap-1.5">
                <span className="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 text-micro font-mono font-bold">
                  KF
                </span>
                <select
                  value={activeParam}
                  onChange={(e) => setActiveParam(e.target.value as TimelineParamType)}
                  className="bg-black/60 border border-white/15 rounded px-1.5 py-0.5 text-micro font-medium text-white focus:outline-none focus:border-amber-400/60"
                  title="Active Keyframe Parameter"
                >
                  <option value="Swap Opacity">Swap Opacity</option>
                  <option value="Mask Feather">Mask Feather</option>
                  <option value="Yaw Erosion">Yaw Erosion</option>
                  <option value="Color Match">Color Match</option>
                </select>
              </div>

              {/* Add Keyframe Button */}
              <button
                type="button"
                onClick={handleAddKeyframeAtCurrent}
                className="p-1 hover:bg-white/10 rounded text-amber-400 hover:text-white transition"
                title={`Add ${activeParam} keyframe at frame #${activeFrame}`}
              >
                <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="12" y1="5" x2="12" y2="19" />
                  <line x1="5" y1="12" x2="19" y2="12" />
                </svg>
              </button>
            </div>

            {/* Keyframe Runway with Connecting Curves */}
            <div className="relative flex-1 h-full overflow-hidden flex items-center bg-black/30">
              {/* Connecting curve line SVG */}
              <svg className="absolute inset-0 w-full h-full pointer-events-none">
                {activeParamKeyframes.map((kf, idx) => {
                  if (idx === activeParamKeyframes.length - 1) return null;
                  const nextKf = activeParamKeyframes[idx + 1];

                  const x1 = frameToFraction(kf.frame_idx) * runwayWidth;
                  const x2 = frameToFraction(nextKf.frame_idx) * runwayWidth;
                  const y = 16; // lane center

                  if (kf.curve === 'Hold') {
                    return (
                      <path
                        key={`line_${kf.id}_${nextKf.id}`}
                        d={`M ${x1} ${y} H ${x2}`}
                        stroke="rgba(251, 191, 36, 0.4)"
                        strokeWidth="1.5"
                        strokeDasharray="3 3"
                        fill="none"
                      />
                    );
                  }
                  if (kf.curve === 'Linear') {
                    return (
                      <line
                        key={`line_${kf.id}_${nextKf.id}`}
                        x1={x1}
                        y1={y}
                        x2={x2}
                        y2={y}
                        stroke="rgba(251, 191, 36, 0.7)"
                        strokeWidth="2"
                      />
                    );
                  }
                  // Ease In/Out smooth bezier
                  const dx = (x2 - x1) * 0.5;
                  return (
                    <path
                      key={`curve_${kf.id}_${nextKf.id}`}
                      d={`M ${x1} ${y} C ${x1 + dx} ${y - 4}, ${x2 - dx} ${y + 4}, ${x2} ${y}`}
                      stroke="rgba(251, 191, 36, 0.85)"
                      strokeWidth="2"
                      fill="none"
                    />
                  );
                })}
              </svg>

              {/* Keyframe Diamonds */}
              {activeParamKeyframes.map((kf) => {
                const kfX = frameToFraction(kf.frame_idx) * 100;
                if (kfX < -5 || kfX > 105) return null;

                const isSelected = selectedKeyframeId === kf.id;
                const paramColor = PARAMETER_CONFIGS[kf.param]?.color || '#fbbf24';

                return (
                  <div
                    key={`kf_${kf.id}`}
                    onMouseDown={(e) => handleKeyframeDrag(kf.id, e)}
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedKeyframeId(kf.id);
                      handleSeekFrame(kf.frame_idx);
                    }}
                    className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3.5 h-3.5 rotate-45 border shadow-md hover:scale-125 transition-transform cursor-ew-resize z-20 ${
                      isSelected
                        ? 'bg-white border-[var(--accent,#e94560)] ring-2 ring-[var(--accent,#e94560)]/80 scale-125'
                        : 'border-black'
                    }`}
                    style={{
                      left: `${kfX}%`,
                      backgroundColor: isSelected ? '#ffffff' : paramColor,
                    }}
                    title={`Keyframe: ${kf.param} = ${kf.value} (${kf.curve}) at frame #${kf.frame_idx} (Drag to move, Alt+Drag to copy)`}
                  />
                );
              })}

              {/* Curve Selector Toolbar if Keyframe Selected */}
              {selectedKeyframe && (
                <div
                  onClick={(e) => e.stopPropagation()}
                  className="absolute right-4 top-1/2 -translate-y-1/2 z-30 flex items-center gap-1.5 px-2 py-0.5 bg-black/80 border border-white/20 rounded shadow-lg backdrop-blur"
                >
                  <span className="text-micro font-mono text-neutral-300">
                    #{selectedKeyframe.frame_idx}:
                  </span>
                  {(['Linear', 'Ease In/Out', 'Hold'] as TimelineCurveType[]).map((c) => (
                    <button
                      key={c}
                      type="button"
                      onClick={() => handleCurveChange(selectedKeyframe.id, c)}
                      className={`px-1.5 py-0.5 rounded text-nano font-medium transition ${
                        selectedKeyframe.curve === c
                          ? 'bg-amber-400 text-black font-bold'
                          : 'bg-white/10 text-neutral-300 hover:text-white'
                      }`}
                    >
                      {c}
                    </button>
                  ))}
                  <button
                    type="button"
                    onClick={() => handleDeleteKeyframe(selectedKeyframe.id)}
                    className="p-1 hover:bg-rose-500/20 text-rose-400 rounded transition"
                    title="Delete Keyframe"
                  >
                    <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <line x1="18" y1="6" x2="6" y2="18" />
                      <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Per-Target Face Track Lanes Track #1..#N ────────────────────────── */}
        <div className="flex flex-col w-full divide-y divide-white/5">
          {trackLanes.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 px-4 text-center text-neutral-400">
              <svg
                className="w-7 h-7 mb-2 text-neutral-600"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
              >
                <path d="M15 10l-4 4l6 6l4 -16l-18 7l4 2l2 6l3 -4" />
              </svg>
              <p className="text-xs font-medium text-neutral-300">
                No ByteTrack Lanes Found
              </p>
              <p className="text-mini text-neutral-500 max-w-sm mt-0.5">
                Run Step 1 Pre-Scan to detect face trajectories, populate ByteTrack lanes,
                and extract pose/occlusion warning ticks.
              </p>
              <button
                type="button"
                onClick={onPreScanClick}
                disabled={isPreScanActive}
                className="mt-2.5 px-3 py-1 bg-[var(--accent,#e94560)] hover:bg-[var(--accent,#e94560)]/90 text-white rounded text-xs font-semibold shadow-lg transition"
              >
                {isPreScanActive ? 'Pre-Scanning…' : 'Start 1 Pre-Scan'}
              </button>
            </div>
          ) : (
            trackLanes.map((track) => {
              const isSelected = selectedTrackId === track.trackId;
              const suggested = assignmentMap.get(track.trackId);
              const fszLabel =
                suggested?.suggestedFileName || track.suggestedFsz || `track_${track.trackId + 1}.fsz`;

              const trackColor = getFacesetColor(
                track.facesetPath || track.suggestedFsz,
                getTrackColor(track.trackId, track.color)
              );
              const trackWarnings = warningsByTrack.get(track.trackId) || [];

              const isTrackLocked = rangeLockIntervals.some(
                (iv) => iv.track_id === track.trackId && iv.locked !== false
              );

              return (
                <div
                  key={`track_lane_${track.trackId}`}
                  onClick={(e) => handleTrackOrRangeClick(track, undefined, e)}
                  className={`flex w-full min-h-[42px] transition-colors cursor-pointer group ${
                    isSelected
                      ? 'bg-[var(--accent,#e94560)]/10 ring-1 ring-inset ring-[var(--accent,#e94560)]/40'
                      : 'hover:bg-white/[0.02]'
                  }`}
                  data-testid={`lane-track-${track.trackId}`}
                >
                  {/* Left Column: Track Info & Assigned / Suggested .fsz Faceset */}
                  <div className="w-56 shrink-0 flex flex-col justify-center px-3 py-1.5 border-r border-white/10 bg-[var(--surface-2,#1a1d26)]/80">
                    <div className="flex items-center justify-between gap-1.5">
                      <div className="flex items-center gap-1.5">
                        <span
                          className="w-2.5 h-2.5 rounded-full shrink-0 shadow-sm"
                          style={{ backgroundColor: trackColor }}
                        />
                        <span className="font-bold text-white text-mini truncate">
                          {track.label}
                        </span>
                      </div>

                      {/* Lock Toggle / Indicator */}
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleTrackOrRangeClick(track, undefined, e);
                        }}
                        className={`p-1 rounded transition ${
                          isTrackLocked
                            ? 'text-[var(--accent,#e94560)] hover:text-white'
                            : 'text-neutral-500 hover:text-neutral-300'
                        }`}
                        title={isTrackLocked ? 'Range Locked (Step 2 Active)' : 'Click to Range Lock'}
                      >
                        <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="currentColor">
                          {isTrackLocked ? (
                            <path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3.1-9H8.9V6c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2z" />
                          ) : (
                            <path d="M12 17c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm6-9h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6h1.9c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm0 12H6V10h12v10z" />
                          )}
                        </svg>
                      </button>
                    </div>

                    {/* Assigned / Suggested .fsz Assignment Chip */}
                    {showSuggestedFsz && (
                      <div className="flex items-center justify-between gap-1 mt-0.5">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            onAssignFsz?.(track.trackId, fszLabel);
                          }}
                          title={`Assigned/Suggested Faceset: ${fszLabel} (${track.via})`}
                          className="flex items-center gap-1 max-w-[170px] px-1.5 py-0.5 rounded bg-black/40 border border-white/10 hover:border-white/20 text-micro font-mono text-neutral-300 truncate transition"
                        >
                          <span style={{ color: trackColor }}>●</span>
                          <span className="truncate">{fszLabel}</span>
                        </button>
                        <span className="text-nano font-mono text-neutral-500">
                          {track.totalFrames}f
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Right Column: Track Runway & Range Bars with Yaw/Occlusion Micro-Indicators */}
                  <div className="relative flex-1 h-full min-h-[42px] overflow-hidden flex items-center">
                    {/* Active Range Presence Bars */}
                    {track.ranges.map(([rStart, rEnd], rangeIndex) => {
                      const start = clampFrameIdx(rStart, 0, maxFrames);
                      const end = clampFrameIdx(rEnd, start, maxFrames);

                      if (end < viewStart || start > viewEnd) return null;

                      const leftPct = Math.max(0, frameToFraction(start) * 100);
                      const rightPct = Math.min(100, frameToFraction(end) * 100);
                      const widthPct = Math.max(0.5, rightPct - leftPct);

                      const activeInterval = rangeLockIntervals.find(
                        (iv) =>
                          iv.track_id === track.trackId &&
                          iv.frame_start === start &&
                          iv.frame_end === end
                      );
                      const isIntervalLocked = activeInterval?.locked !== false && isTrackLocked;

                      return (
                        <div
                          key={`range_bar_${track.trackId}_${rangeIndex}`}
                          onClick={(e) => handleTrackOrRangeClick(track, [start, end], e)}
                          className={`absolute h-7 rounded-md shadow-md flex items-center justify-between px-2 cursor-pointer transition-all border ${
                            isIntervalLocked
                              ? 'border-white/50 ring-2 ring-[var(--accent,#e94560)]/80 brightness-110'
                              : 'border-white/20 hover:border-white/40'
                          }`}
                          style={{
                            left: `${leftPct}%`,
                            width: `${widthPct}%`,
                            background: `linear-gradient(90deg, ${trackColor}cc, ${trackColor}88)`,
                          }}
                          title={`Click to Range Lock [${start}–${end}] (${end - start + 1} frames)`}
                        >
                          {/* Range Bar Label */}
                          <div className="flex items-center gap-1.5 overflow-hidden text-micro font-mono font-semibold text-black/90">
                            <span className="truncate">
                              {start}–{end}
                            </span>
                            {isIntervalLocked && (
                              <span className="text-nano px-1 bg-black/40 text-white rounded">
                                LOCKED
                              </span>
                            )}
                          </div>

                          {/* Range Trim Handles */}
                          {isIntervalLocked && activeInterval && (
                            <>
                              <div
                                onMouseDown={(e) =>
                                  handleRangeTrimDrag(activeInterval.id, 'start', e)
                                }
                                className="absolute left-0 top-0 bottom-0 w-2.5 bg-black/40 hover:bg-black/80 rounded-l cursor-ew-resize flex items-center justify-center text-nano text-white font-bold"
                                title="Trim start of interval"
                              >
                                |
                              </div>
                              <div
                                onMouseDown={(e) =>
                                  handleRangeTrimDrag(activeInterval.id, 'end', e)
                                }
                                className="absolute right-0 top-0 bottom-0 w-2.5 bg-black/40 hover:bg-black/80 rounded-r cursor-ew-resize flex items-center justify-center text-nano text-white font-bold"
                                title="Trim end of interval"
                              >
                                |
                              </div>
                            </>
                          )}
                        </div>
                      );
                    })}

                    {/* Amber / Red Micro-Indicators on Track Runway for Hard Yaw (>50) and Occlusion (>25%) */}
                    {showWarningTicks &&
                      trackWarnings.map((tick, wIdx) => {
                        const tickX = frameToFraction(tick.frame) * 100;
                        if (tickX < 0 || tickX > 100) return null;

                        const isOcclusion = tick.type === 'occlusion' || (tick.value && tick.value > 25);
                        const isHardYaw = tick.type === 'pose' || (tick.value && tick.value > 50);

                        // Red for foreign occlusion (>25%), Amber for hard yaw (>50)
                        const dotColor = isOcclusion ? '#f43f5e' : isHardYaw ? '#f59e0b' : '#38bdf8';
                        const dotLabel = isOcclusion
                          ? `Foreign Occlusion (>25%)`
                          : isHardYaw
                          ? `Hard Yaw (>50°)`
                          : tick.message;

                        return (
                          <div
                            key={`track_warn_${track.trackId}_${wIdx}`}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleSeekFrame(tick.frame);
                              onSelectTrack?.(track.trackId);
                            }}
                            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2.5 h-2.5 rounded-full border border-black shadow-md hover:scale-150 transition-transform cursor-pointer z-20"
                            style={{
                              left: `${tickX}%`,
                              backgroundColor: dotColor,
                            }}
                            title={`[Frame #${tick.frame}] ${dotLabel} - Click to jump`}
                          />
                        );
                      })}
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* ── Magnetic Snapping Visual Guide Line ────────────────────────────── */}
        {snapIndicatorFrame !== null && snapIndicatorFrame >= viewStart && snapIndicatorFrame <= viewEnd && (
          <div
            className="absolute top-0 bottom-0 pointer-events-none z-30"
            style={{
              left: `calc(14rem + (100% - 14rem) * ${frameToFraction(snapIndicatorFrame)})`,
            }}
          >
            <div className="w-px h-full border-l border-dashed border-sky-400 opacity-80" />
            <div className="absolute top-1 -translate-x-1/2 px-1 py-0.5 bg-sky-500 text-black text-nano font-mono font-bold rounded shadow">
              SNAP #{snapIndicatorFrame}
            </div>
          </div>
        )}

        {/* ── Hover Scrub Loupe Preview Popover ──────────────────────────────── */}
        {hoverFrame !== null && hoverClientX !== null && (
          <div
            className="absolute bottom-full mb-1 z-40 pointer-events-none -translate-x-1/2 flex flex-col items-center"
            style={{
              left: `calc(14rem + ${hoverClientX}px)`,
            }}
          >
            <div className="px-2 py-1 bg-black/90 border border-white/20 rounded shadow-xl flex flex-col items-center gap-0.5">
              <span className="text-micro font-mono font-bold text-white">
                {frameToTimecode(hoverFrame, fps)}
              </span>
              <span className="text-nano font-mono text-[var(--accent,#e94560)]">
                Frame #{hoverFrame}
              </span>
            </div>
            <div className="w-2 h-2 rotate-45 bg-black/90 border-r border-b border-white/20 -mt-1" />
          </div>
        )}

        {/* ── Vertical Playhead Scrubber Line ─────────────────────────────────── */}
        {activeFrame >= viewStart && activeFrame <= viewEnd && (
          <div
            className="absolute top-0 bottom-0 pointer-events-none z-30 transition-all duration-75"
            style={{
              left: `calc(14rem + (100% - 14rem) * ${frameToFraction(activeFrame)})`,
            }}
          >
            {/* Scrubber Thumb Marker */}
            <div className="absolute top-0 -translate-x-1/2 w-3.5 h-4 bg-[var(--accent,#e94560)] clip-path-triangle shadow-lg flex items-center justify-center">
              <div className="w-1 h-1 bg-white rounded-full" />
            </div>

            {/* Glowing Scrubber Line */}
            <div className="w-0.5 h-full bg-[var(--accent,#e94560)] shadow-[0_0_8px_rgba(233,69,96,0.8)]" />

            {/* Frame Badge Tooltip */}
            <div className="absolute -top-6 -translate-x-1/2 px-1.5 py-0.5 bg-[var(--accent,#e94560)] text-white text-nano font-mono font-bold rounded shadow whitespace-nowrap">
              #{activeFrame}
            </div>
          </div>
        )}
      </div>

      {/* ── Bottom Footer Status Bar ────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-1.5 bg-[var(--surface-2,#1a1d26)] border-t border-white/5 text-mini text-neutral-400">
        <div className="flex items-center gap-3">
          <span>
            Visible: <span className="font-mono text-white">#{viewStart}</span> –{' '}
            <span className="font-mono text-white">#{viewEnd}</span> ({viewSpan + 1} frames)
          </span>
          <span className="text-neutral-600">|</span>
          <span>
            Zoom: <span className="font-mono text-white">{Math.round((maxFrames / viewSpan) * 100)}%</span>
          </span>
          <span className="text-neutral-600">|</span>
          <span>
            Snap: <span className={snapEnabled ? 'text-sky-300' : 'text-neutral-500'}>{snapEnabled ? 'ON' : 'OFF'}</span>
          </span>
        </div>

        <div className="flex items-center gap-3">
          {selectedTrackId !== null && (
            <span className="text-white font-medium">
              Selected: Track #{typeof selectedTrackId === 'number' ? selectedTrackId + 1 : selectedTrackId}
            </span>
          )}
          <span className="text-neutral-500 text-micro">
            Shortcuts: <kbd className="px-1 bg-black/40 rounded border border-white/10">I</kbd> Trim In ·{' '}
            <kbd className="px-1 bg-black/40 rounded border border-white/10">O</kbd> Trim Out ·{' '}
            <kbd className="px-1 bg-black/40 rounded border border-white/10">M</kbd> Marker ·{' '}
            <kbd className="px-1 bg-black/40 rounded border border-white/10">Ctrl+Wheel</kbd> Zoom
          </span>
        </div>
      </div>
    </div>
  );
};

export default NleTimeline;
