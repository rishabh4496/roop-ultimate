/**
 * NleTimeline.tsx: Non-Linear Editing Multi-Track Timeline for FaceSwap.
 *
 * Capabilities:
 *   1. Displays ByteTrack Lanes `Track #1..#N` populated from Step 1 Pre-Scan (trajectory spans/summary).
 *   2. Displays pose, occlusion, tracking gap, and interpolation warning ticks with instant-seek tooltips.
 *   3. Shows suggested `.fsz` assignments and faceset binding metadata (`via: pre_scan`).
 *   4. Interactivity: Clicking any track lane or range bar selects the track, dispatches `onSelectTrack`
 *      (triggering highlight of the corresponding face badge in `PreviewViewport`), and creates/toggles
 *      Step 2 `Range Lock` active intervals (`RangeLockInterval`).
 *   5. VFR & Exact Math: Uses `ptsFrameMap.ts` for time/frame conversions and ensures trim handles,
 *      playhead positions, and keyframes store exact integer `frame_idx` values with zero drift.
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

// Re-export core interval and keyframe interfaces
export type { RangeLockInterval, KeyframePoint };

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
  selectedTrackId?: number | null;
  selectedRangeId?: string | null;

  // Global clip trim in/out (exact integer frame_idx)
  trimIn?: number;
  trimOut?: number;

  // Identity / parameter keyframes (exact integer frame_idx)
  keyframes?: KeyframePoint[];

  // Event handlers
  onSeek?: (frame: number, timeSeconds?: number) => void;
  onSelectTrack?: (trackId: number | null) => void;
  onSelectRange?: (interval: RangeLockInterval | null) => void;
  onUpdateRangeLockIntervals?: (intervals: RangeLockInterval[]) => void;
  onToggleTrackActive?: (trackId: number, active: boolean) => void;
  onUpdateTrim?: (trimIn: number, trimOut: number) => void;
  onUpdateKeyframes?: (keyframes: KeyframePoint[]) => void;
  onAssignFsz?: (trackId: number, fszFileName: string) => void;
  onPreScanClick?: () => void;

  // Step indicators & UI flags
  isPreScanActive?: boolean;
  isRangeLockActive?: boolean;
  isRenderActive?: boolean;
  showWarningTicks?: boolean;
  showSuggestedFsz?: boolean;

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

// ── Component Implementation ──────────────────────────────────────────────────

export const NleTimeline: React.FC<NleTimelineProps> = ({
  frame: frameProp,
  currentFrame: currentFrameProp,
  currentTime,
  maxFrames: rawMaxFrames,
  fps = 30,
  isPlaying = false,
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
  keyframes = [],
  onSeek,
  onSelectTrack,
  onSelectRange,
  onUpdateRangeLockIntervals,
  onToggleTrackActive,
  onUpdateTrim,
  onUpdateKeyframes,
  onAssignFsz,
  onPreScanClick,
  isPreScanActive = false,
  isRangeLockActive = false,
  isRenderActive = false,
  showWarningTicks = true,
  showSuggestedFsz = true,
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

  // Zoom helpers
  const handleZoom = useCallback((factor: number, anchorFrame?: number) => {
    setViewRange(([start, end]) => {
      const currentSpan = Math.max(1, end - start);
      const targetSpan = Math.max(4, Math.min(maxFrames - 1, Math.round(currentSpan * factor)));
      const anchor = anchorFrame !== undefined ? anchorFrame : Math.round((start + end) / 2);
      const ratio = currentSpan > 0 ? (anchor - start) / currentSpan : 0.5;

      let newStart = Math.round(anchor - targetSpan * ratio);
      let newEnd = newStart + targetSpan;

      if (newStart < 0) {
        newEnd = Math.min(maxFrames - 1, newEnd - newStart);
        newStart = 0;
      }
      if (newEnd >= maxFrames) {
        newStart = Math.max(0, newStart - (newEnd - (maxFrames - 1)));
        newEnd = maxFrames - 1;
      }

      return [clampFrameIdx(newStart, 0, maxFrames), clampFrameIdx(newEnd, 0, maxFrames)];
    });
  }, [maxFrames]);

  const handleResetZoom = useCallback(() => {
    setViewRange([0, maxFrames - 1]);
  }, [maxFrames]);

  // ── Coordinate Conversions ───────────────────────────────────────────────────
  const trackContainerRef = useRef<HTMLDivElement>(null);

  const frameToFraction = useCallback(
    (frameIdx: number) => {
      const clamped = clampFrameIdx(frameIdx, 0, maxFrames);
      if (viewSpan <= 0) return 0;
      return (clamped - viewStart) / viewSpan;
    },
    [viewStart, viewSpan, maxFrames]
  );

  const xToFrame = useCallback(
    (clientX: number) => {
      if (!trackContainerRef.current) return viewStart;
      const rect = trackContainerRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(rect.width, clientX - rect.left));
      const ratio = rect.width > 0 ? x / rect.width : 0;
      const rawFrame = viewStart + ratio * viewSpan;
      return clampFrameIdx(rawFrame, 0, maxFrames);
    },
    [viewStart, viewSpan, maxFrames]
  );

  // ── Seek & Navigation ────────────────────────────────────────────────────────
  const handleSeekFrame = useCallback(
    (targetFrame: number) => {
      const exactFrame = clampFrameIdx(targetFrame, 0, maxFrames);
      const timeSec = frameMap.resolveTimeFromFrame(exactFrame);
      onSeek?.(exactFrame, timeSec);
    },
    [maxFrames, frameMap, onSeek]
  );

  // Scrubber drag handler
  const [isScrubbing, setIsScrubbing] = useState(false);

  const handleScrubberMouseDown = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsScrubbing(true);
      const frame = xToFrame(e.clientX);
      handleSeekFrame(frame);

      const onMouseMove = (ev: globalThis.MouseEvent) => {
        const nextFrame = xToFrame(ev.clientX);
        handleSeekFrame(nextFrame);
      };

      const onMouseUp = () => {
        setIsScrubbing(false);
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
      };

      window.addEventListener('mousemove', onMouseMove);
      window.addEventListener('mouseup', onMouseUp);
    },
    [xToFrame, handleSeekFrame]
  );

  // ── Wheel Zoom & Pan ─────────────────────────────────────────────────────────
  const handleWheel = useCallback(
    (e: WheelEvent<HTMLDivElement>) => {
      if (e.ctrlKey || e.metaKey) {
        // Zoom
        e.preventDefault();
        const anchor = xToFrame(e.clientX);
        const zoomFactor = e.deltaY > 0 ? 1.25 : 0.8;
        handleZoom(zoomFactor, anchor);
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
    [xToFrame, handleZoom, viewSpan, maxFrames]
  );

  // ── Track & Range Lock Interactivity ─────────────────────────────────────────
  /**
   * Handle clicking any track lane or range bar:
   * 1. Updates selectedTrackId and notifies `onSelectTrack` (highlighting face badge in PreviewViewport).
   * 2. Adds or activates a Step 2 `RangeLockInterval` for this track and updates active intervals.
   */
  const handleTrackOrRangeClick = useCallback(
    (track: ByteTrackLane, range?: [number, number], e?: MouseEvent) => {
      if (e) {
        e.stopPropagation();
      }

      // 1. Highlight track badge in PreviewViewport
      onSelectTrack?.(track.trackId);

      // Determine the span interval to lock
      const targetStart = range ? range[0] : track.firstFrame;
      const targetEnd = range ? range[1] : track.lastFrame;

      // Check if an interval already exists for this track & range
      const existingIdx = rangeLockIntervals.findIndex(
        (iv) =>
          iv.track_id === track.trackId &&
          Math.max(iv.frame_start, targetStart) <= Math.min(iv.frame_end, targetEnd)
      );

      let updated: RangeLockInterval[];

      if (existingIdx >= 0) {
        const existing = rangeLockIntervals[existingIdx];
        // Toggle locked state or re-select
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
        // Create new RangeLockInterval
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

  // Range Bar Trim Handle Dragging
  const handleRangeTrimDrag = useCallback(
    (
      intervalId: string,
      handle: 'start' | 'end',
      e: MouseEvent<HTMLDivElement>
    ) => {
      e.stopPropagation();
      e.preventDefault();

      const iv = rangeLockIntervals.find((item) => item.id === intervalId);
      if (!iv) return;

      const onMouseMove = (ev: globalThis.MouseEvent) => {
        const currentFrame = xToFrame(ev.clientX);
        const nextIntervals = rangeLockIntervals.map((item) => {
          if (item.id !== intervalId) return item;
          if (handle === 'start') {
            const newStart = Math.min(currentFrame, item.frame_end);
            return clampRangeLockInterval(
              { ...item, frame_start: newStart },
              0,
              maxFrames
            );
          } else {
            const newEnd = Math.max(currentFrame, item.frame_start);
            return clampRangeLockInterval(
              { ...item, frame_end: newEnd },
              0,
              maxFrames
            );
          }
        });
        onUpdateRangeLockIntervals?.(nextIntervals);
      };

      const onMouseUp = () => {
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
      };

      window.addEventListener('mousemove', onMouseMove);
      window.addEventListener('mouseup', onMouseUp);
    },
    [rangeLockIntervals, xToFrame, maxFrames, onUpdateRangeLockIntervals]
  );

  // ── Global Trim Handles Dragging ─────────────────────────────────────────────
  const handleGlobalTrimDrag = useCallback(
    (handle: 'in' | 'out', e: MouseEvent<HTMLDivElement>) => {
      e.stopPropagation();
      e.preventDefault();

      const onMouseMove = (ev: globalThis.MouseEvent) => {
        const f = xToFrame(ev.clientX);
        if (handle === 'in') {
          const newIn = Math.min(f, trimOut);
          onUpdateTrim?.(clampFrameIdx(newIn, 0, maxFrames), trimOut);
        } else {
          const newOut = Math.max(f, trimIn);
          onUpdateTrim?.(trimIn, clampFrameIdx(newOut, trimIn, maxFrames));
        }
      };

      const onMouseUp = () => {
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
      };

      window.addEventListener('mousemove', onMouseMove);
      window.addEventListener('mouseup', onMouseUp);
    },
    [xToFrame, trimIn, trimOut, maxFrames, onUpdateTrim]
  );

  // ── Keyframe Dragging ────────────────────────────────────────────────────────
  const handleKeyframeDrag = useCallback(
    (kfId: string, e: MouseEvent<HTMLDivElement>) => {
      e.stopPropagation();
      e.preventDefault();

      const onMouseMove = (ev: globalThis.MouseEvent) => {
        const f = xToFrame(ev.clientX);
        const updatedKfs = keyframes.map((k) =>
          k.id === kfId ? clampKeyframe({ ...k, frame_idx: f }, 0, maxFrames) : k
        );
        onUpdateKeyframes?.(normalizeKeyframePoints(updatedKfs, maxFrames));
      };

      const onMouseUp = () => {
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
      };

      window.addEventListener('mousemove', onMouseMove);
      window.addEventListener('mouseup', onMouseUp);
    },
    [keyframes, xToFrame, maxFrames, onUpdateKeyframes]
  );

  // ── Minimap Scrubbing & Panning ──────────────────────────────────────────────
  const minimapRef = useRef<HTMLDivElement>(null);

  const handleMinimapMouseDown = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      e.preventDefault();
      if (!minimapRef.current) return;
      const rect = minimapRef.current.getBoundingClientRect();
      const clickRatio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const targetFrame = Math.round(clickRatio * (maxFrames - 1));

      // Center the viewport window around click position
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

  // ── Ticks Generation for Adaptive Ruler ──────────────────────────────────────
  const rulerTicks = useMemo(() => {
    const ticks: Array<{ frame: number; label: string; major: boolean }> = [];
    const span = viewSpan;

    // Pick an integer step based on visible span
    let step = 1;
    if (span > 3000) step = 500;
    else if (span > 1500) step = 250;
    else if (span > 600) step = 100;
    else if (span > 300) step = 50;
    else if (span > 150) step = 25;
    else if (span > 60) step = 10;
    else if (span > 30) step = 5;
    else if (span > 15) step = 2;
    else step = 1;

    const firstTick = Math.ceil(viewStart / step) * step;
    for (let f = firstTick; f <= viewEnd; f += step) {
      const isMajor = f % (step * 5) === 0 || f === 0;
      let label = '';
      if (span <= 60) {
        label = `#${f}`;
      } else if (span <= 500) {
        label = frameToTimecode(f, fps).slice(3); // MM:SS:FF
      } else {
        label = frameToTimecode(f, fps);
      }
      ticks.push({ frame: f, label, major: isMajor });
    }
    return ticks;
  }, [viewStart, viewEnd, viewSpan, fps]);

  // Suggested .fsz assignment map
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

  // Active playhead time string
  const activeTimecode = useMemo(() => frameToTimecode(activeFrame, fps), [activeFrame, fps]);
  const activeSeconds = useMemo(
    () => formatSeconds(frameMap.resolveTimeFromFrame(activeFrame)),
    [activeFrame, frameMap]
  );

  // Active Range Lock counts
  const lockedIntervalsCount = useMemo(
    () => rangeLockIntervals.filter((iv) => iv.locked !== false).length,
    [rangeLockIntervals]
  );

  return (
    <div
      className={`nle-timeline-root flex flex-col w-full bg-[var(--surface-1,#12141a)] border border-[var(--border-color,rgba(255,255,255,0.08))] rounded-xl overflow-hidden shadow-2xl select-none font-sans text-xs ${className}`}
      style={style}
      onWheel={handleWheel}
      data-testid="nle-timeline"
    >
      {/* ── Top Header Toolbar ──────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-2.5 bg-[var(--surface-2,#1a1d26)] border-b border-[var(--border-color,rgba(255,255,255,0.08))]">
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
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium border transition-colors ${
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
            <span className="font-mono text-[10px] opacity-80">
              ({trackLanes.length} {trackLanes.length === 1 ? 'Track' : 'Tracks'})
            </span>
          </button>

          {/* Step 2 Range Lock Badge */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium border ${
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
            <span className="font-mono text-[10px] opacity-80">
              ({lockedIntervalsCount} {lockedIntervalsCount === 1 ? 'Interval' : 'Intervals'})
            </span>
          </div>
        </div>

        {/* Timecode & Frame Readout */}
        <div className="flex items-center gap-4">
          {/* Exact Timecode and Seconds Display */}
          <div className="flex items-center gap-2 px-3 py-1 bg-black/40 border border-white/10 rounded-lg font-mono">
            <span className="text-[10px] uppercase text-neutral-400 font-semibold tracking-wider">
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
          <div className="hidden sm:flex items-center gap-2">
            <div
              className="flex items-center gap-1.5 px-2 py-1 bg-white/5 border border-white/10 rounded text-[11px] font-mono text-neutral-300"
              title="Trim In Point"
            >
              <span className="text-amber-400 font-bold">IN:</span>
              <span className="tabular-nums">#{trimIn}</span>
            </div>
            <div
              className="flex items-center gap-1.5 px-2 py-1 bg-white/5 border border-white/10 rounded text-[11px] font-mono text-neutral-300"
              title="Trim Out Point"
            >
              <span className="text-amber-400 font-bold">OUT:</span>
              <span className="tabular-nums">#{trimOut}</span>
            </div>
          </div>

          {/* Zoom Controls */}
          <div className="flex items-center gap-1 bg-black/30 border border-white/10 rounded-lg p-0.5">
            <button
              type="button"
              onClick={() => handleZoom(0.7, activeFrame)}
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
              onClick={() => handleZoom(1.4, activeFrame)}
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
              className="px-2 py-0.5 text-[10px] font-medium hover:bg-white/10 rounded text-neutral-300 hover:text-white transition"
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

      {/* ── Main Runway Container (Ruler + Tracks) ──────────────────────────── */}
      <div className="relative flex flex-col w-full overflow-hidden bg-black/20">
        {/* Global Warning Bar (if warning ticks exist) */}
        {showWarningTicks && warningTicks.length > 0 && (
          <div className="relative w-full h-5 bg-black/40 border-b border-white/5 flex items-center px-2">
            <span className="text-[10px] text-amber-400 font-semibold uppercase tracking-wider mr-2 shrink-0">
              Warnings ({warningTicks.length}):
            </span>
            <div className="relative flex-1 h-full overflow-hidden">
              {warningTicks.map((tick, idx) => {
                const tickX = frameToFraction(tick.frame) * 100;
                if (tickX < 0 || tickX > 100) return null;
                const isOcclusion = tick.type === 'occlusion';
                const isPose = tick.type === 'pose';
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

        {/* Ruler Row */}
        <div className="flex w-full h-7 bg-[var(--surface-2,#1a1d26)] border-b border-white/10">
          {/* Header spacer column */}
          <div className="w-56 shrink-0 flex items-center justify-between px-3 border-r border-white/10 text-[11px] font-semibold text-neutral-400 uppercase tracking-wider">
            <span>Lanes</span>
            <span className="text-[10px] text-neutral-500 font-mono">
              fps: {fps}
            </span>
          </div>

          {/* Time/Frame Ruler Track */}
          <div
            ref={trackContainerRef}
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
                    <span className="text-[9px] font-mono text-neutral-400 -translate-x-1/2 select-none">
                      {tick.label}
                    </span>
                  )}
                </div>
              );
            })}

            {/* Global Trim In Bracket Marker */}
            {trimIn >= viewStart && trimIn <= viewEnd && (
              <div
                onMouseDown={(e) => handleGlobalTrimDrag('in', e)}
                className="absolute top-0 bottom-0 w-3 -translate-x-full cursor-ew-resize z-20 flex items-center justify-center group"
                style={{ left: `${frameToFraction(trimIn) * 100}%` }}
                title={`Trim In Handle: #${trimIn} (drag to trim)`}
              >
                <div className="w-1.5 h-5 bg-amber-400 rounded-l group-hover:scale-110 shadow-md flex items-center justify-center text-[9px] font-bold text-black">
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
                title={`Trim Out Handle: #${trimOut} (drag to trim)`}
              >
                <div className="w-1.5 h-5 bg-amber-400 rounded-r group-hover:scale-110 shadow-md flex items-center justify-center text-[9px] font-bold text-black">
                  ]
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ── ByteTrack Lanes Track #1..#N ────────────────────────────────────── */}
        <div className="flex flex-col w-full divide-y divide-white/5">
          {trackLanes.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-10 px-4 text-center text-neutral-400">
              <svg
                className="w-8 h-8 mb-2 text-neutral-600"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
              >
                <path d="M15 10l-4 4l6 6l4 -16l-18 7l4 2l2 6l3 -4" />
              </svg>
              <p className="text-sm font-medium text-neutral-300">
                No ByteTrack Lanes Found
              </p>
              <p className="text-xs text-neutral-500 max-w-sm mt-1">
                Run Step 1 Pre-Scan to detect face trajectories, populate ByteTrack lanes,
                and extract pose/occlusion warning ticks.
              </p>
              <button
                type="button"
                onClick={onPreScanClick}
                disabled={isPreScanActive}
                className="mt-3 px-3.5 py-1.5 bg-[var(--accent,#e94560)] hover:bg-[var(--accent,#e94560)]/90 text-white rounded-lg text-xs font-semibold shadow-lg transition"
              >
                {isPreScanActive ? 'Pre-Scanning…' : 'Start 1 Pre-Scan'}
              </button>
            </div>
          ) : (
            trackLanes.map((track) => {
              const isSelected = selectedTrackId === track.trackId;
              const trackColor = getTrackColor(track.trackId, track.color);
              const trackWarnings = warningsByTrack.get(track.trackId) || [];
              const suggested = assignmentMap.get(track.trackId);
              const fszLabel =
                suggested?.suggestedFileName || track.suggestedFsz || `track_${track.trackId + 1}.fsz`;

              // Check if track has an active RangeLockInterval
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
                  {/* Left Column: Track Info & Suggested .fsz Assignment */}
                  <div className="w-56 shrink-0 flex flex-col justify-center px-3 py-1.5 border-r border-white/10 bg-[var(--surface-2,#1a1d26)]/80">
                    <div className="flex items-center justify-between gap-1.5">
                      <div className="flex items-center gap-1.5">
                        <span
                          className="w-2.5 h-2.5 rounded-full shrink-0 shadow-sm"
                          style={{ backgroundColor: trackColor }}
                        />
                        <span className="font-bold text-white text-[12px] truncate">
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

                    {/* Suggested .fsz Assignment Chip */}
                    {showSuggestedFsz && (
                      <div className="flex items-center justify-between gap-1 mt-0.5">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            onAssignFsz?.(track.trackId, fszLabel);
                          }}
                          title={`Suggested Faceset assignment: ${fszLabel} (${track.via})`}
                          className="flex items-center gap-1 max-w-[170px] px-1.5 py-0.5 rounded bg-black/40 border border-white/10 hover:border-white/20 text-[10px] font-mono text-neutral-300 truncate transition"
                        >
                          <span className="text-[var(--accent,#e94560)]">✨</span>
                          <span className="truncate">{fszLabel}</span>
                        </button>
                        <span className="text-[9px] font-mono text-neutral-500">
                          {track.totalFrames}f
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Right Column: Track Runway & Range Bars */}
                  <div className="relative flex-1 h-full min-h-[42px] overflow-hidden flex items-center">
                    {/* Active Range Bars */}
                    {track.ranges.map(([rStart, rEnd], rangeIndex) => {
                      const start = clampFrameIdx(rStart, 0, maxFrames);
                      const end = clampFrameIdx(rEnd, start, maxFrames);

                      // Check if this range lies within the visible view
                      if (end < viewStart || start > viewEnd) return null;

                      const leftPct = Math.max(0, frameToFraction(start) * 100);
                      const rightPct = Math.min(100, frameToFraction(end) * 100);
                      const widthPct = Math.max(0.5, rightPct - leftPct);

                      // Match corresponding RangeLockInterval
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
                          <div className="flex items-center gap-1.5 overflow-hidden text-[10px] font-mono font-semibold text-black/90">
                            <span className="truncate">
                              {start}–{end}
                            </span>
                            {isIntervalLocked && (
                              <span className="text-[9px] px-1 bg-black/40 text-white rounded">
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
                                className="absolute left-0 top-0 bottom-0 w-2.5 bg-black/40 hover:bg-black/80 rounded-l cursor-ew-resize flex items-center justify-center text-[8px] text-white font-bold"
                                title="Trim start of interval"
                              >
                                |
                              </div>
                              <div
                                onMouseDown={(e) =>
                                  handleRangeTrimDrag(activeInterval.id, 'end', e)
                                }
                                className="absolute right-0 top-0 bottom-0 w-2.5 bg-black/40 hover:bg-black/80 rounded-r cursor-ew-resize flex items-center justify-center text-[8px] text-white font-bold"
                                title="Trim end of interval"
                              >
                                |
                              </div>
                            </>
                          )}
                        </div>
                      );
                    })}

                    {/* Warning Ticks on Track Runway */}
                    {showWarningTicks &&
                      trackWarnings.map((tick, wIdx) => {
                        const tickX = frameToFraction(tick.frame) * 100;
                        if (tickX < 0 || tickX > 100) return null;
                        const isOcclusion = tick.type === 'occlusion';
                        const isPose = tick.type === 'pose';
                        const dotColor = isOcclusion ? '#f43f5e' : isPose ? '#f59e0b' : '#38bdf8';

                        return (
                          <div
                            key={`track_warn_${track.trackId}_${wIdx}`}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleSeekFrame(tick.frame);
                              onSelectTrack?.(track.trackId);
                            }}
                            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2 h-2 rounded-full border border-black shadow-sm hover:scale-150 transition cursor-pointer z-10"
                            style={{
                              left: `${tickX}%`,
                              backgroundColor: dotColor,
                            }}
                            title={`[Frame #${tick.frame}] ${tick.message}`}
                          />
                        );
                      })}
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* ── Keyframes Lane (if keyframes provided) ──────────────────────────── */}
        {keyframes.length > 0 && (
          <div className="flex w-full h-6 bg-black/30 border-t border-white/5">
            <div className="w-56 shrink-0 flex items-center px-3 text-[10px] font-semibold text-neutral-400 uppercase tracking-wider border-r border-white/10">
              Keyframes
            </div>
            <div className="relative flex-1 h-full overflow-hidden">
              {keyframes.map((kf) => {
                const kfX = frameToFraction(kf.frame_idx) * 100;
                if (kfX < 0 || kfX > 100) return null;

                return (
                  <div
                    key={`kf_${kf.id}`}
                    onMouseDown={(e) => handleKeyframeDrag(kf.id, e)}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleSeekFrame(kf.frame_idx);
                    }}
                    className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3 h-3 rotate-45 bg-amber-400 border border-black shadow hover:scale-125 transition-transform cursor-ew-resize z-20"
                    style={{ left: `${kfX}%` }}
                    title={`Keyframe at frame #${kf.frame_idx} (drag to reposition)`}
                  />
                );
              })}
            </div>
          </div>
        )}

        {/* ── Vertical Playhead Scrubber Line ─────────────────────────────────── */}
        {activeFrame >= viewStart && activeFrame <= viewEnd && (
          <div
            className="absolute top-0 bottom-0 pointer-events-none z-30 transition-all duration-75"
            style={{ left: `calc(14rem + (100% - 14rem) * ${frameToFraction(activeFrame)})` }}
          >
            {/* Scrubber Thumb Marker */}
            <div className="absolute top-0 -translate-x-1/2 w-3.5 h-4 bg-[var(--accent,#e94560)] clip-path-triangle shadow-lg flex items-center justify-center">
              <div className="w-1 h-1 bg-white rounded-full" />
            </div>

            {/* Glowing Scrubber Line */}
            <div className="w-0.5 h-full bg-[var(--accent,#e94560)] shadow-[0_0_8px_rgba(233,69,96,0.8)]" />

            {/* Frame Badge Tooltip */}
            <div className="absolute -top-6 -translate-x-1/2 px-1.5 py-0.5 bg-[var(--accent,#e94560)] text-white text-[9px] font-mono font-bold rounded shadow whitespace-nowrap">
              #{activeFrame}
            </div>
          </div>
        )}
      </div>

      {/* ── Bottom Footer Status Bar ────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-1.5 bg-[var(--surface-2,#1a1d26)] border-t border-white/5 text-[11px] text-neutral-400">
        <div className="flex items-center gap-3">
          <span>
            Visible: <span className="font-mono text-white">#{viewStart}</span> –{' '}
            <span className="font-mono text-white">#{viewEnd}</span> ({viewSpan + 1} frames)
          </span>
          <span className="text-neutral-600">|</span>
          <span>
            Zoom: <span className="font-mono text-white">{Math.round((maxFrames / viewSpan) * 100)}%</span>
          </span>
        </div>

        <div className="flex items-center gap-2">
          {selectedTrackId !== null && (
            <span className="text-white font-medium">
              Selected: Track #{selectedTrackId + 1}
            </span>
          )}
          <span className="text-neutral-500">
            Click track or range bar to update Step 2 Range Lock &amp; highlight face in Viewport
          </span>
        </div>
      </div>
    </div>
  );
};

export default NleTimeline;
