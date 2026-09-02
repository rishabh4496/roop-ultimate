import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Scissors,
  Film,
  RotateCcw,
  Sparkles,
  ChevronLeft,
  ChevronRight,
  Eye,
  AlertTriangle,
  CheckCircle2,
} from 'lucide-react';

export interface FaceTrackingRange {
  start: number;
  end: number;
  faceId: string;
  label?: string;
  confidence?: number;
}

export interface TimelineDropoutRange {
  start: number;
  end: number;
}

/**
 * Converts a frame index to SMPTE timecode string: HH:MM:SS:FF
 */
export function frameToTimecode(frame: number, fps = 30): string {
  const f = Math.max(0, Math.round(frame));
  const safeFps = Math.max(1, Math.round(fps));
  const ff = f % safeFps;
  const totalSeconds = Math.floor(f / safeFps);
  const ss = totalSeconds % 60;
  const mm = Math.floor(totalSeconds / 60) % 60;
  const hh = Math.floor(totalSeconds / 3600);
  const p = (n: number) => n.toString().padStart(2, '0');
  return `${p(hh)}:${p(mm)}:${p(ss)}:${p(ff)}`;
}

/**
 * Converts SMPTE timecode string (HH:MM:SS:FF) back to frame index
 */
export function timecodeToFrame(timecode: string, fps = 30): number {
  const parts = timecode.split(':').map((p) => parseInt(p, 10) || 0);
  const safeFps = Math.max(1, Math.round(fps));
  if (parts.length === 4) {
    const [hh, mm, ss, ff] = parts;
    return (hh * 3600 + mm * 60 + ss) * safeFps + ff;
  }
  return 0;
}

/**
 * Computes complementary dropout ranges where no face tracking is present.
 */
export function calculateDropoutRanges(
  totalFrames: number,
  ranges: FaceTrackingRange[]
): TimelineDropoutRange[] {
  if (totalFrames <= 0) return [];
  if (!ranges || ranges.length === 0) {
    return [{ start: 0, end: totalFrames }];
  }

  const sorted = [...ranges].sort((a, b) => a.start - b.start);
  const dropouts: TimelineDropoutRange[] = [];

  let currentPointer = 0;
  for (const r of sorted) {
    const rStart = Math.max(0, Math.min(totalFrames, r.start));
    const rEnd = Math.max(0, Math.min(totalFrames, r.end));

    if (rStart > currentPointer) {
      dropouts.push({ start: currentPointer, end: rStart });
    }
    currentPointer = Math.max(currentPointer, rEnd);
  }

  if (currentPointer < totalFrames) {
    dropouts.push({ start: currentPointer, end: totalFrames });
  }

  return dropouts;
}

/**
 * Computes face tracking coverage percentage across the total duration.
 */
export function calculateCoveragePercentage(
  totalFrames: number,
  ranges: FaceTrackingRange[]
): { trackedPercent: number; dropoutPercent: number } {
  if (totalFrames <= 0) return { trackedPercent: 0, dropoutPercent: 0 };

  const mask = new Uint8Array(totalFrames);
  ranges.forEach((r) => {
    const s = Math.max(0, Math.min(totalFrames - 1, r.start));
    const e = Math.max(0, Math.min(totalFrames, r.end));
    for (let i = s; i < e; i++) {
      mask[i] = 1;
    }
  });

  let trackedCount = 0;
  for (let i = 0; i < totalFrames; i++) {
    if (mask[i] === 1) trackedCount++;
  }

  const trackedPercent = parseFloat(((trackedCount / totalFrames) * 100).toFixed(1));
  const dropoutPercent = parseFloat((100 - trackedPercent).toFixed(1));

  return { trackedPercent, dropoutPercent };
}

export interface TimelineTrackBarProps {
  /** Total number of frames in the sequence or video */
  totalFrames: number;
  /** Current active playhead frame index (0-indexed) */
  currentFrame: number;
  /** Video frame rate for timecode conversion (default: 30) */
  fps?: number;
  /** Array of frame indices where camera scene cuts were detected */
  sceneCuts?: number[];
  /** Continuous ranges where the target face was actively tracked */
  faceTrackingRanges?: FaceTrackingRange[];
  /** Callback fired when user seeks to a new frame */
  onSeek: (frame: number) => void;
  /** Callback fired when the in/out trimmer range boundaries change */
  onRangeChange?: (start: number, end: number) => void;
  /** Initial In-point boundary frame (default: 0) */
  initialStartFrame?: number;
  /** Initial Out-point boundary frame (default: totalFrames) */
  initialEndFrame?: number;
  /** Optional video source URL for real frame thumbnail preview on hover */
  videoUrl?: string;
  /** Optional custom thumbnail provider function: (frame: number) => string URL */
  getThumbnailUrl?: (frame: number) => string | undefined;
  /** Custom CSS styling class names */
  className?: string;
}

type DragMode = 'none' | 'playhead' | 'startHandle' | 'endHandle' | 'rangeSpan';

export function TimelineTrackBar({
  totalFrames = 300,
  currentFrame = 0,
  fps = 30,
  sceneCuts = [],
  faceTrackingRanges = [],
  onSeek,
  onRangeChange,
  initialStartFrame = 0,
  initialEndFrame,
  videoUrl,
  getThumbnailUrl,
  className = '',
}: TimelineTrackBarProps) {
  const safeTotalFrames = Math.max(1, totalFrames);
  const safeFps = Math.max(1, fps);

  // ── Range Trimmer State ([In, Out]) ─────────────────────────────────────
  const [startFrame, setStartFrame] = useState<number>(Math.max(0, initialStartFrame));
  const [endFrame, setEndFrame] = useState<number>(
    initialEndFrame !== undefined ? Math.min(safeTotalFrames, initialEndFrame) : safeTotalFrames
  );

  // Playback Simulation Toggle
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  // ── Hover Scrubber Tooltip State ────────────────────────────────────────
  const [hoverFrame, setHoverFrame] = useState<number | null>(null);
  const [hoverPositionX, setHoverPositionX] = useState<number>(0);
  const [isHovering, setIsHovering] = useState<boolean>(false);

  // Dragging State Ref to eliminate event listener churn
  const dragModeRef = useRef<DragMode>('none');
  const dragStartXRef = useRef<number>(0);
  const dragStartRangeRef = useRef<{ start: number; end: number }>({ start: 0, end: safeTotalFrames });

  const containerRef = useRef<HTMLDivElement | null>(null);
  const offscreenVideoRef = useRef<HTMLVideoElement | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);

  // Ensure endFrame stays synchronized if totalFrames changes
  useEffect(() => {
    if (initialEndFrame === undefined) {
      setEndFrame(safeTotalFrames);
    }
  }, [initialEndFrame, safeTotalFrames]);

  // ── Computations: Dropouts & Tracking Statistics ────────────────────────
  const dropoutRanges = useMemo(
    () => calculateDropoutRanges(safeTotalFrames, faceTrackingRanges),
    [safeTotalFrames, faceTrackingRanges]
  );

  const coverage = useMemo(
    () => calculateCoveragePercentage(safeTotalFrames, faceTrackingRanges),
    [safeTotalFrames, faceTrackingRanges]
  );

  // Sort scene cuts for quick binary/linear seek
  const sortedSceneCuts = useMemo(
    () => [...sceneCuts].sort((a, b) => a - b),
    [sceneCuts]
  );

  // ── Frame & Coordinate Translation ──────────────────────────────────────
  const frameToPercent = useCallback(
    (frame: number) => {
      return (Math.max(0, Math.min(safeTotalFrames, frame)) / safeTotalFrames) * 100;
    },
    [safeTotalFrames]
  );

  const clientXToFrame = useCallback(
    (clientX: number): number => {
      if (!containerRef.current) return 0;
      const rect = containerRef.current.getBoundingClientRect();
      const relativeX = Math.max(0, Math.min(rect.width, clientX - rect.left));
      const ratio = rect.width > 0 ? relativeX / rect.width : 0;
      return Math.round(ratio * safeTotalFrames);
    },
    [safeTotalFrames]
  );

  // ── Pointer Drag Scrubbing Listeners ────────────────────────────────────
  const handlePointerDown = (e: React.PointerEvent, mode: DragMode) => {
    e.preventDefault();
    e.stopPropagation();
    dragModeRef.current = mode;
    dragStartXRef.current = e.clientX;
    dragStartRangeRef.current = { start: startFrame, end: endFrame };

    (e.target as HTMLElement).setPointerCapture(e.pointerId);

    if (mode === 'playhead') {
      const targetFrame = clientXToFrame(e.clientX);
      onSeek(targetFrame);
    }
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    // 1. If actively dragging: update position smoothly
    if (dragModeRef.current !== 'none') {
      const mode = dragModeRef.current;
      const currentTargetFrame = clientXToFrame(e.clientX);

      if (mode === 'playhead') {
        onSeek(currentTargetFrame);
      } else if (mode === 'startHandle') {
        const clampedStart = Math.max(0, Math.min(endFrame - 1, currentTargetFrame));
        setStartFrame(clampedStart);
        onRangeChange?.(clampedStart, endFrame);
      } else if (mode === 'endHandle') {
        const clampedEnd = Math.min(safeTotalFrames, Math.max(startFrame + 1, currentTargetFrame));
        setEndFrame(clampedEnd);
        onRangeChange?.(startFrame, clampedEnd);
      } else if (mode === 'rangeSpan') {
        if (!containerRef.current) return;
        const rect = containerRef.current.getBoundingClientRect();
        const deltaX = e.clientX - dragStartXRef.current;
        const deltaFrames = Math.round((deltaX / rect.width) * safeTotalFrames);
        const spanDuration = dragStartRangeRef.current.end - dragStartRangeRef.current.start;

        let newStart = dragStartRangeRef.current.start + deltaFrames;
        let newEnd = newStart + spanDuration;

        if (newStart < 0) {
          newStart = 0;
          newEnd = spanDuration;
        } else if (newEnd > safeTotalFrames) {
          newEnd = safeTotalFrames;
          newStart = safeTotalFrames - spanDuration;
        }

        setStartFrame(newStart);
        setEndFrame(newEnd);
        onRangeChange?.(newStart, newEnd);
      }
    }

    // 2. Throttled hover calculation for tooltip and preview
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const relX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
      setHoverPositionX(relX);
      const hFrame = Math.max(0, Math.min(safeTotalFrames, Math.round((relX / rect.width) * safeTotalFrames)));
      setHoverFrame(hFrame);

      // Seek offscreen video preview if available
      if (videoUrl && offscreenVideoRef.current) {
        const targetSeconds = hFrame / safeFps;
        if (Math.abs(offscreenVideoRef.current.currentTime - targetSeconds) > 0.2) {
          offscreenVideoRef.current.currentTime = targetSeconds;
        }
      }
    }
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    if (dragModeRef.current !== 'none') {
      dragModeRef.current = 'none';
      try {
        (e.target as HTMLElement).releasePointerCapture(e.pointerId);
      } catch {
        // Pointer capture may have already been released
      }
    }
  };

  // ── Track Click Seeking ────────────────────────────────────────────────
  const handleTrackClick = (e: React.MouseEvent) => {
    if (dragModeRef.current === 'none') {
      const targetFrame = clientXToFrame(e.clientX);
      onSeek(targetFrame);
    }
  };

  // ── Scene Cut Navigation Shortcuts ─────────────────────────────────────
  const jumpToPreviousCut = () => {
    if (sortedSceneCuts.length === 0) return;
    const prevCuts = sortedSceneCuts.filter((cut) => cut < currentFrame - 1);
    if (prevCuts.length > 0) {
      onSeek(prevCuts[prevCuts.length - 1]);
    } else {
      onSeek(0);
    }
  };

  const jumpToNextCut = () => {
    if (sortedSceneCuts.length === 0) return;
    const nextCut = sortedSceneCuts.find((cut) => cut > currentFrame + 1);
    if (nextCut !== undefined) {
      onSeek(nextCut);
    } else {
      onSeek(safeTotalFrames);
    }
  };

  const stepFrame = (delta: number) => {
    const next = Math.max(0, Math.min(safeTotalFrames, currentFrame + delta));
    onSeek(next);
  };

  const handleResetRange = () => {
    setStartFrame(0);
    setEndFrame(safeTotalFrames);
    onRangeChange?.(0, safeTotalFrames);
  };

  // ── Playback Simulation Loop ───────────────────────────────────────────
  useEffect(() => {
    if (!isPlaying) return;
    const intervalMs = Math.round(1000 / safeFps);
    const timer = window.setInterval(() => {
      onSeek((prev) => {
        const next = prev + 1;
        if (next > endFrame) {
          return startFrame;
        }
        return next;
      });
    }, intervalMs);

    return () => clearInterval(timer);
  }, [endFrame, isPlaying, onSeek, safeFps, startFrame]);

  // Video Frame Preview Extraction for Hover Tooltip
  useEffect(() => {
    if (!videoUrl) return;
    const video = document.createElement('video');
    video.src = videoUrl;
    video.muted = true;
    video.crossOrigin = 'anonymous';
    offscreenVideoRef.current = video;

    const handleSeeked = () => {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = 160;
        canvas.height = 90;
        const ctx = canvas.getContext('2d');
        if (ctx && video.videoWidth > 0) {
          ctx.drawImage(video, 0, 0, 160, 90);
          setVideoPreviewUrl(canvas.toDataURL('image/jpeg', 0.6));
        }
      } catch {
        // Canvas extraction fallback
      }
    };

    video.addEventListener('seeked', handleSeeked);
    return () => {
      video.removeEventListener('seeked', handleSeeked);
      video.src = '';
    };
  }, [videoUrl]);

  // Determine active hover properties
  const isHoverOnCut = hoverFrame !== null && sortedSceneCuts.some((cut) => Math.abs(cut - hoverFrame) <= 2);
  const hoverTrackingMatch = hoverFrame !== null ? faceTrackingRanges.find((r) => hoverFrame >= r.start && hoverFrame <= r.end) : null;

  return (
    <div
      className={`select-none rounded-2xl border border-white/10 bg-neutral-950/90 p-5 text-white shadow-2xl backdrop-blur-xl transition-all ${className}`}
    >
      {/* ── Transport Bar & Header Metadata ──────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-4">
        {/* Left: Playhead Transport Controls */}
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={jumpToPreviousCut}
            title="Jump to Previous Scene Cut"
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-white/70 transition-all hover:bg-white/10 hover:text-white active:scale-95"
          >
            <SkipBack className="h-3.5 w-3.5" />
          </button>

          <button
            type="button"
            onClick={() => stepFrame(-1)}
            title="Step Backward 1 Frame (Left Arrow)"
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-white/70 transition-all hover:bg-white/10 hover:text-white active:scale-95"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>

          <button
            type="button"
            onClick={() => setIsPlaying(!isPlaying)}
            title={isPlaying ? 'Pause' : 'Play Timeline'}
            className="flex h-9 w-10 items-center justify-center rounded-xl bg-gradient-to-r from-emerald-500 to-teal-600 font-bold text-white shadow-lg shadow-emerald-950/40 transition-all hover:from-emerald-400 hover:to-teal-500 active:scale-95"
          >
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4 ml-0.5" />}
          </button>

          <button
            type="button"
            onClick={() => stepFrame(1)}
            title="Step Forward 1 Frame (Right Arrow)"
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-white/70 transition-all hover:bg-white/10 hover:text-white active:scale-95"
          >
            <ChevronRight className="h-4 w-4" />
          </button>

          <button
            type="button"
            onClick={jumpToNextCut}
            title="Jump to Next Scene Cut"
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-white/70 transition-all hover:bg-white/10 hover:text-white active:scale-95"
          >
            <SkipForward className="h-3.5 w-3.5" />
          </button>

          {/* Current Frame Counter & Timecode */}
          <div className="ml-2 flex items-baseline gap-2 rounded-xl border border-white/10 bg-neutral-900/80 px-3 py-1.5 font-mono">
            <span className="text-sm font-black tracking-tight text-emerald-400">
              {frameToTimecode(currentFrame, safeFps)}
            </span>
            <span className="text-xs text-white/50">
              Frame <strong className="text-white">{currentFrame}</strong> / {safeTotalFrames}
            </span>
          </div>
        </div>

        {/* Right: Render Range In/Out Stats & Reset */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs font-mono text-white/70">
            <span className="rounded bg-white/5 px-2 py-1 border border-white/10">
              In: <span className="font-semibold text-cyan-300">{frameToTimecode(startFrame, safeFps)}</span> [{startFrame}]
            </span>
            <span className="text-white/30">&bull;</span>
            <span className="rounded bg-white/5 px-2 py-1 border border-white/10">
              Out: <span className="font-semibold text-cyan-300">{frameToTimecode(endFrame, safeFps)}</span> [{endFrame}]
            </span>
            <span className="text-white/30">&bull;</span>
            <span className="text-emerald-300 font-semibold">
              Span: {endFrame - startFrame} frames ({((endFrame - startFrame) / safeFps).toFixed(1)}s)
            </span>
          </div>

          <button
            type="button"
            onClick={handleResetRange}
            title="Reset Trimmer to Entire Clip"
            className="flex items-center gap-1 rounded-xl border border-white/10 bg-white/5 px-2.5 py-1 text-xs font-medium text-white/70 transition-all hover:bg-white/10 hover:text-white"
          >
            <RotateCcw className="h-3 w-3" />
            <span>Reset Span</span>
          </button>
        </div>
      </div>

      {/* ── Metadata Badges Bar ─────────────────────────────────────────── */}
      <div className="my-3 flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-amber-400 font-medium">
            <Scissors className="h-3.5 w-3.5" />
            <span>{sortedSceneCuts.length} Scene Cuts Detected</span>
          </div>

          <div className="flex items-center gap-1.5">
            <Film className="h-3.5 w-3.5 text-white/40" />
            <span className="font-mono text-white/50">{safeFps} FPS</span>
          </div>
        </div>

        {/* Face Tracking Quality Indicator */}
        <div className="flex items-center gap-2 font-mono text-[11px]">
          <span className="flex items-center gap-1 text-emerald-400">
            <CheckCircle2 className="h-3 w-3" />
            {coverage.trackedPercent}% Tracked
          </span>
          {coverage.dropoutPercent > 0 && (
            <span className="flex items-center gap-1 text-rose-400">
              <AlertTriangle className="h-3 w-3" />
              {coverage.dropoutPercent}% Dropout
            </span>
          )}
        </div>
      </div>

      {/* ── Main Multi-Track Stack Container ─────────────────────────────── */}
      <div
        ref={containerRef}
        onPointerDown={(e) => handlePointerDown(e, 'playhead')}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onMouseEnter={() => setIsHovering(true)}
        onMouseLeave={() => {
          setIsHovering(false);
          setHoverFrame(null);
        }}
        onClick={handleTrackClick}
        className="relative mt-2 flex flex-col gap-1.5 cursor-crosshair select-none rounded-xl border border-white/15 bg-neutral-900/90 p-3 pt-6 shadow-inner"
      >
        {/* ── SMPTE Time Ruler Ticks ─────────────────────────────────────── */}
        <div className="absolute inset-x-3 top-1 flex h-4 items-end justify-between pointer-events-none opacity-40">
          {[0, 0.25, 0.5, 0.75, 1.0].map((ratio) => {
            const f = Math.round(ratio * safeTotalFrames);
            return (
              <div key={ratio} className="flex flex-col items-center">
                <span className="font-mono text-[9px] text-white/70">
                  {frameToTimecode(f, safeFps)}
                </span>
                <div className="h-1.5 w-px bg-white/30" />
              </div>
            );
          })}
        </div>

        {/* ── Track 1: Scene Cuts ────────────────────────────────────────── */}
        <div className="relative h-6 w-full rounded-lg border border-white/10 bg-neutral-950/70 overflow-hidden">
          <span className="absolute left-2 top-1 text-[9px] font-bold uppercase tracking-wider text-white/30 pointer-events-none">
            Cuts Track
          </span>

          {/* Render Scene Cut Vertical Ticks */}
          {sortedSceneCuts.map((cutFrame, idx) => {
            const leftPct = frameToPercent(cutFrame);
            return (
              <button
                key={idx}
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onSeek(cutFrame);
                }}
                title={`Scene Cut #${idx + 1} at Frame ${cutFrame} (${frameToTimecode(cutFrame, safeFps)})`}
                className="group absolute top-0 bottom-0 z-10 -ml-1 flex w-2 flex-col items-center justify-start transition-transform hover:scale-125 focus:outline-none"
                style={{ left: `${leftPct}%` }}
              >
                {/* Yellow Amber Flag Indicator */}
                <div className="h-2 w-2 rotate-45 rounded-[1px] bg-amber-400 shadow-sm shadow-amber-950 group-hover:bg-amber-300" />
                <div className="w-0.5 flex-1 bg-amber-400/80 group-hover:bg-amber-300" />
              </button>
            );
          })}
        </div>

        {/* ── Track 2: Face Presence Heatmap ─────────────────────────────── */}
        <div className="relative h-8 w-full rounded-lg border border-white/10 bg-neutral-950/80 overflow-hidden">
          <span className="absolute left-2 top-2 z-10 text-[9px] font-bold uppercase tracking-wider text-white/30 pointer-events-none">
            Face Heatmap
          </span>

          {/* Dropout / Missing Face Segments (Rose/Red Dark Blocks) */}
          {dropoutRanges.map((drop, idx) => {
            const startPct = frameToPercent(drop.start);
            const endPct = frameToPercent(drop.end);
            const widthPct = Math.max(0.2, endPct - startPct);

            return (
              <div
                key={`drop_${idx}`}
                className="absolute top-0 bottom-0 bg-rose-950/40 border-r border-l border-rose-500/20"
                style={{ left: `${startPct}%`, width: `${widthPct}%` }}
                title={`Tracking Dropout: Frames ${drop.start} - ${drop.end} (${frameToTimecode(drop.start, safeFps)})`}
              >
                {/* Diagonal subtle warning hatch */}
                <div className="h-full w-full opacity-20 bg-[repeating-linear-gradient(45deg,#f43f5e,#f43f5e_2px,transparent_2px,transparent_6px)]" />
              </div>
            );
          })}

          {/* Actively Tracked Face Segments (Green/Emerald Blocks) */}
          {faceTrackingRanges.map((range, idx) => {
            const startPct = frameToPercent(range.start);
            const endPct = frameToPercent(range.end);
            const widthPct = Math.max(0.3, endPct - startPct);

            return (
              <div
                key={`track_${idx}`}
                className="group/segment absolute top-0.5 bottom-0.5 rounded-[3px] bg-gradient-to-r from-emerald-500 to-teal-500 shadow-sm shadow-emerald-950/40 transition-all hover:brightness-125"
                style={{ left: `${startPct}%`, width: `${widthPct}%` }}
                title={`Face Locked: ID ${range.faceId} (Frames ${range.start} - ${range.end})`}
              >
                <div className="flex h-full items-center px-1">
                  <span className="truncate text-[8px] font-mono font-extrabold text-neutral-950 opacity-90">
                    {range.faceId}
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* ── Dual Range Trimmer Handles & Active Region Overlay ─────────── */}
        {/* Dimmed Left Mask (Trimmed out before Start Frame) */}
        <div
          className="pointer-events-none absolute inset-y-0 left-0 bg-black/65 backdrop-blur-[1px] border-r border-cyan-400/40"
          style={{ width: `${frameToPercent(startFrame)}%` }}
        />

        {/* Dimmed Right Mask (Trimmed out after End Frame) */}
        <div
          className="pointer-events-none absolute inset-y-0 right-0 bg-black/65 backdrop-blur-[1px] border-l border-cyan-400/40"
          style={{ width: `${100 - frameToPercent(endFrame)}%` }}
        />

        {/* Active Trimmer Middle Span (Draggable whole region) */}
        <div
          onPointerDown={(e) => handlePointerDown(e, 'rangeSpan')}
          className="absolute inset-y-0 cursor-grab border-t-2 border-b-2 border-cyan-400/60 bg-cyan-500/10 active:cursor-grabbing hover:bg-cyan-500/15 transition-colors"
          style={{
            left: `${frameToPercent(startFrame)}%`,
            width: `${Math.max(0, frameToPercent(endFrame) - frameToPercent(startFrame))}%`,
          }}
          title="Drag to shift entire render range"
        />

        {/* In-Point Left Handle ('[') */}
        <div
          onPointerDown={(e) => handlePointerDown(e, 'startHandle')}
          className="group absolute -bottom-1 -top-1 z-30 flex -ml-3 w-6 cursor-ew-resize flex-col items-center justify-center focus:outline-none"
          style={{ left: `${frameToPercent(startFrame)}%` }}
        >
          {/* Floating Timecode Pill above handle */}
          <div className="absolute -top-6 rounded bg-neutral-900 px-1.5 py-0.5 font-mono text-[9px] font-bold text-cyan-300 shadow-md border border-cyan-500/40">
            [{frameToTimecode(startFrame, safeFps)}]
          </div>
          {/* Grip Handle */}
          <div className="flex h-full w-2.5 flex-col items-center justify-center rounded-l-md border border-cyan-400 bg-cyan-500 text-[10px] font-bold text-black shadow-lg shadow-cyan-950/60 group-hover:scale-105">
            [
          </div>
        </div>

        {/* Out-Point Right Handle (']') */}
        <div
          onPointerDown={(e) => handlePointerDown(e, 'endHandle')}
          className="group absolute -bottom-1 -top-1 z-30 flex -ml-3 w-6 cursor-ew-resize flex-col items-center justify-center focus:outline-none"
          style={{ left: `${frameToPercent(endFrame)}%` }}
        >
          {/* Floating Timecode Pill above handle */}
          <div className="absolute -top-6 rounded bg-neutral-900 px-1.5 py-0.5 font-mono text-[9px] font-bold text-cyan-300 shadow-md border border-cyan-500/40">
            [{frameToTimecode(endFrame, safeFps)}]
          </div>
          {/* Grip Handle */}
          <div className="flex h-full w-2.5 flex-col items-center justify-center rounded-r-md border border-cyan-400 bg-cyan-500 text-[10px] font-bold text-black shadow-lg shadow-cyan-950/60 group-hover:scale-105">
            ]
          </div>
        </div>

        {/* ── Playhead Line & Scrubber Marker ────────────────────────────── */}
        <div
          className="pointer-events-none absolute inset-y-0 z-40 flex -ml-1.5 w-3 flex-col items-center"
          style={{ left: `${frameToPercent(currentFrame)}%` }}
        >
          {/* Playhead Arrow / Scrubber Head */}
          <div className="h-3 w-3 rotate-45 rounded-[2px] bg-white shadow-md shadow-neutral-950 border border-neutral-300" />
          <div className="w-0.5 flex-1 bg-white shadow-sm shadow-black" />
        </div>

        {/* ── Hover Scrubber Tooltip & Video Thumbnail ───────────────────── */}
        {isHovering && hoverFrame !== null && dragModeRef.current === 'none' && (
          <div
            className="pointer-events-none absolute -top-32 z-50 flex -translate-x-1/2 flex-col items-center transition-all duration-75"
            style={{
              left: `${Math.max(80, Math.min((containerRef.current?.getBoundingClientRect().width || 400) - 80, hoverPositionX))}px`,
            }}
          >
            <div className="flex flex-col items-center rounded-xl border border-white/20 bg-neutral-900/95 p-1.5 shadow-2xl backdrop-blur-xl">
              {/* Thumbnail Frame Preview */}
              <div className="relative aspect-video w-32 overflow-hidden rounded-lg bg-neutral-950 border border-white/10 flex items-center justify-center">
                {videoPreviewUrl || (getThumbnailUrl && getThumbnailUrl(hoverFrame)) ? (
                  <img
                    src={videoPreviewUrl || getThumbnailUrl?.(hoverFrame)}
                    alt={`Frame ${hoverFrame}`}
                    className="h-full w-full object-cover"
                  />
                ) : (
                  <div className="flex flex-col items-center justify-center p-2 text-center text-white/40">
                    <Eye className="h-4 w-4 mb-1" />
                    <span className="font-mono text-[9px]">Frame {hoverFrame}</span>
                  </div>
                )}

                {/* Overlaid Target Face Bounding Box indicator if tracked */}
                {hoverTrackingMatch && (
                  <div className="absolute inset-2 rounded border-2 border-emerald-400/80 pointer-events-none animate-pulse">
                    <span className="absolute -top-2 left-1 bg-emerald-500 text-black text-[8px] font-extrabold px-1 rounded">
                      {hoverTrackingMatch.faceId}
                    </span>
                  </div>
                )}
              </div>

              {/* Hover Stats */}
              <div className="mt-1 flex flex-col items-center">
                <span className="font-mono text-[10px] font-bold text-white">
                  {frameToTimecode(hoverFrame, safeFps)}
                </span>
                <span className="font-mono text-[9px] text-white/50">
                  Frame {hoverFrame}
                </span>

                {isHoverOnCut && (
                  <span className="mt-0.5 rounded bg-amber-500/20 px-1.5 py-0.2 font-mono text-[8px] font-bold text-amber-300 border border-amber-500/30">
                    Scene Cut
                  </span>
                )}
              </div>
            </div>
            {/* Tooltip caret down */}
            <div className="h-1.5 w-3 bg-neutral-900 rotate-45 -mt-1 border-r border-b border-white/20" />
          </div>
        )}
      </div>

      {/* ── Keyboard Shortcuts Footer ────────────────────────────────────── */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-white/5 pt-3 text-[11px] text-white/40">
        <div className="flex items-center gap-3">
          <span>
            <kbd className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-[10px] text-white/80">Space</kbd> Play / Pause
          </span>
          <span>
            <kbd className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-[10px] text-white/80">&larr;</kbd>{' '}
            <kbd className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-[10px] text-white/80">&rarr;</kbd> Step Frame
          </span>
          <span>
            <kbd className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-[10px] text-white/80">[</kbd>{' '}
            <kbd className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-[10px] text-white/80">]</kbd> Drag Trimmer Handles
          </span>
        </div>

        <div className="flex items-center gap-1 text-[10px]">
          <Sparkles className="h-3 w-3 text-cyan-400" />
          <span>Non-Destructive In/Out Trimming</span>
        </div>
      </div>
    </div>
  );
}

export default TimelineTrackBar;
