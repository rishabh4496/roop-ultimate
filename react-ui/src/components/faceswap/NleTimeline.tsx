/**
 * NleTimeline.tsx: the multi-lane, frame-accurate Non-Linear Editing timeline.
 *
 *   toolbar        step badges (1 Pre-Scan / 2 Range Lock / 3 Render), snap, timecode + frame readout, zoom
 *   transport deck jump / marker / frame / play / loop / speed / J-K-L (when the host gives it playback)
 *   overview       the whole clip with a draggable, resizable window
 *   ruler          canvas: SMPTE timecode + exact frame number, tick density follows pixels
 *   V1 video       adaptive filmstrip, In / Out trim handles, dimmed outside region, duration readout
 *   face coverage  emerald / amber / rose heatmap, click a problem run to jump to it
 *   swap segments  blocks for range locks, queue segments and per-range overrides + draggable markers
 *   A1 audio       peaks waveform
 *   KF             parameter keyframes
 *   face tracks    one lane per ByteTrack id
 *
 * The playhead is one cyan line over every lane (`--accent-cyan`); its cap is a scrub handle, and a
 * floating card with a still, the frame number and the timecode follows it while scrubbing.
 *
 * Geometry: every pixel <-> frame mapping goes through nle/nleGeometry.ts against ONE element (the lane
 * area overlay), so what is drawn and what a click means cannot disagree. 100% offline: no CDN, font or
 * remote URL. Keys: Space / arrows / Home / End / I / O / M / J K L belong to useWorkstationShortcuts (or
 * the Extras adapter); this component owns only Delete for a selected keyframe.
 */

import React, {
  useState,
  useMemo,
  useCallback,
  useRef,
  useEffect,
  useReducer,
  type KeyboardEvent,
} from 'react';

import { API } from '../../api';
import {
  type RangeLockInterval,
  type KeyframePoint,
  type PtsTableRawInput,
  type PtsTableEntry,
  PtsFrameMap,
  createPtsFrameMap,
  clampFrameIdx,
  frameToTimecode,
  formatSeconds,
  createRangeLockInterval,
  clampRangeLockInterval,
} from '../../utils/ptsFrameMap';

import {
  type ChapterMarker,
  type SnapTarget,
  snapToFrame,
  calculateZoomToCursor,
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

import { LANE_HEADER_PX, clientXToFrame, frameToLaneX } from './nle/nleGeometry';
import { RulerCanvas, RULER_HEIGHT } from './nle/RulerCanvas';
import { WaveformCanvas } from './nle/WaveformCanvas';
import { FilmstripCanvas } from './nle/FilmstripCanvas';
import { FaceHealthLane } from './nle/FaceHealthLane';
import { SegmentLane } from './nle/SegmentLane';
import { KeyframeLane } from './nle/KeyframeLane';
import { FaceTrackLane } from './nle/FaceTrackLane';
import { LaneHeader } from './nle/LaneHeader';
import { TransportDeck } from './nle/TransportDeck';
import { MinimapNavigator, type MinimapTrack } from './nle/MinimapNavigator';
import { ScrubTooltip } from './nle/ScrubTooltip';
import { MarkerPopover } from './nle/MarkerPopover';
import { useElementSize } from './nle/useElementSize';
import { useScrubThumbnail, serverFrameFetcher } from './nle/useScrubThumbnail';
import { useFrameGrabber, grabberFetcher, useClientFilmstrip, useWebAudioWaveform } from './nle/clientMedia';
import { healthSegments, summarizeHealth, problemFrames } from './nle/faceHealth';
import { formatFrameLabel } from './nle/rulerModel';
import { neighbourFrame, loopRegion, regionLength, formatDuration } from './nle/transportModel';
import { wheelMinimap, spanToZoom, zoomToSpan, reconcileView } from './nle/minimapModel';
import { COLLAPSED_PX, LANE_HEIGHTS, LANE_LABELS, laneBodyPx, loadLaneStates, reduceLanes, saveLaneStates, type LaneId } from './nle/laneLayout';
import type { SegmentBlock } from './nle/segmentModel';
import { DEFAULT_MARKER_COLOR } from './nle/markerModel';

// Re-export core interval, keyframe, and marker interfaces
export type { RangeLockInterval, KeyframePoint, ChapterMarker, NleKeyframe, TimelineParamType, TimelineCurveType, SegmentBlock };

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
  /** frames this tick stands for (a decimated pose series); the coverage lane paints its neighbourhood */
  span?: number;
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

/** A clip held in the browser (Extras): stills and peaks are built client-side, not requested. */
export interface NleMediaSource {
  url: string;
  file: File;
}

export interface NleTimelineProps {
  // Current playback position
  frame?: number;
  currentFrame?: number;
  currentTime?: number;
  maxFrames: number;
  fps?: number;
  /** the rate was guessed (no metadata), so timecodes are approximate */
  fpsAssumed?: boolean;
  isPlaying?: boolean;

  // Media targeting (for waveform & filmstrip)
  targetIndex?: number | null;
  mediaId?: string | null;
  targetKey?: string;
  /** build stills and peaks in the browser from this local file instead of asking the server */
  mediaSource?: NleMediaSource | null;

  // PTS table & VFR exact mapping
  ptsTable?: PtsTableRawInput | PtsFrameMap | PtsTableEntry[];
  ptsFrameMap?: PtsFrameMap;

  // Tracks, warnings, and suggested assignments from Step 1 Pre-Scan
  trackLanes?: ByteTrackLane[];
  facesets?: Array<{ id: string; name: string; path: string }>;
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

  // Chapter markers: with `onAddMarker` the host owns them (and persists them)
  markers?: ChapterMarker[];

  // Identity / parameter keyframes (exact integer frame_idx)
  keyframes?: NleKeyframe[] | KeyframePoint[];
  activeKeyframeParam?: TimelineParamType;

  // Swap-rule / segment blocks for the segment lane (0-based inclusive frames)
  segmentBlocks?: SegmentBlock[];
  onSegmentClick?: (block: SegmentBlock) => void;
  /** the lane's name: "Swap rules" in the workstation, "Markers" where there are no swap rules (Extras) */
  segmentLaneTitle?: string;
  /** shown beside the In-Out duration when the region does not drive anything but the preview */
  regionNote?: string;
  /**
   * Keys for hosts with no global shortcut layer (Extras). Called first with a key pressed while the
   * timeline has focus; return true when handled. FaceSwap leaves it unset: useWorkstationShortcuts owns
   * Space / arrows / I / O / M / J K L there, and a second handler would double-fire (M would add then toggle off).
   */
  hostKeys?: (e: KeyboardEvent<HTMLDivElement>) => boolean;

  // Transport deck: shown when the host supplies `onTogglePlay`
  playbackRate?: number;
  isLooping?: boolean;
  onTogglePlay?: () => void;
  onStep?: (delta: number) => void;
  onPlaybackRateChange?: (rate: number) => void;
  onToggleLoop?: () => void;
  onShuttle?: (key: 'j' | 'k' | 'l') => void;
  shuttleLabel?: string;
  shuttleDir?: number;

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
  onRangeLockClick?: () => void;
  onRenderClick?: () => void;
  onBatchClick?: () => void;
  stagedJobsCount?: number;
  batchStrategy?: string;

  // Step indicators & UI flags
  isPreScanActive?: boolean;
  isRangeLockActive?: boolean;
  isRenderActive?: boolean;
  /** hide the 1 / 2 / 3 pipeline badges (Extras has no swap pipeline) */
  showPipelineSteps?: boolean;
  showWarningTicks?: boolean;
  showSuggestedFsz?: boolean;
  showFilmstripLane?: boolean;
  showWaveformLane?: boolean;
  showKeyframeLane?: boolean;
  showHealthLane?: boolean;
  showSegmentLane?: boolean;
  showFaceTrackLanes?: boolean;

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

// ── helpers ────────────────────────────────────────────────────────────────────

const SNAP_RADIUS_PX = 6;
const WHEEL_ZOOM_RATE = 0.002;
const clampN = (n: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, n));

/** localStorage can throw on access (blocked storage); the layout simply does not persist then. */
function safeStorage(): Storage | null {
  try {
    return typeof window !== 'undefined' ? window.localStorage : null;
  } catch {
    return null;
  }
}

/** Returns the previous reference while `key` is unchanged, so an unstable caller array does not bust memos. */
function useStableBy<T>(value: T, key: string): T {
  const ref = useRef<{ key: string; value: T }>({ key, value });
  if (ref.current.key !== key) ref.current = { key, value };
  return ref.current.value;
}

const warningsKey = (ticks: ReadonlyArray<WarningTick>) => {
  let h = ticks.length;
  for (const t of ticks) h = (h * 31 + t.frame * 7 + (t.type === 'pose' ? 1 : t.type === 'occlusion' ? 2 : 3) + t.trackId + (t.span ?? 1) * 13) | 0;
  return `${ticks.length}:${h}`;
};
// Every field a lane RENDERS must be in the key, not just the ranges the coverage heatmap reads: a key of ids +
// ranges alone made a faceset re-assignment or a swap mute (same ranges) look unchanged, so the lane kept
// showing the old value.
export const tracksKey = (tracks: ReadonlyArray<ByteTrackLane>) =>
  tracks
    .map((t) => [t.trackId, t.label, t.active, t.color, t.facesetId, t.facesetPath, t.suggestedFsz, t.sourceIndex, t.totalFrames, t.via, t.ranges.join(',')].join('~'))
    .join('|');

interface LaneRowProps {
  id: LaneId;
  testId?: string;
  badge: string;
  badgeClass: string;
  title: string;
  state: ReturnType<typeof loadLaneStates>[LaneId];
  onToggleHidden?: () => void;
  onToggleLocked?: () => void;
  onCycleHeight?: () => void;
  extra?: React.ReactNode;
  status?: React.ReactNode;
  titleVisible?: boolean;
  children?: React.ReactNode;
  heightPx: number;
  className?: string;
}

/** One `[header | lane area]` row. A hidden lane keeps its header strip but renders no body. */
function LaneRow({ id, testId, badge, badgeClass, title, state, onToggleHidden, onToggleLocked, onCycleHeight, extra, status, children, heightPx, className = '' }: LaneRowProps) {
  return (
    <div className={`flex w-full border-b border-white/5 ${className}`} data-testid={testId} data-lane={id} data-hidden={state.hidden ? '1' : '0'} data-height={state.height} data-locked={state.locked ? '1' : '0'} style={{ height: heightPx }}>
      <LaneHeader
        laneId={id}
        badge={badge}
        badgeClass={badgeClass}
        title={title}
        state={state}
        onToggleHidden={onToggleHidden}
        onToggleLocked={onToggleLocked}
        onCycleHeight={onCycleHeight}
        extra={extra}
        status={status}
        width={LANE_HEADER_PX}
        height={heightPx}
      />
      <div className="relative flex-1 overflow-hidden bg-black/25" data-testid={testId ? `${testId}-body` : undefined}>
        {state.hidden ? null : children}
      </div>
    </div>
  );
}

// ── Component Implementation ──────────────────────────────────────────────────

export const NleTimeline: React.FC<NleTimelineProps> = ({
  frame: frameProp,
  currentFrame: currentFrameProp,
  currentTime,
  maxFrames: rawMaxFrames,
  fps = 30,
  fpsAssumed = false,
  isPlaying = false,
  targetIndex = null,
  mediaId = null,
  targetKey = '',
  mediaSource = null,
  ptsTable,
  ptsFrameMap: customPtsFrameMap,
  trackLanes: trackLanesProp = [],
  warningTicks: warningTicksProp = [],
  suggestedAssignments = [],
  rangeLockIntervals = [],
  selectedTrackId = null,
  trimIn: trimInProp,
  trimOut: trimOutProp,
  markers: markersProp = [],
  keyframes: keyframesProp = [],
  activeKeyframeParam: activeParamProp = 'Swap Opacity',
  facesets = [],
  segmentBlocks = [],
  onSegmentClick,
  segmentLaneTitle = 'Swap rules',
  regionNote = '',
  hostKeys,
  playbackRate = 1,
  isLooping = false,
  onTogglePlay,
  onStep,
  onPlaybackRateChange,
  onToggleLoop,
  onShuttle,
  shuttleLabel = '',
  shuttleDir = 0,
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
  onRangeLockClick,
  onRenderClick,
  onBatchClick,
  stagedJobsCount = 0,
  isPreScanActive = false,
  isRangeLockActive = false,
  isRenderActive = false,
  showPipelineSteps = true,
  showWarningTicks = true,
  showSuggestedFsz = true,
  showFilmstripLane = true,
  showWaveformLane = true,
  showKeyframeLane = true,
  showHealthLane = true,
  showSegmentLane = true,
  showFaceTrackLanes = true,
  className = '',
  style,
}) => {
  const maxFrames = Math.max(1, Math.round(rawMaxFrames));
  const lastFrame = maxFrames - 1;
  const trackLanes = useStableBy(trackLanesProp, tracksKey(trackLanesProp));
  const warningTicks = useStableBy(warningTicksProp, warningsKey(warningTicksProp));
  const clientMode = !!mediaSource;

  // Initialize or memoize PTS Frame Map
  const frameMap = useMemo(() => {
    if (customPtsFrameMap instanceof PtsFrameMap) return customPtsFrameMap;
    if (ptsTable) return createPtsFrameMap(ptsTable, fps);
    return createPtsFrameMap(null, fps);
  }, [customPtsFrameMap, ptsTable, fps]);

  // Resolve current active integer frame index
  const activeFrame = useMemo(() => {
    if (typeof frameProp === 'number' && Number.isFinite(frameProp)) return clampFrameIdx(frameProp, 0, maxFrames);
    if (typeof currentFrameProp === 'number' && Number.isFinite(currentFrameProp)) return clampFrameIdx(currentFrameProp, 0, maxFrames);
    if (typeof currentTime === 'number' && Number.isFinite(currentTime)) return frameMap.resolveFrameFromTime(currentTime);
    return 0;
  }, [frameProp, currentFrameProp, currentTime, maxFrames, frameMap]);

  // Trim In / Out points (exact integer frames)
  const trimIn = useMemo(() => clampFrameIdx(typeof trimInProp === 'number' ? trimInProp : 0, 0, maxFrames), [trimInProp, maxFrames]);
  const trimOut = useMemo(() => clampFrameIdx(typeof trimOutProp === 'number' ? trimOutProp : maxFrames - 1, trimIn, maxFrames), [trimOutProp, trimIn, maxFrames]);
  const region = useMemo(() => loopRegion(trimIn, trimOut, maxFrames), [trimIn, trimOut, maxFrames]);
  const regionIsWhole = region.in === 0 && region.out === lastFrame;

  // ── Lane state (hide / lock / height), persisted per viewer ──────────────────
  const [lanes, dispatchLanes] = useReducer(reduceLanes, undefined, () => loadLaneStates(safeStorage()));
  useEffect(() => { saveLaneStates(safeStorage(), lanes); }, [lanes]);
  const laneActions = (id: LaneId) => ({
    onToggleHidden: () => dispatchLanes({ type: 'toggleHidden', id }),
    onToggleLocked: () => dispatchLanes({ type: 'toggleLocked', id }),
    onCycleHeight: () => dispatchLanes({ type: 'cycleHeight', id }),
  });

  // ── Chapter markers ──────────────────────────────────────────────────────────
  // Host-owned when it hands an `onAddMarker`; otherwise an in-memory list so the component stands alone.
  const [internalMarkers, setInternalMarkers] = useState<ChapterMarker[]>([]);
  const markersControlled = !!onAddMarker;
  const markers = markersControlled ? markersProp : internalMarkers;
  const [popover, setPopover] = useState<{ id: string; anchor: { left: number; top: number } } | null>(null);
  const popoverMarker = popover ? markers.find((m) => m.id === popover.id) || null : null;

  const addMarkerAt = useCallback((f: number) => {
    if (markers.some((m) => m.frame === f)) return;
    const marker: ChapterMarker = { id: `marker_${f + 1}`, frame: f, label: 'Marker', color: DEFAULT_MARKER_COLOR, note: '', createdAt: Date.now() };
    if (markersControlled) onAddMarker?.(marker);
    else setInternalMarkers((prev) => [...prev, { ...marker, id: `marker_${f}_${Date.now()}` }].sort((a, b) => a.frame - b.frame));
  }, [markers, markersControlled, onAddMarker]);

  const updateMarker = useCallback((marker: ChapterMarker) => {
    if (markersControlled) onUpdateMarker?.(marker);
    else setInternalMarkers((prev) => prev.map((m) => (m.id === marker.id ? marker : m)).sort((a, b) => a.frame - b.frame));
  }, [markersControlled, onUpdateMarker]);

  const deleteMarker = useCallback((id: string) => {
    if (markersControlled) onDeleteMarker?.(id);
    else setInternalMarkers((prev) => prev.filter((m) => m.id !== id));
    setPopover(null);
  }, [markersControlled, onDeleteMarker]);

  // Active Keyframe Parameter Selection
  const [activeParam, setActiveParam] = useState<TimelineParamType>(activeParamProp);
  useEffect(() => { if (activeParamProp) setActiveParam(activeParamProp); }, [activeParamProp]);

  // Normalized parameter keyframes
  const nleKeyframes = useMemo<NleKeyframe[]>(() => {
    if (!keyframesProp || keyframesProp.length === 0) return [];
    return keyframesProp.map((k: any) => {
      if ('param' in k && 'curve' in k) return k as NleKeyframe;
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
  const activeParamKeyframes = useMemo(() => nleKeyframes.filter((k) => k.param === activeParam), [nleKeyframes, activeParam]);
  const [selectedKeyframeId, setSelectedKeyframeId] = useState<string | null>(null);

  // Snapping
  const [snapEnabled, setSnapEnabled] = useState(initialSnapEnabled);
  const [snapIndicatorFrame, setSnapIndicatorFrame] = useState<number | null>(null);

  // ── Viewport window ──────────────────────────────────────────────────────────
  const [viewRange, setViewRange] = useState<[number, number]>([0, lastFrame]);
  // A window that showed the whole clip keeps showing the whole clip when its length changes (see
  // reconcileView); a different clip always starts fitted.
  const prevLastRef = useRef(lastFrame);
  const clipIdRef = useRef(`${targetKey}|${mediaSource?.url || ''}`);
  useEffect(() => {
    const clipId = `${targetKey}|${mediaSource?.url || ''}`;
    const sameClip = clipIdRef.current === clipId;
    clipIdRef.current = clipId;
    const prevLast = prevLastRef.current;
    prevLastRef.current = lastFrame;
    setViewRange((prev) => (sameClip ? reconcileView(prev, prevLast, lastFrame) : [0, lastFrame]));
  }, [lastFrame, targetKey, mediaSource?.url]);
  const [viewStart, viewEnd] = viewRange;
  const viewSpan = Math.max(1, viewEnd - viewStart);

  // The lane area: one element every pixel <-> frame mapping reads (see nle/nleGeometry.ts).
  const rootRef = useRef<HTMLDivElement>(null);
  const areaRef = useRef<HTMLDivElement>(null);
  const { width: areaWidth } = useElementSize(areaRef);
  const pxPerFrame = areaWidth > 0 ? areaWidth / viewSpan : 1;

  // ── Media: stills + peaks from the server, or built in the browser for a local file ─────────
  const waveServer = useAudioWaveform({
    targetIndex, mediaId, targetKey,
    buckets: Math.min(2000, Math.max(200, Math.round(areaWidth * 1.5))),
    enabled: showWaveformLane && !clientMode && !lanes.a1.hidden,
  });
  const waveClient = useWebAudioWaveform(mediaSource?.file ?? null, showWaveformLane && clientMode && !lanes.a1.hidden);
  const wave = clientMode
    ? { peaks: waveClient.peaks, duration: waveClient.duration, state: waveClient.state, reason: waveClient.reason }
    : { peaks: waveServer.peaks, duration: waveServer.duration, state: waveServer.state, reason: undefined as string | undefined };

  const grabber = useFrameGrabber(mediaSource?.url, fps);
  const filmServer = useFilmstripThumbnails({
    targetIndex, targetKey, viewStart, viewEnd, maxFrames, trackWidth: areaWidth,
    enabled: showFilmstripLane && !clientMode && !lanes.v1.hidden,
  });
  const filmClient = useClientFilmstrip({
    grabber, mediaKey: mediaSource?.url || '', viewStart, viewEnd, maxFrames, trackWidth: areaWidth,
    enabled: showFilmstripLane && clientMode && !lanes.v1.hidden,
  });
  const film = clientMode ? filmClient : filmServer;

  const frameFetcher = useMemo(() => {
    if (clientMode) return grabberFetcher(grabber);
    return targetIndex !== null && targetIndex >= 0 ? serverFrameFetcher(API, targetIndex) : null;
  }, [clientMode, grabber, targetIndex]);

  // ── Face coverage ────────────────────────────────────────────────────────────
  const healthSegs = useMemo(
    () => healthSegments({ maxFrames, trackRanges: trackLanes.flatMap((t) => t.ranges), warnings: warningTicks, scanned: trackLanes.length > 0 }),
    [maxFrames, trackLanes, warningTicks],
  );
  const healthSummary = useMemo(() => summarizeHealth(healthSegs), [healthSegs]);
  const problems = useMemo(() => problemFrames(healthSegs), [healthSegs]);
  const minimapTracks = useMemo<MinimapTrack[]>(
    () => trackLanes.map((t) => ({ trackId: t.trackId, color: getFacesetColor(t.facesetPath || t.suggestedFsz, getTrackColor(t.trackId, t.color)), ranges: t.ranges })),
    [trackLanes],
  );

  // ── Snap targets ─────────────────────────────────────────────────────────────
  const snapTargets = useMemo<SnapTarget[]>(() => {
    if (!snapEnabled) return [];
    const targets: SnapTarget[] = [
      { frame: activeFrame, type: 'playhead', label: 'Playhead' },
      { frame: trimIn, type: 'trim_in', label: 'Trim In' },
      { frame: trimOut, type: 'trim_out', label: 'Trim Out' },
    ];
    markers.forEach((m) => targets.push({ frame: m.frame, type: 'marker', label: m.label }));
    activeParamKeyframes.forEach((k) => targets.push({ frame: k.frame_idx, type: 'keyframe', label: `${k.param} Keyframe` }));
    trackLanes.forEach((t) => t.ranges.forEach(([s, e]) => {
      targets.push({ frame: s, type: 'range_edge', label: `${t.label} Start` });
      targets.push({ frame: e, type: 'range_edge', label: `${t.label} End` });
    }));
    return targets;
  }, [snapEnabled, activeFrame, trimIn, trimOut, markers, activeParamKeyframes, trackLanes]);
  const snapThreshold = Math.max(0, Math.round(SNAP_RADIUS_PX / pxPerFrame));

  // ── Pointer <-> frame ────────────────────────────────────────────────────────
  const dragAreaRef = useRef<{ left: number; width: number } | null>(null);
  const readArea = () => {
    const rect = areaRef.current?.getBoundingClientRect();
    return rect ? { left: rect.left, width: rect.width } : null;
  };
  const xToFrame = useCallback(
    (clientX: number, area: { left: number; width: number } | null = dragAreaRef.current || readArea()) =>
      clientXToFrame(clientX, area, viewStart, viewEnd, maxFrames),
    [viewStart, viewEnd, maxFrames],
  );
  const snapFrame = useCallback((f: number) => snapToFrame(f, snapTargets, snapThreshold), [snapTargets, snapThreshold]);

  // ── Seek (coalesced to one dispatch per animation frame while dragging) ─────────────────────
  const rafSeekRef = useRef<number | null>(null);
  const pendingSeekFrameRef = useRef<number | null>(null);
  const handleSeekFrame = useCallback((target: number, immediate = true) => {
    const exact = clampFrameIdx(target, 0, maxFrames);
    if (immediate) {
      if (rafSeekRef.current !== null) { cancelAnimationFrame(rafSeekRef.current); rafSeekRef.current = null; }
      onSeek?.(exact, frameMap.resolveTimeFromFrame(exact));
      return;
    }
    pendingSeekFrameRef.current = exact;
    if (rafSeekRef.current === null) {
      rafSeekRef.current = requestAnimationFrame(() => {
        rafSeekRef.current = null;
        const f = pendingSeekFrameRef.current;
        pendingSeekFrameRef.current = null;
        if (f !== null) onSeek?.(f, frameMap.resolveTimeFromFrame(f));
      });
    }
  }, [maxFrames, frameMap, onSeek]);
  useEffect(() => () => { if (rafSeekRef.current !== null) cancelAnimationFrame(rafSeekRef.current); }, []);

  // ── Scrubbing + the tooltip that follows it ──────────────────────────────────
  const [scrub, setScrub] = useState<{ frame: number; x: number; top: number } | null>(null);
  const [hoverTip, setHoverTip] = useState<{ frame: number; x: number; top: number } | null>(null);
  const tip = scrub || hoverTip;
  const exactThumb = useScrubThumbnail(!!tip, tip ? tip.frame : null, frameFetcher, `${targetKey}|${mediaSource?.url || ''}`);
  const nearThumb = tip ? film.getThumbnail(tip.frame, 1_000_000) : null;

  const tipAt = (frame: number, area: { left: number; width: number }) => ({
    frame,
    x: area.left + frameToLaneX(frame, viewStart, viewEnd, area.width),
    // the card rises above the RULER (always on screen: it is sticky), not above the timeline's own top,
    // which can be scrolled out of the dock
    top: areaRef.current?.getBoundingClientRect().top ?? 0,
  });

  const handleScrubDown = useCallback((e: React.PointerEvent<HTMLElement>) => {
    if (e.button !== 0) return;
    // preventDefault stops text selection, but it also stops the browser moving focus: without the explicit
    // focus below, focus stays on whatever was clicked before (e.g. the Pro Studio toggle) and the next Space
    // press activates THAT button instead of playing / pausing.
    e.preventDefault();
    rootRef.current?.focus({ preventScroll: true });
    const area = readArea();
    if (!area) return;
    dragAreaRef.current = area;
    const step = (clientX: number, immediate: boolean) => {
      const s = snapFrame(xToFrame(clientX, area));
      setSnapIndicatorFrame(s.snapped ? s.frame : null);
      setScrub(tipAt(s.frame, area));
      handleSeekFrame(s.frame, immediate);
    };
    step(e.clientX, true);
    const move = (ev: PointerEvent) => step(ev.clientX, false);
    const up = () => {
      setScrub(null);
      setSnapIndicatorFrame(null);
      dragAreaRef.current = null;
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', up);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- tipAt/readArea read refs and the view window captured below
  }, [snapFrame, xToFrame, handleSeekFrame, viewStart, viewEnd]);

  const handleRulerHover = (e: React.PointerEvent<HTMLElement>) => {
    if (scrub) return;
    const area = readArea();
    if (!area) return;
    setHoverTip(tipAt(xToFrame(e.clientX, area), area));
  };

  // ── Zoom / pan ───────────────────────────────────────────────────────────────
  const handleZoom = useCallback((factor: number, anchorFraction = 0.5) => {
    setViewRange((prev) => calculateZoomToCursor(prev, factor, anchorFraction, maxFrames, 4));
  }, [maxFrames]);
  const handleResetZoom = useCallback(() => setViewRange([0, lastFrame]), [lastFrame]);

  // Wheel must be NATIVE and non-passive: React's onWheel is registered passive, so preventDefault there
  // is ignored (the page scrolled / browser-zoomed behind the timeline). Two listeners, like Timeline.jsx:
  // `onWheel` stays a pure yield to the app zoom (App.jsx, on window) and `onZoomWheel` claims Ctrl/Cmd
  // for THIS timeline and stops the event so one notch never zooms the track AND the whole UI.
  const wheelCtx = useRef({ viewRange, maxFrames, lastFrame });
  wheelCtx.current = { viewRange, maxFrames, lastFrame };
  useEffect(() => {
    const el = rootRef.current;
    if (!el) return undefined;
    const onWheel = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey) return;
      const W = wheelCtx.current;
      if (W.maxFrames < 2) return;
      const minimap = (e.target as HTMLElement | null)?.closest?.('[data-testid="nle-minimap-area"]') as HTMLElement | null;
      if (minimap) {
        const rect = minimap.getBoundingClientRect();
        if (!rect.width) return;
        e.preventDefault();
        const ratio = clampN((e.clientX - rect.left) / rect.width, 0, 1);
        setViewRange(wheelMinimap(W.viewRange, ratio, Math.exp(clampN(e.deltaY, -240, 240) * WHEEL_ZOOM_RATE), W.maxFrames));
        return;
      }
      // a vertical wheel is the dock's scroll; only a horizontal swipe / Shift+wheel pans the lanes
      if (!(e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY))) return;
      const rect = areaRef.current?.getBoundingClientRect();
      if (!rect || !rect.width) return;
      e.preventDefault();
      const d = e.shiftKey && Math.abs(e.deltaX) < Math.abs(e.deltaY) ? e.deltaY : e.deltaX;
      setViewRange(([s, en]) => {
        const span = en - s;
        const delta = Math.round((d / rect.width) * span);
        const start = clampN(s + delta, 0, Math.max(0, W.lastFrame - span));
        return [start, start + span];
      });
    };
    const onZoomWheel = (e: WheelEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      const W = wheelCtx.current;
      const rect = areaRef.current?.getBoundingClientRect();
      if (W.maxFrames < 2 || !rect || !rect.width) return;
      e.preventDefault();
      e.stopPropagation();
      const fraction = clampN((e.clientX - rect.left) / rect.width, 0, 1);
      const factor = Math.exp(clampN(e.deltaY, -240, 240) * WHEEL_ZOOM_RATE);
      setViewRange((prev) => calculateZoomToCursor(prev, factor, fraction, W.maxFrames, 4));
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    el.addEventListener('wheel', onZoomWheel, { passive: false });
    return () => {
      el.removeEventListener('wheel', onWheel);
      el.removeEventListener('wheel', onZoomWheel);
    };
  }, []);

  // ── In / Out trim handles ────────────────────────────────────────────────────
  const commitTrim = useCallback((nextIn: number, nextOut: number) => {
    onUpdateTrim?.(nextIn, nextOut);
    if (nextIn !== trimIn) onSetTrimIn?.(nextIn);
    if (nextOut !== trimOut) onSetTrimOut?.(nextOut);
  }, [onUpdateTrim, onSetTrimIn, onSetTrimOut, trimIn, trimOut]);

  const handleTrimDrag = useCallback((handle: 'in' | 'out', e: React.PointerEvent<HTMLElement>) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    e.preventDefault();
    (e.currentTarget as HTMLElement).focus?.({ preventScroll: true }); // see handleScrubDown: keep focus on the handle
    const area = readArea();
    if (area) dragAreaRef.current = area;
    const move = (ev: PointerEvent) => {
      const f = snapFrame(xToFrame(ev.clientX)).frame;
      if (handle === 'in') commitTrim(clampFrameIdx(Math.min(f, trimOut), 0, maxFrames), trimOut);
      else commitTrim(trimIn, clampFrameIdx(Math.max(f, trimIn), trimIn, maxFrames));
    };
    const up = () => {
      dragAreaRef.current = null;
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', up);
  }, [snapFrame, xToFrame, commitTrim, trimIn, trimOut, maxFrames]);

  const handleTrimKey = (handle: 'in' | 'out', e: KeyboardEvent<HTMLElement>) => {
    const dir = e.key === 'ArrowLeft' ? -1 : e.key === 'ArrowRight' ? 1 : 0;
    if (!dir) return;
    e.preventDefault();
    e.stopPropagation(); // arrows step the playhead globally; on a focused handle they move the handle
    const delta = dir * (e.shiftKey ? 10 : 1);
    if (handle === 'in') commitTrim(clampFrameIdx(Math.min(trimIn + delta, trimOut), 0, maxFrames), trimOut);
    else commitTrim(trimIn, clampFrameIdx(Math.max(trimOut + delta, trimIn), trimIn, maxFrames));
  };

  // ── Track & Range Lock interactivity ─────────────────────────────────────────
  const handleTrackOrRangeClick = useCallback((track: ByteTrackLane, range?: [number, number], e?: React.MouseEvent) => {
    e?.stopPropagation();
    onSelectTrack?.(track.trackId);
    const targetStart = range ? range[0] : track.firstFrame;
    const targetEnd = range ? range[1] : track.lastFrame;
    const existingIdx = rangeLockIntervals.findIndex(
      (iv) => iv.track_id === track.trackId && Math.max(iv.frame_start, targetStart) <= Math.min(iv.frame_end, targetEnd),
    );
    let updated: RangeLockInterval[];
    if (existingIdx >= 0) {
      const existing = rangeLockIntervals[existingIdx];
      const toggled: RangeLockInterval = { ...existing, locked: existing.locked !== false ? false : true };
      updated = [...rangeLockIntervals.slice(0, existingIdx), toggled, ...rangeLockIntervals.slice(existingIdx + 1)];
      onSelectRange?.(toggled);
    } else {
      const created = createRangeLockInterval({
        id: `rl_${track.trackId}_${targetStart}_${targetEnd}_${Date.now()}`,
        track_id: track.trackId,
        frame_start: clampFrameIdx(targetStart, 0, maxFrames),
        frame_end: clampFrameIdx(targetEnd, targetStart, maxFrames),
        label: `${track.label} [${targetStart}–${targetEnd}]`,
        faceset_id: track.facesetId,
        source_index: track.sourceIndex,
        color: getTrackColor(track.trackId, track.color),
        locked: true,
      }, maxFrames);
      updated = [...rangeLockIntervals, created];
      onSelectRange?.(created);
    }
    onUpdateRangeLockIntervals?.(updated);
  }, [rangeLockIntervals, maxFrames, onSelectTrack, onSelectRange, onUpdateRangeLockIntervals]);

  const handleRangeTrimDrag = useCallback((intervalId: string, handle: 'start' | 'end', e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    e.preventDefault();
    const area = readArea();
    if (area) dragAreaRef.current = area;
    const move = (ev: PointerEvent) => {
      const f = xToFrame(ev.clientX);
      onUpdateRangeLockIntervals?.(rangeLockIntervals.map((item) => {
        if (item.id !== intervalId) return item;
        return handle === 'start'
          ? clampRangeLockInterval({ ...item, frame_start: Math.min(f, item.frame_end) }, 0, maxFrames)
          : clampRangeLockInterval({ ...item, frame_end: Math.max(f, item.frame_start) }, 0, maxFrames);
      }));
    };
    const up = () => {
      dragAreaRef.current = null;
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', up);
  }, [rangeLockIntervals, xToFrame, maxFrames, onUpdateRangeLockIntervals]);

  // Keyboard twin of the edge drag above: ←/→ on a focused Range Lock handle nudges that edge.
  const handleRangeNudge = useCallback((intervalId: string, handle: 'start' | 'end', delta: number) => {
    onUpdateRangeLockIntervals?.(rangeLockIntervals.map((item) => {
      if (item.id !== intervalId) return item;
      return handle === 'start'
        ? clampRangeLockInterval({ ...item, frame_start: Math.min(item.frame_start + delta, item.frame_end) }, 0, maxFrames)
        : clampRangeLockInterval({ ...item, frame_end: Math.max(item.frame_end + delta, item.frame_start) }, 0, maxFrames);
    }));
  }, [rangeLockIntervals, maxFrames, onUpdateRangeLockIntervals]);

  // ── Keyframes (the lane lock refuses every edit) ─────────────────────────────
  const kfLocked = lanes.kf.locked;
  const handleAddKeyframeAtCurrent = useCallback(() => {
    if (kfLocked) return;
    const val = evaluateKeyframeValue(nleKeyframes, activeParam, activeFrame);
    const created = createNleKeyframe(activeParam, activeFrame, val, 'Ease In/Out', maxFrames);
    onUpdateKeyframes?.(sortAndNormalizeKeyframes([...nleKeyframes, created], maxFrames));
    setSelectedKeyframeId(created.id);
  }, [kfLocked, activeParam, activeFrame, nleKeyframes, maxFrames, onUpdateKeyframes]);

  const handleKeyframeDrag = useCallback((kfId: string, e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    e.preventDefault();
    const isCopy = e.altKey; // Alt + Drag = Copy
    setSelectedKeyframeId(kfId);
    const target = nleKeyframes.find((k) => k.id === kfId);
    if (!target) return;
    const area = readArea();
    if (area) dragAreaRef.current = area;
    const move = (ev: PointerEvent) => {
      const f = snapFrame(xToFrame(ev.clientX)).frame;
      if (isCopy) {
        const dup = duplicateKeyframe(target, f - target.frame_idx, maxFrames);
        onUpdateKeyframes?.(sortAndNormalizeKeyframes([...nleKeyframes, dup], maxFrames));
      } else {
        onUpdateKeyframes?.(sortAndNormalizeKeyframes(nleKeyframes.map((k) => (k.id === kfId ? { ...k, frame_idx: clampFrameIdx(f, 0, maxFrames) } : k)), maxFrames));
      }
    };
    const up = () => {
      dragAreaRef.current = null;
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', up);
  }, [nleKeyframes, snapFrame, xToFrame, maxFrames, onUpdateKeyframes]);

  const patchKeyframe = useCallback((id: string, patch: Partial<NleKeyframe>) => {
    if (kfLocked) return;
    onUpdateKeyframes?.(nleKeyframes.map((k) => (k.id === id ? { ...k, ...patch } : k)));
  }, [kfLocked, nleKeyframes, onUpdateKeyframes]);

  const deleteKeyframe = useCallback((id: string) => {
    if (kfLocked) return;
    onUpdateKeyframes?.(nleKeyframes.filter((k) => k.id !== id));
    setSelectedKeyframeId((cur) => (cur === id ? null : cur));
  }, [kfLocked, nleKeyframes, onUpdateKeyframes]);

  // The only key this component owns (everything else is useWorkstationShortcuts / the host).
  const handleKeyDown = useCallback((e: KeyboardEvent<HTMLDivElement>) => {
    const t = e.target as HTMLElement;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT')) return;
    if (hostKeys?.(e)) {
      e.preventDefault();
      e.stopPropagation();
      return;
    }
    if ((e.key === 'Delete' || e.key === 'Backspace') && selectedKeyframeId && !kfLocked) {
      e.preventDefault();
      deleteKeyframe(selectedKeyframeId);
    }
  }, [hostKeys, selectedKeyframeId, kfLocked, deleteKeyframe]);

  // ── Derived display values ───────────────────────────────────────────────────
  const assignmentMap = useMemo(() => new Map(suggestedAssignments.map((s) => [s.trackId, s])), [suggestedAssignments]);
  const warningsByTrack = useMemo(() => {
    const map = new Map<number, WarningTick[]>();
    warningTicks.forEach((tick) => { const list = map.get(tick.trackId) || []; list.push(tick); map.set(tick.trackId, list); });
    return map;
  }, [warningTicks]);
  const activeTimecode = useMemo(() => frameToTimecode(activeFrame, fps), [activeFrame, fps]);
  const activeSeconds = useMemo(() => formatSeconds(frameMap.resolveTimeFromFrame(activeFrame)), [activeFrame, frameMap]);
  const lockedIntervalsCount = useMemo(() => rangeLockIntervals.filter((iv) => iv.locked !== false).length, [rangeLockIntervals]);
  const markerFrames = useMemo(() => markers.map((m) => m.frame), [markers]);
  const regionLabel = regionIsWhole ? 'whole clip' : formatDuration(regionLength(region), fps);

  const segmentLaneBlocks = useMemo<SegmentBlock[]>(() => [
    ...rangeLockIntervals.filter((iv) => iv.locked !== false).map((iv) => ({
      id: `rl_${iv.id}`, start: iv.frame_start, end: iv.frame_end, label: iv.label || `Track ${iv.track_id + 1}`, kind: 'range' as const, color: iv.color,
    })),
    ...segmentBlocks,
  ], [rangeLockIntervals, segmentBlocks]);

  const faceCount = trackLanes.length;
  const faceGroupHeight = lanes.faces.height;
  const faceLaneHeightPx = LANE_HEIGHTS.faces[faceGroupHeight];
  const hv = (id: LaneId) => laneBodyPx(id, lanes[id]);
  const toPct = (f: number) => ((f - viewStart) / viewSpan) * 100;
  const inOutVisibleIn = trimIn >= viewStart && trimIn <= viewEnd;
  const inOutVisibleOut = trimOut >= viewStart && trimOut <= viewEnd;
  const showTransport = !!onTogglePlay;
  const pct1 = (n: number) => (healthSummary.total ? Math.round((n / healthSummary.total) * 100) : 0);

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div
      ref={rootRef}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      className={`nle-timeline-root flex flex-col w-full bg-[var(--surface-1,#12141a)] border border-[var(--border-color,rgba(255,255,255,0.08))] rounded-xl shadow-2xl select-none font-sans text-xs outline-none focus-visible:ring-1 focus-visible:ring-[var(--accent-cyan,#00F0FF)]/50 ${className}`}
      // `clip`, not `hidden`: hidden makes the root a scroll container, which would pin the sticky controls to
      // the root instead of to the dock that actually scrolls. clip still rounds the corners.
      style={{ overflow: 'clip', ...style }}
      data-testid="nle-timeline"
      data-view-start={viewStart}
      data-view-end={viewEnd}
    >
      {/* ── Toolbar ──────────────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-2 bg-[var(--surface-2,#1a1d26)] border-b border-[var(--border-color,rgba(255,255,255,0.08))]">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1.5 font-bold tracking-wide text-white text-plain">
            <svg className="w-4 h-4 text-[var(--accent-cyan,#00F0FF)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <rect x="2" y="4" width="20" height="16" rx="2" /><path d="M10 4v16" /><path d="M14 4v16" />
            </svg>
            <span>NLE Timeline</span>
          </div>

          {showPipelineSteps && (
            <>
              <button
                type="button"
                data-testid="nle-step-prescan"
                onClick={onPreScanClick}
                disabled={isPreScanActive}
                title="Step 1: Run or inspect Pre-Scan ByteTrack spans"
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-mini font-medium border transition-colors ${trackLanes.length > 0 ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300' : isPreScanActive ? 'border-sky-500/50 bg-sky-500/10 text-sky-300 animate-pulse' : 'border-white/10 bg-white/5 text-neutral-400 hover:text-white'}`}
              >
                <span className={`w-2 h-2 rounded-full ${trackLanes.length > 0 ? 'bg-emerald-400' : isPreScanActive ? 'bg-sky-400 animate-ping' : 'bg-neutral-500'}`} />
                <span>1 Pre-Scan</span>
                <span className="font-mono text-micro opacity-80">({trackLanes.length} {trackLanes.length === 1 ? 'Track' : 'Tracks'})</span>
              </button>
              <span aria-hidden="true" className="text-white/25">→</span>
              <button
                type="button"
                data-testid="nle-step-rangelock"
                onClick={onRangeLockClick}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-mini font-medium border transition-colors ${isRangeLockActive || lockedIntervalsCount > 0 ? 'border-[var(--accent,#e94560)]/50 bg-[var(--accent,#e94560)]/10 text-rose-300' : 'border-white/10 bg-white/5 text-neutral-400 hover:text-white'}`}
                title="Step 2: Trim active track intervals and bind facesets (Batch Matrix sync)"
              >
                <span className={`w-2 h-2 rounded-full ${lockedIntervalsCount > 0 ? 'bg-[var(--accent,#e94560)]' : 'bg-neutral-500'}`} />
                <span>2 Range Lock</span>
                <span className="font-mono text-micro opacity-80">({lockedIntervalsCount} {lockedIntervalsCount === 1 ? 'Interval' : 'Intervals'})</span>
              </button>
              <span aria-hidden="true" className="text-white/25">→</span>
              <button
                type="button"
                data-testid="nle-step-render"
                onClick={onRenderClick}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-mini font-medium border transition-colors ${isRenderActive ? 'border-emerald-500/50 bg-emerald-500/15 text-emerald-300 shadow-[0_0_8px_rgba(16,185,129,0.3)] animate-pulse' : 'border-white/10 bg-white/5 text-neutral-400 hover:text-white'}`}
                title="Step 3: Main Swap Render & Audio Remux"
              >
                <span className={`w-2 h-2 rounded-full ${isRenderActive ? 'bg-emerald-400 animate-ping' : 'bg-neutral-500'}`} />
                <span>3 Render &amp; Mux</span>
              </button>
              <button
                type="button"
                data-testid="nle-batch-matrix-btn"
                onClick={() => { if (onBatchClick) onBatchClick(); else if (typeof window !== 'undefined') window.location.hash = '#/batch'; }}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-mini font-medium border border-sky-400/30 bg-sky-400/10 text-sky-200 hover:border-sky-400/50 transition-colors"
                title="Batch Matrix Strategies: Staged track ranges and matrix jobs"
              >
                <span>#/batch</span>
                {stagedJobsCount > 0 ? <span className="font-mono text-micro opacity-90">({stagedJobsCount} staged)</span> : null}
              </button>
            </>
          )}

          <button
            type="button"
            onClick={() => setSnapEnabled((prev) => !prev)}
            aria-pressed={snapEnabled}
            className={`flex items-center gap-1 px-2 py-0.5 rounded border text-mini transition ${snapEnabled ? 'border-sky-500/50 bg-sky-500/15 text-sky-300' : 'border-white/10 bg-white/5 text-neutral-400 hover:text-white'}`}
            title={snapEnabled ? 'Magnetic Snapping ON (Playhead, In/Out, Markers, KF)' : 'Magnetic Snapping OFF'}
          >
            <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="M4 14v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4" /><polyline points="7 9 12 4 17 9" /><line x1="12" y1="4" x2="12" y2="16" /></svg>
            <span>Snap</span>
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1 bg-black/40 border border-white/10 rounded-lg font-mono" data-testid="nle-readout">
            <span className="text-micro uppercase text-neutral-400 font-semibold tracking-wider">TC:</span>
            <span className="text-plain font-bold text-white tracking-wider tabular-nums" data-testid="nle-readout-tc">{fpsAssumed ? '≈' : ''}{activeTimecode}</span>
            <span className="text-neutral-500">|</span>
            <span className="text-mini text-neutral-300 tabular-nums">{activeSeconds}</span>
            <span className="text-neutral-500">|</span>
            <span className="text-mini text-[var(--accent-cyan,#00F0FF)] font-semibold tabular-nums" data-testid="nle-readout-frame">
              {formatFrameLabel(activeFrame)}<span className="text-neutral-500 font-normal"> / {lastFrame}</span>
            </span>
          </div>

          <div className="hidden sm:flex items-center gap-1.5">
            <button type="button" onClick={() => handleSeekFrame(trimIn)} className="flex items-center gap-1 px-2 py-0.5 bg-white/5 border border-white/10 hover:border-amber-400/40 rounded text-mini font-mono text-neutral-300 transition" title="Click to jump to Trim In point ('I' sets at playhead, '[' sets the range start)">
              <span className="text-amber-400 font-bold">IN:</span><span className="tabular-nums">#{trimIn}</span>
            </button>
            <button type="button" onClick={() => handleSeekFrame(trimOut)} className="flex items-center gap-1 px-2 py-0.5 bg-white/5 border border-white/10 hover:border-amber-400/40 rounded text-mini font-mono text-neutral-300 transition" title="Click to jump to Trim Out point ('O' sets at playhead, ']' sets the range end)">
              <span className="text-amber-400 font-bold">OUT:</span><span className="tabular-nums">#{trimOut}</span>
            </button>
            <span className="px-2 py-0.5 rounded border border-amber-400/30 bg-amber-400/10 text-mini font-mono text-amber-200 tabular-nums" data-testid="nle-region-duration" title={regionNote ? `Duration of the In–Out region (${regionNote})` : 'Duration of the In–Out region'}>{regionLabel}</span>
            {regionNote ? <span className="text-mini text-neutral-400" data-testid="nle-region-note">{regionNote}</span> : null}
          </div>

          <button
            type="button"
            onClick={() => addMarkerAt(activeFrame)}
            disabled={lanes.segments.locked}
            className="flex items-center gap-1 px-2 py-1 bg-white/5 border border-white/10 hover:border-sky-400/50 rounded text-mini text-neutral-300 hover:text-white transition disabled:opacity-40"
            title={lanes.segments.locked ? 'The markers lane is locked' : "Add Chapter Marker at playhead ('M')"}
            data-testid="nle-add-marker"
          >
            <svg className="w-3.5 h-3.5 text-sky-400" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M17 3H7c-1.1 0-1.99.9-1.99 2L5 21l7-3 7 3V5c0-1.1-.9-2-2-2z" /></svg>
            <span>+ Marker</span>
            <span className="font-mono text-micro text-neutral-500">({markers.length})</span>
          </button>

          <div className="flex items-center gap-1 bg-black/30 border border-white/10 rounded-lg p-0.5">
            <button type="button" onClick={() => handleZoom(0.7, 0.5)} className="p-1 hover:bg-white/10 rounded text-neutral-300 hover:text-white transition" title="Zoom In (Ctrl + Wheel Up)" aria-label="Zoom In" data-testid="nle-zoom-in">
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /><line x1="11" y1="8" x2="11" y2="14" /><line x1="8" y1="11" x2="14" y2="11" /></svg>
            </button>
            <button type="button" onClick={() => handleZoom(1.4, 0.5)} className="p-1 hover:bg-white/10 rounded text-neutral-300 hover:text-white transition" title="Zoom Out (Ctrl + Wheel Down)" aria-label="Zoom Out" data-testid="nle-zoom-out">
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /><line x1="8" y1="11" x2="14" y2="11" /></svg>
            </button>
            <input
              type="range"
              aria-label="Timeline zoom"
              data-testid="nle-zoom-slider"
              min={0}
              max={100}
              value={spanToZoom(viewSpan, lastFrame)}
              onChange={(e) => {
                // 0 = whole clip, 100 = 4 frames across, log scale; zooms about the window's centre
                const span = zoomToSpan(Number(e.target.value), lastFrame);
                const centre = (viewStart + viewEnd) / 2;
                const start = clampN(Math.round(centre - span / 2), 0, Math.max(0, lastFrame - span));
                setViewRange([start, Math.min(lastFrame, start + span)]);
              }}
              className="w-24 accent-cyan-400"
              title="Zoom (Ctrl + Scroll)"
            />
            <button type="button" onClick={handleResetZoom} className="px-2 py-0.5 text-mini font-medium hover:bg-white/10 rounded text-neutral-300 hover:text-white transition" title="Fit entire clip to window" data-testid="nle-zoom-fit">Fit</button>
          </div>
        </div>
      </div>

      {/* ── Sticky controls: transport, overview and ruler stay put while the lanes scroll under them.
            The dock that hosts the timeline is a short scrolling window (260 px by default) and the full
            stack is taller than that; without this the controls you scrub with scroll out of sight. ── */}
      <div className="sticky top-0 z-40 bg-[var(--surface-1,#12141a)]" data-testid="nle-sticky">
      {showTransport && (
        <TransportDeck
          frame={activeFrame}
          maxFrames={maxFrames}
          fps={fps}
          isPlaying={isPlaying}
          playbackRate={playbackRate}
          loop={isLooping}
          trimIn={trimIn}
          trimOut={trimOut}
          markerFrames={markerFrames}
          shuttleLabel={shuttleLabel}
          shuttleDir={shuttleDir}
          onSeek={(f) => handleSeekFrame(f)}
          onStep={(d) => (onStep ? onStep(d) : handleSeekFrame(activeFrame + d))}
          onTogglePlay={onTogglePlay!}
          onRate={(r) => onPlaybackRateChange?.(r)}
          onToggleLoop={() => onToggleLoop?.()}
          onShuttle={onShuttle}
        />
      )}

      {/* ── Overview navigator ───────────────────────────────────────────────── */}
      <MinimapNavigator
        maxFrames={maxFrames}
        view={viewRange}
        onViewChange={setViewRange}
        frame={activeFrame}
        trimIn={trimIn}
        trimOut={trimOut}
        health={healthSegs}
        tracks={minimapTracks}
        markers={markers}
        headerWidth={LANE_HEADER_PX}
      />

        {/* Ruler */}
        <div className="relative flex w-full border-b border-white/10 bg-[var(--surface-2,#1a1d26)]" style={{ height: RULER_HEIGHT }}>
          <div className="shrink-0 flex items-center justify-between px-3 border-r border-white/10 text-mini font-semibold text-neutral-400 uppercase tracking-wider z-10" style={{ width: LANE_HEADER_PX }}>
            <span>Ruler</span>
            <span className="text-mini text-neutral-400 font-mono normal-case" title={fpsAssumed ? 'Frame rate not read yet: it is measured while the clip plays, and 30 fps is assumed until then (timecodes and frame counts are approximate)' : 'Frame rate'}>{fpsAssumed ? '≈ ' : ''}{Number.isInteger(fps) ? fps : fps.toFixed(3)} fps</span>
          </div>
          <div
            className="relative flex-1 cursor-col-resize"
            onPointerDown={handleScrubDown}
            onPointerMove={handleRulerHover}
            onPointerLeave={() => setHoverTip(null)}
            data-testid="nle-ruler"
          >
            <RulerCanvas viewStart={viewStart} viewEnd={viewEnd} fps={fps} />
            {markers.map((m) => (m.frame >= viewStart && m.frame <= viewEnd ? (
              <span key={m.id} className="absolute top-0 w-0 h-0 pointer-events-none" style={{ left: `${toPct(m.frame)}%`, borderLeft: '4px solid transparent', borderRight: '4px solid transparent', borderTop: `6px solid ${m.color || DEFAULT_MARKER_COLOR}`, transform: 'translateX(-4px)' }} />
            ) : null))}
            {inOutVisibleIn && (
              <div
                role="slider"
                tabIndex={0}
                aria-label="Trim In handle"
                aria-valuemin={0}
                aria-valuemax={lastFrame}
                aria-valuenow={trimIn}
                onPointerDown={(e) => handleTrimDrag('in', e)}
                onKeyDown={(e) => handleTrimKey('in', e)}
                className="absolute top-0 bottom-0 w-3 cursor-ew-resize z-20 flex items-center justify-center group focus:outline-none"
                style={{ left: `${toPct(trimIn)}%` }}
                title={`Trim In Handle: #${trimIn} (drag, ←/→ to nudge, or press 'I')`}
                data-testid="nle-trim-in"
              >
                <div className="w-2 h-6 bg-amber-400 rounded-l group-hover:scale-110 group-focus-visible:ring-2 ring-white shadow-md flex items-center justify-center text-micro font-bold text-black">[</div>
              </div>
            )}
            {inOutVisibleOut && (
              <div
                role="slider"
                tabIndex={0}
                aria-label="Trim Out handle"
                aria-valuemin={0}
                aria-valuemax={lastFrame}
                aria-valuenow={trimOut}
                onPointerDown={(e) => handleTrimDrag('out', e)}
                onKeyDown={(e) => handleTrimKey('out', e)}
                className="absolute top-0 bottom-0 w-3 -translate-x-full cursor-ew-resize z-20 flex items-center justify-center group focus:outline-none"
                style={{ left: `${toPct(trimOut)}%` }}
                title={`Trim Out Handle: #${trimOut} (drag, ←/→ to nudge, or press 'O')`}
                data-testid="nle-trim-out"
              >
                <div className="w-2 h-6 bg-amber-400 rounded-r group-hover:scale-110 group-focus-visible:ring-2 ring-white shadow-md flex items-center justify-center text-micro font-bold text-black">]</div>
              </div>
            )}
          </div>

          {/* Ruler layer: exactly the lane area (everything right of the header column). EVERY pixel <-> frame
              mapping reads this element's rect. It carries the playhead cap and the top of the playhead line. */}
          <div ref={areaRef} className="absolute top-0 bottom-0 right-0 pointer-events-none z-30" style={{ left: LANE_HEADER_PX }} data-testid="nle-lane-area">
            {activeFrame >= viewStart && activeFrame <= viewEnd && (
              <div className="absolute top-0 bottom-0 w-0" style={{ left: `${toPct(activeFrame)}%` }}>
                <div className="absolute top-0 bottom-0 w-px -translate-x-1/2 bg-[var(--accent-cyan,#00F0FF)]" style={{ boxShadow: '0 0 6px var(--accent-cyan,#00F0FF), 0 0 1px #fff' }} data-testid="nle-playhead-line" />
                <div
                  onPointerDown={handleScrubDown}
                  className="absolute top-0 -translate-x-1/2 w-[15px] h-[17px] pointer-events-auto cursor-ew-resize bg-[var(--accent-cyan,#00F0FF)]"
                  style={{ clipPath: 'polygon(0 0, 100% 0, 100% 58%, 50% 100%, 0 58%)', boxShadow: '0 0 8px var(--accent-cyan,#00F0FF)' }}
                  role="presentation"
                  data-testid="nle-playhead-cap"
                  title="Drag to scrub"
                />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Runway: the lane stack, with a layer over the lane area for the In/Out dimming, snap guide and playhead ── */}
      <div className="relative flex flex-col w-full bg-black/20" data-testid="nle-runway">
        {/* Lane 1: video filmstrip */}
        {showFilmstripLane && (
          <LaneRow
            id="v1"
            testId="lane-v1-filmstrip"
            badge="V1"
            badgeClass="bg-blue-500/20 text-blue-400"
            title="Video"
            state={lanes.v1}
            {...laneActions('v1')}
            onToggleLocked={undefined}
            heightPx={hv('v1')}
            status={<span data-testid="nle-v1-status">{film.slots.length} stills · {regionLabel}</span>}
          >
            <div className="absolute inset-0 cursor-col-resize" onPointerDown={handleScrubDown}>
              <FilmstripCanvas slots={film.slots} getThumbnail={film.getThumbnail} version={film.version} />
              {/* the clip outside In/Out is dimmed on every lane (see the overlay layer); here it is also labelled */}
              {!regionIsWhole && (
                <div
                  className="absolute top-1 -translate-x-1/2 px-2 py-0.5 rounded-full bg-black/80 border border-amber-400/50 text-amber-200 text-mini font-mono tabular-nums pointer-events-none"
                  style={{ left: `${clampN(toPct((region.in + region.out) / 2), 8, 92)}%` }}
                >
                  {regionLabel}
                </div>
              )}
              {inOutVisibleIn && (
                <div role="presentation" onPointerDown={(e) => handleTrimDrag('in', e)} className="absolute top-0 bottom-0 w-3 cursor-ew-resize z-20 bg-amber-400/90 rounded-l flex items-center justify-center text-micro font-bold text-black" style={{ left: `${toPct(trimIn)}%` }} title={`Trim In: #${trimIn}`}>[</div>
              )}
              {inOutVisibleOut && (
                <div role="presentation" onPointerDown={(e) => handleTrimDrag('out', e)} className="absolute top-0 bottom-0 w-3 -translate-x-full cursor-ew-resize z-20 bg-amber-400/90 rounded-r flex items-center justify-center text-micro font-bold text-black" style={{ left: `${toPct(trimOut)}%` }} title={`Trim Out: #${trimOut}`}>]</div>
              )}
            </div>
          </LaneRow>
        )}

        {/* Lane 2: face tracking & confidence heatmap */}
        {showHealthLane && (
          <LaneRow
            id="health"
            testId="lane-health"
            badge="FT"
            badgeClass="bg-emerald-500/20 text-emerald-400"
            title="Face coverage"
            state={lanes.health}
            {...laneActions('health')}
            onToggleLocked={undefined}
            heightPx={hv('health')}
            status={trackLanes.length === 0 ? 'not scanned' : <span data-testid="nle-health-status" title="Share of the clip: face tracked / extreme pose / no face (they add up to 100%)">{pct1(healthSummary.face)}% ok · {pct1(healthSummary.risk)}% risk · {pct1(healthSummary.lost)}% lost</span>}
            extra={
              <div className="flex items-center gap-0.5">
                <button type="button" disabled={neighbourFrame(problems, activeFrame, -1) === null} onClick={() => { const f = neighbourFrame(problems, activeFrame, -1); if (f !== null) handleSeekFrame(f); }} aria-label="Previous problem frame" title="Jump to the previous amber / rose run" data-testid="nle-health-prev" className="w-5 h-6 rounded text-neutral-300 hover:text-white hover:bg-white/10 disabled:opacity-30">‹</button>
                <button type="button" disabled={neighbourFrame(problems, activeFrame, 1) === null} onClick={() => { const f = neighbourFrame(problems, activeFrame, 1); if (f !== null) handleSeekFrame(f); }} aria-label="Next problem frame" title="Jump to the next amber / rose run" data-testid="nle-health-next" className="w-5 h-6 rounded text-neutral-300 hover:text-white hover:bg-white/10 disabled:opacity-30">›</button>
              </div>
            }
          >
            <FaceHealthLane segments={healthSegs} viewStart={viewStart} viewEnd={viewEnd} maxFrames={maxFrames} onJump={(f) => handleSeekFrame(f)} onScrubDown={handleScrubDown} />
          </LaneRow>
        )}

        {/* Lane 3: swap segments / overrides + markers */}
        {showSegmentLane && (
          <LaneRow
            id="segments"
            testId="lane-segments"
            badge={segmentLaneTitle === 'Swap rules' ? 'SW' : 'MK'}
            badgeClass="bg-violet-500/20 text-violet-300"
            title={segmentLaneTitle}
            state={lanes.segments}
            {...laneActions('segments')}
            heightPx={hv('segments')}
            status={`${segmentLaneBlocks.length} blocks · ${markers.length} markers`}
          >
            <SegmentLane
              blocks={segmentLaneBlocks}
              markers={markers}
              viewStart={viewStart}
              viewEnd={viewEnd}
              maxFrames={maxFrames}
              locked={lanes.segments.locked}
              selectedMarkerId={popover?.id ?? null}
              onSeek={(f) => handleSeekFrame(f)}
              onBlockClick={onSegmentClick}
              onMarkerSelect={(m, anchor) => setPopover({ id: m.id, anchor })}
              onMarkerMove={(m, f) => { setPopover(null); updateMarker({ ...m, frame: f }); }}
              snap={(f) => snapFrame(f).frame}
              onScrubDown={handleScrubDown}
              emptyHint={trackLanes.length === 0 ? 'Range locks, segments and overrides appear here. Press M or "+ Marker" to flag a frame.' : undefined}
            />
          </LaneRow>
        )}

        {/* Lane 4: audio */}
        {showWaveformLane && (
          <LaneRow
            id="a1"
            testId="lane-a1-waveform"
            badge="A1"
            badgeClass="bg-emerald-500/20 text-emerald-400"
            title="Audio"
            state={lanes.a1}
            {...laneActions('a1')}
            onToggleLocked={undefined}
            heightPx={hv('a1')}
            status={wave.state === 'ready' ? 'peaks' : wave.state === 'none' ? 'no audio' : wave.state}
          >
            <div className="absolute inset-0 cursor-col-resize" onPointerDown={handleScrubDown}>
              {wave.state === 'ready' && wave.peaks.length > 0 ? (
                <WaveformCanvas
                  peaks={wave.peaks}
                  duration={wave.duration}
                  tStart={frameMap.resolveTimeFromFrame(viewStart)}
                  tEnd={frameMap.resolveTimeFromFrame(viewEnd)}
                />
              ) : wave.state === 'none' ? (
                <div className="flex h-full items-center px-4 text-mini text-neutral-500 italic">{wave.reason ? `No waveform: ${wave.reason}` : '(No audio track detected in source media)'}</div>
              ) : wave.state === 'error' ? (
                <div className="flex h-full items-center px-4 text-mini text-rose-300">Waveform could not be loaded</div>
              ) : (
                <div className="flex h-full items-center px-4 text-mini text-sky-300 animate-pulse">{clientMode ? 'Decoding audio…' : 'Loading waveform…'}</div>
              )}
            </div>
          </LaneRow>
        )}

        {/* Keyframes */}
        {showKeyframeLane && (
          <LaneRow
            id="kf"
            testId="lane-kf-editor"
            badge="KF"
            badgeClass="bg-amber-500/20 text-amber-400"
            title="Keyframes"
            state={lanes.kf}
            {...laneActions('kf')}
            heightPx={hv('kf')}
            extra={
              <>
                <select
                  value={activeParam}
                  onChange={(e) => setActiveParam(e.target.value as TimelineParamType)}
                  onClick={(e) => e.stopPropagation()}
                  className="min-w-0 max-w-[104px] bg-black/60 border border-white/15 rounded px-1 py-0.5 text-mini font-medium text-white focus:outline-none focus:border-amber-400/60"
                  title="Active Keyframe Parameter"
                  aria-label="Active keyframe parameter"
                >
                  {Object.values(PARAMETER_CONFIGS).map((config) => <option key={config.id} value={config.id}>{config.label}</option>)}
                </select>
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); handleAddKeyframeAtCurrent(); }}
                  disabled={kfLocked}
                  className="p-1 hover:bg-white/10 rounded text-amber-400 hover:text-white transition disabled:opacity-40"
                  title={kfLocked ? 'The keyframe lane is locked' : `Add ${activeParam} keyframe at frame #${activeFrame}`}
                  aria-label={`Add ${activeParam} keyframe`}
                >
                  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
                </button>
              </>
            }
          >
            <KeyframeLane
              keyframes={activeParamKeyframes}
              viewStart={viewStart}
              viewEnd={viewEnd}
              widthPx={areaWidth}
              heightPx={hv('kf')}
              selectedId={selectedKeyframeId}
              locked={kfLocked}
              onSelect={setSelectedKeyframeId}
              onSeek={(f) => handleSeekFrame(f)}
              onDragStart={handleKeyframeDrag}
              onPatch={patchKeyframe}
              onDelete={deleteKeyframe}
              onScrubDown={handleScrubDown}
            />
          </LaneRow>
        )}

        {/* Per-target face track lanes */}
        {showFaceTrackLanes && (
          <div className="flex flex-col w-full" data-testid="nle-face-tracks" data-hidden={lanes.faces.hidden ? '1' : '0'} data-height={lanes.faces.height}>
            <div className="flex w-full border-b border-white/5" style={{ height: COLLAPSED_PX + 12 }}>
              <LaneHeader
                laneId="faces"
                badge="FS"
                badgeClass="bg-pink-500/20 text-pink-300"
                title={LANE_LABELS.faces}
                state={lanes.faces}
                {...laneActions('faces')}
                onToggleLocked={undefined}
                width={LANE_HEADER_PX}
                height={COLLAPSED_PX + 12}
                status={`${faceCount} ${faceCount === 1 ? 'track' : 'tracks'}`}
              />
              <div className="flex-1 bg-black/20" />
            </div>
            {!lanes.faces.hidden && (
              <div className="flex flex-col w-full divide-y divide-white/5">
                {trackLanes.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-6 px-4 text-center text-neutral-400">
                    <p className="text-compact font-medium text-neutral-300">No ByteTrack Lanes Found</p>
                    <p className="text-mini text-neutral-500 max-w-sm mt-0.5">Run Step 1 Pre-Scan to detect face trajectories, populate ByteTrack lanes, and extract pose/occlusion warning ticks.</p>
                    {showPipelineSteps && (
                      <button type="button" onClick={onPreScanClick} disabled={isPreScanActive} className="mt-2.5 px-3 py-1 bg-[var(--accent,#e94560)] hover:bg-[var(--accent,#e94560)]/90 text-white rounded text-mini font-semibold shadow-lg transition">
                        {isPreScanActive ? 'Pre-Scanning…' : 'Start 1 Pre-Scan'}
                      </button>
                    )}
                  </div>
                ) : (
                  trackLanes.map((track) => {
                    const isTrackLocked = rangeLockIntervals.some((iv) => iv.track_id === track.trackId && iv.locked !== false);
                    return (
                      <FaceTrackLane
                        key={`track_lane_${track.trackId}`}
                        track={track}
                        color={getFacesetColor(track.facesetPath || track.suggestedFsz, getTrackColor(track.trackId, track.color))}
                        isSelected={selectedTrackId === track.trackId}
                        isLocked={isTrackLocked}
                        suggested={assignmentMap.get(track.trackId)}
                        warnings={warningsByTrack.get(track.trackId) || []}
                        showWarnings={showWarningTicks}
                        showSuggested={showSuggestedFsz}
                        facesets={facesets}
                        intervals={rangeLockIntervals}
                        viewStart={viewStart}
                        viewEnd={viewEnd}
                        maxFrames={maxFrames}
                        height={faceGroupHeight}
                        laneHeightPx={faceLaneHeightPx}
                        headerWidth={LANE_HEADER_PX}
                        onRowClick={(e) => handleTrackOrRangeClick(track, undefined, e)}
                        onRangeClick={(range, e) => handleTrackOrRangeClick(track, range, e)}
                        onToggleRangeLock={(e) => handleTrackOrRangeClick(track, undefined, e)}
                        onToggleActive={onToggleTrackActive ? (active) => onToggleTrackActive(track.trackId, active) : undefined}
                        onAssign={(fsz) => onAssignFsz?.(track.trackId, fsz)}
                        onTrimDrag={handleRangeTrimDrag}
                        onTrimNudge={handleRangeNudge}
                        onWarningClick={(tick) => { handleSeekFrame(tick.frame); onSelectTrack?.(tick.trackId); }}
                      />
                    );
                  })
                )}
              </div>
            )}
          </div>
        )}

        {/* ── Lanes layer: same left / right as the ruler layer, so a frame is at the same x in both ───── */}
        <div className="absolute top-0 bottom-0 right-0 pointer-events-none z-30" style={{ left: LANE_HEADER_PX }} data-testid="nle-lane-layer">
          {/* the clip outside In / Out, dimmed across every lane */}
          {!regionIsWhole && (
            <>
              <div className="absolute top-0 bottom-0 left-0 bg-black/45" style={{ width: `${clampN(toPct(region.in), 0, 100)}%` }} data-testid="nle-dim-before" />
              <div className="absolute top-0 bottom-0 right-0 bg-black/45" style={{ width: `${clampN(100 - toPct(region.out), 0, 100)}%` }} data-testid="nle-dim-after" />
            </>
          )}

          {/* snapping guide */}
          {snapIndicatorFrame !== null && snapIndicatorFrame >= viewStart && snapIndicatorFrame <= viewEnd && (
            <div className="absolute top-0 bottom-0" style={{ left: `${toPct(snapIndicatorFrame)}%` }}>
              <div className="w-px h-full border-l border-dashed border-sky-400 opacity-80" />
              <div className="absolute top-8 -translate-x-1/2 px-1 py-0.5 bg-sky-500 text-black text-micro font-mono font-bold rounded shadow">SNAP {formatFrameLabel(snapIndicatorFrame)}</div>
            </div>
          )}

          {/* playhead: the cyan line continues down through every lane (cap and top of the line are in the ruler layer) */}
          {activeFrame >= viewStart && activeFrame <= viewEnd && (
            <div className="absolute top-0 bottom-0 w-0" style={{ left: `${toPct(activeFrame)}%` }} data-testid="nle-playhead" data-frame={activeFrame}>
              <div className="absolute top-0 bottom-0 w-px -translate-x-1/2 bg-[var(--accent-cyan,#00F0FF)]" style={{ boxShadow: '0 0 6px var(--accent-cyan,#00F0FF), 0 0 1px #fff' }} />
            </div>
          )}
        </div>
      </div>

      {/* ── Footer ───────────────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-1.5 bg-[var(--surface-2,#1a1d26)] border-t border-white/5 text-mini text-neutral-400">
        <div className="flex items-center gap-3">
          <span>Visible: <span className="font-mono text-white">#{viewStart}</span> – <span className="font-mono text-white">#{viewEnd}</span> ({viewSpan + 1} frames)</span>
          <span className="text-neutral-600">|</span>
          <span>Zoom: <span className="font-mono text-white">{Math.round((maxFrames / viewSpan) * 100)}%</span></span>
          <span className="text-neutral-600">|</span>
          <span>Snap: <span className={snapEnabled ? 'text-sky-300' : 'text-neutral-500'}>{snapEnabled ? 'ON' : 'OFF'}</span></span>
        </div>
        <div className="flex items-center gap-3">
          {selectedTrackId !== null && (
            <span className="text-white font-medium">Selected: Track #{typeof selectedTrackId === 'number' ? selectedTrackId + 1 : selectedTrackId}</span>
          )}
          <span className="text-neutral-500 text-micro">
            <kbd className="px-1 bg-black/40 rounded border border-white/10">Space</kbd> Play ·{' '}
            <kbd className="px-1 bg-black/40 rounded border border-white/10">I</kbd>/<kbd className="px-1 bg-black/40 rounded border border-white/10">O</kbd> In/Out ·{' '}
            <kbd className="px-1 bg-black/40 rounded border border-white/10">M</kbd> Marker ·{' '}
            <kbd className="px-1 bg-black/40 rounded border border-white/10">Ctrl+Wheel</kbd> Zoom
          </span>
        </div>
      </div>

      {/* ── Floating: scrub card + marker editor (portalled, so overflow-hidden cannot clip them) ── */}
      <ScrubTooltip
        visible={!!tip}
        clientX={tip?.x ?? 0}
        anchorTop={tip?.top ?? 0}
        frame={tip?.frame ?? 0}
        timecode={tip ? frameToTimecode(tip.frame, fps) : ''}
        seconds={tip ? formatSeconds(frameMap.resolveTimeFromFrame(tip.frame)) : ''}
        bitmap={exactThumb || nearThumb}
        approximate={!exactThumb}
      />
      {popover && popoverMarker && (
        <MarkerPopover
          marker={popoverMarker}
          timecode={frameToTimecode(popoverMarker.frame, fps)}
          anchor={popover.anchor}
          locked={lanes.segments.locked}
          onChange={updateMarker}
          onDelete={deleteMarker}
          onGoTo={(f) => handleSeekFrame(f)}
          onClose={() => setPopover(null)}
        />
      )}
    </div>
  );
};

export default NleTimeline;
