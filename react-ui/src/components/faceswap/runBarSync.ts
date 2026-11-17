/**
 * Run Bar Sync: Unified 3-Step Run Bar, Batch Matrix, and NLE Timeline / Preview Bridge.
 *
 * Pipeline Flow:
 *   Step 1: Pre-Scan      — Triggers pre-scan / fetches /api/trajectory/spans & summary,
 *                           populating ByteTrack lanes, pose/occlusion warning ticks, and suggested .fsz files.
 *   Step 2: Range Lock    — Selects/trims track intervals, syncs them to Batch Matrix staged jobs / face mappings,
 *                           and sets active selected track ID for PreviewViewport.
 *   Step 3: Render & Mux  — Dispatches Stage 3 Main Swap & Stage 4 Mux pipeline.
 *
 * Parity Bridge:
 *   Monitors /api/preview/frame and /ws/preview telemetry for `tracks_source === 'trajectory_plan'`
 *   and verified active TrajectoryPlan cache, bridging the Preview Export Parity HUD badge.
 */

import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import {
  type RangeLockInterval,
  createRangeLockInterval,
  clampRangeLockInterval,
} from '../../utils/ptsFrameMap';
import { stageRangeLockTracks } from './batchMatrix.js';
import { getJSON, API, timeoutFor } from '../../api.js';

export type { RangeLockInterval };

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
  type: 'pose' | 'occlusion' | 'interpolated' | 'coasted' | 'gap';
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

export interface PreviewExportParityState {
  tracksSource: 'trajectory_plan' | 'none' | string;
  trajectoryPlanActive: boolean;
  isParityVerified: boolean;
  badgeText: string;
  badgeTone: 'verified' | 'unverified' | 'idle';
  reason?: string;
  cachedFaces?: number;
  cachedFrames?: [number, number] | null;
  trackIds?: number[];
  projectId?: string | null;
}

export interface TrajectorySummary {
  available: boolean;
  faces?: number;
  bytes?: number;
  first_frame?: number | null;
  last_frame?: number | null;
  track_ids?: number[];
  project_id?: string | null;
  reason?: string;
  [key: string]: any;
}

export interface TrajectoryTrackSpan {
  track: number;
  first: number;
  last: number;
  frames: number;
  ranges: Array<[number, number]>;
  source_index: number | null;
  faceset_id: string | null;
  faceset_path: string | null;
  via: string;
  reason: string;
}

export interface TrajectorySpansResponse {
  available: boolean;
  relative?: boolean;
  tracks: TrajectoryTrackSpan[];
  assigned?: number;
  named?: number;
  project_id?: string | null;
  reason?: string;
}

export type RunBarStepKey = 'prescan' | 'rangelock' | 'render' | 'all';

// Distinct, high-contrast palette for NLE ByteTrack timeline lanes
export const TRACK_PALETTE = Object.freeze([
  '#38bdf8', // Sky 400
  '#34d399', // Emerald 400
  '#fbbf24', // Amber 400
  '#f472b6', // Pink 400
  '#a78bfa', // Violet 400
  '#2dd4bf', // Teal 400
  '#fb923c', // Orange 400
  '#e879f9', // Fuchsia 400
]);

// ── API Fetchers ──────────────────────────────────────────────────────────────

/**
 * Fetch cached TrajectoryPlan summary from backend (GET /api/trajectory/summary).
 */
export async function fetchTrajectorySummary(apiBase?: string): Promise<TrajectorySummary> {
  const base = apiBase || '';
  const url = `${base}/api/trajectory/summary`;
  try {
    const res = await getJSON(url, { timeout: timeoutFor('GET', '/api/trajectory/summary') });
    return res as TrajectorySummary;
  } catch (err: any) {
    return {
      available: false,
      reason: err?.message || 'Trajectory summary unavailable',
    };
  }
}

/**
 * Fetch ByteTrack active frame spans from backend (GET /api/trajectory/spans).
 */
export async function fetchTrajectorySpans(
  apiBase?: string,
  relative = true,
  maxGap = 1
): Promise<TrajectorySpansResponse> {
  const base = apiBase || '';
  const url = `${base}/api/trajectory/spans?relative=${relative ? 1 : 0}&max_gap=${maxGap}`;
  try {
    const res = await getJSON(url, { timeout: timeoutFor('GET', '/api/trajectory/spans') });
    return res as TrajectorySpansResponse;
  } catch (err: any) {
    return {
      available: false,
      tracks: [],
      reason: err?.message || 'Trajectory spans unavailable',
    };
  }
}

/**
 * Fetch pose landmarks & head orientation for a single frame (GET /api/trajectory/pose).
 */
export async function fetchTrajectoryPose(
  frame: number,
  apiBase?: string,
  absolute = false
): Promise<any> {
  const base = apiBase || '';
  const url = `${base}/api/trajectory/pose?frame=${Math.max(0, Math.round(frame))}&absolute=${absolute ? 1 : 0}`;
  try {
    return await getJSON(url, { timeout: timeoutFor('GET', '/api/trajectory/pose') });
  } catch (err: any) {
    return {
      available: false,
      reason: err?.message || 'Trajectory pose unavailable',
    };
  }
}

// ── Transform & Mapping Helpers ───────────────────────────────────────────────

/**
 * Format raw trajectory track spans into NleTimeline ByteTrack lanes.
 */
export function formatTrackLanes(
  spans: TrajectorySpansResponse | null,
  sourceFacesInfo: any[] = []
): ByteTrackLane[] {
  if (!spans || !Array.isArray(spans.tracks) || spans.tracks.length === 0) {
    return [];
  }

  return spans.tracks.map((t, idx) => {
    const trackId = Number(t.track);
    const label = `Track #${trackId + 1}`;
    const color = TRACK_PALETTE[trackId % TRACK_PALETTE.length];

    let suggestedFsz: string | null = null;
    let sourceIdx = t.source_index;

    if (t.faceset_path) {
      suggestedFsz = t.faceset_path.split(/[\\/]/).pop() || t.faceset_path;
    } else if (t.faceset_id) {
      suggestedFsz = `${t.faceset_id}.fsz`;
    } else if (
      Number.isInteger(sourceIdx) &&
      sourceIdx !== null &&
      sourceIdx >= 0 &&
      sourceIdx < sourceFacesInfo.length
    ) {
      const srcName = sourceFacesInfo[sourceIdx]?.name || `face_${sourceIdx + 1}`;
      suggestedFsz = srcName.endsWith('.fsz') ? srcName : `${srcName}.fsz`;
    } else if (sourceFacesInfo.length > 0) {
      const candidateIdx = trackId % sourceFacesInfo.length;
      const candidate = sourceFacesInfo[candidateIdx];
      const name = candidate?.name || `face_${candidateIdx + 1}`;
      suggestedFsz = name.endsWith('.fsz') ? name : `${name}.fsz`;
      sourceIdx = candidateIdx;
    } else {
      suggestedFsz = `track_${trackId + 1}.fsz`;
    }

    const ranges: Array<[number, number]> =
      Array.isArray(t.ranges) && t.ranges.length > 0
        ? t.ranges.map(([s, e]) => [Math.round(s), Math.round(e)])
        : [[Math.round(t.first), Math.round(t.last)]];

    return {
      trackId,
      label,
      firstFrame: Math.round(t.first),
      lastFrame: Math.round(t.last),
      totalFrames: Math.round(t.frames),
      ranges,
      sourceIndex: sourceIdx,
      facesetId: t.faceset_id,
      facesetPath: t.faceset_path,
      suggestedFsz,
      via: t.via || 'pre_scan',
      reason: t.reason,
      active: true,
      color,
    };
  });
}

/**
 * Extract pose, occlusion, and track gap warning ticks from spans and telemetry.
 */
export function extractWarningTicks(
  spans: TrajectorySpansResponse | null,
  summary?: TrajectorySummary | null,
  telemetryFaces?: any[]
): WarningTick[] {
  const warnings: WarningTick[] = [];

  // 1. Detect gaps in multi-range spans (track lost or occluded)
  if (spans && Array.isArray(spans.tracks)) {
    spans.tracks.forEach((t) => {
      const trackId = Number(t.track);
      if (Array.isArray(t.ranges) && t.ranges.length > 1) {
        for (let i = 0; i < t.ranges.length - 1; i++) {
          const currentEnd = Math.round(t.ranges[i][1]);
          const nextStart = Math.round(t.ranges[i + 1][0]);
          const gapLen = nextStart - currentEnd - 1;
          if (gapLen > 0) {
            warnings.push({
              frame: currentEnd + 1,
              trackId,
              type: 'gap',
              severity: gapLen > 10 ? 'warning' : 'info',
              message: `Track #${trackId + 1} tracking gap: ${gapLen} frames missing between ${currentEnd} and ${nextStart}`,
              value: gapLen,
            });
          }
        }
      }
    });
  }

  // 2. Telemetry face warnings (extreme pose angle & high occlusion)
  if (Array.isArray(telemetryFaces)) {
    telemetryFaces.forEach((face) => {
      const trackId = face.track_id != null ? Number(face.track_id) : 0;
      const frame = Math.round(face.frame_idx ?? face.frame ?? 0);

      // Extreme pose check
      if (face.pose) {
        const yaw = Math.abs(face.pose.yaw ?? 0);
        const pitch = Math.abs(face.pose.pitch ?? 0);
        if (yaw > 45 || pitch > 30) {
          warnings.push({
            frame,
            trackId,
            type: 'pose',
            severity: yaw > 60 || pitch > 40 ? 'warning' : 'info',
            message: `Extreme head angle on Track #${trackId + 1}: yaw ${yaw.toFixed(0)}°, pitch ${pitch.toFixed(0)}°`,
            value: yaw,
          });
        }
      }

      // High occlusion check
      const occl = face.occlusion_coverage_pct;
      if (typeof occl === 'number' && occl > 35) {
        warnings.push({
          frame,
          trackId,
          type: 'occlusion',
          severity: occl > 60 ? 'warning' : 'info',
          message: `Heavy occlusion on Track #${trackId + 1}: ${occl.toFixed(0)}% occluded`,
          value: occl,
        });
      }
    });
  }

  return warnings.sort((a, b) => a.frame - b.frame);
}

/**
 * Derive suggested .fsz assignments for all ByteTrack lanes.
 */
export function deriveSuggestedAssignments(
  spans: TrajectorySpansResponse | null,
  sourceFacesInfo: any[] = []
): SuggestedFszAssignment[] {
  if (!spans || !Array.isArray(spans.tracks)) return [];

  return spans.tracks.map((t) => {
    const trackId = Number(t.track);
    let srcIdx = t.source_index;
    let name = '';

    if (
      Number.isInteger(srcIdx) &&
      srcIdx !== null &&
      srcIdx >= 0 &&
      srcIdx < sourceFacesInfo.length
    ) {
      name = sourceFacesInfo[srcIdx]?.name || `Face ${srcIdx + 1}`;
    } else if (sourceFacesInfo.length > 0) {
      srcIdx = trackId % sourceFacesInfo.length;
      name = sourceFacesInfo[srcIdx]?.name || `Face ${srcIdx + 1}`;
    } else {
      srcIdx = 0;
      name = `Identity ${trackId + 1}`;
    }

    const cleanBase = name.replace(/\.fsz$/i, '').trim();
    const suggestedFileName = `${cleanBase || `identity_track_${trackId + 1}`}.fsz`;

    return {
      trackId,
      sourceIndex: srcIdx ?? 0,
      facesetId: t.faceset_id || (sourceFacesInfo[srcIdx]?.id ?? null),
      facesetPath: t.faceset_path || null,
      name,
      suggestedFileName,
      confidence: t.source_index !== null ? 1.0 : 0.85,
    };
  });
}

/**
 * Bridge preview telemetry and trajectory summary to derive Preview Export Parity HUD state.
 */
export function bridgePreviewExportParity(
  telemetry: any,
  summary?: TrajectorySummary | null
): PreviewExportParityState {
  const tracksSource = telemetry?.tracks_source || 'none';
  const hasPlanInTelemetry = tracksSource === 'trajectory_plan';
  const hasActiveSummary = Boolean(summary?.available && (summary?.faces ?? 0) > 0);
  const isParityVerified = hasPlanInTelemetry && (hasActiveSummary || summary?.available !== false);

  let badgeText = 'Preview Export Parity: Unverified';
  let badgeTone: 'verified' | 'unverified' | 'idle' = 'idle';

  if (isParityVerified) {
    badgeText = 'Preview Export Parity: 100% Locked (TrajectoryPlan)';
    badgeTone = 'verified';
  } else if (hasActiveSummary && !hasPlanInTelemetry) {
    badgeText = 'Trajectory Cache Active (Frame Unbound)';
    badgeTone = 'unverified';
  } else if (hasPlanInTelemetry) {
    badgeText = 'Trajectory Plan Tracks Active';
    badgeTone = 'verified';
  }

  const cachedFrames: [number, number] | null =
    summary?.first_frame != null && summary?.last_frame != null
      ? [summary.first_frame, summary.last_frame]
      : null;

  return {
    tracksSource,
    trajectoryPlanActive: hasPlanInTelemetry || hasActiveSummary,
    isParityVerified,
    badgeText,
    badgeTone,
    reason: isParityVerified
      ? 'Current preview frame reads track IDs and alignment directly from sealed TrajectoryPlan cache'
      : (summary?.reason || 'No trajectory plan active on current frame'),
    cachedFaces: summary?.faces,
    cachedFrames,
    trackIds: summary?.track_ids,
    projectId: summary?.project_id,
  };
}

/**
 * Convert ByteTrack lanes into Step 2 RangeLockInterval objects.
 */
export function convertTrackLanesToRangeLockIntervals(
  lanes: ByteTrackLane[]
): RangeLockInterval[] {
  const intervals: RangeLockInterval[] = [];

  lanes.forEach((lane) => {
    lane.ranges.forEach(([start, end], idx) => {
      intervals.push(
        createRangeLockInterval({
          id: `rl_${lane.trackId}_${start}_${end}_${idx}`,
          track_id: lane.trackId,
          frame_start: start,
          frame_end: end,
          label: `${lane.label} [${start}–${end}]`,
          faceset_id: lane.facesetId,
          source_index: lane.sourceIndex,
          color: lane.color,
          locked: true,
        })
      );
    });
  });

  return intervals;
}

/**
 * Sync Range Lock active intervals with Batch Matrix staged jobs.
 */
export function syncRangeLockToBatchMatrix(options: {
  targets: any[];
  targetIndex: number;
  activeIntervals: RangeLockInterval[];
  trackLanes?: ByteTrackLane[];
  sourceFacesInfo: any[];
  createJobPayload?: any;
  swapMode?: string;
  settings?: any;
}): { jobs: any[]; error: string | null } {
  return (stageRangeLockTracks as any)({
    targets: options.targets,
    targetIndex: options.targetIndex,
    activeIntervals: options.activeIntervals.map((iv) => ({
      trackId: iv.track_id,
      startFrame: iv.frame_start,
      endFrame: iv.frame_end,
      sourceIndex: iv.source_index,
      facesetId: iv.faceset_id,
      label: iv.label,
      active: iv.locked !== false,
    })),
    sourceFacesInfo: options.sourceFacesInfo,
    createJobPayload: options.createJobPayload,
    swapMode: options.swapMode,
    settings: options.settings,
  });
}

/**
 * Full trigger execution for Step 1: Pre-Scan.
 */
export async function triggerPreScan(
  targetIndex = 0,
  options: {
    relative?: boolean;
    maxGap?: number;
    apiBase?: string;
    sourceFacesInfo?: any[];
  } = {}
): Promise<{
  summary: TrajectorySummary;
  spans: TrajectorySpansResponse;
  lanes: ByteTrackLane[];
  warnings: WarningTick[];
  suggestedAssignments: SuggestedFszAssignment[];
}> {
  const [summary, spans] = await Promise.all([
    fetchTrajectorySummary(options.apiBase),
    fetchTrajectorySpans(options.apiBase, options.relative ?? true, options.maxGap ?? 1),
  ]);

  const lanes = formatTrackLanes(spans, options.sourceFacesInfo);
  const warnings = extractWarningTicks(spans, summary);
  const suggestedAssignments = deriveSuggestedAssignments(spans, options.sourceFacesInfo);

  return { summary, spans, lanes, warnings, suggestedAssignments };
}

// ── Custom Hook: useRunBarSync ────────────────────────────────────────────────

export interface UseRunBarSyncProps {
  targets?: any[];
  targetIndex?: number;
  sourceFacesInfo?: any[];
  settings?: any;
  telemetry?: any;
  onStageBatchJobs?: (jobs: any[]) => void;
  onStartRender?: () => void;
  onSelectTrackForPreview?: (trackId: number | null) => void;
  onSelectFaceForPreview?: (faceIndex: number | null) => void;
  notify?: (message: string, type?: 'info' | 'success' | 'warning' | 'error') => void;
}

export interface UseRunBarSyncReturn {
  activeStep: 1 | 2 | 3;
  setActiveStep: (step: 1 | 2 | 3) => void;
  isPreScanning: boolean;
  preScanError: string | null;

  trajectorySummary: TrajectorySummary | null;
  trajectorySpans: TrajectorySpansResponse | null;
  trackLanes: ByteTrackLane[];
  warningTicks: WarningTick[];
  suggestedAssignments: SuggestedFszAssignment[];

  activeIntervals: RangeLockInterval[];
  selectedTrackId: number | null;
  setSelectedTrackId: (trackId: number | null) => void;

  parityState: PreviewExportParityState;

  stagedJobs: any[];
  stagedJobsError: string | null;

  handleStepClick: (stepKey: RunBarStepKey | string, stepObj?: any) => Promise<void>;
  handlePreScan: () => Promise<void>;
  handleSelectTrack: (trackId: number | null) => void;
  handleToggleInterval: (intervalId: string) => void;
  handleUpdateIntervals: (newIntervals: RangeLockInterval[]) => void;
  handleAssignTrackFaceset: (trackId: number, sourceIndex: number, facesetId?: string) => void;
  handleRenderMux: () => void;
  handleSyncBatchMatrix: () => { jobs: any[]; error: string | null };
}

/**
 * Hook to connect the 3-step run bar (1 Pre-Scan -> 2 Range Lock -> 3 Render & Mux)
 * and Batch Matrix controls with NleTimeline.tsx and PreviewViewport.tsx.
 */
export function useRunBarSync({
  targets = [],
  targetIndex = 0,
  sourceFacesInfo = [],
  settings = {},
  telemetry = null,
  onStageBatchJobs,
  onStartRender,
  onSelectTrackForPreview,
  onSelectFaceForPreview,
  notify,
}: UseRunBarSyncProps): UseRunBarSyncReturn {
  const [activeStep, setActiveStep] = useState<1 | 2 | 3>(1);
  const [isPreScanning, setIsPreScanning] = useState(false);
  const [preScanError, setPreScanError] = useState<string | null>(null);

  const [trajectorySummary, setTrajectorySummary] = useState<TrajectorySummary | null>(null);
  const [trajectorySpans, setTrajectorySpans] = useState<TrajectorySpansResponse | null>(null);
  const [trackLanes, setTrackLanes] = useState<ByteTrackLane[]>([]);
  const [warningTicks, setWarningTicks] = useState<WarningTick[]>([]);
  const [suggestedAssignments, setSuggestedAssignments] = useState<SuggestedFszAssignment[]>([]);

  const [activeIntervals, setActiveIntervals] = useState<RangeLockInterval[]>([]);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);

  const [stagedJobs, setStagedJobs] = useState<any[]>([]);
  const [stagedJobsError, setStagedJobsError] = useState<string | null>(null);

  // Compute Preview Export Parity state
  const parityState = useMemo(() => {
    return bridgePreviewExportParity(telemetry, trajectorySummary);
  }, [telemetry, trajectorySummary]);

  // Sync active intervals with Batch Matrix staged jobs helper
  const syncToBatch = useCallback(
    (intervals: RangeLockInterval[], lanes: ByteTrackLane[]) => {
      if (!targets.length) return { jobs: [], error: null };

      const result = syncRangeLockToBatchMatrix({
        targets,
        targetIndex,
        activeIntervals: intervals,
        trackLanes: lanes,
        sourceFacesInfo,
        settings,
      });

      setStagedJobs(result.jobs);
      setStagedJobsError(result.error);

      if (result.jobs.length > 0 && onStageBatchJobs) {
        onStageBatchJobs(result.jobs);
      }

      return result;
    },
    [targets, targetIndex, sourceFacesInfo, settings, onStageBatchJobs]
  );

  // Step 1: Pre-Scan Handler
  const handlePreScan = useCallback(async () => {
    setIsPreScanning(true);
    setPreScanError(null);
    notify?.('Starting Pre-Scan: probing ByteTrack trajectory spans...', 'info');

    try {
      const res = await triggerPreScan(targetIndex, {
        sourceFacesInfo,
      });

      setTrajectorySummary(res.summary);
      setTrajectorySpans(res.spans);
      setTrackLanes(res.lanes);
      setWarningTicks(res.warnings);
      setSuggestedAssignments(res.suggestedAssignments);

      const intervals = convertTrackLanesToRangeLockIntervals(res.lanes);
      setActiveIntervals(intervals);

      // Advance to Step 2 Range Lock once scan completes
      setActiveStep(2);

      // Sync to Batch Matrix
      syncToBatch(intervals, res.lanes);

      const trackCount = res.lanes.length;
      const warnCount = res.warnings.length;
      notify?.(
        `Pre-Scan complete: ${trackCount} track${trackCount === 1 ? '' : 's'} locked, ${warnCount} warning${warnCount === 1 ? '' : 's'} identified`,
        'success'
      );
    } catch (err: any) {
      const msg = err?.message || 'Pre-Scan failed to fetch trajectory spans';
      setPreScanError(msg);
      notify?.(`Pre-Scan error: ${msg}`, 'error');
    } finally {
      setIsPreScanning(false);
    }
  }, [targetIndex, sourceFacesInfo, syncToBatch, notify]);

  // Select a track lane from NleTimeline
  const handleSelectTrack = useCallback(
    (trackId: number | null) => {
      setSelectedTrackId(trackId);

      // Trigger highlight in PreviewViewport
      if (onSelectTrackForPreview) {
        onSelectTrackForPreview(trackId);
      }

      // If a face index corresponds to this track in current telemetry, select it
      if (trackId !== null && onSelectFaceForPreview && telemetry?.faces) {
        const faceIndex = telemetry.faces.findIndex((f: any) => f.track_id === trackId);
        if (faceIndex >= 0) {
          onSelectFaceForPreview(faceIndex);
        }
      }

      // Ensure activeStep is Range Lock (Step 2)
      if (activeStep !== 2) {
        setActiveStep(2);
      }
    },
    [activeStep, onSelectTrackForPreview, onSelectFaceForPreview, telemetry]
  );

  // Toggle interval locked / active state
  const handleToggleInterval = useCallback(
    (intervalId: string) => {
      setActiveIntervals((prev) => {
        const next = prev.map((iv) => {
          if (iv.id === intervalId) {
            return { ...iv, locked: !iv.locked };
          }
          return iv;
        });
        syncToBatch(next, trackLanes);
        return next;
      });
    },
    [syncToBatch, trackLanes]
  );

  // Update intervals (e.g. after trim handle drag in NleTimeline)
  const handleUpdateIntervals = useCallback(
    (newIntervals: RangeLockInterval[]) => {
      setActiveIntervals(newIntervals);
      syncToBatch(newIntervals, trackLanes);
    },
    [syncToBatch, trackLanes]
  );

  // Assign a source faceset to a track
  const handleAssignTrackFaceset = useCallback(
    (trackId: number, sourceIndex: number, facesetId?: string) => {
      setTrackLanes((prev) =>
        prev.map((lane) => {
          if (lane.trackId === trackId) {
            const srcName = sourceFacesInfo[sourceIndex]?.name || `face_${sourceIndex + 1}`;
            return {
              ...lane,
              sourceIndex,
              facesetId: facesetId || sourceFacesInfo[sourceIndex]?.id || null,
              suggestedFsz: srcName.endsWith('.fsz') ? srcName : `${srcName}.fsz`,
            };
          }
          return lane;
        })
      );

      setActiveIntervals((prev) => {
        const next = prev.map((iv) => {
          if (iv.track_id === trackId) {
            return {
              ...iv,
              source_index: sourceIndex,
              faceset_id: facesetId || sourceFacesInfo[sourceIndex]?.id || null,
            };
          }
          return iv;
        });
        syncToBatch(next, trackLanes);
        return next;
      });

      notify?.(
        `Track #${trackId + 1} assigned to ${sourceFacesInfo[sourceIndex]?.name || `Faceset #${sourceIndex + 1}`}`,
        'info'
      );
    },
    [sourceFacesInfo, syncToBatch, trackLanes, notify]
  );

  // Step 3: Render & Mux Handler
  const handleRenderMux = useCallback(() => {
    setActiveStep(3);
    if (onStartRender) {
      onStartRender();
    }
  }, [onStartRender]);

  // Click handler for 3-step run bar steps
  const handleStepClick = useCallback(
    async (stepKey: RunBarStepKey | string, _stepObj?: any) => {
      switch (stepKey) {
        case 'prescan':
        case '1':
          await handlePreScan();
          break;
        case 'rangelock':
        case '2':
          setActiveStep(2);
          if (trackLanes.length === 0 && !isPreScanning) {
            await handlePreScan();
          }
          break;
        case 'render':
        case '3':
          handleRenderMux();
          break;
        default:
          break;
      }
    },
    [handlePreScan, handleRenderMux, trackLanes.length, isPreScanning]
  );

  // Re-sync with Batch Matrix
  const handleSyncBatchMatrix = useCallback(() => {
    return syncToBatch(activeIntervals, trackLanes);
  }, [syncToBatch, activeIntervals, trackLanes]);

  return {
    activeStep,
    setActiveStep,
    isPreScanning,
    preScanError,
    trajectorySummary,
    trajectorySpans,
    trackLanes,
    warningTicks,
    suggestedAssignments,
    activeIntervals,
    selectedTrackId,
    setSelectedTrackId,
    parityState,
    stagedJobs,
    stagedJobsError,
    handleStepClick,
    handlePreScan,
    handleSelectTrack,
    handleToggleInterval,
    handleUpdateIntervals,
    handleAssignTrackFaceset,
    handleRenderMux,
    handleSyncBatchMatrix,
  };
}

export default useRunBarSync;
