/**
 * FaceTrackLane: one ByteTrack lane (Track #N): header (swap on/off, range lock, faceset assignment)
 * and the presence bars with their Range Lock trim handles and pose / occlusion micro-indicators.
 *
 * What the header's two toggles do (both existed before this stage, now both are on the header):
 *   eye   the swap for this track on/off (`onToggleActive`, into the studio bindings: a muted track's
 *         faces are left untouched by the render)
 *   lock  Range Lock this track's ranges (Step 2): the same toggle a click on its bars makes
 * `detail` follows the group's lane height: compact keeps the name and toggles, normal adds the
 * faceset select and file chip (the pre-existing layout), tall makes the bars and dots larger.
 */

import React, { useMemo } from 'react';
import type { ByteTrackLane, SuggestedFszAssignment, WarningTick } from '../NleTimeline';
import { groupWarningRuns } from './faceHealth';
import type { RangeLockInterval } from '../../../utils/ptsFrameMap';
import type { LaneHeight } from './laneLayout';

export interface FaceTrackLaneProps {
  track: ByteTrackLane;
  color: string;
  isSelected: boolean;
  isLocked: boolean;
  suggested?: SuggestedFszAssignment;
  warnings: ReadonlyArray<WarningTick>;
  showWarnings: boolean;
  showSuggested: boolean;
  facesets: ReadonlyArray<{ id: string; name: string; path: string }>;
  intervals: ReadonlyArray<RangeLockInterval>;
  viewStart: number;
  viewEnd: number;
  maxFrames: number;
  height: LaneHeight;
  laneHeightPx: number;
  headerWidth: number;
  onRowClick: (e: React.MouseEvent) => void;
  onRangeClick: (range: [number, number], e: React.MouseEvent) => void;
  onToggleRangeLock: (e: React.MouseEvent) => void;
  onToggleActive?: (active: boolean) => void;
  onAssign?: (fsz: string) => void;
  onTrimDrag: (intervalId: string, handle: 'start' | 'end', e: React.PointerEvent<HTMLDivElement>) => void;
  /** keyboard nudge of a Range Lock edge, in frames (negative = earlier) */
  onTrimNudge: (intervalId: string, handle: 'start' | 'end', delta: number) => void;
  onWarningClick: (tick: WarningTick) => void;
}

const clampI = (n: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Math.round(n)));

export function FaceTrackLane({
  track, color, isSelected, isLocked, suggested, warnings, showWarnings, showSuggested, facesets, intervals,
  viewStart, viewEnd, maxFrames, height, laneHeightPx, headerWidth,
  onRowClick, onRangeClick, onToggleRangeLock, onToggleActive, onAssign, onTrimDrag, onTrimNudge, onWarningClick,
}: FaceTrackLaneProps) {
  const span = Math.max(1, viewEnd - viewStart);
  const pct = (f: number) => ((f - viewStart) / span) * 100;
  const fszLabel = suggested?.suggestedFileName || track.suggestedFsz || `track_${track.trackId + 1}.fsz`;
  const swapOn = track.active !== false;
  const detail = height !== 'compact';
  const barH = Math.max(14, Math.min(34, laneHeightPx - 14));
  const runs = useMemo(() => groupWarningRuns(warnings), [warnings]);

  return (
    // The row-wide click (select the track, toggle its Range Lock) is a POINTER shortcut: every action on it has a
    // keyboard path of its own (the Range Lock button in the header, the focusable range bars below).
    // eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-static-element-interactions
    <div
      onClick={onRowClick}
      className={`flex w-full cursor-pointer group transition-colors ${isSelected ? 'bg-[var(--accent,#e94560)]/10 ring-1 ring-inset ring-[var(--accent,#e94560)]/40' : 'hover:bg-white/[0.02]'} ${swapOn ? '' : 'opacity-60'}`}
      style={{ minHeight: laneHeightPx }}
      data-testid={`lane-track-${track.trackId}`}
    >
      <div className="shrink-0 flex flex-col justify-center gap-0.5 px-3 py-1.5 border-r border-white/10 bg-[var(--surface-2,#1a1d26)]/80" style={{ width: headerWidth }}>
        <div className="flex items-center justify-between gap-1.5">
          <div className="flex items-center gap-1.5 min-w-0">
            <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
            <span className="font-bold text-white text-mini truncate">{track.label}</span>
          </div>
          <div className="flex items-center gap-1 shrink-0">
            {onToggleActive && (
              <button
                type="button"
                aria-pressed={!swapOn}
                aria-label={swapOn ? `Mute swap for ${track.label}` : `Unmute swap for ${track.label}`}
                title={swapOn ? 'Swap ON for this face. Click to mute it (the render leaves it untouched)' : 'Swap MUTED for this face. Click to enable'}
                data-testid={`lane-track-${track.trackId}-mute`}
                onClick={(e) => { e.stopPropagation(); onToggleActive(!swapOn); }}
                className={`p-1 rounded transition ${swapOn ? 'text-neutral-400 hover:text-white' : 'text-amber-300 bg-amber-500/15'}`}
              >
                <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12z" /><circle cx="12" cy="12" r="3" />{!swapOn && <path d="M3 3l18 18" />}
                </svg>
              </button>
            )}
            <button
              type="button"
              aria-pressed={isLocked}
              aria-label={isLocked ? `Range lock on for ${track.label}` : `Range lock ${track.label}`}
              onClick={(e) => { e.stopPropagation(); onToggleRangeLock(e); }}
              className={`p-1 rounded transition ${isLocked ? 'text-[var(--accent,#e94560)] hover:text-white' : 'text-neutral-500 hover:text-neutral-300'}`}
              title={isLocked ? 'Range Locked (Step 2 Active)' : 'Click to Range Lock'}
            >
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                {isLocked
                  ? <path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3.1-9H8.9V6c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2z" />
                  : <path d="M12 17c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm6-9h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6h1.9c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm0 12H6V10h12v10z" />}
              </svg>
            </button>
          </div>
        </div>

        {detail && facesets.length > 0 && (
          <select
            aria-label={`Faceset for Track #${track.trackId + 1}`}
            value={track.facesetPath || track.facesetId || ''}
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => onAssign?.(e.target.value)}
            className="w-full bg-black/40 text-white text-mini"
          >
            <option value="" disabled>Choose faceset</option>
            {facesets.map((s) => <option key={s.id} value={s.path}>{s.name}</option>)}
          </select>
        )}
        {detail && showSuggested && (
          <div className="flex items-center justify-between gap-1">
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); onAssign?.(fszLabel); }}
              title={`Assigned/Suggested Faceset: ${fszLabel} (${track.via})`}
              className="flex items-center gap-1 max-w-[190px] px-1.5 py-0.5 rounded bg-black/40 border border-white/10 hover:border-white/20 text-micro font-mono text-neutral-300 truncate transition"
            >
              <span style={{ color }}>●</span>
              <span className="truncate">{fszLabel}</span>
            </button>
            <span className="text-micro font-mono text-neutral-500">{track.totalFrames}f</span>
          </div>
        )}
      </div>

      <div className="relative flex-1 overflow-hidden flex items-center" style={{ minHeight: laneHeightPx }}>
        {track.ranges.map(([rStart, rEnd], i) => {
          const start = clampI(rStart, 0, maxFrames - 1);
          const end = clampI(rEnd, start, maxFrames - 1);
          if (end < viewStart || start > viewEnd) return null;
          const left = Math.max(0, pct(start));
          const right = Math.min(100, pct(end));
          const activeInterval = intervals.find((iv) => iv.track_id === track.trackId && iv.frame_start === start && iv.frame_end === end);
          const locked = activeInterval?.locked !== false && isLocked;
          const widthPct = Math.max(0.5, right - left);
          const barStyle = { height: barH, left: `${left}%`, width: `${widthPct}%` };
          return (
            <React.Fragment key={`${track.trackId}_${i}`}>
              <div
                role="button"
                tabIndex={0}
                aria-pressed={locked}
                aria-label={`Range lock frames ${start} to ${end} of ${track.label}`}
                onClick={(e) => onRangeClick([start, end], e)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); onRangeClick([start, end], e as unknown as React.MouseEvent); }
                }}
                className={`absolute rounded-md shadow-md flex items-center justify-between px-2 cursor-pointer transition-all border focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-cyan,#00F0FF)] ${locked ? 'border-white/50 ring-2 ring-[var(--accent,#e94560)]/80 brightness-110' : 'border-white/20 hover:border-white/40'}`}
                style={{ ...barStyle, background: `linear-gradient(90deg, ${color}cc, ${color}88)` }}
                title={`Click to Range Lock [${start}–${end}] (${end - start + 1} frames)`}
              >
                <div className="flex items-center gap-1.5 overflow-hidden text-mini font-mono font-semibold text-black/90 pl-2.5">
                  <span className="truncate">{start}–{end}</span>
                  {locked && <span className="text-micro px-1 bg-black/40 text-white rounded">LOCKED</span>}
                </div>
              </div>
              {/* The trim handles are SIBLINGS of the bar, not children: an interactive control inside a button is
                  invalid (axe `nested-interactive`) and its clicks needed stopPropagation to avoid toggling the lock. */}
              {locked && activeInterval && (
                <>
                  <div
                    role="slider" tabIndex={0} aria-label="Trim start of interval" aria-valuemin={0} aria-valuemax={maxFrames - 1} aria-valuenow={activeInterval.frame_start}
                    onPointerDown={(e) => onTrimDrag(activeInterval.id, 'start', e)}
                    onKeyDown={(e) => { const d = e.key === 'ArrowLeft' ? -1 : e.key === 'ArrowRight' ? 1 : 0; if (d) { e.preventDefault(); e.stopPropagation(); onTrimNudge(activeInterval.id, 'start', d * (e.shiftKey ? 10 : 1)); } }}
                    className="absolute w-2.5 bg-black/50 hover:bg-black/80 rounded-l cursor-ew-resize flex items-center justify-center text-micro text-white font-bold z-10 focus:outline-none focus-visible:ring-2 focus-visible:ring-white"
                    style={{ height: barH, left: `${left}%` }}
                    title="Trim start of interval (drag, or ←/→)"
                  >|</div>
                  <div
                    role="slider" tabIndex={0} aria-label="Trim end of interval" aria-valuemin={0} aria-valuemax={maxFrames - 1} aria-valuenow={activeInterval.frame_end}
                    onPointerDown={(e) => onTrimDrag(activeInterval.id, 'end', e)}
                    onKeyDown={(e) => { const d = e.key === 'ArrowLeft' ? -1 : e.key === 'ArrowRight' ? 1 : 0; if (d) { e.preventDefault(); e.stopPropagation(); onTrimNudge(activeInterval.id, 'end', d * (e.shiftKey ? 10 : 1)); } }}
                    className="absolute w-2.5 bg-black/50 hover:bg-black/80 rounded-r cursor-ew-resize flex items-center justify-center text-micro text-white font-bold z-10 focus:outline-none focus-visible:ring-2 focus-visible:ring-white"
                    style={{ height: barH, left: `${left + widthPct}%`, transform: 'translateX(-100%)' }}
                    title="Trim end of interval (drag, or ←/→)"
                  >|</div>
                </>
              )}
            </React.Fragment>
          );
        })}

        {showWarnings && runs.map((run, i) => {
          if (run.end < viewStart || run.start > viewEnd) return null;
          const left = Math.max(0, pct(run.start));
          const right = Math.min(100, pct(run.end + 1));
          // Classify by TYPE. `value` means different things per type (yaw in degrees for a pose tick,
          // percent occluded for an occlusion tick), so the old `value > 25` test painted every pose
          // tick (they are only emitted above 45 degrees of yaw) as a red "Foreign Occlusion".
          const occlusion = run.type === 'occlusion';
          const hardYaw = run.type === 'pose' && (run.peak === 0 || run.peak > 50);
          const color = occlusion ? '#f43f5e' : hardYaw ? '#f59e0b' : '#38bdf8';
          const what = occlusion ? 'Foreign Occlusion (>25%)' : hardYaw ? 'Hard Yaw (>50°)' : run.message || 'Warning';
          const where = run.start === run.end ? `Frame #${run.start}` : `Frames #${run.start}–#${run.end}`;
          const samples = run.count > 1 ? ` · ${run.count} samples` : '';
          return (
            <div
              key={`w_${i}`}
              // not a tab stop (a clip can have dozens of runs): the keyboard way to walk them is the coverage lane's prev / next
              role="button"
              tabIndex={-1}
              aria-label={`${what}, ${where}`}
              onClick={(e) => { e.stopPropagation(); onWarningClick({ frame: run.start, trackId: track.trackId, type: run.type, severity: 'warning', message: what }); }}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); onWarningClick({ frame: run.start, trackId: track.trackId, type: run.type, severity: 'warning', message: what }); } }}
              className="absolute bottom-[3px] rounded-full border border-black/70 shadow-md hover:brightness-125 cursor-pointer z-20"
              style={{ left: `${left}%`, width: `${Math.max(0.2, right - left)}%`, minWidth: 6, height: height === 'tall' ? 7 : 5, backgroundColor: color }}
              title={`[${where}] ${what}${samples} - Click to jump`}
              data-testid="nle-warning-run"
              data-run-type={run.type}
            />
          );
        })}
      </div>
    </div>
  );
}

export default FaceTrackLane;
