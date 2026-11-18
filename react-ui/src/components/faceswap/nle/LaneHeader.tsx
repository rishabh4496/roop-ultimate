/**
 * LaneHeader: the left track header of one lane: a badge and name, then Hide (or Mute) / Lock / Height.
 *
 * The three toggles are real state (see laneLayout.ts for what each one does to its lane), exposed as
 * toggle buttons (`aria-pressed`) so they are operable by keyboard and announced by a screen reader.
 */

import React, { type ReactNode } from 'react';
import type { LaneHeight, LaneState } from './laneLayout';

export const LANE_HEADER_CLASS =
  'shrink-0 flex items-center gap-1.5 px-2 border-r border-white/10 bg-[var(--surface-2,#1a1d26)] z-10';

const ICON = { width: 14, height: 14, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const };

const EyeIcon = ({ off }: { off: boolean }) => (
  <svg {...ICON} aria-hidden="true">
    <path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12z" />
    <circle cx="12" cy="12" r="3" />
    {off && <path d="M3 3l18 18" />}
  </svg>
);
const SpeakerIcon = ({ off }: { off: boolean }) => (
  <svg {...ICON} aria-hidden="true">
    <path d="M11 5L6 9H2v6h4l5 4V5z" />
    {off ? <path d="M23 9l-6 6M17 9l6 6" /> : <path d="M15.5 8.5a5 5 0 0 1 0 7M19 5a10 10 0 0 1 0 14" />}
  </svg>
);
const LockIcon = ({ on }: { on: boolean }) => (
  <svg {...ICON} aria-hidden="true">
    <rect x="4" y="11" width="16" height="10" rx="2" />
    {on ? <path d="M8 11V7a4 4 0 0 1 8 0v4" /> : <path d="M8 11V7a4 4 0 0 1 7.5-2" />}
  </svg>
);
const HeightIcon = ({ level }: { level: LaneHeight }) => (
  <svg {...ICON} aria-hidden="true">
    <path d="M12 3v18M8 7l4-4 4 4M8 17l4 4 4-4" opacity={level === 'compact' ? 0.55 : 1} />
    <path d={level === 'tall' ? 'M4 3h16M4 21h16' : level === 'normal' ? 'M6 6h12M6 18h12' : 'M8 9h8M8 15h8'} />
  </svg>
);

const btn =
  'inline-flex items-center justify-center w-6 h-6 rounded border transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-cyan,#00F0FF)]';
const on = 'border-[var(--accent-cyan,#00F0FF)]/60 bg-[var(--accent-cyan,#00F0FF)]/15 text-[var(--accent-cyan,#00F0FF)]';
const off = 'border-white/10 bg-white/5 text-neutral-400 hover:text-white hover:border-white/25';

export interface LaneHeaderProps {
  laneId: string;
  badge: string;
  badgeClass: string;
  title: string;
  state: LaneState;
  /** the eye is a Mute speaker on the audio lane */
  hideMode?: 'hide' | 'mute';
  onToggleHidden?: () => void;
  onToggleLocked?: () => void;
  onCycleHeight?: () => void;
  /** appears between the title and the toggles */
  extra?: ReactNode;
  /** short readout under the title (e.g. "93% face") */
  status?: ReactNode;
  width: number;
  height: number;
}

export function LaneHeader({
  laneId,
  badge,
  badgeClass,
  title,
  state,
  hideMode = 'hide',
  onToggleHidden,
  onToggleLocked,
  onCycleHeight,
  extra,
  status,
  width,
  height,
}: LaneHeaderProps) {
  const hideLabel = hideMode === 'mute' ? (state.hidden ? `Unmute ${title}` : `Mute ${title}`) : state.hidden ? `Show ${title}` : `Hide ${title}`;
  return (
    <div className={LANE_HEADER_CLASS} style={{ width, minHeight: height }} data-testid={`lane-${laneId}-header`}>
      <span className={`px-1.5 py-0.5 rounded font-mono font-bold text-micro ${badgeClass}`}>{badge}</span>
      <div className="min-w-0 flex-1 leading-tight">
        <div className="truncate font-semibold text-white text-mini">{title}</div>
        {status ? <div className="truncate text-neutral-400 text-micro font-mono">{status}</div> : null}
      </div>
      {extra}
      {onToggleHidden && (
        <button
          type="button"
          className={`${btn} ${state.hidden ? on : off}`}
          aria-pressed={state.hidden}
          aria-label={hideLabel}
          title={hideLabel}
          data-testid={`lane-${laneId}-hide`}
          onClick={(e) => { e.stopPropagation(); onToggleHidden(); }}
        >
          {hideMode === 'mute' ? <SpeakerIcon off={state.hidden} /> : <EyeIcon off={state.hidden} />}
        </button>
      )}
      {onToggleLocked && (
        <button
          type="button"
          className={`${btn} ${state.locked ? on : off}`}
          aria-pressed={state.locked}
          aria-label={state.locked ? `Unlock ${title}` : `Lock ${title}`}
          title={state.locked ? `Unlock ${title} (edits allowed)` : `Lock ${title} (refuse edits)`}
          data-testid={`lane-${laneId}-lock`}
          onClick={(e) => { e.stopPropagation(); onToggleLocked(); }}
        >
          <LockIcon on={state.locked} />
        </button>
      )}
      {onCycleHeight && (
        <button
          type="button"
          className={`${btn} ${off}`}
          aria-label={`${title} height: ${state.height}. Click for the next size`}
          title={`Track height: ${state.height} (click to cycle)`}
          data-testid={`lane-${laneId}-height`}
          data-height={state.height}
          onClick={(e) => { e.stopPropagation(); onCycleHeight(); }}
        >
          <HeightIcon level={state.height} />
        </button>
      )}
    </div>
  );
}

export default LaneHeader;
