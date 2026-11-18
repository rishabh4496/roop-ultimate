/**
 * TransportDeck: the studio transport bar above the lanes.
 *
 *   |<  <M  <|  PLAY  |>  M>  >|     [loop In-Out 2s 04f]     0.25x 0.5x 1x 2x 4x     J K L
 *
 * It owns nothing. Play, rate, loop and stepping are the SAME state the keyboard drives (FaceSwap's
 * isPlaying / playbackRate / isLooping and useWorkstationShortcuts.stepFrame), so a button press and a
 * key press cannot disagree. Where a key exists its hint is in the tooltip; the deck does not add key
 * handlers of its own (Space / arrows / Home / End / J K L live in useWorkstationShortcuts).
 */

import React from 'react';
import { PLAYBACK_RATES, formatDuration, formatRate, loopRegion, neighbourFrame, regionLength, snapRate } from './transportModel';

export interface TransportDeckProps {
  frame: number;
  maxFrames: number;
  fps: number;
  isPlaying: boolean;
  playbackRate: number;
  loop: boolean;
  trimIn: number;
  trimOut: number;
  /** 0-based frames of every marker */
  markerFrames: ReadonlyArray<number>;
  /** "▶ 2×" / "◀ 1×" while a J/K/L shuttle is active */
  shuttleLabel?: string;
  /** -1 reverse shuttle active, 0 none, 1 forward */
  shuttleDir?: number;
  onSeek: (frame: number) => void;
  onStep: (delta: number) => void;
  onTogglePlay: () => void;
  onRate: (rate: number) => void;
  onToggleLoop: () => void;
  onShuttle?: (key: 'j' | 'k' | 'l') => void;
  className?: string;
}

const I = { width: 16, height: 16, viewBox: '0 0 24 24', fill: 'currentColor', 'aria-hidden': true } as const;
const Icons = {
  start: <svg {...I}><path d="M6 5h2v14H6zM20 5v14L9 12z" /></svg>,
  end: <svg {...I}><path d="M16 5h2v14h-2zM4 5v14l11-7z" /></svg>,
  prevMarker: <svg {...I}><path d="M5 3h2v18H5zM9 4v7l7-3.5zM9 13v7l7-3.5z" opacity=".9" /></svg>,
  nextMarker: <svg {...I}><path d="M17 3h2v18h-2zM15 4v7L8 7.5zM15 13v7l-7-3.5z" opacity=".9" /></svg>,
  stepBack: <svg {...I}><path d="M15 6l-8 6 8 6V6z" /></svg>,
  stepFwd: <svg {...I}><path d="M9 6l8 6-8 6V6z" /></svg>,
  play: <svg {...I} width={20} height={20}><path d="M8 5v14l11-7z" /></svg>,
  pause: <svg {...I} width={20} height={20}><path d="M7 5h4v14H7zM13 5h4v14h-4z" /></svg>,
  loop: <svg {...I} fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M17 2l4 4-4 4" /><path d="M3 11V9a3 3 0 0 1 3-3h15" /><path d="M7 22l-4-4 4-4" /><path d="M21 13v2a3 3 0 0 1-3 3H3" /></svg>,
};

const base =
  'inline-flex items-center justify-center h-8 min-w-8 px-2 rounded-md border transition-colors select-none focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-cyan,#00F0FF)] disabled:opacity-35 disabled:pointer-events-none';
const idle = 'border-white/10 bg-white/5 text-neutral-200 hover:bg-white/10 hover:text-white hover:border-white/25 active:translate-y-px';
const active = 'border-[var(--accent-cyan,#00F0FF)]/70 bg-[var(--accent-cyan,#00F0FF)]/15 text-[var(--accent-cyan,#00F0FF)]';

function TransportDeckImpl({
  frame, maxFrames, fps, isPlaying, playbackRate, loop, trimIn, trimOut, markerFrames,
  shuttleLabel = '', shuttleDir = 0, onSeek, onStep, onTogglePlay, onRate, onToggleLoop, onShuttle, className = '',
}: TransportDeckProps) {
  const last = Math.max(0, maxFrames - 1);
  const prevMarker = neighbourFrame(markerFrames, frame, -1);
  const nextMarker = neighbourFrame(markerFrames, frame, 1);
  const region = loopRegion(trimIn, trimOut, maxFrames);
  const isWhole = region.in === 0 && region.out === last;
  const rate = snapRate(playbackRate);

  return (
    <div
      role="toolbar"
      aria-label="Transport"
      data-testid="nle-transport"
      className={`flex flex-wrap items-center justify-center gap-x-4 gap-y-1 px-3 py-1 bg-[var(--surface-2,#1a1d26)] border-b border-white/10 ${className}`}
    >
      <div className="flex items-center gap-1" role="group" aria-label="Playhead">
        <button type="button" className={`${base} ${idle}`} onClick={() => onSeek(0)} disabled={frame <= 0} aria-label="Jump to start" title="Jump to start (Home)" data-testid="transport-start">{Icons.start}</button>
        <button type="button" className={`${base} ${idle}`} onClick={() => prevMarker !== null && onSeek(prevMarker)} disabled={prevMarker === null} aria-label="Previous marker" title="Jump to the previous marker" data-testid="transport-prev-marker">{Icons.prevMarker}</button>
        <button type="button" className={`${base} ${idle}`} onClick={() => onStep(-1)} disabled={frame <= 0} aria-label="Previous frame" title="Previous frame (←). Shift+← steps 10" data-testid="transport-step-back">{Icons.stepBack}</button>
        <button
          type="button"
          className={`${base} w-11 ${isPlaying ? active : idle}`}
          onClick={onTogglePlay}
          aria-label={isPlaying ? 'Pause' : 'Play'}
          aria-pressed={isPlaying}
          title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
          data-testid="transport-play"
        >
          {isPlaying ? Icons.pause : Icons.play}
        </button>
        <button type="button" className={`${base} ${idle}`} onClick={() => onStep(1)} disabled={frame >= last} aria-label="Next frame" title="Next frame (→). Shift+→ steps 10" data-testid="transport-step-fwd">{Icons.stepFwd}</button>
        <button type="button" className={`${base} ${idle}`} onClick={() => nextMarker !== null && onSeek(nextMarker)} disabled={nextMarker === null} aria-label="Next marker" title="Jump to the next marker" data-testid="transport-next-marker">{Icons.nextMarker}</button>
        <button type="button" className={`${base} ${idle}`} onClick={() => onSeek(last)} disabled={frame >= last} aria-label="Jump to end" title="Jump to end (End)" data-testid="transport-end">{Icons.end}</button>
      </div>

      <button
        type="button"
        className={`${base} gap-1.5 ${loop ? active : idle}`}
        onClick={onToggleLoop}
        aria-pressed={loop}
        aria-label={loop ? 'Loop In to Out: on' : 'Loop In to Out: off'}
        title={loop ? 'Looping the In–Out region. Click to play through' : 'Loop the In–Out region'}
        data-testid="transport-loop"
      >
        {Icons.loop}
        <span className="text-mini font-mono tabular-nums">
          {isWhole ? 'whole clip' : `${formatDuration(regionLength(region), fps)}`}
        </span>
      </button>

      <div className="flex items-center gap-1" role="radiogroup" aria-label="Playback speed" data-testid="transport-rates">
        {PLAYBACK_RATES.map((r) => (
          <button
            key={r}
            type="button"
            role="radio"
            aria-checked={rate === r}
            className={`${base} text-mini font-mono tabular-nums ${rate === r ? active : idle}`}
            onClick={() => onRate(r)}
            title={`Play at ${formatRate(r)}`}
            data-testid={`transport-rate-${r}`}
          >
            {formatRate(r)}
          </button>
        ))}
      </div>

      {onShuttle && (
        <div className="flex items-center gap-1" role="group" aria-label="Shuttle (J K L)">
          <button type="button" className={`${base} font-mono text-mini ${shuttleDir < 0 ? active : idle}`} onClick={() => onShuttle('j')} title="Shuttle backward. Press again to go faster (J)" aria-label="Shuttle backward" data-testid="transport-shuttle-j">J</button>
          <button type="button" className={`${base} font-mono text-mini ${idle}`} onClick={() => onShuttle('k')} title="Stop shuttle (K)" aria-label="Stop shuttle" data-testid="transport-shuttle-k">K</button>
          <button type="button" className={`${base} font-mono text-mini ${shuttleDir > 0 ? active : idle}`} onClick={() => onShuttle('l')} title="Shuttle forward. Press again to go faster (L)" aria-label="Shuttle forward" data-testid="transport-shuttle-l">L</button>
          <span className="min-w-12 text-mini font-mono text-[var(--accent-cyan,#00F0FF)]" aria-live="polite" data-testid="transport-shuttle-label">{shuttleLabel}</span>
        </div>
      )}
    </div>
  );
}

export const TransportDeck = React.memo(TransportDeckImpl);
export default TransportDeck;
