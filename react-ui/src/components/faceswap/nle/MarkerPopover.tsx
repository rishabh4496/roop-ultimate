/**
 * MarkerPopover: edit one marker's colour tag and note. Portalled and `position: fixed`, so the
 * timeline's `overflow-hidden` cannot clip it. Edits apply as you type (the store is a few bytes of
 * localStorage); Escape or a click outside closes it.
 */

import React, { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import type { ChapterMarker } from '../timeline/timelineSnap';
import { MARKER_COLORS, MAX_NOTE_CHARS } from './markerModel';
import { formatFrameLabel } from './rulerModel';

export interface MarkerPopoverProps {
  marker: ChapterMarker;
  timecode: string;
  /** viewport rect of the marker's flag */
  anchor: { left: number; top: number };
  locked: boolean;
  onChange: (marker: ChapterMarker) => void;
  onDelete: (id: string) => void;
  onGoTo: (frame: number) => void;
  onClose: () => void;
}

const W = 252;

export function MarkerPopover({ marker, timecode, anchor, locked, onChange, onDelete, onGoTo, onClose }: MarkerPopoverProps) {
  const boxRef = useRef<HTMLDivElement>(null);
  const noteRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    noteRef.current?.focus({ preventScroll: true });
  }, [marker.id]);

  useEffect(() => {
    const onDown = (e: PointerEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
      }
    };
    document.addEventListener('pointerdown', onDown, true);
    document.addEventListener('keydown', onKey, true);
    return () => {
      document.removeEventListener('pointerdown', onDown, true);
      document.removeEventListener('keydown', onKey, true);
    };
  }, [onClose]);

  if (typeof document === 'undefined') return null;
  const vw = window.innerWidth;
  const left = Math.max(8, Math.min(vw - W - 8, anchor.left - W / 2));
  const bottom = Math.max(8, window.innerHeight - anchor.top + 10);

  return createPortal(
    <div
      ref={boxRef}
      role="dialog"
      aria-label={`Marker at ${formatFrameLabel(marker.frame)}`}
      data-testid="nle-marker-popover"
      className="fixed z-[2100] rounded-xl border border-white/15 bg-[#12141a] p-3 shadow-2xl text-white"
      style={{ left, bottom, width: W }}
    >
      <div className="flex items-baseline justify-between gap-2 mb-2">
        <span className="text-plain font-bold font-mono tabular-nums" style={{ color: marker.color }}>{formatFrameLabel(marker.frame)}</span>
        <span className="text-mini font-mono text-neutral-400">{timecode}</span>
      </div>
      <div className="flex items-center gap-1.5 mb-2" role="radiogroup" aria-label="Marker colour">
        {MARKER_COLORS.map((c) => (
          <button
            key={c.id}
            type="button"
            role="radio"
            aria-checked={marker.color === c.hex}
            aria-label={c.label}
            title={c.label}
            disabled={locked}
            data-testid={`marker-color-${c.id}`}
            onClick={() => onChange({ ...marker, color: c.hex })}
            className={`w-6 h-6 rounded-full border-2 transition-transform disabled:opacity-40 ${marker.color === c.hex ? 'border-white scale-110' : 'border-transparent hover:scale-110'}`}
            style={{ backgroundColor: c.hex }}
          />
        ))}
      </div>
      <label className="block text-mini text-neutral-300 mb-1" htmlFor="nle-marker-note">Note</label>
      <textarea
        id="nle-marker-note"
        ref={noteRef}
        value={marker.note || ''}
        maxLength={MAX_NOTE_CHARS}
        readOnly={locked}
        rows={3}
        placeholder={locked ? 'Markers lane is locked' : 'Dialogue beat, problem frame, review note…'}
        onChange={(e) => onChange({ ...marker, note: e.target.value })}
        className="w-full resize-none rounded-md border border-white/15 bg-black/50 px-2 py-1 text-compact text-white placeholder:text-neutral-500 focus:outline-none focus:border-[var(--accent-cyan,#00F0FF)]/70"
        data-testid="marker-note"
      />
      <div className="mt-2 flex items-center justify-between gap-2">
        <button type="button" onClick={() => onGoTo(marker.frame)} className="px-2 py-1 rounded-md border border-white/15 bg-white/5 hover:bg-white/10 text-mini">Go to</button>
        <span className="text-mini text-neutral-500 font-mono tabular-nums">{(marker.note || '').length}/{MAX_NOTE_CHARS}</span>
        <button
          type="button"
          disabled={locked}
          onClick={() => onDelete(marker.id)}
          className="px-2 py-1 rounded-md border border-rose-400/40 bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 text-mini disabled:opacity-40"
          data-testid="marker-delete"
        >
          Delete
        </button>
      </div>
    </div>,
    document.body,
  );
}

export default MarkerPopover;
