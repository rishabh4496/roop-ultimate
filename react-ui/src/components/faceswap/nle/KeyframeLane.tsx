/**
 * KeyframeLane: body of the KF lane (parameter keyframes with their interpolation curves).
 * Ported from NleTimeline, positioned on the shared lane area; the lane lock refuses add / move /
 * delete / value edits (selection and seeking still work).
 */

import React from 'react';
import { PARAMETER_CONFIGS, type NleKeyframe, type TimelineCurveType } from '../timeline/parameterKeyframes';

export interface KeyframeLaneProps {
  keyframes: ReadonlyArray<NleKeyframe>;
  viewStart: number;
  viewEnd: number;
  widthPx: number;
  heightPx: number;
  selectedId: string | null;
  locked: boolean;
  onSelect: (id: string) => void;
  onSeek: (frame: number) => void;
  onDragStart: (id: string, e: React.PointerEvent<HTMLDivElement>) => void;
  onPatch: (id: string, patch: Partial<NleKeyframe>) => void;
  onDelete: (id: string) => void;
  onScrubDown: (e: React.PointerEvent<HTMLDivElement>) => void;
}

const CURVES: TimelineCurveType[] = ['Linear', 'Ease In/Out', 'Hold'];

export function KeyframeLane({ keyframes, viewStart, viewEnd, widthPx, heightPx, selectedId, locked, onSelect, onSeek, onDragStart, onPatch, onDelete, onScrubDown }: KeyframeLaneProps) {
  const span = Math.max(1, viewEnd - viewStart);
  const xPx = (f: number) => ((f - viewStart) / span) * widthPx;
  const y = heightPx / 2;
  const selected = keyframes.find((k) => k.id === selectedId) || null;
  const cfg = selected ? PARAMETER_CONFIGS[selected.param] : null;

  return (
    <div className="absolute inset-0 overflow-hidden" onPointerDown={onScrubDown}>
      <svg className="absolute inset-0 w-full h-full pointer-events-none" aria-hidden="true">
        {keyframes.map((kf, idx) => {
          const next = keyframes[idx + 1];
          if (!next) return null;
          const x1 = xPx(kf.frame_idx);
          const x2 = xPx(next.frame_idx);
          if (kf.curve === 'Hold') {
            return <path key={`h_${kf.id}`} d={`M ${x1} ${y} H ${x2}`} stroke="rgba(251,191,36,0.4)" strokeWidth="1.5" strokeDasharray="3 3" fill="none" />;
          }
          if (kf.curve === 'Linear') {
            return <line key={`l_${kf.id}`} x1={x1} y1={y} x2={x2} y2={y} stroke="rgba(251,191,36,0.7)" strokeWidth="2" />;
          }
          const dx = (x2 - x1) * 0.5;
          return <path key={`e_${kf.id}`} d={`M ${x1} ${y} C ${x1 + dx} ${y - 4}, ${x2 - dx} ${y + 4}, ${x2} ${y}`} stroke="rgba(251,191,36,0.85)" strokeWidth="2" fill="none" />;
        })}
      </svg>

      {keyframes.map((kf) => {
        const left = ((kf.frame_idx - viewStart) / span) * 100;
        if (left < -5 || left > 105) return null;
        const isSelected = selectedId === kf.id;
        const color = PARAMETER_CONFIGS[kf.param]?.color || '#fbbf24';
        return (
          <div
            key={kf.id}
            role="button"
            tabIndex={0}
            aria-label={`${kf.param} keyframe at frame ${kf.frame_idx}, value ${kf.value}, ${kf.curve}`}
            data-testid="nle-keyframe"
            data-frame={kf.frame_idx}
            onPointerDown={(e) => { if (!locked) onDragStart(kf.id, e); else e.stopPropagation(); }}
            onClick={(e) => { e.stopPropagation(); onSelect(kf.id); onSeek(kf.frame_idx); }}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); onSelect(kf.id); onSeek(kf.frame_idx); } }}
            className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3.5 h-3.5 rotate-45 border shadow-md hover:scale-125 transition-transform z-20 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-cyan,#00F0FF)] ${locked ? 'cursor-pointer' : 'cursor-ew-resize'} ${isSelected ? 'border-[var(--accent-cyan,#00F0FF)] ring-2 ring-[var(--accent-cyan,#00F0FF)]/80 scale-125' : 'border-black'}`}
            style={{ left: `${left}%`, backgroundColor: isSelected ? '#ffffff' : color }}
            title={`Keyframe: ${kf.param} = ${kf.value} (${kf.curve}) at frame #${kf.frame_idx}${locked ? ' (lane locked)' : ' (Drag to move, Alt+Drag to copy)'}`}
          />
        );
      })}

      {selected && cfg && (
        <div
          onPointerDown={(e) => e.stopPropagation()}
          className="absolute right-3 top-1/2 -translate-y-1/2 z-30 flex items-center gap-1.5 px-2 py-0.5 bg-black/85 border border-white/20 rounded shadow-lg backdrop-blur"
        >
          <span className="text-mini font-mono text-neutral-300">#{selected.frame_idx}:</span>
          {cfg.choices ? (
            <select aria-label="Keyframe value" disabled={locked} value={selected.value} onChange={(e) => onPatch(selected.id, { value: Number(e.target.value), curve: 'Hold' })} className="bg-black text-white text-mini">
              {cfg.choices.map((v, i) => <option key={i} value={i}>{String(v)}</option>)}
            </select>
          ) : (
            <input
              aria-label="Keyframe value"
              type="number"
              disabled={locked}
              className="w-16 bg-black text-white text-mini"
              min={cfg.min}
              max={cfg.max}
              step={cfg.step}
              value={selected.value}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (Number.isFinite(v)) onPatch(selected.id, { value: Math.max(cfg.min, Math.min(cfg.max, v)) });
              }}
            />
          )}
          {CURVES.map((c) => (
            <button
              key={c}
              type="button"
              disabled={locked}
              onClick={() => onPatch(selected.id, { curve: c })}
              className={`px-1.5 py-0.5 rounded text-mini font-medium transition disabled:opacity-40 ${selected.curve === c ? 'bg-amber-400 text-black font-bold' : 'bg-white/10 text-neutral-300 hover:text-white'}`}
            >
              {c}
            </button>
          ))}
          <button type="button" disabled={locked} onClick={() => onDelete(selected.id)} className="p-1 hover:bg-rose-500/20 text-rose-400 rounded transition disabled:opacity-40" title="Delete Keyframe" aria-label="Delete keyframe">
            <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
          </button>
        </div>
      )}
    </div>
  );
}

export default KeyframeLane;
