import React, { useRef, useState } from 'react';
import { curvePath, uiFrameOf } from '../identityKeyframes';
import { frameAtX, normalizeView, viewSpan } from './viewMath';

// The Identity Keyframe Track: one row beneath the film strip (where the chapter markers are carried down), with a
// diamond per identity keyframe and, between neighbours, a glyph of the curve that segment follows (a step for
// Hold, a diagonal for Linear, the eased curve for Bezier).
//
// Gestures: double-click empty track adds a keyframe there (seeded with the blend you can already see, so adding one
// changes nothing by itself); click a diamond to select it and park the playhead on it; drag a diamond to move it
// (a ghost follows the pointer and the move commits on release, so one drag is one recipe change, not one per
// pixel); with a diamond focused, Left / Right nudge a frame (Shift: ten), Delete removes it.
//
// Geometry is DOM, not canvas: a clip holds at most MAX_KEYFRAMES (64) keyframes, every one a real button with a
// label, so it is keyboard- and screen-reader-reachable, unlike the canvas lanes whose contents are bulk data.

export const IDENTITY_LANE_H = 30;
const DIAMOND = 12;

export default function IdentityKeyframeLane({
  view, maxFrames, frame, keyframes, selected, canKey, markers = [],
  onSeek, onAdd, onMove, onRemove, onSelect, emptyNote,
}) {
  const laneRef = useRef(null);
  const dragRef = useRef(null);
  const [ghost, setGhost] = useState(null);           // { frameIndex, uiFrame } while a diamond is being dragged
  const v = normalizeView(view, maxFrames);
  const span = viewSpan(v);
  const pct = (uiFrame) => ((uiFrame - v.start) / span) * 100;
  const inView = (uiFrame) => uiFrame >= v.start - 1 && uiFrame <= v.end + 1;

  const frameFrom = (e) => {
    const rect = laneRef.current?.getBoundingClientRect();
    return rect && rect.width ? frameAtX(e.clientX - rect.left, rect.width, v, maxFrames) : frame;
  };

  const onLaneClick = (e) => { if (e.target === e.currentTarget) onSeek?.(frameFrom(e)); };
  const onLaneDouble = (e) => {
    if (e.target !== e.currentTarget || !canKey) return;
    onAdd?.(frameFrom(e));
  };

  const down = (kf) => (e) => {
    e.stopPropagation();
    e.currentTarget.setPointerCapture?.(e.pointerId);
    dragRef.current = { frameIndex: kf.frame_index, moved: false, startX: e.clientX };
    onSelect?.(kf.frame_index);
    onSeek?.(uiFrameOf(kf));
  };
  const drag = (e) => {
    const d = dragRef.current;
    if (!d || !(e.buttons & 1)) return;
    if (!d.moved && Math.abs(e.clientX - d.startX) < 3) return;       // a click is not a drag
    d.moved = true;
    setGhost({ frameIndex: d.frameIndex, uiFrame: frameFrom(e) });
  };
  const up = (e) => {
    const d = dragRef.current;
    dragRef.current = null;
    e.currentTarget.releasePointerCapture?.(e.pointerId);
    if (d?.moved) {
      const to = frameFrom(e);
      setGhost(null);
      if (to !== d.frameIndex + 1) onMove?.(d.frameIndex, to);
    }
  };

  const key = (kf) => (e) => {
    const step = e.shiftKey ? 10 : 1;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault();
      const to = Math.max(1, Math.min(maxFrames, uiFrameOf(kf) + (e.key === 'ArrowLeft' ? -step : step)));
      const moved = onMove?.(kf.frame_index, to);
      // A keyframe's identity IS its frame (it is the React key), so a move remounts its button and drops focus; put it
      // back on the moved one, or a second arrow press goes nowhere.
      if (moved !== false) {
        requestAnimationFrame(() => laneRef.current?.querySelector(`[data-testid="identity-keyframe-${to - 1}"]`)?.focus());
      }
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      onRemove?.(kf.frame_index);
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onSelect?.(kf.frame_index);
      onSeek?.(uiFrameOf(kf));
    }
  };

  const shown = keyframes.map((k) => (ghost && ghost.frameIndex === k.frame_index
    ? { ...k, ghostFrame: ghost.uiFrame } : k));

  return (
    <div
      ref={laneRef}
      data-testid="lane-identity-keyframes"
      role="group"
      aria-label={`Identity keyframes: ${keyframes.length}. Double-click the track to add one at that frame.`}
      className="relative w-full overflow-hidden rounded-md border border-[var(--border-color)] bg-[var(--input-bg)]"
      style={{ height: IDENTITY_LANE_H }}
    >
      {/* The pointer hit layer: click parks the playhead, double-click adds a keyframe. A mouse affordance only, so it
          is presentation; the keyboard routes are the "+ Key at playhead" button in the lane header and the
          keyframe buttons themselves (focus one: arrows nudge, Delete removes, Enter selects). */}
      <div role="presentation" data-testid="identity-keyframe-hit" onClick={onLaneClick} onDoubleClick={onLaneDouble}
           className="absolute inset-0" style={{ cursor: canKey ? 'copy' : 'default' }} />

      {/* Chapter markers, carried down so a keyframe can be placed on a shot boundary by eye. */}
      {markers.filter(inView).slice(0, 400).map((m) => (
        <span key={m} className="pointer-events-none absolute inset-y-0 w-px bg-amber-300/40"
              style={{ left: `${pct(m)}%` }} />
      ))}

      {/* Curve glyphs between neighbours: this segment's easing, drawn in its own box. */}
      {keyframes.slice(0, -1).map((k, i) => {
        const nx = keyframes[i + 1];
        const a = pct(uiFrameOf(k));
        const b = pct(uiFrameOf(nx));
        if (b < -5 || a > 105) return null;
        return (
          <svg key={`seg-${k.frame_index}`} className="pointer-events-none absolute inset-y-0"
               style={{ left: `${a}%`, width: `${Math.max(0, b - a)}%` }}
               width="100%" height={IDENTITY_LANE_H} viewBox={`0 0 100 ${IDENTITY_LANE_H}`} preserveAspectRatio="none"
               aria-hidden="true">
            <path d={curvePath(k.curve, k.bezier, 100, IDENTITY_LANE_H, 6)} fill="none"
                  stroke="var(--accent)" strokeOpacity="0.55" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
          </svg>
        );
      })}

      {!keyframes.length && (
        <span className="pointer-events-none absolute inset-0 flex items-center px-3 text-note text-[var(--text-muted)]">
          {emptyNote || (canKey ? 'Double-click to key the blended identity at a frame' : 'Add two or more sources to the Identity Blender to key them across the clip')}
        </span>
      )}

      {shown.map((k) => {
        const f = k.ghostFrame ?? uiFrameOf(k);
        if (!inView(f)) return null;
        const on = selected === k.frame_index;
        return (
          <button
            key={k.frame_index}
            type="button"
            data-testid={`identity-keyframe-${k.frame_index}`}
            aria-label={`Identity keyframe at frame ${uiFrameOf(k)}, ${k.curve.replace('_', ' ')} to the next`}
            aria-pressed={on}
            title={`Frame ${uiFrameOf(k)} - ${k.curve.replace('_', ' ')}. Drag to move, arrows nudge, Delete removes.`}
            onPointerDown={down(k)}
            onPointerMove={drag}
            onPointerUp={up}
            onPointerCancel={() => { dragRef.current = null; setGhost(null); }}
            onKeyDown={key(k)}
            onClick={(e) => e.stopPropagation()}
            onDoubleClick={(e) => e.stopPropagation()}
            className={`absolute top-1/2 touch-none border outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] ${
              on ? 'border-white bg-[var(--accent)]' : 'border-black/60 bg-white/85 hover:bg-white'} ${k.ghostFrame ? 'opacity-70' : ''}`}
            style={{ left: `${pct(f)}%`, width: DIAMOND, height: DIAMOND, transform: 'translate(-50%, -50%) rotate(45deg)',
              cursor: 'grab' }}
          />
        );
      })}

      {/* The playhead's frame reads as a tick when there is a keyframe under it (a quick "am I on one?") */}
      {keyframes.some((k) => uiFrameOf(k) === frame) && (
        <span className="pointer-events-none absolute inset-x-0 bottom-0 h-0.5 bg-[var(--accent)]/60" />
      )}
    </div>
  );
}

