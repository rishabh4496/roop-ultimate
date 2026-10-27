import React, { useEffect, useMemo, useRef } from 'react';
import { useLaneCanvas } from './useLaneCanvas';
import {
  LANE_H, MAX_PRESENCE_ROWS, drawCacheStrip, drawFilmstrip, drawOverrides, drawRuler, drawScenes, sceneLaneHeight,
} from './lanes';
import { overrideLaneHit, sceneLaneHit } from './hit';
import { fmtTimecode, frameAtX, normalizeView, placeView, viewSpan } from './viewMath';
import { scenesFromCuts } from './laneModel';
import { describeOverride } from './overrides';

// The lane components. Each is one <canvas> that repaints from plain data when the view or the data
// changes; none of them re-renders on a playback tick (the playhead is a DOM overlay owned by
// Timeline), so playing a clip does not repaint any lane.

const frameFromEvent = (e, el, view, maxFrames) => {
  const rect = el.getBoundingClientRect();
  return frameAtX(e.clientX - rect.left, rect.width, view, maxFrames);
};

export function RulerLane({ view, maxFrames, fps, onPointerDown, onPointerMove, onPointerLeave }) {
  const ref = useLaneCanvas(
    (ctx, g) => {
      const ticks = drawRuler(ctx, g, { view, maxFrames, fps });
      // What the ruler is currently reading in (timecode / seconds / frames), for the specs: a canvas
      // has no text to assert on, and "the labels switched to frame numbers" is the thing to prove.
      if (ref.current) ref.current.dataset.mode = ticks.mode;
    },
    [view.start, view.end, maxFrames, fps],
  );
  return (
    <canvas
      ref={ref}
      data-testid="lane-ruler"
      role="img"
      aria-label="Timecode ruler. Click or drag to move the playhead."
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerLeave={onPointerLeave}
      className="block w-full cursor-ew-resize"
      style={{ height: LANE_H.ruler }}
    />
  );
}

export function CacheLane({ view, maxFrames, spans }) {
  const ref = useLaneCanvas(
    (ctx, g) => { drawCacheStrip(ctx, g, { view, maxFrames, spans }); },
    [view.start, view.end, maxFrames, spans],
  );
  return (
    <canvas
      ref={ref}
      data-testid="lane-cache"
      role="img"
      aria-label="Render and smart-skip cache strip"
      className="block w-full rounded-full"
      style={{ height: LANE_H.cache }}
    />
  );
}

export function FilmCanvas({ view, maxFrames, fps, slots, tileFor, tileVersion, waveCoarse, waveFine, thumbsHidden }) {
  const ref = useLaneCanvas(
    (ctx, g) => { drawFilmstrip(ctx, g, { view, maxFrames, fps, slots, tileFor, waveCoarse, waveFine, thumbsHidden }); },
    [view.start, view.end, maxFrames, fps, slots, tileFor, tileVersion, waveCoarse, waveFine, thumbsHidden],
  );
  return (
    <canvas
      ref={ref}
      data-testid="lane-film"
      className="absolute inset-0 block h-full w-full pointer-events-none"
    />
  );
}

export function SceneLane({
  view, maxFrames, fps, cuts, presence, selectedPersonId, selectedInterval, emptyNote,
  onSeek, onJumpCut, onPresenceClick,
}) {
  const scenes = useMemo(() => scenesFromCuts(cuts, maxFrames), [cuts, maxFrames]);
  const h = sceneLaneHeight(presence.length);
  const ref = useLaneCanvas(
    (ctx, g) => { drawScenes(ctx, g, { view, maxFrames, cuts, presence, selectedPersonId, selectedInterval, emptyNote }); },
    [view.start, view.end, maxFrames, cuts, presence, selectedPersonId, selectedInterval, emptyNote],
  );

  const hitAt = (e) => {
    const el = ref.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    const v = normalizeView(view, maxFrames);
    const frame = frameFromEvent(e, el, v, maxFrames);
    const pxPerFrame = rect.width / viewSpan(v);
    return sceneLaneHit({
      frame, y: e.clientY - rect.top, cuts, rows: presence.slice(0, MAX_PRESENCE_ROWS), scenes,
      tolFrames: Math.max(0.5, 5 / Math.max(pxPerFrame, 1e-6)),
    });
  };

  // Native tooltip, written straight to the element so hovering never re-renders anything.
  const onMove = (e) => {
    const el = ref.current;
    if (!el) return;
    const hit = hitAt(e);
    let tip = '';
    if (hit?.kind === 'cut') tip = `Scene cut at frame ${hit.frame} (${fmtTimecode(hit.frame, fps)}) - click to jump`;
    else if (hit?.kind === 'presence') tip = `${hit.row.name}: frames ${hit.interval[0]}-${hit.interval[1]} - click to select this person and set In/Out to this scene`;
    else if (hit?.kind === 'scene') tip = `Scene ${hit.scene.index + 1}: frames ${hit.scene.start}-${hit.scene.end} - right-click to add an override`;
    if (el.title !== tip) el.title = tip;
    el.style.cursor = hit && hit.kind !== 'scene' ? 'pointer' : 'default';
  };

  const onClick = (e) => {
    const hit = hitAt(e);
    if (hit?.kind === 'cut') onJumpCut?.(hit.frame);
    else if (hit?.kind === 'presence') onPresenceClick?.(hit.row, hit.interval, hit.frame);
    else if (ref.current) onSeek?.(frameFromEvent(e, ref.current, normalizeView(view, maxFrames), maxFrames));
  };

  return (
    <canvas
      ref={ref}
      data-testid="lane-scenes"
      role="img"
      aria-label="Scene cuts and who is on screen"
      onPointerMove={onMove}
      onClick={onClick}
      className="block w-full"
      style={{ height: h }}
    />
  );
}

export function OverrideLane({
  view, maxFrames, overrides, selectedId, inout, emptyNote, onSelect, onUpdate, onSeek,
}) {
  const dragRef = useRef(null);
  const liveRef = useRef(null);
  liveRef.current = { view, maxFrames, overrides, onUpdate };
  const ref = useLaneCanvas(
    (ctx, g) => { drawOverrides(ctx, g, { view, maxFrames, overrides, selectedId, inout, emptyNote }); },
    [view.start, view.end, maxFrames, overrides, selectedId, inout?.start, inout?.end, emptyNote],
  );

  useEffect(() => {
    const onMove = (e) => {
      const d = dragRef.current;
      const L = liveRef.current;
      const el = ref.current;
      if (!d || !L || !el) return;
      const f = frameFromEvent(e, el, normalizeView(L.view, L.maxFrames), L.maxFrames);
      if (d.part === 'left') L.onUpdate(d.id, { start: Math.min(f, d.end) });
      else if (d.part === 'right') L.onUpdate(d.id, { end: Math.max(f, d.start) });
      else {
        const span = d.end - d.start;
        const placed = placeView(f - d.grab, span, L.maxFrames);
        L.onUpdate(d.id, { start: placed.start, end: placed.start + span });
      }
    };
    const onUp = () => { dragRef.current = null; };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
    };
  }, [ref]);

  const onPointerDown = (e) => {
    if (e.button !== 0) return;
    const el = ref.current;
    const rect = el.getBoundingClientRect();
    const v = normalizeView(view, maxFrames);
    const f = frameFromEvent(e, el, v, maxFrames);
    const edgeFrames = Math.max(0, Math.round(6 / Math.max(rect.width / viewSpan(v), 1e-6)));
    const hit = overrideLaneHit(overrides, f, edgeFrames);
    if (!hit) { onSelect(null); onSeek?.(f); return; }
    const { override: o, part } = hit;
    onSelect(o.id);
    dragRef.current = { id: o.id, part, start: o.start, end: o.end, grab: f - o.start };
  };

  const onMove = (e) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const v = normalizeView(view, maxFrames);
    const f = frameFromEvent(e, el, v, maxFrames);
    const edgeFrames = Math.max(0, Math.round(6 / Math.max(rect.width / viewSpan(v), 1e-6)));
    const hit = overrideLaneHit(overrides, f, edgeFrames);
    el.style.cursor = !hit ? 'default' : hit.part === 'body' ? 'grab' : 'ew-resize';
    const tip = hit ? `${hit.override.label}: frames ${hit.override.start}-${hit.override.end} - ${describeOverride(hit.override)}` : '';
    if (el.title !== tip) el.title = tip;
  };

  return (
    <canvas
      ref={ref}
      data-testid="lane-overrides"
      role="img"
      aria-label="Per-scene recipe overrides"
      onPointerDown={onPointerDown}
      onPointerMove={onMove}
      className="block w-full"
      style={{ height: LANE_H.over }}
    />
  );
}
