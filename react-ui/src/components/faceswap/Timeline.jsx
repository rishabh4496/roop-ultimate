import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import useThrottledFrameRequest from './useThrottledFrameRequest';
import Minimap from './timeline/Minimap';
import OverrideInspector from './timeline/OverrideInspector';
import IdentityKeyframeLane from './timeline/IdentityKeyframeLane';
import IdentityKeyframeInspector from './timeline/IdentityKeyframeInspector';
import CurveEditor from './curves/CurveEditor';
import { AudioLane, CacheLane, FilmCanvas, IdentityLane, OverrideLane, RulerLane, SceneLane } from './timeline/Lanes.jsx';
import useAudioTrackWave from './timeline/useAudioTrackWave';
import { clampOffset, offsetText, offsetMeaning, trackShiftSeconds } from './mouthCustody';
import { useApi } from '../../useApi';
import { allAmber, nextAmber } from './identityLock';
import { LANE_H, STATE_COLORS, STATE_LABELS } from './timeline/lanes.js';
import { useSceneOverrides } from './timeline/useSceneOverrides';
import { useFilmstrip, useTimelineTracks, useWaveform } from './timeline/useTimelineData';
import {
  adjacentCut, buildRenderStrip, mergeIntervals, renderedForPieces, scenesFromCuts, sceneAt,
} from './timeline/laneModel';
import { overrideCandidates } from './timeline/hit';
import { IDLE, reverseTick, shuttleKey, shuttleLabel, shuttleRate, stepSize } from './timeline/shuttle';
import {
  clamp, fitRange, fmtSeconds, fmtTimecode, followFrame, frameAtX, isZoomed, normalizeView, panView, placeView,
  spanAtFraction, thumbSlots, viewSpan, zoomFraction, zoomView,
} from './timeline/viewMath';

/**
 * The clip timeline - a multi-track NLE surface for the target video.
 *
 * Top to bottom: the whole-clip MINIMAP (always visible, with zoom handles), an adaptive RULER
 * (timecode -> seconds -> exact frame numbers as you zoom), a 4 px render / smart-skip CACHE strip,
 * the FILMSTRIP + audio waveform (the scrub surface), the SCENE CUTS + FACE PRESENCE lane, and the
 * per-segment OVERRIDE lane. Everything between the minimap and the overrides is drawn on canvases
 * from plain data, and every loop in the drawing is bounded by what is on screen, so a 50,000-frame
 * clip costs what a 100-frame one does.
 *
 * Scrub/drag geometry still lives in FaceSwap (it owns the pointer capture and the magnetic snap);
 * this component takes the ref and the three handlers and renders on top of them, so the
 * interaction contract is unchanged. The playhead is moved imperatively between React commits
 * (playheadApiRef) exactly as before.
 *
 * Gestures: Ctrl/Cmd + wheel zooms about the cursor; Shift + wheel (or a horizontal trackpad
 * swipe, or middle-button drag) scrolls; dragging the minimap window scrolls and its handles zoom.
 * Keys (when no field has focus): J / K / L shuttle, , and . step a frame (Shift: ten), I and O set
 * In and Out, Up / Down jump to the previous / next scene cut, M drops a chapter marker.
 */

const SPEEDS = [0.25, 0.5, 1, 2, 4];

/** m:ss under an hour, h:mm:ss above it. */
const fmtTC = (f, fps) => fmtSeconds(f + 1, fps);

const markersKey = (k) => `roop_markers_${k || 'default'}`;
const EMPTY = Object.freeze([]);

/** Chip-shaped inline number field - the header's In/Out are editable in place. */
function RangeField({ label, value, min, max, onCommit, accent = false }) {
  const [draft, setDraft] = useState(null);
  const commit = () => {
    if (draft !== null) {
      const v = clamp(parseInt(draft, 10) || value, min, max);
      if (v !== value) onCommit(v);
    }
    setDraft(null);
  };
  return (
    <label
      className={`inline-flex items-center gap-1.5 rounded-lg border px-2 py-1 transition-colors cursor-text ${
        accent
          ? 'border-[var(--accent)]/30 bg-[var(--accent)]/10'
          : 'border-[var(--border-color)] bg-[var(--surface-2)] hover:border-[var(--border-strong)]'
      }`}
      title={`${label} point - click to type a frame number`}
    >
      <span className="text-nano font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        value={draft ?? value}
        onFocus={(e) => { setDraft(String(value)); e.target.select(); }}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') e.currentTarget.blur();
          if (e.key === 'Escape') { setDraft(null); e.currentTarget.blur(); }
        }}
        className="w-[7ch] bg-transparent text-right text-mini font-mono font-semibold tabular-nums text-[var(--text-main)] outline-none"
      />
    </label>
  );
}

/** Icon button used across the toolbar clusters. */
const IconBtn = ({ title, onClick, active = false, children, className = '', disabled = false }) => (
  <button
    type="button"
    onClick={onClick}
    title={title}
    aria-label={title}
    disabled={disabled}
    className={`grid place-items-center h-8 w-8 rounded-lg transition-colors disabled:opacity-30 ${
      active
        ? 'bg-[var(--accent)]/12 text-accent'
        : 'text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.06]'
    } ${className}`}
  >
    {children}
  </button>
);

const Ico = ({ d, fill = false, size = 15 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24"
       fill={fill ? 'currentColor' : 'none'} stroke={fill ? 'none' : 'currentColor'}
       strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    {typeof d === 'string' ? <path d={d} /> : d}
  </svg>
);

const LaneHeader = ({ title, children }) => (
  <div className="flex items-center gap-2 pt-1.5 pb-0.5 min-h-[22px]">
    <span className="text-nano font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">{title}</span>
    <span className="flex-1" />
    {children}
  </div>
);

const smallBtn = 'rounded-md px-1.5 py-0.5 text-note font-semibold text-[var(--text-muted)] hover:bg-white/[0.06] hover:text-[var(--text-main)] disabled:opacity-40 transition-colors';

export default function Timeline({
  fps = 25,
  maxFrames = 1,
  frame = 1,
  setFrame,
  startFrame = 1,
  endFrame = 1,
  setFrameMarkerVal,
  timelineRef,
  onPointerDown,
  onPointerMove,
  onPointerLeave,
  hoverFrame = null,
  isScrubbing = false,
  isPlaying = false,
  setIsPlaying,
  buffering = false,
  isLooping = false,
  setIsLooping,
  playbackRate = 1,
  setPlaybackRate,
  thumbUrl,
  targetKey = '',
  targetIndex = null,
  view,
  setView,
  segments = [],
  onSegmentClick,
  sceneCuts = EMPTY,
  /** True while a render is in flight: the filmstrip stops fetching and the track maps refresh. */
  running = false,
  /** The user's match distance (Max face distance): what "near the threshold" is measured against. */
  matchThreshold = null,
  /** [{start, end, fp}] finished renders of this clip, and the fingerprints in force NOW, per stretch. */
  renderLog = EMPTY,
  fpPieces = EMPTY,
  /** {start, end, done} while a render is in flight, so the strip fills in as frames land. */
  liveRange = null,
  /** useSceneOverrides(): {overrides, selectedId, setSelectedId, add, update, remove}. */
  sceneOverrides,
  swapModels = EMPTY,
  detectionMode = '',
  baseValues,
  onPresenceSelect,
  onRenderRange,
  onRenderOverrides,
  renderBlockedReason = '',
  /**
   * A ref the parent calls `.current(frame)` on to move the playhead WITHOUT a React commit.
   * During a playhead drag the parent commits `frame` a few times a second (the preview can
   * only follow that fast anyway) while the pointer produces one position per display frame;
   * this is how the playhead and its timecode badge keep up with the pointer between commits.
   */
  playheadApiRef = null,
  /**
   * Stage 17: the delivered audio as its own row, with the slip controls.
   * {offsetMs, onOffsetChange(ms), trackPath, trackName, hasTargetAudio}. Absent = no audio row (the Batch page, tests).
   */
  audioTrack = null,
  /**
   * Stage 20: the Identity Keyframe Track. useIdentityKeyframes() plus `recipe`, `components` and `labelOf`
   * (FaceSwap composes them). Absent = no track (the Batch page, tests).
   */
  identityKeys = null,
  /**
   * Stage 25: the Curve Editor. {open, onToggle, api (useParameterCurves), tracks, enabled, settings, morph, planNote}.
   * `open` lives in FaceSwap because the C key that toggles it is owned there (one owner per bare key). Absent = no
   * editor (the Batch page, tests).
   */
  curveEditor = null,
  /**
   * A ref the parent reads `.current.sceneAt(frame)` from: the scene under a frame as this timeline draws it (its cuts
   * are the merge of the scan's and the backend's), so an action elsewhere on the page - the face menu's "bypass on
   * this scene" - cuts at the same shots the lanes show. Absent = nothing is published.
   */
  sceneApiRef = null,
}) {
  // ── View window ───────────────────────────────────────────────────────────
  // The lanes show [view.start, view.end] rather than the whole clip. Fall back to the whole clip
  // for a missing or degenerate window - the parent seeds it in an effect, so the first paint after
  // a target change would otherwise map every position against a one-frame span.
  const v = useMemo(() => normalizeView(view, maxFrames), [view, maxFrames]);
  const vStart = v.start;
  const vEnd = v.end;
  const vSpan = viewSpan(v);
  const zoomed = isZoomed(v, maxFrames);
  const pctOf = (f) => ((f - vStart) / vSpan) * 100;
  const inView = (f) => f >= vStart && f <= vEnd;

  // The frame the playhead is DRAWN at. Between drag commits the parent's `frame` lags the pointer,
  // so while scrubbing a re-render must not snap the playhead back to it.
  const liveFrameRef = useRef(frame);
  if (!isScrubbing) liveFrameRef.current = frame;
  const shownFrame = isScrubbing ? liveFrameRef.current : frame;

  const startPct = pctOf(startFrame);
  const endPct = pctOf(endFrame);
  const currentPct = pctOf(shownFrame);
  const rangeLen = Math.max(0, endFrame - startFrame + 1);
  const rangeShare = Math.round((rangeLen / maxFrames) * 100);

  // ── Geometry: the lanes share one x-extent, measured from the film track ───
  const [trackW, setTrackW] = useState(0);
  useEffect(() => {
    const el = timelineRef?.current;
    if (!el) return undefined;
    setTrackW(el.getBoundingClientRect().width);
    if (typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(([e]) => setTrackW(e.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, [timelineRef, maxFrames]);

  const setViewSafe = useCallback((next) => { if (setView) setView(next); }, [setView]);

  /** Re-window around a pivot frame. factor > 1 zooms out, < 1 zooms in. */
  const zoomAround = useCallback((pivot, factor, ratio) => {
    if (!setView || maxFrames < 2) return;
    setView(zoomView(v, maxFrames, trackW, factor, pivot, ratio));
  }, [setView, maxFrames, trackW, v]);
  const panBy = useCallback((deltaFrames) => { if (setView) setView(panView(v, maxFrames, deltaFrames)); }, [setView, v, maxFrames]);
  const resetView = () => setViewSafe({ start: 1, end: maxFrames });

  // Wheel over the lanes. Must be a NATIVE non-passive listener: React's onWheel is registered
  // passive, so preventDefault there is ignored and the page scrolls behind the zoom. The handler
  // reads the current window from a ref and the listener is bound once, so a fast spin never hands
  // several events to a listener still holding the previous window.
  const stackRef = useRef(null);
  const wheelRef = useRef(null);
  wheelRef.current = { v, vSpan, zoomAround, panBy, maxFrames, frame };
  useEffect(() => {
    const el = stackRef.current;
    if (!el) return undefined;
    const onWheel = (e) => {
      const W = wheelRef.current;
      if (!W || W.maxFrames < 2) return;
      // Ctrl/Cmd + wheel is the app zoom (App.jsx, on window), which this
      // listener would otherwise fire alongside — zooming the track and the
      // whole UI on one gesture.
      if (e.ctrlKey || e.metaKey) return;
      const rect = el.getBoundingClientRect();
      if (!rect.width) return;
      e.preventDefault();
      const ratio = clamp((e.clientX - rect.left) / rect.width, 0, 1);
      const pivot = W.v.start + ratio * W.vSpan;
      // A horizontal swipe (trackpad) or Shift + wheel scrolls.
      if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        const d = e.shiftKey && Math.abs(e.deltaX) < Math.abs(e.deltaY) ? e.deltaY : e.deltaX;
        W.panBy(Math.round((d / rect.width) * W.vSpan));
        return;
      }
      // Smooth for both a notched mouse wheel (|dy| ~ 100) and a trackpad pinch (|dy| ~ 1-10).
      const factor = Math.exp(clamp(e.deltaY, -240, 240) * 0.002);
      W.zoomAround(pivot, factor, ratio);
    };
    // Ctrl/Cmd + wheel is the timeline's zoom gesture, claimed by its OWN listener rather than by a
    // branch of the one above (which stays a pure yield to the app zoom, as test_ui_zoom_pan pins).
    // The element listener runs before App.jsx's window listener, and stopPropagation is what keeps
    // that from also firing: one gesture, one zoom - this track's, never the whole UI's.
    const onZoomWheel = (e) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      const W = wheelRef.current;
      const rect = el.getBoundingClientRect();
      if (!W || W.maxFrames < 2 || !rect.width) return;
      e.preventDefault();
      e.stopPropagation();
      const ratio = clamp((e.clientX - rect.left) / rect.width, 0, 1);
      // Anchored at the cursor; a trackpad pinch (|dy| ~ 1-10) and a notched wheel (~100) both land smoothly.
      W.zoomAround(W.v.start + ratio * W.vSpan, Math.exp(clamp(e.deltaY, -240, 240) * 0.002), ratio);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    el.addEventListener('wheel', onZoomWheel, { passive: false });
    return () => {
      el.removeEventListener('wheel', onWheel);
      el.removeEventListener('wheel', onZoomWheel);
    };
  }, []);

  // Middle-button drag scrolls. Window-level move/up so the drag survives leaving the lanes.
  const panDragRef = useRef(null);
  useEffect(() => {
    const el = stackRef.current;
    if (!el) return undefined;
    const onDown = (e) => {
      if (e.button !== 1) return;
      e.preventDefault();                       // no browser auto-scroll cursor
      panDragRef.current = { x: e.clientX, start: wheelRef.current.v.start, w: el.getBoundingClientRect().width };
    };
    const onMove = (e) => {
      const d = panDragRef.current;
      if (!d) return;
      const W = wheelRef.current;
      const delta = -((e.clientX - d.x) / Math.max(1, d.w)) * W.vSpan;
      setView?.(placeView(d.start + delta, W.vSpan, W.maxFrames));
    };
    const onUp = () => { panDragRef.current = null; };
    el.addEventListener('pointerdown', onDown);
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    return () => {
      el.removeEventListener('pointerdown', onDown);
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
  }, [setView]);

  // Keep the playhead on screen while playing - a zoomed-in track that the playhead has already run
  // off is just a still picture.
  useEffect(() => {
    if (!isPlaying || !zoomed || !setView) return;
    const next = followFrame(v, maxFrames, frame);
    // followFrame hands back an equal window when the playhead is still on screen; committing that
    // would be a state change (and a re-render of the whole panel) on every playback tick.
    if (next.start !== v.start || next.end !== v.end) setView(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame, isPlaying, zoomed]);

  // ── Data: tracks, waveform, stills ─────────────────────────────────────────
  const { tracks, scanPresence, scanning } = useTimelineTracks({
    targetIndex, targetKey, maxFrames, poll: running, threshold: matchThreshold,
  });
  // Every amber frame across people, for the prev / next buttons beside the strip.
  const amberFrames = useMemo(() => allAmber(tracks.identity), [tracks.identity]);
  const cuts = useMemo(() => {
    const merged = new Set([...(tracks.sceneCuts || []), ...(sceneCuts || [])].filter(Number.isFinite));
    return [...merged].sort((a, b) => a - b);
  }, [tracks.sceneCuts, sceneCuts]);
  const scenes = useMemo(() => scenesFromCuts(cuts, maxFrames), [cuts, maxFrames]);

  const wave = useWaveform({ targetIndex, targetKey, fps, maxFrames, view: v, trackW, enabled: maxFrames > 1 });
  // Stage 17: the audio row shows what will PLAY - the bound external track if there is one, else the target's own audio.
  const { getJSON: getAudioJSON } = useApi();
  const trackPath = audioTrack?.trackPath || null;
  const trackWave = useAudioTrackWave(getAudioJSON, trackPath);
  const audioWaves = useMemo(
    () => (trackPath ? [trackWave.coarse] : [wave.fine, wave.coarse]),
    [trackPath, trackWave.coarse, wave.fine, wave.coarse],
  );
  const audioOffsetMs = clampOffset(audioTrack?.offsetMs);
  const showAudioRow = !!audioTrack && maxFrames > 1 && (!!trackPath || audioTrack.hasTargetAudio !== false);
  const nudgeAudio = (d) => audioTrack?.onOffsetChange?.(clampOffset(audioOffsetMs + d));

  const slotH = LANE_H.film;
  const slots = useMemo(
    () => thumbSlots(v, maxFrames, trackW, Math.round((slotH * 16) / 9)),
    [v, maxFrames, trackW, slotH],
  );
  const filmstrip = useFilmstrip({ targetIndex, targetKey, slots, enabled: !running });

  // ── The cache strip ───────────────────────────────────────────────────────
  const spans = useMemo(() => {
    const rendered = renderedForPieces(renderLog, fpPieces);
    if (liveRange && liveRange.done > 0) rendered.push([liveRange.start, liveRange.start + liveRange.done - 1]);
    const mapsApply = tracks.renderVideoMatches;
    return buildRenderStrip({
      lo: 1,
      hi: Math.max(1, maxFrames),
      rendered: mergeIntervals(rendered, 1, maxFrames),
      bypass: mapsApply ? tracks.faceFree : EMPTY,
      reuse: mapsApply ? tracks.reuse : EMPTY,
    });
  }, [renderLog, fpPieces, liveRange, tracks.faceFree, tracks.reuse, tracks.renderVideoMatches, maxFrames]);

  // ── Chapter markers, per clip ─────────────────────────────────────────────
  const [markers, setMarkers] = useState([]);
  useEffect(() => {
    try {
      const raw = localStorage.getItem(markersKey(targetKey));
      const list = raw ? JSON.parse(raw) : null;
      setMarkers(Array.isArray(list) ? list.filter((n) => Number.isFinite(n)) : []);
    } catch { setMarkers([]); }
  }, [targetKey]);

  const persistMarkers = (list) => {
    const next = [...new Set(list)].filter((f) => f >= 1 && f <= maxFrames).sort((a, b) => a - b);
    setMarkers(next);
    try {
      if (next.length) localStorage.setItem(markersKey(targetKey), JSON.stringify(next));
      else localStorage.removeItem(markersKey(targetKey));
    } catch { /* private mode / quota - markers are a convenience, not state we own */ }
  };

  const markerAtPlayhead = markers.includes(frame);
  const toggleMarker = () =>
    persistMarkers(markerAtPlayhead ? markers.filter((m) => m !== frame) : [...markers, frame]);

  const jumpMarker = (dir) => {
    const next = dir > 0 ? markers.find((m) => m > frame) : [...markers].reverse().find((m) => m < frame);
    if (next !== undefined) setFrame(next);
  };
  // ── Identity keyframes: the selected one and its successor, for the inspector under the track ──
  const idKf = identityKeys?.keyframes || EMPTY;
  const idSel = identityKeys ? idKf.findIndex((k) => k.frame_index === identityKeys.selected) : -1;
  const jumpCut = (dir) => {
    const next = adjacentCut(cuts, frame, dir);
    if (next !== null) setFrame(next);
    return next !== null;
  };

  // ── Frame read-out field ──────────────────────────────────────────────────
  const [frameDraft, setFrameDraft] = useState(null);

  // Hover thumbnail, loaded one at a time and aborted when superseded (see git history for why:
  // it competes with the playhead's own frame for the same decoder).
  const { frameSrc: hoverSrc } = useThrottledFrameRequest(
    hoverFrame !== null && thumbUrl ? thumbUrl(hoverFrame) : '',
    { throttleMs: 150 });

  // Labels at the very edges would hang off the track; anchor those to the edge instead of centring.
  const anchor = (pct) => (pct < 3 ? 'translateX(0)' : pct > 97 ? 'translateX(-100%)' : 'translateX(-50%)');
  // Where a frame's marker sits along the track: a percentage of the (full-width) wrapper it is applied to.
  const slide = (pct) => `translate3d(${pct}%, 0, 0)`;

  // Imperative playhead: writes the same things the JSX below renders straight to the DOM. Re-registered on
  // every render so it closes over the CURRENT view window and fps.
  //
  // EVERYTHING THAT FOLLOWS THE FRAME MOVES BY TRANSFORM, NOT `left`. Each of these is a full-width, zero-content
  // wrapper (`left-0 w-full`) whose child sits at its left edge, so `translate3d(P%, 0, 0)` -- a percentage of the
  // wrapper's own width, i.e. of the track -- puts the child at P% of the track exactly where `left: P%` did.
  // The difference is the pipeline: `left` re-runs layout for the lane stack on every playback tick; a transform on
  // a `will-change: transform` layer is composited and never touches layout. The percentage maths is unchanged, so
  // it is still independent of the app's CSS zoom (no pixel measurement anywhere).
  const playheadElRef = useRef(null);
  const stackLineRef = useRef(null);
  const badgeElRef = useRef(null);
  const badgeInnerRef = useRef(null);
  const badgeTextRef = useRef(null);
  useLayoutEffect(() => {
    if (!playheadApiRef) return undefined;
    playheadApiRef.current = (f) => {
      liveFrameRef.current = f;
      const pct = pctOf(f);
      if (playheadElRef.current) playheadElRef.current.style.transform = slide(pct);
      if (stackLineRef.current) {
        stackLineRef.current.style.transform = slide(pct);
        stackLineRef.current.style.display = inView(f) ? '' : 'none';
      }
      const badge = badgeElRef.current;
      if (badge) {
        badge.style.transform = slide(pct);
        badge.style.display = inView(f) ? '' : 'none';
      }
      if (badgeInnerRef.current) badgeInnerRef.current.style.transform = anchor(pct);
      if (badgeTextRef.current) badgeTextRef.current.textContent = fmtTimecode(f, fps);
    };
    return () => { if (playheadApiRef) playheadApiRef.current = null; };
  });

  const step = (d) => setFrame((f) => clamp(f + d, 1, maxFrames));

  // ── Scene overrides ───────────────────────────────────────────────────────
  const fallbackOv = useSceneOverrides(targetKey, maxFrames);
  const ov = sceneOverrides || fallbackOv;
  const selectedOverride = ov.overrides.find((o) => o.id === ov.selectedId) || null;
  const inout = useMemo(() => ({ start: startFrame, end: endFrame }), [startFrame, endFrame]);

  const [menu, setMenu] = useState(null);       // {x, y, frame, items}
  useEffect(() => {
    if (!menu) return undefined;
    const close = () => setMenu(null);
    const onKey = (e) => { if (e.code === 'Escape') close(); };
    window.addEventListener('pointerdown', close);
    window.addEventListener('keydown', onKey);
    window.addEventListener('blur', close);
    return () => {
      window.removeEventListener('pointerdown', close);
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('blur', close);
    };
  }, [menu]);

  const onContextMenu = (e) => {
    if (maxFrames < 2) return;
    const el = stackRef.current;
    const rect = el.getBoundingClientRect();
    e.preventDefault();
    const f = frameAtX(e.clientX - rect.left, rect.width, v, maxFrames);
    const items = overrideCandidates({ frame: f, scenes, startFrame, endFrame, maxFrames })
      .map((c) => ({ id: `add-${c.kind}`, label: `+ Add Scene Override - ${c.label} (frames ${c.start}-${c.end})`, run: () => ov.add({ start: c.start, end: c.end }, c.label) }));
    const hit = ov.overrides.filter((o) => f >= o.start && f <= o.end).pop();
    if (hit) {
      items.push({ id: 'edit', label: `Edit "${hit.label}"`, run: () => ov.setSelectedId(hit.id) });
      items.push({ id: 'remove', label: `Remove "${hit.label}"`, run: () => ov.remove(hit.id) });
    }
    items.push({ id: 'cut', label: `Set In to ${f}`, run: () => setFrameMarkerVal('start', f) });
    items.push({ id: 'cut-out', label: `Set Out to ${f}`, run: () => setFrameMarkerVal('end', f) });
    setMenu({
      x: clamp(((e.clientX - rect.left) / rect.width) * 100, 0, 100),
      y: clamp(((e.clientY - rect.top) / rect.height) * 100, 0, 100),
      frame: f,
      items,
    });
  };

  // ── Presence selection ────────────────────────────────────────────────────
  const [pick, setPick] = useState(null);     // {id, interval}
  useEffect(() => { setPick(null); }, [targetKey]);
  const onPresenceClick = (row, interval, at) => {
    setPick({ id: row.id, interval });
    // "That scene": the shot under the click, limited to the stretch this person is in. A person who
    // stays on screen across several shots has ONE long pill, and In/Out to the whole of it would be
    // the clip, not a scene; the cuts are what say where this scene starts and ends.
    const shot = Number.isFinite(at) ? sceneAt(scenes, at) : null;
    const from = shot ? Math.max(interval[0], shot.start) : interval[0];
    const to = shot ? Math.min(interval[1], shot.end) : interval[1];
    setFrameMarkerVal('start', from);
    setFrameMarkerVal('end', to);
    setFrame(from);
    onPresenceSelect?.(row, [from, to]);
  };

  // ── Shuttle (J / K / L) ───────────────────────────────────────────────────
  const [shuttle, setShuttle] = useState(IDLE);
  const shuttleRef = useRef(shuttle);
  shuttleRef.current = shuttle;
  const latest = useRef({});
  latest.current = {
    frame, fps, isLooping, startFrame, endFrame, setFrame, playheadApiRef, markers, toggleMarker,
    maxFrames, jumpCut, step, setFrameMarkerVal, setIsPlaying, setPlaybackRate,
  };

  // The buffered player owns forward playback; anything else that stops or starts it (Space, the
  // end of the clip, the Play button) ends the shuttle so the read-out never lies.
  useEffect(() => {
    if (!isPlaying && shuttle.dir > 0) setShuttle(IDLE);
    if (isPlaying && shuttle.dir < 0) setShuttle(IDLE);
  }, [isPlaying, shuttle.dir]);

  // Reverse has no buffer to ride, so the playhead is stepped on a clock. The playhead moves every
  // animation frame through the imperative API; the frame is committed at most every 100 ms (the
  // same pace as a scrub drag), which is all the preview can follow.
  useEffect(() => {
    if (shuttle.dir >= 0) return undefined;
    let raf = 0;
    let last = performance.now();
    let cur = latest.current.frame;
    let carry = 0;
    let lastCommit = 0;
    const rate = shuttleRate(shuttle);
    const tick = (now) => {
      const L = latest.current;
      const floor = L.isLooping ? L.startFrame : 1;
      const r = reverseTick({ frame: cur, carry }, now - last, rate, L.fps, floor);
      last = now;
      carry = r.carry;
      cur = r.frame;
      L.playheadApiRef?.current?.(cur);
      if (now - lastCommit >= 100 || r.atStart) { lastCommit = now; L.setFrame(cur); }
      if (r.atStart) {
        if (L.isLooping && L.endFrame > cur) { cur = L.endFrame; carry = 0; }
        else { setShuttle(IDLE); return; }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [shuttle]);

  // ── Keyboard ──────────────────────────────────────────────────────────────
  // Same guards as the app's global shortcut handler so it never fires while typing or behind a
  // modal. Reached through a ref so the listener is bound once instead of re-subscribing on every
  // frame the playback loop paints. Each key is tested in the `e.key... === 'x'` form on purpose:
  // test_ui_shortcut_keys.py reads those to prove no two components claim the same bare key.
  useEffect(() => {
    const onKey = (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
      if (document.querySelector('[role="dialog"]')) return;
      const L = latest.current;
      if (L.maxFrames < 2) return;
      const shuttleTo = (next) => {
        setShuttle(next);
        if (next.dir > 0) { L.setPlaybackRate?.(shuttleRate(next)); L.setIsPlaying?.(true); }
        else L.setIsPlaying?.(false);
      };
      if (e.key.toLowerCase() === 'm') { e.preventDefault(); L.toggleMarker(); }
      else if (e.key.toLowerCase() === 'j') { e.preventDefault(); shuttleTo(shuttleKey(shuttleRef.current, 'j')); }
      else if (e.key.toLowerCase() === 'k') { e.preventDefault(); shuttleTo(shuttleKey(shuttleRef.current, 'k')); }
      else if (e.key.toLowerCase() === 'l') { e.preventDefault(); shuttleTo(shuttleKey(shuttleRef.current, 'l')); }
      else if (e.key.toLowerCase() === 'i') { e.preventDefault(); L.setFrameMarkerVal('start', L.frame); }
      else if (e.key.toLowerCase() === 'o') { e.preventDefault(); L.setFrameMarkerVal('end', L.frame); }
      else if (e.key === ',' || e.key === '<') { e.preventDefault(); L.step(-stepSize(e.shiftKey || e.key === '<')); }
      else if (e.key === '.' || e.key === '>') { e.preventDefault(); L.step(stepSize(e.shiftKey || e.key === '>')); }
      else if (e.key === 'ArrowUp') { if (L.jumpCut(-1)) e.preventDefault(); }
      else if (e.key === 'ArrowDown') { if (L.jumpCut(1)) e.preventDefault(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // ── Zoom slider ───────────────────────────────────────────────────────────
  const zoomT = zoomFraction(v, maxFrames, trackW);
  const onZoomSlider = (e) => {
    const t = Number(e.target.value);
    const span = spanAtFraction(t, maxFrames, trackW);
    setViewSafe(zoomView(v, maxFrames, trackW, span / vSpan, clamp(frame, v.start, v.end)));
  };

  const scene = sceneAt(scenes, frame);
  const hasCuts = cuts.length > 0;
  useLayoutEffect(() => {
    if (!sceneApiRef) return undefined;
    sceneApiRef.current = {
      hasCuts,
      sceneAt: (f) => {
        const s = sceneAt(scenes, f);
        return s ? { start: s.start, end: s.end, label: `Scene ${s.index + 1}`, index: s.index } : null;
      },
    };
    return () => { sceneApiRef.current = null; };
  }, [sceneApiRef, scenes, hasCuts]);
  const presenceNote = tracks.presenceSource === 'none' && !tracks.presence.length
    ? 'Who is on screen: press "Scan faces" to fill this lane'
    : '';
  const overrideNote = 'Right-click a scene or the In/Out range and choose + Add Scene Override';
  const rangeIsWhole = startFrame <= 1 && endFrame >= maxFrames;

  return (
    <div className="contain-panel space-y-2 select-none" data-testid="timeline"
         data-view-start={vStart} data-view-end={vEnd} data-zoomed={zoomed ? 'true' : 'false'}>
      {/* ── Header: identity + the single, editable statement of the range ── */}
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2">
        <div className="flex items-baseline gap-2.5 min-w-0">
          <span className="text-mini font-semibold uppercase tracking-[0.18em] text-[var(--text-muted)]">Timeline</span>
          <span className="font-mono text-note tabular-nums text-muted truncate">
            {maxFrames.toLocaleString()} frames · {fps} fps · {fmtTC(maxFrames, fps)}
          </span>
          {zoomed && (
            <span className="rounded-md border border-[var(--border-color)] bg-[var(--surface-2)] px-1.5 py-0.5 font-mono text-note tabular-nums text-[var(--text-muted)]"
                  title="Visible range - the track is zoomed in">
              showing {fmtTC(vStart, fps)}-{fmtTC(vEnd, fps)} · {Math.round((maxFrames - 1) / vSpan)}×
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          {curveEditor && maxFrames > 1 && (
            <button type="button" onClick={curveEditor.onToggle} aria-pressed={curveEditor.open} data-testid="curve-editor-toggle"
                    title="Show the parameter curve editor: keyframe a setting across the clip (C)"
                    className={`inline-flex items-center gap-1.5 rounded-lg border px-2 py-1 text-mini font-semibold transition-colors ${
                      curveEditor.open
                        ? 'border-[var(--accent)]/30 bg-[var(--accent)]/10 text-accent'
                        : 'border-[var(--border-color)] bg-[var(--surface-2)] text-[var(--text-muted)] hover:border-[var(--border-strong)] hover:text-[var(--text-main)]'}`}>
              <Ico size={13} d="M3 18c4 0 5-12 9-12s5 12 9 12" />
              Curve Editor (C)
              {curveEditor.api?.any && <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" aria-label="has keyframes" />}
            </button>
          )}
          <RangeField label="In" value={startFrame} min={1} max={endFrame}
                      onCommit={(n) => setFrameMarkerVal('start', n)} />
          <RangeField label="Out" value={endFrame} min={startFrame} max={maxFrames}
                      onCommit={(n) => setFrameMarkerVal('end', n)} />
          <span className="inline-flex items-baseline gap-1.5 rounded-lg border border-[var(--accent)]/25 bg-[var(--accent)]/[0.08] px-2 py-1 font-mono text-mini tabular-nums text-accent"
                title="Length of the selected range - this is what gets rendered">
            {fmtTC(rangeLen, fps)}
            <span className="text-note">{rangeLen.toLocaleString()} f · {rangeShare}%</span>
          </span>
        </div>
      </div>

      {/* ── Minimap: the whole clip, with the visible window and its zoom handles ── */}
      <Minimap
        maxFrames={maxFrames}
        view={v}
        setView={setViewSafe}
        frame={frame}
        startFrame={startFrame}
        endFrame={endFrame}
        spans={spans}
        cuts={cuts}
        presence={tracks.presence}
        trackW={trackW}
      />

      {/* ── Lane stack. One x-extent for every lane, so one playhead crosses them all. ── */}
      <div ref={stackRef} className="relative" onContextMenu={onContextMenu} data-testid="timeline-stack">
        <RulerLane
          view={v} maxFrames={maxFrames} fps={fps}
          onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerLeave={onPointerLeave}
        />
        <CacheLane view={v} maxFrames={maxFrames} spans={spans} />

        {/* Filmstrip + waveform: the scrub surface */}
        <div className="relative mt-1">
          {/* Hover scrub thumbnail */}
          {hoverFrame !== null && thumbUrl && hoverSrc && (
            <div
              className="absolute bottom-[90px] z-50 flex flex-col items-center pointer-events-none -translate-x-1/2"
              style={{ left: `${clamp(pctOf(hoverFrame), 9, 91)}%` }}
            >
              <div className="rounded-xl border border-[var(--border-strong)] bg-[var(--card-bg)] p-1.5 backdrop-blur-xl shadow-[0_10px_30px_rgba(0,0,0,0.55)]">
                <img
                  src={hoverSrc}
                  alt=""
                  className="w-44 h-[99px] object-cover rounded-lg bg-black/60"
                  onError={(e) => { e.currentTarget.style.visibility = 'hidden'; }}
                  onLoad={(e) => { e.currentTarget.style.visibility = 'visible'; }}
                />
                <div className="mt-1 flex items-baseline justify-between px-0.5 font-mono text-note tabular-nums">
                  <span className="text-[var(--text-main)] font-semibold">{fmtTimecode(hoverFrame, fps)}</span>
                  <span className="text-muted">f {hoverFrame.toLocaleString()}</span>
                </div>
              </div>
              <span className="w-2.5 h-2.5 rotate-45 -mt-[6px] border-b border-r border-[var(--border-strong)] bg-[var(--card-bg)]" />
            </div>
          )}

          <div
            ref={timelineRef}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerLeave={onPointerLeave}
            className="relative w-full rounded-lg bg-[var(--input-bg)] border border-[var(--border-color)] overflow-hidden cursor-ew-resize timeline-track"
            style={{ height: LANE_H.film }}
          >
            <FilmCanvas
              view={v} maxFrames={maxFrames} fps={fps} slots={slots}
              tileFor={filmstrip.tileFor} tileVersion={filmstrip.version}
              waveCoarse={wave.coarse} waveFine={wave.fine}
              thumbsHidden={false}
            />

            {/* Out-of-range: dimmed AND desaturated, the way an NLE greys the material you have
                trimmed away. Percentages are clamped because once the view is zoomed the In/Out
                points can sit outside it: a negative width is invalid CSS. */}
            <div className="absolute inset-y-0 left-0 z-10 pointer-events-none bg-black/55 backdrop-saturate-0"
                 style={{ width: `${clamp(startPct, 0, 100)}%` }} />
            <div className="absolute inset-y-0 right-0 z-10 pointer-events-none bg-black/55 backdrop-saturate-0"
                 style={{ left: `${clamp(endPct, 0, 100)}%` }} />

            {/* Active range rails */}
            <div className="absolute z-10 pointer-events-none border-y-2 border-[var(--accent)]/70 inset-y-0"
                 style={{ left: `${clamp(startPct, 0, 100)}%`,
                          width: `${Math.max(0, clamp(endPct, 0, 100) - clamp(startPct, 0, 100))}%` }} />

            {/* In / Out handles - only when the point is actually on screen */}
            {[['start', startFrame, startPct], ['end', endFrame, endPct]]
              .filter(([, f]) => inView(f))
              .map(([which, , pct]) => (
              <div key={which} className="absolute inset-y-0 z-20 w-[3px] -translate-x-1/2 bg-white/90 pointer-events-none"
                   style={{ left: `${pct}%` }}>
                <span className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-9 w-2.5 rounded-full bg-white shadow-[0_1px_4px_rgba(0,0,0,0.7)] grid place-items-center gap-[2px]">
                  <span className="block h-[7px] w-px bg-black/25" />
                </span>
                <span className={`absolute top-1 rounded px-1 bg-black/70 text-nano font-bold uppercase tracking-wider text-white ${
                  which === 'start' ? 'left-1.5' : 'right-1.5'}`}>
                  {which === 'start' ? 'In' : 'Out'}
                </span>
              </div>
            ))}

            {/* Chapter markers carried down onto the track (bounded: only those on screen) */}
            {markers.filter(inView).slice(0, 400).map((m) => (
              <span key={m} className="absolute inset-y-0 w-px bg-amber-300/45 z-10 pointer-events-none"
                    style={{ left: `${pctOf(m)}%` }} />
            ))}

            {/* Segment bands: a thin strip along the bottom so several stay readable. */}
            {segments.map((s, i) => {
              const from = Math.max(s.start, vStart);
              const to = Math.min(s.end, vEnd);
              if (to < from) return null;
              return (
                <button
                  key={s.id}
                  type="button"
                  onClick={(e) => { e.stopPropagation(); onSegmentClick?.(s); }}
                  onPointerDown={(e) => e.stopPropagation()}
                  className="absolute bottom-0 h-1.5 z-20 rounded-sm bg-[var(--accent)]/75 hover:bg-[var(--accent)] border-x border-black/40 transition-colors"
                  style={{ left: `${pctOf(from)}%`, width: `${Math.max(0.4, pctOf(to) - pctOf(from))}%` }}
                  title={`Segment ${i + 1}: frames ${s.start}-${s.end}`}
                  aria-label={`Segment ${i + 1}, frames ${s.start} to ${s.end}`}
                />
              );
            })}

            {/* Hover line */}
            {hoverFrame !== null && (
              <div className="absolute inset-y-0 left-0 w-full z-20 pointer-events-none will-change-transform"
                   style={{ transform: slide(pctOf(hoverFrame)) }}>
                <div className="absolute inset-y-0 left-0 w-px -translate-x-1/2 bg-white/35" />
              </div>
            )}

            {/* Playhead */}
            <div ref={playheadElRef}
                 data-testid="timeline-playhead"
                 className={`absolute inset-y-0 left-0 w-full z-30 pointer-events-none will-change-transform ${isScrubbing ? '' : 'transition-transform duration-100 ease-out'}`}
                 style={{ transform: slide(currentPct) }}>
              <div className="absolute inset-y-0 left-0 -translate-x-1/2 w-[2px] bg-white shadow-[0_0_0_1px_rgba(0,0,0,0.45)]" />
              <div className="absolute -top-px left-0 -translate-x-1/2 h-0 w-0 border-x-[6px] border-x-transparent border-t-[8px] border-t-[var(--accent)]" />
              <div className="absolute bottom-1 left-0 -translate-x-1/2 h-2 w-2 rounded-full bg-white shadow-[0_1px_3px_rgba(0,0,0,0.6)]" />
            </div>
          </div>

          {/* Playhead timecode, riding under the film track. Always mounted (hidden with display:none
              when out of view) so the imperative API has an element to write to. */}
          <div ref={badgeElRef}
               className={`absolute inset-x-0 -bottom-2 z-30 pointer-events-none will-change-transform ${isScrubbing ? '' : 'transition-transform duration-100 ease-out'}`}
               style={{ transform: slide(currentPct), display: inView(shownFrame) ? undefined : 'none' }}>
            {/* The label is anchored to the marker (edge-aware: see `anchor`), inside the sliding wrapper. */}
            <div ref={badgeInnerRef} className="absolute bottom-0 left-0 w-max" style={{ transform: anchor(currentPct) }}>
              <span ref={badgeTextRef} className="rounded-md bg-[var(--accent)] px-1.5 py-0.5 font-mono text-note font-bold tabular-nums text-white shadow-[0_2px_6px_rgba(0,0,0,0.5)]">
                {fmtTimecode(shownFrame, fps)}
              </span>
            </div>
          </div>
        </div>

        {/* Identity keyframes: the blended identity as a function of the frame, directly beneath the film track and the
            chapter markers carried down onto it. */}
        {identityKeys && (
          <>
            <LaneHeader title="Identity keyframes">
              <span className="font-mono text-note tabular-nums text-[var(--text-muted)]" data-testid="identity-keyframe-count">
                {idKf.length ? `${idKf.length} keyframe${idKf.length === 1 ? '' : 's'}` : 'none'}
                {idKf.length && identityKeys.recipe && !identityKeys.recipe.enabled ? ' · blender is off, not applied' : ''}
              </span>
              <button type="button" className={smallBtn} disabled={!identityKeys.canKey} data-testid="identity-keyframe-add"
                      onClick={() => identityKeys.add(frame)}
                      title={identityKeys.canKey ? 'Key the blended identity at the playhead (double-click the track to key any frame)' : 'Add two or more sources to the Identity Blender first'}>
                + Key at playhead
              </button>
              <button type="button" className={smallBtn} disabled={!idKf.length} data-testid="identity-keyframe-clear"
                      onClick={identityKeys.clear} title="Remove every identity keyframe on this clip">
                Clear
              </button>
            </LaneHeader>
            <IdentityKeyframeLane
              view={v} maxFrames={maxFrames} frame={frame} keyframes={idKf} selected={identityKeys.selected}
              canKey={identityKeys.canKey} markers={markers}
              onSeek={setFrame} onAdd={identityKeys.add} onMove={identityKeys.move}
              onRemove={identityKeys.remove} onSelect={identityKeys.setSelected}
            />
            {idSel >= 0 && (
              <IdentityKeyframeInspector
                keyframe={idKf[idSel]} nextKeyframe={idKf[idSel + 1] || null}
                components={identityKeys.components || EMPTY} recipe={identityKeys.recipe || { keyframes: idKf }}
                labelOf={identityKeys.labelOf || ((id) => id)}
                onUpdate={identityKeys.update} onRemove={identityKeys.remove} onClose={() => identityKeys.setSelected(null)}
              />
            )}
          </>
        )}

        {/* Parameter curves: a value-vs-frame graph on the same x-axis as every lane above and below it */}
        {curveEditor?.open && maxFrames > 1 && (
          <CurveEditor
            view={v} maxFrames={maxFrames} frame={frame}
            tracks={curveEditor.tracks} enabled={curveEditor.enabled} onEnabledChange={curveEditor.api.setEnabled}
            settings={curveEditor.settings} api={curveEditor.api} onSeek={setFrame}
            onClose={curveEditor.onToggle} morph={curveEditor.morph} planNote={curveEditor.planNote}
          />
        )}

        {/* Audio: what will play, drawn where it will play (the slip moves it against the video) */}
        {showAudioRow && (
          <>
            <LaneHeader title="Audio">
              <span className="min-w-0 truncate text-note text-[var(--text-muted)]" data-testid="audio-lane-source"
                    title={trackPath ? String(trackPath) : "The target video's own audio"}>
                {trackPath ? `External: ${audioTrack.trackName || String(trackPath).split(/[\\/]/).pop()}` : 'Target audio'}
                {trackPath && trackWave.status === 'loading' ? ' (reading…)' : ''}
                {trackPath && trackWave.status === 'error' ? ' (waveform unavailable)' : ''}
              </span>
              <span className="font-mono text-note tabular-nums text-[var(--text-main)]" data-testid="audio-lane-offset"
                    title={offsetMeaning(audioOffsetMs)}>{offsetText(audioOffsetMs)}</span>
              {[-10, -1, 1, 10].map((d) => (
                <button key={d} type="button" className={smallBtn} onClick={() => nudgeAudio(d)}
                        disabled={!audioTrack.onOffsetChange || clampOffset(audioOffsetMs + d) === audioOffsetMs}
                        aria-label={`Slip audio ${d > 0 ? 'later' : 'earlier'} by ${Math.abs(d)} ms`}
                        title={`Slip the audio ${d > 0 ? 'later' : 'earlier'} by ${Math.abs(d)} ms`}>
                  {d > 0 ? `+${d}` : `−${Math.abs(d)}`}
                </button>
              ))}
              <button type="button" className={smallBtn} onClick={() => audioTrack.onOffsetChange?.(0)}
                      disabled={!audioTrack.onOffsetChange || audioOffsetMs === 0} title="Put the audio back in step with the video">Reset</button>
            </LaneHeader>
            <AudioLane view={v} maxFrames={maxFrames} fps={fps} waves={audioWaves}
                       slipSeconds={trackShiftSeconds({ audio_offset_ms: audioOffsetMs })} active={!trackPath || trackWave.status === 'ready'}
                       label={`Audio waveform, ${offsetText(audioOffsetMs)} against the video`} />
          </>
        )}

        {/* Scene cuts + who is on screen */}
        <LaneHeader title="Scene cuts & faces">
          <span className="font-mono text-note tabular-nums text-[var(--text-muted)]">
            {cuts.length ? `${cuts.length} cut${cuts.length === 1 ? '' : 's'}` : 'no cuts found'}
            {scene ? ` · scene ${scene.index + 1}/${scenes.length}` : ''}
          </span>
          <button type="button" className={smallBtn} disabled={!hasCuts} onClick={() => jumpCut(-1)}
                  title="Jump to the previous scene cut (Up Arrow)">↑ Prev cut</button>
          <button type="button" className={smallBtn} disabled={!hasCuts} onClick={() => jumpCut(1)}
                  title="Jump to the next scene cut (Down Arrow)">↓ Next cut</button>
          <button type="button" className={smallBtn} disabled={scanning || targetIndex === null} onClick={scanPresence}
                  title="Sample the clip, cluster the faces and show where each person appears">
            {scanning ? 'Scanning…' : tracks.presence.length ? 'Rescan faces' : 'Scan faces'}
          </button>
        </LaneHeader>
        <SceneLane
          view={v} maxFrames={maxFrames} fps={fps} cuts={cuts} presence={tracks.presence}
          selectedPersonId={pick?.id} selectedInterval={pick?.interval} emptyNote={presenceNote}
          onSeek={setFrame} onJumpCut={setFrame} onPresenceClick={onPresenceClick}
        />

        {/* Identity confidence: where the match was weak, in amber */}
        {tracks.identity.length > 0 && (
          <>
            <LaneHeader title="Identity confidence">
              <span className="font-mono text-note tabular-nums text-[var(--text-muted)]" data-testid="identity-amber-count">
                {amberFrames.length ? `${amberFrames.length} amber` : 'all clear'}
              </span>
              <button type="button" className={smallBtn} disabled={!amberFrames.length}
                      data-testid="identity-prev"
                      onClick={() => { const f = nextAmber(amberFrames, frame, -1); if (f !== null) setFrame(f); }}
                      title="Jump to the previous frame where the identity match is near the threshold">◂ Prev amber</button>
              <button type="button" className={smallBtn} disabled={!amberFrames.length}
                      data-testid="identity-next"
                      onClick={() => { const f = nextAmber(amberFrames, frame, 1); if (f !== null) setFrame(f); }}
                      title="Jump to the next frame where the identity match is near the threshold">Next amber ▸</button>
            </LaneHeader>
            <IdentityLane view={v} maxFrames={maxFrames} rows={tracks.identity} onSeek={setFrame} />
          </>
        )}

        {/* Per-segment overrides */}
        <LaneHeader title="Scene overrides">
          {ov.overrides.length > 0 && (
            <>
              <span className="font-mono text-note tabular-nums text-[var(--text-muted)]">{ov.overrides.length}</span>
              <label className="inline-flex items-center gap-1 text-note text-[var(--text-muted)]"
                     title="Off: the overrides stay on the timeline, but the preview, the cache strip and the render use the clip's own settings">
                <input type="checkbox" checked={ov.enabled !== false} aria-label="Apply scene overrides"
                       onChange={(e) => ov.setEnabled?.(e.target.checked)} className="accent-[var(--accent)]" />
                Apply
              </label>
            </>
          )}
          <button type="button" className={smallBtn}
                  disabled={rangeIsWhole && !scene}
                  onClick={() => {
                    const r = rangeIsWhole ? { start: scene.start, end: scene.end, label: `Scene ${scene.index + 1}` }
                      : { start: startFrame, end: endFrame, label: 'In/Out range' };
                    ov.add({ start: r.start, end: r.end }, r.label);
                  }}
                  title={rangeIsWhole ? 'Add an override on the scene under the playhead' : 'Add an override on the In/Out range'}>
            + Add Scene Override
          </button>
          <button type="button" className={smallBtn}
                  disabled={!ov.overrides.length || !onRenderOverrides || !!renderBlockedReason}
                  onClick={onRenderOverrides}
                  title={renderBlockedReason || 'Queue the In/Out range as one job per piece (overrides applied), ready to join'}>
            Queue with overrides
          </button>
        </LaneHeader>
        <OverrideLane
          view={v} maxFrames={maxFrames} overrides={ov.overrides} selectedId={ov.selectedId}
          inout={inout} emptyNote={overrideNote}
          onSelect={ov.setSelectedId} onUpdate={ov.update} onSeek={setFrame}
        />

        {/* Playhead continued across every lane */}
        <div ref={stackLineRef}
             className="absolute inset-y-0 left-0 w-full z-30 pointer-events-none will-change-transform"
             style={{ transform: slide(currentPct), display: inView(shownFrame) ? undefined : 'none' }}>
          <div className="absolute inset-y-0 left-0 w-px -translate-x-1/2 bg-white/70" />
        </div>

        {/* Right-click menu. Positioned in percent of the stack so the app-level CSS zoom cannot offset it. */}
        {menu && (
          <div
            role="menu"
            aria-label="Timeline actions"
            data-testid="timeline-menu"
            onPointerDown={(e) => e.stopPropagation()}
            className="absolute z-[60] min-w-[220px] max-w-[320px] rounded-lg border border-[var(--border-strong)] bg-[var(--card-bg)] p-1 shadow-[0_12px_32px_rgba(0,0,0,0.55)] backdrop-blur-xl"
            style={{ left: `${menu.x}%`, top: `${menu.y}%`, transform: `translate(${menu.x > 60 ? '-100%' : '0'}, ${menu.y > 55 ? '-100%' : '0'})` }}
          >
            {menu.items.map((it) => (
              <button
                key={it.id} type="button" role="menuitem"
                onClick={() => { it.run(); setMenu(null); }}
                className="block w-full rounded-md px-2.5 py-1.5 text-left text-mini text-[var(--text-main)] hover:bg-white/[0.08]"
              >
                {it.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-note text-[var(--text-muted)]" data-testid="timeline-legend">
        {Object.keys(STATE_COLORS).map((k) => (
          <span key={k} className="inline-flex items-center gap-1.5">
            <span className="h-1.5 w-3 rounded-full" style={{ background: STATE_COLORS[k], opacity: k === 'dirty' ? 0.6 : 1 }} />
            {STATE_LABELS[k]}
          </span>
        ))}
        {!tracks.renderVideoMatches && <span className="opacity-70">· face-free / smart-skip fill in after a run</span>}
      </div>

      {selectedOverride && (
        <OverrideInspector
          override={selectedOverride}
          maxFrames={maxFrames}
          swapModels={swapModels}
          detectionMode={detectionMode}
          baseValues={baseValues}
          onChange={(patch) => ov.update(selectedOverride.id, patch)}
          onRemove={() => ov.remove(selectedOverride.id)}
          onJump={() => setFrame(selectedOverride.start)}
          onUseAsRange={() => { setFrameMarkerVal('start', selectedOverride.start); setFrameMarkerVal('end', selectedOverride.end); }}
        />
      )}

      {/* ── Toolbar: one surface, three zones ────────────────────────────── */}
      <div className="mt-2 flex flex-wrap items-center justify-between gap-x-4 gap-y-2.5 rounded-xl border border-[var(--border-color)] bg-[var(--card-bg)] px-3 py-2">
        {/* Read-out */}
        <div className="flex items-baseline gap-2 font-mono">
          <span className="text-lead font-semibold tabular-nums text-[var(--text-main)]" data-testid="timecode-readout">{fmtTimecode(frame, fps)}</span>
          <span className="text-mini tabular-nums text-muted">/ {fmtTimecode(maxFrames, fps)}</span>
          <span className="mx-1 h-4 w-px bg-[var(--border-color)] self-center" />
          <input
            type="number"
            min={1}
            max={maxFrames}
            value={frameDraft ?? frame}
            onFocus={(e) => { setFrameDraft(String(frame)); e.target.select(); }}
            onChange={(e) => setFrameDraft(e.target.value)}
            onBlur={() => {
              if (frameDraft !== null) setFrame(clamp(parseInt(frameDraft, 10) || frame, 1, maxFrames));
              setFrameDraft(null);
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') e.currentTarget.blur();
              if (e.key === 'Escape') { setFrameDraft(null); e.currentTarget.blur(); }
            }}
            className="w-[8ch] rounded-md border border-[var(--border-color)] bg-[var(--input-bg)] py-0.5 text-center text-mini font-semibold tabular-nums text-[var(--text-main)] outline-none transition-colors focus:border-[var(--accent)]"
            title="Type a frame number and press Enter to jump"
            aria-label="Frame number"
          />
          <span className="text-note tabular-nums text-muted">/ {maxFrames.toLocaleString()}</span>
          {shuttle.dir !== 0 && (
            <span className="ml-1 rounded-md bg-[var(--accent)]/15 px-1.5 py-0.5 text-note font-bold text-accent" data-testid="shuttle-readout">
              {shuttleLabel(shuttle)}
            </span>
          )}
        </div>

        {/* Transport */}
        <div className="spring-cluster flex items-center gap-0.5">
          <IconBtn title="Jump to In point" onClick={() => setFrame(startFrame)}>
            <Ico d={<><polygon points="19 20 9 12 19 4" fill="currentColor" stroke="none" /><line x1="5" y1="19" x2="5" y2="5" /></>} />
          </IconBtn>
          <IconBtn title="Previous marker" onClick={() => jumpMarker(-1)} className={markers.length ? '' : 'opacity-30'}>
            <Ico d="M15 6l-6 6 6 6" />
          </IconBtn>
          <IconBtn title="Previous frame (←)" onClick={() => step(-1)}>
            <Ico d={<><rect x="5" y="5" width="2.4" height="14" rx="1" fill="currentColor" stroke="none" /><polygon points="20 5 20 19 10 12" fill="currentColor" stroke="none" /></>} />
          </IconBtn>

          <button
            type="button"
            onClick={() => setIsPlaying(!isPlaying)}
            title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
            className={`mx-1 grid place-items-center h-10 w-10 rounded-full transition-colors ${
              isPlaying
                ? 'bg-[var(--surface-2)] border border-[var(--border-strong)] text-[var(--text-main)] hover:bg-white/[0.08]'
                : 'bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] shadow-[0_2px_10px_var(--accent-glow)]'
            }`}
          >
            {isPlaying && buffering
              // Held waiting for frames. Saying so is the difference between "this is loading" and
              // "this button is broken".
              ? <span className="h-4 w-4 rounded-full border-2 border-white/25 border-t-white animate-spin" />
              : isPlaying
                ? <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16" rx="1.2" /><rect x="14" y="4" width="4" height="16" rx="1.2" /></svg>
                : <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor" className="ml-0.5"><polygon points="6 3 21 12 6 21" /></svg>}
          </button>

          <IconBtn title="Next frame (→)" onClick={() => step(1)}>
            <Ico d={<><rect x="16.6" y="5" width="2.4" height="14" rx="1" fill="currentColor" stroke="none" /><polygon points="4 5 4 19 14 12" fill="currentColor" stroke="none" /></>} />
          </IconBtn>
          <IconBtn title="Next marker" onClick={() => jumpMarker(1)} className={markers.length ? '' : 'opacity-30'}>
            <Ico d="M9 6l6 6-6 6" />
          </IconBtn>
          <IconBtn title="Jump to Out point" onClick={() => setFrame(endFrame)}>
            <Ico d={<><polygon points="5 4 15 12 5 20" fill="currentColor" stroke="none" /><line x1="19" y1="5" x2="19" y2="19" /></>} />
          </IconBtn>
        </div>

        {/* Range + playback options */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="spring-cluster flex items-center gap-0.5 rounded-lg border border-[var(--border-color)] bg-[var(--surface-2)] p-0.5">
            <button type="button" onClick={() => setFrameMarkerVal('start', frame)}
                    title="Set In point to the current frame (I or [)"
                    className="px-2 py-1 rounded-md text-mini font-semibold text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.06] transition-colors">
              Set in
            </button>
            <button type="button" onClick={() => setFrameMarkerVal('end', frame)}
                    title="Set Out point to the current frame (O or ])"
                    className="px-2 py-1 rounded-md text-mini font-semibold text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.06] transition-colors">
              Set out
            </button>
            <button type="button"
                    onClick={async () => { await setFrameMarkerVal('start', 1); await setFrameMarkerVal('end', maxFrames); }}
                    title="Reset the range to the whole clip (R)"
                    className="px-2 py-1 rounded-md text-mini font-semibold text-muted hover:text-[var(--text-main)] hover:bg-white/[0.06] transition-colors">
              Full
            </button>
          </div>

          <button type="button"
                  onClick={() => setIsLooping(!isLooping)}
                  aria-pressed={isLooping}
                  title="Loop the work area: playback wraps from Out back to In"
                  className={`rounded-lg border px-2 py-1.5 text-mini font-semibold transition-colors ${
                    isLooping
                      ? 'border-[var(--accent)]/40 bg-[var(--accent)]/12 text-accent'
                      : 'border-[var(--border-color)] bg-[var(--surface-2)] text-[var(--text-muted)] hover:text-[var(--text-main)]'}`}>
            Loop Work Area
          </button>

          {onRenderRange && (
            <button type="button"
                    onClick={onRenderRange}
                    disabled={!!renderBlockedReason}
                    data-testid="render-range"
                    title={renderBlockedReason || `Render only frames ${startFrame}-${endFrame}`}
                    className="rounded-lg bg-[var(--accent)] px-2.5 py-1.5 text-mini font-semibold text-white hover:bg-[var(--accent-hover)] disabled:opacity-40 transition-colors">
              Render In→Out Range Only
            </button>
          )}

          <div className="spring-cluster flex items-center gap-0.5 rounded-lg border border-[var(--border-color)] bg-[var(--surface-2)] p-0.5">
            <IconBtn title={markerAtPlayhead ? 'Remove the marker here (M)' : 'Add a marker at the playhead (M)'}
                     onClick={toggleMarker} active={markerAtPlayhead}>
              <Ico size={13} d={<path d="M5 3v18l7-5 7 5V3z" fill={markerAtPlayhead ? 'currentColor' : 'none'} />} />
            </IconBtn>
            {markers.length > 0 && (
              <button type="button" onClick={() => persistMarkers([])}
                      title={`Clear all ${markers.length} markers`}
                      className="px-1.5 py-1 rounded-md font-mono text-note font-bold tabular-nums text-amber-300/80 hover:text-amber-200 hover:bg-white/[0.06] transition-colors">
                {markers.length}✕
              </button>
            )}
          </div>

          <div className="spring-cluster flex items-center rounded-lg border border-[var(--border-color)] bg-[var(--surface-2)] p-0.5" title="Playback speed (J / K / L shuttle)">
            {SPEEDS.map((r) => (
              <button key={r} type="button" onClick={() => setPlaybackRate(r)}
                      className={`px-1.5 py-1 rounded-md text-note font-bold tabular-nums transition-colors ${
                        playbackRate === r
                          ? 'bg-[var(--accent)] text-white'
                          : 'text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.06]'}`}>
                {r}×
              </button>
            ))}
          </div>

          {/* Zoom */}
          <div className="spring-cluster flex items-center gap-1 rounded-lg border border-[var(--border-color)] bg-[var(--surface-2)] p-0.5 pr-2"
               title="Zoom the timeline - Ctrl/Cmd + wheel zooms about the cursor, Shift + wheel scrolls">
            <IconBtn title="Zoom out" onClick={() => zoomAround(frame, 1.6)}>
              <Ico d={<><circle cx="11" cy="11" r="7" /><line x1="20" y1="20" x2="16.2" y2="16.2" /><line x1="8" y1="11" x2="14" y2="11" /></>} />
            </IconBtn>
            <input
              type="range" min={0} max={1} step={0.005} value={zoomT}
              onChange={onZoomSlider}
              aria-label="Timeline zoom"
              data-testid="timeline-zoom"
              className="h-1 w-24 accent-[var(--accent)]"
            />
            <IconBtn title="Zoom in on the playhead" onClick={() => zoomAround(frame, 0.6)}>
              <Ico d={<><circle cx="11" cy="11" r="7" /><line x1="20" y1="20" x2="16.2" y2="16.2" /><line x1="8" y1="11" x2="14" y2="11" /><line x1="11" y1="8" x2="11" y2="14" /></>} />
            </IconBtn>
            <button type="button"
                    onClick={() => (zoomed ? resetView() : setViewSafe(fitRange(startFrame, endFrame, maxFrames, trackW)))}
                    title={zoomed ? 'Zoom out to the whole clip' : 'Zoom to the selected In/Out range'}
                    className="px-2 py-1 rounded-md text-note font-bold text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.06] transition-colors">
              {zoomed ? 'All' : 'Range'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
