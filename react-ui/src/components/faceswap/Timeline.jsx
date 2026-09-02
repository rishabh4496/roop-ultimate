import React, { useEffect, useMemo, useRef, useState } from 'react';
import useThrottledFrameRequest from './useThrottledFrameRequest';

/**
 * The clip timeline — an editor-style scrubber for the target video.
 *
 * Replaces the earlier inline block, which read dated for three reasons: the
 * "ruler" was five evenly-spaced timecodes that lined up with nothing on the
 * track, the trim range was stated twice (chips in the header AND a numeric
 * Range box in the toolbar), and the toolbar was four separately-bordered pill
 * groups on one wrapping row, so the eye had no order to follow.
 *
 * What it is now: a measured tick ruler whose marks sit at real time positions,
 * one editable statement of In/Out (the header chips ARE the inputs), a taller
 * filmstrip with pro-editor trim treatment (out-of-range is dimmed AND
 * desaturated), a playhead carrying its own timecode, and a single toolbar
 * surface split into read-out / transport / range zones. New: chapter markers
 * (add at the playhead, click to jump, alt-click to delete) persisted per clip.
 *
 * Scrub/drag geometry still lives in FaceSwap (it owns the pointer capture and
 * the magnetic snap); this component takes the ref and the three handlers and
 * renders on top of them, so the interaction contract is unchanged.
 */

// Label steps a human would choose. The ruler picks the smallest one that keeps
// labels at least ~92px apart at the current track width, so the same clip is
// legible in a narrow rail and a wide window without re-tuning.
const NICE_SECONDS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200];
// Used instead of NICE_SECONDS once the view is zoomed past ~1s per label.
const NICE_FRAMES = [1, 2, 5, 10, 15, 25, 50, 100];
const SPEEDS = [0.25, 0.5, 1, 2, 4];
const MINOR_PER_MAJOR = 4;

const clamp = (v, lo, hi) => Math.max(lo, Math.min(v, hi));

/** m:ss under an hour, h:mm:ss above it. */
const fmtTC = (f, fps) => {
  const total = Math.max(0, f) / (fps || 25);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = Math.floor(total % 60);
  return h > 0
    ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
    : `${m}:${String(s).padStart(2, '0')}`;
};

/** Timecode with the frame remainder — what the playhead badge shows. */
const fmtTCF = (f, fps) => {
  const r = (fps || 25);
  const idx = Math.max(0, f - 1);
  return `${fmtTC(f, r)}:${String(Math.floor(idx % r)).padStart(2, '0')}`;
};

const markersKey = (k) => `roop_markers_${k || 'default'}`;

/** Chip-shaped inline number field — the header's In/Out are editable in place. */
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
      title={`${label} point — click to type a frame number`}
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
const IconBtn = ({ title, onClick, active = false, children, className = '' }) => (
  <button
    type="button"
    onClick={onClick}
    title={title}
    aria-label={title}
    className={`grid place-items-center h-8 w-8 rounded-lg transition-colors ${
      active
        ? 'bg-[var(--accent)]/12 text-[var(--accent)]'
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

/**
 * Whole-clip overview with the visible window drawn on it.
 *
 * Its own pointer handling (the main track's belongs to FaceSwap and means
 * "scrub"): press inside the window to drag it, on an edge to resize it,
 * anywhere else to centre it there.
 */
function OverviewStrip({ maxFrames, vStart, vEnd, frame, startFrame, endFrame, markers, setView }) {
  const ref = useRef(null);
  const dragRef = useRef(null);
  const span = Math.max(1, maxFrames - 1);
  const pct = (f) => ((clamp(f, 1, maxFrames) - 1) / span) * 100;
  const vSpan = Math.max(1, vEnd - vStart);

  const frameAt = (clientX) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect || !rect.width) return 1;
    return clamp(Math.round(1 + ((clientX - rect.left) / rect.width) * span), 1, maxFrames);
  };
  const place = (start, width) => {
    if (!setView) return;
    const s = clamp(Math.round(start), 1, Math.max(1, maxFrames - width));
    setView({ start: s, end: Math.min(maxFrames, s + width) });
  };

  // The drag handler needs the CURRENT window, but binding the listeners to it
  // would re-subscribe on every playback tick (the auto-follow moves the window
  // while playing). Latest values go through a ref so the listeners bind once.
  const liveRef = useRef(null);
  liveRef.current = { vStart, vEnd, vSpan, frameAt, place, setView, maxFrames };
  useEffect(() => {
    const onMove = (e) => {
      const d = dragRef.current;
      const L = liveRef.current;
      if (!d || !L || !L.setView) return;
      const f = L.frameAt(e.clientX);
      if (d.mode === 'move') L.place(f - d.grab, L.vSpan);
      // Never let a resize collapse the window below the zoom floor.
      else if (d.mode === 'left') L.setView({ start: clamp(f, 1, L.vEnd - 24), end: L.vEnd });
      else L.setView({ start: L.vStart, end: clamp(f, L.vStart + 24, L.maxFrames) });
    };
    const onUp = () => { dragRef.current = null; };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
  }, []);

  const onDown = (e) => {
    const f = frameAt(e.clientX);
    const rect = ref.current?.getBoundingClientRect();
    const edgePx = 6;
    const edgeFrames = rect ? Math.max(1, Math.round((edgePx / rect.width) * span)) : 1;
    if (Math.abs(f - vStart) <= edgeFrames) dragRef.current = { mode: 'left' };
    else if (Math.abs(f - vEnd) <= edgeFrames) dragRef.current = { mode: 'right' };
    else if (f > vStart && f < vEnd) dragRef.current = { mode: 'move', grab: f - vStart };
    else { place(f - Math.round(vSpan / 2), vSpan); dragRef.current = { mode: 'move', grab: Math.round(vSpan / 2) }; }
  };

  return (
    <div
      ref={ref}
      onPointerDown={onDown}
      className="relative mt-3 h-6 w-full rounded-md border border-[var(--border-color)] bg-[var(--input-bg)] overflow-hidden cursor-pointer"
      title="Whole clip — drag the lit window to pan, its edges to zoom"
    >
      {/* selected render range, for orientation */}
      <div className="absolute inset-y-0 bg-[var(--accent)]/15 pointer-events-none"
           style={{ left: `${pct(startFrame)}%`, width: `${Math.max(0, pct(endFrame) - pct(startFrame))}%` }} />
      {markers.map((m) => (
        <span key={m} className="absolute inset-y-0 w-px bg-amber-300/50 pointer-events-none" style={{ left: `${pct(m)}%` }} />
      ))}
      <span className="absolute inset-y-0 w-px bg-white/80 pointer-events-none" style={{ left: `${pct(frame)}%` }} />
      {/* the visible window */}
      <div className="absolute inset-y-0 rounded-[3px] border border-white/45 bg-white/[0.10] pointer-events-none"
           style={{ left: `${pct(vStart)}%`, width: `${Math.max(1.2, pct(vEnd) - pct(vStart))}%` }}>
        <span className="absolute inset-y-0 -left-px w-[3px] rounded-full bg-white/70" />
        <span className="absolute inset-y-0 -right-px w-[3px] rounded-full bg-white/70" />
      </div>
    </div>
  );
}

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
  storyboardThumbs = [],
  isPlaying = false,
  setIsPlaying,
  buffering = false,
  isLooping = false,
  setIsLooping,
  playbackRate = 1,
  setPlaybackRate,
  thumbUrl,
  targetKey = '',
  view,
  setView,
  segments = [],
  onSegmentClick,
}) {
  // ── View window ───────────────────────────────────────────────────────────
  // The track shows [view.start, view.end] rather than the whole clip. `pctOf`
  // is therefore a position on the VISIBLE track and can fall outside 0..100
  // for frames scrolled off either edge — callers either clamp it (labels) or
  // hide the element (handles, markers, playhead).
  // Fall back to the whole clip for a missing or degenerate window — the parent
  // seeds it in an effect, so the first paint after a target change would
  // otherwise map every position against a one-frame span.
  const valid = view && view.end > view.start;
  const vStart = valid ? view.start : 1;
  const vEnd = valid ? Math.min(view.end, maxFrames) : maxFrames;
  const vSpan = Math.max(1, vEnd - vStart);
  const zoomed = vSpan < maxFrames - 1;
  const pctOf = (f) => ((f - vStart) / vSpan) * 100;
  const inView = (f) => f >= vStart && f <= vEnd;

  const startPct = pctOf(startFrame);
  const endPct = pctOf(endFrame);
  const currentPct = pctOf(frame);
  const rangeLen = Math.max(0, endFrame - startFrame + 1);
  const rangeShare = Math.round((rangeLen / maxFrames) * 100);

  /** Re-window around a pivot frame. factor > 1 zooms out, < 1 zooms in. */
  const zoomAround = (pivot, factor) => {
    if (!setView || maxFrames < 2) return;
    // Floor at 24 visible frames — below that the filmstrip is a handful of
    // near-identical stills and the frame box is the better tool anyway.
    const nextSpan = clamp(Math.round(vSpan * factor), Math.min(24, maxFrames - 1), maxFrames - 1);
    const ratio = clamp((pivot - vStart) / vSpan, 0, 1);
    let s = Math.round(pivot - ratio * nextSpan);
    s = clamp(s, 1, Math.max(1, maxFrames - nextSpan));
    setView({ start: s, end: Math.min(maxFrames, s + nextSpan) });
  };

  const panBy = (deltaFrames) => {
    if (!setView) return;
    const s = clamp(vStart + deltaFrames, 1, Math.max(1, maxFrames - vSpan));
    setView({ start: s, end: Math.min(maxFrames, s + vSpan) });
  };

  const resetView = () => setView && setView({ start: 1, end: maxFrames });

  // Wheel over the track zooms about the cursor. Must be a NATIVE non-passive
  // listener: React's onWheel is registered passive, so preventDefault there is
  // ignored and the page scrolls behind the zoom.
  //
  // The handler reads the current window from a ref and the listener is bound
  // ONCE. Keying it to [vStart, vSpan] instead meant re-subscribing on every
  // zoom step, and with React free to defer the effect flush a fast wheel spin
  // could hand several events to a listener still holding the previous window —
  // each computing its step from a stale base, which reads as the zoom
  // stuttering or fighting the cursor.
  const wheelRef = useRef(null);
  wheelRef.current = { vStart, vSpan, zoomAround, panBy, maxFrames };
  useEffect(() => {
    const el = timelineRef?.current;
    if (!el) return;
    const onWheel = (e) => {
      const W = wheelRef.current;
      if (!W || W.maxFrames < 2) return;
      // Ctrl/Cmd + wheel is the app zoom (App.jsx, on window), which this
      // listener would otherwise fire alongside — zooming the track and the
      // whole UI on one gesture.
      if (e.ctrlKey || e.metaKey) return;
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      if (!rect.width) return;
      const pivot = W.vStart + clamp((e.clientX - rect.left) / rect.width, 0, 1) * W.vSpan;
      // Shift-wheel pans instead of zooming, matching every NLE.
      if (e.shiftKey) W.panBy(Math.round((e.deltaY > 0 ? 0.15 : -0.15) * W.vSpan));
      else W.zoomAround(pivot, e.deltaY > 0 ? 1.25 : 0.8);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [timelineRef]);

  // Keep the playhead on screen while playing — a zoomed-in track that the
  // playhead has already run off is just a still picture.
  useEffect(() => {
    if (!isPlaying || !zoomed || !setView) return;
    if (frame >= vStart && frame <= vEnd) return;
    const s = clamp(Math.round(frame - vSpan * 0.25), 1, Math.max(1, maxFrames - vSpan));
    setView({ start: s, end: Math.min(maxFrames, s + vSpan) });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame, isPlaying, zoomed]);

  // ── Ruler ticks, measured from the real track width ───────────────────────
  const [trackW, setTrackW] = useState(0);
  useEffect(() => {
    const el = timelineRef?.current;
    if (!el) return;
    setTrackW(el.getBoundingClientRect().width);
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(([e]) => setTrackW(e.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, [timelineRef, maxFrames]);

  const ticks = useMemo(() => {
    if (!trackW || maxFrames < 2) return { major: [], minor: [], frameMode: false };
    const want = Math.max(2, Math.floor(trackW / 92));
    let step;
    let frameMode = false;
    if (vSpan / fps / want < 1) {
      // Zoomed past one second per label — step in frames and show the frame
      // remainder, otherwise every tick would read the same second.
      const raw = vSpan / want;
      step = NICE_FRAMES.find((n) => n >= raw) ?? Math.max(1, Math.round(fps));
      frameMode = true;
    } else {
      const sec = NICE_SECONDS.find((s) => s >= vSpan / fps / want) ?? NICE_SECONDS[NICE_SECONDS.length - 1];
      step = Math.max(1, Math.round(sec * fps));
    }
    // Ticks stay pinned to absolute clip time (multiples of `step` from frame 1)
    // so they don't slide about under the material while panning.
    const first = 1 + Math.ceil((vStart - 1) / step) * step;
    const major = [];
    const minor = [];
    for (let f = first - step; f <= vEnd + step; f += step) {
      if (f >= vStart && f <= vEnd && f >= 1) major.push(f);
      for (let k = 1; k < MINOR_PER_MAJOR; k++) {
        const mf = Math.round(f + (step * k) / MINOR_PER_MAJOR);
        if (mf >= vStart && mf <= vEnd && mf >= 1 && mf <= maxFrames) minor.push(mf);
      }
    }
    return { major, minor, frameMode };
  }, [trackW, maxFrames, fps, vStart, vEnd, vSpan]);

  // ── Chapter markers, per clip ─────────────────────────────────────────────
  const [markers, setMarkers] = useState([]);
  useEffect(() => {
    try {
      const raw = localStorage.getItem(markersKey(targetKey));
      const list = raw ? JSON.parse(raw) : [];
      setMarkers(Array.isArray(list) ? list.filter((n) => Number.isFinite(n)) : []);
    } catch { setMarkers([]); }
  }, [targetKey]);

  const persistMarkers = (list) => {
    const next = [...new Set(list)].filter((f) => f >= 1 && f <= maxFrames).sort((a, b) => a - b);
    setMarkers(next);
    try {
      if (next.length) localStorage.setItem(markersKey(targetKey), JSON.stringify(next));
      else localStorage.removeItem(markersKey(targetKey));
    } catch { /* private mode / quota — markers are a convenience, not state we own */ }
  };

  const markerAtPlayhead = markers.includes(frame);
  const toggleMarker = () =>
    persistMarkers(markerAtPlayhead ? markers.filter((m) => m !== frame) : [...markers, frame]);

  // 'M' toggles a marker at the playhead. Same guards as the app's global
  // shortcut handler so it never fires while typing or behind a modal. The
  // handler is reached through a ref so the listener is bound once instead of
  // re-subscribing on every frame the playback loop paints.
  const toggleRef = useRef(toggleMarker);
  toggleRef.current = toggleMarker;
  useEffect(() => {
    const onKey = (e) => {
      if (e.key.toLowerCase() !== 'm' || e.ctrlKey || e.metaKey || e.altKey) return;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
      if (document.querySelector('[role="dialog"]')) return;
      e.preventDefault();
      toggleRef.current();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const jumpMarker = (dir) => {
    const next = dir > 0 ? markers.find((m) => m > frame) : [...markers].reverse().find((m) => m < frame);
    if (next !== undefined) setFrame(next);
  };

  // ── Frame read-out field ──────────────────────────────────────────────────
  const [frameDraft, setFrameDraft] = useState(null);

  // Hover thumbnail, loaded one at a time. Sweeping the pointer across the track
  // names a new frame roughly every pixel, and each one is a video seek on the
  // same decoder the playhead's own frame needs — so an unthrottled hover strip
  // could bury the main preview under a hundred requests for stills that were
  // out of date before they finished.
  //
  // This is the SAME decoder the scrubber competes for, so it gets the same
  // treatment: 150 ms throttle, one in flight, and the superseded request
  // ABORTED rather than left to run. A hover thumbnail is decoration; it must
  // never be able to delay the frame the user is actually looking at, and the
  // previous single-flight-only version could, because an <img> load cannot be
  // cancelled. The <img> below points at the same URL and is answered from the
  // browser cache the request already filled.
  const { frameSrc: hoverSrc } = useThrottledFrameRequest(
    hoverFrame !== null && thumbUrl ? thumbUrl(hoverFrame) : '',
    { throttleMs: 150 });

  // Labels at the very edges would hang off the track; anchor those to the edge
  // instead of centring them.
  const anchor = (pct) => (pct < 3 ? 'translateX(0)' : pct > 97 ? 'translateX(-100%)' : 'translateX(-50%)');

  const step = (d) => setFrame((f) => clamp(f + d, 1, maxFrames));

  return (
    <div className="space-y-2.5 select-none">
      {/* ── Header: identity + the single, editable statement of the range ── */}
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2">
        <div className="flex items-baseline gap-2.5 min-w-0">
          <span className="text-mini font-semibold uppercase tracking-[0.18em] text-[var(--text-muted)]">Timeline</span>
          <span className="font-mono text-micro tabular-nums text-[var(--text-muted)]/55 truncate">
            {maxFrames.toLocaleString()} frames · {fps} fps · {fmtTC(maxFrames, fps)}
          </span>
          {zoomed && (
            <span className="rounded-md border border-[var(--border-color)] bg-[var(--surface-2)] px-1.5 py-0.5 font-mono text-nano tabular-nums text-[var(--text-muted)]"
                  title="Visible range — the track is zoomed in">
              showing {fmtTC(vStart, fps)}–{fmtTC(vEnd, fps)} · {Math.round((maxFrames - 1) / vSpan)}×
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          <RangeField label="In" value={startFrame} min={1} max={endFrame}
                      onCommit={(v) => setFrameMarkerVal('start', v)} />
          <RangeField label="Out" value={endFrame} min={startFrame} max={maxFrames}
                      onCommit={(v) => setFrameMarkerVal('end', v)} />
          <span className="inline-flex items-baseline gap-1.5 rounded-lg border border-[var(--accent)]/25 bg-[var(--accent)]/[0.08] px-2 py-1 font-mono text-mini tabular-nums text-[var(--accent)]"
                title="Length of the selected range — this is what gets rendered">
            {fmtTC(rangeLen, fps)}
            <span className="text-nano opacity-60">{rangeLen.toLocaleString()} f · {rangeShare}%</span>
          </span>
        </div>
      </div>

      {/* ── Ruler: real ticks + marker flags ─────────────────────────────── */}
      <div className="relative h-6">
        <div className="absolute inset-x-0 bottom-0 h-px bg-[var(--border-color)]" />
        {ticks.minor.map((f, i) => (
          <span key={`n${i}`} className="absolute bottom-0 w-px h-1.5 bg-[var(--text-muted)]/20"
                style={{ left: `${pctOf(f)}%` }} />
        ))}
        {ticks.major.map((f) => {
          const pct = pctOf(f);
          return (
            <React.Fragment key={f}>
              <span className="absolute bottom-0 w-px h-2.5 bg-[var(--text-muted)]/40" style={{ left: `${pct}%` }} />
              <span className="absolute bottom-3 font-mono text-nano tabular-nums text-[var(--text-muted)]/60 whitespace-nowrap"
                    style={{ left: `${pct}%`, transform: anchor(pct) }}>
                {ticks.frameMode ? fmtTCF(f, fps) : fmtTC(f, fps)}
              </span>
            </React.Fragment>
          );
        })}

        {/* Chapter markers — click to jump, alt-click to delete. */}
        {markers.filter(inView).map((m) => (
          <button
            key={m}
            type="button"
            onClick={(e) => { if (e.altKey) persistMarkers(markers.filter((x) => x !== m)); else setFrame(m); }}
            title={`Marker at frame ${m} (${fmtTC(m, fps)}) — click to jump, alt-click to remove`}
            className="absolute bottom-0 -translate-x-1/2 h-3.5 w-2.5 z-10 grid place-items-end"
            style={{ left: `${pctOf(m)}%` }}
          >
            <span className={`block h-2.5 w-2 rounded-[2px] transition-transform hover:scale-125 ${
              m === frame ? 'bg-amber-300' : 'bg-amber-400/70'}`} />
          </button>
        ))}
      </div>

      {/* ── Track ────────────────────────────────────────────────────────── */}
      <div className="relative">
        {/* Hover scrub thumbnail */}
        {hoverFrame !== null && thumbUrl && hoverSrc && (
          <div
            className="absolute bottom-[104px] z-50 flex flex-col items-center pointer-events-none -translate-x-1/2"
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
              <div className="mt-1 flex items-baseline justify-between px-0.5 font-mono text-micro tabular-nums">
                <span className="text-[var(--text-main)] font-semibold">{fmtTC(hoverFrame, fps)}</span>
                <span className="text-[var(--text-muted)]/60">f {hoverFrame.toLocaleString()}</span>
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
          className="relative h-24 w-full rounded-xl bg-[var(--input-bg)] border border-[var(--border-color)] overflow-hidden cursor-ew-resize timeline-track"
        >
          {/* Filmstrip. Shown at full strength — the old 45% wash made every clip
              look like grey mud; legibility comes from the scrim below instead. */}
          {storyboardThumbs.length > 0 && (
            <div className="absolute inset-0 flex pointer-events-none">
              {/* An empty entry is a frame the server could not read. It keeps
                  its slot (so the stills stay under the right timecodes) but
                  must not become <img src="">, which re-requests the page. */}
              {storyboardThumbs.map((url, i) => (
                url
                  ? <img key={i} src={url} alt="" loading="lazy"
                         className="flex-1 min-w-0 h-full object-cover"
                         onError={(e) => { e.currentTarget.style.opacity = '0'; }} />
                  : <span key={i} className="flex-1 min-w-0 h-full" />
              ))}
            </div>
          )}
          <div className="absolute inset-0 pointer-events-none bg-gradient-to-t from-black/70 via-black/10 to-black/25" />

          {/* Out-of-range: dimmed AND desaturated, the way an NLE greys the
              material you have trimmed away — reads instantly as "not rendered".
              Percentages are clamped because once the view is zoomed the In/Out
              points can sit outside it: a negative width is invalid CSS, which
              the browser drops, and the dim would silently vanish. */}
          <div className="absolute inset-y-0 left-0 z-10 pointer-events-none bg-black/55 backdrop-saturate-0"
               style={{ width: `${clamp(startPct, 0, 100)}%` }} />
          <div className="absolute inset-y-0 right-0 z-10 pointer-events-none bg-black/55 backdrop-saturate-0"
               style={{ left: `${clamp(endPct, 0, 100)}%` }} />

          {/* Active range rails */}
          <div className="absolute z-10 pointer-events-none border-y-2 border-[var(--accent)]/70 inset-y-0"
               style={{ left: `${clamp(startPct, 0, 100)}%`,
                        width: `${Math.max(0, clamp(endPct, 0, 100) - clamp(startPct, 0, 100))}%` }} />

          {/* In / Out handles — only when the point is actually on screen */}
          {[['start', startFrame, startPct], ['end', endFrame, endPct]]
            .filter(([, f]) => inView(f))
            .map(([which, , pct]) => (
            <div key={which} className="absolute inset-y-0 z-20 w-[3px] -translate-x-1/2 bg-white/90 pointer-events-none"
                 style={{ left: `${pct}%` }}>
              <span className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-9 w-2.5 rounded-full bg-white shadow-[0_1px_4px_rgba(0,0,0,0.7)] grid place-items-center gap-[2px]">
                <span className="block h-[7px] w-px bg-black/25" />
              </span>
              <span className={`absolute top-1 text-nano font-bold uppercase tracking-wider text-white/70 ${
                which === 'start' ? 'left-1.5' : 'right-1.5'}`}>
                {which === 'start' ? 'In' : 'Out'}
              </span>
            </div>
          ))}

          {/* Marker ticks carried down onto the track */}
          {markers.filter(inView).map((m) => (
            <span key={m} className="absolute inset-y-0 w-px bg-amber-300/45 z-10 pointer-events-none"
                  style={{ left: `${pctOf(m)}%` }} />
          ))}

          {/* Segment bands. Drawn as a thin strip along the bottom rather than a
              full-height tint so several of them stay readable over the
              filmstrip, and so they never compete with the In/Out shading that
              says what THIS run will render. Clipped to the visible window. */}
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
                title={`Segment ${i + 1}: frames ${s.start}–${s.end}`}
                aria-label={`Segment ${i + 1}, frames ${s.start} to ${s.end}`}
              />
            );
          })}

          {/* Hover line */}
          {hoverFrame !== null && (
            <div className="absolute inset-y-0 w-px -translate-x-1/2 bg-white/35 z-20 pointer-events-none"
                 style={{ left: `${pctOf(hoverFrame)}%` }} />
          )}

          {/* Playhead */}
          <div className={`absolute inset-y-0 z-30 pointer-events-none ${isScrubbing ? '' : 'transition-[left] duration-100 ease-out'}`}
               style={{ left: `${currentPct}%` }}>
            <div className="absolute inset-y-0 left-0 -translate-x-1/2 w-[2px] bg-white shadow-[0_0_0_1px_rgba(0,0,0,0.45)]" />
            <div className="absolute -top-px left-0 -translate-x-1/2 h-0 w-0 border-x-[6px] border-x-transparent border-t-[8px] border-t-[var(--accent)]" />
            <div className="absolute bottom-1 left-0 -translate-x-1/2 h-2 w-2 rounded-full bg-white shadow-[0_1px_3px_rgba(0,0,0,0.6)]" />
          </div>
        </div>

        {/* Playhead timecode, riding under the track. Hidden when the playhead
            has been scrolled out of the window — it sits outside the track's
            overflow clip, so it would otherwise float free of its own line. */}
        {inView(frame) && (
          <div className={`absolute -bottom-2 z-30 pointer-events-none ${isScrubbing ? '' : 'transition-[left] duration-100 ease-out'}`}
               style={{ left: `${currentPct}%`, transform: anchor(currentPct) }}>
            <span className="rounded-md bg-[var(--accent)] px-1.5 py-0.5 font-mono text-nano font-bold tabular-nums text-white shadow-[0_2px_6px_rgba(0,0,0,0.5)]">
              {fmtTCF(frame, fps)}
            </span>
          </div>
        )}
      </div>

      {/* ── Overview strip ───────────────────────────────────────────────────
          The whole clip at a glance with the visible window drawn on it: the
          one thing a zoomed track cannot tell you is where in the clip you are.
          Drag the window to pan, drag its edges to resize, click anywhere to
          jump the window there. Only shown once zoomed — at full extent it
          would just restate the track. */}
      {zoomed && (
        <OverviewStrip
          maxFrames={maxFrames}
          vStart={vStart}
          vEnd={vEnd}
          frame={frame}
          startFrame={startFrame}
          endFrame={endFrame}
          markers={markers}
          setView={setView}
        />
      )}

      {/* ── Toolbar: one surface, three zones ────────────────────────────── */}
      <div className="mt-4 flex flex-wrap items-center justify-between gap-x-4 gap-y-2.5 rounded-xl border border-[var(--border-color)] bg-[var(--card-bg)] px-3 py-2">
        {/* Read-out */}
        <div className="flex items-baseline gap-2 font-mono">
          <span className="text-lead font-semibold tabular-nums text-[var(--text-main)]">{fmtTC(frame, fps)}</span>
          <span className="text-mini tabular-nums text-[var(--text-muted)]/45">/ {fmtTC(maxFrames, fps)}</span>
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
          />
          <span className="text-micro tabular-nums text-[var(--text-muted)]/40">/ {maxFrames.toLocaleString()}</span>
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
              // Held waiting for frames. Saying so is the difference between
              // "this is loading" and "this button is broken".
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
        <div className="flex items-center gap-2">
          <div className="spring-cluster flex items-center gap-0.5 rounded-lg border border-[var(--border-color)] bg-[var(--surface-2)] p-0.5">
            <button type="button" onClick={() => setFrameMarkerVal('start', frame)}
                    title="Set In point to the current frame ([)"
                    className="px-2 py-1 rounded-md text-mini font-semibold text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.06] transition-colors">
              Set in
            </button>
            <button type="button" onClick={() => setFrameMarkerVal('end', frame)}
                    title="Set Out point to the current frame (])"
                    className="px-2 py-1 rounded-md text-mini font-semibold text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.06] transition-colors">
              Set out
            </button>
            <button type="button"
                    onClick={async () => { await setFrameMarkerVal('start', 1); await setFrameMarkerVal('end', maxFrames); }}
                    title="Reset the range to the whole clip (R)"
                    className="px-2 py-1 rounded-md text-mini font-semibold text-[var(--text-muted)]/70 hover:text-[var(--text-main)] hover:bg-white/[0.06] transition-colors">
              Full
            </button>
          </div>

          <div className="spring-cluster flex items-center gap-0.5 rounded-lg border border-[var(--border-color)] bg-[var(--surface-2)] p-0.5">
            <IconBtn title={markerAtPlayhead ? 'Remove the marker here (M)' : 'Add a marker at the playhead (M)'}
                     onClick={toggleMarker} active={markerAtPlayhead}>
              <Ico size={13} d={<path d="M5 3v18l7-5 7 5V3z" fill={markerAtPlayhead ? 'currentColor' : 'none'} />} />
            </IconBtn>
            {markers.length > 0 && (
              <button type="button" onClick={() => persistMarkers([])}
                      title={`Clear all ${markers.length} markers`}
                      className="px-1.5 py-1 rounded-md font-mono text-micro font-bold tabular-nums text-amber-300/80 hover:text-amber-200 hover:bg-white/[0.06] transition-colors">
                {markers.length}✕
              </button>
            )}
          </div>

          <div className="spring-cluster flex items-center rounded-lg border border-[var(--border-color)] bg-[var(--surface-2)] p-0.5" title="Playback speed">
            {SPEEDS.map((r) => (
              <button key={r} type="button" onClick={() => setPlaybackRate(r)}
                      className={`px-1.5 py-1 rounded-md text-micro font-bold tabular-nums transition-colors ${
                        playbackRate === r
                          ? 'bg-[var(--accent)] text-white'
                          : 'text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.06]'}`}>
                {r}×
              </button>
            ))}
          </div>

          <IconBtn title="Loop playback over the selected range" onClick={() => setIsLooping(!isLooping)} active={isLooping}
                   className={`border ${isLooping ? 'border-[var(--accent)]/30' : 'border-[var(--border-color)]'}`}>
            <Ico d={<><polyline points="17 1 21 5 17 9" /><path d="M3 11V9a4 4 0 0 1 4-4h14" /><polyline points="7 23 3 19 7 15" /><path d="M21 13v2a4 4 0 0 1-4 4H3" /></>} />
          </IconBtn>

          {/* Zoom */}
          <div className="spring-cluster flex items-center gap-0.5 rounded-lg border border-[var(--border-color)] bg-[var(--surface-2)] p-0.5"
               title="Zoom the timeline — wheel over the track zooms about the cursor, shift-wheel pans">
            <IconBtn title="Zoom out" onClick={() => zoomAround(frame, 1.6)}>
              <Ico d={<><circle cx="11" cy="11" r="7" /><line x1="20" y1="20" x2="16.2" y2="16.2" /><line x1="8" y1="11" x2="14" y2="11" /></>} />
            </IconBtn>
            <IconBtn title="Zoom in on the playhead" onClick={() => zoomAround(frame, 0.6)}>
              <Ico d={<><circle cx="11" cy="11" r="7" /><line x1="20" y1="20" x2="16.2" y2="16.2" /><line x1="8" y1="11" x2="14" y2="11" /><line x1="11" y1="8" x2="11" y2="14" /></>} />
            </IconBtn>
            <button type="button"
                    onClick={() => (zoomed ? resetView() : setView && setView({
                      start: Math.max(1, startFrame), end: Math.max(startFrame + 1, endFrame) }))}
                    title={zoomed ? 'Zoom out to the whole clip' : 'Zoom to the selected In/Out range'}
                    className="px-2 py-1 rounded-md text-micro font-bold text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/[0.06] transition-colors">
              {zoomed ? 'All' : 'Range'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
