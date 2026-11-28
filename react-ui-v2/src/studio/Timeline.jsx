import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { ChevronsLeft, ChevronLeft, ChevronRight, ChevronsRight, Play, Pause, Flag, FlagOff, Scissors, RotateCcw, FileDown } from 'lucide-react';
import { getJSON, postJSON } from '@engine/api';
import { cutListDocument, downloadText } from '../engine/interchange';
import { notify } from '../ui/controls';
import { useStudio, currentTarget, isVideoTarget } from '../engine/studio';
import { timecode, frameUrl, hueOf } from './people';
import FrameImg from '../ui/FrameImg';
import { DIRECT_PARAMS } from '@engine/components/faceswap/curves/curveModel.js';
import { describeOverride } from '@engine/components/faceswap/timeline/overrides.js';
import { clipFxOf, clipKeyOf, inOutOf } from '../engine/payload';

// ── The timeline ────────────────────────────────────────────────────────────
// A filmstrip of real frames across the clip; a playhead you drag (raw frames while dragging, the
// swap once you let go); in/out handles that bound what renders (saved to the engine with
// /api/target/set_frame); scene cuts and per-person presence lanes from /api/target/timeline_tracks;
// a ruler in timecode. Keyboard: ←/→ one frame, Shift ±10, Home/End, I / O set in / out, Space play.

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

function niceStep(frames, fps, px) {
  const targetTicks = Math.max(4, Math.floor(px / 90));
  const raw = frames / targetTicks;
  const steps = [1, 2, 5, 10, 15, 30, 60, 120, 150, 300, 600, 900, 1800, 3600, 9000, 18000].map((k) => (k <= 2 ? k : Math.round(k * (fps || 30) / 30)));
  return steps.find((s) => s >= raw) || Math.ceil(raw);
}

export default function Timeline({ onScrub }) {
  const target = useStudio(currentTarget);
  const frame = useStudio((s) => s.frame);
  const sel = useStudio((s) => s.selectedTarget);
  const mediaId = useStudio((s) => s.targetMediaId);
  const setFrame = useStudio((s) => s.setFrame);
  const setRange = useStudio((s) => s.setRange);
  // Primitives / stable references only: clipFxOf builds a fresh object, so it is derived, not selected.
  const clipFx = useStudio((st) => st.clipFx);
  const clipKey = useStudio(clipKeyOf);
  const fx = useMemo(() => clipFxOf({ clipFx, targetMediaId: clipKey, selectedTarget: 0 }), [clipFx, clipKey]);
  const total = Math.max(1, target?.frames || 1);
  const fps = target?.fps || 30;
  const video = isVideoTarget(target);

  const strip = useRef(null);
  const [w, setW] = useState(800);
  useLayoutEffect(() => {
    const el = strip.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, [video]);

  const xOf = (f) => ((f - 1) / Math.max(1, total - 1)) * w;
  const fOf = useCallback((clientX) => {
    const r = strip.current.getBoundingClientRect();
    return clamp(Math.round(1 + ((clientX - r.left) / r.width) * (total - 1)), 1, total);
  }, [total]);

  // In / out in UI frames (1-based, inclusive), read through inOutOf.
  const { a: engineIn, b: engineOut } = inOutOf(target);
  const [range, setRangeLocal] = useState({ a: engineIn, b: engineOut });
  useEffect(() => { setRangeLocal({ a: engineIn, b: engineOut }); }, [engineIn, engineOut, total, sel]);

  // Tracks (scene cuts + presence lanes)
  const [tracks, setTracks] = useState(null);
  useEffect(() => {
    if (!video || !mediaId) { setTracks(null); return undefined; }
    let alive = true;
    getJSON(`/api/target/timeline_tracks?media_id=${encodeURIComponent(mediaId)}`, { timeout: 8000 })
      .then((t) => { if (alive) setTracks(t); }).catch(() => { if (alive) setTracks(null); });
    return () => { alive = false; };
  }, [video, mediaId, sel]);

  // Filmstrip frames
  const thumbs = useMemo(() => {
    const n = clamp(Math.floor(w / 92), 3, 24);
    return Array.from({ length: n }, (_, k) => Math.round(1 + (k * (total - 1)) / (n - 1)));
  }, [w, total]);

  // Scrubbing
  const [scrubbing, setScrubbing] = useState(false);
  const [hover, setHover] = useState(null);
  const startScrub = (e) => {
    if (e.button !== 0 || e.target.closest('[data-handle]')) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    setScrubbing(true); onScrub?.(true);
    setFrame(fOf(e.clientX));
  };
  const moveScrub = (e) => {
    const f = fOf(e.clientX);
    setHover({ f, x: xOf(f) });
    if (scrubbing) setFrame(f);
  };
  const endScrub = () => { if (scrubbing) { setScrubbing(false); onScrub?.(false); } };

  const dragHandle = (which) => (e) => {
    e.preventDefault(); e.stopPropagation();
    const move = (ev) => {
      const f = fOf(ev.clientX);
      setRangeLocal((r) => (which === 'start' ? { ...r, a: Math.min(f, r.b) } : { ...r, b: Math.max(f, r.a) }));
    };
    const up = (ev) => {
      window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up);
      const f = fOf(ev.clientX);
      setRange(which, which === 'start' ? Math.min(f, range.b) : Math.max(f, range.a));
    };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
  };

  // Playback over raw frames (cheap GETs), swap preview when it stops.
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    if (!playing) return undefined;
    onScrub?.(true);
    const stride = Math.max(1, Math.round(fps / 8));
    const h = setInterval(() => {
      const f = useStudio.getState().frame;
      const end = range.b || total;
      if (f >= end) { setPlaying(false); return; }
      setFrame(Math.min(end, f + stride));
    }, 125);
    return () => { clearInterval(h); onScrub?.(false); };
  }, [playing]); // eslint-disable-line react-hooks/exhaustive-deps

  // Keys (ignored while typing)
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.closest('input, textarea, select, [contenteditable]')) return;
      const f = useStudio.getState().frame;
      const step = e.shiftKey ? 10 : 1;
      if (e.key === 'ArrowRight') { setFrame(f + step); e.preventDefault(); }
      else if (e.key === 'ArrowLeft') { setFrame(f - step); e.preventDefault(); }
      else if (e.key === 'Home') setFrame(range.a || 1);
      else if (e.key === 'End') setFrame(range.b || total);
      else if (e.key === ' ' && video) { setPlaying((p) => !p); e.preventDefault(); }
      else if (e.key.toLowerCase() === 'i' && video) setRange('start', f);
      else if (e.key.toLowerCase() === 'o' && video) setRange('end', f);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [range, total, video, setFrame, setRange]);

  if (!target) return null;

  const step = niceStep(total, fps, w);
  const ticks = [];
  for (let f = 1; f <= total; f += Math.max(1, Math.round(step / 5))) ticks.push(f);
  const presence = Array.isArray(tracks?.presence) ? tracks.presence : [];
  const cuts = Array.isArray(tracks?.scene_cuts) ? tracks.scene_cuts : [];

  return (
    <div className="tl" data-testid="timeline">
      {/* Transport */}
      <div className="mb-2 flex flex-wrap items-center gap-1.5">
        <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="To in point" onClick={() => setFrame(range.a || 1)}><ChevronsLeft size={16} /></button>
        <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="Previous frame" onClick={() => setFrame(frame - 1)}><ChevronLeft size={16} /></button>
        <button type="button" className={`btn btn-sm btn-icon ${playing ? 'btn-primary' : ''}`} aria-label={playing ? 'Pause playback' : 'Play'} disabled={!video} onClick={() => setPlaying((p) => !p)}>
          {playing ? <Pause size={15} /> : <Play size={15} />}
        </button>
        <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="Next frame" onClick={() => setFrame(frame + 1)}><ChevronRight size={16} /></button>
        <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="To out point" onClick={() => setFrame(range.b || total)}><ChevronsRight size={16} /></button>
        <div className="ml-1 flex items-center gap-1.5 rounded-lg border border-[var(--rule-strong)] bg-black/50 px-2 py-1 font-mono text-[12px]">
          <span className="tnum text-white">{timecode(frame, fps)}</span>
          <span className="text-[var(--fg-faint)]">/</span>
          <span className="tnum text-[var(--fg-dim)]">{timecode(total, fps)}</span>
          <span className="mx-1 h-3 w-px bg-white/15" />
          <label className="sr-only" htmlFor="tl-frame">Frame</label>
          <input id="tl-frame" type="number" min={1} max={total} value={frame} onChange={(e) => setFrame(Number(e.target.value))}
            className="w-16 bg-transparent text-right text-[var(--accent-2)] outline-none tnum" />
          <span className="text-[var(--fg-faint)]">f</span>
        </div>
        {video && (
          <div className="ml-auto flex flex-wrap items-center gap-1.5">
            <span className="chip"><Scissors size={12} /> in <b className="tnum">{range.a}</b> · out <b className="tnum">{range.b}</b> · {Math.max(0, range.b - range.a + 1)} f</span>
            <button type="button" className="btn btn-sm" onClick={() => setRange('start', frame)} title="Set in at playhead (I)"><Flag size={13} /> In</button>
            <button type="button" className="btn btn-sm" onClick={() => setRange('end', frame)} title="Set out at playhead (O)"><FlagOff size={13} /> Out</button>
            <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="Reset range to the whole clip"
              onClick={async () => { await setRange('start', 1); await setRange('end', total); }}><RotateCcw size={14} /></button>
          </div>
        )}
      </div>

      {/* Ruler */}
      <div className="tl-ruler" aria-hidden="true">
        {ticks.map((f) => {
          const major = (f - 1) % step === 0;
          return (
            <span key={f} className={`tl-tick ${major ? 'major' : ''}`} style={{ left: xOf(f), height: major ? 10 : 5 }}>
              {major && <span>{timecode(f, fps)}</span>}
            </span>
          );
        })}
      </div>

      {/* Strip + playhead + range */}
      <div ref={strip} className="tl-strip flex" role="slider" tabIndex={0} aria-label="Timeline" aria-valuemin={1} aria-valuemax={total} aria-valuenow={frame}
        onPointerDown={startScrub} onPointerMove={moveScrub} onPointerUp={endScrub} onPointerLeave={() => setHover(null)} data-testid="timeline-strip">
        {thumbs.map((f) => <FrameImg key={`${sel}-${f}`} src={frameUrl(sel, f, 160)} loading="lazy" draggable={false} className="h-full min-w-0 flex-1 object-cover" />)}
        {video && (
          <>
            <div className="tl-out" style={{ left: 0, width: xOf(range.a) }} />
            <div className="tl-out" style={{ left: xOf(range.b), right: 0 }} />
            <div className="tl-range" style={{ left: xOf(range.a), width: Math.max(2, xOf(range.b) - xOf(range.a)) }} />
            <div data-handle className="tl-handle" style={{ left: xOf(range.a) - 6 }} onPointerDown={dragHandle('start')} role="slider" aria-label="In point" aria-valuenow={range.a} tabIndex={0} />
            <div data-handle className="tl-handle" style={{ left: xOf(range.b) - 6 }} onPointerDown={dragHandle('end')} role="slider" aria-label="Out point" aria-valuenow={range.b} tabIndex={0} />
          </>
        )}
        {cuts.map((c) => <span key={`c${c}`} className="tl-cut" style={{ left: xOf(c) }} title={`Scene cut · frame ${c}`} />)}
        <div className="tl-playhead" style={{ left: xOf(frame) - 1 }} />
        {hover && !scrubbing && video && (
          <div className="tl-hover" style={{ left: clamp(hover.x, 70, w - 70) }}>
            <img src={frameUrl(sel, hover.f, 240)} alt="" className="block h-[78px] w-[138px] object-cover" />
            <div className="px-2 py-0.5 text-center font-mono text-[10.5px] text-white">{timecode(hover.f, fps)} · f{hover.f}</div>
          </div>
        )}
      </div>

      {/* Tool lanes: scene overrides, curve keys, pins (what will render differently, and where). */}
      {video && (fx.overrides.length > 0 || DIRECT_PARAMS.some((p) => (fx.curves[p.id] || []).length) || fx.pins.length > 0) && (
        <div className="mt-2 space-y-1" aria-label="Tool lanes" data-testid="tool-lanes">
          {fx.overrides.length > 0 && (
            <div className="flex items-center gap-2">
              <span className="w-24 shrink-0 text-[11px] font-medium text-[var(--accent-2)]">Scenes</span>
              <div className="tl-lane flex-1">
                {fx.overrides.map((o) => (
                  <i key={o.id} title={`${o.label}: ${describeOverride(o)}`} style={{ left: `${((o.start - 1) / Math.max(1, total - 1)) * 100}%`, width: `${Math.max(0.5, ((o.end - o.start + 1) / total) * 100)}%`, background: o.bypass ? 'var(--fg-faint)' : 'var(--accent-2)' }} />
                ))}
              </div>
            </div>
          )}
          {DIRECT_PARAMS.filter((p) => (fx.curves[p.id] || []).length).map((p) => (
            <div key={p.id} className="flex items-center gap-2">
              <span className="w-24 shrink-0 truncate text-[11px] font-medium" style={{ color: p.color }}>{p.label}</span>
              <div className="tl-lane flex-1">
                {fx.curves[p.id].map((k) => <i key={k.frame} title={`f${k.frame}: ${k.value}`} style={{ left: `calc(${((k.frame - 1) / Math.max(1, total - 1)) * 100}% - 3px)`, width: 6, background: p.color, borderRadius: 2 }} />)}
              </div>
            </div>
          ))}
          {fx.pins.length > 0 && (
            <div className="flex items-center gap-2">
              <span className="w-24 shrink-0 text-[11px] font-medium text-[var(--warn)]">Pins</span>
              <div className="tl-lane flex-1">
                {fx.pins.map((pn) => <i key={pn.id} title={`pin at f${pn.frame}`} style={{ left: `calc(${((pn.frame - 1) / Math.max(1, total - 1)) * 100}% - 3px)`, width: 6, background: 'var(--warn)' }} />)}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Presence lanes */}
      {presence.length > 0 && (
        <div className="mt-2 space-y-1" aria-label="Who appears when">
          {presence.slice(0, 6).map((p, i) => (
            <div key={p.person_id || i} className="flex items-center gap-2">
              <span className="w-24 shrink-0 truncate text-[11px] font-medium" style={{ color: hueOf(i) }} title={p.name}>{p.name}</span>
              <div className="tl-lane flex-1" style={{ width: w }}>
                {(p.intervals || []).map(([a, b], k) => (
                  <i key={k} style={{ left: `${((a - 1) / Math.max(1, total - 1)) * 100}%`, width: `${Math.max(0.4, ((b - a + 1) / total) * 100)}%`, background: hueOf(i), boxShadow: `0 0 8px ${hueOf(i)}` }} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
      {video && !presence.length && (
        <div className="mt-1.5 text-[11.5px] text-[var(--fg-faint)]">Presence lanes appear once the engine has scanned this clip for people.</div>
      )}
      {video && (
        <div className="mt-2 flex flex-wrap items-center gap-1.5" data-testid="cutlist-export">
          <span className="text-[11px] text-[var(--fg-faint)]">Cut list for your editor{presence.length ? ` (${presence.length} ${presence.length === 1 ? 'person' : 'people'}, In to Out)` : ' (source only: no presence lanes yet)'}:</span>
          {[['edl', 'EDL', '/api/timeline/export_edl', 'edl', 'text/plain'], ['otio', 'OTIO', '/api/timeline/export_otio', 'otio', 'application/json']].map(([k, label, route, field, type]) => (
            <button key={k} type="button" className="btn btn-ghost btn-sm !h-7" data-testid={`export-${k}`}
              title={k === 'edl' ? "CMX3600 EDL: the source on V1, each person's swapped stretches on V2" : 'OpenTimelineIO JSON: the source on V1, swapped stretches on V2'}
              onClick={async () => {
                const { document: doc, fps } = cutListDocument(target, presence, { a: engineIn, b: engineOut });
                try {
                  const r = await postJSON(route, { document: doc, fps });
                  const text = r?.[field];
                  if (!text) { notify(`The engine returned no ${label}`, 'error'); return; }
                  downloadText(text, r.filename || `${doc.name}.${k}`, type);
                  notify(`Exported ${label}: ${doc.timeline.face_segments.length} swapped stretch${doc.timeline.face_segments.length === 1 ? '' : 'es'}`);
                } catch (e) { notify(e.message, 'error'); }
              }}>
              <FileDown size={12} /> {label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
