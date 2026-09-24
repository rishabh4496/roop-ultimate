import React, { useCallback, useEffect, useRef, useState } from 'react';
import { GlRenderer } from './glRenderer';

// ── <VideoCompareStage> — original vs. result, one WebGL canvas ───────────
//
// Two <video> elements, the OUTPUT (master: its clock, its audio, its
// controls' state) and the ORIGINAL target (follower, muted), drawn as two
// textures into one canvas by GlRenderer:
//
//   'wipe' — an interactive split slider: original left of the divider, the
//            swapped result right of it, the same pixels of both pictures
//   'side' — both in full, side by side
//
// Why WebGL rather than two <video>s with a clip-path: a clip-path wipe makes
// the compositor re-rasterise a stage-sized layer on every pointer move, and
// two separately-composited videos drift apart by a frame or more on their own
// presentation clocks. Here both pictures are sampled in the SAME draw, so the
// wipe is a uniform and the two sides can never show different moments of
// the draw loop.
//
// SYNC. The follower is steered to `master.currentTime + offsetS`, where
// `offsetS` is where the render's first frame sits in the original (a trimmed
// render starts at start_frame / fps — the same offset its audio was cut at).
// Small drift is closed by nudging the follower's playbackRate (a seek stalls
// the decoder and shows as a hitch); drift past HARD_SYNC_S is a seek.
//
// The videos stay in the layout at 1x1 px, transparent — not display:none,
// which lets Chromium stop presenting (and so stop decoding) video frames.

const SOFT_SYNC_S = 0.03;
const HARD_SYNC_S = 0.25;
const MAX_EDGE = 2560;

const fmt = (s) => {
  if (!Number.isFinite(s) || s < 0) return '0:00';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, '0')}`;
};

export default function VideoCompareStage({
  outputUrl,
  sourceUrl,
  offsetS = 0,
  mode = 'wipe',           // 'wipe' | 'side'
  labelA = 'Original',
  labelB = 'Swapped',
  onError,
  className = '',
}) {
  const hostRef = useRef(null);
  const canvasRef = useRef(null);
  const masterRef = useRef(null);
  const followerRef = useRef(null);
  const rendererRef = useRef(null);
  const dividerRef = useRef(null);
  const handleRef = useRef(null);
  const seekRef = useRef(null);
  const timeRef = useRef(null);
  const splitRef = useRef(0.5);
  const drawRef = useRef(() => {});
  const [playing, setPlaying] = useState(false);
  const [dims, setDims] = useState(null);
  const [glKind, setGlKind] = useState('');

  // ── renderer lifetime ──────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    const r = new GlRenderer(canvas);
    rendererRef.current = r;
    setGlKind(r.kind);
    const fit = () => {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const s = Math.min(1, MAX_EDGE / Math.max(rect.width * dpr, rect.height * dpr));
      if (r.resize(rect.width * dpr * s, rect.height * dpr * s)) drawRef.current(true);
    };
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(fit) : null;
    ro?.observe(canvas);
    fit();
    return () => {
      ro?.disconnect();
      r.destroy();
      rendererRef.current = null;
    };
  }, []);

  useEffect(() => {
    const r = rendererRef.current;
    if (!r) return;
    r.setMode(mode === 'side' ? 'side' : 'wipe');
    r.setSplit(splitRef.current);
    drawRef.current(true);
  }, [mode, dims]);

  // ── draw: upload both current frames, one draw call ────────────────────
  const lastTimeRef = useRef(-1);
  const draw = useCallback((force = false) => {
    const r = rendererRef.current;
    const m = masterRef.current;
    const f = followerRef.current;
    if (!r || !m) return;
    // HAVE_CURRENT_DATA (2): there is a decoded frame to sample.
    const t = m.currentTime;
    if (!force && t === lastTimeRef.current) return;
    lastTimeRef.current = t;
    if (f && f.readyState >= 2) r.setLayer(0, f);
    if (m.readyState >= 2) r.setLayer(1, m);
    r.draw();
  }, []);
  drawRef.current = draw;

  // ── sync + render loop, only while playing ─────────────────────────────
  useEffect(() => {
    if (!playing) return undefined;
    let raf = 0;
    const loop = () => {
      const m = masterRef.current;
      const f = followerRef.current;
      if (m && f && f.readyState >= 1) {
        const want = m.currentTime + offsetS;
        const drift = f.currentTime - want;
        if (Math.abs(drift) > HARD_SYNC_S) {
          f.currentTime = want;
          f.playbackRate = m.playbackRate;
        } else if (Math.abs(drift) > SOFT_SYNC_S) {
          // Ahead -> slow down, behind -> speed up; converges without a seek.
          f.playbackRate = m.playbackRate * (drift > 0 ? 0.95 : 1.05);
        } else if (f.playbackRate !== m.playbackRate) {
          f.playbackRate = m.playbackRate;
        }
      }
      draw();
      // The seek bar and clock, written straight to the DOM (no re-render).
      if (m && seekRef.current && m.duration) {
        seekRef.current.value = String((m.currentTime / m.duration) * 1000);
      }
      if (m && timeRef.current) timeRef.current.textContent = `${fmt(m.currentTime)} / ${fmt(m.duration)}`;
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing, offsetS, draw]);

  // ── master events: mirror onto the follower ────────────────────────────
  useEffect(() => {
    const m = masterRef.current;
    const f = followerRef.current;
    if (!m || !f) return undefined;
    const align = () => {
      const want = m.currentTime + offsetS;
      if (Math.abs(f.currentTime - want) > 1 / 120) f.currentTime = want;
    };
    const onPlay = () => { align(); f.play().catch(() => {}); setPlaying(true); };
    const onPause = () => { f.pause(); align(); setPlaying(false); draw(true); };
    const onSeek = () => { align(); };
    const onSeeked = () => { draw(true); if (timeRef.current) timeRef.current.textContent = `${fmt(m.currentTime)} / ${fmt(m.duration)}`; };
    const onRate = () => { f.playbackRate = m.playbackRate; };
    const onMeta = () => {
      if (m.videoWidth && m.videoHeight) setDims({ w: m.videoWidth, h: m.videoHeight });
      align();
      if (timeRef.current) timeRef.current.textContent = `${fmt(m.currentTime)} / ${fmt(m.duration)}`;
    };
    const onData = () => draw(true);
    const onEnded = () => { f.pause(); setPlaying(false); };
    m.addEventListener('play', onPlay);
    m.addEventListener('pause', onPause);
    m.addEventListener('seeking', onSeek);
    m.addEventListener('seeked', onSeeked);
    m.addEventListener('ratechange', onRate);
    m.addEventListener('loadedmetadata', onMeta);
    m.addEventListener('loadeddata', onData);
    m.addEventListener('ended', onEnded);
    f.addEventListener('seeked', onData);
    f.addEventListener('loadeddata', onData);
    if (m.readyState >= 1) onMeta();
    return () => {
      m.removeEventListener('play', onPlay);
      m.removeEventListener('pause', onPause);
      m.removeEventListener('seeking', onSeek);
      m.removeEventListener('seeked', onSeeked);
      m.removeEventListener('ratechange', onRate);
      m.removeEventListener('loadedmetadata', onMeta);
      m.removeEventListener('loadeddata', onData);
      m.removeEventListener('ended', onEnded);
      f.removeEventListener('seeked', onData);
      f.removeEventListener('loadeddata', onData);
    };
  }, [offsetS, draw, outputUrl, sourceUrl]);

  // Release both decoders and abort their range requests on unmount. A
  // detached <video> otherwise holds its connection and its decoded buffers
  // until garbage collection gets round to it. Unmount ONLY: React writes a
  // new `src` before it runs the previous effect's cleanup, so a cleanup keyed
  // on the URLs would strip the NEW source. The parent keys this component by
  // its URLs, so a new file is a new mount.
  useEffect(() => {
    const m = masterRef.current;
    const f = followerRef.current;
    return () => {
      for (const v of [m, f]) {
        if (!v) continue;
        try { v.pause(); v.removeAttribute('src'); v.load(); } catch { /* ignore */ }
      }
    };
  }, []);

  // ── the wipe handle: pointer + keyboard, no React state per move ───────
  const applySplit = useCallback((frac) => {
    const v = Math.max(0, Math.min(1, frac));
    splitRef.current = v;
    if (dividerRef.current) dividerRef.current.style.left = `${v * 100}%`;
    if (handleRef.current) {
      handleRef.current.setAttribute('aria-valuenow', String(Math.round(v * 100)));
      handleRef.current.setAttribute('aria-valuetext', `${Math.round(v * 100)}% ${labelA}`);
    }
    const r = rendererRef.current;
    if (r) { r.setSplit(v); drawRef.current(true); }
  }, [labelA]);

  const onPointerDown = (e) => {
    if (mode !== 'wipe') return;
    const host = hostRef.current;
    if (!host) return;
    e.preventDefault();
    host.setPointerCapture?.(e.pointerId);
    const move = (ev) => {
      const rect = host.getBoundingClientRect();
      applySplit((ev.clientX - rect.left) / rect.width);
    };
    move(e);
    const up = () => {
      host.removeEventListener('pointermove', move);
      host.removeEventListener('pointerup', up);
      host.removeEventListener('pointercancel', up);
    };
    host.addEventListener('pointermove', move);
    host.addEventListener('pointerup', up);
    host.addEventListener('pointercancel', up);
  };

  const onHandleKey = (e) => {
    const step = e.shiftKey ? 0.1 : 0.02;
    if (e.key === 'ArrowLeft') { e.preventDefault(); applySplit(splitRef.current - step); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); applySplit(splitRef.current + step); }
    else if (e.key === 'Home') { e.preventDefault(); applySplit(0); }
    else if (e.key === 'End') { e.preventDefault(); applySplit(1); }
  };

  const togglePlay = () => {
    const m = masterRef.current;
    if (!m) return;
    if (m.paused) m.play().catch((err) => onError?.(err));
    else m.pause();
  };

  const aspect = dims ? (mode === 'side' ? `${dims.w * 2} / ${dims.h}` : `${dims.w} / ${dims.h}`) : '16 / 9';

  return (
    <div className={`space-y-2 ${className}`}>
      <div
        ref={hostRef}
        className={`relative w-full max-h-[52vh] mx-auto overflow-hidden rounded-xl border border-white/5 bg-black select-none ${mode === 'wipe' ? 'cursor-ew-resize' : ''}`}
        style={{ aspectRatio: aspect, touchAction: 'none' }}
        onPointerDown={onPointerDown}
        onDoubleClick={togglePlay}
      >
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" role="img"
          aria-label={`${labelA} and ${labelB} compared (${mode === 'side' ? 'side by side' : 'split'})`} />
        {mode === 'wipe' && (
          <div ref={dividerRef} className="absolute inset-y-0 w-0 pointer-events-none" style={{ left: `${splitRef.current * 100}%` }}>
            <div className="absolute inset-y-0 -translate-x-1/2 w-0.5 bg-white/85 shadow-[0_0_8px_rgba(0,0,0,0.6)]" />
            <div
              ref={handleRef}
              role="slider"
              tabIndex={0}
              aria-label="Comparison split"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(splitRef.current * 100)}
              onKeyDown={onHandleKey}
              className="pointer-events-auto absolute top-1/2 -translate-x-1/2 -translate-y-1/2 h-9 w-9 rounded-full bg-white/90 text-black grid place-items-center shadow-lg focus:outline focus:outline-2 focus:outline-[var(--accent)]"
            >
              <span aria-hidden="true" className="text-xs font-bold">⇆</span>
            </div>
          </div>
        )}
        <span className="absolute top-2 left-2 px-2 py-0.5 rounded-md bg-black/70 text-micro font-bold uppercase tracking-wider text-white/85 pointer-events-none">{labelA}</span>
        <span className="absolute top-2 right-2 px-2 py-0.5 rounded-md bg-[var(--accent)]/90 text-micro font-bold uppercase tracking-wider text-white pointer-events-none">{labelB}</span>

        {/* The two decoders. Laid out (1x1, transparent), not display:none. */}
        <video ref={followerRef} src={sourceUrl} muted playsInline preload="auto" crossOrigin="anonymous"
          aria-hidden="true" tabIndex={-1}
          className="absolute left-0 top-0 w-px h-px opacity-0 pointer-events-none"
          onError={(e) => onError?.(e.currentTarget.error)} />
        <video ref={masterRef} src={outputUrl} playsInline preload="auto" crossOrigin="anonymous"
          aria-hidden="true" tabIndex={-1}
          className="absolute left-0 top-0 w-px h-px opacity-0 pointer-events-none"
          onError={(e) => onError?.(e.currentTarget.error)} />
      </div>

      <div className="flex items-center gap-3 text-xs">
        <button type="button" onClick={togglePlay}
          className="px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/15 border border-white/10 font-bold text-white"
          aria-label={playing ? 'Pause' : 'Play'}>
          {playing ? '❚❚' : '▶'}
        </button>
        <input
          ref={seekRef}
          type="range" min={0} max={1000} step={1} defaultValue={0}
          aria-label="Seek"
          className="flex-1 accent-[var(--accent)]"
          onInput={(e) => {
            const m = masterRef.current;
            if (m && m.duration) m.currentTime = (Number(e.currentTarget.value) / 1000) * m.duration;
          }}
        />
        <span ref={timeRef} className="font-mono tabular-nums text-white/60 whitespace-nowrap">0:00 / 0:00</span>
        {glKind === '2d' && <span className="text-amber-300/80" title="WebGL is unavailable in this webview">2D</span>}
      </div>
    </div>
  );
}
