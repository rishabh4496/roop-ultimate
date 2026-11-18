import React, { useEffect, useRef, useState } from 'react';
import { pillsFor, quadLayout, clampRatio } from './mediaMeta';
import { fmtClock } from '../hoverScrub';

// Compare 2 to 4 outputs side by side.
//
// One transport drives every tile: play, pause, and a shared scrub bar seek them
// together, and while playing each tile is pulled back into step whenever it
// drifts more than a fifth of a second from the first. Tiles are muted, because
// four audio tracks at once is not a comparison anyone can hear. Each tile carries
// its own pills, and its resolution is the one the file reports on load.
//
// `items`: [{ name, url, kind: 'video' | 'image', entry }] where `entry` is the
// run-history record for that output (or null when the run was not recorded).

const DRIFT_S = 0.2;

export default function QuadLightbox({ items, onClose, onRemove }) {
  const layout = quadLayout(items.length);
  const vids = useRef([]);
  const [dims, setDims] = useState({});
  const [playing, setPlaying] = useState(false);
  const [ratio, setRatio] = useState(0);
  const [clock, setClock] = useState({ t: 0, d: 0 });

  const list = items.slice(0, layout.count);
  const videos = () => vids.current.filter(Boolean);
  const lead = () => videos().find((v) => Number.isFinite(v.duration) && v.duration > 0) || null;

  const seek = (r) => {
    const x = clampRatio(r);
    setRatio(x);
    for (const v of videos()) {
      if (Number.isFinite(v.duration) && v.duration > 0) v.currentTime = x * v.duration;
    }
  };

  const togglePlay = () => {
    const vs = videos();
    if (!vs.length) return;
    if (playing) {
      vs.forEach((v) => v.pause());
      setPlaying(false);
    } else {
      // Start from the leader's position so the tiles begin in step.
      const head = lead();
      if (head) vs.forEach((v) => { if (v !== head && Number.isFinite(v.duration)) v.currentTime = head.currentTime; });
      Promise.allSettled(vs.map((v) => v.play())).then(() => setPlaying(true));
    }
  };

  // Keep the tiles in step while playing, without fighting a user who is scrubbing.
  useEffect(() => {
    if (!playing) return undefined;
    const id = setInterval(() => {
      const head = lead();
      if (!head) return;
      for (const v of videos()) {
        if (v !== head && Math.abs(v.currentTime - head.currentTime) > DRIFT_S) v.currentTime = head.currentTime;
      }
    }, 500);
    return () => clearInterval(id);
  }, [playing]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') { e.preventDefault(); onClose?.(); }
      if (e.key === ' ' && e.target === document.body) { e.preventDefault(); togglePlay(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }); // re-bind each render so togglePlay sees current state

  const onTime = (i) => (e) => {
    if (i !== 0) return;
    const v = e.currentTarget;
    setClock({ t: v.currentTime, d: v.duration || 0 });
    if (Number.isFinite(v.duration) && v.duration > 0) setRatio(v.currentTime / v.duration);
  };

  const pills = list.map((it, i) => pillsFor(it.entry, dims[i] || null));

  return (
    <div role="dialog" aria-modal="true" aria-label={`Compare ${list.length} renders`}
         data-testid="quad-lightbox"
         className="fixed inset-0 z-[120] flex flex-col bg-black/92 backdrop-blur-sm p-3 sm:p-5 gap-3">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="text-mini uppercase tracking-[0.18em] text-accent font-semibold">Compare {list.length} renders</div>
          <div className="text-micro text-muted">Same clock for every tile. Tiles are muted.</div>
        </div>
        <div className="flex items-center gap-2">
          {list.some((it) => it.kind === 'video') && (
            <button type="button" onClick={togglePlay} data-testid="quad-play"
              className="px-3 py-1.5 rounded-lg bg-[var(--accent)] text-white text-xs font-bold">
              {playing ? '❚❚ Pause all' : '▶ Play all'}
            </button>
          )}
          <button type="button" onClick={onClose} aria-label="Close comparison"
            className="px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white text-xs font-bold">Close (Esc)</button>
        </div>
      </header>

      {list.some((it) => it.kind === 'video') && (
        <div className="flex items-center gap-3 font-mono text-micro text-white/70">
          <span className="tabular-nums w-24">{fmtClock(clock.t)}</span>
          <input type="range" min={0} max={1} step={0.001} value={ratio}
                 onChange={(e) => seek(Number(e.target.value))}
                 aria-label="Shared playhead"
                 className="flex-1 accent-[var(--accent)]" />
          <span className="tabular-nums w-24 text-right">{fmtClock(clock.d)}</span>
        </div>
      )}

      <div className={`grid gap-3 min-h-0 flex-1 grid-cols-1 ${layout.cols === 2 ? 'sm:grid-cols-2' : ''}`}
           style={{ gridTemplateRows: `repeat(${layout.rows}, minmax(0, 1fr))` }}>
        {list.map((it, i) => (
          <figure key={it.name} className="m-0 relative min-h-0 rounded-xl overflow-hidden border border-white/10 bg-black flex flex-col" data-tile={i}>
            <div className="relative flex-1 min-h-0 bg-black">
              {it.kind === 'video' ? (
                <video ref={(el) => { vids.current[i] = el; }} src={it.url} muted playsInline preload="metadata"
                       className="absolute inset-0 w-full h-full object-contain"
                       onLoadedMetadata={(e) => {
                         const v = e.currentTarget;
                         setDims((d) => ({ ...d, [i]: { width: v.videoWidth, height: v.videoHeight } }));
                       }}
                       onTimeUpdate={onTime(i)} />
              ) : (
                <img src={it.url} alt={it.name} className="absolute inset-0 w-full h-full object-contain"
                     onLoad={(e) => {
                       const im = e.currentTarget;
                       setDims((d) => ({ ...d, [i]: { width: im.naturalWidth, height: im.naturalHeight } }));
                     }} />
              )}
            </div>
            <figcaption className="flex flex-wrap items-center gap-1.5 px-3 py-2 bg-black/70 border-t border-white/10">
              <span className="text-xs font-bold text-white truncate min-w-0 flex-1" title={it.name}>{it.name}</span>
              {pills[i].map((p) => (
                <span key={p.key} title={p.title}
                      className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-nano font-mono text-white/80 whitespace-nowrap"
                      data-pill={p.key}>
                  <span className="text-white/65 mr-1">{p.label}</span>{p.value}
                </span>
              ))}
              {onRemove && (
                <button type="button" onClick={() => onRemove(it.name)} aria-label={`Remove ${it.name} from the comparison`}
                        className="text-nano text-white/60 hover:text-white px-1">✕</button>
              )}
            </figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}
