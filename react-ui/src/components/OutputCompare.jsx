import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Button, Card } from './ui';
import { LABELS, fmtDur, fmtVal, diffSettings } from './settingsDiff';

/**
 * Put two finished renders side by side.
 *
 * The comparison suite in the Face Swap tab compares before/after of the CURRENT
 * preview, and Run History diffs two runs' SETTINGS. Neither answers the
 * question you actually have after trying two swappers on the same clip: which
 * of these two files looks better. This does — pixels first, with the settings
 * that produced them underneath, so "B is sharper" comes with "because B used
 * Restoreformer++ and 512px pixel boost".
 *
 * Videos are compared through two real <video> elements driven from one clock:
 * seeking or playing drives A and the effect mirrors it onto B. Sharing a clock
 * matters more than it sounds — comparing two renders frame-for-frame is the
 * entire point, and two independently-playing videos drift apart within seconds.
 */

// Hoisted out of the component on purpose. Declared inside the render body it
// would be a NEW component type on every render, so React would unmount and
// remount the <video> each time — resetting playback the instant anything else
// on the panel changed, which is fatal for the one thing this view is for.
function Media({ file, url, mediaRef, className, style, onMeta }) {
  return file.kind === 'video' ? (
    <video
      ref={mediaRef}
      src={url}
      className={className}
      style={style}
      loop
      playsInline
      preload="metadata"
      onLoadedMetadata={(e) => onMeta?.(e.target.videoWidth, e.target.videoHeight)}
    />
  ) : (
    <img
      ref={mediaRef}
      src={url}
      alt={file.name}
      className={className}
      style={style}
      draggable={false}
      onLoad={(e) => onMeta?.(e.target.naturalWidth, e.target.naturalHeight)}
    />
  );
}

// Also hoisted — same reason as Media above, and it keeps the render body to
// behaviour rather than markup.
function Label({ side, file, entry }) {
  return (
    <div className="min-w-0">
      <div className="text-nano font-semibold uppercase tracking-[0.14em] text-white/45">{side}</div>
      <div className="text-sm font-semibold text-white/85 truncate" title={file.name}>{file.name}</div>
      <div className="text-micro font-mono text-white/45 truncate">
        {entry?.duration_s > 0 && <>{fmtDur(entry.duration_s)} · </>}
        {entry?.fps > 0 && <span className="text-emerald-400/80">{entry.fps.toFixed(1)} fps · </span>}
        {entry?.settings?.swap_model || '—'}
      </div>
    </div>
  );
}

const MODES = [
  { id: 'slider', label: 'Slider', hint: 'Wipe between them' },
  { id: 'side', label: '⬍ Side by side', hint: 'Both in full' },
  { id: 'blend', label: 'Blend', hint: 'Cross-fade' },
  { id: 'diff', label: 'Difference', hint: 'Where they differ at all' },
  { id: 'heatmap', label: '🔥 Heatmap', hint: 'High contrast pixel change heatmap' },
];

export default function OutputCompare({ a, b, aUrl, bUrl, historyA, historyB, onClose, onSwap }) {
  const [mode, setMode] = useState('slider');
  const [pos, setPos] = useState(50);
  const [dragging, setDragging] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [dims, setDims] = useState(null);
  const boxRef = useRef(null);
  const aRef = useRef(null);
  const bRef = useRef(null);

  const isVideo = a.kind === 'video' && b.kind === 'video';
  const mixed = a.kind !== b.kind;
  // Which of the two stage layouts is mounted. They are separate subtrees, so
  // crossing this boundary destroys both <video> elements and builds new ones —
  // which is why the sync effect below has to re-run on it.
  const sideBySide = mode === 'side' || mixed;

  const { changed } = useMemo(
    () => diffSettings(historyA?.settings, historyB?.settings),
    [historyA, historyB],
  );

  // Keep B on A's clock. Re-synced on every play/pause and whenever A seeks —
  // decoders start at slightly different speeds, so a one-off sync at load time
  // is visibly out by the end of a long clip.
  //
  // `sideBySide` is a dependency because switching stage layout REPLACES both
  // video elements: the refs come back pointing at the new nodes while these
  // listeners stay bound to the detached old ones. B then stops following A —
  // and `playing` freezes with it, since that flag is set from A's play/pause
  // events — which loses the shared clock that is this whole view's reason to
  // exist, silently, the first time anyone tries the other mode.
  useEffect(() => {
    if (!isVideo) return undefined;
    const va = aRef.current, vb = bRef.current;
    if (!va || !vb) return undefined;
    // Layouts swap while A is playing; carry that over rather than stranding
    // the new B paused behind a playing A.
    setPlaying(!va.paused);
    if (!va.paused) vb.play().catch(() => {});
    const sync = () => {
      if (Math.abs(vb.currentTime - va.currentTime) > 0.06) vb.currentTime = va.currentTime;
    };
    const onPlay = () => { sync(); vb.play().catch(() => {}); setPlaying(true); };
    const onPause = () => { vb.pause(); sync(); setPlaying(false); };
    va.addEventListener('play', onPlay);
    va.addEventListener('pause', onPause);
    va.addEventListener('seeked', sync);
    va.addEventListener('timeupdate', sync);
    return () => {
      va.removeEventListener('play', onPlay);
      va.removeEventListener('pause', onPause);
      va.removeEventListener('seeked', sync);
      va.removeEventListener('timeupdate', sync);
    };
  }, [isVideo, aUrl, bUrl, sideBySide]);

  // Keys are handled ON THE DIALOG, not on window. A modal's shortcuts belong to
  // the modal: bound to window they would fire for whatever is behind it too,
  // and Escape/Space are already spoken for by the Face Swap tab. Focus is moved
  // into the dialog on open so the handler receives keys immediately (and so a
  // keyboard user is not left navigating the page underneath).
  const dialogRef = useRef(null);
  useEffect(() => {
    const el = dialogRef.current;
    const restore = document.activeElement;
    el?.focus();
    return () => { if (restore instanceof HTMLElement) restore.focus(); };
  }, []);

  const onDialogKeyDown = (e) => {
    if (e.key === 'Escape') { e.stopPropagation(); onClose(); return; }
    if (e.key !== ' ' || !isVideo) return;
    // Space on a focused button is that button's own activation, not ours.
    const tag = e.target?.tagName;
    if (tag === 'BUTTON' || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    e.preventDefault();
    e.stopPropagation();
    const va = aRef.current;
    if (!va) return;
    if (va.paused) va.play().catch(() => {}); else va.pause();
  };

  const move = (clientX) => {
    const box = boxRef.current?.getBoundingClientRect();
    if (!box || !box.width) return;
    setPos(Math.max(0, Math.min(((clientX - box.left) / box.width) * 100, 100)));
  };

  useEffect(() => {
    if (!dragging) return undefined;
    const onMove = (e) => move(e.clientX ?? e.touches?.[0]?.clientX);
    const onUp = () => setDragging(false);
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
  }, [dragging]);

  const overlayStyle =
    mode === 'blend' ? { opacity: pos / 100 }
    : mode === 'diff' ? { mixBlendMode: 'difference' }
    : mode === 'heatmap' ? { mixBlendMode: 'difference', filter: 'contrast(500%) saturate(400%) hue-rotate(180deg)' }
    : { clipPath: `polygon(${pos}% 0, 100% 0, 100% 100%, ${pos}% 100%)` };

  return (
    <div
      ref={dialogRef}
      tabIndex={-1}
      onKeyDown={onDialogKeyDown}
      className="fixed inset-0 z-[80] bg-black/85 backdrop-blur-md flex items-center justify-center p-4 outline-none"
      role="dialog"
      aria-modal="true"
      aria-label={`Comparing ${a.name} with ${b.name}`}
      onClick={onClose}
    >
      <Card
        className="w-full max-w-6xl max-h-[92vh] overflow-y-auto p-5 border border-white/10 shadow-2xl flex flex-col gap-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4">
          <div className="grid grid-cols-2 gap-6 flex-1 min-w-0">
            <Label side="A" file={a} entry={historyA} />
            <Label side="B" file={b} entry={historyB} />
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <Button size="sm" variant="secondary" onClick={onSwap} title="Swap which is A and which is B">
              ⇄ Swap A/B
            </Button>
            <button
              type="button"
              onClick={onClose}
              className="text-white/40 hover:text-white text-xl leading-none px-1"
              aria-label="Close comparison"
              title="Close (Esc)"
            >
              ✕
            </button>
          </div>
        </div>

        {mixed && (
          <p className="text-mini text-amber-300/80 m-0">
            One of these is a video and the other an image — only side-by-side is meaningful here.
          </p>
        )}

        <div className="flex flex-wrap items-center gap-1.5">
          {MODES.map((m) => (
            <button
              key={m.id}
              type="button"
              onClick={() => setMode(m.id)}
              disabled={mixed && m.id !== 'side'}
              title={m.hint}
              aria-pressed={mode === m.id}
              className={`px-2.5 py-1 rounded-lg text-mini font-bold transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
                mode === m.id ? 'bg-[var(--accent)] text-white shadow-md' : 'bg-white/10 text-white/70 hover:text-white'
              }`}
            >
              {m.label}
            </button>
          ))}
          {(mode === 'slider' || mode === 'blend') && (
            <div className="flex items-center gap-2 ml-2">
              <input
                type="range"
                min={0}
                max={100}
                value={pos}
                onChange={(e) => setPos(Number(e.target.value))}
                className="w-36 accent-[var(--accent)] cursor-pointer"
                aria-label={mode === 'slider' ? 'Wipe position' : 'Blend amount'}
              />
              <span className="text-micro font-mono tabular-nums text-white/50 w-9">{Math.round(pos)}%</span>
            </div>
          )}
          {isVideo && (
            <Button
              size="sm"
              variant="secondary"
              className="ml-auto"
              onClick={() => {
                const va = aRef.current;
                if (!va) return;
                if (va.paused) va.play().catch(() => {}); else va.pause();
              }}
            >
              {playing ? '⏸ Pause both' : '▶ Play both'}
            </Button>
          )}
        </div>

        {/* Stage */}
        {sideBySide ? (
          <div className="grid grid-cols-2 gap-2">
            {[[a, aUrl, aRef], [b, bUrl, bRef]].map(([f, url, r], i) => (
              <div key={i} className="relative bg-black/60 rounded-xl overflow-hidden border border-white/10">
                <Media file={f} url={url} mediaRef={r} className="w-full max-h-[58vh] object-contain" />
                <span className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/75 text-micro font-bold text-white/80">
                  {i === 0 ? 'A' : 'B'}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div
            ref={boxRef}
            className="relative bg-black/60 rounded-xl overflow-hidden border border-white/10 select-none"
            style={{ aspectRatio: dims ? `${dims.w}/${dims.h}` : '16/9', maxHeight: '58vh' }}
            onPointerDown={(e) => { setDragging(true); move(e.clientX); }}
          >
            <Media
              file={a} url={aUrl} mediaRef={aRef}
              className="absolute inset-0 w-full h-full object-contain"
              onMeta={(w, h) => setDims({ w, h })}
            />
            <div className="absolute inset-0" style={overlayStyle}>
              <Media file={b} url={bUrl} mediaRef={bRef} className="absolute inset-0 w-full h-full object-contain" />
            </div>
            {mode === 'slider' && (
              <div
                className="absolute inset-y-0 w-0.5 bg-gradient-to-b from-[var(--accent)] via-white to-[var(--accent)] z-10 pointer-events-none"
                style={{ left: `${pos}%` }}
              >
                <span className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-black/80 border border-white/30 grid place-items-center text-micro text-white/70">
                  ⇄
                </span>
              </div>
            )}
            <span className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/75 text-micro font-bold text-white/80 z-10">A</span>
            <span className="absolute bottom-2 right-2 px-2 py-0.5 rounded bg-[var(--accent)]/90 text-micro font-bold text-white z-10">B</span>
          </div>
        )}

        {/* What was different about them */}
        <div>
          <div className="text-nano font-semibold uppercase tracking-[0.14em] text-white/45 mb-1.5">
            {historyA && historyB
              ? `${changed.length} setting${changed.length === 1 ? '' : 's'} differ`
              : 'Settings'}
          </div>
          {!historyA || !historyB ? (
            <p className="text-note text-white/45 m-0">
              No run-history entry for {!historyA ? a.name : b.name} — it was rendered before
              history was recorded, or by another install.
            </p>
          ) : changed.length === 0 ? (
            <p className="text-note text-white/45 m-0">These two were rendered with identical settings.</p>
          ) : (
            <div className="rounded-lg border border-white/[0.07] overflow-hidden divide-y divide-white/[0.05]">
              {changed.map(({ k, va, vb }) => (
                <div key={k} className="grid grid-cols-[1fr_1fr_1fr] gap-2 px-3 py-1.5 text-mini font-mono items-baseline">
                  <span className="text-white/40 truncate" title={k}>{LABELS[k] || k}</span>
                  <span className="text-amber-300/80 truncate text-right">{fmtVal(va)}</span>
                  <span className="text-emerald-300/80 truncate text-right">{fmtVal(vb)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
