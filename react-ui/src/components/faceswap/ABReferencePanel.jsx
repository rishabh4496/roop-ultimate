import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '../ui';
import { clearReference, loadReference, referenceCovers, referenceTime, saveReference, subscribeReference } from './abReference';

// A/B reference: a previous render (sent from Outputs) wiped against the LIVE preview, frame for frame.
//
//   A (left of the divider)   the reference - an image, or a video held on the frame matching the playhead
//   B (right of the divider)  what the Face Swap preview shows right now, with the current recipe
//
// Drag the divider (or use the slider / arrow keys). A reference video is looked up at
// `(frame - 1 - offset) / fps`: a render of a trimmed range starts `offset` frames into the clip, which a
// file cannot say about itself, so the offset is a field and "Align to In" sets it from the In point.
// A frame the reference does not cover shows its nearest end and says so, instead of looking like a match.

export default function ABReferencePanel({ previewSrc, frame = 1, fps = 25, startFrame = 1 }) {
  const [ref, setRef] = useState(loadReference);
  const [pos, setPos] = useState(50);
  const [duration, setDuration] = useState(Infinity);
  const [open, setOpen] = useState(true);
  const hostRef = useRef(null);
  const videoRef = useRef(null);
  const raf = useRef(0);

  useEffect(() => subscribeReference(setRef), []);

  // Hold the reference video on the frame that matches the playhead (one seek per animation frame).
  const target = ref?.kind === 'video' ? referenceTime(ref, frame, fps, duration) : null;
  useEffect(() => {
    const v = videoRef.current;
    if (!v || target === null) return undefined;
    if (raf.current) cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(() => {
      raf.current = 0;
      if (Math.abs(v.currentTime - target) > 0.001) v.currentTime = target;
    });
    return () => { if (raf.current) cancelAnimationFrame(raf.current); };
  }, [target, ref?.url]);

  const dragTo = useCallback((e) => {
    const rect = hostRef.current?.getBoundingClientRect();
    if (!rect || !rect.width) return;
    setPos(Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100)));
  }, []);

  if (!ref) return null;
  const covered = referenceCovers(ref, frame, fps, duration);
  const setOffset = (n) => saveReference({ ...ref, offset: Math.max(0, Math.round(Number(n) || 0)) });

  return (
    <section className="rounded-xl border border-white/10 bg-black/20 p-3 space-y-2" aria-label="A/B reference" data-testid="ab-reference">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0">
          <h4 className="m-0 text-sm font-bold text-white/90">A/B reference</h4>
          <p className="m-0 truncate text-mini text-muted" title={ref.name}>
            <span className="text-sky-300">A</span> {ref.name} · <span className="text-pink-300">B</span> live preview, frame {frame}
          </p>
        </div>
        <div className="flex items-center gap-1.5">
          <Button size="sm" variant="ghost" onClick={() => setOpen((o) => !o)} aria-expanded={open}>{open ? 'Hide' : 'Show'}</Button>
          <Button size="sm" variant="ghost" onClick={clearReference} data-testid="ab-clear" title="Stop comparing against this render">Remove</Button>
        </div>
      </div>

      {open && (
        <>
          <div
            ref={hostRef}
            data-testid="ab-wipe-surface"
            onPointerDown={(e) => { e.currentTarget.setPointerCapture?.(e.pointerId); dragTo(e); }}
            onPointerMove={(e) => { if (e.buttons & 1) dragTo(e); }}
            className="relative aspect-video w-full cursor-ew-resize select-none overflow-hidden rounded-lg bg-black touch-none"
          >
            {previewSrc
              ? <img src={previewSrc} alt="Live preview (B)" className="absolute inset-0 h-full w-full object-contain pointer-events-none" draggable={false} />
              : <div className="absolute inset-0 grid place-items-center text-mini text-muted">Waiting for a preview…</div>}
            <div className="absolute inset-0 pointer-events-none" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
              {ref.kind === 'video' ? (
                <video
                  ref={videoRef} src={ref.url} muted playsInline preload="auto"
                  onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
                  className="h-full w-full bg-black object-contain" data-testid="ab-video"
                />
              ) : (
                <img src={ref.url} alt={`Reference (A): ${ref.name}`} className="h-full w-full bg-black object-contain" draggable={false} />
              )}
            </div>
            <div className="absolute inset-y-0 w-0.5 -translate-x-1/2 bg-white/90 shadow-[0_0_0_1px_rgba(0,0,0,0.5)] pointer-events-none" style={{ left: `${pos}%` }} />
            <span className="absolute left-2 top-2 rounded bg-black/70 px-1.5 py-0.5 text-nano font-bold text-sky-300 pointer-events-none">A · reference</span>
            <span className="absolute right-2 top-2 rounded bg-black/70 px-1.5 py-0.5 text-nano font-bold text-pink-300 pointer-events-none">B · live</span>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <label className="flex min-w-[180px] flex-1 items-center gap-2 text-mini text-muted">
              Wipe
              <input
                type="range" min={0} max={100} value={Math.round(pos)} onChange={(e) => setPos(Number(e.target.value))}
                aria-label="A/B wipe position" data-testid="ab-slider" className="h-1 flex-1 accent-[var(--accent)]"
              />
              <span className="w-9 text-right font-mono tabular-nums text-white/70">{Math.round(pos)}%</span>
            </label>
            {ref.kind === 'video' && (
              <div className="flex items-center gap-1.5 text-mini text-muted">
                <label className="flex items-center gap-1.5">
                  Starts at clip frame
                  <input
                    type="number" min={1} value={ref.offset + 1} aria-label="Frame of the clip where the reference render starts"
                    onChange={(e) => setOffset((Number(e.target.value) || 1) - 1)} data-testid="ab-offset"
                    className="w-20 rounded-md border border-white/10 bg-black/30 px-1.5 py-0.5 text-right font-mono tabular-nums text-white outline-none focus:border-[var(--accent)]"
                  />
                </label>
                <Button size="sm" variant="ghost" onClick={() => setOffset(startFrame - 1)} data-testid="ab-align"
                        title="Treat the reference as rendered from the current In point">Align to In</Button>
              </div>
            )}
          </div>
          {!covered && (
            <p className="m-0 text-mini text-amber-300" role="status" data-testid="ab-uncovered">
              The reference does not cover frame {frame} - A shows its nearest end, not this frame.
            </p>
          )}
        </>
      )}
    </section>
  );
}
