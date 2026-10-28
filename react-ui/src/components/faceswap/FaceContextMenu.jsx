import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { clampMenuPosition } from './faceContextModel';
import { uiScale } from './zoomPan';

// The right-click menu on a detected face. Rendered in a portal on <body> and positioned `fixed`, so the stage's
// overflow clipping, its zoom / pan transform and its stacking context cannot cut it off, and clamped to the window.
//
// COORDINATES. The app applies its UI zoom as CSS `zoom` on <html>. getBoundingClientRect, clientX / clientY and
// window.innerWidth are all in visual pixels, while `left` / `top` are in layout pixels, which the zoom scales. So the
// placement is worked out in visual pixels from the menu's MEASURED rect (faceContextModel.clampMenuPosition) and
// divided by uiScale() only at the very end. It never assumes the zoom is 1.
//
// DISMISSAL: Escape, a press anywhere outside, the window losing focus, a resize, or a wheel / scroll anywhere. A new
// right-click elsewhere dismisses it on the press, then opens a fresh one.

const SETTLE_MS = 250;
const row = 'flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-mini text-[var(--text-main)] outline-none transition-colors hover:bg-white/[0.08] focus-visible:bg-white/[0.1] aria-disabled:cursor-not-allowed aria-disabled:opacity-45 aria-disabled:hover:bg-transparent';

export default function FaceContextMenu({ menu, onClose, face, sources = [], actions }) {
  const ref = useRef(null);
  const [pos, setPos] = useState(null);                       // { left, top, maxHeight } in layout px; null until measured
  const [picking, setPicking] = useState(false);

  useEffect(() => { setPicking(false); setPos(null); }, [menu?.faceIndex, menu?.x, menu?.y]);

  // Place the menu from its measured size, and again whenever that size changes (the source picker opening).
  useLayoutEffect(() => {
    const el = ref.current;
    if (!menu || !el) return undefined;
    const place = () => {
      const s = uiScale(el) || 1;
      const r = el.getBoundingClientRect();
      // Natural size: when a max-height is already applied the rect is the clamped one, so read the content height.
      const h = Math.max(r.height, el.scrollHeight * s);
      const c = clampMenuPosition({ x: menu.x, y: menu.y, w: r.width, h, vw: window.innerWidth, vh: window.innerHeight });
      setPos((cur) => {
        const next = { left: c.left / s, top: c.top / s, maxHeight: c.maxHeight === null ? null : c.maxHeight / s };
        return cur && cur.left === next.left && cur.top === next.top && cur.maxHeight === next.maxHeight ? cur : next;
      });
    };
    place();
    if (typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(place);
    ro.observe(el);
    return () => ro.disconnect();
  }, [menu, picking]);

  useEffect(() => {
    if (!menu) return undefined;
    const outside = (e) => { if (!ref.current?.contains(e.target)) onClose(); };
    const onKey = (e) => { if (e.code === 'Escape') { e.stopPropagation(); onClose(); } };
    const close = () => onClose();
    // The menu's own source list scrolls; that must not dismiss it. And a scroll that was already under way when the
    // menu opened (the browser delivers the event a frame after the scroll itself, i.e. after the contextmenu event
    // that opened this) is not the user scrolling AWAY from it - it is ignored for a moment, or it would close the menu
    // the instant it appeared.
    const openedAt = performance.now();
    const closeOutside = (e) => {
      if (performance.now() - openedAt < SETTLE_MS || ref.current?.contains(e.target)) return;
      onClose();
    };
    window.addEventListener('pointerdown', outside, true);
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('blur', close);
    window.addEventListener('resize', close);
    window.addEventListener('wheel', closeOutside, { passive: true, capture: true });
    window.addEventListener('scroll', closeOutside, true);
    return () => {
      window.removeEventListener('pointerdown', outside, true);
      window.removeEventListener('keydown', onKey, true);
      window.removeEventListener('blur', close);
      window.removeEventListener('resize', close);
      window.removeEventListener('wheel', closeOutside, { capture: true });
      window.removeEventListener('scroll', closeOutside, true);
    };
  }, [menu, onClose]);

  // Focus the first enabled item once the menu is placed, so the keyboard can drive it straight away.
  useEffect(() => {
    if (menu && pos) ref.current?.querySelector('[role="menuitem"]:not([aria-disabled="true"])')?.focus({ preventScroll: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [menu?.faceIndex, menu?.x, menu?.y, !!pos]);

  if (!menu || !face) return null;

  const onKeyDown = (e) => {
    const list = [...ref.current.querySelectorAll('[role="menuitem"]:not([aria-disabled="true"])')];
    if (!list.length) return;
    const at = list.indexOf(document.activeElement);
    const go = (i) => { e.preventDefault(); list[(i + list.length) % list.length].focus(); };
    if (e.key === 'ArrowDown') go(at + 1);
    else if (e.key === 'ArrowUp') go(at < 0 ? -1 : at - 1);
    else if (e.key === 'Home') go(0);
    else if (e.key === 'End') go(list.length - 1);
  };

  const run = (it) => () => {
    if (it.reason) return;
    if (it.id === 'assign') { setPicking((p) => !p); return; }
    onClose();
    it.run?.();
  };

  const entries = [
    { id: 'assign', label: 'Assign / Change Source Identity', hint: picking ? '▴' : '▾', sub: face.sourceName ? `Now: ${face.sourceName}` : '',
      reason: sources.length ? '' : 'Add a source face to the gallery first' },
    { id: 'pin', label: face.pinned ? `Unpin Identity from ${face.trackText}` : `Pin / Lock Identity to ${face.trackText}`,
      sub: face.pinned ? 'The renderer decides who this is again' : '', reason: face.pinReason, run: actions.pin },
    { id: 'bypass', label: 'Bypass Swap on This Face', sub: face.bypassSub, reason: face.bypassReason, run: actions.bypass },
    { id: 'extract', label: 'Extract Face Crop to Face Manager Bank',
      sub: 'Adds the full-resolution crop to #/facemgr', reason: face.extractReason, run: actions.extract },
    { id: 'skin', label: 'Auto-Match Skin Tone / Sample White Balance',
      sub: 'Cheek + forehead chroma vs. the swapped result', reason: face.skinReason, run: actions.skin },
    { id: 'focus', label: 'Focus & Zoom 200% on This Face', run: actions.focus },
  ];

  return createPortal(
    <div
      ref={ref}
      role="menu"
      tabIndex={-1}
      aria-label={`Actions for ${face.badge}`}
      data-testid="face-context-menu"
      onKeyDown={onKeyDown}
      onContextMenu={(e) => e.preventDefault()}
      className="fixed z-[200] w-[min(300px,calc(100vw-16px))] overflow-y-auto rounded-xl border border-[var(--border-strong)] bg-[var(--card-bg)] p-1 shadow-[0_14px_36px_rgba(0,0,0,0.6)] backdrop-blur-xl"
      style={{
        left: pos ? pos.left : 0,
        top: pos ? pos.top : 0,
        maxHeight: pos?.maxHeight ?? undefined,
        visibility: pos ? 'visible' : 'hidden',      // measured while hidden, so it never flashes at the wrong place
      }}
    >
      <div className="px-2.5 pb-1 pt-1.5 text-nano font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]" data-testid="face-menu-badge">
        {face.badge}
      </div>
      {entries.map((it) => {
        const disabled = !!it.reason;
        return (
          <React.Fragment key={it.id}>
            <button
              type="button" role="menuitem" data-testid={`face-menu-${it.id}`}
              aria-disabled={disabled} aria-expanded={it.id === 'assign' ? picking : undefined}
              title={it.reason || undefined}
              onClick={run(it)}
              className={row}
            >
              <span className="min-w-0 flex-1">
                <span className="block truncate">{it.label}</span>
                {(it.reason || it.sub) && (
                  <span className={`block truncate text-note ${it.reason ? 'text-amber-300/90' : 'text-[var(--text-muted)]'}`}>{it.reason || it.sub}</span>
                )}
              </span>
              {it.hint && <span className="text-[var(--text-muted)]" aria-hidden="true">{it.hint}</span>}
            </button>
            {it.id === 'assign' && picking && (
              <div role="group" aria-label="Source identity" data-testid="face-menu-picker"
                   className="mx-1 mb-1 max-h-48 overflow-y-auto rounded-lg border border-[var(--border-color)] bg-[var(--input-bg)] p-1">
                {sources.map((s) => (
                  <button key={s.index} type="button" role="menuitemradio" aria-checked={s.index === face.sourceIndex}
                          data-testid={`face-menu-source-${s.index}`}
                          onClick={() => { onClose(); actions.assign(s.index); }}
                          className={`${row} !py-1`}>
                    {s.thumb
                      ? <img src={s.thumb} alt="" className="h-7 w-7 shrink-0 rounded-md object-cover" />
                      : <span className="h-7 w-7 shrink-0 rounded-md bg-white/10" />}
                    <span className="min-w-0 flex-1 truncate">{s.name}</span>
                    {s.index === face.sourceIndex && <span aria-hidden="true">{'✓'}</span>}
                  </button>
                ))}
                <button type="button" role="menuitemradio" aria-checked={face.excluded} data-testid="face-menu-source-skip"
                        onClick={() => { onClose(); actions.assign(-1); }} className={`${row} !py-1`}>
                  <span className="grid h-7 w-7 shrink-0 place-items-center rounded-md bg-white/10 text-[var(--text-muted)]" aria-hidden="true">{'∅'}</span>
                  <span className="min-w-0 flex-1 truncate">Skip - never swap this person</span>
                  {face.excluded && <span aria-hidden="true">{'✓'}</span>}
                </button>
              </div>
            )}
          </React.Fragment>
        );
      })}
    </div>,
    document.body,
  );
}
