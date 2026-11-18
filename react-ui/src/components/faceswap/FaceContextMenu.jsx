import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { clampMenuPosition, placeBesideRect, getPoseBucket, formatPoseAngles } from './faceContextModel';
import { ROLE_CHOICES } from './faceRoleChoices';
import { uiScale } from './zoomPan';

// The right-click HUD on a detected face. Rendered in a portal on <body> and positioned `fixed`, so the stage's overflow
// clipping, its zoom / pan transform and its stacking context cannot cut it off, and kept inside the window.
//
// STAGE 12: it opens BESIDE the face (to its right, or its left when there is no room), not under the pointer, so the face
// stays in view while its menu is used. It is a compact glass panel: a role row (Primary / Secondary / Reference / Exclude), the
// face's own telemetry, and the per-face actions.
//
// COORDINATES. The app applies its UI zoom as CSS `zoom` on <html>. getBoundingClientRect, clientX / clientY and
// window.innerWidth are all in visual pixels, while `left` / `top` are in layout pixels, which the zoom scales. So the
// placement is worked out in visual pixels from the menu's MEASURED rect (faceContextModel.placeBesideRect /
// clampMenuPosition) and divided by uiScale() only at the very end. It never assumes the zoom is 1.
//
// DISMISSAL: Escape, a press anywhere outside, the window losing focus, a resize, or a wheel / scroll anywhere. A new
// right-click elsewhere dismisses it on the press, then opens a fresh one.

const SETTLE_MS = 250;
const NAV = '[role^="menuitem"]:not([aria-disabled="true"])';
const row = 'flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-mini text-[var(--text-main)] outline-none transition-colors hover:bg-white/[0.09] focus-visible:bg-white/[0.12] aria-disabled:cursor-not-allowed aria-disabled:opacity-45 aria-disabled:hover:bg-transparent';

export default function FaceContextMenu({ menu, onClose, face, sources = [], actions }) {
  const ref = useRef(null);
  const [pos, setPos] = useState(null);                       // { left, top, maxHeight, side } in layout px; null until measured
  const [picking, setPicking] = useState(false);
  const [hint, setHint] = useState(null);                     // the role whose explanation is shown

  useEffect(() => { setPicking(false); setPos(null); setHint(null); }, [menu?.faceIndex, menu?.x, menu?.y]);

  // Place the menu from its measured size, and again whenever that size changes (the source picker opening).
  useLayoutEffect(() => {
    const el = ref.current;
    if (!menu || !el) return undefined;
    const place = () => {
      const s = uiScale(el) || 1;
      const r = el.getBoundingClientRect();
      // Natural size: when a max-height is already applied the rect is the clamped one, so read the content height.
      const h = Math.max(r.height, el.scrollHeight * s);
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const c = menu.rect
        ? placeBesideRect({ rect: menu.rect, w: r.width, h, vw, vh, point: { x: menu.x, y: menu.y } })
        : { ...clampMenuPosition({ x: menu.x, y: menu.y, w: r.width, h, vw, vh }), side: 'point' };
      setPos((cur) => {
        const next = { left: c.left / s, top: c.top / s, maxHeight: c.maxHeight === null ? null : c.maxHeight / s, side: c.side };
        return cur && cur.left === next.left && cur.top === next.top && cur.maxHeight === next.maxHeight && cur.side === next.side ? cur : next;
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

  // Focus the first enabled action once the menu is placed, so the keyboard can drive it straight away.
  useEffect(() => {
    if (menu && pos) ref.current?.querySelector('[role="menuitem"]:not([aria-disabled="true"])')?.focus({ preventScroll: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [menu?.faceIndex, menu?.x, menu?.y, !!pos]);

  if (!menu || !face) return null;

  // The face's own telemetry: only what was MEASURED. The pose comes from the preview response; there is no per-face occlusion
  // figure anywhere in it, so none is shown (an earlier version printed "Occ: 0.0%" for every face).
  const trackMatch = typeof face.trackText === 'string' ? face.trackText.match(/\d+/) : null;
  const trackId = face.trackId ?? (trackMatch ? trackMatch[0] : (face.personId != null ? face.personId : (face.index != null ? face.index + 1 : '1')));
  const p = face.pose;
  const arr = Array.isArray(p) ? p : (p && typeof p === 'object' ? [p.yaw, p.pitch, p.roll] : null);
  const havePose = !!arr && Number.isFinite(Number(arr[0])) && Number.isFinite(Number(arr[1]));
  const yaw = havePose ? Number(arr[0]) : null;
  const pitch = havePose ? Number(arr[1]) : null;
  const poseBucket = havePose ? (face.poseBucket || face.pose_bucket || getPoseBucket(yaw, pitch)) : null;
  const occlusion = Number.isFinite(Number(face.occlusionPct)) && face.occlusionPct !== null ? Number(face.occlusionPct) : null;

  const onKeyDown = (e) => {
    const list = [...ref.current.querySelectorAll(NAV)];
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
    if (it.id === 'assign') { setPicking((v) => !v); return; }
    onClose();
    it.run?.();
  };

  const entries = [
    { id: 'assign', label: 'Assign Source Face / .fsz', hint: picking ? '▴' : '▾', sub: face.sourceName ? `Now: ${face.sourceName}` : '',
      reason: sources.length ? '' : 'Add a source face to the gallery first' },
    { id: 'loupe', label: 'Solo Face Crop in Loupe', sub: 'Zooms the stage to this face and switches the magnifier on',
      reason: face.loupeReason, run: actions.loupe },
    { id: 'copypose', label: 'Copy Pose to Filter', sub: face.copyPose?.sub, reason: face.copyPose?.reason, run: actions.copyPose },
    { id: 'paint', label: 'Paint Manual Occlusion Mask', sub: 'Zooms to this face and picks the Paint brush', reason: face.loupeReason, run: actions.paint },
    { id: 'pin', label: face.pinned ? `Unpin Identity from ${face.trackText}` : `Pin / Lock Identity to ${face.trackText}`,
      sub: face.pinned ? 'The renderer decides who this is again' : '', reason: face.pinReason, run: actions.pin },
    { id: 'bypass', label: 'Bypass Swap on This Face', sub: face.bypassSub, reason: face.bypassReason, run: actions.bypass },
    { id: 'extract', label: 'Extract Face Crop to Face Manager Bank',
      sub: 'Adds the full-resolution crop to #/facemgr', reason: face.extractReason, run: actions.extract },
    { id: 'skin', label: 'Auto-Match Skin Tone / Sample White Balance',
      sub: 'Cheek + forehead chroma vs. the swapped result', reason: face.skinReason, run: actions.skin },
    { id: 'focus', label: 'Focus & Zoom 200% on This Face', run: actions.focus },
    { id: 'reset', label: 'Reset Face Overrides', danger: true, sub: face.reset?.sub, reason: face.reset?.reason, run: actions.reset },
  ];

  const shownHint = ROLE_CHOICES.find((c) => c.id === (hint || face.choice)) || null;
  const hintReason = shownHint ? (face.roleBlocked?.[shownHint.id] || '') : '';

  return createPortal(
    <div
      ref={ref}
      role="menu"
      tabIndex={-1}
      aria-label={`Actions for ${face.badge}`}
      data-testid="face-context-menu"
      data-side={pos?.side || ''}
      onKeyDown={onKeyDown}
      onContextMenu={(e) => e.preventDefault()}
      className="fixed z-[200] w-[min(320px,calc(100vw-16px))] overflow-y-auto rounded-2xl border border-white/15 bg-[linear-gradient(160deg,rgba(24,30,48,0.82),rgba(8,10,18,0.78))] p-1.5 shadow-[0_18px_48px_rgba(0,0,0,0.65),inset_0_1px_0_rgba(255,255,255,0.08)] backdrop-blur-2xl backdrop-saturate-150"
      style={{
        left: pos ? pos.left : 0,
        top: pos ? pos.top : 0,
        maxHeight: pos?.maxHeight ?? undefined,
        visibility: pos ? 'visible' : 'hidden',      // measured while hidden, so it never flashes at the wrong place
      }}
    >
      <div className="flex items-center gap-2 px-2.5 pb-1 pt-1.5">
        <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--accent)] shadow-[0_0_8px_var(--accent)]" aria-hidden="true" />
        <div className="min-w-0 flex-1 truncate text-nano font-semibold uppercase tracking-[0.14em] text-white/70" data-testid="face-menu-badge">
          {face.badge}
        </div>
      </div>

      {/* The face's own telemetry. Only measured values: a pose when the preview returned one, a track id always. */}
      <div className="mx-1 mb-1.5 space-y-1 rounded-xl border border-white/10 bg-black/35 p-2 font-mono text-micro" data-testid="face-menu-telemetry">
        <div className="flex items-center justify-between text-white/90">
          <span className="font-bold text-[var(--accent)]" data-testid="face-menu-track-id">Track ID: #{trackId}</span>
          {poseBucket && <span className="rounded bg-white/10 px-1.5 py-0.5 text-white/80" data-testid="face-menu-pose-bucket">Bucket: {poseBucket}</span>}
        </div>
        <div className="flex items-center justify-between text-white/70">
          <span data-testid="face-menu-pose">{havePose ? formatPoseAngles(arr) : 'Pose not measured on this frame'}</span>
          {occlusion !== null && (
            <span className={occlusion > 25 ? 'font-bold text-amber-400' : 'text-emerald-400'} data-testid="face-menu-occlusion">Occ: {occlusion.toFixed(1)}%</span>
          )}
        </div>
      </div>

      {/* Role badges: what this face is for the render. Each is a real change to the mapping, not a label. */}
      <div className="mx-1 mb-1.5" role="group" aria-label="Face role" data-testid="face-menu-roles">
        <div className="grid grid-cols-2 gap-1">
          {ROLE_CHOICES.map((c) => {
            const blocked = face.roleBlocked?.[c.id] || '';
            const current = face.choice === c.id;
            return (
              <button
                key={c.id}
                type="button"
                role="menuitemradio"
                aria-checked={current}
                aria-disabled={!!blocked && !current}
                data-testid={`face-role-${c.id}`}
                title={current ? `${c.label} (current)` : blocked || c.label}
                onMouseEnter={() => setHint(c.id)}
                onMouseLeave={() => setHint(null)}
                onFocus={() => setHint(c.id)}
                onBlur={() => setHint(null)}
                onClick={() => { if (blocked || current) return; onClose(); actions.role?.(c.id); }}
                className={`flex items-center gap-1.5 rounded-lg border px-2 py-1.5 text-left text-micro font-semibold outline-none transition-colors focus-visible:ring-2 focus-visible:ring-[var(--accent)] ${
                  current ? 'text-white' : 'border-white/10 bg-white/[0.04] text-white/80 hover:bg-white/[0.09] aria-disabled:hover:bg-white/[0.04]'} aria-disabled:cursor-not-allowed aria-disabled:opacity-40`}
                style={current ? { borderColor: c.color, background: `${c.color}26`, boxShadow: `0 0 12px ${c.color}40` } : undefined}
              >
                <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: c.color }} aria-hidden="true" />
                <span className="min-w-0 flex-1 truncate">{c.short}{c.id === 'exclude' ? ' (never swap)' : ''}</span>
                {current && <span aria-hidden="true">{'✓'}</span>}
              </button>
            );
          })}
        </div>
        {shownHint && (
          <p className={`mt-1 px-1 text-note leading-snug ${hintReason && hint ? 'text-amber-300/90' : 'text-white/55'}`} data-testid="face-role-hint">
            <span className="font-semibold" style={{ color: shownHint.color }}>{shownHint.label}.</span>{' '}
            {hint && hintReason ? hintReason : shownHint.hint}
          </p>
        )}
      </div>

      {/* 1-Click .fsz faceset lock / swap toggles (Stage 25) */}
      <div className="mx-1 mb-1.5 grid grid-cols-2 gap-1" data-testid="face-menu-toggles">
        <button
          type="button"
          role="menuitemcheckbox"
          data-testid="face-menu-lock-toggle"
          aria-checked={!!face.pinned}
          disabled={!!face.pinReason}
          onClick={() => { onClose(); actions.pin?.(); }}
          className="flex items-center justify-center gap-1 rounded-lg border border-white/10 bg-white/[0.05] px-2 py-1 text-left text-micro font-semibold transition-all hover:border-amber-400/40 hover:bg-white/[0.09] disabled:cursor-not-allowed disabled:opacity-40"
          title={face.pinReason || (face.pinned ? '1-Click Unlock .fsz Faceset' : '1-Click Lock .fsz Faceset')}
        >
          <span className={face.pinned ? 'text-amber-300' : 'text-white/60'}>{face.pinned ? '🔒' : '🔓'}</span>
          <span className="truncate">{face.pinned ? 'Unlock .fsz' : 'Lock .fsz'}</span>
        </button>
        <button
          type="button"
          role="menuitemcheckbox"
          data-testid="face-menu-swap-toggle"
          aria-checked={!face.excluded}
          disabled={!!face.bypassReason}
          onClick={() => { onClose(); actions.bypass?.(); }}
          className="flex items-center justify-center gap-1 rounded-lg border border-white/10 bg-white/[0.05] px-2 py-1 text-left text-micro font-semibold transition-all hover:border-sky-400/40 hover:bg-white/[0.09] disabled:cursor-not-allowed disabled:opacity-40"
          title={face.bypassReason || (face.excluded ? '1-Click Enable Swap' : '1-Click Bypass Swap')}
        >
          <span className={face.excluded ? 'text-rose-400' : 'text-emerald-400'}>{face.excluded ? '⊘' : '✓'}</span>
          <span className="truncate">{face.excluded ? 'Enable Swap' : 'Bypass Swap'}</span>
        </button>
      </div>

      {entries.map((it) => {
        const disabled = !!it.reason;
        return (
          <React.Fragment key={it.id}>
            {it.id === 'pin' && <div className="mx-2 my-1 h-px bg-white/10" role="separator" />}
            {it.id === 'reset' && <div className="mx-2 my-1 h-px bg-white/10" role="separator" />}
            <button
              type="button" role="menuitem" data-testid={`face-menu-${it.id}`}
              aria-disabled={disabled} aria-expanded={it.id === 'assign' ? picking : undefined}
              title={it.reason || undefined}
              onClick={run(it)}
              className={`${row} ${it.danger ? 'text-rose-200' : ''}`}
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
                   className="mx-1 mb-1 max-h-48 overflow-y-auto rounded-lg border border-white/10 bg-black/30 p-1">
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
