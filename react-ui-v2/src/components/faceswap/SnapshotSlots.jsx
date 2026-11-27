import React, { useEffect, useId, useRef, useState } from 'react';
import { useSlotStore, SLOT_IDS } from '../../store/slotStore';
import { useHistoryStore } from '../../store/historyStore';

// ── Four instant recipe snapshots: A / B / C / D ────────────────────────────
//
//   click an empty slot            capture the recipe (and the frame on screen)
//   click a filled slot            recall it, as one undoable action
//   Shift+click a filled slot      overwrite it with the current state
//   keys 1..4 over the stage       the same as clicking A..D
//   right-click / Menu key         Assign as Split-Wipe Reference, overwrite, clear
//
// The pill shows a thumbnail of the kept frame, a dot when the slot has been edited away from, and
// an R when it is the split-wipe reference.
const THUMB_W = 28;
const THUMB_H = 16;

function Thumb({ slot }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current;
    const bm = slot?.bitmap;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, c.width, c.height);
    if (!bm) return;
    try { ctx.drawImage(bm, 0, 0, c.width, c.height); } catch { /* bitmap closed under us: leave it blank */ }
  }, [slot]);
  return <canvas ref={ref} width={THUMB_W} height={THUMB_H} aria-hidden="true" className="h-4 w-7 rounded-[3px] bg-white/10" />;
}

export default function SnapshotSlots({ api, className = '', label = 'Snapshot slots', supportsReference = true }) {
  const bank = api.bank;
  const slots = useSlotStore((s) => s.banks[bank]);
  const active = useSlotStore((s) => s.active[bank]);
  const activeHeadId = useSlotStore((s) => s.activeHeadId[bank]);
  const reference = useSlotStore((s) => s.reference[bank]);
  const headId = useHistoryStore((s) => s.headId);
  const [menu, setMenu] = useState(null);          // slot id whose menu is open
  const rootRef = useRef(null);
  const menuId = useId();

  // A menu opened by right-click has no focus inside it, so Escape would go nowhere: move focus
  // to its first item, which is also what a keyboard user opening it with the Menu key needs.
  useEffect(() => {
    if (menu) rootRef.current?.querySelector('[role="menu"] button:not([disabled])')?.focus();
  }, [menu]);

  useEffect(() => {
    if (!menu) return undefined;
    const onDown = (e) => { if (rootRef.current && !rootRef.current.contains(e.target)) setMenu(null); };
    window.addEventListener('pointerdown', onDown, true);
    return () => {
      window.removeEventListener('pointerdown', onDown, true);
    };
  }, [menu]);

  const itemCls = 'block w-full px-3 py-1.5 text-left text-mini text-white/80 hover:bg-white/10 hover:text-white';

  return (
    <div ref={rootRef} role="group" aria-label={label} className={`relative flex items-center gap-1.5 ${className}`} data-testid="snapshot-slots">
      <span className="text-micro font-bold uppercase tracking-wide text-white/50" aria-hidden="true">Slots</span>
      {SLOT_IDS.map((id, i) => {
        const slot = slots[id];
        const isActive = active === id;
        const modified = isActive && activeHeadId !== headId;
        const isRef = reference === id;
        const detail = !slot ? 'empty: click to capture'
          : `${slot.bitmap ? `frame kept${slot.frameKey ? ` (${String(slot.frameKey).split('_').slice(-1)[0]})` : ''}, ${slot.width}x${slot.height}` : 'recipe only'}`;
        return (
          <div key={id} className="relative">
            <button
              type="button"
              onClick={(e) => api.activate(id, { overwrite: e.shiftKey })}
              onContextMenu={(e) => { e.preventDefault(); if (slot) setMenu(id); }}
              onKeyDown={(e) => { if (slot && (e.key === 'ContextMenu' || (e.shiftKey && e.key === 'F10'))) { e.preventDefault(); setMenu(id); } }}
              aria-pressed={isActive}
              aria-haspopup={slot ? 'menu' : undefined}
              aria-label={slot
                ? `Slot ${id}, ${detail}${isRef ? ', split-wipe reference' : ''}${modified ? ', modified since' : ''}. Click to recall, Shift click to overwrite, context menu for more`
                : `Slot ${id}, empty. Click to capture the current recipe and frame`}
              title={slot
                ? `Slot ${id} (key ${i + 1}) — ${detail}. Click: recall · Shift+click: overwrite · right-click: more`
                : `Slot ${id} (key ${i + 1}) — empty. Click to capture the current recipe and frame`}
              className={`flex h-7 items-center gap-1.5 rounded-lg border px-1.5 text-mini font-bold transition-colors ${
                isActive ? 'border-[var(--accent)] bg-[var(--accent)]/20 text-white'
                  : slot ? 'border-white/15 bg-white/[0.06] text-white/85 hover:bg-white/10'
                    : 'border-dashed border-white/15 bg-transparent text-white/55 hover:border-white/30 hover:text-white'
              }`}
              data-slot={id}
              data-slot-filled={slot ? '1' : '0'}
            >
              <span>{id}</span>
              {slot?.bitmap ? <Thumb slot={slot} /> : null}
              {isRef && <span className="rounded bg-cyan-500/30 px-1 text-nano text-cyan-100" title="Split-wipe reference">R</span>}
              {modified && <span className="h-1.5 w-1.5 rounded-full bg-amber-400" title="Edited since this slot was applied" />}
            </button>
            {menu === id && slot && (
              <div id={`${menuId}-${id}`} role="menu" tabIndex={-1} aria-label={`Slot ${id} options`} onKeyDown={(e) => { if (e.key === 'Escape') { e.stopPropagation(); setMenu(null); rootRef.current?.querySelector(`[data-slot="${id}"]`)?.focus(); } }} className="absolute left-0 top-full z-50 mt-1 w-60 overflow-hidden rounded-xl border border-white/10 bg-[#10131c]/95 py-1 shadow-2xl backdrop-blur-xl">
                {!supportsReference ? null : isRef ? (
                  <button type="button" role="menuitem" className={itemCls} onClick={() => { api.setReference(null); setMenu(null); }}>Clear split-wipe reference</button>
                ) : (
                  <button type="button" role="menuitem" className={itemCls} onClick={() => { api.setReference(id); setMenu(null); }} disabled={!slot.bitmap} title={slot.bitmap ? undefined : 'This slot kept no rendered frame to compare against'}>
                    Assign as Split-Wipe Reference
                  </button>
                )}
                <button type="button" role="menuitem" className={itemCls} onClick={() => { api.activate(id, { overwrite: true }); setMenu(null); }}>Overwrite with current state</button>
                <button type="button" role="menuitem" className={itemCls} onClick={() => { api.clear(id); setMenu(null); }}>Clear slot</button>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
