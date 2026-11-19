import React, { useEffect, useMemo, useState } from 'react';
import {
  ANSI_MAIN, ANSI_NAV, GROUPS, GROUP_BY_ID, LAYERS, LAYER_BY_ID, analyzeKeymap, keyView, modifierUse,
} from '../../shortcuts/ansiLayout';
import { formatCombo } from '../../shortcuts/combo';
import { tokenOfEvent } from '../../shortcuts/keyboardModel';

// The key remapper's keyboard (Settings -> Keyboard Shortcuts, and the command palette's "Open the key remapper"): a full ANSI
// layout where every key a workstation shortcut uses is lit in its family's colour -
//
//   Transport · Viewport · Masking · Pipeline Soloing · Snapshots · Workspace & Timeline
//
// - and a key that two actions can fire together is ringed red with a "!" the moment the keymap makes it so (a remap, an
// import). The keyboard shows ONE modifier layer at a time (Plain / Ctrl / Alt / Shift / Ctrl+Shift): press a modifier key on
// the drawing, or a layer tab, to see what lives under it - Alt is where Alt+1..5 (saved workspace layouts) sit.
//
// Everything lit is read from the live keymap (shortcuts/ansiLayout.analyzeKeymap), so a remap moves its highlight and nothing
// here is a second copy of the defaults. The keys are a mouse affordance (click one to list its actions); the keyboard route
// to every action is the list beside it, so the 80-odd drawn keys add no tab stops.

const FLASH_MS = 240;
const KU = 'var(--ku, 2.5rem)';          // one key unit

const MOD_LAYER = { Mod: 'ctrl', Alt: 'alt', Shift: 'shift' };

export default function AnsiKeyboard({ overrides = {}, layerId, onLayerChange, selected, onSelect }) {
  const analysis = useMemo(() => analyzeKeymap(overrides), [overrides]);
  const layer = LAYER_BY_ID[layerId] || LAYER_BY_ID.plain;
  const modUse = useMemo(() => modifierUse(analysis), [analysis]);
  const [hide, setHide] = useState(() => new Set());          // family ids switched off in the legend
  const [pressed, setPressed] = useState({});

  // Flash the key under each press (capture phase, so it shows even if a handler stops the event).
  useEffect(() => {
    const timers = new Set();
    const flash = (token, on) => setPressed((p) => {
      if (on) return { ...p, [token]: true };
      const next = { ...p };
      delete next[token];
      return next;
    });
    const onDown = (e) => {
      const token = tokenOfEvent(e);
      if (!token) return;
      flash(token, true);
      const t = setTimeout(() => flash(token, false), FLASH_MS);
      timers.add(t);
    };
    window.addEventListener('keydown', onDown, true);
    return () => {
      window.removeEventListener('keydown', onDown, true);
      timers.forEach((t) => clearTimeout(t));
    };
  }, []);

  const toggleGroup = (id) => setHide((h) => {
    const next = new Set(h);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });

  const renderKey = (key, i) => {
    if (key.gap !== undefined) return <span key={`g${i}`} aria-hidden="true" style={{ width: `calc(${KU} * ${key.gap})`, flex: 'none' }} />;
    const width = `calc(${KU} * ${key.w} - 4px)`;
    const base = 'relative flex h-[calc(var(--ku,2.5rem)-4px)] flex-none items-center justify-center rounded-md border font-mono text-nano font-bold tabular-nums select-none transition-[background-color,border-color,box-shadow,transform] duration-150 ease-out motion-reduce:transition-none';

    if (key.mod) {
      const lit = layer.mods.includes(key.mod);
      const used = modUse[key.mod] || 0;
      return (
        <button
          key={`${key.t}${i}`}
          type="button"
          tabIndex={-1}
          data-key={key.l}
          data-lit={lit ? 'yes' : 'no'}
          aria-label={`${key.l}: ${used} binding${used === 1 ? '' : 's'} use it. Click to show the ${key.l} layer`}
          title={`${key.l} — ${used} binding${used === 1 ? '' : 's'}. Click to show that layer.`}
          onClick={() => onLayerChange?.(lit ? 'plain' : MOD_LAYER[key.mod])}
          style={{ width, marginInline: 2 }}
          className={`${base} text-white ${lit ? 'border-white/70 bg-white/25' : used ? 'border-white/20 bg-white/[0.08]' : 'border-white/[0.07] bg-white/[0.03] text-white/65'}`}
        >
          {key.l}
        </button>
      );
    }
    if (key.dead) {
      return <span key={`${key.t}${i}`} aria-hidden="true" style={{ width, marginInline: 2 }} className={`${base} border-white/[0.05] bg-transparent text-white/60`}>{key.l}</span>;
    }

    const view = keyView(key.t, analysis, layer);
    const shown = view.bound && view.groups.some((g) => !hide.has(g));
    const primary = view.primary ? GROUP_BY_ID[view.primary] : null;
    const isSel = selected === key.t;
    const count = view.actions.length;
    const style = { width, marginInline: 2 };
    if (shown && primary) {
      style.background = `color-mix(in srgb, ${primary.color} 30%, transparent)`;
      style.borderColor = `color-mix(in srgb, ${primary.color} 75%, transparent)`;
      style.boxShadow = `0 0 10px color-mix(in srgb, ${primary.color} 35%, transparent)`;
    }
    if (view.conflict && shown) {
      style.borderColor = '#ef4444';
      style.boxShadow = '0 0 0 2px rgba(239,68,68,0.55)';
    }
    const title = view.bound
      ? view.entries.map((e) => `${formatCombo(e.combo)}: ${e.actions.map((a) => a.label).join(', ')}`).join('\n')
      : (view.elsewhere ? `Free here — ${view.elsewhere} binding${view.elsewhere === 1 ? '' : 's'} in another layer` : 'Not bound');
    return (
      <button
        key={`${key.t}${i}`}
        type="button"
        tabIndex={-1}
        data-key={key.t}
        data-lit={shown ? 'yes' : 'no'}
        data-group={shown ? view.primary : undefined}
        data-conflict={view.conflict ? 'yes' : undefined}
        aria-pressed={isSel}
        aria-label={`${key.l}: ${view.bound ? `${count} action${count === 1 ? '' : 's'}${view.conflict ? ', conflict' : ''}` : 'not bound'}`}
        title={title}
        onClick={() => onSelect?.(isSel ? null : key.t)}
        style={style}
        className={`${base} ${shown ? 'text-white' : view.bound ? 'border-white/10 bg-white/[0.04] text-white/60' : 'border-white/[0.07] bg-white/[0.02] text-white/60'} ${isSel ? 'outline outline-2 outline-white/80' : ''} ${pressed[key.t] ? 'scale-[0.94] motion-reduce:scale-100 brightness-150' : ''}`}
      >
        {key.l}
        {shown && view.groups.length > 1 && (
          <span aria-hidden="true" className="absolute bottom-0.5 left-1/2 flex -translate-x-1/2 gap-px">
            {view.groups.slice(1, 4).map((g) => <i key={g} className="h-1 w-1 rounded-full" style={{ background: GROUP_BY_ID[g].color }} />)}
          </span>
        )}
        {shown && count > 1 && <span aria-hidden="true" className="absolute right-0.5 top-0 text-[0.55rem] opacity-70">{count}</span>}
        {view.conflict && shown && <span aria-hidden="true" className="absolute -right-1 -top-1 grid h-3.5 w-3.5 place-items-center rounded-full bg-red-500 text-[0.6rem] font-black text-white">!</span>}
      </button>
    );
  };

  const rowsOf = (rows) => rows.map((row, r) => (
    <div key={r} className="flex items-center" style={{ minHeight: KU }}>
      {row.map(renderKey)}
    </div>
  ));

  return (
    <div className="space-y-3" data-testid="ansi-keyboard" data-layer={layer.id}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div role="tablist" aria-label="Modifier layer" className="flex items-center gap-1 rounded-lg border border-white/10 bg-black/25 p-0.5">
          {LAYERS.map((l) => (
            <button
              key={l.id}
              type="button"
              role="tab"
              aria-selected={layer.id === l.id}
              data-layer-tab={l.id}
              onClick={() => onLayerChange?.(l.id)}
              className={`rounded-md px-2.5 py-1 text-mini font-semibold apple-transition ${layer.id === l.id ? 'bg-[var(--accent)]/25 text-white shadow-[inset_0_0_0_1px_color-mix(in_srgb,var(--accent)_45%,transparent)]' : 'text-muted hover:text-white hover:bg-white/[0.06]'}`}
            >
              {l.label} <span className="tabular-nums text-white/70">{analysis.layerCount[l.id]}</span>
            </button>
          ))}
        </div>
        <ul className="m-0 flex list-none flex-wrap items-center gap-1.5 p-0" aria-label="Key colours">
          {GROUPS.map((g) => (
            <li key={g.id}>
              <button
                type="button"
                aria-pressed={!hide.has(g.id)}
                data-group-chip={g.id}
                onClick={() => toggleGroup(g.id)}
                title={hide.has(g.id) ? `Show ${g.label} keys` : `Hide ${g.label} keys`}
                className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-nano font-semibold apple-transition ${hide.has(g.id) ? 'border-white/10 text-white/60' : 'border-white/20 text-white/85'}`}
              >
                <i className="h-2 w-2 rounded-sm" style={{ background: hide.has(g.id) ? 'transparent' : g.color, boxShadow: `inset 0 0 0 1px ${g.color}` }} />
                {g.label}
              </button>
            </li>
          ))}
          <li className="inline-flex items-center gap-1.5 rounded-full border border-red-500/50 px-2 py-0.5 text-nano font-semibold text-red-300">
            <i className="grid h-3 w-3 place-items-center rounded-full bg-red-500 text-[0.55rem] font-black text-white">!</i> Conflict
          </li>
        </ul>
      </div>

      <div className="overflow-x-auto rounded-xl border border-white/10 bg-black/30 p-3 sm:p-4" role="group" aria-label="Keyboard, ANSI layout">
        <div className="mx-auto flex w-max gap-[calc(var(--ku,2.5rem)*0.5)]" style={{ '--ku': 'clamp(1.9rem, 2.4vw, 2.6rem)' }}>
          <div>{rowsOf(ANSI_MAIN)}</div>
          <div>{rowsOf(ANSI_NAV)}</div>
        </div>
      </div>
    </div>
  );
}
