import { useShallow } from 'zustand/react/shallow';
import React, { useState, useRef, useEffect, useMemo } from 'react';
import { LAYOUT_PRESETS, useWorkspaceStore } from '../../store/workspaceStore';
import { useShortcutStore } from '../../store/shortcutStore';
import { bindingsOf } from '../../shortcuts/keymap';
import { formatCombo } from '../../shortcuts/combo';
import { LAYOUT_SLOT_ACTION_IDS } from '../../shortcuts/actions';
import {
  ASPECTS, ASPECT_ANY, ASPECT_HINT, HOTKEY_SLOTS, classifyAspect, fitsWindow, orderForAspect,
} from '../../workspace/layoutModel';
import { Icon } from '../../icons';

// The workspace layout manager (header). The first three presets stay in the header bar; everything else is in the menu:
//
//   presets          the built-in workstation layouts
//   saved layouts    the user's own split-pane proportions. Each has a name (renameable), the monitor shape it is for
//                    (16:9 / 21:9 / 32:9 / any - the layouts for THIS window's shape are listed first) and an Alt+1..5
//                    slot: pressing the slot anywhere in the app switches to it (App.jsx). A layout saved on one monitor
//                    keeps its proportions on another monitor of the same shape (store/workspaceStore.applyLayout).
//
// Choosing a layout never leaves the current hash route.

const useWindowSize = () => {
  const read = () => ({ w: window.innerWidth, h: window.innerHeight });
  const [size, setSize] = useState(read);
  useEffect(() => {
    const on = () => setSize(read());
    window.addEventListener('resize', on);
    return () => window.removeEventListener('resize', on);
  }, []);
  return size;
};

const field = 'rounded-md bg-black/60 border border-white/20 px-1.5 py-1 text-xs text-white focus:outline-none focus:border-accent';

function LayoutRow({ layout, active, hereAspect, width, slotKeys, onApply, onClose }) {
  const { renameCustomLayout, bindLayoutHotkey, setLayoutAspect, overwriteCustomLayout, deleteCustomLayout } = useWorkspaceStore.getState();
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(layout.name);
  const [error, setError] = useState('');
  const fit = fitsWindow(layout.sizes, width);
  const otherShape = layout.aspect !== ASPECT_ANY && layout.aspect !== hereAspect;

  const submitName = (e) => {
    e.preventDefault();
    const r = renameCustomLayout(layout.id, name);
    if (!r.ok) { setError(r.error); return; }
    setError('');
  };

  return (
    <div
      data-testid="layout-row"
      data-layout-id={layout.id}
      data-aspect={layout.aspect}
      className={`rounded-lg text-xs transition-colors ${active ? 'bg-[var(--accent)]/20 text-white' : 'text-white/80 hover:bg-white/10'}`}
    >
      <div className="flex items-center justify-between gap-1 px-2.5 py-1.5">
        <button type="button" onClick={() => { onApply(layout.id); onClose?.(); }} className="flex-1 text-left min-w-0 pr-1" aria-label={`Switch to layout ${layout.name}`}>
          <div className={`truncate ${active ? 'font-semibold' : 'font-medium'}`}>{layout.name}</div>
          <div className="text-micro text-white/60 truncate font-mono">
            {layout.sizes.leftWidth}px · {layout.sizes.rightWidth}px · {layout.sizes.bottomHeight}px
            {!fit.fits && <span className="ml-1 text-amber-300" title={`On this ${width}px-wide window the centre stage would be ${fit.centre}px; the side panels shrink to keep 360px`}>· tight here</span>}
          </div>
        </button>
        <div className="flex items-center gap-1 shrink-0">
          <span
            data-testid="layout-aspect-badge"
            title={ASPECT_HINT[layout.aspect]}
            className={`rounded px-1 text-micro font-bold ${otherShape ? 'bg-white/5 text-white/60' : 'bg-white/10 text-white/80'}`}
          >{layout.aspect === ASPECT_ANY ? 'any' : layout.aspect}</span>
          {layout.hotkey && (
            <kbd data-testid="layout-hotkey-badge" className="rounded bg-[var(--accent)]/25 px-1 font-mono text-micro text-white" title={`Press ${slotKeys[layout.hotkey - 1]} to switch here`}>
              {slotKeys[layout.hotkey - 1]}
            </kbd>
          )}
          {active && <Icon.done size={14} className="text-accent" />}
          <button
            type="button"
            onClick={() => { setEditing((v) => !v); setName(layout.name); setError(''); }}
            aria-expanded={editing}
            aria-label={`Edit layout ${layout.name}: rename, hotkey, monitor shape`}
            className={`rounded px-1.5 py-0.5 text-micro font-semibold ${editing ? 'bg-white/15 text-white' : 'text-white/65 hover:text-white hover:bg-white/10'}`}
          >
            Edit
          </button>
        </div>
      </div>

      {editing && (
        <div className="space-y-1.5 border-t border-white/10 px-2.5 py-2" data-testid="layout-edit">
          <form onSubmit={submitName} className="flex items-center gap-1">
            <input
              type="text"
              value={name}
              maxLength={40}
              aria-label="Layout name"
              aria-invalid={error ? true : undefined}
              onChange={(e) => { setName(e.target.value); setError(''); }}
              className={`${field} flex-1 min-w-0`}
            />
            <button type="submit" disabled={!name.trim() || name.trim() === layout.name} className="rounded-md bg-accent px-2 py-1 text-xs font-semibold text-white disabled:opacity-40">Rename</button>
          </form>
          {error && <div role="alert" className="text-micro text-amber-300">{error}</div>}
          <div className="grid grid-cols-2 gap-1.5">
            <label className="flex flex-col gap-0.5 text-micro text-white/65">
              Hotkey
              <select
                value={layout.hotkey ?? ''}
                aria-label="Hotkey slot"
                onChange={(e) => bindLayoutHotkey(layout.id, e.target.value === '' ? null : Number(e.target.value))}
                className={field}
              >
                <option value="">None</option>
                {HOTKEY_SLOTS.map((n) => <option key={n} value={n}>{slotKeys[n - 1]}</option>)}
              </select>
            </label>
            <label className="flex flex-col gap-0.5 text-micro text-white/65">
              Monitor
              <select value={layout.aspect} aria-label="Monitor shape" onChange={(e) => setLayoutAspect(layout.id, e.target.value)} className={field}>
                {ASPECTS.map((a) => <option key={a} value={a}>{a}</option>)}
                <option value={ASPECT_ANY}>Any</option>
              </select>
            </label>
          </div>
          <div className="flex items-center justify-between gap-2">
            <button
              type="button"
              onClick={() => overwriteCustomLayout(layout.id)}
              title="Replace this layout's panel sizes with the ones on screen now"
              className="text-micro font-medium text-accent hover:text-accent/80"
            >
              Update from current panels
            </button>
            <button
              type="button"
              onClick={() => deleteCustomLayout(layout.id)}
              aria-label={`Delete layout ${layout.name}`}
              className="flex items-center gap-1 text-micro text-white/65 hover:text-red-400"
            >
              <Icon.trash size={12} /> Delete
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default function LayoutSwitcher() {
  const {
    mode,
    activeLayoutId,
    customLayouts,
    applyLayout,
    saveCustomLayout,
    resetPanelSizes,
    panelSizes,
  } = useWorkspaceStore(
    useShallow((s) => ({
      mode: s.workspaceMode,
      activeLayoutId: s.activeLayoutId,
      customLayouts: s.customLayouts,
      applyLayout: s.applyLayout,
      saveCustomLayout: s.saveCustomLayout,
      resetPanelSizes: s.resetPanelSizes,
      panelSizes: s.panelSizes,
    }))
  );
  const overrides = useShortcutStore((s) => s.overrides);

  const win = useWindowSize();
  const hereAspect = classifyAspect(win.w, win.h);
  // What Alt+<n> is called right now, remaps included: the slot badge must say the key that works.
  const slotKeys = useMemo(
    () => LAYOUT_SLOT_ACTION_IDS.map((id, i) => (bindingsOf(id, overrides)[0] ? formatCombo(bindingsOf(id, overrides)[0]) : `slot ${i + 1}`)),
    [overrides],
  );

  const [open, setOpen] = useState(false);
  const [newLayoutName, setNewLayoutName] = useState('');
  const [newAspect, setNewAspect] = useState('');          // '' = this window's shape
  const [isSaving, setIsSaving] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    const handleClickOutside = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setOpen(false);
        setIsSaving(false);
      }
    };
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        setOpen(false);
        setIsSaving(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [open]);

  const handleSelect = (id) => {
    applyLayout(id);
    // Layout selection applies without leaving the current hash route.
    setOpen(false);
  };

  const handleSave = (e) => {
    e.preventDefault();
    if (!newLayoutName.trim()) return;
    saveCustomLayout(newLayoutName.trim(), { aspect: newAspect || hereAspect });
    setNewLayoutName('');
    setNewAspect('');
    setIsSaving(false);
  };

  // Primary 3 shown directly in the quick bar
  const quickPresets = LAYOUT_PRESETS.slice(0, 3);
  const activeCustom = customLayouts.find((c) => c.id === activeLayoutId);
  const ordered = useMemo(() => orderForAspect(customLayouts, hereAspect), [customLayouts, hereAspect]);

  return (
    <div className="relative inline-flex items-center" ref={menuRef}>
      <div
        role="group"
        aria-label="Workspace layout"
        className="flex items-center gap-0.5 rounded-lg border border-white/10 bg-black/25 p-0.5"
      >
        {quickPresets.map((l) => {
          const active = activeLayoutId === l.id || (activeLayoutId === 'standard' && mode === l.mode && !activeCustom);
          return (
            <button
              key={l.id}
              type="button"
              aria-pressed={active}
              title={`${l.label}: ${l.hint}`}
              onClick={() => handleSelect(l.id)}
              className={`rounded-md px-2.5 py-1 text-mini font-semibold whitespace-nowrap transition-colors max-xl:px-2 ${
                active
                  ? 'bg-[var(--accent)]/20 text-white shadow-[inset_0_0_0_1px_color-mix(in_srgb,var(--accent)_45%,transparent)]'
                  : 'text-muted hover:text-white/90 hover:bg-white/[0.06]'
              }`}
            >
              {l.label}
            </button>
          );
        })}

        {activeCustom && (
          <span
            data-testid="active-custom-layout"
            className="rounded-md px-2 py-0.5 text-mini font-semibold text-accent bg-accent/15 border border-accent/30 max-w-[140px] truncate"
          >
            {activeCustom.name}{activeCustom.hotkey ? ` · ${slotKeys[activeCustom.hotkey - 1]}` : ''}
          </span>
        )}

        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-label="More workspace layouts and custom presets"
          title="More layouts & custom presets"
          data-testid="layout-menu-toggle"
          className={`flex items-center gap-1 rounded-md px-1.5 py-1 text-mini font-semibold transition-colors ${
            open ? 'bg-white/10 text-white' : 'text-muted hover:text-white hover:bg-white/[0.06]'
          }`}
        >
          <Icon.down size={13} className={`transition-transform duration-200 ${open ? 'rotate-180' : ''}`} />
        </button>
      </div>

      {/* Dropdown Menu */}
      {open && (
        <div
          data-testid="layout-menu"
          className="absolute right-0 top-full mt-1.5 w-80 max-w-[calc(100vw-1rem)] rounded-xl border border-white/15 bg-neutral-900/95 p-2 shadow-2xl backdrop-blur-xl z-50 animate-in fade-in zoom-in-95 duration-100"
        >
          <div className="px-2 py-1 text-nano uppercase tracking-wider font-bold text-white/65 border-b border-white/10 flex items-center justify-between gap-2">
            <span>Workspace Layouts</span>
            <span className="font-mono text-micro text-accent" data-testid="layout-window-aspect" title={ASPECT_HINT[hereAspect]}>
              {hereAspect} · {win.w}×{win.h}
            </span>
          </div>
          <div className="px-2 pt-1 text-micro font-mono text-white/60">
            Panels now: {panelSizes.leftWidth}px / {panelSizes.rightWidth}px / {panelSizes.bottomHeight}px
          </div>

          <div className="py-1 max-h-[60vh] overflow-y-auto custom-scrollbar space-y-0.5">
            <div className="px-2 pt-1 pb-0.5 text-micro font-semibold text-muted uppercase tracking-wider">
              Workstation Presets
            </div>
            {LAYOUT_PRESETS.map((p) => {
              const active = activeLayoutId === p.id;
              return (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => handleSelect(p.id)}
                  className={`w-full text-left px-2.5 py-1.5 rounded-lg text-xs flex items-center justify-between transition-colors ${
                    active ? 'bg-[var(--accent)]/20 text-white font-semibold' : 'text-white/80 hover:bg-white/10 hover:text-white'
                  }`}
                >
                  <div className="min-w-0 pr-2">
                    <div className="truncate font-medium">{p.label}</div>
                    <div className="text-micro text-white/60 truncate">{p.hint}</div>
                  </div>
                  {active && <Icon.done size={14} className="text-accent shrink-0" />}
                </button>
              );
            })}

            {ordered.length > 0 && (
              <>
                <div className="px-2 pt-2 pb-0.5 text-micro font-semibold text-muted uppercase tracking-wider border-t border-white/5">
                  Saved Layouts · {hereAspect} first
                </div>
                {ordered.map((c) => (
                  <LayoutRow
                    key={c.id}
                    layout={c}
                    active={activeLayoutId === c.id}
                    hereAspect={hereAspect}
                    width={win.w}
                    slotKeys={slotKeys}
                    onApply={applyLayout}
                    onClose={() => setOpen(false)}
                  />
                ))}
              </>
            )}
          </div>

          <div className="pt-2 mt-1 border-t border-white/10 space-y-1.5">
            {isSaving ? (
              <form onSubmit={handleSave} className="flex items-center gap-1 px-1">
                <input
                  type="text"
                  placeholder="Layout name (e.g. Roto 21:9)"
                  aria-label="New layout name"
                  value={newLayoutName}
                  maxLength={40}
                  onChange={(e) => setNewLayoutName(e.target.value)}
                  className={`${field} flex-1 min-w-0 placeholder-white/55`}
                />
                <select value={newAspect || hereAspect} aria-label="Monitor shape for the new layout" onChange={(e) => setNewAspect(e.target.value)} className={field}>
                  {ASPECTS.map((a) => <option key={a} value={a}>{a}</option>)}
                  <option value={ASPECT_ANY}>Any</option>
                </select>
                <button
                  type="submit"
                  disabled={!newLayoutName.trim()}
                  className="rounded-md bg-accent px-2 py-1 text-xs font-semibold text-white hover:bg-accent/80 disabled:opacity-50"
                >
                  Save
                </button>
                <button
                  type="button"
                  onClick={() => setIsSaving(false)}
                  aria-label="Cancel"
                  className="rounded-md px-1.5 py-1 text-xs text-muted hover:text-white"
                >
                  ✕
                </button>
              </form>
            ) : (
              <div className="flex items-center justify-between gap-1 px-1">
                <button
                  type="button"
                  onClick={() => setIsSaving(true)}
                  data-testid="layout-save-open"
                  className="flex items-center gap-1.5 text-xs font-medium text-accent hover:text-accent/80 py-1 transition-colors"
                >
                  <Icon.plus size={13} />
                  <span>Save Current Proportions</span>
                </button>
                <button
                  type="button"
                  onClick={resetPanelSizes}
                  title="Reset panel widths and heights to default"
                  className="text-mini text-white/60 hover:text-white py-1 transition-colors"
                >
                  Reset Sizes
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
