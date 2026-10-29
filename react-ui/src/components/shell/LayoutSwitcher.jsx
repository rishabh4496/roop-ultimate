import { useShallow } from 'zustand/react/shallow';
import React, { useState, useRef, useEffect } from 'react';
import { LAYOUT_PRESETS, useWorkspaceStore } from '../../store/workspaceStore';
import { Icon } from '../../icons';

// The workstation workspace layout switcher & custom layout manager.
//
// Allows switching between built-in presets (Standard, Max Canvas, Inspection Suite,
// My Roto Setup, Dual-Monitor Color, Minimal Laptop) and saving custom user layouts
// with persisted panel proportions (Left Media Rail, Right Inspector, Bottom Timeline).
export default function LayoutSwitcher({ tab, onNavigate }) {
  const {
    mode,
    activeLayoutId,
    customLayouts,
    applyLayout,
    saveCustomLayout,
    deleteCustomLayout,
    resetPanelSizes,
    panelSizes,
  } = useWorkspaceStore(
    useShallow((s) => ({
      mode: s.workspaceMode,
      activeLayoutId: s.activeLayoutId,
      customLayouts: s.customLayouts,
      applyLayout: s.applyLayout,
      saveCustomLayout: s.saveCustomLayout,
      deleteCustomLayout: s.deleteCustomLayout,
      resetPanelSizes: s.resetPanelSizes,
      panelSizes: s.panelSizes,
    }))
  );

  const [open, setOpen] = useState(false);
  const [newLayoutName, setNewLayoutName] = useState('');
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
    if (tab !== 'faceswap') onNavigate('faceswap');
    setOpen(false);
  };

  const handleSave = (e) => {
    e.preventDefault();
    if (!newLayoutName.trim()) return;
    saveCustomLayout(newLayoutName.trim());
    setNewLayoutName('');
    setIsSaving(false);
  };

  // Primary 3 shown directly in the quick bar
  const quickPresets = LAYOUT_PRESETS.slice(0, 3);
  const activeCustom = customLayouts.find((c) => c.id === activeLayoutId);

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
          <span className="rounded-md px-2 py-0.5 text-mini font-semibold text-accent bg-accent/15 border border-accent/30 max-w-[120px] truncate">
            {activeCustom.name}
          </span>
        )}

        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-label="More workspace layouts and custom presets"
          title="More layouts & custom presets"
          className={`flex items-center gap-1 rounded-md px-1.5 py-1 text-mini font-semibold transition-colors ${
            open ? 'bg-white/10 text-white' : 'text-muted hover:text-white hover:bg-white/[0.06]'
          }`}
        >
          <Icon.down size={13} className={`transition-transform duration-200 ${open ? 'rotate-180' : ''}`} />
        </button>
      </div>

      {/* Dropdown Menu */}
      {open && (
        <div className="absolute right-0 top-full mt-1.5 w-72 rounded-xl border border-white/15 bg-neutral-900/95 p-2 shadow-2xl backdrop-blur-xl z-50 animate-in fade-in zoom-in-95 duration-100">
          <div className="px-2 py-1 text-nano uppercase tracking-wider font-bold text-white/50 border-b border-white/10 flex items-center justify-between">
            <span>Workspace Layouts</span>
            <span className="font-mono text-micro text-accent">
              {panelSizes.leftWidth}px / {panelSizes.rightWidth}px / {panelSizes.bottomHeight}px
            </span>
          </div>

          <div className="py-1 max-h-64 overflow-y-auto custom-scrollbar space-y-0.5">
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
                    <div className="text-micro text-white/40 truncate">{p.hint}</div>
                  </div>
                  {active && <Icon.done size={14} className="text-accent shrink-0" />}
                </button>
              );
            })}

            {customLayouts.length > 0 && (
              <>
                <div className="px-2 pt-2 pb-0.5 text-micro font-semibold text-muted uppercase tracking-wider border-t border-white/5">
                  Saved Custom Layouts
                </div>
                {customLayouts.map((c) => {
                  const active = activeLayoutId === c.id;
                  return (
                    <div
                      key={c.id}
                      className={`group w-full px-2.5 py-1.5 rounded-lg text-xs flex items-center justify-between transition-colors ${
                        active ? 'bg-[var(--accent)]/20 text-white font-semibold' : 'text-white/80 hover:bg-white/10'
                      }`}
                    >
                      <button
                        type="button"
                        onClick={() => handleSelect(c.id)}
                        className="flex-1 text-left min-w-0 pr-1 truncate"
                      >
                        <div className="truncate font-medium">{c.name}</div>
                        <div className="text-micro text-white/40 truncate font-mono">
                          {c.sizes.leftWidth}px · {c.sizes.rightWidth}px · {c.sizes.bottomHeight}px
                        </div>
                      </button>
                      <div className="flex items-center gap-1 shrink-0">
                        {active && <Icon.done size={14} className="text-accent" />}
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteCustomLayout(c.id);
                          }}
                          aria-label="Delete custom layout" title="Delete custom layout"
                          className="opacity-0 group-hover:opacity-100 hover:text-red-400 p-0.5 rounded transition-opacity"
                        >
                          <Icon.trash size={12} />
                        </button>
                      </div>
                    </div>
                  );
                })}
              </>
            )}
          </div>

          <div className="pt-2 mt-1 border-t border-white/10 space-y-1.5">
            {isSaving ? (
              <form onSubmit={handleSave} className="flex items-center gap-1 px-1">
                <input
                  type="text"
                  placeholder="Layout name (e.g. My Roto Setup)"
                  value={newLayoutName}
                  onChange={(e) => setNewLayoutName(e.target.value)}
                  className="flex-1 rounded-md bg-black/60 border border-white/20 px-2 py-1 text-xs text-white placeholder-white/40 focus:outline-none focus:border-accent"
                />
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
                  className="flex items-center gap-1.5 text-xs font-medium text-accent hover:text-accent/80 py-1 transition-colors"
                >
                  <Icon.plus size={13} />
                  <span>Save Current Proportions</span>
                </button>
                <button
                  type="button"
                  onClick={resetPanelSizes}
                  title="Reset panel widths and heights to default"
                  className="text-mini text-white/40 hover:text-white/80 py-1 transition-colors"
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
