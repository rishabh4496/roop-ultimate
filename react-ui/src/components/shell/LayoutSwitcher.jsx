import { useShallow } from 'zustand/react/shallow';
import React from 'react';
import { LAYOUT_PRESETS, useWorkspaceStore } from '../../store/workspaceStore';

// The workspace layout switcher: Standard / Max Canvas / Inspection Suite.
//
// These are names over the SAME mode table the Face Swap dock's menu uses
// (store/workspaceStore.js WORKSPACE_LAYOUT), not a second layout system, so a
// choice here is what the dock shows and vice versa. The layout only exists on
// Face Swap, so picking one from another tab also goes there — a switch that
// quietly changes nothing visible would read as broken.
//
// A mode the switcher does not name (the dock's "dual") leaves all three
// unselected rather than lying about which is active.
export default function LayoutSwitcher({ tab, onNavigate }) {
  const { mode, setMode } = useWorkspaceStore(
    useShallow((s) => ({ mode: s.workspaceMode, setMode: s.setWorkspaceMode }))
  );

  return (
    <div
      role="group"
      aria-label="Workspace layout"
      className="flex items-center gap-0.5 rounded-lg border border-white/10 bg-black/25 p-0.5"
    >
      {LAYOUT_PRESETS.map((l) => {
        const active = mode === l.mode;
        return (
          <button
            key={l.id}
            type="button"
            aria-pressed={active}
            title={`${l.label}: ${l.hint}`}
            onClick={() => {
              setMode(l.mode);
              if (tab !== 'faceswap') onNavigate('faceswap');
            }}
            className={`rounded-md px-2.5 py-1 text-mini font-semibold whitespace-nowrap transition-colors max-xl:px-2 ${
              active ? 'bg-[var(--accent)]/20 text-white shadow-[inset_0_0_0_1px_color-mix(in_srgb,var(--accent)_45%,transparent)]' : 'text-muted hover:text-white/90 hover:bg-white/[0.06]'
            }`}
          >
            {l.label}
          </button>
        );
      })}
    </div>
  );
}
