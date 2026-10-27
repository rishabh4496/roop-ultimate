import { useShallow } from 'zustand/react/shallow';
import { useState } from 'react';
import { WORKSPACE_LAYOUT, WORKSPACE_MODES, useWorkspaceStore } from '../../store/workspaceStore';

// ── What each workspace mode actually does ────────────────────────────────
// The dock offered four modes but only 'cinema' changed anything; 'dual' and
// 'timeline' were menu entries that did nothing at all, and the dock's third
// drawer button toggled a `bottom` flag nothing read. Each mode is one row of
// the table in store/workspaceStore.js and every panel reads its visibility
// from here, so a mode cannot quietly become decorative again.
//
// The MODE itself lives in that store, not in this hook: the header's layout
// switcher sets it from outside this tab, and it has to survive leaving Face
// Swap (it used to reset to 'default' on every tab change).
export { WORKSPACE_MODES };

export default function useWorkspaceLayout() {
  const { workspaceMode, setWorkspaceMode } = useWorkspaceStore(
    useShallow((s) => ({ workspaceMode: s.workspaceMode, setWorkspaceMode: s.setWorkspaceMode }))
  );
  const [ambilightEnabled, setAmbilightEnabled] = useState(true);
  const [drawers, setDrawers] = useState({ left: true, right: true, bottom: true });

  const layout = WORKSPACE_LAYOUT[workspaceMode] || WORKSPACE_LAYOUT.default;

  // The dock's drawer buttons stay authoritative: a mode sets the baseline, and
  // closing a drawer by hand still closes it.
  return {
    workspaceMode, setWorkspaceMode,
    ambilightEnabled, setAmbilightEnabled,
    drawers, setDrawers,
    showLeftPanel: layout.left && drawers.left,
    showRightPanel: layout.right && drawers.right,
    showTimelineDeck: layout.bottom && drawers.bottom,
  };
}
