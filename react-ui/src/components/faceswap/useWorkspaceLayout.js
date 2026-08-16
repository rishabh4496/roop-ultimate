import { useState } from 'react';

// ── What each workspace mode actually does ────────────────────────────────
// The dock offered four modes but only 'cinema' changed anything; 'dual' and
// 'timeline' were menu entries that did nothing at all, and the dock's third
// drawer button toggled a `bottom` flag nothing read. Each mode is one row of
// this table and every panel reads its visibility from here, so a mode cannot
// quietly become decorative again.
//
//            left faces   right settings   timeline deck
//  default       yes            yes              yes
//  cinema        no             no               no       (all canvas)
//  dual          yes            yes              no       (faces + params)
//  timeline      no             no               yes      (precision scrub)
const WORKSPACE_LAYOUT = {
  default: { left: true, right: true, bottom: true },
  cinema: { left: false, right: false, bottom: false },
  dual: { left: true, right: true, bottom: false },
  timeline: { left: false, right: false, bottom: true },
};

export const WORKSPACE_MODES = Object.keys(WORKSPACE_LAYOUT);

export default function useWorkspaceLayout() {
  const [workspaceMode, setWorkspaceMode] = useState('default');
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
