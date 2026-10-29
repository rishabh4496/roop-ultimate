import { create } from 'zustand';

// ── Workstation chrome state ────────────────────────────────────────────────
//
// Two things the shell remembers across tabs and reloads:
//
//   workspaceMode   which panels the Face Swap workspace shows. This is the SAME
//                   table the dock's mode menu used (faceswap/useWorkspaceLayout
//                   used to hold it in component state, so it reset to "default"
//                   on every tab change). The header's layout switcher and the
//                   dock now both read and write this one value.
//   railCollapsed   the left navigation rail: icon + label, or icon only.
//
// Persisted to localStorage on a try/catch: a private window, blocked site data
// or a throwing accessor must leave the app working with in-memory defaults.

//            left faces   right settings   timeline deck
//  default       yes            yes              yes
//  cinema        no             no               no       (all canvas)
//  dual          yes            yes              no       (faces + params)
//  timeline      no             no               yes      (precision scrub)
export const WORKSPACE_LAYOUT = {
  default: { left: true, right: true, bottom: true },
  cinema: { left: false, right: false, bottom: false },
  dual: { left: true, right: true, bottom: false },
  timeline: { left: false, right: false, bottom: true },
};

export const WORKSPACE_MODES = Object.keys(WORKSPACE_LAYOUT);

/** What the header switcher offers: three names over the table's modes. `dual` stays a dock-only mode. */
export const LAYOUT_PRESETS = [
  { id: 'standard', mode: 'default', label: 'Standard', hint: 'Faces, settings, preview and timeline' },
  { id: 'canvas', mode: 'cinema', label: 'Max Canvas', hint: 'Preview only: every panel hidden' },
  { id: 'inspect', mode: 'timeline', label: 'Inspection Suite', hint: 'Preview and timeline for frame-by-frame inspection' },
];

/** The Settings tab's sections. Held here so the palette and the telemetry popover can open one directly. */
export const SETTINGS_TABS = ['providers', 'cpu', 'paths', 'diagnostics', 'general', 'shortcuts'];

const KEY = 'roop_workspace';

const readSaved = () => {
  try {
    const v = JSON.parse(window.localStorage.getItem(KEY) || '{}');
    return v && typeof v === 'object' ? v : {};
  } catch {
    return {};
  }
};

const saved = readSaved();

// Below this LAYOUT width the rail starts icon-only until the user says otherwise. Face Swap is three columns
// (inspector 350 + media 330 + the stage); at 1280 a 200px rail left the stage 282px wide, narrow enough that
// its own HUD buttons wrapped across the picture and covered the centre of it. Layout width, not window width:
// UI zoom is CSS `zoom` on <html>, which media queries cannot see (the same reason the old header divided by it).
export const RAIL_AUTO_COLLAPSE_BELOW = 1400;
const autoCollapsed = () => {
  try {
    const zoom = parseFloat(window.localStorage.getItem('roop_zoom')) || 1;
    return window.innerWidth / zoom < RAIL_AUTO_COLLAPSE_BELOW;
  } catch {
    return false;
  }
};

const persist = (state) => {
  try {
    window.localStorage.setItem(KEY, JSON.stringify({
      workspaceMode: state.workspaceMode,
      // Only an explicit choice is stored: writing the width-derived default here (as a side effect of
      // changing the layout or the Settings tab) would turn it into a choice the user never made.
      railCollapsed: state.railChosen ? state.railCollapsed : undefined,
      settingsTab: state.settingsTab,
    }));
  } catch { /* storage unavailable: the choice lasts for this session only */ }
};

export const useWorkspaceStore = create((set, get) => ({
  workspaceMode: WORKSPACE_MODES.includes(saved.workspaceMode) ? saved.workspaceMode : 'default',
  // An explicit choice (saved) wins; with none, the window's width decides.
  railCollapsed: typeof saved.railCollapsed === 'boolean' ? saved.railCollapsed : autoCollapsed(),
  railChosen: typeof saved.railCollapsed === 'boolean',
  settingsTab: SETTINGS_TABS.includes(saved.settingsTab) ? saved.settingsTab : 'providers',

  setWorkspaceMode: (mode) => {
    if (!WORKSPACE_MODES.includes(mode) || mode === get().workspaceMode) return;
    set({ workspaceMode: mode });
    persist(get());
  },
  setRailCollapsed: (collapsed) => {
    set({ railCollapsed: !!collapsed, railChosen: true });
    persist(get());
  },
  setSettingsTab: (tab) => {
    if (!SETTINGS_TABS.includes(tab) || tab === get().settingsTab) return;
    set({ settingsTab: tab });
    persist(get());
  },
  toggleRail: () => {
    set({ railCollapsed: !get().railCollapsed, railChosen: true });
    persist(get());
  },
}));
