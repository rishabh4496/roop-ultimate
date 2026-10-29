import { create } from 'zustand';

// ── Workstation chrome state ────────────────────────────────────────────────
//
// Persistent workstation preferences remembered across tabs and reloads:
//   workspaceMode   which panels the Face Swap workspace shows (default, cinema, dual, timeline)
//   railCollapsed   the left navigation rail: icon + label, or icon only
//   panelSizes      resizable dimensions for Left Inspector, Right Media Rail, Bottom Timeline
//   customLayouts   user-saved named workspace layouts with custom proportions
//   activeLayoutId  id of the active preset or custom layout
//
// Persisted to localStorage under KEY = 'roop_workspace'.

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

export const DEFAULT_PANEL_SIZES = {
  leftWidth: 350,
  rightWidth: 330,
  bottomHeight: 280,
};

export const PANEL_LIMITS = {
  leftWidth: { min: 240, max: 620 },
  rightWidth: { min: 240, max: 580 },
  bottomHeight: { min: 140, max: 600 },
};

export const clampPanelSize = (panel, val) => {
  const limits = PANEL_LIMITS[panel];
  if (!limits) return val;
  const num = Number(val);
  if (!Number.isFinite(num)) return limits.min;
  return Math.max(limits.min, Math.min(limits.max, Math.round(num)));
};

/** Built-in workspace presets including specialized workstation profiles. */
export const LAYOUT_PRESETS = [
  { id: 'standard', mode: 'default', label: 'Standard', hint: 'Faces, settings, preview and timeline', sizes: { leftWidth: 350, rightWidth: 330, bottomHeight: 280 } },
  { id: 'canvas', mode: 'cinema', label: 'Max Canvas', hint: 'Preview only: every panel hidden', sizes: { leftWidth: 350, rightWidth: 330, bottomHeight: 280 } },
  { id: 'inspect', mode: 'timeline', label: 'Inspection Suite', hint: 'Preview and timeline for frame-by-frame inspection', sizes: { leftWidth: 350, rightWidth: 330, bottomHeight: 380 } },
  { id: 'roto', mode: 'default', label: 'My Roto Setup', hint: 'Expanded Inspector & high timeline for precision roto', sizes: { leftWidth: 440, rightWidth: 300, bottomHeight: 360 } },
  { id: 'dual_color', mode: 'dual', label: 'Dual-Monitor Color', hint: 'Wide inspector and faceset rail for grade/color', sizes: { leftWidth: 460, rightWidth: 400, bottomHeight: 220 } },
  { id: 'laptop', mode: 'default', label: 'Minimal Laptop', hint: 'Compact rails maximizing central stage on 1080p', sizes: { leftWidth: 280, rightWidth: 270, bottomHeight: 200 } },
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
      railCollapsed: state.railChosen ? state.railCollapsed : undefined,
      settingsTab: state.settingsTab,
      panelSizes: state.panelSizes,
      customLayouts: state.customLayouts,
      activeLayoutId: state.activeLayoutId,
    }));
  } catch { /* storage unavailable */ }
};

const initialSizes = {
  leftWidth: clampPanelSize('leftWidth', saved.panelSizes?.leftWidth ?? DEFAULT_PANEL_SIZES.leftWidth),
  rightWidth: clampPanelSize('rightWidth', saved.panelSizes?.rightWidth ?? DEFAULT_PANEL_SIZES.rightWidth),
  bottomHeight: clampPanelSize('bottomHeight', saved.panelSizes?.bottomHeight ?? DEFAULT_PANEL_SIZES.bottomHeight),
};

const initialCustom = Array.isArray(saved.customLayouts) ? saved.customLayouts : [];

export const useWorkspaceStore = create((set, get) => ({
  workspaceMode: WORKSPACE_MODES.includes(saved.workspaceMode) ? saved.workspaceMode : 'default',
  railCollapsed: typeof saved.railCollapsed === 'boolean' ? saved.railCollapsed : autoCollapsed(),
  railChosen: typeof saved.railCollapsed === 'boolean',
  settingsTab: SETTINGS_TABS.includes(saved.settingsTab) ? saved.settingsTab : 'providers',
  panelSizes: initialSizes,
  customLayouts: initialCustom,
  activeLayoutId: saved.activeLayoutId || 'standard',

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

  setPanelSize: (panel, val) => {
    const clamped = clampPanelSize(panel, val);
    const cur = get().panelSizes;
    if (cur[panel] === clamped) return;
    const next = { ...cur, [panel]: clamped };
    set({ panelSizes: next });
    persist(get());
  },

  applyLayout: (layoutId) => {
    const preset = LAYOUT_PRESETS.find((l) => l.id === layoutId);
    const custom = get().customLayouts.find((l) => l.id === layoutId);
    const layout = preset || custom;
    if (!layout) return;

    const nextMode = WORKSPACE_MODES.includes(layout.mode) ? layout.mode : get().workspaceMode;
    const nextSizes = layout.sizes
      ? {
          leftWidth: clampPanelSize('leftWidth', layout.sizes.leftWidth ?? DEFAULT_PANEL_SIZES.leftWidth),
          rightWidth: clampPanelSize('rightWidth', layout.sizes.rightWidth ?? DEFAULT_PANEL_SIZES.rightWidth),
          bottomHeight: clampPanelSize('bottomHeight', layout.sizes.bottomHeight ?? DEFAULT_PANEL_SIZES.bottomHeight),
        }
      : get().panelSizes;

    set({
      workspaceMode: nextMode,
      panelSizes: nextSizes,
      activeLayoutId: layoutId,
    });
    persist(get());
  },

  saveCustomLayout: (name) => {
    const trimmed = String(name || '').trim();
    if (!trimmed) return null;
    const id = `custom_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    const newLayout = {
      id,
      name: trimmed,
      label: trimmed,
      mode: get().workspaceMode,
      sizes: { ...get().panelSizes },
    };
    const next = [...get().customLayouts.filter((l) => l.name.toLowerCase() !== trimmed.toLowerCase()), newLayout];
    set({
      customLayouts: next,
      activeLayoutId: id,
    });
    persist(get());
    return newLayout;
  },

  deleteCustomLayout: (layoutId) => {
    const next = get().customLayouts.filter((l) => l.id !== layoutId);
    const active = get().activeLayoutId === layoutId ? 'standard' : get().activeLayoutId;
    set({ customLayouts: next, activeLayoutId: active });
    persist(get());
  },

  resetPanelSizes: () => {
    set({ panelSizes: { ...DEFAULT_PANEL_SIZES } });
    persist(get());
  },
}));
