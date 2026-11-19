import { create } from 'zustand';
import {
  ASPECTS, assignHotkey, classifyAspect, layoutForSlot, newLayoutId, normalizeLayouts, renameLayout, scaleSizes, setAspect,
} from '../workspace/layoutModel';

// ── Workstation chrome state ────────────────────────────────────────────────
//
// Saved under roop_workspace: navigation width, selected layout, pane sizes and
// custom presets. Cinema and timeline retain the bottom deck with icon rails.
// A fresh wide display selects dual stage; an explicit saved choice wins.
export const WORKSPACE_LAYOUT = {
  default: { left: true, right: true, bottom: true },
  cinema: { left: false, right: false, bottom: true },
  dual: { left: true, right: true, bottom: true },
  timeline: { left: false, right: false, bottom: true },
};

export const WORKSPACE_MODES = Object.keys(WORKSPACE_LAYOUT);

export const DEFAULT_PANEL_SIZES = {
  leftWidth: 260,
  rightWidth: 340,
  bottomHeight: 260,
  queueWidth: 280,
};

export const PANEL_LIMITS = {
  leftWidth: { min: 240, max: 620 },
  rightWidth: { min: 240, max: 580 },
  bottomHeight: { min: 140, max: 600 },
  queueWidth: { min: 220, max: 500 },
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
  { id: 'standard', mode: 'default', label: 'Standard NLE', hint: 'Assets 260 · fluid viewport · inspector 340 · timeline 260', sizes: DEFAULT_PANEL_SIZES },
  { id: 'canvas', mode: 'cinema', label: 'Cinema Preview', hint: 'Collapsed icon rails with viewport and timeline', sizes: DEFAULT_PANEL_SIZES },
  { id: 'ultrawide', mode: 'dual', label: 'Ultra-Wide Dual Stage', hint: 'Source / face bank, dual A/B canvas, inspector and live queue', sizes: DEFAULT_PANEL_SIZES },
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
  queueWidth: clampPanelSize('queueWidth', saved.panelSizes?.queueWidth ?? DEFAULT_PANEL_SIZES.queueWidth),
};

// Stored layouts from before Stage 15 have no aspect / hotkey / viewport: normalizeLayouts gives them 'any' / null / null.
const initialCustom = normalizeLayouts(saved.customLayouts, clampPanelSize);

const currentViewport = () => (typeof window !== 'undefined' && window.innerWidth > 0
  ? { w: Math.round(window.innerWidth), h: Math.round(window.innerHeight) } : null);

export const useWorkspaceStore = create((set, get) => ({
  workspaceMode: WORKSPACE_MODES.includes(saved.workspaceMode) ? saved.workspaceMode : (typeof window !== 'undefined' && window.innerWidth > 2200 ? 'dual' : 'default'),
  railCollapsed: typeof saved.railCollapsed === 'boolean' ? saved.railCollapsed : true,
  railChosen: typeof saved.railCollapsed === 'boolean',
  settingsTab: SETTINGS_TABS.includes(saved.settingsTab) ? saved.settingsTab : 'providers',
  panelSizes: initialSizes,
  customLayouts: initialCustom,
  activeLayoutId: saved.activeLayoutId || (typeof window !== 'undefined' && window.innerWidth > 2200 ? 'ultrawide' : 'standard'),
  layoutEpoch: 0,

  setWorkspaceMode: (mode) => {
    if (!WORKSPACE_MODES.includes(mode) || mode === get().workspaceMode) return;
    set({ workspaceMode: mode, activeLayoutId: { default: 'standard', cinema: 'canvas', dual: 'ultrawide', timeline: 'inspect' }[mode], layoutEpoch: get().layoutEpoch + 1 });
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

  // Cinema mode (Tab): the Canvas layout with the side rails collapsed. Toggling again returns to the layout and rail
  // state it came from. Not persisted, so a reload leaves cinema mode (the saved layout is the one to come back to).
  cinemaReturn: null,
  toggleCinema: () => {
    const s = get();
    if (s.cinemaReturn) {
      const back = s.cinemaReturn;
      set({ cinemaReturn: null, railCollapsed: back.railCollapsed, railChosen: true });
      get().applyLayout(back.layoutId);
      return;
    }
    set({ cinemaReturn: { layoutId: s.activeLayoutId, railCollapsed: s.railCollapsed } });
    get().applyLayout('canvas');
    set({ railCollapsed: true, railChosen: true });
  },

  applyLayout: (layoutId) => {
    const preset = LAYOUT_PRESETS.find((l) => l.id === layoutId);
    const custom = get().customLayouts.find((l) => l.id === layoutId);
    const layout = preset || custom;
    if (!layout) return;

    const nextMode = WORKSPACE_MODES.includes(layout.mode) ? layout.mode : get().workspaceMode;
    // A saved layout carries the window it was made in: on another monitor of the same shape its panels keep their
    // PROPORTIONS (a 3440-wide layout on a 2560-wide screen), not their pixels. Presets have no viewport and apply as written.
    const here = currentViewport();
    const source = !preset && layout.viewport && here ? scaleSizes(layout.sizes, layout.viewport, here, clampPanelSize) : layout.sizes;
    const nextSizes = source
      ? {
          leftWidth: clampPanelSize('leftWidth', source.leftWidth ?? DEFAULT_PANEL_SIZES.leftWidth),
          rightWidth: clampPanelSize('rightWidth', source.rightWidth ?? DEFAULT_PANEL_SIZES.rightWidth),
          bottomHeight: clampPanelSize('bottomHeight', source.bottomHeight ?? DEFAULT_PANEL_SIZES.bottomHeight),
          queueWidth: clampPanelSize('queueWidth', source.queueWidth ?? DEFAULT_PANEL_SIZES.queueWidth),
        }
      : get().panelSizes;

    set({
      workspaceMode: nextMode,
      panelSizes: nextSizes,
      activeLayoutId: layoutId,
      layoutEpoch: get().layoutEpoch + 1,
    });
    persist(get());
  },

  // Save the CURRENT proportions as a named layout. `aspect` is the monitor shape it is for (default: the one this window
  // is); `hotkey` an Alt+<1..5> slot. Saving under a name that exists replaces that layout and keeps its slot.
  saveCustomLayout: (name, { aspect, hotkey } = {}) => {
    const trimmed = String(name || '').trim().slice(0, 40);
    if (!trimmed) return null;
    const here = currentViewport();
    const previous = get().customLayouts.find((l) => l.name.toLowerCase() === trimmed.toLowerCase());
    const id = previous?.id || newLayoutId();
    const newLayout = {
      id,
      name: trimmed,
      label: trimmed,
      mode: get().workspaceMode,
      sizes: { ...get().panelSizes },
      aspect: ASPECTS.includes(aspect) ? aspect : (previous?.aspect && previous.aspect !== 'any' ? previous.aspect : (here ? classifyAspect(here.w, here.h) : 'any')),
      hotkey: previous?.hotkey ?? null,
      viewport: here,
    };
    let next = [...get().customLayouts.filter((l) => l.id !== id), newLayout];
    if (hotkey !== undefined) next = assignHotkey(next, id, hotkey);
    set({ customLayouts: next, activeLayoutId: id });
    persist(get());
    return next.find((l) => l.id === id);
  },

  /** Replace a saved layout's proportions with the ones on screen now (keeps its name, slot and aspect). */
  overwriteCustomLayout: (layoutId) => {
    const here = currentViewport();
    const next = get().customLayouts.map((l) => (l.id === layoutId
      ? { ...l, mode: get().workspaceMode, sizes: { ...get().panelSizes }, viewport: here }
      : l));
    set({ customLayouts: next, activeLayoutId: layoutId });
    persist(get());
  },

  /** { ok: true } or { ok: false, error } - an empty name or another layout's name is refused. */
  renameCustomLayout: (layoutId, name) => {
    const r = renameLayout(get().customLayouts, layoutId, name);
    if (!r.ok) return r;
    set({ customLayouts: r.layouts });
    persist(get());
    return { ok: true };
  },

  /** Bind Alt+<slot> (1..5) to a saved layout; null unbinds. The slot moves off whichever layout had it. */
  bindLayoutHotkey: (layoutId, slot) => {
    const next = assignHotkey(get().customLayouts, layoutId, slot);
    if (next === get().customLayouts) return;
    set({ customLayouts: next });
    persist(get());
  },

  setLayoutAspect: (layoutId, aspect) => {
    const next = setAspect(get().customLayouts, layoutId, aspect);
    if (next === get().customLayouts) return;
    set({ customLayouts: next });
    persist(get());
  },

  /** Alt+<slot>: switch to the layout bound to it. Returns that layout, or null when the slot is empty. */
  applyLayoutSlot: (slot) => {
    const layout = layoutForSlot(get().customLayouts, slot);
    if (!layout) return null;
    get().applyLayout(layout.id);
    return layout;
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
