import { create } from 'zustand';

const STORAGE_KEY = 'roop_studio_workspace_layout_v1';

export const WORKSPACE_PRESETS = {
  default: {
    label: 'Studio Default',
    description: 'Balanced arrangement with Preview, Timeline, Face Bank, and Telemetry HUD.',
    panels: { preview: true, timeline: true, facebank: true, queue: false, telemetry: true },
  },
  cinema: {
    label: 'Cinematic Focus',
    description: 'Full-canvas Preview and precision Timeline for maximum visual inspection.',
    panels: { preview: true, timeline: true, facebank: false, queue: false, telemetry: false },
  },
  facebank: {
    label: 'Actor Routing Hub',
    description: 'Face Bank cluster tray and Preview side-by-side for rapid identity mapping.',
    panels: { preview: true, timeline: false, facebank: true, queue: false, telemetry: false },
  },
  grading: {
    label: 'Color Grading (Rec.709)',
    description: 'Neutral gray reference borders, zero optical tint, with Telemetry diagnostics.',
    panels: { preview: true, timeline: true, facebank: false, queue: false, telemetry: true },
  },
  batch: {
    label: 'Batch Render Deck',
    description: 'Render Queue drawer expanded with real-time throughput metrics.',
    panels: { preview: true, timeline: false, facebank: false, queue: true, telemetry: true },
  },
};

const DEFAULT_STATE = {
  activePreset: 'default',
  panels: {
    preview: true,
    timeline: true,
    facebank: true,
    queue: false,
    telemetry: true,
  },
  panelSizes: {
    facebankWidth: 360,
    timelineHeight: 220,
    queueHeight: 280,
  },
  studioTheme: 'obsidian', // 'obsidian' | 'rec709_neutral'
};

function loadStoredState() {
  if (typeof window === 'undefined') return DEFAULT_STATE;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_STATE;
    const parsed = JSON.parse(raw);
    return {
      activePreset: parsed.activePreset in WORKSPACE_PRESETS ? parsed.activePreset : 'default',
      panels: { ...DEFAULT_STATE.panels, ...(parsed.panels || {}) },
      panelSizes: { ...DEFAULT_STATE.panelSizes, ...(parsed.panelSizes || {}) },
      studioTheme: parsed.studioTheme === 'rec709_neutral' ? 'rec709_neutral' : 'obsidian',
    };
  } catch {
    return DEFAULT_STATE;
  }
}

function persistState(state) {
  if (typeof window === 'undefined') return;
  try {
    const toSave = {
      activePreset: state.activePreset,
      panels: state.panels,
      panelSizes: state.panelSizes,
      studioTheme: state.studioTheme,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave));
  } catch {
    // Ignored
  }
}

export const useWorkspaceLayoutStore = create((set) => ({
  ...loadStoredState(),

  setPreset: (presetName) => {
    const preset = WORKSPACE_PRESETS[presetName];
    if (!preset) return;
    set((state) => {
      const isGrading = presetName === 'grading';
      const next = {
        ...state,
        activePreset: presetName,
        panels: { ...preset.panels },
        studioTheme: isGrading ? 'rec709_neutral' : state.studioTheme,
      };
      persistState(next);
      return next;
    });
  },

  togglePanel: (panelKey) => {
    set((state) => {
      const current = !!state.panels[panelKey];
      const nextPanels = { ...state.panels, [panelKey]: !current };
      const next = {
        ...state,
        activePreset: 'custom',
        panels: nextPanels,
      };
      persistState(next);
      return next;
    });
  },

  setPanelVisible: (panelKey, visible) => {
    set((state) => {
      const next = {
        ...state,
        panels: { ...state.panels, [panelKey]: !!visible },
      };
      persistState(next);
      return next;
    });
  },

  setPanelSize: (panelKey, size) => {
    set((state) => {
      const next = {
        ...state,
        panelSizes: { ...state.panelSizes, [panelKey]: Math.max(120, Number(size) || 120) },
      };
      persistState(next);
      return next;
    });
  },

  setStudioTheme: (theme) => {
    set((state) => {
      const next = {
        ...state,
        studioTheme: theme === 'rec709_neutral' ? 'rec709_neutral' : 'obsidian',
      };
      persistState(next);
      return next;
    });
  },

  resetLayout: () => {
    set(() => {
      persistState(DEFAULT_STATE);
      return { ...DEFAULT_STATE };
    });
  },
}));
