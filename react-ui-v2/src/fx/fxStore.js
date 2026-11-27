// React UI 2.0 — the visual-effects preferences.
//
// Presentation only: nothing here is an engine setting or changes what a control does.
//
// The preferences are PER BROWSER (localStorage), not engine settings: they say how this window
// looks, and the 1.0 client on the same machine must not be affected by them.
import { create } from 'zustand';

const KEY = 'roop_v2_fx';

export const TRANSITIONS = [
  { id: 'depth', label: 'Depth', hint: 'Panels recede into Z and the next one arrives from behind the glass' },
  { id: 'flip', label: 'Card flip', hint: '3D rotate around the vertical axis' },
  { id: 'portal', label: 'Portal', hint: 'Circular iris reveal from the rail' },
  { id: 'glitch', label: 'Glitch', hint: 'RGB-split slice-in, terminal style' },
  { id: 'warp', label: 'Warp', hint: 'Fast stretch along the travel axis' },
  { id: 'none', label: 'None', hint: 'Instant cut' },
];

export const AMBIENT_QUALITY = [
  { id: 'low', label: 'Low', scale: 0.35, fps: 24 },
  { id: 'balanced', label: 'Balanced', scale: 0.5, fps: 30 },
  { id: 'high', label: 'High', scale: 0.75, fps: 60 },
];

export const PALETTES = [
  { id: 'magenta', label: 'Magenta', a: '#f84fcc', b: '#00dbe4' },
  { id: 'cyan', label: 'Cyan', a: '#00dbe4', b: '#9ff4ff' },
  { id: 'ember', label: 'Ember', a: '#ff582d', b: '#fea92f' },
  { id: 'aurora', label: 'Aurora', a: '#51e57e', b: '#4cd7c8' },
  { id: 'mono', label: 'Mono', a: '#e4e4e7', b: '#a1a1aa' },
];

export const DEFAULT_FX = {
  ambient: true,          // the WebGL horizon behind the app
  ambientQuality: 'balanced',
  ambientDuringRender: false, // keep animating while a render runs (off: freeze to one still frame)
  parallax: true,         // the horizon follows the pointer
  transition: 'depth',
  terminal: true,         // the terminal dock exists at all
  terminalOpen: false,    // ...and is expanded
  terminalHeight: 260,
  typewriter: true,       // new terminal lines type themselves out
  crt: false,             // scanlines + vignette over the terminal
  grain: true,            // film grain over the whole page
  palette: 'magenta',     // accent family: magenta, cyan, ember, aurora, mono
  boot: true,             // the boot sequence on the first load of a session
  sunray: true,           // the light sweep across titles
};

function load() {
  try {
    const raw = JSON.parse(window.localStorage.getItem(KEY) || 'null');
    return raw && typeof raw === 'object' ? { ...DEFAULT_FX, ...raw } : { ...DEFAULT_FX };
  } catch {
    return { ...DEFAULT_FX };
  }
}

function save(state) {
  try {
    const out = {};
    Object.keys(DEFAULT_FX).forEach((k) => { out[k] = state[k]; });
    window.localStorage.setItem(KEY, JSON.stringify(out));
  } catch { /* private window / blocked storage: the defaults still render */ }
}

export const useFxStore = create((set, get) => ({
  ...load(),
  setFx: (patch) => { set(patch); save(get()); },
  toggleFx: (key) => { set({ [key]: !get()[key] }); save(get()); },
  resetFx: () => { set({ ...DEFAULT_FX }); save(get()); },
}));

/** The OS asked for less motion. Effects that MOVE honour it; static styling does not change. */
export const prefersReducedMotion = () => {
  try { return window.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch { return false; }
};
