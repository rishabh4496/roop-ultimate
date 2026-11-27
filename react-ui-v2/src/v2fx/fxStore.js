// React UI 2.0 — the visual-effects preferences.
//
// Everything in src/v2fx/ is presentation only: no component here talks to the backend, reads a
// setting the engine uses, or changes what a control does. That is the rule that keeps 2.0 a
// reskin of 1.0 rather than a second client that can drift (the first React UI 2.0 was removed
// for exactly that drift - see docs/UI_V2_VISUAL_AUDIT.md).
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
  spotlight: true,        // a pointer-following glow on panel borders
  tilt: true,             // 3D tilt on cards marked as tiltable
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
