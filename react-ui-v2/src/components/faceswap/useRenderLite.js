import { useCallback, useEffect, useSyncExternalStore } from 'react';

// ── Render-lite: stop the UI competing with the GPU ─────────────────────────
// This window is composited by Chromium on the SAME GPU that runs the swap, and
// the swap phase sits at ~98% utilisation — so every frame the compositor
// paints is taken off the render. The costly parts are the backdrop-filter
// blurs (a full readback + blur of whatever is behind each panel, re-done
// whenever anything under them changes — and something under them changes every
// second, because the progress poll lands) and the never-ending keyframe
// animations, which keep the compositor awake at 60 Hz for a render nobody is
// watching frame-by-frame. There are 59 blurred panels in this app.
//
// Switching to the Terminal makes all of that stop, which is exactly why the
// render speeds up there. This makes the UI cost that little while it is on
// screen instead. Scoped to `data-render-lite` in index.css.
//
// The attribute goes on <html>, not on a component's own subtree, because the
// panels it needs to reach are portalled and fixed-position — which is also why
// this is a hook with an effect rather than a className somewhere.
//
// ── Why three modes rather than a switch ───────────────────────────────────
// The same compositing cost is paid when nothing is rendering at all: the blurs
// are re-done on every state change, which on this app means every keystroke and
// every settings edit. So the strip is worth having while idle too — but turning
// it on by default would change how the whole application LOOKS, and that is a
// design decision, not a performance one.
//
//   auto    lite during a run only. The default, and the behaviour this hook
//           had before: nothing about the idle appearance changes.
//   always  lite all the time, for a machine where the idle UI is the problem.
//   off     never, for comparing against.
//
// The preference lives in a module-level store rather than component state
// because two places read it — the Processing dock's button and the
// command palette, which is mounted in App, several trees away. Prop-drilling a
// preference between those two is how a control ends up wired to one of them and
// silently not the other.

const KEY = 'roop_render_lite';
export const RENDER_LITE_MODES = ['auto', 'always', 'off'];

function readStored() {
  try {
    const raw = localStorage.getItem(KEY);
    // Back-compatible with the original boolean: '0' meant "never", anything
    // else (including absent) meant "during a run". Those map onto off/auto, so
    // nobody's existing choice is silently reinterpreted.
    if (raw === '0') return 'off';
    if (raw === '1' || raw === null) return 'auto';
    return RENDER_LITE_MODES.includes(raw) ? raw : 'auto';
  } catch {
    return 'auto';
  }
}

let mode = readStored();
const listeners = new Set();

function setMode(next) {
  if (!RENDER_LITE_MODES.includes(next) || next === mode) return;
  mode = next;
  try { localStorage.setItem(KEY, next); } catch { /* private mode */ }
  listeners.forEach((fn) => fn());
}

function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function cycleRenderLiteMode() {
  setMode(RENDER_LITE_MODES[(RENDER_LITE_MODES.indexOf(mode) + 1) % RENDER_LITE_MODES.length]);
}

export function renderLiteMode() {
  return mode;
}

export const RENDER_LITE_LABEL = {
  auto: 'Lite UI: during renders',
  always: 'Lite UI: always',
  off: 'Lite UI: off',
};

export const RENDER_LITE_HINT = {
  auto: 'Panel blurs and looping animations switch off while a render is running, so the browser leaves the GPU to the swap. The idle interface is unchanged.',
  always: 'Panel blurs and looping animations are off at all times. The interface is plainer and cheaper to composite — use this if the UI itself feels sluggish.',
  off: 'The full interface is always composited, including during a render, on the same GPU as the swap.',
};

// ── One attribute, several mounters ────────────────────────────────────────
// THREE components mount this hook — App (always mounted, so 'always' mode
// holds on every screen), the Processing tab and the Face Swap tab. They all
// drive the same single `data-render-lite` attribute on <html>.
//
// So the effect cannot simply remove the attribute on cleanup. Switching away
// from the Face Swap tab mid-render would run ITS cleanup, strip the attribute,
// and leave it stripped: the other instances' effects do not re-run, because
// none of their dependencies changed. The result is a control that works until
// you change tabs and then silently stops.
//
// A refcount of the mounters currently ASKING for lite is the fix: the
// attribute is present exactly while at least one of them wants it, and any
// instance mounting, unmounting or changing its mind just re-evaluates.
let liteWanters = 0;

function applyLiteAttribute() {
  const el = document.documentElement;
  if (liteWanters > 0) el.setAttribute('data-render-lite', '');
  else el.removeAttribute('data-render-lite');
}

export default function useRenderLite(processing) {
  const current = useSyncExternalStore(subscribe, renderLiteMode, renderLiteMode);

  useEffect(() => {
    const lite = current === 'always' || (current === 'auto' && processing);
    if (!lite) return undefined;
    liteWanters += 1;
    applyLiteAttribute();
    return () => {
      liteWanters = Math.max(0, liteWanters - 1);
      applyLiteAttribute();
    };
  }, [processing, current]);

  const toggleRenderLite = useCallback(() => cycleRenderLiteMode(), []);

  return {
    mode: current,
    // Kept for the existing call sites, which render a two-state button: true
    // whenever this mode does anything at all.
    renderLite: current !== 'off',
    label: RENDER_LITE_LABEL[current],
    hint: RENDER_LITE_HINT[current],
    toggleRenderLite,
  };
}
