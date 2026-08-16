import { useEffect, useState } from 'react';

// ── Render-lite: stop the UI competing with the render for the GPU ─────────
// This window is composited by Chromium on the SAME GPU that is running the
// swap, and the swap phase sits at ~98% utilisation — so every frame the
// compositor paints is taken off the render. The costly parts are the
// backdrop-filter blurs (a full readback + blur of whatever is behind each
// panel, re-done whenever anything under them changes — and something under
// them changes every second, because the progress poll lands) and the
// never-ending keyframe animations, which keep the compositor awake at 60 Hz
// for a render nobody is watching frame-by-frame.
//
// Switching to the Terminal makes all of that stop, which is exactly why the
// render speeds up there. This makes the UI cost that little while it is on
// screen instead. Scoped to `data-render-lite` in index.css, applied only
// during a run, and switchable from the dock so the difference can be measured
// rather than taken on trust.
//
// The attribute goes on <html>, not on a component's own subtree, because the
// panels it needs to reach are portalled and fixed-position — which is also why
// this is a hook with an effect rather than a className somewhere.

const KEY = 'roop_render_lite';

export default function useRenderLite(processing) {
  // Defaults ON: the setting exists to make the common case fast, and someone
  // who wants the full-fat UI during a render can say so and be remembered.
  const [renderLite, setRenderLite] = useState(() => {
    try { return localStorage.getItem(KEY) !== '0'; } catch { return true; }
  });

  const toggleRenderLite = () => {
    setRenderLite((on) => {
      try { localStorage.setItem(KEY, on ? '0' : '1'); } catch { /* private mode */ }
      return !on;
    });
  };

  useEffect(() => {
    const el = document.documentElement;
    if (processing && renderLite) el.setAttribute('data-render-lite', '');
    else el.removeAttribute('data-render-lite');
    // Cleanup matters on unmount as well as on change: leaving the tab
    // mid-render would otherwise strand the whole app in the stripped-down
    // styling with nothing left mounted to take it off again.
    return () => el.removeAttribute('data-render-lite');
  }, [processing, renderLite]);

  return { renderLite, toggleRenderLite };
}
