// WHERE THE WIPE LINE IS, AS A TRANSFORM. The compare divider used to be positioned with `left`/`top` (a percentage),
// which re-runs layout for the stage on every pointer move of a drag and on every tick of the auto-swipe. It is now a
// full-stage-sized wrapper (`left: 0; width: 100%` / `top: 0; height: 100%`), and the wipe position is a `translate3d`
// of it: a percentage in a translate resolves against the element's OWN size, which here is the stage's, so
// `translate3d(P%, 0, 0)` lands the line exactly where `left: P%` did. Compositor-only; no layout.
//
// The rotated (diagonal) wipe needs a distance along a direction. That cannot be a percentage (the wrapper is 200vmax
// wide there), so it is in px from the stage's layout size (offsetWidth/Height: local CSS px, unaffected by the app's
// CSS zoom or by transforms), then `translate(-50%, -50%) rotate()` as before.
//
// Pure (the host is only read for its offsetWidth/offsetHeight) so .render-check/render-stage8-logic.mjs can pin the
// geometry against the `left: P%` it replaced.

/**
 * @param {number} pct         0..100, how much of the stage shows the processed side
 * @param {number} angle       0 = linear curtain, otherwise the wipe direction in degrees
 * @param {boolean} horizontal linear curtain direction (ignored when angle != 0)
 * @param {{offsetWidth:number, offsetHeight:number}|null} host the stage element (null before first mount)
 */
export function dividerTransform(pct, angle, horizontal, host) {
  if (angle !== 0) {
    const rad = (angle * Math.PI) / 180;
    const normDist = (100 - pct - 50) / 100;                 // -0.5 .. +0.5 of the stage
    const dx = normDist * Math.cos(rad) * (host?.offsetWidth || 0);
    const dy = normDist * Math.sin(rad) * (host?.offsetHeight || 0);
    return `translate3d(${dx}px, ${dy}px, 0) translate(-50%, -50%) rotate(${angle + 90}deg)`;
  }
  return horizontal ? `translate3d(0, ${100 - pct}%, 0)` : `translate3d(${100 - pct}%, 0, 0)`;
}
