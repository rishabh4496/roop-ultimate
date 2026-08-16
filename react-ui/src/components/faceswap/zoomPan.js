/**
 * Shared zoom/pan maths for the preview stage and the comparison grids.
 *
 * Both surfaces render their content as `scale(z) translate(pan.x/z, pan.y/z)`
 * about the centre. Under that transform a content point sitting `u` layout
 * pixels from the centre lands at `u * z + pan` on screen — so `pan` is a
 * straight screen-pixel offset, and every helper here is written in those
 * terms.
 *
 * Everything is a pure function of the stage element plus the current state.
 * Keeping it out of the components is what stops the two of them drifting
 * apart, which is how they ended up with two different (both wrong) answers to
 * "where should a double-click land".
 */

/**
 * The visual scale the app-level CSS `zoom` is currently applying to `el`.
 *
 * `getBoundingClientRect()` reports zoomed pixels while a transform's
 * `translate()` is in unzoomed layout pixels. Every pointer delta therefore has
 * to be divided by this factor, or at any app zoom other than 100% the content
 * slides away from the cursor by exactly the zoom ratio. Derived from the
 * element itself (rect width vs `offsetWidth`) rather than read off
 * `document.documentElement.style.zoom`, so it stays right whichever way the
 * browser chooses to report zoom.
 */
export function uiScale(el) {
  if (!el) return 1;
  const w = el.offsetWidth;
  if (!w) return 1;
  const s = el.getBoundingClientRect().width / w;
  return Number.isFinite(s) && s > 0.01 ? s : 1;
}

/** Stage size in layout pixels, i.e. with the app zoom divided back out. */
export function stageSize(stage) {
  if (!stage) return { w: 0, h: 0 };
  const r = stage.getBoundingClientRect();
  const s = uiScale(stage);
  return { w: r.width / s, h: r.height / s };
}

/**
 * Layout size of an element, via offsetWidth/offsetHeight.
 *
 * Unlike `stageSize`, this is safe on an element INSIDE the zoom/pan transform:
 * a rect there already carries the zoom, layout does not. Same reasoning as
 * `zoomToActual` — measuring the transformed rect and then solving for a bound
 * makes the bound depend on the zoom you happened to be at.
 */
export function contentSize(el) {
  if (!el) return { w: 0, h: 0 };
  return { w: el.offsetWidth, h: el.offsetHeight };
}

/**
 * The furthest the content can travel before its own edge crosses the stage
 * edge. Without this the content can be flung completely out of the box, and
 * the only way back is the reset button — which is not obviously a zoom control
 * to someone who has just lost the image.
 *
 * `content` is the thing actually being looked at, which is NOT the stage
 * whenever the media is letterboxed inside it — and the preview stage is a
 * fixed 16:9 box, so anything that is not 16:9 is letterboxed. Bounding against
 * the stage instead treats the empty bars as image: a 9:16 phone clip is 32% of
 * the stage width, so at 2x it does not overflow horizontally at all, yet the
 * stage-derived bound still permitted half a stage-width of travel — far enough
 * to drag half the picture under the `overflow-hidden` edge. Omit it and the
 * content is taken to fill the stage, which is the old behaviour exactly.
 */
export function panBounds(stage, zoom, content) {
  if (!stage || zoom <= 1) return { x: 0, y: 0 };
  const s = stageSize(stage);
  const c = content ? contentSize(content) : s;
  return {
    x: Math.max(0, (c.w * zoom - s.w) / 2),
    y: Math.max(0, (c.h * zoom - s.h) / 2),
  };
}

export function clampPan(pan, zoom, stage, content) {
  const b = panBounds(stage, zoom, content);
  return {
    x: Math.max(-b.x, Math.min(b.x, pan.x)),
    y: Math.max(-b.y, Math.min(b.y, pan.y)),
  };
}

/**
 * The pan that keeps whatever is under `client` pinned there while the zoom
 * goes `prevZoom` → `nextZoom`.
 *
 * Solving `s = u * z + pan` for the pan that leaves the pointer's own screen
 * offset `s` unchanged gives `pan' = s - (s - pan) * (z' / z)`. Zooming about
 * the cursor instead of the centre is the difference between a magnifier and a
 * shove: the old centre-anchored version pushed the region you were inspecting
 * off the edge on the way in.
 */
export function panAnchoredAt(client, stage, prevZoom, nextZoom, pan) {
  if (!stage || !prevZoom) return pan;
  const rect = stage.getBoundingClientRect();
  const s = uiScale(stage);
  const sx = (client.x - (rect.left + rect.width / 2)) / s;
  const sy = (client.y - (rect.top + rect.height / 2)) / s;
  const k = nextZoom / prevZoom;
  return { x: sx - (sx - pan.x) * k, y: sy - (sy - pan.y) * k };
}

/**
 * Pan that centres the point under `client` at `zoom` — what a double-click to
 * zoom in should do. This is `panAnchoredAt` with the pointer taken to the
 * centre rather than held in place: `pan = -u * z`, where `u` is the clicked
 * offset in layout pixels.
 *
 * Both call sites used to hardcode a `* 1.5` here while zooming to 2.5x, so the
 * clicked detail landed 40% of the way off centre — far enough that on a face
 * you would double-click an eye and get the ear.
 */
export function panCenteringAt(client, stage, zoom, content) {
  if (!stage) return { x: 0, y: 0 };
  const rect = stage.getBoundingClientRect();
  const s = uiScale(stage);
  const u = {
    x: (client.x - (rect.left + rect.width / 2)) / s,
    y: (client.y - (rect.top + rect.height / 2)) / s,
  };
  return clampPan({ x: -u.x * zoom, y: -u.y * zoom }, zoom, stage, content);
}

/**
 * One wheel notch, as a MULTIPLICATIVE step. The old additive `± 0.15` meant 47
 * notches to cross the 1→8 range and a step that felt 8x coarser at 1x than at
 * 8x; a ratio gives the same perceived travel everywhere.
 *
 * Returns null when the wheel cannot move the zoom any further. That is the
 * signal for the caller NOT to preventDefault, so a scroll that reaches the
 * bottom of the zoom range keeps scrolling the page instead of being swallowed
 * by a stage that has nothing left to do with it.
 */
export function wheelZoom(deltaY, zoom, max, step = 0.2) {
  const raw = deltaY < 0 ? zoom * (1 + step) : zoom / (1 + step);
  const next = Math.min(Math.max(raw, 1), max);
  return Math.abs(next - zoom) < 1e-4 ? null : next;
}

/** The transform every zoom/pan stage renders with. */
export function transformFor(zoom, pan) {
  return {
    transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
    transformOrigin: 'center',
  };
}
