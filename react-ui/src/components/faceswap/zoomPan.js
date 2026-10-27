/**
 * Shared zoom/pan maths for the preview stage and the comparison grids.
 *
 * Both surfaces render their content as `scale(z) translate(pan.x/z, pan.y/z)`
 * or direct affine 2D transformation matrices about the centre. Under that
 * transform a content point sitting `u` layout pixels from the centre lands
 * at `u * z + pan` on screen — so `pan` is a straight screen-pixel offset, and
 * every helper here is written in those terms.
 *
 * Everything is a pure function of the stage element plus the current state.
 */

export const ZOOM_MIN = 0.1;
export const ZOOM_MAX = 16.0;

/**
 * The visual scale the app-level CSS `zoom` is currently applying to `el`.
 *
 * `getBoundingClientRect()` reports zoomed pixels while a transform's
 * `translate()` is in unzoomed layout pixels. Every pointer delta therefore has
 * to be divided by this factor, or at any app zoom other than 100% the content
 * slides away from the cursor by exactly the zoom ratio.
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
 */
export function contentSize(el) {
  if (!el) return { w: 0, h: 0 };
  return { w: el.offsetWidth, h: el.offsetHeight };
}

/**
 * The furthest the content can travel before its own edge crosses the stage
 * edge.
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
 * offset `s` unchanged gives `pan' = s - (s - pan) * (z' / z)`.
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
 * Pan that centres the point under `client` at `zoom`.
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
 * One wheel notch, as a MULTIPLICATIVE step from min (0.1x) to max (16x).
 *
 * Returns null when the wheel cannot move the zoom any further.
 */
export function wheelZoom(deltaY, zoom, max = ZOOM_MAX, min = ZOOM_MIN, step = 0.15) {
  const raw = deltaY < 0 ? zoom * (1 + step) : zoom / (1 + step);
  const next = Math.min(Math.max(raw, min), max);
  return Math.abs(next - zoom) < 1e-4 ? null : next;
}

/** The transform string every zoom/pan stage renders with. */
export function transformFor(zoom, pan) {
  return {
    transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
    transformOrigin: 'center',
  };
}

/** Direct 2D affine matrix representation for high-frequency transforms. */
export function affineMatrix(zoom, pan) {
  return {
    transform: `matrix(${zoom}, 0, 0, ${zoom}, ${pan.x}, ${pan.y})`,
    transformOrigin: 'center center',
  };
}

/**
 * Maps a pointer screen coordinate through the viewport matrix back into
 * the native media pixel coordinates.
 */
export function screenToMedia(clientX, clientY, stage, imgDim, zoom, pan) {
  if (!stage || !imgDim || !imgDim.w || !imgDim.h) return null;
  const rect = stage.getBoundingClientRect();
  const s = uiScale(stage);

  // Screen offset from stage center in layout pixels
  const sx = (clientX - (rect.left + rect.width / 2)) / s;
  const sy = (clientY - (rect.top + rect.height / 2)) / s;

  // Invert the zoom/pan transform: u = (s - pan) / zoom
  const ux = (sx - pan.x) / zoom;
  const uy = (sy - pan.y) / zoom;

  // Media aspect contain fit inside stage
  const ss = stageSize(stage);
  const scale = Math.min(ss.w / imgDim.w, ss.h / imgDim.h);
  const mw = imgDim.w * scale;
  const mh = imgDim.h * scale;

  // Coordinate on image plane (-mw/2 to mw/2) -> (0 to imgDim.w)
  const mx = (ux / mw + 0.5) * imgDim.w;
  const my = (uy / mh + 0.5) * imgDim.h;

  const onImage = mx >= 0 && mx <= imgDim.w && my >= 0 && my <= imgDim.h;
  return { x: mx, y: my, onImage };
}

/**
 * Calculates smoothed camera target zoom and pan to lock and center on a face.
 * Uses 5-point ArcFace landmark telemetry when available, falling back to bbox.
 */
export function faceTrackingTarget(bbox, kps, imgDim, stage) {
  if (!imgDim || !imgDim.w || !imgDim.h || !stage) return null;
  let cx = 0;
  let cy = 0;
  let fw = 0;
  let fh = 0;

  if (kps && kps.length >= 5) {
    const [eyeL, eyeR, nose, mouthL, mouthR] = kps;
    cx = (eyeL[0] + eyeR[0] + nose[0] + mouthL[0] + mouthR[0]) / 5;
    cy = (eyeL[1] + eyeR[1] + nose[1] + mouthL[1] + mouthR[1]) / 5;
    fw = Math.max(Math.abs(eyeR[0] - eyeL[0]) * 2.2, 80);
    fh = Math.max(Math.abs((mouthL[1] + mouthR[1]) / 2 - (eyeL[1] + eyeR[1]) / 2) * 2.5, 80);
  } else if (bbox && bbox.length >= 4) {
    const [sx, sy, ex, ey] = bbox;
    cx = (sx + ex) / 2;
    cy = (sy + ey) / 2;
    fw = Math.max(ex - sx, 60);
    fh = Math.max(ey - sy, 60);
  } else {
    return null;
  }

  const ss = stageSize(stage);
  // Fit face to ~40% of the stage height for comfortable viewing
  const scale = Math.min(ss.w / imgDim.w, ss.h / imgDim.h);
  const faceDisplayH = Math.max(fw * scale, fh * scale);
  const desiredZoom = Math.min(Math.max((ss.h * 0.45) / Math.max(faceDisplayH, 1), 1.5), 6.0);

  // Offset from image center in normalized [-0.5, 0.5]
  const normX = cx / imgDim.w - 0.5;
  const normY = cy / imgDim.h - 0.5;

  const contentW = imgDim.w * scale;
  const contentH = imgDim.h * scale;
  const ux = normX * contentW;
  const uy = normY * contentH;

  const targetPan = {
    x: -ux * desiredZoom,
    y: -uy * desiredZoom,
  };

  return { zoom: desiredZoom, pan: clampPan(targetPan, desiredZoom, stage, { offsetWidth: contentW, offsetHeight: contentH }) };
}
