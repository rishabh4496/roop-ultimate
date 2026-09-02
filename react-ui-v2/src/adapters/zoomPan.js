/**
 * Pure geometric mathematics for canvas/image zoom, pan, lens loupe, and viewport scale.
 */

export const clamp = (v, lo, hi) => Math.max(lo, Math.min(v, hi));

/**
 * Visual-to-layout pixel scale ratio.
 * Dividing by uiScale ensures panning and magnifier overlays don't drift on zoomed displays.
 */
export const uiScale = (el) => {
  if (!el || typeof window === 'undefined') return 1;
  const zoom = parseFloat(window.getComputedStyle(document.documentElement).zoom) || 1;
  return zoom;
};

export const clampPan = (pan, zoom, container, image) => {
  if (zoom <= 1 || !container) return { x: 0, y: 0 };
  const cw = container.clientWidth || 800;
  const ch = container.clientHeight || 600;
  const iw = (image?.clientWidth || cw) * zoom;
  const ih = (image?.clientHeight || ch) * zoom;

  const maxPanX = Math.max(0, (iw - cw) / 2);
  const maxPanY = Math.max(0, (ih - ch) / 2);

  return {
    x: clamp(pan.x, -maxPanX, maxPanX),
    y: clamp(pan.y, -maxPanY, maxPanY),
  };
};

export const panAnchoredAt = (cursor, container, currentZoom, nextZoom, currentPan) => {
  if (!container || currentZoom === nextZoom) return currentPan;
  const rect = container.getBoundingClientRect();
  const scale = uiScale(container);
  const cx = (cursor.x - rect.left) / scale - rect.width / 2;
  const cy = (cursor.y - rect.top) / scale - rect.height / 2;

  const factor = nextZoom / currentZoom;
  return {
    x: cx - (cx - currentPan.x) * factor,
    y: cy - (cy - currentPan.y) * factor,
  };
};

export const panCenteringAt = (cursor, container, targetZoom, image) => {
  if (!container) return { x: 0, y: 0 };
  const rect = container.getBoundingClientRect();
  const scale = uiScale(container);
  const cx = (cursor.x - rect.left) / scale - rect.width / 2;
  const cy = (cursor.y - rect.top) / scale - rect.height / 2;

  return clampPan({ x: -cx * (targetZoom - 1), y: -cy * (targetZoom - 1) }, targetZoom, container, image);
};

export const wheelZoom = (deltaY, currentZoom, maxZoom = 8) => {
  if (Math.abs(deltaY) < 1e-3) return null;
  const factor = deltaY < 0 ? 1.25 : 1 / 1.25;
  const next = clamp(currentZoom * factor, 1, maxZoom);
  return Math.abs(next - currentZoom) > 1e-3 ? next : null;
};

export const transformFor = (zoom, pan) => ({
  transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
  transformOrigin: 'center center',
});
