// Pure helpers for the Outputs gallery: the metadata pills on a card, the compare
// selection rule, and the quad-view grid.
//
// Where each value really comes from, so nothing is guessed:
//   resolution  the video element's own videoWidth / videoHeight (read on load)
//   render fps  the run's recorded throughput (history entry: frames / duration).
//               It is how fast the RENDER went, not the clip's frame rate, and the
//               pill says so.
//   model       the swap model that run used (history entry settings.swap_model)
//   duration    how long the render took (history entry duration_s)
// A value that was not recorded is omitted, not shown as zero.

/** "1920×1080", or null until the media has reported its dimensions. */
export function fmtResolution(width, height) {
  const w = Number(width);
  const h = Number(height);
  if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return null;
  return `${Math.round(w)}×${Math.round(h)}`;
}

/** A render duration as "41 s", "3m 07s" or "1h 02m". Null when unknown. */
export function fmtRenderDuration(seconds) {
  const s = Number(seconds);
  if (!Number.isFinite(s) || s <= 0) return null;
  if (s < 60) return `${Math.round(s)} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(Math.round(s % 60)).padStart(2, '0')}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
}

/** Render throughput with one decimal, e.g. "12.4 fps". Null when unknown. */
export function fmtRenderFps(fps) {
  const n = Number(fps);
  if (!Number.isFinite(n) || n <= 0) return null;
  return `${n.toFixed(1)} fps`;
}

/**
 * The pills for one output. `dims` is {width, height} from the element, or null.
 * Returns [{ key, label, value, title }] in display order, omitting unknowns.
 */
export function pillsFor(entry, dims = null) {
  const pills = [];
  const res = dims ? fmtResolution(dims.width, dims.height) : null;
  if (res) pills.push({ key: 'res', label: 'Res', value: res, title: 'Frame size read from the file' });
  const fps = entry ? fmtRenderFps(entry.fps) : null;
  if (fps) pills.push({ key: 'fps', label: 'Render', value: fps, title: 'Render throughput: frames divided by render time' });
  const model = entry?.settings?.swap_model;
  if (model) pills.push({ key: 'model', label: 'Model', value: String(model), title: 'Swap model used for this render' });
  const dur = entry ? fmtRenderDuration(entry.duration_s) : null;
  if (dur) pills.push({ key: 'dur', label: 'Took', value: dur, title: 'How long this render took' });
  return pills;
}

/**
 * Whether a selection can open the quad view: 2 to 4 renders. Returns
 * `{ ok, reason }` so the toolbar can say why a button is absent.
 */
export function compareSelection(count) {
  if (count < 2) return { ok: false, reason: 'Select 2 to 4 renders to compare them' };
  if (count > 4) return { ok: false, reason: `${count} selected: the quad view compares up to 4` };
  return { ok: true, reason: '' };
}

/** Grid for a quad view of n tiles: two columns, as many rows as needed (2 -> 1 row, 3-4 -> 2 rows). */
export function quadLayout(n) {
  const count = Math.max(1, Math.min(4, Math.round(Number(n) || 0)));
  return { cols: 2, rows: count <= 2 ? 1 : 2, count };
}

/** Clamp a scrub position (0..1) for the synchronised playhead. */
export function clampRatio(r) {
  const n = Number(r);
  if (!Number.isFinite(n)) return 0;
  return Math.min(1, Math.max(0, n));
}
