import { API } from '../api';

// ── The URL of a rendered output, versioned by the FILE, not the clock ────
//
// Three call sites each built this inline, busting the cache three different
// ways: `?t=${progress.progress}`, `?t=${started_at || Date.now()}`, and
// OutputVideoPlayer's own `Date.now()` on top. The Date.now() ones made every
// remount (every Pinokio tab switch reloads the webview) a brand-new URL, so a
// finished multi-GB render was downloaded again each time it was looked at,
// and `|| Date.now()` evaluated during render made the URL change on ANY
// re-render — remounting the <video> and throwing away its buffer and position.
//
// The backend now reports `version` = `<size>-<mtime_ns>` of the file
// (api._record_last_output), which changes exactly when the file's contents do.
// The server also answers with ETag + `Cache-Control: no-cache`, so even with
// no version the browser revalidates instead of trusting a stale copy.

export function outputBasePath(out) {
  if (!out) return '';
  if (out.url) return out.url;
  if (out.path?.startsWith('/')) return out.path;
  return out.path ? `/api/file?path=${encodeURIComponent(out.path)}` : '';
}

export function withVersion(path, version) {
  if (!path || !version) return path;
  return `${path}${path.includes('?') ? '&' : '?'}v=${encodeURIComponent(version)}`;
}

/** Absolute URL for an output record from /api/progress (`progress.output`). */
export function outputMediaUrl(out) {
  const base = outputBasePath(out);
  return base ? `${API}${withVersion(base, out.version)}` : '';
}

/**
 * The original target the output was rendered from, for compare mode, or
 * null. `offsetS` lines the two clocks up on a trimmed render: output time 0
 * is source frame `start_frame`, the same offset the audio is cut at.
 */
export function outputSource(out) {
  const src = out?.source;
  if (!src?.url) return null;
  const fps = Number(src.fps) || 0;
  return {
    url: `${API}${src.url}`,
    kind: src.kind || 'video',
    name: src.name || '',
    offsetS: fps > 0 ? Math.max(0, Number(src.start_frame) || 0) / fps : 0,
  };
}
