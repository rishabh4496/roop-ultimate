// Pure model for the Studio Launchpad (#/home): the module deep links and the
// model-weight readiness summary.
//
// Readiness comes from GET /api/models/offline, which checks every manifest entry
// on disk ("ok", "missing", or "corrupt"). A required model that is missing or
// corrupt is what stops a render, so it is named; an optional one is counted but
// not alarming.

/**
 * Every module the launchpad links to. `tab` must be one of the App's tab ids;
 * the render-check pins that, so a renamed tab cannot leave a dead link.
 */
export const MODULE_LINKS = [
  { tab: 'faceswap', label: 'Face Swap', blurb: 'Single-clip swap with the full settings deck' },
  { tab: 'batch', label: 'Batch Matrix', blurb: 'Source × target × recipe grid, split and queued' },
  { tab: 'processing', label: 'Processing', blurb: 'Render queue mission control and the live run' },
  { tab: 'facemgr', label: 'Face Manager', blurb: 'Identity workstation: facesets, targets, coverage' },
  { tab: 'extras', label: 'Editor', blurb: 'Timeline, scrubbers, effect rack and routing' },
  { tab: 'gallery', label: 'Outputs', blurb: 'Media wall, hover scrub and the quad compare' },
  { tab: 'history', label: 'History', blurb: 'Project checkpoints and past run settings' },
  { tab: 'settings', label: 'Settings', blurb: 'Providers, hardware sweep and diagnostics' },
];

/**
 * Summarise GET /api/models/offline's `models` map into a readiness verdict.
 * `models` is { [file]: { status: 'ok' | 'missing' | 'corrupt', ... } }.
 */
export function readinessOf(models) {
  const entries = Object.entries(models && typeof models === 'object' ? models : {});
  const total = entries.length;
  const ok = entries.filter(([, m]) => m?.status === 'ok').length;
  const blocking = entries
    .filter(([, m]) => m?.status && m.status !== 'ok')
    .map(([file, m]) => ({ file, status: m.status }));
  let tone = 'ready';
  if (total === 0) tone = 'unknown';
  else if (blocking.length) tone = 'blocked';
  return {
    tone,
    total,
    ok,
    blocking,
    label: total === 0
      ? 'Model check not available'
      : blocking.length
        ? (blocking.length === 1 ? '1 model needs attention' : `${blocking.length} models need attention`)
        : `All ${total} model weights present`,
  };
}

/** Clock-free "x ago" for a unix-seconds timestamp, shared by the launchpad cards. */
export function agoLabel(ts, now = Date.now()) {
  if (!ts) return '';
  const s = Math.max(0, (now - Number(ts) * 1000) / 1000);
  if (s < 90) return 'just now';
  const m = s / 60;
  if (m < 60) return `${Math.round(m)} min ago`;
  const h = m / 60;
  if (h < 24) return `${Math.round(h)} h ago`;
  return `${Math.round(h / 24)} d ago`;
}
