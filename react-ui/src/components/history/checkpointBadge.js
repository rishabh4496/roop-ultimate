// The validation badge on a persistent project checkpoint (#/history).
//
// The backend decides what may be resumed (project_checkpoint.validate) and
// reports `recoverable` plus one reason per problem. This only reads that answer:
//
//   green  resumable: every source, the target and the partial output are present and match
//   amber  one or more INPUT paths are missing, and nothing else is wrong: a one-click relink fixes it
//   red    something is changed, unreadable, or a partial output is gone: relink cannot repair that
//
// A relink is only offered for a missing source or target, because those are the
// only files the user can point at again. Changed content is never offered a
// relink: a different file would be a different render.

// The reason strings the backend writes (project_checkpoint._same_file):
//   "source 2 is missing: C:\path\clip.mp4"
//   "target is missing: no path recorded"
const MISSING = /^(source (\d+)|target) is missing: (.*)$/;

/** A missing-input reason as { role, index, label, path }, or null for any other reason. */
export function parseMissing(reason) {
  const m = MISSING.exec(String(reason || '').trim());
  if (!m) return null;
  const isSource = !!m[2];
  const rawPath = (m[3] || '').trim();
  return {
    role: isSource ? 'source' : 'target',
    index: isSource ? Number(m[2]) - 1 : null,
    label: m[1],
    path: rawPath === 'no path recorded' ? '' : rawPath,
  };
}

/** The badge for one checkpoint summary (the shape GET /api/projects returns). */
export function checkpointBadge(summary) {
  if (summary?.recoverable) {
    return { tone: 'green', label: 'Resumable', detail: 'Every input and the partial output are present and match.', relink: [], reasons: [] };
  }
  const reasons = Array.isArray(summary?.validation_errors) ? summary.validation_errors : [];
  const parsed = reasons.map(parseMissing);
  const relink = parsed.filter(Boolean);
  const blocking = reasons.filter((_, i) => !parsed[i]);
  if (reasons.length === 0) {
    return { tone: 'red', label: 'Not resumable', detail: 'The record could not be validated.', relink: [], reasons: [] };
  }
  if (blocking.length === 0 && relink.length > 0) {
    return {
      tone: 'amber',
      label: relink.length === 1 ? 'Missing path' : `${relink.length} missing paths`,
      detail: 'Point the missing file at its new location to make this resumable again.',
      relink,
      reasons,
    };
  }
  return {
    tone: 'red',
    label: 'Changed or incomplete',
    detail: blocking[0],
    relink: [],
    reasons,
  };
}

/** Tailwind classes for each tone. */
export const BADGE_TONE = {
  green: 'text-emerald-300 bg-emerald-500/10 border-emerald-400/35',
  amber: 'text-amber-300 bg-amber-500/10 border-amber-400/40',
  red: 'text-red-300 bg-red-500/10 border-red-400/35',
};
