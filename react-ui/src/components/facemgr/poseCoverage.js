// What the .fsz gallery badge says, as DATA, from the `pose_coverage` field of a /api/faceset/library entry
// (app/roop/faceset_coverage.py):
//
//   { left, front, right, deep_left, deep_right, unsolved, total, missing: [...], complete }   or   null
//
// `null` means the backend could not measure it (a legacy archive with no pose data, or an unreadable file). That is
// shown as "unknown", never as "front only" and never as "complete": a badge that guessed would tell the user a
// faceset is fine when nothing was measured.
//
// The sides are the archive's own left / center / right split (+-25 deg yaw). `deep_*` counts references past 50 deg,
// so "Left 2 (1 profile)" can be told apart from two 30-degree three-quarter views.

export const SIDES = ['left', 'front', 'right'];
const SHORT = { left: 'L', front: 'F', right: 'R' };
const NAME = { left: 'Left', front: 'Front', right: 'Right' };

const count = (v) => (Number.isFinite(Number(v)) && Number(v) > 0 ? Math.floor(Number(v)) : 0);
const isCoverage = (c) => c && typeof c === 'object' && !Array.isArray(c);

export function coverageBadge(entry) {
  const c = entry ? entry.pose_coverage : undefined;
  if (!isCoverage(c)) {
    return {
      state: 'unknown',
      pips: SIDES.map((side) => ({ side, short: SHORT[side], label: NAME[side], count: 0, deep: 0, lit: false })),
      text: 'Angles unknown',
      title: 'Angles unknown: this archive records no head-pose data, so left / right coverage cannot be measured.',
    };
  }
  const deep = { left: count(c.deep_left), front: 0, right: count(c.deep_right) };
  const pips = SIDES.map((side) => {
    const n = count(c[side]);
    return { side, short: SHORT[side], label: NAME[side], count: n, deep: deep[side], lit: n > 0 };
  });
  const missing = pips.filter((p) => !p.lit).map((p) => p.side);
  const unsolved = count(c.unsolved);
  const measured = pips.reduce((sum, p) => sum + p.count, 0);
  const state = measured === 0 ? 'unknown' : missing.length === 0 ? 'complete' : 'partial';

  const parts = pips.map((p) => {
    if (!p.lit) return `${p.label}: none`;
    const profile = p.deep > 0 ? ` (${p.deep} true profile${p.deep === 1 ? '' : 's'})` : '';
    return `${p.label}: ${p.count}${profile}`;
  });
  let title = parts.join(' · ');
  if (unsolved > 0) title += ` · ${unsolved} reference${unsolved === 1 ? '' : 's'} with no solved pose`;
  if (state === 'partial') title += `. Missing ${missing.join(' and ')} angles: turned heads on that side fall back to the nearest angle held.`;
  if (state === 'unknown') title = 'Angles unknown: no reference in this archive has a solved head pose.';

  const text = state === 'complete' ? 'L · F · R'
    : state === 'partial' ? `No ${missing.join('/')}`
      : 'Angles unknown';
  return { state, pips, text, title };
}
