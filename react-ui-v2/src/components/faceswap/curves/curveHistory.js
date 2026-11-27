import { paramById } from './curveModel';

// The undo-history label for a change to the parameter curves. The channel's value is { clipKey: { paramId: keyframes } }
// (only clips edited this session appear), so the label is found by comparing the one track that differs.

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

export function describeCurvesChange(before, after) {
  const clips = new Set([...Object.keys(before || {}), ...Object.keys(after || {})]);
  for (const clip of clips) {
    const b = before?.[clip] || {};
    const a = after?.[clip] || {};
    const ids = new Set([...Object.keys(b), ...Object.keys(a)]);
    for (const id of ids) {
      if (same(b[id], a[id])) continue;
      const name = paramById(id)?.label || id;
      const nb = b[id]?.length || 0;
      const na = a[id]?.length || 0;
      if (na > nb) return `Added ${name} Keyframe`;
      if (na < nb) return na === 0 ? `Cleared ${name} Curve` : `Removed ${name} Keyframe`;
      const moved = (b[id] || []).some((k, i) => k.frame !== a[id][i].frame || k.value !== a[id][i].value);
      return moved ? `Moved ${name} Keyframe` : `Changed ${name} Interpolation`;
    }
  }
  return 'Edited Parameter Curves';
}
