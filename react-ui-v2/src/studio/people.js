// Target people, as the studio shows them: one colour each, a display name, and who they become.
export const PERSON_HUES = ['#f84fcc', '#00dbe4', '#fbbf24', '#4ade80', '#a78bfa', '#fb923c', '#38bdf8', '#f472b6'];

/**
 * Unique people of the current target, in capture order: [{ id, rank, faceIndex, name, thumb, legacy }].
 * The engine names people by stable id (target_person_ids). An older target context has only
 * target_groups (a person RANK per captured face); then the rank is the id, exactly as React UI
 * 1.0's faceMapping helpers key it, and selection goes by face index.
 */
export function peopleOf(s) {
  const out = [];
  const seen = new Set();
  const ids = s.targetPersonIds || [];
  const legacy = !ids.some(Boolean);
  const keys = legacy ? (s.targetGroups || []) : ids;
  keys.forEach((id, faceIndex) => {
    if (id === null || id === undefined || id === '') return;
    if (seen.has(id)) { out.find((p) => p.id === id)?.faceIndices.push(faceIndex); return; }
    seen.add(id);
    const rank = out.length;
    out.push({
      id, rank, faceIndex, legacy,
      // The engine's target_names is indexed by person RANK (_target_names_ranked), not by captured
      // face: with two angles of one person, a face index names the wrong person.
      name: s.targetNames?.[rank] || s.targetFacesInfo?.[faceIndex]?.name || `Person ${rank + 1}`,
      named: !!s.targetNames?.[rank],
      faceIndices: [faceIndex],     // every captured angle of this person
      thumb: s.targetFaces?.[faceIndex] || null,
    });
  });
  return out;
}

/** Whether person `p` is the selected one (by id, or by face index in a legacy context). */
export const isSelectedPerson = (s, p) => (p.legacy
  ? (s.targetGroups || [])[s.selTargetFace] === p.id
  : p.id === s.selectedTargetPersonId);

export const hueOf = (rank) => PERSON_HUES[((rank % PERSON_HUES.length) + PERSON_HUES.length) % PERSON_HUES.length];

/** Which source person `id` becomes: an index, -1 (left untouched) or null (not mapped yet). */
export function mappedSource(s, id) {
  const m = s.faceMapping || {};
  if (!Object.prototype.hasOwnProperty.call(m, id)) return null;
  const v = Array.isArray(m[id]) ? m[id][0] : m[id];
  const n = Number(v);
  return Number.isInteger(n) ? n : null;
}

export function timecode(frame, fps) {
  const f = Math.max(0, (frame || 1) - 1);
  const r = fps > 0 ? fps : 30;
  const total = f / r;
  const m = Math.floor(total / 60);
  const sec = Math.floor(total % 60);
  const ff = Math.floor(f % Math.round(r));
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}.${String(ff).padStart(2, '0')}`;
}

// The engine's boxes are [x1, y1, x2, y2] (insightface). A box whose far corner is not past its
// near one cannot be that, so it is read as [x, y, w, h] - only the mock backend sends that form.
export const asCorners = (b) => {
  const [a, c, d, e] = (b || []).map(Number);
  return d > a && e > c ? [a, c, d, e] : [a, c, a + Math.abs(d), c + Math.abs(e)];
};

export const frameUrl = (index, frame, width = 0) => `/api/target/preview?index=${index}&frame=${frame}${width ? `&width=${width}` : ''}`;
