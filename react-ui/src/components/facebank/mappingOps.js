// Pure mapping mutations behind FaceBankRouter, kept out of the JSX so the node
// check (.render-check/cinematic-facebank-check.mjs) exercises the real code.
//
// A mapping is { [clusterId]: sourceId | sourceId[] | -1 }, -1 = unassigned.
// Every function returns a NEW mapping and never mutates its input.

export const UNASSIGNED = -1;

/** Assign sourceId to clusterId; append=true adds it to the existing set. */
export function assignSource(mapping, clusterId, sourceId, append = false) {
  const key = String(clusterId);
  const existing = mapping[key];
  if (append && existing !== undefined && existing !== UNASSIGNED) {
    const arr = Array.isArray(existing) ? [...existing] : [existing];
    if (!arr.includes(sourceId)) arr.push(sourceId);
    return { ...mapping, [key]: arr };
  }
  return { ...mapping, [key]: sourceId };
}

/**
 * Remove one source from clusterId. Two left -> one collapses to a scalar;
 * none left -> UNASSIGNED. Unknown cluster -> the mapping unchanged.
 */
export function removeSource(mapping, clusterId, sourceId) {
  const key = String(clusterId);
  const existing = mapping[key];
  if (existing === undefined) return mapping;
  if (Array.isArray(existing)) {
    const filtered = existing.filter((id) => String(id) !== String(sourceId));
    return {
      ...mapping,
      [key]: filtered.length > 0 ? (filtered.length === 1 ? filtered[0] : filtered) : UNASSIGNED,
    };
  }
  return { ...mapping, [key]: UNASSIGNED };
}

/** Per-cluster override values with their defaults filled in. */
export function normalizeOverrides(existing = {}) {
  return {
    cosineThreshold: existing.cosineThreshold !== undefined ? existing.cosineThreshold : 0.60,
    maskOffset: existing.maskOffset !== undefined ? existing.maskOffset : 0,
    action: existing.action || 'swap', // 'swap' | 'keep' | 'censor'
  };
}
