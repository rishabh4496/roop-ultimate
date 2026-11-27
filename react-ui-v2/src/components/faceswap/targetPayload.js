// Apply a target-faces API payload (the shape every /api/target/* mutation
// returns, via api._target_faces_payload) to Face Swap's parallel state.
//
// Lifted out of PersonGroups so that anything else which changes a person's
// angle bank (the Biometric Angle HUD's "Add to angle bank") updates the UI
// through the SAME steps as the auto-capture button, rather than a copy that
// could drift from it. Every handler is optional; a missing one is skipped.
export function applyTargetFacesPayload(res, h) {
  if (!res) return;
  if (h.applyTargetContext) {
    h.applyTargetContext(res, res.target_media_id || h.targetMediaId);
  }
  if (res.target_faces) h.setTargetFaces?.(res.target_faces);
  if (res.target_groups && h.setTargetGroups) {
    const flat = res.target_groups.map((g) => Array.isArray(g) ? (g[0] ?? 0) : (typeof g === 'number' ? g : parseInt(g, 10) || 0));
    h.setTargetGroups(flat);
  }
  if (res.target_names !== undefined && h.setTargetNames) h.setTargetNames(res.target_names || []);
  if (res.target_faces_info !== undefined && h.setTargetFacesInfo) h.setTargetFacesInfo(res.target_faces_info || []);
  if (res.target_person_source_mapping !== undefined && h.setFaceMapping) {
    h.setFaceMapping(res.target_person_source_mapping || {});
  } else if (res.face_mapping && !Array.isArray(res.face_mapping) && h.setFaceMapping) {
    h.setFaceMapping(res.face_mapping || {});
  }
  if (res.target_person_ids && h.setTargetPersonIds) h.setTargetPersonIds(res.target_person_ids);
  if (res.target_reference_face_ids && h.setTargetReferenceFaceIds) h.setTargetReferenceFaceIds(res.target_reference_face_ids);
  if (res.selected_target_person_id && h.setSelectedTargetPersonId) h.setSelectedTargetPersonId(res.selected_target_person_id);
  if (h.setSelectedReferenceFaceId) {
    const refId = res.selected_reference_face_id
      ?? (res.target_reference_face_ids?.[res.selected_target_face_index ?? 0] || null);
    if (refId !== undefined) {
      h.setSelectedReferenceFaceId(refId);
    }
  }
  if (h.clearPreviewCache) h.clearPreviewCache();
}
