/**
 * React UI 2.0's engine payloads (plain node, no browser):
 *
 *   node .render-check/run-module.mjs render-studio.mjs
 *
 * buildEnginePayload / buildPreviewPayload / buildQueueJob / readiness, driven with studio states
 * shaped like the engine's /api/state + target context. The selection fields come from React UI
 * 1.0's pure builders; these checks pin that 2.0 hands them the right inputs - the failure this
 * project keeps meeting is a render that "succeeds" having swapped nobody.
 */
globalThis.window = globalThis.window || { location: { origin: 'http://127.0.0.1', host: '127.0.0.1' } };

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};

const { buildEnginePayload, buildPreviewPayload, buildQueueJob, readiness, sourceIdAt } = await import('../src/engine/payload.js');

const settings = {
  face_detection_mode: 'Selected face', selected_enhancer: 'GPEN', max_face_distance: 0.65, blend_ratio: 0.9,
  subsample_upscale: '256px', mask_engine: 'DFL XSeg', mask_engine_2: 'None', mask_clip_text: 'hair',
  autorotate_faces: true, swap_model: 'realswap', num_swap_steps: 1, some_engine_key: 42,
};
const base = {
  settings,
  sources: ['data:a', 'data:b'],
  sourcesInfo: [{ id: 'src-anna', name: 'Anna' }, { id: 'src-ben', name: 'Ben' }],
  selectedSource: 1,
  targets: [{ media_id: 'm1', name: 'clip.mp4', frames: 300 }],
  selectedTarget: 0,
  targetMediaId: 'm1',
  targetGroups: [0, 1],
  targetPersonIds: ['tp_a', 'tp_b'],
  faceMapping: {},
  selTargetFace: 1,
  selectedTargetPersonId: 'tp_b',
  selectionVersion: 1000,
};

console.log('── selected face ───────────────────────────────────────────');
{
  const p = buildEnginePayload(base);
  ok('the selected person gets the selected source, the other is skipped', JSON.stringify(p.face_mapping) === '[-1,1]', JSON.stringify(p.face_mapping));
  ok('selection_state says "selected" and names the person', p.selection_state.selection_mode === 'selected' && p.selection_state.person_id === 'tp_b', JSON.stringify(p.selection_state));
  ok('processing_selection targets that person', p.processing_selection.target_person_id === 'tp_b' && p.processing_selection.target_media_id === 'm1');
  ok('processing_selection carries the source identity id', p.processing_selection.source_identity_id === 'src-ben');
  ok('the stable mapping maps the person to the source id', p.target_person_source_mapping.tp_b === 'src-ben', JSON.stringify(p.target_person_source_mapping));
  ok('names and ids follow the mapping', JSON.stringify(p.source_mapping_names) === '[null,"Ben"]' && JSON.stringify(p.source_mapping_ids) === '[null,"src-ben"]');
  ok('selected source name/id', p.selected_source_name === 'Ben' && p.selected_source_id === 'src-ben');
  ok('selection version travels', p.selection_version === 1000 && p.processing_selection.selection_version === 1000);
}

console.log('── renamed keys (what the engine reads) ────────────────────');
{
  const p = buildEnginePayload(base);
  ok('enhancer <- selected_enhancer', p.enhancer === 'GPEN');
  ok('face_distance <- max_face_distance', p.face_distance === 0.65);
  ok('upscale <- subsample_upscale', p.upscale === '256px');
  ok('clip_text <- mask_clip_text', p.clip_text === 'hair');
  ok('autorotate <- autorotate_faces', p.autorotate === true);
  ok('detection mirrors face_detection_mode', p.detection === 'Selected face' && p.face_detection_mode === 'Selected face');
  ok('every other setting rides along untouched', p.some_engine_key === 42 && p.swap_model === 'realswap');
  ok('mouth-custody request fields are present', 'mouth_custody_mode' in p && 'audio_offset_ms' in p);
  ok('target media id', p.target_media_id === 'm1');
}

console.log('── explicit mapping, several people ────────────────────────');
{
  const s = { ...base, settings: { ...settings, face_detection_mode: 'Selected people' }, faceMapping: { tp_a: 0, tp_b: -1 } };
  const p = buildEnginePayload(s);
  ok('an explicit choice wins: A -> Anna, B untouched', JSON.stringify(p.face_mapping) === '[0,-1]', JSON.stringify(p.face_mapping));
  ok('multi_person selection lists only the mapped person', p.selection_state.selection_mode === 'multi_person'
    && JSON.stringify(p.selection_state.person_ids) === '["tp_a"]', JSON.stringify(p.selection_state));
  const auto = buildEnginePayload({ ...s, faceMapping: { tp_a: 0 } });
  ok('a person with NO key (automatic) is not forced to skip', auto.face_mapping[0] === 0, JSON.stringify(auto.face_mapping));
}

console.log('── preview + queue bodies ──────────────────────────────────');
{
  const pv = buildPreviewPayload(base, { frame: 42, swap: true });
  ok('preview carries index, frame and fake_preview', pv.index === 0 && pv.target_index === 0 && pv.frame === 42 && pv.fake_preview === true);
  ok('preview has the same selection as the swap', pv.processing_selection.target_person_id === 'tp_b');
  ok('an original-frame preview asks for no swap', buildPreviewPayload(base, { frame: 1, swap: false }).fake_preview === false);
  const job = buildQueueJob(base);
  ok('a queue job freezes the swap body', job.payload && job.payload.face_mapping && job.target_media_id === 'm1');
  ok('the job carries names for the queue list', job.target_name === 'clip.mp4' && job.source_name === 'Ben' && job.source_id === 'src-ben');
  ok('the job carries the frozen selection + its request id', job.processing_selection === job.payload.processing_selection
    && job.request_id === job.payload.processing_selection.request_id);
}

console.log('── readiness (refuse a render that would swap nobody) ──────');
{
  ok('no source', readiness({ ...base, sources: [] }) === 'Add a source face to the cast');
  ok('no target', /target media/.test(readiness({ ...base, targets: [] }) || ''));
  ok('Selected face with no person picked', /Pick the target person/.test(readiness({ ...base, targetPersonIds: [], targetGroups: [] }) || ''));
  ok('every person set to skip', /Skip/.test(readiness({ ...base, settings: { ...settings, face_detection_mode: 'Selected people' }, faceMapping: { tp_a: -1, tp_b: -1 } }) || ''));
  ok('a complete setup is ready', readiness(base) === null);
  ok('"All faces" needs no picking', readiness({ ...base, settings: { ...settings, face_detection_mode: 'All faces' }, targetPersonIds: [], targetGroups: [] }) === null);
  ok('a source without an id still has a stable slot id', sourceIdAt({ sourcesInfo: [] }, 3) === 'memory-slot-3');
}

console.log(failures ? `studio: ${failures} check(s) FAILED` : 'studio: all checks passed');
globalThis.__RENDER_CHECK_FAILURES__ = failures;
