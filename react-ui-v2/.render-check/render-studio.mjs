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

const { buildEnginePayload, buildPreviewPayload, buildQueueJob, readiness, sourceIdAt, settingsAtFrame, renderPlan, planJobs, planBlocked, maskJsonOf, inOutOf, engineFrameFor } = await import('../src/engine/payload.js');

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

console.log('── tools in the render ─────────────────────────────────────');
{
  const zone = { id: 'z1', mode: 'exclude', feather: 8, frames: null, closed: true, enabled: true,
    points: [{ x: 10, y: 10 }, { x: 60, y: 10 }, { x: 60, y: 50 }] };
  const pin = { id: 'p1', frame: 12, bbox: [1, 2, 30, 40], source_index: 1, person_id: 'tp_b', track_id: null };
  const s = {
    ...base,
    mask: { dataUrl: 'data:image/png;base64,AAA', refKps: [[1, 2], [3, 4]], version: 3 },
    clipFx: { m1: { roto: [zone], pins: [pin], overrides: [], curves: {} }, other: { roto: [{ ...zone, id: 'leak' }] } },
    identityBlend: { enabled: true, components: [{ source_id: 'src-anna', weight: 1 }, { source_id: 'src-ben', weight: 0.5 }], mode: 'slerp' },
  };
  const p = buildEnginePayload(s);
  const mask = JSON.parse(p.imagemask);
  ok('imagemask is {"<selected source>": {exclude, canonical:false, ref_kps}}', mask['1']?.exclude === 'data:image/png;base64,AAA' && mask['1'].canonical === false && mask['1'].ref_kps?.length === 2, p.imagemask.slice(0, 90));
  ok('no mask -> empty imagemask', maskJsonOf({ ...s, mask: { dataUrl: null } }) === '');
  ok('roto zones belong to this clip only', p.roto_polygons.length === 1 && p.roto_polygons[0].id === 'z1' && p.roto_polygons[0].points.length === 3);
  ok('an open zone is not sent', buildEnginePayload({ ...s, clipFx: { m1: { roto: [{ ...zone, closed: false }] } } }).roto_polygons.length === 0);
  ok('pins serialize with frame, bbox and source', p.identity_pins.length === 1 && p.identity_pins[0].frame === 12 && p.identity_pins[0].source_index === 1);
  ok('the identity blend is normalised into the payload', p.identity_blend.enabled === true && p.identity_blend.components.length === 2 && Array.isArray(p.identity_blend.keyframes));
  ok('a different clip carries none of it', buildEnginePayload({ ...s, targetMediaId: 'm9' }).roto_polygons.length === 0);
}

console.log('── scene overrides + curves ────────────────────────────────');
{
  const t = { media_id: 'm1', name: 'clip.mp4', frames: 300, start_frame: 1, end_frame: 300 };
  const ov = { id: 'o1', start: 101, end: 200, label: 'Close-up', bypass: false, params: { mask_feather: 60 }, order: 0 };
  const s = { ...base, targets: [t], settings: { ...settings, face_mask_blend: 20, blend_ratio: 1 },
    clipFx: { m1: { roto: [], pins: [], overrides: [ov], curves: {} } } };
  ok('inside the override the preview reads its value', settingsAtFrame(s, 150).face_mask_blend === 60);
  ok('outside it, the clip settings', settingsAtFrame(s, 50).face_mask_blend === 20);
  const plan = renderPlan(s);
  ok('an override cuts the range into 3 pieces', plan.has && plan.pieces.length === 3, JSON.stringify(plan.pieces.map((pc) => [pc.start, pc.end])));
  const jobs = planJobs(s, plan);
  ok('each piece is a job with its own range and patched payload', jobs.length === 3 && jobs[1].frame_start === 101 && jobs[1].frame_end === 200
    && jobs[1].payload.face_mask_blend === 60 && jobs[0].payload.face_mask_blend === 20, JSON.stringify(jobs.map((j) => [j.frame_start, j.payload.face_mask_blend])));
  ok('pieces share a segment group so the queue can join them', jobs.every((j) => j.segment && j.segment.group === jobs[0].segment.group && j.segment.count === 3));
  ok('no override, no curve -> one plain render', renderPlan({ ...s, clipFx: {} }).has === false);
  const curved = { ...s, clipFx: { m1: { overrides: [], curves: { swap_opacity: [{ frame: 1, value: 1, interp: 'linear' }, { frame: 300, value: 0.2, interp: 'linear' }] } } } };
  ok('a curve changes the preview along the clip', settingsAtFrame(curved, 1).blend_ratio === 1 && settingsAtFrame(curved, 300).blend_ratio < 0.3, `${settingsAtFrame(curved, 300).blend_ratio}`);
  const cp = renderPlan(curved);
  ok('a curve renders as stepped pieces', cp.has && cp.pieces.length > 2 && cp.pieces.length <= 24, `${cp.pieces.length} pieces`);
  const bypass = { ...s, settings: { ...s.settings, face_detection_mode: 'All faces' }, clipFx: { m1: { overrides: [{ ...ov, bypass: true, params: {} }] } } };
  ok('bypass outside the targeted modes is refused', !!planBlocked(bypass, renderPlan(bypass)));
}

console.log('── In/Out frames (1-based inclusive at the API) ─────────────');
{
  // Live engine, 11-28: set_frame used to store In raw as the 0-based seek, so In 300 / Out 329 rendered
  // 29 frames (301..329). The API now takes and reports 1-based frames (app/api.py target_set_frame).
  ok('In at frame 300 is sent as 300', engineFrameFor('start', 300) === 300);
  ok('Out at frame 329 is sent as 329', engineFrameFor('end', 329) === 329);
  const full = inOutOf({ frames: 81507, start_frame: 1, end_frame: 81507 });
  ok('the full-clip default reads as 1..frames', full.a === 1 && full.b === 81507, JSON.stringify(full));
  const set = inOutOf({ frames: 81507, start_frame: 300, end_frame: 329 });
  ok('In 300 / Out 329 counts 30 frames', set.a === 300 && set.b === 329 && set.b - set.a + 1 === 30, JSON.stringify(set));
  ok('end_frame 0 (unset) means the whole clip', inOutOf({ frames: 50, start_frame: 1, end_frame: 0 }).b === 50);
  const s = { ...base, targets: [{ media_id: 'm1', frames: 300, start_frame: 100, end_frame: 200 }], clipFx: {} };
  const plan = renderPlan(s);
  ok('the render plan covers 100..200', plan.lo === 100 && plan.hi === 200, `${plan.lo}..${plan.hi}`);
}

console.log(failures ? `studio: ${failures} check(s) FAILED` : 'studio: all checks passed');
globalThis.__RENDER_CHECK_FAILURES__ = failures;
