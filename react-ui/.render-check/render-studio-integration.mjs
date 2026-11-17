import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { STUDIO_CONTROLS, studioPayload } from '../src/components/faceswap/studioControls.js';
import StudioParameters from '../src/components/faceswap/StudioParameters.jsx';
import { PARAMETER_CONFIGS, createNleKeyframe, evaluateKeyframeValue } from '../src/components/faceswap/timeline/parameterKeyframes.ts';
import { serializeStudioTimeline, restoreAutomationProjects } from '../src/components/faceswap/studioProjectModel.js';
import { readStudioCheckpoint, writeStudioCheckpoint, STUDIO_PROJECT_KEY } from '../src/components/faceswap/useStudioProject.js';

let failed = 0;
function check(name, run) {
  try { run(); console.log(`PASS ${name}`); }
  catch (error) { failed++; console.error(`FAIL ${name}`, error); }
}
check('every Studio control renders and has a keyframe configuration', () => {
  const html = renderToStaticMarkup(React.createElement(StudioParameters, { values: {}, onChange() {} }));
  for (const control of STUDIO_CONTROLS) {
    assert.ok(html.includes(control.label));
    assert.ok(PARAMETER_CONFIGS[control.wire]);
  }
  assert.equal(Object.keys(studioPayload({})).length, 15);
});
check('sparse numeric, enum and boolean keys serialize together without coercion', () => {
  const project = { keyframes: [createNleKeyframe('pixel_boost', 0, 2), createNleKeyframe('multi_angle_detect', 0, 1),
    createNleKeyframe('restore_identity_weight', 0, .3, 'Hold'), createNleKeyframe('grain_match_strength', 0, .6, 'Linear')], bindings: [] };
  const [key] = serializeStudioTimeline(project).keyframes;
  assert.equal(key.frame, 0); assert.equal(key.pixel_boost, 768); assert.equal(key.multi_angle_detect, true);
  assert.equal(key.interpolations.restore_identity_weight, 'hold'); assert.equal(key.interpolations.grain_match_strength, 'linear');
  const keys = [createNleKeyframe('pixel_boost', 0, 0), createNleKeyframe('pixel_boost', 100, 2)];
  assert.equal(evaluateKeyframeValue(keys, 'pixel_boost', 99), 0);
  assert.equal(evaluateKeyframeValue(keys, 'pixel_boost', 100), 2);
});
check('frontend easing agrees with the backend smoothstep away from the midpoint', () => {
  const keys = [createNleKeyframe('grain_match_strength', 0, 0), createNleKeyframe('grain_match_strength', 100, 1)];
  assert.equal(evaluateKeyframeValue(keys, 'grain_match_strength', 25), .15625);
});
check('trimmed track intervals and disabled tracks reach backend bindings', () => {
  const timeline = serializeStudioTimeline({ bindings: [{ track_id: 4, faceset_path: 'actor.fsz', enabled: true }],
    intervals: [{ track_id: 4, frame_start: 30, frame_end: 90, locked: true }] });
  assert.deepEqual(timeline.bindings[0].active_ranges, [[30, 90]]);
  assert.equal(serializeStudioTimeline({ bindings: timeline.bindings, intervals: [{ track_id: 4, locked: false }] }).bindings[0].enabled, false);
});
check('checkpoint restores clips, bindings, painted masks, strokes, keys and trims', () => {
  const values = new Map();
  const storage = { getItem: (key) => values.get(key) || null, setItem: (key, value) => values.set(key, value) };
  const projects = { clipA: { mask: { dataUrl: 'data:image/png;base64,AA==' }, strokes: [{ frame_idx: 0, points: [[1, 2], [3, 4]] }],
    bindings: [{ track_id: 1, faceset_path: 'one.fsz' }], keyframes: [createNleKeyframe('Swap Opacity', 5, .5)], trim: { start: 1, end: 90 } },
    clipB: { bindings: [{ track_id: 2, faceset_path: 'two.fsz' }] } };
  writeStudioCheckpoint(storage, projects);
  writeStudioCheckpoint(storage, { ...projects, clipC: {} });
  assert.deepEqual(readStudioCheckpoint(storage).clipA, projects.clipA);
  values.set(STUDIO_PROJECT_KEY, 'broken json');
  assert.deepEqual(readStudioCheckpoint(storage), projects);
});
check('undoing the first project edit clears keys while retaining the brush checkpoint', () => {
  const restored = restoreAutomationProjects({ clipA: { keyframes: [{ frame_idx: 5 }], mask: 'png' } }, {});
  assert.deepEqual(restored.clipA.keyframes, []); assert.equal(restored.clipA.mask, 'png');
});
globalThis.__RENDER_CHECK_FAILURES__ = failed;
