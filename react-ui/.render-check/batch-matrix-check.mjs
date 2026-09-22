// Behavioural checks for the Batch Matrix staging logic (per-file matrix,
// grouped, one-to-many, recipes). These import and RUN the module BatchSwap.jsx
// ships (src/components/faceswap/batchMatrix.js), so a refactor that keeps the
// toast strings but pairs the wrong faceset with a target fails here.
//
//   node .render-check/batch-matrix-check.mjs            run the checks
//   node .render-check/batch-matrix-check.mjs --emit F   also write every
//        scenario's POST /api/queue/add_batch body to F, for
//        app/tests/test_batch_matrix_queue.py to push through the real queue.
//
// The fixture is the one docs/development/BATCH_MATRIX_DATA_FLOW.md walks:
// three targets, two facesets, and the recipe split A+alice, B+bob, C+alice.
import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import process from 'node:process';
import {
  ERR_NO_GROUP_TARGETS,
  ERR_NO_MATRIX_FILES,
  ERR_NO_SOURCES,
  autoMatchMatrixConfig,
  buildBatchJobPayload,
  defaultMatrixRow,
  queueRequestFromStagedJobs,
  stageCartesian,
  stageGrouped,
  stageMatrix,
  stageOneToMany,
  stageSegments,
  stageSequential,
} from '../src/components/faceswap/batchMatrix.js';

let pass = 0;
const fails = [];
const check = (name, fn) => {
  try { fn(); pass++; console.log(`  PASS  ${name}`); }
  catch (e) { fails.push(name); console.log(`  FAIL  ${name}\n        ${e.message}`); }
};
const group = (t) => console.log(`\n── ${t} ${'─'.repeat(Math.max(0, 50 - t.length))}`);

// ── Fixture ───────────────────────────────────────────────────────────────
// /api/state as the strategies read it. Target A and C have two captured
// people, B has one; the ACTIVE target at page load is A (targetGroups=[0,1]).
const TARGETS = [
  { id: 0, media_id: 'media-a', name: 'clip_a.mp4', frames: 300, start_frame: 1, end_frame: 300, fps: 30 },
  { id: 1, media_id: 'media-b', name: 'clip_b.mp4', frames: 120, start_frame: 1, end_frame: 120, fps: 30 },
  { id: 2, media_id: 'media-c', name: 'clip_c.mp4', frames: 600, start_frame: 1, end_frame: 600, fps: 30 },
];
const SOURCES = [
  { id: '/facesets/alice.fsz', name: 'alice.fsz', count: 3, poses: [] },
  { id: '/facesets/bob.fsz', name: 'bob.fsz', count: 2, poses: [] },
];
const ALICE = 0;
const BOB = 1;
const SETTINGS = { selected_enhancer: 'None', max_face_distance: 0.65 };

const payloadBuilder = (targetGroups) => (mappings, swapMode, overrides = {}) =>
  buildBatchJobPayload({
    mappings, swapMode, overrides, settings: SETTINGS,
    sourceCount: SOURCES.length, sourceFacesInfo: SOURCES, targetGroups,
  });
const common = (targetGroups = [0, 1]) => ({
  targets: TARGETS, sourceCount: SOURCES.length, sourceFacesInfo: SOURCES,
  createJobPayload: payloadBuilder(targetGroups),
});

// One job reduced to the triple the queue must preserve.
const triple = (j) => ({
  target: j.target_name,
  target_index: j.target_index,
  source: j.source_name,
  source_id: j.source_id ?? j.payload.selected_source_id,
  face_mapping: j.payload.face_mapping,
  ids: j.payload.source_mapping_ids,
});

const emitted = {
  fixture: { targets: TARGETS, sources: SOURCES, active_target_groups: [0, 1] },
  scenarios: {},
};
const emit = (name, jobs, expected) => {
  emitted.scenarios[name] = { jobs: queueRequestFromStagedJobs(jobs), expected };
};

// ── Recipe: sequential split A+alice, B+bob, C+alice ─────────────────────
group('Recipe matrix: sequential match');
{
  const { jobs, error } = stageSequential(common());
  check('3 targets x 2 facesets -> exactly 3 jobs, one per target', () => {
    assert.equal(error, null);
    assert.equal(jobs.length, 3);
  });
  check('the pairing is A+alice, B+bob, C+alice (tIdx % sourceCount)', () => {
    assert.deepEqual(jobs.map(triple), [
      { target: 'clip_a.mp4', target_index: 0, source: 'alice.fsz', source_id: SOURCES[ALICE].id,
        face_mapping: [ALICE], ids: [SOURCES[ALICE].id] },
      { target: 'clip_b.mp4', target_index: 1, source: 'bob.fsz', source_id: SOURCES[BOB].id,
        face_mapping: [BOB], ids: [SOURCES[BOB].id] },
      { target: 'clip_c.mp4', target_index: 2, source: 'alice.fsz', source_id: SOURCES[ALICE].id,
        face_mapping: [ALICE], ids: [SOURCES[ALICE].id] },
    ]);
  });
  check('each job carries its own target frame range', () => {
    assert.deepEqual(jobs.map((j) => [j.frame_start, j.frame_end]), [[1, 300], [1, 120], [1, 600]]);
  });
  check('the wire form names the target and the primary faceset by id', () => {
    const wire = queueRequestFromStagedJobs(jobs);
    assert.deepEqual(wire.map((w) => [w.target_name, w.source_index, w.source_id]), [
      ['clip_a.mp4', ALICE, SOURCES[ALICE].id],
      ['clip_b.mp4', BOB, SOURCES[BOB].id],
      ['clip_c.mp4', ALICE, SOURCES[ALICE].id],
    ]);
    assert.ok(wire.every((w) => !('target_media_id' in w)), 'F1: media id is not forwarded (documented)');
  });
  check('payload objects are distinct per job (no shared mutable state)', () => {
    jobs[0].payload.face_mapping.push(99);
    assert.deepEqual(jobs[1].payload.face_mapping, [BOB]);
    assert.deepEqual(jobs[2].payload.face_mapping, [ALICE]);
    jobs[0].payload.face_mapping.pop();
  });
  emit('sequential', jobs, jobs.map(triple));
}

group('Recipe matrix: cartesian');
{
  const { jobs } = stageCartesian(common());
  check('3 x 2 -> 6 jobs, target-major order', () => {
    assert.deepEqual(jobs.map((j) => `${j.target_name}->${j.source_name}`), [
      'clip_a.mp4->alice.fsz', 'clip_a.mp4->bob.fsz',
      'clip_b.mp4->alice.fsz', 'clip_b.mp4->bob.fsz',
      'clip_c.mp4->alice.fsz', 'clip_c.mp4->bob.fsz',
    ]);
    assert.deepEqual(jobs.map((j) => j.target_index), [0, 0, 1, 1, 2, 2]);
  });
  emit('cartesian', jobs, jobs.map(triple));
}

group('Recipe matrix: segment splitter');
{
  const { jobs } = stageSegments({ ...common(), targetIndex: 2, segmentCount: 4 });
  check('600 frames / 4 -> contiguous, non-overlapping ranges on ONE target', () => {
    assert.deepEqual(jobs.map((j) => [j.frame_start, j.frame_end]),
      [[1, 150], [151, 300], [301, 450], [451, 600]]);
    assert.ok(jobs.every((j) => j.target_index === 2 && j.target_name === 'clip_c.mp4'));
  });
  check('a 5-way split of 120 frames does not run past the end', () => {
    const { jobs: five } = stageSegments({ ...common(), targetIndex: 1, segmentCount: 5 });
    assert.deepEqual(five.map((j) => [j.frame_start, j.frame_end]),
      [[1, 24], [25, 48], [49, 72], [73, 96], [97, 120]]);
  });
}

// ── Per-file matrix ──────────────────────────────────────────────────────
group('Per-file matrix');
{
  const matrixConfig = {
    0: { ...defaultMatrixRow(TARGETS[0], SETTINGS),
      mappings: [{ personRank: 0, sourceIdx: BOB }, { personRank: 1, sourceIdx: ALICE }],
      swapMode: 'Selected people', frameStart: 5, frameEnd: 50, enhancer: 'GPEN' },
    1: { ...defaultMatrixRow(TARGETS[1], SETTINGS), enabled: false,
      mappings: [{ personRank: 0, sourceIdx: BOB }] },
    2: { ...defaultMatrixRow(TARGETS[2], SETTINGS),
      mappings: [{ personRank: 0, sourceIdx: ALICE }], faceDistance: 0.4 },
  };
  const { jobs, error } = stageMatrix({ ...common(), matrixConfig });
  check('a disabled row is skipped and the NEXT row keeps its own index', () => {
    assert.equal(error, null);
    assert.deepEqual(jobs.map((j) => [j.target_name, j.target_index]),
      [['clip_a.mp4', 0], ['clip_c.mp4', 2]]);
  });
  check("row A: two people -> face_mapping [bob, alice] in rank order", () => {
    assert.deepEqual(jobs[0].payload.face_mapping, [BOB, ALICE]);
    assert.deepEqual(jobs[0].payload.source_mapping_ids, [SOURCES[BOB].id, SOURCES[ALICE].id]);
    assert.equal(jobs[0].source_index, BOB, 'primary = first mapping row');
    assert.equal(jobs[0].source_name, 'bob.fsz');
    assert.deepEqual(jobs[0].payload.selection_state.person_ids, [0, 1]);
    assert.equal(jobs[0].payload.selection_state.selection_mode, 'multi_person');
  });
  check('row A: per-row frame range, enhancer and swap mode reach the payload', () => {
    assert.deepEqual([jobs[0].frame_start, jobs[0].frame_end, jobs[0].total_frames], [5, 50, 46]);
    assert.equal(jobs[0].payload.enhancer, 'GPEN');
    assert.equal(jobs[0].payload.detection, 'Selected people');
  });
  check('row C: its own mapping, not row A\'s and not row B\'s', () => {
    assert.deepEqual(jobs[1].payload.face_mapping, [ALICE]);
    assert.equal(jobs[1].source_name, 'alice.fsz');
    assert.equal(jobs[1].payload.face_distance, 0.4);
    assert.equal(jobs[1].payload.enhancer, 'None', 'settings.selected_enhancer is the row default');
    assert.deepEqual([jobs[1].frame_start, jobs[1].frame_end], [1, 600]);
  });
  check('a row whose target index has no config is skipped (sparse config)', () => {
    const { jobs: sparse } = stageMatrix({ ...common(), matrixConfig: { 2: matrixConfig[2] } });
    assert.deepEqual(sparse.map((j) => j.target_index), [2]);
  });
  check('all rows disabled -> the documented error, no jobs', () => {
    const off = Object.fromEntries(Object.entries(matrixConfig).map(([k, v]) => [k, { ...v, enabled: false }]));
    assert.deepEqual(stageMatrix({ ...common(), matrixConfig: off }), { jobs: [], error: ERR_NO_MATRIX_FILES });
  });
  check('no facesets loaded -> refused before any job is built', () => {
    assert.deepEqual(stageMatrix({ ...common(), sourceCount: 0, sourceFacesInfo: [], matrixConfig }),
      { jobs: [], error: ERR_NO_SOURCES });
  });
  check('a sparse rank list is dense on the wire: rank 1 only -> [SKIP, src]', () => {
    const { jobs: one } = stageMatrix({ ...common(), matrixConfig: {
      2: { ...matrixConfig[2], mappings: [{ personRank: 1, sourceIdx: BOB }] } } });
    assert.deepEqual(one[0].payload.face_mapping, [-1, BOB]);
    assert.deepEqual(one[0].payload.source_mapping_ids, [null, SOURCES[BOB].id]);
    assert.equal(one[0].payload.selection_state.person_id, 1);
  });
  check('a sourceIdx past the gallery becomes SKIP, never source 0 -- and a stale PRIMARY row makes source_index -1', () => {
    const { jobs: stale } = stageMatrix({ ...common(), matrixConfig: {
      2: { ...matrixConfig[2], mappings: [{ personRank: 0, sourceIdx: 7 }] } } });
    assert.deepEqual(stale[0].payload.face_mapping, [-1]);
    assert.equal(stale[0].source_index, -1);
    assert.equal(stale[0].payload.selected_source_id, null);
    emit('stale_primary', stale, stale.map(triple));
  });
  emit('matrix', jobs, jobs.map(triple));
}

group('Per-file matrix: auto-match by filename token');
{
  const named = [
    { id: 0, name: 'alice_beach.mp4', frames: 10 },
    { id: 1, name: 'party.mp4', frames: 10 },
    { id: 2, name: 'bob-and-alice.mp4', frames: 10 },
  ];
  const { matrixConfig: cfg, matchCount } = autoMatchMatrixConfig({
    targets: named, sourceFacesInfo: SOURCES, matrixConfig: { 1: { enabled: false, mappings: [] } },
  });
  check('matches by token, first faceset wins a tie, unmatched rows untouched', () => {
    assert.equal(matchCount, 2);
    assert.deepEqual(cfg[0].mappings, [{ personRank: 0, sourceIdx: ALICE }]);
    assert.deepEqual(cfg[2].mappings, [{ personRank: 0, sourceIdx: ALICE }], 'alice is index 0 so wins over bob');
    assert.deepEqual(cfg[1], { enabled: false, mappings: [] });
  });
}

// ── Grouped ──────────────────────────────────────────────────────────────
group('Grouped');
{
  const groups = [
    { id: 1, label: 'Group X', targetIndices: [0, 2],
      mappings: [{ personRank: 0, sourceIdx: BOB }], swapMode: 'Selected face',
      enhancer: 'CodeFormer', faceDistance: 0.5 },
    { id: 2, label: 'Group Y', targetIndices: [1, 2],
      mappings: [{ personRank: 0, sourceIdx: ALICE }], swapMode: 'All faces',
      enhancer: 'None', faceDistance: 0.9 },
  ];
  const { jobs, error } = stageGrouped({ ...common(), groups });
  check('two groups sharing target C -> 4 jobs, group-major then target order', () => {
    assert.equal(error, null);
    assert.deepEqual(jobs.map((j) => [j.target_index, j.source_name]),
      [[0, 'bob.fsz'], [2, 'bob.fsz'], [1, 'alice.fsz'], [2, 'alice.fsz']]);
  });
  check("each job carries ITS group's mapping and settings, not the other group's", () => {
    assert.deepEqual(jobs.map((j) => j.payload.face_mapping), [[BOB], [BOB], [ALICE], [ALICE]]);
    assert.deepEqual(jobs.map((j) => j.payload.enhancer), ['CodeFormer', 'CodeFormer', 'None', 'None']);
    assert.deepEqual(jobs.map((j) => j.payload.face_distance), [0.5, 0.5, 0.9, 0.9]);
    assert.deepEqual(jobs.map((j) => j.payload.detection),
      ['Selected face', 'Selected face', 'All faces', 'All faces']);
    assert.deepEqual(jobs.map((j) => j.payload.selection_state.selection_mode),
      ['selected', 'selected', 'none', 'none']);
  });
  check('labels name the group and the target', () => {
    assert.equal(jobs[1].label, 'Group X | clip_c.mp4 (P#1➔F#2)');
    assert.equal(jobs[2].label, 'Group Y | clip_b.mp4 (P#1➔F#1)');
  });
  check('a group with no targets contributes nothing; every group empty -> error', () => {
    const { jobs: some } = stageGrouped({ ...common(), groups: [{ ...groups[0], targetIndices: [] }, groups[1]] });
    assert.deepEqual(some.map((j) => j.target_index), [1, 2]);
    assert.deepEqual(stageGrouped({ ...common(), groups: [{ ...groups[0], targetIndices: [] }] }),
      { jobs: [], error: ERR_NO_GROUP_TARGETS });
  });
  check('a target index that is no longer loaded is dropped from the group (documented: silently)', () => {
    const { jobs: dropped } = stageGrouped({ ...common(), groups: [{ ...groups[0], targetIndices: [0, 7, 2] }] });
    assert.deepEqual(dropped.map((j) => j.target_index), [0, 2]);
  });
  check('a two-person group mapping -> dense rank-ordered face_mapping on every member', () => {
    const { jobs: two } = stageGrouped({ ...common(), groups: [{ ...groups[0],
      mappings: [{ personRank: 1, sourceIdx: ALICE }, { personRank: 0, sourceIdx: BOB }],
      swapMode: 'Selected people' }] });
    assert.deepEqual(two.map((j) => j.payload.face_mapping), [[BOB, ALICE], [BOB, ALICE]]);
    assert.equal(two[0].source_index, ALICE, 'primary is the FIRST mapping row, not rank 0');
    assert.deepEqual(two[0].payload.selection_state.person_ids, [0, 1]);
  });
  emit('grouped', jobs, jobs.map(triple));
}

// ── One-to-many ──────────────────────────────────────────────────────────
group('One-to-many');
{
  const { jobs } = stageOneToMany({ ...common(), selectedTargets: [2, 0],
    mappings: [{ personRank: 0, sourceIdx: ALICE }, { personRank: 1, sourceIdx: BOB }],
    swapMode: 'Selected people', enhancer: 'GFPGAN', faceDistance: 0.7 });
  check('selected targets in the order selected, the same mapping on each', () => {
    assert.deepEqual(jobs.map((j) => [j.target_index, j.target_name]), [[2, 'clip_c.mp4'], [0, 'clip_a.mp4']]);
    assert.deepEqual(jobs.map((j) => j.payload.face_mapping), [[ALICE, BOB], [ALICE, BOB]]);
    assert.deepEqual(jobs.map((j) => j.payload.enhancer), ['GFPGAN', 'GFPGAN']);
  });
  emit('one_to_many', jobs, jobs.map(triple));
}

// ── The active person bank vs. the job's own target ──────────────────────
// Ranks address people on the JOB'S target and are resolved server-side at
// dispatch; the active target's bank must not decide whether a rank exists.
group('Ranks are validated against the job\'s target, not the active one');
{
  // Active target = B (one captured person). Staging a per-file-matrix row
  // for A that addresses A's SECOND person.
  const active = [0];
  const { jobs } = stageMatrix({ ...common(active), matrixConfig: {
    0: { ...defaultMatrixRow(TARGETS[0], SETTINGS), mappings: [{ personRank: 1, sourceIdx: BOB }] } } });
  check('matrix: rank 1 for target A survives when the ACTIVE target has one person', () => {
    assert.deepEqual(jobs[0].payload.face_mapping, [-1, BOB]);
    assert.equal(jobs[0].payload.selection_state.person_id, 1);
    assert.deepEqual(jobs[0].payload.selection_state.person_ids, [1]);
    assert.equal(jobs[0].payload.selection_state.valid, true);
  });
  const { jobs: grouped } = stageGrouped({ ...common(active), groups: [{ id: 1, label: 'G', targetIndices: [2],
    mappings: [{ personRank: 0, sourceIdx: ALICE }, { personRank: 1, sourceIdx: BOB }],
    swapMode: 'Selected people', enhancer: 'None', faceDistance: 0.75 }] });
  check('grouped: a two-person mapping for target C survives when the ACTIVE target has one person', () => {
    assert.deepEqual(grouped[0].payload.selection_state.person_ids, [0, 1]);
    assert.equal(grouped[0].payload.selection_state.valid, true);
  });
  emit('matrix_rank1_active_bank_one_person', jobs, jobs.map(triple));
  emit('grouped_two_people_active_bank_one_person', grouped, grouped.map(triple));
}

// ── Report ───────────────────────────────────────────────────────────────
const emitAt = process.argv.indexOf('--emit');
if (emitAt !== -1 && process.argv[emitAt + 1]) {
  writeFileSync(process.argv[emitAt + 1], JSON.stringify(emitted, null, 2));
  console.log(`\nemitted ${Object.keys(emitted.scenarios).length} scenarios -> ${process.argv[emitAt + 1]}`);
}
console.log(`\n${pass} passed, ${fails.length} failed`);
if (fails.length) {
  console.log('FAILED: ' + fails.join(', '));
  process.exit(1);
}
console.log('ALL GREEN');
