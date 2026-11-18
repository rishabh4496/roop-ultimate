/**
 * Stage 12: the pure logic behind the VFX editing upgrades (plain node).
 *
 *   node .render-check/run-module.mjs render-stage12-logic.mjs        (npm run test:stage12)
 *
 *   1. history   non-destructive branching: a new edit after an undo PARKS the redo tail, switching is reversible,
 *                the ring cap strands (and prunes) a branch whose fork fell off, entries are classified by what moved
 *   2. (curves, slots, faces: appended below as each lands)
 */
globalThis.window = globalThis;
globalThis.location = { protocol: 'http:', host: '127.0.0.1:42003', origin: 'http://127.0.0.1:42003', hash: '' };
globalThis.window.location = globalThis.location;

const E = await import('../src/history/historyEngine.js');
const M = await import('../src/history/historyStackModel.js');
const { useHistoryStore, resetHistoryForTests } = await import('../src/store/historyStore.js');
const C = await import('../src/history/historyChannels.js');

let failures = 0;
const ok = (name, cond, extra = '') => {
  if (!cond) { failures += 1; console.error(`  FAIL  ${name}${extra ? ` — ${extra}` : ''}`); } else console.log(`  PASS  ${name}`);
};
const tick = () => new Promise((r) => queueMicrotask(r));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

console.log('── history: branching (engine) ──');
{
  let h = E.createHistory({ a: 0 }, { now: 0 });
  for (let i = 1; i <= 4; i += 1) h = E.commit(h, { a: i }, { now: i * 1000, label: `a${i}` });   // 0 1 2 3 4
  ok('start: five entries, no branches', h.entries.length === 5 && h.index === 4 && E.branchesOf(h).length === 0);

  const back = E.jump(h, 2);                                           // cursor on a2; a3, a4 are the redo tail
  let h2 = E.commit(back.h, { a: 90 }, { now: 9000, label: 'ninety' });
  ok('a new edit after an undo keeps the tail as ONE branch', E.branchesOf(h2).length === 1);
  const [br] = E.branchesOf(h2);
  ok('the branch forks at the entry we left and holds both parked steps', br.forkIndex === 2 && br.count === 2 && br.label === 'a3', JSON.stringify(br));
  ok('the live path is the new edit', h2.entries.length === 4 && E.head(h2).state.a === 90 && E.redo(h2) === null);

  const sw = E.switchBranch(h2, br.id);
  ok('switching lands on the branch tip', sw && sw.state.a === 4 && sw.h.index === sw.h.entries.length - 1 && sw.h.entries.length === 5);
  const back2 = E.branchesOf(sw.h);
  ok('the path we left is now a branch (a switch is reversible)', back2.length === 1 && back2[0].count === 1 && back2[0].label === 'ninety' && back2[0].forkIndex === 2, JSON.stringify(back2));
  const sw2 = E.switchBranch(sw.h, back2[0].id);
  ok('switching back restores the original path exactly', sw2 && sw2.state.a === 90 && sw2.h.entries.map((e) => e.state.a).join() === '0,1,2,90');
  ok('an unknown branch id is null', E.switchBranch(h2, 999) === null);

  // a branch of a branch
  const jumped = E.jump(sw2.h, 1);
  ok('jumping does not make branches', E.branchesOf(jumped.h).length === 1);
  let h3 = E.commit(jumped.h, { a: 55 }, { now: 12000, label: 'fifty-five' });   // forks at a1, parking 2,90
  ok('a second fork parks the tail again (2 kept, only the one forking off the live path is offered)',
    h3.branches.length === 2 && E.branchesOf(h3).length === 1 && E.branchesOf(h3)[0].count === 2, JSON.stringify(E.branchesOf(h3)));
  const nested = (h3.branches || []).find((b) => b.entries[0].state.a === 2);
  ok('the parked tail still contains the earlier fork point (branches nest)', nested && nested.entries.map((e) => e.state.a).join() === '2,90');
  const viaNested = E.switchBranch(h3, nested.id);
  ok('the branch parked earlier can be switched to again', viaNested && viaNested.state.a === 90);
  ok('and the stranded sub-branch of it is offered again once it is on the path', E.branchesOf(viaNested.h).some((b) => b.count === 2 && b.label === 'a3'),
    JSON.stringify(E.branchesOf(viaNested.h)));

  // ring cap: a branch whose fork fell off is pruned, not left dangling
  let big = E.createHistory({ n: 0 }, { now: 0, cap: 6 });
  for (let i = 1; i <= 5; i += 1) big = E.commit(big, { n: i }, { now: i * 1000 });
  const fk = E.jump(big, 3);
  big = E.commit(fk.h, { n: 100 }, { now: 7000 });                       // parks n4, n5 at fork n3
  ok('under the cap the branch exists', E.branchesOf(big).length === 1);
  for (let i = 0; i < 8; i += 1) big = E.commit(big, { n: 200 + i }, { now: 8000 + i * 1000 });
  ok('once the ring drops its fork the branch is pruned', big.branches.length === 0 && big.entries.length === 6, `${big.branches.length} ${big.entries.length}`);

  // the count of branches is bounded
  let many = E.createHistory({ n: 0 }, { now: 0, cap: 50 });
  many = E.commit(many, { n: 1 }, { now: 1000 });
  for (let i = 0; i < E.MAX_BRANCHES + 5; i += 1) {
    many = E.commit(many, { n: 10 + i }, { now: 2000 + i * 1000 });
    many = E.jump(many, 1).h;                                           // back to the fork each time
  }
  ok('at most MAX_BRANCHES are parked', many.branches.length <= E.MAX_BRANCHES, String(many.branches.length));

  // a no-op commit makes no branch
  const nop = E.commit(E.jump(h, 2).h, { a: 2 }, { now: 1 });
  ok('committing the state you are already on parks nothing', E.branchesOf(nop).length === 0);
}

console.log('── history: what an entry changed ──');
{
  const mk = (before, state, label = 'x') => ({ id: 1, label, ts: 0, before, state });
  ok('a slider is a parameter edit', E.entryKind(mk({ settings: { blend_ratio: 1 } }, { settings: { blend_ratio: 0.8 } })) === 'parameter');
  ok('only expression keys is an expression edit',
    E.entryKind(mk({ settings: { expression_restore_strength: 0.2, pupil_gaze_retention: 0.1 } }, { settings: { expression_restore_strength: 0.6, pupil_gaze_retention: 0.1 } })) === 'expression');
  ok('an expression key AND a normal key is a parameter edit',
    E.entryKind(mk({ settings: { expression_restore_strength: 0.2, blend_ratio: 1 } }, { settings: { expression_restore_strength: 0.6, blend_ratio: 0.5 } })) === 'parameter');
  ok('a stroke is a brush edit', E.entryKind(mk({ maskStrokes: [] }, { maskStrokes: [{ x: 1 }] })) === 'brush');
  ok('a curve key is a curve edit', E.entryKind(mk({ curves: {} }, { curves: { c: { swap_opacity: [] } } })) === 'curve');
  ok('a roto shape is a roto edit', E.entryKind(mk({ roto: {} }, { roto: { m: [1] } })) === 'roto');
  ok('a pin is a pin edit', E.entryKind(mk({ pins: {} }, { pins: { m: [1] } })) === 'pin');
  ok('a slot recall is its own kind', E.entryKind(mk({ settings: { a: 1 } }, { settings: { a: 2 } }, 'Recalled Slot B')) === 'slot');
  ok('the first entry has nothing to compare to', E.entryKind({ id: 0, label: 'Initial state', before: null, state: { a: 1 } }) === 'other');
  ok('an identical-valued rebuilt object is not a change', E.changedChannels(mk({ settings: { a: 1 } }, { settings: { a: 1 } })).length === 0);
}

console.log('── history: the stack rows ──');
{
  let h = E.createHistory({ settings: { a: 0 }, maskStrokes: [] }, { now: 0 });
  h = E.commit(h, { settings: { a: 1 }, maskStrokes: [] }, { now: 1000, label: 'Changed A (0 -> 1)' });
  h = E.commit(h, { settings: { a: 1 }, maskStrokes: [{ p: 1 }] }, { now: 2000, label: 'Painted Occlusion Mask Stroke' });
  h = E.commit(h, { settings: { a: 7 }, maskStrokes: [{ p: 1 }] }, { now: 3000, label: 'Changed A (1 -> 7)' });
  const cur = E.jump(h, 1);                                             // sit on the first edit
  let h2 = E.commit(cur.h, { settings: { a: 5 }, maskStrokes: [] }, { now: 4000, label: 'Changed A (1 -> 5)' });
  const rows = M.buildRows(h2.entries, h2.index, E.branchesOf(h2));
  ok('rows are newest first', rows[0].i === h2.index && rows[rows.length - 1].base === true);
  ok('exactly one row is current', rows.filter((r) => r.current).length === 1 && rows[0].current);
  ok('rows carry a kind', rows[0].kind === 'parameter' && rows.at(-1).kind === 'other');
  const fork = rows.find((r) => r.i === 1);
  ok('the branch hangs off the row it forked from', fork.branches.length === 1 && fork.branches[0].count === 2 && fork.branches[0].label === 'Painted Occlusion Mask Stroke', JSON.stringify(fork.branches));
  ok('the switched-away stroke is a brush row once switched in',
    (() => { const s = E.switchBranch(h2, fork.branches[0].id); return M.buildRows(s.h.entries, s.h.index, E.branchesOf(s.h)).some((r) => r.kind === 'brush'); })());
  ok('position text', M.positionText(h2.entries, h2.index) === 'Step 2 of 2' && M.positionText([{}], 0) === 'No edits yet');
  ok('step text', M.stepsText(1) === '1 step' && M.stepsText(3) === '3 steps');
  ok('every kind has a chip and a colour', Object.values(M.KIND_META).every((m) => m.chip && /^#/.test(m.color)));
}

console.log('── history: the store ──');
{
  resetHistoryForTests();
  let live = { v: 0 };
  const apply = (v) => { live = v; };
  const unregister = useHistoryStore.getState().register('t', { apply, scope: 'global', describe: (b, a) => `v ${b?.v}->${a?.v}`, coalesce: false });
  const H = () => useHistoryStore.getState();
  H().observe('t', live);
  for (const v of [1, 2, 3]) { live = { v }; H().observe('t', live); await tick(); await tick(); }
  ok('three edits recorded', H().entries.length === 4 && H().index === 3, String(H().entries.length));
  H().jumpTo(1);                                                         // back on v=1; v=2,3 are the redo tail
  ok('time-travel applied v=1', live.v === 1 && H().canRedo);
  live = { v: 50 }; H().observe('t', live); await tick(); await tick();
  ok('a new edit parks the tail: one branch published on the store', H().branches.length === 1 && H().branches[0].count === 2 && !H().canRedo, JSON.stringify(H().branches));
  const id = H().branches[0].id;
  ok('switchBranch applies the branch tip to the owner', H().switchBranch(id) === true && live.v === 3 && H().index === H().entries.length - 1);
  ok('the path left behind is now the branch', H().branches.length === 1 && H().branches[0].count === 1);
  ok('switching back applies the original edit', H().switchBranch(H().branches[0].id) === true && live.v === 50);
  ok('an unknown branch is refused', H().switchBranch(12345) === false);
  unregister();
  resetHistoryForTests();
  ok('the test seam clears branches', H().branches.length === 0);
}

// ═══ curves ═════════════════════════════════════════════════════════════════
const K = await import('../src/components/faceswap/curves/curveModel.js');
const P = await import('../src/components/faceswap/curves/curvePieces.js');
const CH = await import('../src/components/faceswap/curves/curveHistory.js');
const { bezierEase } = await import('../src/components/faceswap/identityKeyframes.js');
const near = (a, b, tol = 1e-9) => Math.abs(a - b) <= tol;

console.log('── curves: custom Bezier segments ──');
{
  const op = K.paramById('swap_opacity');
  const key = (frame, value, interp = 'linear', extra = {}) => ({ frame, value, interp, ...extra });

  const n = K.normalizeTrack([key(1, 0, 'bezier', { bz: [0.8, -3, 0.2, 9] }), key(21, 1, 'linear', { bz: [0.1, 0.1, 0.1, 0.1] }), key(31, 0.5, 'bezier')], op, 50);
  ok('a bezier key keeps its handles, held to [0,1]', n[0].interp === 'bezier' && n[0].bz.join() === '0.8,0,0.2,1', JSON.stringify(n[0]));
  ok('a non-bezier key drops stray handles', n[1].bz === undefined);
  ok('a bezier key with no handles gets ease-in-out', n[2].bz.join() === '0.42,0,0.58,1');
  ok('malformed handles are repaired, not kept', K.cleanBz([1, 2]).join() === '0.42,0,0.58,1' && K.cleanBz(['x', 0, 0, 0]).join() === '0.42,0,0.58,1');

  // evaluation
  const hand = [key(1, 0, 'bezier', { bz: [0.8, 0.05, 0.9, 0.2] }), key(11, 1)];
  const lin = [key(1, 0, 'linear'), key(11, 1)];
  const evHand = K.makeEvaluator(hand);
  ok('a hand-shaped segment follows bezierEase exactly', near(evHand(6), bezierEase(0.5, 0.8, 0.05, 0.9, 0.2)) && near(evHand(3), bezierEase(0.2, 0.8, 0.05, 0.9, 0.2)));
  ok('and is not the straight line', Math.abs(evHand(4) - K.evaluate(lin, 4)) > 0.05);
  ok('handles on the diagonal ARE the straight line', (() => {
    const d = [key(1, 0, 'bezier', { bz: [1 / 3, 1 / 3, 2 / 3, 2 / 3] }), key(11, 1)];
    return [1, 2, 4, 6, 9, 11].every((f) => near(K.evaluate(d, f), K.evaluate(lin, f), 1e-6));
  })());
  ok('keys are still hit exactly and held outside', evHand(1) === 0 && evHand(11) === 1 && evHand(-5) === 0 && evHand(99) === 1);

  // switching to Custom Bezier must not move the curve
  const three = [key(1, 0.1, 'smooth'), key(20, 0.9, 'smooth'), key(40, 0.3, 'smooth'), key(60, 0.8, 'smooth')];
  const asBz = K.setInterp(three, 20, 'bezier');
  ok('smooth -> bezier seeds the handles from the live shape', asBz[1].interp === 'bezier' && asBz[1].bz.length === 4);
  ok('and the curve is unchanged there (to the handles\' 3-decimal precision)', [22, 27, 33, 39].every((f) => near(K.evaluate(asBz, f), K.evaluate(three, f), 2e-3)),
    [22, 27, 33, 39].map((f) => (K.evaluate(asBz, f) - K.evaluate(three, f)).toFixed(4)).join());
  const lin3 = [key(1, 0.2, 'linear'), key(30, 0.8, 'linear')];
  const lb = K.setInterp(lin3, 1, 'bezier');
  ok('linear -> bezier starts on the diagonal', lb[0].bz.join() === [1 / 3, 1 / 3, 2 / 3, 2 / 3].map((x) => Math.round(x * 1000) / 1000).join() && [5, 15, 25].every((f) => near(K.evaluate(lb, f), K.evaluate(lin3, f), 2e-3)));
  ok('bezier -> smooth drops the handles', K.setInterp(asBz, 20, 'smooth')[1].bz === undefined);
  ok('setting the same interp is the same list', K.setInterp(three, 20, 'smooth') === three);

  // handles
  ok('a hold, the last key and a flat segment have no handles',
    K.segmentHandles([key(1, 0, 'hold'), key(9, 1)], 0) === null && K.segmentHandles(lin, 1) === null && K.segmentHandles([key(1, 0.5), key(9, 0.5)], 0) === null);
  ok('a linear segment reports its diagonal', K.segmentHandles(lin, 0).map((x) => Math.round(x * 1000) / 1000).join() === '0.333,0.333,0.667,0.667');
  const sb = K.setSegmentBezier(three, 20, [0.5, 0, 0.5, 1]);
  ok('setSegmentBezier makes the segment after the key custom', sb[1].interp === 'bezier' && sb[1].bz.join() === '0.5,0,0.5,1' && sb[0] === three[0]);
  ok('the last key cannot be shaped, an unknown frame is refused', K.setSegmentBezier(three, 60, [0, 0, 1, 1]) === three && K.setSegmentBezier(three, 21, [0, 0, 1, 1]) === three);
  ok('the same handles are the same list', K.setSegmentBezier(sb, 20, [0.5, 0, 0.5, 1]) === sb);
  const a = three[1];
  const b = three[2];
  const pt = K.handlePoint(a, b, 0.25, 0.75);
  ok('a handle sits inside its segment\'s box', near(pt.frame, 20 + 0.25 * 20) && near(pt.value, 0.9 + 0.75 * (0.3 - 0.9)));
  const back = K.handleFromPoint(a, b, pt.frame, pt.value);
  ok('handleFromPoint inverts handlePoint (also on a falling segment)', near(back[0], 0.25) && near(back[1], 0.75));
  ok('a handle dragged outside the box is held on its edge', K.handleFromPoint(a, b, -50, 5).join() === '0,0');

  // render bake follows the hand-shaped curve
  const bakeTracks = { swap_opacity: [key(1, 0, 'bezier', { bz: [0.9, 0, 0.95, 0.1] }), key(40, 1)] };
  const baked = P.bakeCurvePieces([{ start: 1, end: 40, override: null }], bakeTracks);
  const ev = K.makeEvaluator(bakeTracks.swap_opacity);
  ok('the render bake cuts a custom curve into steps', baked.pieces.length > 3, String(baked.pieces.length));
  ok('each baked piece carries the evaluator\'s value at its middle', baked.pieces.every((pc) => near(pc.curves.swap_opacity, K.clampValue(op, ev((pc.start + pc.end) / 2)))));
  ok('a back-loaded curve is still near 0 for most of the clip (it is not linear)', baked.pieces.find((pc) => pc.start <= 20 && pc.end >= 20).curves.swap_opacity < 0.2);

  // identity morph compile
  const morph = K.morphToIdentityKeyframes([key(1, 0, 'bezier', { bz: [0.7, 0.1, 0.3, 0.9] }), key(20, 1)], [{ source_id: 'a' }, { source_id: 'b' }], 100);
  ok('a custom segment compiles to the backend\'s cubic_bezier with ITS handles', morph[0].curve === 'cubic_bezier' && morph[0].bezier.join() === '0.7,0.1,0.3,0.9');

  ok('history says "Reshaped" when only handles moved', CH.describeCurvesChange(
    { c: { swap_opacity: [key(1, 0, 'bezier', { bz: [0.2, 0, 0.8, 1] }), key(9, 1)] } },
    { c: { swap_opacity: [key(1, 0, 'bezier', { bz: [0.6, 0, 0.8, 1] }), key(9, 1)] } }) === 'Reshaped Swap Opacity Curve');
}

console.log('── curves: grid snapping ──');
{
  const op = K.paramById('swap_opacity');
  const px = K.paramById('pixel_boost');
  ok('value snap off just clamps', K.snapValue(op, 0.437, 0) === 0.44 && K.snapValue(op, 7, 0) === 1);
  ok('value snap to quarters', K.snapValue(op, 0.37, 4) === 0.25 && K.snapValue(op, 0.4, 4) === 0.5 && K.snapValue(op, 0.9, 4) === 1);
  ok('value snap to tenths', K.snapValue(op, 0.337, 10) === 0.3 && K.snapValue(op, 0.96, 10) === 1);
  ok('the grid follows the parameter\'s own range', K.snapValue(K.paramById('mask_feather'), 47, 4) === 50 && K.snapValue(px, 400, 4) === 384);
  ok('a signed range snaps through zero', K.snapValue(K.paramById('yaw_erosion_bias'), 0.04, 4) === 0 && K.snapValue(K.paramById('yaw_erosion_bias'), 0.3, 4) === 0.5);
  ok('frame snap off rounds, on lands on multiples', K.snapFrame(7.4, 1, 100) === 7 && K.snapFrame(7, 5, 100) === 5 && K.snapFrame(8, 5, 100) === 10 && K.snapFrame(13, 10, 100) === 10);
  ok('frame 1 and the last frame stay reachable', K.snapFrame(2, 5, 100) === 1 && K.snapFrame(99, 25, 100) === 100 && K.snapFrame(500, 5, 100) === 100);
  ok('the snap menus offer an off position', K.SNAP_VALUE_DIVS[0] === 0 && K.SNAP_FRAME_STEPS[0] === 1);
}

console.log('── curves: box selection and group edits ──');
{
  const key = (frame, value, interp = 'linear') => ({ frame, value, interp });
  const tracks = {
    swap_opacity: [key(10, 0.2), key(20, 0.5), key(30, 0.8), key(40, 1)],
    mask_feather: [key(15, 20), key(25, 100), key(35, 180)],          // 0.1, 0.5, 0.9 of 0..200
  };
  const box = (f0, f1, n0, n1) => K.keysInBox(tracks, ['swap_opacity', 'mask_feather'], { f0, f1, n0, n1 });
  ok('a box picks the keys inside it on both axes', [...box(12, 28, 0.4, 0.6)].sort().join() === 'mask_feather:25,swap_opacity:20');
  ok('corner order does not matter', [...box(28, 12, 0.6, 0.4)].sort().join() === [...box(12, 28, 0.4, 0.6)].sort().join());
  ok('an empty box picks nothing', box(1, 5, 0, 1).size === 0);
  ok('only the listed curves are searched', K.keysInBox(tracks, ['mask_feather'], { f0: 0, f1: 99, n0: 0, n1: 1 }).size === 3);
  ok('each curve is boxed on its OWN 0..1 scale', [...box(0, 99, 0.85, 1)].sort().join() === 'mask_feather:35,swap_opacity:40'    // 180px = 0.9 of its range; 0.8 of opacity is not
    && [...box(0, 99, 0.75, 0.85)].sort().join() === 'swap_opacity:30');

  const sel = new Set([K.selKey('swap_opacity', 20), K.selKey('swap_opacity', 30)]);
  ok('selection keys round-trip', K.parseSelKey(K.selKey('swap_opacity', 20)).paramId === 'swap_opacity' && K.parseSelKey(K.selKey('swap_opacity', 20)).frame === 20);
  ok('a group is held back by its unselected neighbours', (() => { const b = K.shiftBounds(tracks, sel, 100); return b.min === -9 && b.max === 9; })(), JSON.stringify(K.shiftBounds(tracks, sel, 100)));
  const moved = K.moveKeys(tracks, sel, 5, 0, 100);
  ok('the group slides rigidly', moved.swap_opacity.map((k) => k.frame).join() === '10,25,35,40' && moved.mask_feather === tracks.mask_feather);
  ok('and keeps its order and untouched keys', moved.swap_opacity[0] === tracks.swap_opacity[0] && moved.swap_opacity[3] === tracks.swap_opacity[3]);
  const tooFar = K.moveKeys(tracks, sel, 50, 0, 100);
  ok('a slide past a neighbour stops one frame short of it', tooFar.swap_opacity.map((k) => k.frame).join() === '10,29,39,40', tooFar.swap_opacity.map((k) => k.frame).join());
  const early = K.moveKeys(tracks, new Set([K.selKey('swap_opacity', 10)]), -50, 0, 100);
  ok('the first key stops on frame 1', early.swap_opacity[0].frame === 1);
  const late = K.moveKeys(tracks, new Set([K.selKey('swap_opacity', 40)]), 500, 0, 100);
  ok('the last key stops on the clip\'s last frame', late.swap_opacity[3].frame === 100);
  const cross = new Set([K.selKey('swap_opacity', 40), K.selKey('mask_feather', 35)]);
  ok('a group across curves moves as one block, held by the tightest limit', (() => {
    const m = K.moveKeys(tracks, cross, 500, 0, 100);
    return m.swap_opacity[3].frame === 100 && m.mask_feather[2].frame === 95;
  })());
  const up = K.moveKeys(tracks, new Set([K.selKey('swap_opacity', 20), K.selKey('mask_feather', 25)]), 0, 0.1, 100);
  ok('a vertical slide is a fraction of EACH curve\'s own range', near(up.swap_opacity[1].value, 0.6) && near(up.mask_feather[1].value, 120), `${up.swap_opacity[1].value} ${up.mask_feather[1].value}`);
  ok('values are held inside the parameter range', K.moveKeys(tracks, new Set([K.selKey('swap_opacity', 40)]), 0, 5, 100).swap_opacity[3].value === 1);
  ok('a move that moves nothing is the same map', K.moveKeys(tracks, sel, 0, 0, 100) === tracks && K.moveKeys(tracks, new Set(), 5, 0.1, 100) === tracks);
  ok('an unknown curve in the selection is ignored', K.moveKeys(tracks, new Set(['nope:3']), 5, 0, 100) === tracks);

  const rem = K.removeKeys(tracks, new Set([K.selKey('mask_feather', 15), K.selKey('mask_feather', 25), K.selKey('mask_feather', 35), K.selKey('swap_opacity', 10)]));
  ok('removing every key of a curve drops the curve', rem.mask_feather === undefined && rem.swap_opacity.length === 3);
  ok('removing nothing is the same map', K.removeKeys(tracks, new Set(['swap_opacity:999'])) === tracks);
  const ip = K.setInterpKeys(tracks, sel, 'hold');
  ok('a group interp change sets every selected key', ip.swap_opacity[1].interp === 'hold' && ip.swap_opacity[2].interp === 'hold' && ip.swap_opacity[0].interp === 'linear');
  ok('and is the same map when already so', K.setInterpKeys(ip, sel, 'hold') === ip);
  ok('no selected key crosses another curve\'s... (property: 200 random group moves keep every track sorted, unique, in range)', (() => {
    let seed = 7;
    const rnd = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
    for (let t = 0; t < 200; t += 1) {
      const trk = { swap_opacity: [], mask_feather: [] };
      for (const id of Object.keys(trk)) {
        const p = K.paramById(id);
        const frames = new Set();
        while (frames.size < 2 + Math.floor(rnd() * 6)) frames.add(1 + Math.floor(rnd() * 99));
        trk[id] = [...frames].sort((x, y) => x - y).map((f) => ({ frame: f, value: p.min + rnd() * (p.max - p.min), interp: 'linear' }));
      }
      const s = new Set();
      for (const id of Object.keys(trk)) for (const k of trk[id]) if (rnd() < 0.4) s.add(K.selKey(id, k.frame));
      const out = K.moveKeys(trk, s, Math.floor(rnd() * 200 - 100), rnd() * 2 - 1, 100);
      for (const id of Object.keys(out)) {
        const l = out[id];
        for (let i = 0; i < l.length; i += 1) {
          if (l[i].frame < 1 || l[i].frame > 100 || (i && l[i].frame <= l[i - 1].frame)) return false;
        }
        if (l.length !== trk[id].length) return false;
      }
    }
    return true;
  })());
}

// ═══ snapshot bank / quad compare ═══════════════════════════════════════════
const Q = await import('../src/components/faceswap/slotDelta.js');

console.log('── slots: where two slots differ ──');
{
  const slot = (id, settings, extra = {}, more = {}) => ({
    id, state: { settings, ...extra }, bitmap: more.bitmap === undefined ? {} : more.bitmap, frameKey: more.frameKey ?? 'clip_12', ts: more.ts ?? 1_700_000_000_000 + id.charCodeAt(0), label: `Slot ${id}`,
  });
  const A = slot('A', { blend_ratio: 1, face_mask_blend: 20, theme_name: 'dark', provider: 'cuda' });
  const B = slot('B', { blend_ratio: 0.7, face_mask_blend: 20, theme_name: 'light', provider: 'tensorrt' });
  const C = slot('C', { blend_ratio: 1, face_mask_blend: 20, theme_name: 'dark', provider: 'cuda' }, { curves: { clip: { swap_opacity: [{ frame: 1, value: 0.5 }] } } });
  const D = slot('D', { blend_ratio: 0.4, face_mask_blend: 60, expression_restore_strength: 0.9 }, {}, { bitmap: null, frameKey: 'clip_99' });

  const d = Q.settingDeltas(A, B);
  ok('a changed slider is one delta, with labels and values', d.length === 1 && d[0].key === 'blend_ratio' && d[0].fromText === '1' && d[0].toText === '0.70' && d[0].label.length > 0, JSON.stringify(d));
  ok('machine settings (theme, provider) are never a difference between looks', !d.some((x) => /theme|provider/.test(x.key)));
  ok('identical recipes have no deltas', Q.settingDeltas(A, A).length === 0 && Q.settingDeltas(A, C).length === 0);
  ok('a key present in only one slot is a difference', Q.settingDeltas(A, D).some((x) => x.key === 'expression_restore_strength' && x.fromText === 'none' && x.toText === '0.90'));
  ok('non-settings channels are reported by name', Q.channelDeltas(A, C).join() === 'Parameter curves' && Q.channelDeltas(A, B).length === 0);
  ok('a channel in neither slot is not a difference', Q.channelDeltas(A, A).length === 0);
  ok('frame and clock come from the slot', Q.slotFrame(A) === 12 && Q.slotFrame({}) === null && /\d/.test(Q.slotClock(A.ts)) && Q.slotClock(0) === '');

  const slots = { A, B, C, D };
  ok('the baseline defaults to the first filled slot', Q.pickBaseline(slots) === 'A' && Q.pickBaseline({ A: null, B, C: null, D: null }) === 'B');
  ok('a requested baseline wins when filled, else falls back', Q.pickBaseline(slots, 'C') === 'C' && Q.pickBaseline({ A, B: null, C: null, D: null }, 'C') === 'A' && Q.pickBaseline({ A: null, B: null, C: null, D: null }) === null);

  const quad = Q.buildQuad(slots, 'A', 'clip_12');
  ok('four quadrants in A B C D order', quad.map((q) => q.id).join('') === 'ABCD');
  ok('the baseline carries no deltas and says so', quad[0].baseline && quad[0].deltas.length === 0 && !quad[0].same);
  ok('B differs by exactly the slider', quad[1].deltas.length === 1 && quad[1].deltas[0].key === 'blend_ratio' && !quad[1].same);
  ok('C differs only by its curves', quad[2].deltas.length === 0 && quad[2].channels.join() === 'Parameter curves' && !quad[2].same);
  ok('D differs by two settings, has no picture and is of another frame', quad[3].deltas.length === 3 && !quad[3].hasPicture && quad[3].otherFrame && quad[3].frame === 99);
  ok('a slot on the current frame is not flagged', quad[0].otherFrame === false);
  const twin = Q.buildQuad({ A, B: { ...A, id: 'B' }, C: null, D: null }, 'A', 'clip_12');
  ok('an identical recipe is flagged "same"', twin[1].same === true && twin[2].empty && twin[2].deltas.length === 0 && !twin[2].same);
  ok('moving the baseline recomputes every delta', Q.buildQuad(slots, 'B', 'clip_12')[0].deltas.some((x) => x.key === 'blend_ratio' && x.fromText === '0.70' && x.toText === '1'));
  ok('an empty bank has four empty quadrants', Q.buildQuad({ A: null, B: null, C: null, D: null }).every((q) => q.empty && !q.baseline));

  const table = Q.diffTable(slots, 'A');
  const row = table.find((r) => r.key === 'blend_ratio');
  ok('the table has a row per setting that differs anywhere', table.map((r) => r.key).sort().join() === 'blend_ratio,expression_restore_strength,face_mask_blend');
  ok('a row shows every slot\'s own value', row.values.A === '1' && row.values.B === '0.70' && row.values.C === '1' && row.values.D === '0.40');
  ok('and which slots differ from the baseline', [...row.differs].sort().join() === 'B,D');
  ok('an empty slot shows nothing', Q.diffTable({ A, B, C: null, D: null }, 'A').every((r) => r.values.C === '' && r.values.D === ''));
  ok('one filled slot has nothing to compare', Q.diffTable({ A, B: null, C: null, D: null }).length === 0);

  const s1 = Q.deltaSummary(quad[1], 'A');
  ok('summary: count, then the first few as "label: from → to"', s1.headline === '1 difference from A' && /→/.test(s1.names[0]) && s1.more === 0);
  ok('summary: caps the list and counts the rest', (() => { const s = Q.deltaSummary(quad[3], 'A', 2); return s.names.length === 2 && s.more === 1 && s.headline === '3 differences from A'; })());
  ok('summary: baseline / same / empty', Q.deltaSummary(quad[0], 'A').headline === 'Baseline' && Q.deltaSummary(twin[1], 'A').headline === 'Same recipe as A' && Q.deltaSummary(twin[2], 'A').headline === 'Empty slot');
}

// ═══ face menu: placement, roles, pose, reset ═══════════════════════════════
const FM = await import('../src/components/faceswap/faceContextModel.js');
const RC = await import('../src/components/faceswap/faceRoleChoices.js');
const FR = await import('../src/components/faceswap/faceRoles.js');

console.log('── faces: the menu opens beside the face ──');
{
  const rect = { left: 400, top: 200, right: 520, bottom: 330 };
  const base = { rect, w: 300, h: 400, vw: 1280, vh: 720 };
  const r = FM.placeBesideRect(base);
  ok('it opens on the right of the box, a gap away, top-aligned', r.side === 'right' && r.left === 530 && r.top === 200, JSON.stringify(r));
  const l = FM.placeBesideRect({ ...base, rect: { left: 1000, top: 200, right: 1100, bottom: 330 } });
  ok('with no room on the right it flips to the left of the box', l.side === 'left' && l.left === 1000 - 10 - 300 && l.left + 300 <= 1000, JSON.stringify(l));
  const low = FM.placeBesideRect({ ...base, rect: { left: 400, top: 600, right: 520, bottom: 700 } });
  ok('a box near the bottom keeps the menu on screen', low.top + 400 <= 720 - 8 && low.top >= 8, JSON.stringify(low));
  const huge = FM.placeBesideRect({ rect: { left: 20, top: 10, right: 1260, bottom: 710 }, w: 300, h: 400, vw: 1280, vh: 720, point: { x: 640, y: 360 } });
  ok('a box that fills the window falls back to the click point, still on screen', huge.side === 'point' && huge.left >= 8 && huge.left + 300 <= 1280 - 8 && huge.top >= 8 && huge.top + 400 <= 720 - 8, JSON.stringify(huge));
  const tall = FM.placeBesideRect({ ...base, h: 900 });
  ok('a menu taller than the window is pinned to the top and told to scroll', tall.top === 8 && tall.maxHeight === 720 - 16, JSON.stringify(tall));
  ok('the menu never overlaps the box when placed beside it', [r, l].every((p) => p.left >= rect.right || p.left + 300 <= rect.left));
  let every = true;
  for (let vw = 400; vw <= 1600; vw += 100) {
    for (let x = 0; x < vw - 60; x += 120) {
      const p = FM.placeBesideRect({ rect: { left: x, top: 100, right: x + 60, bottom: 220 }, w: 300, h: 380, vw, vh: 600, point: { x: x + 30, y: 150 } });
      if (p.left < 8 - 1e-9 || p.left + 300 > vw - 8 + 1e-9) every = false;
    }
  }
  ok('across window widths and box positions it is always fully horizontally on screen', every);
}

console.log('── faces: zoom to a face, pose to the frontalize gate ──');
{
  const img = { w: 1000, h: 500 };
  const stage = { w: 800, h: 400 };                                       // fit = 0.8
  const z = FM.faceFitZoom({ bbox: [400, 150, 500, 300], imgDim: img, stage });
  ok('a face fills 60% of the stage\'s shorter side', Math.abs(z - (0.6 * Math.min(800 / (100 * 0.8), 400 / (150 * 0.8)))) < 0.01 || z === 8, String(z));
  ok('the zoom is held to the range, and is 2 when anything is unknown', FM.faceFitZoom({ bbox: [0, 0, 1000, 500], imgDim: img, stage }) === 1
    && FM.faceFitZoom({ bbox: [400, 400, 400.5, 400.5], imgDim: img, stage }) === 2 && FM.faceFitZoom({ bbox: null, imgDim: img, stage }) === 2 && FM.faceFitZoom({ bbox: [1, 1, 50, 50], imgDim: null, stage }) === 2);
  ok('a tiny face zooms no more than 8x', FM.faceFitZoom({ bbox: [10, 10, 14, 14], imgDim: img, stage }) <= 8);

  const p = FM.poseToFilterAngle([12.4, -38.6, 3]);
  ok('the pose angle is the larger of |yaw| and |pitch|, on the slider\'s 5 degree step', p.angle === 40 && p.exact === 38.6 && p.yaw === 12.4);
  ok('and is held to the slider\'s 10..60 range', FM.poseToFilterAngle([2, 1, 0]).angle === 10 && FM.poseToFilterAngle([-88, 0, 0]).angle === 60);
  ok('an {yaw, pitch} object works too', FM.poseToFilterAngle({ yaw: 31, pitch: 4 }).angle === 30);
  ok('an unmeasured pose is null (never an invented angle)', FM.poseToFilterAngle(null) === null && FM.poseToFilterAngle([]) === null && FM.poseToFilterAngle(['x', 1]) === null && FM.poseToFilterAngle(undefined) === null);
}

console.log('── faces: the four roles ──');
{
  ok('four roles, each with a label, colour and a hint', RC.ROLE_CHOICES.length === 4 && RC.ROLE_CHOICES.every((c) => c.label && c.short && /^#/.test(c.color) && c.hint.length > 20));
  ok('the labels are the ones in the brief', RC.ROLE_CHOICES.map((c) => c.label).join('|') === 'Primary Swap Target|Secondary Target|Reference Only|Exclude (Never Swap)');
  ok('Reference Only says it renders like Exclude', /exactly like Exclude/i.test(RC.choiceMeta('reference').hint));

  const of = (o) => RC.choiceOf({ personId: 'p1', selectedTargetPersonId: 'p1', ...o });
  ok('the pinned target is primary', of({ role: 'target' }) === 'primary');
  ok('a swapped person who is not the pinned target is secondary', of({ role: 'swapped', personId: 'p2' }) === 'secondary');
  ok('a swapped person who IS the selected target is primary', of({ role: 'swapped', personId: 'p1' }) === 'primary');
  ok('an excluded person is exclude, or reference when tagged', of({ role: 'bystander', excluded: true }) === 'exclude' && of({ role: 'bystander', excluded: true, tagged: true }) === 'reference');
  ok('a tag on a person who is NOT excluded means nothing', of({ role: 'bystander', excluded: false, tagged: true }) === null);
  ok('an unmapped bystander has no role', of({ role: 'bystander' }) === null);

  const sw = { role: 'swapped', choice: 'secondary', reason: '' };
  ok('picking the current role is refused', RC.choiceBlocked('secondary', sw) !== '');
  ok('picking another role is allowed', RC.choiceBlocked('exclude', { ...sw, canExclude: true }) === '' && RC.choiceBlocked('primary', sw) === '');
  ok('secondary needs a source', /source face/.test(RC.choiceBlocked('secondary', { role: 'bystander', choice: null }, { hasSources: false })));
  const target = { role: 'target', choice: 'primary', reason: 'This is the pinned target. Pin another face to change it.' };
  ok('the pinned target cannot be excluded or made reference, with the reason', /pinned target/.test(RC.choiceBlocked('exclude', target)) && /pinned target/.test(RC.choiceBlocked('reference', target)));
  ok('the primary cannot be made secondary', /primary/.test(RC.choiceBlocked('secondary', target)));
  ok('a face nobody captured can be given any role (capturing it is the first step)', ['primary', 'secondary', 'exclude', 'reference'].every((c) => RC.choiceBlocked(c, { role: 'bystander', choice: null }) === ''));
  ok('the on-canvas badge', RC.badgeFor('exclude').text === 'Exclude' && RC.badgeFor(null) === null && RC.badgeFor('nope') === null);

  // the real faceRole() output feeds choiceOf(): excluded via the mapping really reads as excluded
  const mapping = FR.excludePerson({}, 'p1');
  const info = FR.faceRole({ personId: 'p1', faceSelection: 'Selected people', selectedTargetPersonId: null, faceMapping: mapping, targetGroups: [], targetPersonIds: ['p1'], sourceCount: 2, sourceIdentityIds: ['a', 'b'] });
  ok('faceRoles + choiceOf agree: an excluded person is Exclude', info.excluded === true && RC.choiceOf({ ...info, personId: 'p1' }) === 'exclude');

  const rs = RC.resetSummary({ overrides: 2, pins: 1, mapped: true, tagged: false });
  ok('the reset summary counts and names what goes', rs.total === 4 && /2 per-person overrides/.test(rs.text) && /1 identity pin\b/.test(rs.text) && /source \/ skip/.test(rs.text));
  ok('and says nothing when there is nothing', RC.resetSummary({}).total === 0 && RC.resetSummary({}).text === '');
}

globalThis.__RENDER_CHECK_FAILURES__ = failures;
console.log(failures === 0 ? '\nstage 12 logic: all passed' : `\nstage 12 logic: ${failures} FAILED`);
