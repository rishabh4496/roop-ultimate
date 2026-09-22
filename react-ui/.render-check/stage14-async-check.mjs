// Stage 14: stale asynchronous state — delayed-response races on the client.
//
// These import and RUN the real modules the UI ships (processingSelection.js,
// faceMapping.js). The race section replays FaceSwap.refreshPreview's control
// flow — single flight, coalesced pending request, wanted-selection ref,
// classifyPreviewResponse verdict — against a fake /api/preview whose replies
// resolve out of order with real timers. The source-level section then checks
// that FaceSwap.jsx actually wires that flow the way the model does, so the
// model cannot quietly drift from the component.
//
// Run with: node .render-check/stage14-async-check.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import process from 'node:process';
import {
  buildProcessingSelection,
  classifyPreviewResponse,
  nextSelectionVersion,
  reconcileRestoredSelection,
  sameSelectionIdentity,
  selectionIdentity,
} from '../src/components/faceswap/processingSelection.js';
import {
  buildTargetSelectionState,
  defaultSourceIndexForTarget,
} from '../src/components/faceswap/faceMapping.js';

const here = dirname(fileURLToPath(import.meta.url));

let pass = 0;
const fails = [];
const check = async (name, fn) => {
  try { await fn(); pass++; console.log(`  PASS  ${name}`); }
  catch (e) { fails.push(name); console.log(`  FAIL  ${name}\n        ${e.message}`); }
};
const group = (t) => console.log(`\n── ${t} ${'─'.repeat(Math.max(0, 56 - t.length))}`);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ── A UI context, the way FaceSwap.jsx holds it ──────────────────────────
const SOURCES = ['src-a', 'src-b', 'src-c'];
const ctxA = { mediaId: 'media-A', groups: [0, 1], people: ['tp_a1', 'tp_a2'], refs: ['tr_a1', 'tr_a2'] };
const ctxB = { mediaId: 'media-B', groups: [0, 1], people: ['tp_b1', 'tp_b2'], refs: ['tr_b1', 'tr_b2'] };

const selectionFor = (ctx, { person, source = 0, mapping = {}, mode = 'Selected face', version = 1 }) => {
  const selectionState = buildTargetSelectionState({
    faceSelection: mode, targetGroups: ctx.groups, faceMapping: mapping,
    sourceCount: SOURCES.length, selTargetFace: ctx.people.indexOf(person),
    selectedSource: source, targetPersonIds: ctx.people,
    selectedTargetPersonId: person, sourceIdentityIds: SOURCES,
  });
  return buildProcessingSelection({
    targetMediaId: ctx.mediaId, targetGroups: ctx.groups, targetPersonIds: ctx.people,
    faceMapping: mapping, sourceCount: SOURCES.length, sourceIdentityIds: SOURCES,
    faceSelection: mode, selectedTargetPersonId: person,
    selectedReferenceFaceId: ctx.refs[ctx.people.indexOf(person)],
    selectedSource: source, selectionState, selectionVersion: version,
  });
};

// ── Canonical selection ──────────────────────────────────────────────────
group('Canonical selection');
await check('carries media, person, source identity, mapping, mode and version', () => {
  const s = selectionFor(ctxA, { person: 'tp_a2', source: 1, version: 7 });
  assert.equal(s.target_media_id, 'media-A');
  assert.equal(s.target_person_id, 'tp_a2');
  assert.deepEqual(s.target_person_ids, ['tp_a2']);
  assert.equal(s.target_reference_face_id, 'tr_a2');
  assert.equal(s.source_identity_id, 'src-b');
  assert.equal(s.detection_mode, 'Selected face');
  assert.deepEqual(s.target_person_source_mapping, { tp_a2: 'src-b' });
  assert.equal(s.selection_version, 7);
  assert.ok(s.request_id);
  assert.equal(s.selection_state.selection_mode, 'selected');
  assert.equal(s.selection_state.person_id, 'tp_a2');
});
await check('two builds from one context differ only by request id', () => {
  const a = selectionFor(ctxA, { person: 'tp_a1' });
  const b = selectionFor(ctxA, { person: 'tp_a1' });
  assert.notEqual(a.request_id, b.request_id);
  assert.ok(sameSelectionIdentity(a, b));
  assert.deepEqual(selectionIdentity(a), selectionIdentity(b));
});
await check('person, source and media changes each change the identity', () => {
  const base = selectionFor(ctxA, { person: 'tp_a1', source: 0 });
  assert.ok(!sameSelectionIdentity(base, selectionFor(ctxA, { person: 'tp_a2', source: 0 })));
  assert.ok(!sameSelectionIdentity(base, selectionFor(ctxA, { person: 'tp_a1', source: 2 })));
  assert.ok(!sameSelectionIdentity(base, selectionFor(ctxB, { person: 'tp_b1', source: 0 })));
});
await check('a selected person unknown to the context is not sent as identity', () => {
  const s = selectionFor(ctxA, { person: 'tp_deleted' });
  assert.equal(s.target_person_id, null);
  assert.deepEqual(s.target_person_ids, []);
});
await check('selection versions are strictly increasing across bumps and reloads', () => {
  let v = nextSelectionVersion(0);
  const first = v;
  for (let i = 0; i < 50; i += 1) { const n = nextSelectionVersion(v); assert.ok(n > v); v = n; }
  assert.ok(nextSelectionVersion(0) >= first, 'a fresh session must not fall behind an old one');
});

// ── The preview flow model (mirrors FaceSwap.refreshPreview) ─────────────
// `server(body)` returns a promise of the backend reply; the harness echoes
// the request's identity like app/api.py's _selection_response_fields does.
function makePreviewFlow(server) {
  const flow = {
    ui: { ctx: ctxA, person: 'tp_a1', source: 0, frame: 1, mapping: {} },
    displayed: null,            // what the stage shows
    cache: new Map(),           // key -> response
    log: [],
    busy: false, pending: null, seq: 0,
    wanted() {
      const s = selectionFor(this.ui.ctx, { person: this.ui.person, source: this.ui.source, mapping: this.ui.mapping });
      const key = `${this.ui.ctx.mediaId}_${this.ui.frame}_${JSON.stringify(selectionIdentity(s))}`;
      return { key, mediaId: this.ui.ctx.mediaId, frame: this.ui.frame };
    },
    inFlight: [],
    async refresh(opts = {}) {
      // Like the component: while busy only the caller's raw opts are kept;
      // the re-dispatch resolves everything else from the ui as it is THEN.
      if (this.busy) { this.pending = { ...opts }; return; }
      const ctx = opts.ctx || this.ui.ctx;
      const frame = opts.frame ?? this.ui.frame;
      this.busy = true;
      this.seq += 1;
      const selection = selectionFor(ctx, { person: this.ui.person, source: this.ui.source, mapping: this.ui.mapping });
      const request = {
        seq: this.seq, requestId: selection.request_id,
        key: `${ctx.mediaId}_${frame}_${JSON.stringify(selectionIdentity(selection))}`,
        mediaId: ctx.mediaId, frame, selection,
      };
      const p = (async () => {
        try {
          const res = await server({ frame, processing_selection: selection, target_media_id: ctx.mediaId });
          const verdict = classifyPreviewResponse({ request, response: res, wanted: this.wanted() });
          this.log.push({ seq: request.seq, ...verdict, person: selection.target_person_id, frame });
          if (!verdict.accept) {
            if (verdict.cache) this.cache.set(request.key, res);
            return;
          }
          this.displayed = res;
          this.cache.set(request.key, res);
        } finally {
          this.busy = false;
          if (this.pending) { const n = this.pending; this.pending = null; this.refresh(n); }
        }
      })();
      this.inFlight.push(p);
    },
    async settle() {
      // Wait until nothing is in flight or pending.
      for (let i = 0; i < 200; i += 1) {
        await Promise.all(this.inFlight);
        await sleep(5);
        if (!this.busy && !this.pending) return;
      }
      throw new Error('flow did not settle');
    },
  };
  return flow;
}

// A backend whose reply time is chosen per request, echoing identity like
// api.py does (request_id, target_media_id, frame, processing_selection).
const makeServer = (delayFor) => async (body) => {
  const delay = delayFor(body);
  await sleep(delay);
  const s = body.processing_selection;
  return {
    request_id: s.request_id, target_media_id: body.target_media_id, frame: body.frame,
    processing_selection: s, image: `img:${s.target_person_id}:${s.source_identity_id}:${body.target_media_id}:f${body.frame}`,
    preview_signature: `sig:${JSON.stringify(selectionIdentity(s))}:${body.frame}`,
  };
};

group('RACE 1: preview A, then B; A returns after B');
await check('B remains current, A is never displayed', async () => {
  // Two genuinely concurrent requests (no single flight) so A can land last.
  const flow = makePreviewFlow(makeServer((b) => (b.processing_selection.target_person_id === 'tp_a1' ? 60 : 10)));
  flow.busy = false;
  const send = async (person) => {
    flow.ui.person = person;
    const selection = selectionFor(ctxA, { person });
    const request = { requestId: selection.request_id, key: `${ctxA.mediaId}_1_${JSON.stringify(selectionIdentity(selection))}`, mediaId: ctxA.mediaId, frame: 1, selection };
    const res = await makeServer((b) => (b.processing_selection.target_person_id === 'tp_a1' ? 60 : 10))({ frame: 1, processing_selection: selection, target_media_id: ctxA.mediaId });
    const verdict = classifyPreviewResponse({ request, response: res, wanted: flow.wanted() });
    if (verdict.accept) flow.displayed = res;
    else if (verdict.cache) flow.cache.set(request.key, res);
    return verdict;
  };
  const pa = send('tp_a1');           // slow
  await sleep(1);
  const pb = send('tp_a2');           // fast, and now the wanted person
  const [va, vb] = await Promise.all([pa, pb]);
  assert.equal(vb.accept, true);
  assert.equal(va.accept, false);
  assert.equal(va.reason, 'superseded');
  assert.equal(flow.displayed.image, 'img:tp_a2:src-a:media-A:f1');
  // A's late answer was filed under A's own key, never under B's.
  const keys = [...flow.cache.keys()];
  assert.equal(keys.length, 1);
  assert.ok(keys[0].includes('"target_person_id":"tp_a1"'));
});

group('RACE 1 through the single-flight flow (coalesced)');
await check('A in flight, switch to B: A is discarded, B is rendered once', async () => {
  const flow = makePreviewFlow(makeServer(() => 20));
  await flow.refresh();                     // A (tp_a1) in flight
  flow.ui.person = 'tp_a2';                 // user clicks B
  await flow.refresh();                     // coalesced as pending
  await flow.settle();
  assert.equal(flow.displayed.image, 'img:tp_a2:src-a:media-A:f1');
  const verdicts = flow.log.map((l) => `${l.person}:${l.reason}`);
  assert.deepEqual(verdicts, ['tp_a1:superseded', 'tp_a2:current']);
});

group('RACE 2: preview A running, switch target media to B');
await check('A result is discarded; B is what the stage shows', async () => {
  const flow = makePreviewFlow(makeServer(() => 20));
  await flow.refresh();                     // media-A, tp_a1
  flow.ui.ctx = ctxB; flow.ui.person = 'tp_b1'; flow.ui.frame = 1;
  await flow.refresh({ ctx: ctxB, frame: 1 });
  await flow.settle();
  assert.equal(flow.displayed.target_media_id, 'media-B');
  assert.equal(flow.log[0].accept, false);
  assert.equal(flow.log[0].reason, 'superseded');
});
await check('a response whose media does not match the request is dropped, not cached', () => {
  const selection = selectionFor(ctxA, { person: 'tp_a1' });
  const verdict = classifyPreviewResponse({
    request: { requestId: selection.request_id, key: 'k', mediaId: 'media-A', frame: 1, selection },
    response: { request_id: selection.request_id, target_media_id: 'media-B', frame: 1 },
    wanted: { key: 'k', mediaId: 'media-A', frame: 1 },
  });
  assert.deepEqual(verdict, { accept: false, cache: false, reason: 'media_mismatch' });
});
await check('a response answering a different request id is dropped, not cached', () => {
  const selection = selectionFor(ctxA, { person: 'tp_a1' });
  const verdict = classifyPreviewResponse({
    request: { requestId: 'req-1', key: 'k', mediaId: 'media-A', frame: 1, selection },
    response: { request_id: 'req-2', target_media_id: 'media-A', frame: 1 },
    wanted: { key: 'k', mediaId: 'media-A', frame: 1 },
  });
  assert.deepEqual(verdict, { accept: false, cache: false, reason: 'request_id_mismatch' });
});
await check('the server echoing a different person than asked is dropped', () => {
  const selection = selectionFor(ctxA, { person: 'tp_a1' });
  const verdict = classifyPreviewResponse({
    request: { requestId: selection.request_id, key: 'k', mediaId: 'media-A', frame: 1, selection },
    response: { request_id: selection.request_id, target_media_id: 'media-A', frame: 1,
      processing_selection: { ...selection, target_person_id: 'tp_a2' } },
    wanted: { key: 'k', mediaId: 'media-A', frame: 1 },
  });
  assert.equal(verdict.accept, false);
  assert.equal(verdict.cache, false);
});
await check('a diagnostic answer (server person null) for the wanted request is still shown', () => {
  const selection = selectionFor(ctxA, { person: 'tp_a1' });
  const verdict = classifyPreviewResponse({
    request: { requestId: selection.request_id, key: 'k', mediaId: 'media-A', frame: 1, selection },
    response: { request_id: selection.request_id, target_media_id: 'media-A', frame: 1,
      selection_diagnostic: 'invalid_person_id',
      processing_selection: { ...selection, target_person_id: null } },
    wanted: { key: 'k', mediaId: 'media-A', frame: 1 },
  });
  assert.equal(verdict.accept, true);
});

group('Source A -> source B');
await check('new target previews pair with its own source by target index', () => {
  assert.equal(defaultSourceIndexForTarget(0, 2), 0);
  assert.equal(defaultSourceIndexForTarget(1, 2), 1);
  assert.equal(defaultSourceIndexForTarget(2, 2), 0);
  assert.equal(defaultSourceIndexForTarget(null, 2, 1), 1);
});
await check('the source-A answer is discarded; source B is displayed', async () => {
  const flow = makePreviewFlow(makeServer(() => 20));
  await flow.refresh();
  flow.ui.source = 2;
  await flow.refresh();
  await flow.settle();
  assert.equal(flow.displayed.image, 'img:tp_a1:src-c:media-A:f1');
  assert.deepEqual(flow.log.map((l) => l.reason), ['superseded', 'current']);
});

group('Frame 1 -> 2 -> 3');
await check('only frame 3 is displayed; frames 1 is cached under its own key, 2 never requested', async () => {
  const flow = makePreviewFlow(makeServer(() => 15));
  await flow.refresh({ frame: 1 });
  flow.ui.frame = 2; await flow.refresh({ frame: 2 });
  flow.ui.frame = 3; await flow.refresh({ frame: 3 });   // replaces pending 2
  await flow.settle();
  assert.equal(flow.displayed.image, 'img:tp_a1:src-a:media-A:f3');
  // The cache key embeds the frame, so a frame change is a key change first.
  assert.deepEqual(flow.log.map((l) => `${l.frame}:${l.reason}`), ['1:superseded', '3:current']);
  assert.equal([...flow.cache.keys()].filter((k) => k.startsWith('media-A_1_')).length, 1);
  assert.equal([...flow.cache.keys()].filter((k) => k.startsWith('media-A_2_')).length, 0);
});

group('RACE 7: rapid A -> B -> A');
await check('the newest request is authoritative; nothing stale is displayed', async () => {
  const flow = makePreviewFlow(makeServer(() => 20));
  await flow.refresh();                    // A in flight
  flow.ui.person = 'tp_a2'; await flow.refresh();   // pending B
  flow.ui.person = 'tp_a1'; await flow.refresh();   // pending replaced by A again
  await flow.settle();
  assert.equal(flow.displayed.image, 'img:tp_a1:src-a:media-A:f1');
  // The first A answer IS the wanted selection again by the time it lands:
  // identity equality (not request order) is what makes it authoritative.
  assert.equal(flow.log[0].accept, true);
  // Then the coalesced re-request for A: also current.
  assert.ok(flow.log.every((l) => l.accept));
  assert.ok(flow.log.every((l) => l.person === 'tp_a1'), 'B was never rendered');
});
await check('A -> B -> A with a slow first A and the second A resolving first', async () => {
  let n = 0;
  const flow = makePreviewFlow(makeServer(() => { n += 1; return n === 1 ? 40 : 5; }));
  await flow.refresh();
  flow.ui.person = 'tp_a2'; await flow.refresh();
  flow.ui.person = 'tp_a1'; await flow.refresh();
  await flow.settle();
  assert.equal(flow.displayed.image, 'img:tp_a1:src-a:media-A:f1');
  assert.ok(flow.log.every((l) => l.person === 'tp_a1'));
});

group('RACE 3 (client half): the queued job carries the selection as of queueing');
await check('a job built while B is shown keeps B after the UI moves to A', () => {
  const jobSelection = selectionFor(ctxB, { person: 'tp_b2', source: 2, version: 10 });
  const job = { target_media_id: 'media-B', processing_selection: jobSelection, payload: { processing_selection: jobSelection } };
  // The UI moves on…
  const later = selectionFor(ctxA, { person: 'tp_a1', source: 0, version: 11 });
  assert.ok(!sameSelectionIdentity(job.processing_selection, later));
  assert.equal(job.processing_selection.target_person_id, 'tp_b2');
  assert.equal(job.processing_selection.source_identity_id, 'src-c');
  assert.deepEqual(job.processing_selection.target_person_source_mapping, { tp_b2: 'src-c' });
});

group('RACE 5/6: session restore reconciliation');
await check('exact person/reference/mapping are restored when the backend still knows them', () => {
  const r = reconcileRestoredSelection({
    savedPersonId: 'tp_a2', savedReferenceId: 'tr_a2', savedMapping: { tp_a2: 'src-b' },
    personIds: ['tp_a1', 'tp_a2'], referenceIds: ['tr_a1', 'tr_a2'], sourceIdentityIds: SOURCES,
    serverPersonId: 'tp_a2', serverReferenceId: 'tr_a2',
  });
  assert.deepEqual(r, { selectedTargetPersonId: 'tp_a2', selectedReferenceFaceId: 'tr_a2', faceMapping: { tp_a2: 'src-b' } });
});
await check('the backend\'s selected person wins over a stale client memory', () => {
  const r = reconcileRestoredSelection({
    savedPersonId: 'tp_a1', personIds: ['tp_a1', 'tp_a2'], serverPersonId: 'tp_a2',
  });
  assert.equal(r.selectedTargetPersonId, 'tp_a2');
});
await check('a deleted person cannot come back from client memory', () => {
  const r = reconcileRestoredSelection({
    savedPersonId: 'tp_a2', savedReferenceId: 'tr_a2', savedMapping: { tp_a2: 'src-b', tp_a1: 'src-a' },
    personIds: ['tp_a1'], referenceIds: ['tr_a1'], sourceIdentityIds: SOURCES,
    serverPersonId: null, serverReferenceId: null,
  });
  assert.equal(r.selectedTargetPersonId, null);
  assert.equal(r.selectedReferenceFaceId, null);
  assert.deepEqual(r.faceMapping, { tp_a1: 'src-a' });
});
await check('a mapping to a deleted source is dropped', () => {
  const r = reconcileRestoredSelection({
    savedMapping: { tp_a1: 'src-gone' }, personIds: ['tp_a1'], sourceIdentityIds: SOURCES,
  });
  assert.deepEqual(r.faceMapping, {});
});

// ── Source-level wiring: the component must use the model above ──────────
group('FaceSwap.jsx wiring');
const faceSwap = readFileSync(join(here, '..', 'src', 'components', 'FaceSwap.jsx'), 'utf8');
const refresh = faceSwap.slice(faceSwap.indexOf('const refreshPreview = async'), faceSwap.indexOf('// ── Comparison-grid preview loaders'));
await check('refreshPreview captures the request before awaiting and judges with classifyPreviewResponse', () => {
  const capture = refresh.indexOf('const request = {');
  const post = refresh.indexOf("await postJSON('/api/preview'");
  const judge = refresh.indexOf('classifyPreviewResponse({');
  assert.ok(capture > 0 && post > capture && judge > post, 'request snapshot must precede the await, verdict must follow it');
  assert.ok(refresh.includes('wanted: wantedPreviewRef.current'));
});
await check('the coalesced request is re-dispatched through the newest closure with raw opts only', () => {
  assert.ok(refresh.includes('previewPendingRef.current = { ...opts };'));
  assert.ok(refresh.includes('(refreshPreviewRef.current || refreshPreview)(next)'));
  assert.ok(faceSwap.includes('refreshPreviewRef.current = refreshPreview;'));
});
await check('nothing reaches the stage or the live cache key before verdict.accept', () => {
  const judge = refresh.indexOf('classifyPreviewResponse({');
  const accept = refresh.indexOf('if (!verdict.accept)');
  const show = refresh.indexOf('setPreviewSrc(blobSrc)');
  assert.ok(judge < accept && accept < show);
  // A superseded response is cached only under the REQUEST's own key.
  const stale = refresh.slice(accept, refresh.indexOf('return;', accept));
  assert.ok(stale.includes('setCachedPreviewByKey(request.key'));
  assert.ok(!stale.includes('setPreviewSrc('));
  assert.ok(!stale.includes('wantedPreviewRef.current.key'));
});
await check('preview, swap and queue payloads all carry processing_selection', () => {
  const previewBody = faceSwap.slice(faceSwap.indexOf('const buildPreviewPayload = ('), faceSwap.indexOf('const resetTrackerSliders'));
  const swapBody = faceSwap.slice(faceSwap.indexOf('const buildSwapPayload = ('), faceSwap.indexOf('const currentJob = ('));
  const jobBody = faceSwap.slice(faceSwap.indexOf('const currentJob = ('), faceSwap.indexOf('const addToQueue = '));
  for (const [name, body] of [['preview', previewBody], ['swap', swapBody], ['job', jobBody]]) {
    assert.ok(body.includes('processing_selection:'), `${name} payload lacks processing_selection`);
  }
  assert.ok(jobBody.includes('processing_selection: payload.processing_selection'));
});
await check('automatic target queue rebuilds canonical source selection per target', () => {
  const swapBody = faceSwap.slice(faceSwap.indexOf('const buildSwapPayload = ('), faceSwap.indexOf('const currentJob = ('));
  const addBody = faceSwap.slice(faceSwap.indexOf('const applyTargetAdd = async'), faceSwap.indexOf('const onAddTarget = async'));
  assert.ok(swapBody.includes('sourceIndex = selSource'));
  assert.ok(swapBody.includes('processing_selection: buildProcessingSelection(sp, {'));
  assert.ok(addBody.includes('const targetPayload = buildSwapPayload(p, {'));
  assert.ok(addBody.includes('sourceIndex: srcIdx'));
  assert.ok(addBody.includes('targetMediaId'));
  assert.ok(addBody.includes('payload: {\n              ...targetPayload'));
});
await check('the preview signature strips per-request bookkeeping', () => {
  const sig = faceSwap.slice(faceSwap.indexOf('const previewSignature = ('), faceSwap.indexOf('const previewKey ='));
  assert.ok(sig.includes('selectionIdentity('));
  assert.ok(sig.includes('selection_version: null'));
});
await check('person selection and mapping edits go through the versioned commit path', () => {
  const personGroups = readFileSync(join(here, '..', 'src', 'components', 'PersonGroups.jsx'), 'utf8');
  assert.ok(personGroups.includes('commitTargetContext({'));
  assert.ok(personGroups.includes('selected_target_person_id: personId'));
  assert.ok(faceSwap.includes('selection_version: bumpSelectionVersion()'));
  assert.ok(faceSwap.includes('commitTargetContext={commitTargetContext}'));
});
await check('a preview face-box click captures that face (or selects its person), never a bare highlight', () => {
  assert.ok(faceSwap.includes('onSelectFace={onFaceBoxClick}'), 'the stage must be wired to onFaceBoxClick');
  const body = faceSwap.slice(faceSwap.indexOf('const onFaceBoxClick = async'), faceSwap.indexOf('const capturedBoxesRef'));
  assert.ok(body.includes('await captureTargetFaceFromFrame({ faceIndex })'));
  assert.ok(body.includes('capturedBoxesRef.current[boxKey]'), 'a repeated click must select, not capture twice');
  assert.ok(body.includes('captureBusyRef.current'), 'clicks during an in-flight capture are ignored');
});
await check('capturing multiple people switches the preview to multi-person mode', () => {
  const body = faceSwap.slice(
    faceSwap.indexOf('const captureTargetFaceFromFrame = async'),
    faceSwap.indexOf('const useFaceFromFrame = async'),
  );
  assert.ok(body.includes('new Set('), 'capture must count stable target people');
  assert.ok(body.includes("capturedPeople.size > 1 ? 'Selected people' : 'Selected face'"),
    'multi-person capture must not remain in single-face mode');
  assert.ok(body.includes('previewing all captured people'),
    'the UI should confirm that the preview includes every captured person');
});
await check('session restore goes through reconcileRestoredSelection and restores the source by id', () => {
  assert.ok(faceSwap.includes('reconcileRestoredSelection({'));
  assert.ok(faceSwap.includes('st.selected_source_id'));
});
await check('target context preserves selSource across target selection switches', () => {
  const rememberBody = faceSwap.slice(faceSwap.indexOf('const rememberTargetContext ='), faceSwap.indexOf('const applyTargetContext ='));
  const applyBody = faceSwap.slice(faceSwap.indexOf('const applyTargetContext ='), faceSwap.indexOf('// Keep the ref current for changes'));
  assert.ok(rememberBody.includes('selSource,'), 'rememberTargetContext must include selSource');
  assert.ok(applyBody.includes('saved.selSource'), 'applyTargetContext must restore saved.selSource');
  assert.ok(applyBody.includes('setSelSource(restoredSelSource)'), 'applyTargetContext must call setSelSource');
  assert.ok(applyBody.includes('defaultSourceIndexForTarget'), 'new targets need a deterministic source default');
  assert.ok(applyBody.includes('target_selected_source_id'), 'backend target source selection must be restored');
  assert.ok(faceSwap.includes('selected_source_id: sourceIdAt(selSource)'), 'preview must carry the selected source identity');
});

console.log(`\n${fails.length ? `FAILED: ${fails.length} (${fails.join(', ')})` : `ALL GREEN: ${pass}/${pass} checks passed`}`);
process.exit(fails.length ? 1 : 0);
