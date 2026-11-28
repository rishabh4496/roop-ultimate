// The studio's state and every action that changes it - one store, so the viewport, timeline,
// cast, targets and people panels all read the same truth.
//
// Every action is a real engine call with the request shapes React UI 1.0 uses (its Face Swap
// screen is the reference for each body below). Responses that carry the target context are
// folded in by applyTargetContext - the one place those ~12 fields are read - exactly as 1.0's
// applyTargetFacesPayload does, so a capture, a select and a refresh can never disagree.
import { create } from 'zustand';
import { getJSON, postJSON, postFiles } from '@engine/api';
import { nextSelectionVersion } from '@engine/components/faceswap/processingSelection.js';
import { buildPreviewPayload, clipKeyOf, inOutOf, engineFrameFor } from './payload';
import { normalizeBlend, EMPTY_BLEND } from '@engine/components/faceswap/identityBlend.js';

// Per-clip tool state (roto zones, pins, scene overrides, curves), kept per browser and keyed by the
// target's media id, so a zone drawn round a microphone in one clip never rides into another's render.
const CLIP_FX_KEY = 'roop_v2_clipfx';
const loadClipFx = () => { try { return JSON.parse(localStorage.getItem(CLIP_FX_KEY) || '{}') || {}; } catch { return {}; } };
const saveClipFx = (v) => { try { localStorage.setItem(CLIP_FX_KEY, JSON.stringify(v)); } catch { /* quota / private: in-memory only */ } };
import { notify } from '../ui/controls';
import { termLog } from '../fx/termFeed';

const flatGroups = (groups) => (groups || []).map((g) => (Array.isArray(g) ? (g[0] ?? 0)
  : typeof g === 'number' ? g : parseInt(g, 10) || 0));

function contextPatch(res) {
  const p = {};
  if (!res || typeof res !== 'object') return p;
  if (res.target_media_id !== undefined || res.active_target_media_id !== undefined) {
    p.targetMediaId = res.target_media_id || res.active_target_media_id || null;
  }
  if (res.target_faces !== undefined) p.targetFaces = res.target_faces || [];
  if (res.target_groups !== undefined) p.targetGroups = flatGroups(res.target_groups);
  if (res.target_names !== undefined) p.targetNames = res.target_names || [];
  if (res.target_faces_info !== undefined) p.targetFacesInfo = res.target_faces_info || [];
  if (res.target_person_ids !== undefined) p.targetPersonIds = res.target_person_ids || [];
  if (res.target_reference_face_ids !== undefined) p.targetReferenceFaceIds = res.target_reference_face_ids || [];
  if (res.target_person_source_mapping !== undefined && res.target_person_source_mapping !== null) {
    p.faceMapping = res.target_person_source_mapping || {};
  } else if (res.face_mapping && !Array.isArray(res.face_mapping)) {
    p.faceMapping = res.face_mapping;
  }
  if (res.selected_target_person_id !== undefined) p.selectedTargetPersonId = res.selected_target_person_id || null;
  if (res.selected_reference_face_id !== undefined) p.selectedReferenceFaceId = res.selected_reference_face_id || null;
  if (Number.isInteger(res.selected_target_face_index)) p.selTargetFace = res.selected_target_face_index;
  if (Array.isArray(res.scene_cuts)) p.sceneCuts = res.scene_cuts;
  return p;
}

const EMPTY_PREVIEW = { image: null, faces: [], personIds: [], frame: null, loading: false, error: null, audit: null, diagnostic: null, message: null, size: null };

export const useStudio = create((set, get) => ({
  booted: false,
  meta: null,
  settings: null,
  sources: [], sourcesInfo: [], selectedSource: 0,
  targets: [], selectedTarget: 0, targetMediaId: null,
  targetFaces: [], targetGroups: [], targetNames: [], targetFacesInfo: [], targetPersonIds: [],
  targetReferenceFaceIds: [], faceMapping: {}, selTargetFace: 0, selectedTargetPersonId: null,
  selectedReferenceFaceId: null, sceneCuts: [],
  selectionVersion: nextSelectionVersion(0),
  frame: 1,
  preview: EMPTY_PREVIEW,
  previewMode: 'swap',          // swap | original | compare
  busy: '',                     // the action in flight, for spinners
  tool: 'pick',                 // viewport tool: pick | mask | roto | pin
  brush: { size: 28, erase: false },
  mask: { dataUrl: null, refKps: null, version: 0 },
  clipFx: loadClipFx(),
  identityBlend: EMPTY_BLEND,
  blendDiag: null,
  reference: null,              // { url, name, kind } - an output to A/B against

  // ── boot / refresh ─────────────────────────────────────────────────────────
  boot: async () => {
    const [meta, settings] = await Promise.all([getJSON('/api/meta'), getJSON('/api/settings')]);
    set({ meta, settings });
    // The engine's live identity blend is the state to restore (a reload must not drop it).
    getJSON('/api/identity/blend').then((r) => { if (r?.recipe) set({ identityBlend: normalizeBlend(r.recipe) }); }).catch(() => {});
    await get().refresh();
    set({ booted: true });
    return { meta, settings };
  },
  refresh: async () => {
    const st = await getJSON('/api/state');
    get().applyState(st);
    return st;
  },
  applyState: (st) => {
    if (!st) return;
    const targets = st.targets || [];
    const sel = Math.min(Math.max(0, st.selected_target_index ?? 0), Math.max(0, targets.length - 1));
    set({
      sources: st.source_faces || [],
      sourcesInfo: st.source_faces_info || [],
      selectedSource: st.selected_source_index ?? 0,
      targets,
      selectedTarget: sel,
      ...contextPatch(st),
      ...(targets[sel] && !st.target_media_id ? { targetMediaId: targets[sel].media_id || targets[sel].id || null } : {}),
    });
    if (st.desync) notify(`Source gallery out of step with the engine: ${st.desync.message || 'reload the cast'}`, 'warn');
  },
  applyTargetContext: (res) => set(contextPatch(res)),

  // ── settings ───────────────────────────────────────────────────────────────
  // Applied locally at once; saved as a debounced partial patch (POST /api/settings accepts one).
  setSetting: (key, value) => get().setSettings({ [key]: value }),
  setSettings: (patch) => {
    set((s) => ({ settings: { ...(s.settings || {}), ...patch } }));
    pendingPatch = { ...pendingPatch, ...patch };
    clearTimeout(saveTimer);
    saveTimer = setTimeout(flushSettings, 450);
  },

  // ── sources (the cast) ─────────────────────────────────────────────────────
  addSources: async (files) => {
    if (!files?.length) return;
    set({ busy: 'sources' });
    try {
      await postFiles('/api/source/add', files, undefined, { timeout: 0 });
      await get().refresh();
      notify(`Added ${files.length} source image${files.length > 1 ? 's' : ''} to the cast`);
    } catch (e) { notify(e.message, 'error'); } finally { set({ busy: '' }); }
  },
  removeSource: async (index) => {
    try { await postJSON('/api/source/remove', { index }); await get().refresh(); } catch (e) { notify(e.message, 'error'); }
  },
  selectSource: async (index) => {
    set({ selectedSource: index });
    try { await postJSON('/api/source/select', { index }); } catch (e) { notify(e.message, 'error'); }
  },
  clearSources: async () => {
    try { await postJSON('/api/source/clear', {}); await get().refresh(); } catch (e) { notify(e.message, 'error'); }
  },

  // ── targets (media bin) ────────────────────────────────────────────────────
  addTargets: async (files) => {
    if (!files?.length) return;
    set({ busy: 'targets' });
    try {
      const res = await postFiles('/api/target/add', files, undefined, { timeout: 0 });
      await get().refresh();
      if (res?.errors?.length) notify(`${res.errors.length} file(s) were refused: ${res.errors.map((e) => e.file).join(', ')}`, 'warn');
      else notify(`Added ${files.length} target${files.length > 1 ? 's' : ''}`);
      set({ frame: 1, preview: EMPTY_PREVIEW });
    } catch (e) { notify(e.message, 'error'); } finally { set({ busy: '' }); }
  },
  addTargetPaths: async (paths) => {
    const list = (paths || []).map((p) => String(p).trim().replace(/^"|"$/g, '')).filter(Boolean);
    if (!list.length) return;
    set({ busy: 'targets' });
    try {
      await postJSON('/api/target/add_path', { paths: list }, { timeout: 0 });
      await get().refresh();
      notify(`Added ${list.length} target${list.length > 1 ? 's' : ''} by path`);
    } catch (e) { notify(e.message, 'error'); } finally { set({ busy: '' }); }
  },
  selectTarget: async (index) => {
    const t = get().targets[index];
    if (!t) return;
    try {
      const res = await postJSON('/api/target/select', {
        index, target_media_id: t.media_id || t.id, selected_target_face_index: get().selTargetFace,
      });
      const targets = res.targets || get().targets;
      const sel = res.selected_target_index ?? index;
      set({ targets, selectedTarget: sel, frame: inOutOf(targets[sel]).a, preview: EMPTY_PREVIEW, ...contextPatch(res) });
      if (!res.target_media_id) set({ targetMediaId: targets[sel]?.media_id || targets[sel]?.id || null });
    } catch (e) { notify(e.message, 'error'); }
  },
  removeTarget: async (index) => {
    const t = get().targets[index];
    try {
      await postJSON('/api/target/remove', { index, target_media_id: t?.media_id || t?.id });
      await get().refresh();
      set({ preview: EMPTY_PREVIEW });
    } catch (e) { notify(e.message, 'error'); }
  },
  /** `frame` is the UI's frame (1-based, inclusive), which is what set_frame takes. */
  setRange: async (which, frame) => {
    try {
      const res = await postJSON('/api/target/set_frame', { which, frame: engineFrameFor(which, frame), target_media_id: get().targetMediaId });
      if (res.targets) set({ targets: res.targets });
      termLog(`${which === 'start' ? 'in' : 'out'} = frame ${Math.round(frame)}`, 'info', 'range');
    } catch (e) { notify(e.message, 'error'); }
  },

  // ── target people ──────────────────────────────────────────────────────────
  /** Make face `faceIndex` on the current frame a target person (or every face: all=true). */
  captureFace: async ({ faceIndex = null, all = false } = {}) => {
    const s = get();
    try {
      const res = await postJSON('/api/target/use_face', {
        index: s.selectedTarget, frame: s.frame, target_media_id: s.targetMediaId,
        ...(all ? { capture_all: true } : { face_index: faceIndex }),
      }, { timeout: 0 });
      set({ ...contextPatch(res), selTargetFace: Math.max(0, (res.target_faces || []).length - 1) });
      const people = new Set((res.target_person_ids || []).filter(Boolean));
      // As 1.0 does: one captured person previews that person; several preview all of them.
      get().setSetting('face_detection_mode', people.size > 1 ? 'Selected people' : 'Selected face');
      notify(all ? `Added ${res.count ?? people.size} target people` : `Face ${faceIndex + 1} is now a target person`);
      return res.selected_target_person_id || null;
    } catch (e) { notify(e.message, 'error'); return null; }
  },
  selectPerson: (personId, faceIndex) => {
    // A legacy context (person ranks, no ids) selects by face index; a current one by stable id.
    const legacy = typeof personId === 'number';
    set({ ...(legacy ? {} : { selectedTargetPersonId: personId }), ...(Number.isInteger(faceIndex) ? { selTargetFace: faceIndex } : {}) });
    get().commitContext(legacy ? { selected_target_face_index: faceIndex } : { selected_target_person_id: personId });
  },
  /** Who person `personId` becomes: a source index, or -1 to leave them untouched. */
  mapPerson: (personId, sourceIndex) => {
    // undefined = "automatic": the key must be REMOVED, not set to undefined - 1.0's mapPerson
    // treats any present key as an explicit choice, and an undefined one normalises to Skip.
    const next = { ...get().faceMapping };
    if (sourceIndex === undefined || sourceIndex === null) delete next[personId];
    else next[personId] = sourceIndex;
    set({ faceMapping: next });
    get().commitContext({ face_mapping: next, target_person_source_mapping: next });
  },
  /** Give person `p` (from peopleOf) a display name; '' clears it. The engine keys it by stable id. */
  renamePerson: async (p, name) => {
    const s = get();
    try {
      const res = await postJSON('/api/target/name', {
        target_media_id: s.targetMediaId, name: String(name || '').trim().slice(0, 40),
        ...(p.legacy ? { person: p.rank } : { target_person_id: p.id }),
      });
      set(contextPatch(res));
      return true;
    } catch (e) { notify(e.message, 'error'); return false; }
  },
  /** Forget person `p`: every captured angle of them, highest face index first so the rest keep theirs. */
  removePerson: async (p) => {
    const indices = [...(p.faceIndices || [p.faceIndex])].sort((a, b) => b - a);
    try {
      let res = null;
      for (const face_index of indices) {
        res = await postJSON('/api/target/remove_face', { target_media_id: get().targetMediaId, face_index, index: face_index });
      }
      // The engine's own mapping wins; the mock sends none, so drop the person from ours.
      const patch = contextPatch(res);
      const next = { ...get().faceMapping };
      delete next[p.id];
      set({ ...patch, faceMapping: patch.faceMapping || next });
      notify(`Removed ${p.name}`);
      return true;
    } catch (e) { notify(e.message, 'error'); await get().refresh().catch(() => {}); return false; }
  },
  // ── people tools (1.0's PersonGroups, on the same routes) ───────────────────
  /** Scan the whole clip and capture everyone from THEIR most frontal frame, replacing the people. */
  autoCapture: async () => {
    const s = get();
    set({ busy: 'people' });
    try {
      const res = await postJSON('/api/target/auto_capture', { index: s.selectedTarget, replace: true, target_media_id: s.targetMediaId }, { timeout: 0 });
      set({ ...contextPatch(res), selTargetFace: 0 });
      if (!res.count) { notify(res.message || 'Auto-capture found nobody in this clip', 'warn'); return res; }
      const people = new Set((res.target_person_ids || []).filter(Boolean));
      if (people.size > 1) get().setSetting('face_detection_mode', 'Selected people');
      // `separation` predicts whether identity can work at all: two people 0.12 apart are one
      // identity to the matcher, and the failure shows up much later as wrong or missing swaps.
      const sep = res.separation;
      const quality = sep == null ? '' : sep >= 0.7 ? `, clearly distinct (${sep.toFixed(2)})` : sep >= 0.4 ? `, usable but close (${sep.toFixed(2)})` : `. WARNING: only ${sep.toFixed(2)} apart, they will be mixed up`;
      const angles = res.angles_added ? `, plus ${res.angles_added} angle${res.angles_added === 1 ? '' : 's'}` : '';
      notify(`Captured ${res.count} ${res.count === 1 ? 'person' : 'people'}${angles}${res.scanned ? ` from ${res.scanned} frames` : ''}${quality}`, sep != null && sep < 0.4 ? 'warn' : 'success');
      (res.notes || []).forEach((n) => { if (/WARNING|never clearly apart|overlap/.test(n)) notify(n, 'warn'); });
      return res;
    } catch (e) { notify(e.message, 'error'); return null; } finally { set({ busy: '' }); }
  },
  /** Re-group every captured face into people by embedding. */
  autoGroup: async () => {
    set({ busy: 'people' });
    try {
      const res = await postJSON('/api/target/autocluster', { target_media_id: get().targetMediaId }, { timeout: 0 });
      set(contextPatch(res));
      const n = res.people ?? new Set((res.target_person_ids?.some(Boolean) ? res.target_person_ids : res.target_groups || []).filter((x) => x !== null && x !== '')).size;
      notify(`Grouped into ${n} ${n === 1 ? 'person' : 'people'}`);
      return res;
    } catch (e) { notify(e.message, 'error'); return null; } finally { set({ busy: '' }); }
  },
  /** Capture person `p` again from the frame on screen: one more angle in their bank. */
  addAngle: async (p) => {
    const s = get();
    try {
      const res = await postJSON('/api/target/add_angle', {
        index: s.selectedTarget, frame: s.frame, target_media_id: s.targetMediaId,
        ...(p.legacy ? { person: p.rank } : { target_person_id: p.id }),
      }, { timeout: 0 });
      set(contextPatch(res));
      if (res?.message && !res.count && !(res.target_faces || []).length) notify(res.message, 'warn');
      else notify(`Added an angle of ${p.name} from frame ${s.frame}`);
      return res;
    } catch (e) { notify(e.message, 'error'); return null; }
  },
  /** Scan the clip for person `p` at many poses and bank the new angles. */
  findAngles: async (p) => {
    const s = get();
    set({ busy: `angles:${p.id}` });
    try {
      const res = await postJSON('/api/target/auto_angles', {
        index: s.selectedTarget, target_media_id: s.targetMediaId,
        ...(p.legacy ? { person: p.rank } : { target_person_id: p.id }),
      }, { timeout: 0 });
      set(contextPatch(res));
      const detail = res.scanned ? ` (scanned ${res.scanned} frames${res.bins ? `, ${res.bins} pose bins` : ''})` : '';
      if (res.count) {
        notify(`Found ${res.count} new angle${res.count === 1 ? '' : 's'} of ${p.name}${detail}`);
        // Every match takes the minimum over a person's angles, so one wrong angle speaks for all.
        if (res.review?.length) notify(`${res.review.length} of them sit far from the original capture: check ${p.name}'s angles and remove any that are not them.`, 'warn');
      } else {
        const why = Object.entries(res.rejected || {}).map(([r, c]) => `${c} ${r}`).join(', ');
        notify(`${res.message || 'No new angles found'}${detail}${why ? `; turned away: ${why}` : ''}`, 'warn');
      }
      return res;
    } catch (e) { notify(e.message, 'error'); return null; } finally { set({ busy: '' }); }
  },
  /** Merge person `from` into person `into`: every captured angle of `from` becomes `into`'s. */
  mergePerson: async (from, into) => {
    const s = get();
    if (!from || !into || from.id === into.id) return null;
    const legacy = from.legacy || into.legacy;
    const body = legacy
      ? { groups: (s.targetGroups || []).map((g) => (g === from.id ? into.id : g)) }
      : { target_person_ids: (s.targetPersonIds || []).map((id) => (id === from.id ? into.id : id)) };
    try {
      const res = await postJSON('/api/target/group', { target_media_id: s.targetMediaId, ...body });
      const patch = contextPatch(res);
      // The engine sends the regrouped context; the mock's legacy answer is the groups we sent.
      set({ ...patch, ...(legacy && !patch.targetGroups ? { targetGroups: body.groups } : {}) });
      const next = { ...get().faceMapping };
      if (Object.prototype.hasOwnProperty.call(next, from.id) && !patch.faceMapping) { delete next[from.id]; set({ faceMapping: next }); }
      notify(`Merged ${from.name} into ${into.name}`);
      return res;
    } catch (e) { notify(e.message, 'error'); return null; }
  },
  clearPeople: async () => {
    try {
      const res = await postJSON('/api/target/clear_faces', { target_media_id: get().targetMediaId });
      set({ ...contextPatch(res), faceMapping: {}, selTargetFace: 0, selectedTargetPersonId: null });
    } catch (e) { notify(e.message, 'error'); }
  },
  // The single write path for target-scoped identity state, with a fresh selection_version so the
  // engine drops any older write (a preview built before this click) that lands after it.
  commitContext: async (patch) => {
    const mediaId = get().targetMediaId;
    if (!mediaId) return null;
    const version = nextSelectionVersion(get().selectionVersion);
    set({ selectionVersion: version });
    try {
      return await postJSON('/api/target/context', { target_media_id: mediaId, selection_version: version, ...patch });
    } catch (e) { notify(e.message, 'error'); return null; }
  },

  // ── tools ────────────────────────────────────────────────────────────────
  setTool: (tool) => set({ tool }),
  setBrush: (patch) => set((x) => ({ brush: { ...x.brush, ...patch } })),
  /** Commit the painted mask (a white-on-transparent PNG at frame size; the engine reads it as grayscale). */
  applyMask: (dataUrl) => {
    const x = get();
    const kps = x.preview.kps?.[x.selTargetFace] || x.preview.kps?.[0] || null;
    set({ mask: { dataUrl: dataUrl || null, refKps: dataUrl ? kps : null, version: x.mask.version + 1 } });
  },
  /** Patch the selected clip's tool state: fn(fx) -> partial fx. */
  updateClipFx: (fn) => {
    const x = get();
    const key = clipKeyOf(x);
    const cur = { roto: [], pins: [], overrides: [], curves: {}, ...(x.clipFx[key] || {}) };
    const next = { ...x.clipFx, [key]: { ...cur, ...fn(cur) } };
    set({ clipFx: next });
    saveClipFx(next);
  },
  setIdentityBlend: (recipe) => {
    const next = normalizeBlend(recipe);
    set({ identityBlend: next });
    clearTimeout(blendTimer);
    blendTimer = setTimeout(async () => {
      try { set({ blendDiag: await postJSON('/api/identity/blend', next) }); }
      catch (e) { set({ blendDiag: { error: e.message } }); }
    }, 300);
  },
  setReference: (reference) => set({ reference }),

  // ── frame + preview ────────────────────────────────────────────────────────
  setFrame: (frame) => {
    const t = get().targets[get().selectedTarget];
    const max = Math.max(1, t?.frames || 1);
    set({ frame: Math.min(max, Math.max(1, Math.round(frame))) });
  },
  setPreviewMode: (previewMode) => set({ previewMode }),
  requestPreview: async () => {
    const s = get();
    if (!s.targets.length || !s.settings) return;
    previewCtrl?.abort();
    const ctrl = new AbortController();
    previewCtrl = ctrl;
    const frame = s.frame;
    const swap = s.previewMode !== 'original' && s.sources.length > 0;
    set((x) => ({ preview: { ...x.preview, loading: true, error: null } }));
    try {
      const res = await postJSON('/api/preview', buildPreviewPayload(s, { frame, swap }), { timeout: 0, signal: ctrl.signal });
      if (previewCtrl !== ctrl) return;
      set({
        preview: {
          image: res?.image || null, faces: res?.faces || [], personIds: res?.person_ids || [], kps: res?.kps || [], frame,
          loading: false, error: null, audit: res?.swap_audit || null,
          diagnostic: res?.selection_diagnostic || (res?.target_required ? 'target_required' : null),
          message: res?.message || null, swapped: swap,
        },
      });
    } catch (e) {
      if (e?.name === 'AbortError' || previewCtrl !== ctrl) return;
      // 409 render_active: the engine refuses a preview beside a running render (it would share the GPU).
      if (e?.status === 409 && e?.body?.render_active) {
        set((x) => ({ preview: { ...x.preview, loading: false, error: null, message: 'Render in progress - the swap preview resumes when it ends' } }));
        return;
      }
      set((x) => ({ preview: { ...x.preview, loading: false, error: e.message } }));
    }
  },
}));

let previewCtrl = null;
let blendTimer = null;
let saveTimer = null;
let pendingPatch = {};
async function flushSettings() {
  const patch = pendingPatch;
  pendingPatch = {};
  if (!Object.keys(patch).length) return;
  try {
    await postJSON('/api/settings', patch);
    termLog(`saved ${Object.keys(patch).join(', ')}`, 'info', 'settings');
  } catch (e) { notify(`Saving settings failed: ${e.message}`, 'error'); }
}

/** Selectors */
export const currentTarget = (s) => s.targets[s.selectedTarget] || null;
export const isVideoTarget = (t) => !!t && (t.frames || 0) > 1;
