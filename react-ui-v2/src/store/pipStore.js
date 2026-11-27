import { create } from 'zustand';
import { getJSON, postJSON } from '../api';
import { decodeFrame, releaseFrame } from '../components/faceswap/frameDecoder';
import { newRequestId, nextSelectionVersion } from '../components/faceswap/processingSelection';
import { onMutation } from '../mutationBus';
import { changesLiveSource, nextCorner, replayBody, sanitizeSaved } from '../pip/pipModel';

// The floating mini-viewport's state: whether it is on, where it sits, the frame it shows, and how that frame is renewed.
//
// THE FRAME. Face Swap's preview stage decodes each picture once, off-thread, into an ImageBitmap that PreviewCanvas paints.
// When the stage receives its "after" picture it offers it here (`offerFrame`); this store keeps ITS OWN copy
// (createImageBitmap of a bitmap is a pixel copy, not a request and not a second decode). A copy, not a share, because the
// owner closes its bitmap the moment it is replaced or the panel unmounts - and the panel unmounting is exactly the moment
// the PiP exists for. No network is involved at any point: the picture the PiP shows is the one the stage already has.
//
// Nothing is copied while the PiP is switched off, so the feature costs nothing until it is turned on.
//
// THE REFRESH. While Face Swap is mounted it renders previews and the PiP just follows them. When it is NOT (the user is in
// Face Manager), a change that re-selects the live source (Morph Lab's Preview, a faceset load - see pipModel.changesLiveSource)
// makes the picture stale, so the PiP replays the last /api/preview request itself - same frame, settings and people, the
// source the backend now has selected - after a short debounce. At most one such request is ever in flight, and a result
// older than a frame the stage has offered since is dropped rather than shown over it.

const KEY = 'roop_pip_v1';
const REFRESH_DEBOUNCE_MS = 700;

const readSaved = () => {
  try {
    return sanitizeSaved(JSON.parse(localStorage.getItem(KEY) || 'null'));
  } catch {
    return sanitizeSaved(null);
  }
};
const writeSaved = (s) => {
  try {
    localStorage.setItem(KEY, JSON.stringify({ enabled: s.enabled, w: s.w, corner: s.corner, x: s.x, y: s.y }));
  } catch { /* the layout lasts for this session */ }
};

let seq = 0;                 // increases with every frame offered / refreshed in
let refreshTimer = null;
const closeLater = (bmp) => { if (bmp) setTimeout(() => releaseFrame(bmp), 400); };   // after the paint that stopped using it

export const usePipStore = create((set, get) => {
  const saved = readSaved();
  const persist = () => writeSaved(get());

  return {
    enabled: saved.enabled,
    w: saved.w,
    corner: saved.corner,          // 'bottom-right' etc., or null while free-floating
    x: saved.x,
    y: saved.y,

    frame: null,                   // { bitmap, w, h, seq, at } - owned by this store
    context: { frame: null, total: null },
    tab: 'home',
    stageVisible: true,
    faceswapMounted: false,
    lastRequest: null,             // the body of the last /api/preview Face Swap sent
    refreshing: false,
    refreshError: '',

    setEnabled: (enabled) => {
      set({ enabled: !!enabled });
      if (!enabled) get().clearFrame();
      persist();
    },
    toggle: () => get().setEnabled(!get().enabled),

    /** Size / place the window. Pass `corner: null` for a free position. */
    setGeometry: (patch) => { set(patch); persist(); },
    dock: (corner) => { set({ corner, x: null, y: null }); persist(); },
    cycleCorner: () => get().dock(nextCorner(get().corner)),

    setTab: (tab) => { if (tab !== get().tab) set({ tab }); },
    setStageVisible: (stageVisible) => { if (stageVisible !== get().stageVisible) set({ stageVisible }); },
    setFaceswapMounted: (faceswapMounted) => set({ faceswapMounted }),
    setContext: (context) => set({ context: { ...get().context, ...context } }),
    rememberRequest: (body) => set({ lastRequest: body }),

    clearFrame: () => {
      const old = get().frame;
      set({ frame: null });
      closeLater(old?.bitmap);
    },

    /**
     * The stage received a new "after" picture. `source` is an ImageBitmap / canvas / image; it stays the caller's.
     * createImageBitmap snapshots the pixels when it is CALLED, so it is invoked before anything is awaited - the owner may
     * close its bitmap on the very next paint.
     */
    offerFrame: (source) => {
      if (!get().enabled || !source || typeof createImageBitmap !== 'function') return;
      const mine = ++seq;
      let pending;
      try { pending = createImageBitmap(source); } catch { return; }
      pending.then((bitmap) => {
        if (mine < seq || !get().enabled) { releaseFrame(bitmap); return; }   // a newer one was offered meanwhile
        const old = get().frame;
        set({ frame: { bitmap, w: bitmap.width, h: bitmap.height, seq: mine, at: Date.now() }, refreshError: '' });
        closeLater(old?.bitmap);
      }).catch(() => { /* the source was closed first: the next offer replaces it */ });
    },

    /** Re-render the picture because the live source changed. No-op while Face Swap is mounted (it owns previews). */
    refresh: async () => {
      const s = get();
      if (!s.enabled || !s.lastRequest || s.refreshing || s.faceswapMounted) return false;
      const started = ++seq;                                  // claims a sequence number: anything offered after this wins
      set({ refreshing: true, refreshError: '' });
      try {
        let state = null;
        try { state = await getJSON('/api/state'); } catch { /* without it, the last source stands */ }
        const prevVersion = s.lastRequest?.processing_selection?.selection_version;
        const body = replayBody(s.lastRequest, {
          state, requestId: newRequestId(), version: nextSelectionVersion(prevVersion),
        });
        const res = await postJSON('/api/preview', body, { timeout: 0 });
        if (res?.error) throw new Error(res.message || res.error);
        if (!res?.image) throw new Error('the preview came back without a picture');
        const bitmap = await decodeFrame(res.image);
        if (seq !== started) { releaseFrame(bitmap); return false; }           // something newer arrived while this rendered
        const old = get().frame;
        set({ frame: { bitmap, w: bitmap.width, h: bitmap.height, seq: started, at: Date.now() } });
        closeLater(old?.bitmap);
        return true;
      } catch (err) {
        set({ refreshError: String(err?.message || err) });
        return false;
      } finally {
        set({ refreshing: false });
      }
    },
  };
});

// A source change in Face Manager (or a faceset load) makes the frame stale: refresh it once the burst settles.
let unsubscribe = null;
export function installPipRefresh() {
  if (unsubscribe) return unsubscribe;
  const off = onMutation(({ method, path }) => {
    if (!changesLiveSource(method, path)) return;
    const s = usePipStore.getState();
    if (!s.enabled || s.faceswapMounted) return;
    if (refreshTimer) clearTimeout(refreshTimer);
    refreshTimer = setTimeout(() => { refreshTimer = null; usePipStore.getState().refresh(); }, REFRESH_DEBOUNCE_MS);
  });
  unsubscribe = () => { off(); unsubscribe = null; if (refreshTimer) { clearTimeout(refreshTimer); refreshTimer = null; } };
  return unsubscribe;
}
