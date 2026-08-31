import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { getJSON, postJSON, postFile, postFiles, API } from '../api';
import { Section, Select, Slider, Toggle, TextInput, Button, FaceGallery, Card, Skeleton } from './ui';
import { Icon } from '../icons';
import PersonGroups from './PersonGroups';
import QualityReport from './QualityReport';
import FileDrop from './faceswap/FileDrop';
import CompareGrid from './faceswap/CompareGrid';
import ParserRegions from './faceswap/ParserRegions';
import InteractivePreview from './faceswap/InteractivePreview';
import useQueue from './faceswap/useQueue';
import QueuePanel from './faceswap/QueuePanel';
import useSegments from './faceswap/useSegments';
import SegmentBar from './faceswap/SegmentBar';
import SliderTrackerBar from './faceswap/SliderTrackerBar';
import Timeline from './faceswap/Timeline';
import FacesetLibrary from './faceswap/FacesetLibrary';
import AmbilightGlow from './faceswap/AmbilightGlow';
import FloatingActionDock from './faceswap/FloatingActionDock';
import MediaTabSessionBar from './faceswap/MediaTabSessionBar';
import PresetStudioModal from './faceswap/PresetStudioModal';
import { popoutManager } from './faceswap/PopoutPreviewManager';
import { setLastPreview } from './faceswap/lastPreview';
import { num, fmtTime } from './faceswap/utils';
import useProfiles from './faceswap/useProfiles';
import useTelemetry from './faceswap/useTelemetry';
import useSequentialImage from './faceswap/useSequentialImage';
import useCompareGrid from './faceswap/useCompareGrid';
import useRenderLite from './faceswap/useRenderLite';
import useLiveCam from './faceswap/useLiveCam';
import useClipAdvisor, { ADVISOR_LABELS, fmtAdviceVal } from './faceswap/useClipAdvisor';
import useRuntimeEstimate from './faceswap/useRuntimeEstimate';
import useUserDefaults from './faceswap/useUserDefaults';
import useViewPersistence from './faceswap/useViewPersistence';
import usePlaybackBuffer from './faceswap/usePlaybackBuffer';
import useGridPreviewLoader from './faceswap/useGridPreviewLoader';
import useWorkspaceLayout from './faceswap/useWorkspaceLayout';
import { TRACKER_DEFAULT_VALUES, TRACKER_BYPASS_VALUES } from './faceswap/trackerConfig';
import { TiltCard } from '../motion';

// AI upscale models folded into the swap pass (mirrors the Extras post-processor
// list). value = backend subtype, label = friendly name.
const AI_UPSCALE_MODELS = [
  { value: 'esrganx2', label: 'Real-ESRGAN ×2' },
  { value: 'esrganx4', label: 'Real-ESRGAN ×4' },
  { value: 'esrgan_anime_x4', label: 'Real-ESRGAN Anime ×4' },
  { value: 'ultrasharp_x4', label: 'Ultra-Sharp ×4' },
  { value: 'lsdirx4', label: 'LSDIR ×4' },
  { value: 'clear_reality_x4', label: 'Clear Reality ×4' },
  { value: 'span_x4', label: 'SPAN ×4' },
  { value: 'compact_x4', label: 'Compact ×4 (fast AI)' },
  { value: 'nomos8k_x4', label: 'Nomos8k ×4' },
  { value: 'lanczos_x2', label: 'Fast Lanczos ×2 (no AI)' },
  { value: 'lanczos_x4', label: 'Fast Lanczos ×4 (no AI)' },
  { value: 'fsr_x2', label: 'FSR-lite ×2 · Lanczos+CAS (no AI)' },
  { value: 'fsr_x4', label: 'FSR-lite ×4 · Lanczos+CAS (no AI)' },
  { value: 'spline_x2', label: 'Spline36 ×2 (no AI)' },
  { value: 'spline_x4', label: 'Spline36 ×4 (no AI)' },
  { value: 'sinc_x2', label: 'Sinc ×2 · sharpest (no AI)' },
  { value: 'sinc_x4', label: 'Sinc ×4 · sharpest (no AI)' },
];

export default function FaceSwap({
  meta, settings, setSettings, notify, registerFileListener,
  progress, setProgress, startTime, setStartTime, onOpenProcessing
}) {
  const [sourceFaces, setSourceFaces] = useState([]);
  const [sourceFacesInfo, setSourceFacesInfo] = useState([]);
  const [targetFaces, setTargetFaces] = useState([]);
  const [targetGroups, setTargetGroups] = useState([]);
  const [targetNames, setTargetNames] = useState([]);
  const [targetFacesInfo, setTargetFacesInfo] = useState([]);
  const [targets, setTargets] = useState([]);
  const [selSource, setSelSource] = useState(0);
  const [selTarget, setSelTarget] = useState(0);
  const [selTargetFace, setSelTargetFace] = useState(0);
  const [frame, setFrame] = useState(1);
  const [maxFrames, setMaxFrames] = useState(1);
  const [previewSrc, setPreviewSrc] = useState('');
  // Which view `previewSrc` was actually rendered for, as `${index}_${frame}`.
  // A render lags the playhead by however long the swap takes, so without this
  // the stage had no way to tell "the swap for the frame you are looking at"
  // from "the swap for the frame you just left" and happily showed the latter.
  const [previewFor, setPreviewFor] = useState('');
  const [previewFaces, setPreviewFaces] = useState([]);
  const [previewPersonIds, setPreviewPersonIds] = useState([]);
  const [previewKps, setPreviewKps] = useState([]);
  // (yaw, pitch, roll) per face, solved server-side by the SAME solve_pose_5pt
  // the non-frontal mask router gates on — see the
  // preview endpoint for why it is not re-derived here.
  const [previewPose, setPreviewPose] = useState([]);
  // Manual mask painted in the preview box (PNG data URL) + the face keypoints
  // it was painted against. maskVersion is a cheap token for the preview cache
  // key so the multi-KB data URL never has to be stringified per render.
  const [manualMask, setManualMask] = useState(null);
  const [maskRefKps, setMaskRefKps] = useState(null);
  const [maskVersion, setMaskVersion] = useState(0);
  // Single-frame AI-upscale spot-check (result shown in a modal).
  const [upscaledSrc, setUpscaledSrc] = useState('');
  const [upscaledDims, setUpscaledDims] = useState(null);
  const [upscaling, setUpscaling] = useState(false);
  const [fakePreview, setFakePreview] = useState(true);
  const [uploadingSrc, setUploadingSrc] = useState(false);
  const [uploadingTgt, setUploadingTgt] = useState(false);
  const [uploadingLipsyncAudio, setUploadingLipsyncAudio] = useState(false);
  // Live {loaded, total, phase} from the XHR uploader, plus the controller that
  // cancels it. A multi-GB target used to post with no progress at all — see
  // api.js for why that could not be fixed without leaving fetch.
  const [srcProgress, setSrcProgress] = useState(null);
  const [tgtProgress, setTgtProgress] = useState(null);
  const srcAbortRef = useRef(null);
  const tgtAbortRef = useRef(null);

  const [previewing, setPreviewing] = useState(false);
  const [previewSecs, setPreviewSecs] = useState(0);
  const [compare, setCompare] = useState(false);
  const [sliderEffectEnabled, setSliderEffectEnabled] = useState(true);
  const [splitView, setSplitView] = useState(false);
  // The four comparison grids (enhancers, masks, swappers, upscalers) all hold
  // the same eight pieces of state, so they share one hook and differ only in
  // their storage key and defaults. Destructured back to the names the rest of
  // this file already uses — see faceswap/useCompareGrid.
  const {
    comparing: comparingEnhancers, setComparing: setComparingEnhancers,
    selected: selectedGridEnhancers, setSelected: setSelectedGridEnhancers,
    previews: enhancerPreviews, setPreviews: setEnhancerPreviews,
    times: enhancerTimes, setTimes: setEnhancerTimes,
    timers: liveRenderingTimers, setTimers: setLiveRenderingTimers,
    intervalsRef: activeIntervalsRef,
  } = useCompareGrid({
    storageKey: 'roop_grid_enhancers',
    defaults: ['None', 'GPEN', 'Restoreformer++', 'GFPGAN'],
  });

  // Which panels the current workspace mode shows, and the drawer toggles that
  // override it. See faceswap/useWorkspaceLayout.
  const {
    workspaceMode, setWorkspaceMode,
    ambilightEnabled, setAmbilightEnabled,
    drawers, setDrawers,
    showLeftPanel, showRightPanel, showTimelineDeck,
  } = useWorkspaceLayout();

  const [showPresetStudio, setShowPresetStudio] = useState(false);
  const [pinnedIdentities, setPinnedIdentities] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('roop_pinned_identities') || '[]');
    } catch { return []; }
  });

  const togglePinCurrentSource = () => {
    if (selSource < 0 || !sourceFaces[selSource]) return;
    const faceData = sourceFaces[selSource];
    setPinnedIdentities((prev) => {
      const exists = prev.some((p) => p.data === faceData);
      const updated = exists
        ? prev.filter((p) => p.data !== faceData)
        : [{ id: Date.now(), data: faceData, label: `Identity ${prev.length + 1}` }, ...prev];
      localStorage.setItem('roop_pinned_identities', JSON.stringify(updated));
      notify?.(exists ? 'Unpinned identity' : 'Pinned identity to Quick Pin Bar!', 'info');
      return updated;
    });
  };

  // The run-complete chime and notification live in App, which is the only
  // component always mounted — a run that ends while you are on another tab
  // still has to announce itself. Mounting the hook here as well is what gave
  // every finished run two chimes.

  // Keeps the UI from competing with the render for the GPU while a job runs —
  // see faceswap/useRenderLite for why that is worth doing. This tab now stays
  // fully drawn during a run, so it matters here more than it used to; the
  // switch for it is on the Processing tab.
  useRenderLite(progress.processing);

  // ── Mask-engine comparison grid (mirrors the enhancer grid) ──
  const {
    comparing: comparingMasks, setComparing: setComparingMasks,
    selected: selectedGridMasks, setSelected: setSelectedGridMasks,
    previews: maskPreviews, setPreviews: setMaskPreviews,
    times: maskTimes, setTimes: setMaskTimes,
    timers: maskRenderTimers, setTimers: setMaskRenderTimers,
    intervalsRef: maskIntervalsRef,
  } = useCompareGrid({
    storageKey: 'roop_grid_masks',
    defaults: ['None', 'DFL XSeg', 'Face Occluder', 'Face Parser (BiSeNet)'],
  });

  // ── Swapper-model comparison grid (mirrors the enhancer/mask grid) ──
  const {
    comparing: comparingSwappers, setComparing: setComparingSwappers,
    selected: selectedGridSwappers, setSelected: setSelectedGridSwappers,
    previews: swapperPreviews, setPreviews: setSwapperPreviews,
    times: swapperTimes, setTimes: setSwapperTimes,
    timers: swapperRenderTimers, setTimers: setSwapperRenderTimers,
    intervalsRef: swapperIntervalsRef,
  } = useCompareGrid({
    storageKey: 'roop_grid_swappers',
    defaults: ['inswapper', 'reswapper', 'hyperswap', 'simswap'],
  });

  // ── AI-upscale comparison grid (mirrors the enhancer/mask/swapper grid) ──
  // Keyed by the friendly MODEL LABEL (e.g. "Real-ESRGAN ×2"), which is also
  // the caption CompareGrid shows; label→subtype is resolved when calling the
  // backend. Each cell swaps the frame ONCE then upscales it with one model.
  //
  // The only grid whose stored selection is validated against a fixed set
  // rather than "any string": these labels index into AI_UPSCALE_MODELS, so a
  // stale one from an older build would render a cell that can never resolve.
  const {
    comparing: comparingUpscalers, setComparing: setComparingUpscalers,
    selected: selectedGridUpscalers, setSelected: setSelectedGridUpscalers,
    previews: upscalePreviews, setPreviews: setUpscalePreviews,
    times: upscaleTimes, setTimes: setUpscaleTimes,
    timers: upscaleRenderTimers, setTimers: setUpscaleRenderTimers,
    intervalsRef: upscaleIntervalsRef,
  } = useCompareGrid({
    storageKey: 'roop_grid_upscalers',
    defaults: AI_UPSCALE_MODELS.slice(0, 2).map((m) => m.label),
    isValid: (x) => AI_UPSCALE_MODELS.some((m) => m.label === x),
  });

  // Telemetry HUD — GPU/VRAM/CPU/RAM/threads poller (see faceswap/useTelemetry).
  const telemetry = useTelemetry();

  // Target-to-Source visual mapping state
  const [faceMapping, setFaceMapping] = useState({});

  const getFaceMappingArray = () => {
    const uniqPersons = Array.from(new Set(targetGroups))
      .filter(x => typeof x === 'number')
      .sort((a, b) => a - b);
    return uniqPersons.map(pId => {
      const mappedSrc = faceMapping[pId];
      return mappedSrc !== undefined ? mappedSrc : pId;
    });
  };

  // Profile Management — named setting presets (see faceswap/useProfiles).
  const {
    profiles, newProfileName, setNewProfileName,
    saveProfile, loadProfile, deleteProfile, exportProfiles, importProfiles,
  } = useProfiles({ settings, setSettings, notify });

  // Keyboard Shortcuts Modal visibility
  const [showShortcutHUD, setShowShortcutHUD] = useState(false);

  // Drag and drop overlay state removed (managed globally by App.jsx)

  // Batch queue. The list and the runner both live on the server now
  // (app/routes_queue.py); this is a view onto them, so a reload, a Pinokio tab
  // switch or a closed window no longer throws the batch away mid-render.
  const queue = useQueue({ notify });

  // Named frame ranges on the current clip, persisted per target. Declared
  // beside the queue because that is what consumes them: each segment becomes
  // one queued job (see queueSegments below).
  const segments = useSegments(targets[selTarget]?.name || String(selTarget), maxFrames);

  // Pasted Files Dialog State
  const [pastedFiles, setPastedFiles] = useState(null);

  // Preview Cache Ref
  const previewCacheRef = useRef({});
  // Set when a preview refresh is skipped because a swap run owns the GPU.
  const previewDeferredRef = useRef(false);

  const clearPreviewCache = () => {
    previewCacheRef.current = {};
  };

  // Content signature of a face gallery. The count alone is NOT enough to
  // identify which faces are loaded: uploading a new faceset and then deleting
  // the old one leaves the count unchanged, so a length-keyed preview cache kept
  // serving — and a length-keyed refresh effect kept skipping — the *previous*
  // face. Hashing the thumbnail data URLs (size + tail, cheap) also catches
  // reorders and in-place replacements.
  const gallerySig = (faces) =>
    `${faces.length}:` + faces.map((f) => (f ? `${f.length}.${f.slice(-24)}` : '-')).join('|');
  const sourceSig = useMemo(() => gallerySig(sourceFaces), [sourceFaces]);
  const targetSig = useMemo(() => gallerySig(targetFaces), [targetFaces]);

  // Everything a cached preview depends on EXCEPT the target index, the frame
  // and the settings — shared by the main preview and by all three comparison
  // grids, which each used to re-spell it. A comment already claimed it was
  // shared; it was copied, so a fix to one copy silently missed the other.
  //
  // The engine version leads it because the rest of the key is settings, and
  // settings do not change when the BACKEND does. Restarting the server after a
  // pipeline change while leaving this webview open left every rendered cell
  // keyed identically, so a grid could show one model freshly rendered beside
  // another served from before the change and labelled "Cached" — read as a
  // difference between the two models. Comparing a new render against a stale
  // one is the one thing a comparison grid must not do.
  const cacheSuffix =
    `${meta?.git_version || 'dev'}_${sourceSig}_${targetSig}_${selSource}_${selTargetFace}`;

  const getCacheKey = (idx = selTarget, fr = frame) => {
    return `${idx}_${fr}_${previewKey}_${cacheSuffix}`;
  };

  const getCachedPreview = (idx = selTarget, fr = frame) => {
    const key = getCacheKey(idx, fr);
    return previewCacheRef.current[key];
  };

  const setCachedPreview = (idx, fr, data) => {
    const key = getCacheKey(idx, fr);
    const cache = previewCacheRef.current;
    const keys = Object.keys(cache);
    if (keys.length > 200) {
      delete cache[keys[0]]; // cap at 200 items to prevent memory bloat
    }
    cache[key] = data;
  };

  // Invalidate cache when source/target faces or selections change
  useEffect(() => {
    clearPreviewCache();
  }, [sourceSig, targetSig, selSource, selTargetFace]);

  // ── Shareable "recipe": full current settings + person→source mapping ──
  const exportRecipe = () => {
    try {
      const recipe = {
        type: 'roop-recipe',
        version: 1,
        exported_at: new Date().toISOString(),
        settings: { ...p },
        face_mapping: getFaceMappingArray(),
      };
      const blob = new Blob([JSON.stringify(recipe, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `roop_recipe_${Date.now()}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      notify('Recipe exported — share the .json to reproduce this exact setup');
    } catch (e) { notify('Failed to export recipe: ' + e.message, 'error'); }
  };

  const importRecipe = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const r = JSON.parse(event.target.result);
        if (r.type !== 'roop-recipe' || !r.settings) throw new Error('Not a valid roop recipe file.');
        setSettings((s) => ({ ...s, ...r.settings }));
        if (Array.isArray(r.face_mapping)) {
          const fm = {};
          r.face_mapping.forEach((v, i) => { fm[i] = v; });
          setFaceMapping(fm);
        }
        clearPreviewCache();
        notify('Recipe applied — settings and mapping restored');
      } catch (err) {
        notify('Failed to import recipe: ' + err.message, 'error');
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  // Register clipboard & drop listener with global App registry
  useEffect(() => {
    if (!registerFileListener) return;
    return registerFileListener((files) => {
      setPastedFiles(files);
      return true; // consumed
    });
  }, [registerFileListener]);

  // ── The one place a /api/swap body is built ──────────────────────────────
  // start() posts this directly; a queued job stores it verbatim and the server
  // replays it (app/routes_queue.py). One builder rather than two means a new
  // setting cannot reach a hand-started run and miss a queued one — which is
  // exactly what the queue's private copy of this list used to do.
  // `target_index` is deliberately absent: the queue resolves it from the target
  // NAME at dispatch time, because a stored index goes stale as soon as a target
  // is removed and the job then renders a different file than the one it names.
  const buildSwapPayload = (params = p) => {
    const sp = withSliderBypass(params);
    return {
      ...sp,
      enhancer: sp.selected_enhancer, detection: sp.face_detection_mode,
      output_method: sp.output_method, video_method: sp.video_swapping_method,
      upscale: sp.subsample_upscale, mask_engine: sp.mask_engine,
      mask_engine_2: sp.mask_engine_2, clip_text: sp.mask_clip_text,
      sam2_model_size: sp.sam2_model_size, track_identities: sp.track_identities,
      autorotate: sp.autorotate_faces,
      face_distance: num(sp.max_face_distance, 0.75), blend_ratio: num(sp.blend_ratio, 0.8),
      num_swap_steps: num(sp.num_swap_steps, 1),
      face_mapping: getFaceMappingArray(),
      imagemask: maskJson,
    };
  };

  // Describe the current setup as a queue job. `extra` carries a segment
  // (frame_start/frame_end) when one range of several is being queued.
  const currentJob = (extra = {}) => ({
    target_name: targets[selTarget]?.name || '',
    source_index: selSource,
    source_name: sourceFaces[selSource] ? `Face ${selSource + 1}` : 'Selected face',
    payload: buildSwapPayload(),
    ...extra,
  });

  const addToQueue = async () => {
    if (targets.length === 0) { notify('Load target media first', 'error'); return; }
    if (sourceFaces.length === 0) { notify('Add a source face first', 'error'); return; }
    const job = currentJob();
    if (await queue.add(job)) notify(`Added "${job.target_name}" to the batch queue`);
  };

  // Put a queued job's settings back into the editor so it can be adjusted and
  // re-queued (or the queued one updated in place).
  const loadJobSettings = (job) => {
    if (!job?.payload) return;
    setSettings((s) => ({ ...(s || {}), ...job.payload }));
    const idx = targets.findIndex((t) => t.name === job.target_name);
    if (idx >= 0) selectTarget(idx);
  };

  const startQueue = () => queue.start();

  // ── Segments ─────────────────────────────────────────────────────────────
  // One run renders one range, because the trim lives on the target entry
  // rather than in the swap payload. So N ranges = N queued jobs, each of which
  // sets its own trim just before it dispatches, and an optional join at the
  // end to hand back a single file.
  const queueSegments = async () => {
    if (targets.length === 0 || sourceFaces.length === 0) {
      notify('Load a target and a source face first', 'error');
      return;
    }
    if (segments.segments.length === 0) return;
    const payload = buildSwapPayload();
    const name = targets[selTarget]?.name || '';
    await queue.addMany(segments.segments.map((s, i) => ({
      target_name: name,
      source_index: selSource,
      source_name: sourceFaces[selSource] ? `Face ${selSource + 1}` : 'Selected face',
      label: `${name} — segment ${i + 1}`,
      payload,
      frame_start: s.start,
      frame_end: s.end,
    })));
    notify(`Queued ${segments.segments.length} segment${segments.segments.length === 1 ? '' : 's'} of "${name}"`);
  };

  // Finished segment jobs for THIS target, in timeline order — the join has to
  // concatenate them in clip order, not in the order they happened to render.
  const joinableJobs = React.useMemo(() => {
    const name = targets[selTarget]?.name;
    if (!name) return [];
    return queue.jobs
      .filter((j) => j.target_name === name && j.status === 'finished'
                     && j.frame_start != null && (j.outputs || []).length > 0)
      .sort((a, b) => a.frame_start - b.frame_start);
  }, [queue.jobs, targets, selTarget]);

  const joinSegments = async () => {
    try {
      const res = await postJSON('/api/queue/join', { ids: joinableJobs.map((j) => j.id) });
      notify(`Joined ${res.segments} segments into ${res.name}`, 'success');
    } catch (e) { notify(e.message, 'error'); }
  };

  const jumpToSegment = (s) => {
    setFrameMarkerVal('start', s.start);
    setFrameMarkerVal('end', s.end);
    setFrame(s.start);
  };

  // Custom Timeline and Playback States
  // Buffered, video-style playback: the play/loop/rate state, the rolling frame
  // buffer behind it, and the rAF clock that drives the playhead. See
  // faceswap/usePlaybackBuffer.
  //
  // Declared HERE, where the state it replaced was, and not down beside the
  // effects it used to live among. `isPlaying` is read by dependency arrays
  // several hundred lines above those effects, and a dependency array is
  // evaluated as the component body runs — so a later declaration is a
  // temporal-dead-zone throw on the first render, which neither the build nor
  // oxlint's no-undef reports.
  const {
    isPlaying, setIsPlaying,
    isLooping, setIsLooping,
    playbackRate, setPlaybackRate,
    bufferedSrc, playStalled,
  } = usePlaybackBuffer({ frame, setFrame, selTarget, maxFrames, targets });

  const [isScrubbing, setIsScrubbing] = useState(false);
  const [dragType, setDragType] = useState('playhead'); // 'playhead', 'start', 'end'
  const [storyboardThumbs, setStoryboardThumbs] = useState([]);
  const storyboardRef = useRef([]);   // latest strip, for blob cleanup on unmount
  const [hoverFrame, setHoverFrame] = useState(null);
  // Timeline view window (inclusive frame range the track currently shows).
  // On a 95k-frame clip the full-width track puts ~100 frames in every pixel,
  // so it can only ever be a rough locator; zooming makes it a working surface.
  // Every frame<->pixel conversion below goes through this, and it is read via
  // a ref so a zoom during a drag can't tear down the scrub listeners.
  const [view, setView] = useState({ start: 1, end: 1 });
  const viewRef = useRef(view);
  viewRef.current = view;
  const timelineRef = useRef(null);
  // Coalesces timeline hover/scrub pointer-move work to one update per frame.
  // Pointer-move fires faster than this large component can re-render, so
  // without batching the events pile up and scrubbing/hovering feels sticky.
  const timelineRafRef = useRef(null);
  const timelinePendingRef = useRef(null);
  const [isGeneratingPreviewClip, setIsGeneratingPreviewClip] = useState(false);
  const [origStartEnd, setOrigStartEnd] = useState(null);


  const previewBusyRef = useRef(false);   // a /api/preview call is in flight
  const previewPendingRef = useRef(null); // latest queued request while busy (coalesced)

  // p = the swap parameters, seeded from CFG (settings) and patched locally.
  const p = settings || {};
  const set = (k, v) => setSettings((s) => ({ ...s, [k]: v }));
  // Several keys in ONE update. Two sequential set() calls each build off the
  // same stale snapshot, so the second silently drops the first — which for
  // the region panel would mean toggling a region also reverted its grow.
  const setMany = (patch) => setSettings((s) => ({ ...s, ...patch }));

  // Manual mask painted in the preview box. ProcessMgr already understands this
  // format ({"<faceset>": {exclude, canonical, ref_kps}}) — it was simply never
  // being sent: both /api/preview and /api/swap passed a literal None for the
  // imagemask, so the brush was a drawing toy. Painted areas KEEP THE ORIGINAL
  // face (exclude semantics), which is what the red paint has always implied.
  const maskJson = useMemo(() => {
    if (!manualMask) return '';
    const entry = { exclude: manualMask, canonical: false };
    // ref_kps lets the backend warp a full-frame mask into the face-crop space.
    // Without it the mask can only be interpreted as already-canonical, which a
    // frame-space painting is not.
    if (maskRefKps) entry.ref_kps = maskRefKps;
    return JSON.stringify({ [String(selSource)]: entry });
  }, [manualMask, maskRefKps, selSource]);

  const applyManualMask = (dataUrl) => {
    setManualMask(dataUrl);
    // Bind the mask to the currently selected target face's keypoints so the
    // backend can map frame-space paint onto the aligned crop.
    setMaskRefKps(dataUrl ? (previewKps[selTargetFace] || previewKps[0] || null) : null);
    setMaskVersion((v) => v + 1);
  };

  // "Slider Effect: BYPASSED" neutralises the eight Slider Tracker values.
  // This MUST be applied to the render payload too, not just the preview —
  // otherwise the preview shows a bypassed result while Run quietly renders
  // with the slider values still applied, and the two disagree.
  const withSliderBypass = (params) =>
    sliderEffectEnabled ? params : { ...params, ...TRACKER_BYPASS_VALUES };

  // Flip the toggle only. The refresh CANNOT happen inside the setState updater
  // (as it used to): an updater must be pure, React 19 StrictMode runs it twice
  // in dev, and — the real defect — refreshPreview() called from in there still
  // closes over the PREVIOUS sliderEffectEnabled, so the toggle rendered a
  // preview of the state you just left. Refreshing from an effect means the new
  // value is committed before the request is built.
  const toggleSliderEffect = () => setSliderEffectEnabled((v) => !v);
  const sliderEffectMounted = useRef(false);
  useEffect(() => {
    if (!sliderEffectMounted.current) { sliderEffectMounted.current = true; return; }
    refreshPreview({ force: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sliderEffectEnabled]);

  // While face-swap preview is on, auto-refresh when the swapped result would
  // change: new source faces, target faces, or any swap/mask parameter.
  // ── The one place UI params become an /api/preview request ──────────────
  // Both the request body and the cache/refresh signature are built from this
  // single function, and the signature is literally the request body with the
  // frame coordinates zeroed. That makes the failure this replaces structurally
  // impossible: a setting cannot be sent to the backend without also
  // invalidating the cache (a control that silently does nothing), nor keyed
  // without being sent. There used to be six hand-maintained copies of this
  // list -- and they had already drifted: the three comparison-grid keys were
  // missing sam2_model_size, face_detector_nms and face_mapping, so those grids
  // served stale cells when any of the three changed.
  //
  // `params` is the settings object to read from: `p` normally, or a grid's
  // `localParams` (p with that grid's one override applied), which reproduces
  // the override in the request for free.
  const buildPreviewPayload = (params, { index, frame: fr, fake, ...overrides } = {}) => {
    const activeParams = withSliderBypass(params);
    return {
      index, frame: fr, fake_preview: fake,
      enhancer: activeParams.selected_enhancer, codeformer_fidelity: num(activeParams.codeformer_fidelity, 0.5),
      detection: activeParams.face_detection_mode,
      face_distance: num(activeParams.max_face_distance, 0.75), blend_ratio: num(activeParams.blend_ratio, 0.8),
      mask_engine: activeParams.mask_engine, mask_engine_2: activeParams.mask_engine_2,
      clip_text: activeParams.mask_clip_text,
      no_face_action: activeParams.no_face_action, vr_mode: activeParams.vr_mode, autorotate: activeParams.autorotate_faces,
      show_mask_offsets: activeParams.show_mask_offsets, restore_original_mouth: activeParams.restore_original_mouth,
      num_swap_steps: num(activeParams.num_swap_steps, 1), upscale: activeParams.subsample_upscale,
      use_3d_recon: activeParams.use_3d_recon, use_source_bank: activeParams.use_source_bank,
      use_frontalization: activeParams.use_frontalization, frontalization_threshold: num(activeParams.frontalization_threshold, 30),
      jaw_reshape: activeParams.jaw_reshape, jaw_reshape_strength: num(activeParams.jaw_reshape_strength, 0.5),
      detail_transfer_strength: num(activeParams.detail_transfer_strength, 0),
      enhancer_align: activeParams.enhancer_align,
      color_match_after_enhance: activeParams.color_match_after_enhance,
      lipsync_enabled: activeParams.lipsync_enabled,
      lipsync_audio_source: activeParams.lipsync_audio_source,
      lipsync_audio_path: activeParams.lipsync_audio_path,
      restore_original_eyes: activeParams.restore_original_eyes,
      eyes_blend_amount: num(activeParams.eyes_blend_amount, 1),
      eyes_feather_blend: num(activeParams.eyes_feather_blend, 25),
      eyes_size_factor: num(activeParams.eyes_size_factor, 1),
      eyes_radius_x: num(activeParams.eyes_radius_x, 1),
      eyes_radius_y: num(activeParams.eyes_radius_y, 1),
      parser_regions: activeParams.parser_regions,
      parser_region_grow: activeParams.parser_region_grow,
      merger_hist_match: num(activeParams.merger_hist_match, 0),
      merger_sharpen: num(activeParams.merger_sharpen, 0),
      merger_motion_blur: num(activeParams.merger_motion_blur, 0),
      merger_grain_match: num(activeParams.merger_grain_match, 0),
      merger_degrade: num(activeParams.merger_degrade, 0),
      merger_clarity: num(activeParams.merger_clarity, 0),
      output_face_scale: num(activeParams.output_face_scale, 0),
      expression_restore_strength: num(activeParams.expression_restore_strength, 0),
      expression_restore_region: activeParams.expression_restore_region || 'all',
      swap_model: activeParams.swap_model, default_det_size: activeParams.default_det_size,
      face_detector_size: activeParams.face_detector_size, face_detector_threshold: activeParams.face_detector_threshold,
      face_detector_nms: activeParams.face_detector_nms,
      color_transfer_mode: activeParams.color_transfer_mode,
      target_conditioned_appearance: !!activeParams.target_conditioned_appearance,
      target_conditioned_appearance_strength: num(activeParams.target_conditioned_appearance_strength, 0.75),
      target_conditioned_appearance_temporal_alpha: num(activeParams.target_conditioned_appearance_temporal_alpha, 0.30),
      sam2_model_size: activeParams.sam2_model_size,
      refine_landmarks: activeParams.refine_landmarks,
      swap_model_mask_strength: activeParams.swap_model_mask_strength,
      rescue_small_faces: activeParams.rescue_small_faces,
      detector_engine: activeParams.detector_engine,
      face_mapping: getFaceMappingArray(),
      mask_top: activeParams.mask_top,
      mask_bottom: activeParams.mask_bottom,
      mask_left: activeParams.mask_left,
      mask_right: activeParams.mask_right,
      face_mask_blend: activeParams.face_mask_blend,
      mouth_mask_blend: activeParams.mouth_mask_blend,
      mouth_top_scale: activeParams.mouth_top_scale,
      mouth_bottom_scale: activeParams.mouth_bottom_scale,
      mouth_left_scale: activeParams.mouth_left_scale,
      mouth_right_scale: activeParams.mouth_right_scale,
      // Manual brush mask (JSON string, '' when nothing is painted).
      imagemask: maskJson,
      ...overrides,
    };
  };

  const resetTrackerSliders = () => {
    Object.entries(TRACKER_DEFAULT_VALUES).forEach(([k, v]) => set(k, v));
    refreshPreview({ force: true });
  };

  // index/frame are separate cache dimensions, so they are zeroed here rather
  // than being part of the settings signature. imagemask is swapped for its
  // short version token: it is a base64 PNG that can run to tens of kilobytes,
  // and the signature is re-stringified on every render. The token still bumps
  // on every mask edit, so the invalidate-when-it-changes guarantee holds.
  const previewSignature = (params, fake) =>
    JSON.stringify(buildPreviewPayload(params, {
      index: 0, frame: 0, fake, imagemask: `mask:${maskVersion}`,
    }));

  const previewKey = previewSignature(p, fakePreview);

  // One-click speed/quality profiles. Each bundles the core levers (detection
  // resolution, pixel-boost upscale, enhancer, swap steps); other settings (mask
  // engine, target selection, etc.) are left as-is. Pure UI — no runtime cost.
  const PRESETS = {
    Fast:     { default_det_size: false, face_detector_size: '320', face_detector_threshold: 0.50, subsample_upscale: '128px', selected_enhancer: 'None',     num_swap_steps: 1 },
    Balanced: { default_det_size: true,  face_detector_size: '640', face_detector_threshold: 0.50, subsample_upscale: '256px', selected_enhancer: 'GPEN',     num_swap_steps: 1 },
    Quality:  { default_det_size: true,  face_detector_size: '640', face_detector_threshold: 0.50, subsample_upscale: '512px', selected_enhancer: 'UltraMax', swap_model: 'realswap', mask_engine: 'RealityUX', num_swap_steps: 1 },
  };
  const activePreset = Object.keys(PRESETS).find((name) =>
    Object.entries(PRESETS[name]).every(([k, v]) =>
      k === 'default_det_size' ? (p[k] !== false) === (v !== false) : p[k] === v));
  const applyPreset = (name) => {
    setSettings((s) => ({ ...s, ...PRESETS[name] }));
    notify(`Applied ${name} preset`, 'info');
  };

  // The user's own "Save as default" snapshot, and the resets that read it.
  // See faceswap/useUserDefaults.
  const { userDefaults, saveAsDefault, resetToDefaults, clearUserDefault } =
    useUserDefaults({ settings: p, setSettings, notify });

  // ── View state across a webview reload ───────────────────────────────────
  // Where you were looking (target, playhead, rendered frame), remembered
  // across the webview reload a Pinokio tab switch causes. Captured at mount
  // and read once by the rehydrate effect below — see
  // faceswap/useViewPersistence.
  const restoredViewRef = useViewPersistence({ selTarget, frame, previewSrc });

  // ── initial rehydrate ──
  // Pinokio reloads the webview whenever you switch the RUN/DEV/FILES tabs,
  // which remounts this component and wipes its React state. The backend keeps
  // running, so we restore both the faces/targets AND the live job state.
  useEffect(() => {
    getJSON('/api/state').then((st) => {
      checkDesync(st);
      setSourceFaces(st.source_faces || []);
      if (st.source_faces_info) setSourceFacesInfo(st.source_faces_info);
      setTargetFaces(st.target_faces || []);
      setTargetGroups(st.target_groups || []);
      setTargetNames(st.target_names || []);
      setTargetFacesInfo(st.target_faces_info || []);
      const tg = st.targets || [];
      setTargets(tg);
      if (tg.length > 0) {
        // The BACKEND owns which target is selected — never override it from
        // storage. The playhead and the rendered preview are client-only, so
        // they are restored, but only when they belong to that same target and
        // are still in range; otherwise showing them would be a lie about which
        // frame you are looking at.
        const sel = st.selected_target_index || 0;
        const mf = tg[sel]?.frames || 1;
        const v = restoredViewRef.current;
        const sameView = v && v.target === sel && v.frame >= 1 && v.frame <= mf;
        setSelTarget(sel);
        setMaxFrames(mf);
        setFrame(sameView ? v.frame : 1);
        if (sameView && v.image) { setPreviewSrc(v.image); setPreviewFor(`${sel}_${v.frame}`); }
      }
    }).catch(() => {});

    // Restore an in-flight swap so the run bar shows Pause/Resume/Stop and the
    // progress %/desc again instead of falling back to "Start Swapping".
    getJSON('/api/progress').then((pr) => {
      setProgress(pr);
      if (pr.processing) {
        // Prefer the backend's own start time: the run has been going since
        // before this mount, so starting a fresh clock here would report a
        // long render as a few seconds old and skew the ETA with it.
        if (!startTime) setStartTime(pr.started_at ? pr.started_at * 1000 : Date.now());
      }
    }).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const refreshPreview = async (opts = {}) => {
    if (targets.length === 0) { setPreviewSrc(''); setPreviewFor(''); return; }

    const idx = opts.index ?? selTarget;
    const fr = opts.frame ?? frame;
    const fake = opts.fake ?? fakePreview;

    // Check client-side preview cache first.
    //
    // `force` skips it AND evicts the entry. The cache key covers every setting
    // that reaches the backend, so an automatic refresh after a settings change
    // is always a miss and re-renders correctly. But the manual Refresh button
    // used to go through this same lookup, which made it a no-op whenever the
    // cache was warm: nothing the user could do would re-run the swap for the
    // current frame. Anything the key cannot see — a source file edited on disk,
    // a model swapped underneath us, a backend restart — left a stale image with
    // no way to clear it.
    if (!opts.force) {
      const cached = getCachedPreview(idx, fr);
      if (cached) {
        setPreviewFaces(cached.faces);
        setPreviewPersonIds(cached.personIds || []);
        setPreviewKps(cached.kps || []);
        setPreviewPose(cached.pose || []);
        setPreviewSrc(cached.image);
        setPreviewFor(`${idx}_${fr}`);
        return;
      }
    } else {
      delete previewCacheRef.current[getCacheKey(idx, fr)];
    }

    // Single-flight: the backend's live_swap shares one (non-thread-safe)
    // ProcessMgr on the GPU. Two overlapping /api/preview calls corrupt/hang
    // TensorRT/CUDA. So never run two at once — queue the latest request and
    // run it once the current one finishes.
    if (previewBusyRef.current) { 
      previewPendingRef.current = { ...opts, index: idx, frame: fr, fake: fake }; 
      return; 
    }
    
    previewBusyRef.current = true;
    setPreviewing(true);
    // Safety net: the first run of a new model downloads it and builds a
    // TensorRT/CUDA engine (minutes). Abort after 15 min so a genuine hang can
    // never wedge the single-flight guard permanently.
    const ctrl = new AbortController();
    const killer = setTimeout(() => ctrl.abort(), 15 * 60 * 1000);
    try {
      const res = await postJSON('/api/preview', buildPreviewPayload(p, { index: idx, frame: fr, fake }), { signal: ctrl.signal });
      if (res.faces) setPreviewFaces(res.faces);
      setPreviewPersonIds(res.person_ids || []);
      setPreviewKps(res.kps || []);
      setPreviewPose(res.pose || []);
      setPreviewSrc(res.image || '');
      setPreviewFor(res.image ? `${idx}_${fr}` : '');
      if (res.image) {
        setCachedPreview(idx, fr, { faces: res.faces || [], personIds: res.person_ids || [], kps: res.kps || [], pose: res.pose || [], image: res.image });
      }
    } catch (e) {
      notify(e.name === 'AbortError' ? 'Preview timed out (model build took too long)' : e.message, 'error');
    }
    finally {
      clearTimeout(killer);
      previewBusyRef.current = false;
      setPreviewing(false);
      if (previewPendingRef.current) {
        const next = previewPendingRef.current;
        previewPendingRef.current = null;
        refreshPreview(next);
      }
    }
  };

  // ── Comparison-grid preview loaders ─────────────────────────────────────
  // Enhancers, mask engines and swapper models each render one preview per
  // selected variant with everything else held fixed. That was the same seventy
  // lines three times over; it is one hook now, differing only in which settings
  // key is varied and which values are legal. See faceswap/useGridPreviewLoader.
  //
  // The cache key's tail — see `cacheSuffix` above, which the main preview uses
  // too so the two can never drift.
  const gridCommon = {
    settings: p, fakePreview, selTarget, frame, targetCount: targets.length,
    buildPreviewPayload, previewSignature, previewCacheRef,
    cacheSuffix, reloadKey: previewKey,
  };

  useGridPreviewLoader({
    ...gridCommon,
    enabled: comparingEnhancers, selection: selectedGridEnhancers,
    allowed: meta.enhancers, paramKey: 'selected_enhancer',
    setPreviews: setEnhancerPreviews, setTimes: setEnhancerTimes,
    setTimers: setLiveRenderingTimers, intervalsRef: activeIntervalsRef,
  });

  useGridPreviewLoader({
    ...gridCommon,
    enabled: comparingMasks, selection: selectedGridMasks,
    allowed: meta.mask_engines, paramKey: 'mask_engine',
    setPreviews: setMaskPreviews, setTimes: setMaskTimes,
    setTimers: setMaskRenderTimers, intervalsRef: maskIntervalsRef,
  });

  useGridPreviewLoader({
    ...gridCommon,
    enabled: comparingSwappers, selection: selectedGridSwappers,
    allowed: meta.swap_models, paramKey: 'swap_model',
    setPreviews: setSwapperPreviews, setTimes: setSwapperTimes,
    setTimers: setSwapperRenderTimers, intervalsRef: swapperIntervalsRef,
  });

  // ── AI-upscale grid preview loader ─────────────────────────────────────
  // Unlike the enhancer/swapper grids (which re-run the full swap per cell),
  // this swaps the frame ONCE then upscales that single result with each
  // selected model — so the grid isolates the upscaler's effect (and is much
  // cheaper: one swap + N upscales instead of N full swaps).
  const loadUpscalePreviews = async (activeCheck) => {
    if (targets.length === 0) return;
    const labels = AI_UPSCALE_MODELS.map(m => m.label);
    const available = selectedGridUpscalers.filter(l => labels.includes(l));

    const keepOnly = (prev) => {
      const reset = {};
      for (const l of available) if (prev[l]) reset[l] = prev[l];
      return reset;
    };
    setUpscalePreviews(keepOnly);
    setUpscaleTimes(keepOnly);
    setUpscaleRenderTimers(keepOnly);

    // Swap the frame ONCE to get the base image every cell upscales. fake_preview
    // is forced on so the grid always compares upscalers on the swapped result
    // (falls back to the raw frame server-side when there are no source faces).
    let baseImage = '';
    try {
      const baseRes = await postJSON('/api/preview', buildPreviewPayload(p, { index: selTarget, frame, fake: true }));
      baseImage = baseRes.image || '';
    } catch {
      // handled below (no base → nothing to upscale)
    }
    if (!activeCheck()) return;
    if (!baseImage) return;

    for (const label of available) {
      if (!activeCheck()) return;
      const subtype = AI_UPSCALE_MODELS.find(m => m.label === label)?.value || 'esrganx2';
      try {
        const start = Date.now();
        setUpscaleRenderTimers(prev => ({ ...prev, [label]: '0.0s' }));
        upscaleIntervalsRef.current[label] = setInterval(() => {
          setUpscaleRenderTimers(prev => ({ ...prev, [label]: ((Date.now() - start) / 1000).toFixed(1) + 's' }));
        }, 100);

        const res = await postJSON('/api/preview_upscale', { image: baseImage, subtype });

        const duration = ((Date.now() - start) / 1000).toFixed(2);
        if (upscaleIntervalsRef.current[label]) {
          clearInterval(upscaleIntervalsRef.current[label]);
          delete upscaleIntervalsRef.current[label];
        }
        if (!activeCheck()) return;
        if (res.image) {
          setUpscalePreviews((prev) => ({ ...prev, [label]: res.image }));
          setUpscaleTimes((prev) => ({ ...prev, [label]: `${duration}s` }));
          setUpscaleRenderTimers((prev) => ({ ...prev, [label]: null }));
        }
      } catch {
        if (upscaleIntervalsRef.current[label]) {
          clearInterval(upscaleIntervalsRef.current[label]);
          delete upscaleIntervalsRef.current[label];
        }
        setUpscaleRenderTimers((prev) => ({ ...prev, [label]: null }));
        // Fail silently (a model may fail to download or init on a single frame)
      }
    }
  };

  /* eslint-disable react-hooks/exhaustive-deps -- intentional: loadUpscalePreviews is a stable closure invoked on trigger */
  useEffect(() => {
    if (!comparingUpscalers || targets.length === 0) return;
    let active = true;
    loadUpscalePreviews(() => active);
    return () => {
      active = false;
      if (upscaleIntervalsRef.current) {
        Object.values(upscaleIntervalsRef.current).forEach(clearInterval);
        upscaleIntervalsRef.current = {};
      }
    };
  }, [comparingUpscalers, selectedGridUpscalers, frame, selTarget, targets.length, sourceSig, targetSig, selSource, selTargetFace, previewKey]);
  /* eslint-enable react-hooks/exhaustive-deps */

  // Live elapsed timer for the "Rendering…" badge so a slow first run reads as
  // working, not hung.
  useEffect(() => {
    if (!previewing) { setPreviewSecs(0); return; }
    const started = Date.now();
    setPreviewSecs(0);
    const id = setInterval(() => setPreviewSecs(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(id);
  }, [previewing]);

  // Auto-refresh preview when the target list, selected target, or frame
  // changes (targets.length covers initial rehydrate after a page refresh).
  useEffect(() => {
    if (targets.length === 0 || progress.processing || isScrubbing || isPlaying) return;
    const t = setTimeout(() => refreshPreview(), 250);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selTarget, frame, targets.length, isScrubbing, isPlaying]);

  useEffect(() => {
    if (targets.length === 0 || isScrubbing || isPlaying) return;
    // A run is in flight — the GPU is busy, so remember that the preview is now
    // out of date and re-render it once the run finishes. Without this, faces
    // changed during a run leave the previous result frozen on screen.
    if (progress.processing) { previewDeferredRef.current = true; return; }
    const t = setTimeout(() => refreshPreview(), 350);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewKey, sourceSig, targetSig, isScrubbing, isPlaying]);

  useEffect(() => {
    if (progress.processing || !previewDeferredRef.current) return;
    previewDeferredRef.current = false;
    if (targets.length === 0 || isScrubbing || isPlaying) return;
    const t = setTimeout(() => refreshPreview(), 350);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [progress.processing]);

  // ── source / target file handling ──
  // A cancelled upload is an outcome, not a failure: the user asked for it, so
  // it gets no red toast. Everything else still reports.
  const reportUploadError = (err) => {
    if (err?.name === 'AbortError') { notify('Upload cancelled', 'info'); return; }
    notify(err.message, 'error');
  };

  const onAddSource = async (files) => {
    if (!files || !files.length) return;
    const before = sourceFaces.length;
    const ctrl = new AbortController();
    srcAbortRef.current = ctrl;
    setUploadingSrc(true);
    setSrcProgress(null);
    try {
      const res = checkDesync(await postFiles('/api/source/add', files, undefined, {
        onProgress: setSrcProgress, signal: ctrl.signal,
      }));
      setSourceFaces(res.source_faces);
      if (res.source_faces_info) setSourceFacesInfo(res.source_faces_info);
      const added = res.source_faces.length - before;
      if (added > 0) notify(`Loaded ${added} face(s) — ${res.faceset_count} faceset(s) total`);
      else notify('No face detected in the uploaded file(s)', 'error');
    } catch (err) { reportUploadError(err); }
    finally { setUploadingSrc(false); setSrcProgress(null); srcAbortRef.current = null; }
  };

  // Everything that happens once the backend has accepted new targets, however
  // they arrived — uploaded, or referenced by path. Shared so the two routes
  // cannot drift on the auto-queue rule or the preview refresh.
  const applyTargetAdd = async (res, beforeCount) => {
      const newTargetsList = res.targets || [];
      setTargets(newTargetsList);
      setSelTarget(res.selected_target_index || 0);
      const mf = newTargetsList[res.selected_target_index || 0]?.frames || 1;
      setMaxFrames(mf); setFrame(1);
      refreshPreview({ index: res.selected_target_index || 0, frame: 1 });
      notify(`Added ${newTargetsList.length - beforeCount} target(s)`);

      // Automatically add videos to batch queue if more than 1 video is uploaded
      const newTargets = newTargetsList.slice(beforeCount);
      const newVideos = newTargets.filter(t => t.frames > 1);
      if (newVideos.length > 1) {
        const payload = buildSwapPayload();
        await queue.addMany(newVideos.map((t) => ({
          target_name: t.name || '',
          source_index: selSource,
          source_name: sourceFaces[selSource] ? `Face ${selSource + 1}` : 'Selected face',
          payload,
        })));
        notify(`Automatically queued ${newVideos.length} uploaded videos`, 'success');
      }
  };

  const onAddTarget = async (files) => {
    if (!files || !files.length) return;
    const ctrl = new AbortController();
    tgtAbortRef.current = ctrl;
    setUploadingTgt(true);
    setTgtProgress(null);
    try {
      const beforeCount = targets.length;
      await applyTargetAdd(await postFiles('/api/target/add', files, undefined, {
        onProgress: setTgtProgress, signal: ctrl.signal,
      }), beforeCount);
    } catch (err) { reportUploadError(err); }
    finally { setUploadingTgt(false); setTgtProgress(null); tgtAbortRef.current = null; }
  };

  const onAddLipsyncAudio = async (files) => {
    if (!files || !files.length) return;
    setUploadingLipsyncAudio(true);
    try {
      const res = await postFile('/api/lipsync/audio/add', files[0]);
      set('lipsync_audio_path', res.path);
    } catch (err) { notify(err.message, 'error'); }
    finally { setUploadingLipsyncAudio(false); }
  };

  // Targets that are already on this machine. No transfer and no second copy
  // in temp/ — see /api/target/add_path for why that matters on a loopback app.
  const onAddTargetPaths = async (paths) => {
    if (!paths || !paths.length) return;
    setUploadingTgt(true);
    try {
      const beforeCount = targets.length;
      const res = await postJSON('/api/target/add_path', { paths });
      // Report rejects individually: with several paths pasted at once, a
      // silent drop looks like the app ignored the whole thing.
      (res.rejected || []).forEach((r) => notify(`Skipped ${r.path} — ${r.why}`, 'error'));
      if ((res.added || []).length) await applyTargetAdd(res, beforeCount);
    } catch (err) { notify(err.message, 'error'); }
    finally { setUploadingTgt(false); }
  };

  const removeTarget = async (i) => {
    const res = await postJSON('/api/target/remove', { index: i });
    setTargets(res.targets);
    const newSel = res.selected_target_index || 0;
    setSelTarget(newSel);
    if (res.targets.length === 0) { setPreviewSrc(''); setPreviewFor(''); setMaxFrames(1); }
    else { setMaxFrames(res.targets[newSel]?.frames || 1); setFrame(1); }
  };

  const selectTarget = async (i) => {
    setSelTarget(i);
    const res = await postJSON('/api/target/select', { index: i });
    setTargets(res.targets);
    const mf = res.targets[i]?.frames || 1;
    setMaxFrames(mf); setFrame(1);
    refreshPreview({ index: i, frame: 1 });
  };

  // The backend reports it if the faceset list and the gallery thumbnails have
  // fallen out of step. That state used to be invisible — the gallery showed a
  // face the swap didn't have, and the only symptom was that person silently
  // not being swapped — so say so loudly wherever a gallery payload arrives.
  const checkDesync = (res) => {
    if (res?.desync) notify(`Source faces: ${res.desync}`, 'error');
    return res;
  };

  const sourceAction = async (path, body) => {
    try {
      const res = checkDesync(await postJSON(path, body));
      if (res.source_faces) setSourceFaces(res.source_faces);
      if (res.source_faces_info) setSourceFacesInfo(res.source_faces_info);
    } catch (e) {
      // Surface failures (e.g. a 404 when the backend hasn't been restarted to
      // pick up a new endpoint) instead of silently doing nothing.
      notify(`${path.split('/').pop()} failed: ${e.message}. If this is a new feature, restart the app server.`, 'error');
    }
  };

  // Removing a source faceset shifts every later index down by one. The
  // person→source mapping stores raw indices, so without this it keeps pointing
  // at whatever slid into the removed slot (wrong face) or off the end of the
  // list (the backend then swaps in an empty faceset).
  const removeSource = async (i) => {
    await sourceAction('/api/source/remove', { index: i });
    setFaceMapping((prev) => {
      const next = {};
      for (const [pid, src] of Object.entries(prev || {})) {
        if (typeof src !== 'number' || src === i) continue;  // dropped → falls back to default
        next[pid] = src > i ? src - 1 : src;
      }
      return next;
    });
    const nextSel = selSource === i ? 0 : selSource > i ? selSource - 1 : selSource;
    if (nextSel !== selSource) {
      setSelSource(nextSel);
      try { await postJSON('/api/source/select', { index: nextSel }); } catch { /* selection is best-effort */ }
    }
  };

  const selectSource = async (i) => { setSelSource(i); await postJSON('/api/source/select', { index: i }); };

  const useFaceFromFrame = async () => {
    try {
      const res = await postJSON('/api/target/use_face', { index: selTarget, frame });
      setTargetFaces(res.target_faces);
      setTargetGroups(res.target_groups || []);
      setTargetNames(res.target_names || []);
      setTargetFacesInfo(res.target_faces_info || []);
      set('face_detection_mode', 'Selected face');
      notify(`Added ${res.count} target person(s)`);
    } catch (e) { notify(e.message, 'error'); }
  };

  // Spot-check the AI upscale on just the current preview frame — sends the
  // exact swapped image the preview shows to the backend, which upscales the
  // one frame and returns it for inspection in a modal.
  const upscaleThisFrame = async () => {
    if (!previewSrc) { notify('No preview to upscale', 'error'); return; }
    setUpscaling(true);
    try {
      const res = await postJSON('/api/preview_upscale', {
        image: previewSrc,
        subtype: p.upscale_model_after || 'esrganx2',
      });
      if (!res.image) throw new Error(res.message || 'upscale failed');
      setUpscaledSrc(res.image);
      setUpscaledDims(res.width && res.height ? { w: res.width, h: res.height } : null);
    } catch (e) {
      notify(e.message, 'error');
    } finally {
      setUpscaling(false);
    }
  };

  // Click a numbered face box in the live preview to capture just that person
  // as a NEW target face (face_index = its left-to-right box order).
  const addPersonFromBox = async (faceIndex) => {
    try {
      const res = await postJSON('/api/target/use_face', { index: selTarget, frame, face_index: faceIndex });
      if (!res.count) { notify('No face found for that box', 'error'); return; }
      setTargetFaces(res.target_faces);
      setTargetGroups(res.target_groups || []);
      setTargetNames(res.target_names || []);
      setTargetFacesInfo(res.target_faces_info || []);
      set('face_detection_mode', 'Selected face');
      notify(`Added Person ${(previewPersonIds[faceIndex] ?? faceIndex) + 1} to target faces`);
    } catch (e) { notify(e.message, 'error'); }
  };

  const setFrameMarkerVal = async (which, val) => {
    if (!val || isNaN(val)) return;
    try {
      const res = await postJSON('/api/target/set_frame', { which, frame: val });
      if (res.targets) setTargets(res.targets);
      notify(`Set ${which} frame = ${val}`);
    } catch (e) { notify(e.message, 'error'); }
  };

  // ── start / stop ──
  const start = async () => {
    try {
      await postJSON('/api/settings', p);            // persist CFG
      await postJSON('/api/swap', buildSwapPayload());
      setStartTime(Date.now());
      // Ask for desktop-notification permission so we can ping on completion if
      // the tab is backgrounded (no-op if already decided).
      try { if ('Notification' in window && Notification.permission === 'default') Notification.requestPermission(); } catch { /* ignore */ }
      // Optimistically flag processing so the old "Latest output" clears
      // immediately, before the first poll tick (~1s) confirms it.
      setProgress((pr) => ({ ...pr, processing: true, paused: false, progress: 0, desc: 'Starting…' }));
      notify('Processing started');
    } catch (e) { notify(e.message, 'error'); }
  };

  // Stop stays here because the floating dock offers it from this tab. Pause and
  // Resume moved to the Processing tab with the run bar that carried them.
  const stop = async () => { await postJSON('/api/stop', {}); notify('Stopping…', 'info'); };

  // Hide the previous "Latest output" while a job is running so a new upload +
  // run never shows a stale result. The poll keeps reporting the old _last_output
  // until the new job finishes, so we gate on `processing` rather than clearing
  // progress.output (which the next poll tick would just restore).
  const out = progress.processing ? null : progress.output;
  const outUrl = out?.path ? `${API}/api/file?path=${encodeURIComponent(out.path)}&t=${progress.progress}` : '';
  const prog = progress.progress || 0;

  const elapsedMs = progress.processing && startTime ? Date.now() - startTime : 0;

  // "Time left" comes from the terminal's own progress bar (`eta_s`), so the two
  // agree by construction. It is not derived here any more:
  // elapsed * (1 - prog) / prog assumes the whole run so far went at the rate
  // the finished frames went at, and it did not — model loads, TensorRT engine
  // builds and the temporal pre-pass all bill minutes against a frame counter
  // still sitting at zero, then get extrapolated over the remaining 84%. On a
  // real run (12m47s in, 7,233/44,755 frames, 22.5 fps) the bar said 28 minutes
  // and this formula said 66.
  //
  // The old formula stays as the fallback for the windows where no bar is
  // counting frames — start-up, and the encode/mux tail — because a rough
  // number that moves beats a blank. `eta_s` is null in exactly those windows.
  const etaMs = progress.processing
    ? (typeof progress.eta_s === 'number' && progress.eta_s > 0
        ? progress.eta_s * 1000
        : (prog > 0.01 ? (elapsedMs * (1 - prog)) / prog : 0))
    : 0;
  // While actively scrubbing the timeline (or playing back) each frame is a
  // fresh server fetch, so request a lightweight downscaled frame — a full-res
  // HD/4K JPEG per frame makes dragging feel sluggish. Snap back to full
  // resolution the instant the user settles on a frame.
  const scrubbingNow = isScrubbing || isPlaying;
  const rawReqUrl = targets.length > 0
    ? `${API}/api/target/preview?index=${selTarget}&frame=${frame}${scrubbingNow ? '&width=960' : ''}`
    : '';
  // Load raw frames one at a time, latest wins (see useSequentialImage). Binding
  // an <img> directly to rawReqUrl issued a request per intermediate frame of a
  // drag, and since each one is a video seek on a single shared decoder, the
  // frame you stopped on arrived behind every frame you swept past.
  const loadedRawUrl = useSequentialImage(rawReqUrl);
  // Until the first frame of a new target has loaded there is nothing better to
  // show, so fall through to the request URL rather than blanking the box.
  const rawUrl = loadedRawUrl || rawReqUrl;

  // Does the raw frame currently DECODED belong to the view we are looking at?
  // Read the identity out of the URL rather than comparing it to rawReqUrl:
  // settling after a drag drops the &width=960 scrub parameter, so a straight
  // string compare calls the just-loaded frame stale and sends the stage back
  // to the frame before it.
  const rawIsCurrent = useMemo(() => {
    const m = /[?&]index=(\d+)&frame=(\d+)/.exec(loadedRawUrl || '');
    return !!m && Number(m[1]) === selTarget && Number(m[2]) === frame;
  }, [loadedRawUrl, selTarget, frame]);

  // What the stage's "after" layer shows, in strict preference order. The rule
  // this encodes is that the picture may only ever move FORWARD onto the frame
  // the playhead is on — never back onto one already left. Stepping used to
  // show the previous frame's swap (previewSrc lags by a whole render) and then
  // jump forward once the new one landed, which read as the image flicking
  // between two frames.
  const stageAfterSrc = (() => {
    if (isPlaying && bufferedSrc) return bufferedSrc;             // buffered player owns it
    // Mid-drag there is no render for most frames and none is coming; the
    // freshest decoded raw frame is what keeps a scrub feeling continuous.
    if (scrubbingNow) return getCachedPreview(selTarget, frame)?.image || rawUrl;
    if (previewSrc && previewFor === `${selTarget}_${frame}`) return previewSrc;
    const cached = getCachedPreview(selTarget, frame)?.image;
    if (cached) return cached;
    if (rawIsCurrent) return loadedRawUrl;   // right frame, not swapped yet
    // The new frame has not decoded yet either: hold whatever is already up
    // rather than blanking or bouncing. It is replaced the moment either of
    // the two branches above can answer.
    return previewSrc || rawUrl;
  })();

  // Keep a detached pop-out monitor in sync. It used to receive exactly one
  // frame — the one it was opened with — because nothing ever called
  // sendUpdate() again, so it sat frozen on that image while the main window
  // carried on. Only broadcasts while a pop-out is actually open: a rendered
  // preview is a multi-MB data URI.
  useEffect(() => {
    const src = previewSrc || rawUrl;
    if (src && popoutManager.isOpen()) {
      popoutManager.sendUpdate({ type: 'UPDATE_PREVIEW', src });
    }
  }, [previewSrc, rawUrl]);

  // Park the current still where the Processing tab can find it. That tab is
  // mounted only while this one is not, and its live-frame peek needs something
  // to show for the window before the pipeline publishes its first frame — see
  // faceswap/lastPreview.
  useEffect(() => {
    setLastPreview({ previewSrc, rawUrl, frame, maxFrames });
  }, [previewSrc, rawUrl, frame, maxFrames]);

  const revealOutput = async () => {
    try { await postJSON('/api/reveal', { path: out?.path }); }
    catch (e) { notify(e.message, 'error'); }
  };

  // Storyboard loading effect. The strip spans the VISIBLE range, so zooming in
  // re-samples it over the shorter span instead of leaving the same twelve
  // whole-clip stills stretched behind a narrow window. Debounced because a
  // wheel-zoom emits a burst of view updates and each strip is 12 backend
  // seeks; only the range you settle on gets fetched.
  //
  // Skipped outright while a run is in flight. Twelve stills is twelve video
  // seeks on the backend (a seek is a flat ~125-180 ms whatever the distance),
  // so re-drawing the strip costs the render up to two seconds of decode it has
  // to share the machine with — and Pinokio reloads this webview on every tab
  // switch, so simply looking at the UI mid-run re-paid that each time. The
  // strip is decoration and the filmstrip is not even on screen during a run;
  // `progress.processing` is in the deps so it refills once the run ends.
  useEffect(() => {
    if (targets.length === 0 || maxFrames <= 1 || progress.processing) {
      if (targets.length === 0 || maxFrames <= 1) setStoryboardThumbs([]);
      return;
    }
    let cancelled = false;
    const id = setTimeout(async () => {
      // 12 across the strip: enough that the filmstrip reads as the shape of
      // the clip rather than eight stretched stills, still only 12 seeks.
      const numThumbs = 12;
      const lo = Math.max(1, view.start);
      const hi = Math.max(lo, Math.min(view.end, maxFrames));
      const step = (hi - lo) / Math.max(1, numThumbs - 1);
      const frames = [];
      for (let i = 0; i < numThumbs; i++) {
        frames.push(Math.max(1, Math.min(maxFrames, Math.round(lo + i * step))));
      }
      // ONE request for all twelve. As twelve separate <img> URLs they competed
      // for the browser's six connections to this origin, so re-drawing the
      // filmstrip after a zoom could stall the request that actually matters —
      // the frame under the playhead — behind a strip nobody is looking at yet.
      // The server also decodes them in ascending order, which at high zoom
      // means walking a few frames rather than seeking twelve times.
      try {
        const res = await fetch(
          `${API}/api/target/preview_grid?index=${selTarget}&frames=${frames.join(',')}&width=200`);
        if (!res.ok) throw new Error('grid failed');
        const buf = new Uint8Array(await res.arrayBuffer());
        const urls = [];
        let off = 0;
        while (off + 4 <= buf.length) {
          const len = (buf[off] << 24 | buf[off + 1] << 16 | buf[off + 2] << 8 | buf[off + 3]) >>> 0;
          off += 4;
          if (off + len > buf.length) break;
          // A zero-length part is a frame the server could not read; keep its
          // slot so the remaining stills stay under the right timecodes.
          urls.push(len === 0 ? '' :
            URL.createObjectURL(new Blob([buf.slice(off, off + len)], { type: 'image/jpeg' })));
          off += len;
        }
        if (cancelled) { urls.forEach((u) => u && URL.revokeObjectURL(u)); return; }
        // Revoke the outgoing strip HERE, not inside the setState updater:
        // React 19 StrictMode runs updaters twice to check they are pure, and an
        // updater that frees resources is not.
        const outgoing = storyboardRef.current;
        storyboardRef.current = urls;
        setStoryboardThumbs(urls);
        // Safe even though the old <img>s are still on screen for a tick:
        // revoking a blob URL does not disturb an image that has already loaded
        // from it, and any that had not are being unmounted anyway.
        outgoing.forEach((u) => u && u.startsWith('blob:') && URL.revokeObjectURL(u));
      } catch {
        // The strip is decoration; a failed fetch leaves the previous one up.
      }
    }, 220);
    return () => { cancelled = true; clearTimeout(id); };
  }, [selTarget, maxFrames, targets.length, view.start, view.end, progress.processing]);

  // Release the last strip's blobs when the panel goes away. (storyboardRef is
  // written next to the setState above, so it is always the live list.)
  useEffect(() => () => {
    storyboardRef.current.forEach((u) => u && u.startsWith('blob:') && URL.revokeObjectURL(u));
  }, []);

  // ── Frame <-> track-position mapping, through the view window ─────────────
  // `pct` is 0..1 across the visible track, NOT across the clip. With the view
  // at its default (whole clip) these are exactly the old full-range formulas.
  const frameAtPct = (pct) => {
    const { start, end } = viewRef.current;
    const span = Math.max(0, end - start);
    return Math.max(1, Math.min(Math.round(start + pct * span), maxFrames));
  };
  const pctOfFrame = (f) => {
    const { start, end } = viewRef.current;
    return (f - start) / Math.max(1, end - start);
  };
  const viewSpan = () => Math.max(1, viewRef.current.end - viewRef.current.start);

  // A new clip (or a target swap) starts fully zoomed out.
  useEffect(() => {
    setView({ start: 1, end: Math.max(1, maxFrames) });
  }, [selTarget, maxFrames]);

  // Scrubbing event handler setup
  const updateTimelinePos = (targetFrame, type) => {
    if (type === 'start') {
      const end = targets[selTarget]?.end_frame ?? maxFrames;
      const val = Math.max(1, Math.min(targetFrame, end));
      setTargets((prev) => {
        const copy = [...prev];
        if (copy[selTarget]) copy[selTarget] = { ...copy[selTarget], start_frame: val };
        return copy;
      });
    } else if (type === 'end') {
      const start = targets[selTarget]?.start_frame ?? 1;
      const val = Math.max(start, Math.min(targetFrame, maxFrames));
      setTargets((prev) => {
        const copy = [...prev];
        if (copy[selTarget]) copy[selTarget] = { ...copy[selTarget], end_frame: val };
        return copy;
      });
    } else {
      setFrame(Math.max(1, Math.min(targetFrame, maxFrames)));
    }
  };

  const handleTimelinePointerMove = (e) => {
    if (!timelineRef.current) return;
    const clientX = e.clientX ?? e.touches?.[0]?.clientX;
    if (clientX === undefined) return;
    timelinePendingRef.current = clientX;
    if (timelineRafRef.current) return;
    timelineRafRef.current = requestAnimationFrame(() => {
      timelineRafRef.current = null;
      if (!timelineRef.current || timelinePendingRef.current === null) return;
      const rect = timelineRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(timelinePendingRef.current - rect.left, rect.width));
      const f = frameAtPct(x / rect.width);
      setHoverFrame((prev) => (prev === f ? prev : f));
    });
  };

  const handleTimelinePointerLeave = () => {
    if (timelineRafRef.current) { cancelAnimationFrame(timelineRafRef.current); timelineRafRef.current = null; }
    timelinePendingRef.current = null;
    setHoverFrame(null);
  };
  useEffect(() => () => { if (timelineRafRef.current) cancelAnimationFrame(timelineRafRef.current); }, []);

  /* eslint-disable react-hooks/exhaustive-deps -- intentional: scrub handlers bind latest markers via helpers, no re-subscribe wanted */
  useEffect(() => {
    if (!isScrubbing) return;

    // Coalesce scrub moves to one state update per animation frame so dragging
    // doesn't queue up more re-renders of this large component than the display
    // can paint.
    let rafId = null;
    let pendingX = null;
    const flush = () => {
      rafId = null;
      if (!timelineRef.current || pendingX === null) return;
      const rect = timelineRef.current.getBoundingClientRect();
      const pct = Math.max(0, Math.min((pendingX - rect.left) / rect.width, 1));
      let targetFrame = frameAtPct(pct);
      // Magnetic snap: while dragging the playhead, pull it onto the In/Out
      // points and the clip ends when it lands within ~10px, so lining the
      // playhead up with a marker doesn't require pixel-perfect aim. The
      // tolerance is ~10px of the VISIBLE span, so zooming in makes the snap
      // proportionally finer instead of swallowing the precision it bought.
      if (dragType === 'playhead' && maxFrames > 1) {
        const snapFrames = Math.max(1, Math.round((10 / rect.width) * viewSpan()));
        const snapPoints = [1, maxFrames, targets[selTarget]?.start_frame ?? 1, targets[selTarget]?.end_frame ?? maxFrames];
        let best = null, bestDist = Infinity;
        for (const sp of snapPoints) {
          const d = Math.abs(sp - targetFrame);
          if (d <= snapFrames && d < bestDist) { best = sp; bestDist = d; }
        }
        if (best !== null) targetFrame = best;
      }
      updateTimelinePos(targetFrame, dragType);
    };
    const handlePointerMove = (e) => {
      const clientX = e.clientX ?? e.touches?.[0]?.clientX;
      if (clientX === undefined) return;
      pendingX = clientX;
      if (rafId === null) rafId = requestAnimationFrame(flush);
    };

    const handlePointerUp = () => {
      if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
      setIsScrubbing(false);
      if (dragType === 'start') {
        const val = targets[selTarget]?.start_frame ?? 1;
        setFrameMarkerVal('start', val);
      } else if (dragType === 'end') {
        const val = targets[selTarget]?.end_frame ?? maxFrames;
        setFrameMarkerVal('end', val);
      }
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
    return () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, [isScrubbing, dragType, selTarget, maxFrames, targets]);
  /* eslint-enable react-hooks/exhaustive-deps */

  const handleTimelinePointerDown = (e) => {
    if (!timelineRef.current) return;
    const rect = timelineRef.current.getBoundingClientRect();
    const clientX = e.clientX ?? e.touches?.[0]?.clientX;
    if (clientX === undefined) return;
    const x = clientX - rect.left;
    const pct = Math.max(0, Math.min(x / rect.width, 1));
    const targetFrame = frameAtPct(pct);

    const sPct = pctOfFrame(targets[selTarget]?.start_frame ?? 1);
    const ePct = pctOfFrame(targets[selTarget]?.end_frame ?? maxFrames);
    const cPct = pctOfFrame(frame);

    const distStart = Math.abs(pct - sPct);
    const distEnd = Math.abs(pct - ePct);
    const distCurrent = Math.abs(pct - cPct);

    let dragTarget = 'playhead';
    const tolerance = 0.04;

    if (distStart < tolerance && distStart < distEnd && distStart < distCurrent) {
      dragTarget = 'start';
    } else if (distEnd < tolerance && distEnd < distStart && distEnd < distCurrent) {
      dragTarget = 'end';
    } else {
      dragTarget = 'playhead';
    }

    setDragType(dragTarget);
    setIsScrubbing(true);
    updateTimelinePos(targetFrame, dragTarget);
  };

  const renderPreviewClip = async () => {
    if (targets.length === 0) return;
    const fps = targets[selTarget]?.fps || 25;
    const origStart = targets[selTarget]?.start_frame || 1;
    const origEnd = targets[selTarget]?.end_frame || maxFrames;
    setOrigStartEnd({ start: origStart, end: origEnd });
    setIsGeneratingPreviewClip(true);

    const previewStart = frame;
    // 5 seconds preview = frame + 5 * fps
    const previewEnd = Math.min(maxFrames, frame + Math.round(5 * fps));

    try {
      // Set temporary start/end frames in backend
      await postJSON('/api/target/set_frame', { which: 'start', frame: previewStart });
      await postJSON('/api/target/set_frame', { which: 'end', frame: previewEnd });

      // Start the swap with current settings. Through the SAME builder the real
      // run and the queue use — this was a hand-copied second version of that
      // object, so a setting added to one reached a full render and a queued job
      // but silently missed the preview, which is the one place the difference
      // is hardest to notice.
      await postJSON('/api/settings', p);
      await postJSON('/api/swap', {
        ...buildSwapPayload(),
        target_index: selTarget,
        // Quick 5s preview clip: skip the heavy post-swap AI upscale so the
        // preview stays fast (the real render still upscales).
        upscale_after_swap: false,
      });

      setStartTime(Date.now());
      notify('Generating 5-second preview clip...');
    } catch (e) {
      notify(e.message, 'error');
      setIsGeneratingPreviewClip(false);
      setOrigStartEnd(null);
    }
  };

  // Restoration effect when swapping finishes
  /* eslint-disable react-hooks/exhaustive-deps -- intentional: fires on processing-complete transition; notify is stable */
  useEffect(() => {
    if (isGeneratingPreviewClip && !progress.processing && origStartEnd) {
      const restore = async () => {
        try {
          await postJSON('/api/target/set_frame', { which: 'start', frame: origStartEnd.start });
          await postJSON('/api/target/set_frame', { which: 'end', frame: origStartEnd.end });
          
          // Sync targets list
          const res = await getJSON('/api/state');
          if (res.targets) setTargets(res.targets);
        } catch (e) {
          console.error("Failed to restore timeline range markers:", e);
        } finally {
          setIsGeneratingPreviewClip(false);
          setOrigStartEnd(null);
          notify('5-second preview clip generated successfully!');
        }
      };
      restore();
    }
  }, [progress.processing, isGeneratingPreviewClip, origStartEnd]);
  /* eslint-enable react-hooks/exhaustive-deps */

  // Keyboard Escape, Shortcuts HUD, & Global Productivity Hotkeys
  /* eslint-disable react-hooks/exhaustive-deps -- intentional: hotkey handlers read latest callbacks; re-subscribing every render would thrash listeners */
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ignore key events if the user is typing in form controls
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;

      // Ignore while a modal dialog is open (confirm/prompt, command palette),
      // so playback/timeline hotkeys don't fire on the page behind it.
      if (document.querySelector('[role="dialog"]')) return;

      // Toggle shortcuts HUD: '?' or 'h'
      if (e.key === '?' || e.key === 'h' || e.key === 'H') {
        e.preventDefault();
        setShowShortcutHUD((prev) => !prev);
        return;
      }

      // Play/Pause: Space
      if (e.key === ' ') {
        e.preventDefault();
        if (maxFrames > 1 && setIsPlaying) {
          setIsPlaying((p) => !p);
        } else if (setCompare) {
          // Match the 'C' shortcut and the UI toggle: compare and the
          // enhancer grid are mutually exclusive.
          setCompare((c) => {
            const nextVal = !c;
            if (nextVal) { setComparingEnhancers(false); setComparingMasks(false); setComparingSwappers(false); setComparingUpscalers(false); }
            return nextVal;
          });
        }
        return;
      }

      // Step frames left/right: Left/Right Arrow
      if (e.key === 'ArrowLeft' && maxFrames > 1 && setFrame) {
        e.preventDefault();
        setFrame((f) => Math.max(1, f - (e.shiftKey ? 10 : 1)));
        return;
      }
      if (e.key === 'ArrowRight' && maxFrames > 1 && setFrame) {
        e.preventDefault();
        setFrame((f) => Math.min(maxFrames, f + (e.shiftKey ? 10 : 1)));
        return;
      }

      // Jump to start/end: Home / End
      if (e.key === 'Home' && maxFrames > 1 && setFrame) {
        e.preventDefault();
        setFrame(1);
        return;
      }
      if (e.key === 'End' && maxFrames > 1 && setFrame) {
        e.preventDefault();
        setFrame(maxFrames);
        return;
      }

      // Range Trim Markers: '[' to set start, ']' to set end, 'R' to reset range
      if (e.key === '[') {
        e.preventDefault();
        setFrameMarkerVal('start', frame);
        return;
      }
      if (e.key === ']') {
        e.preventDefault();
        setFrameMarkerVal('end', frame);
        return;
      }
      if (e.key.toLowerCase() === 'r' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        setFrameMarkerVal('start', 1);
        setFrameMarkerVal('end', maxFrames);
        return;
      }

      // Comparison modes: 'C' to compare, 'S' to toggle split view
      if (e.key.toLowerCase() === 'c') {
        e.preventDefault();
        setCompare((prev) => {
          const nextVal = !prev;
          if (nextVal) { setComparingEnhancers(false); setComparingMasks(false); setComparingSwappers(false); setComparingUpscalers(false); }
          return nextVal;
        });
        return;
      }
      if (e.key.toLowerCase() === 's') {
        e.preventDefault();
        setSplitView((prev) => !prev);
        return;
      }

      // Add to batch queue: 'Q'
      if (e.key.toLowerCase() === 'q') {
        e.preventDefault();
        addToQueue();
        return;
      }

      // Swapping execution: Ctrl + Enter
      if (e.key === 'Enter' && e.ctrlKey) {
        e.preventDefault();
        if (queue.running) return;
        if (queue.pending.length > 0) {
          startQueue();
        } else {
          start();
        }
        return;
      }
    };

    const handleEsc = (e) => {
      if (e.key === 'Escape') {
        setShowShortcutHUD(false);
        setPastedFiles(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keydown', handleEsc);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keydown', handleEsc);
    };
  }, [
    frame,
    maxFrames,
    isPlaying,
    compare,
    splitView,
    queue.running,
    queue.pending.length,
    setFrame,
    setIsPlaying,
    setCompare,
    setComparingEnhancers,
    setSplitView,
    setFrameMarkerVal,
    addToQueue,
    start,
    startQueue
  ]);
  /* eslint-enable react-hooks/exhaustive-deps */

  // Command-palette action bus (dispatched from App via window 'roop:command').
  // A ref keeps the handler map fresh without re-subscribing every render.
  const cmdRef = useRef({});
  cmdRef.current = {
    start: () => { if (queue.running) return; if (queue.pending.length > 0) startQueue(); else start(); },
    stop,
    queue: addToQueue,
    compare: () => setCompare((v) => { const n = !v; if (n) { setComparingEnhancers(false); setComparingMasks(false); setComparingSwappers(false); setComparingUpscalers(false); } return n; }),
    split: () => setSplitView((v) => !v),
    preview: () => refreshPreview({ force: true }),   // explicit user action — same as Refresh
    shortcuts: () => setShowShortcutHUD(true),
    // Applying a named preset is the one command that takes an argument, hence
    // the detail object being forwarded to the handler below.
    preset: (d) => { if (d?.name) loadProfile(d.name); },
  };
  useEffect(() => {
    const h = (e) => { const fn = cmdRef.current[e.detail?.id]; if (fn) fn(e.detail); };
    window.addEventListener('roop:command', h);
    return () => window.removeEventListener('roop:command', h);
  }, []);

  const startFrame = targets[selTarget]?.start_frame || 1;
  const endFrame = targets[selTarget]?.end_frame || maxFrames;

  // Pre-run estimate (idle only) — heuristic baseline refined by what the
  // backend has measured for these exact settings. See
  // faceswap/useRuntimeEstimate.
  const estFrames = maxFrames > 1 ? Math.max(1, endFrame - startFrame + 1) : (targets.length ? 1 : 0);
  const { calibEst, estPerFrame, estTotalMs, estLearned } = useRuntimeEstimate({
    settings: p,
    estFrames,
    faceCount: previewFaces.length,   // density hint from the current frame
    processing: progress.processing,
    hasTargets: targets.length > 0,
    threads: telemetry?.threads,
  });

  const heavyVram = (p.selected_enhancer && p.selected_enhancer !== 'None') &&
    (parseInt(p.face_detector_size || '640', 10) >= 960);

  // Clip advisor: sample the target, get recommended settings, apply them on
  // request. See faceswap/useClipAdvisor.
  const { advice, setAdvice, advisorBusy, runAdvisor, applyAdvice } = useClipAdvisor({
    targets, selTarget, settings: p, set, notify,
  });

  // Live camera (webcam → live swap → optional OBS virtual camera). The whole
  // session lives on the backend; see faceswap/useLiveCam.
  const {
    liveActive, liveBusy, liveTick,
    liveCamNum, setLiveCamNum,
    liveRes, setLiveRes,
    liveObs, setLiveObs,
    startLiveCam, stopLiveCam,
  } = useLiveCam({ notify });

  // Derived values for the estimation box.
  const estFps = targets[selTarget]?.fps || 0;
  const estDurationS = estFps ? estFrames / estFps : 0;
  const estSourceLabel = estLearned ? 'Learned'
    : (calibEst?.source === 'global' ? 'Global avg' : 'Heuristic');
  const estSourceClass = estLearned ? 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10'
    : (calibEst?.source === 'global' ? 'text-amber-400 border-amber-500/30 bg-amber-500/10'
      : 'text-white/45 border-white/10 bg-white/5');

  return (
    <div className="flex flex-col lg:flex-row gap-6 items-start w-full">

      {/* COLUMN 1: Settings & Controls — sticky sidebar on large viewports so it
          follows the scroll (and never leaves the lower-left area empty) while
          scrolling a taller workspace. Scrolls internally when taller than the
          viewport.

          This used to disappear for the duration of a run, along with the asset
          rail, the timeline and the preview controls — the whole tab became a
          progress panel. Watching a render and setting up the next job are now
          two different tabs, so this one keeps its layout at all times and the
          run has the Processing tab to itself. */}
      <div className={`w-full lg:w-[380px] 3xl:w-[440px] 4xl:w-[520px] shrink-0 pr-0 lg:pr-2 space-y-5 select-none ${!showLeftPanel ? 'hidden' : ''}`}>
        <Section title="Presets">
          <div className="flex flex-wrap gap-2">
            {Object.keys(PRESETS).map((name) => (
              <Button key={name} size="sm"
                variant={activePreset === name ? 'primary' : 'secondary'}
                onClick={() => applyPreset(name)}>
                {name}
              </Button>
            ))}
            <Button size="sm" variant="secondary" onClick={saveAsDefault}
              title="Save the current Face Swap tab settings as your default. 'Reset defaults' will restore to this.">
              Save as default
            </Button>
            <Button size="sm" variant="secondary" onClick={resetToDefaults}
              title={userDefaults
                ? 'Restore every Face Swap tab setting to your saved default'
                : 'Restore every Face Swap tab setting to the factory defaults'}>
              ↩️ {userDefaults ? 'Reset to my default' : 'Reset defaults'}
            </Button>
            {userDefaults && (
              <Button size="sm" variant="secondary" onClick={clearUserDefault}
                title="Forget your saved default and go back to the factory defaults">
                Clear my default
              </Button>
            )}
          </div>
          <div className="text-xs text-[var(--text-muted)] mt-2">
            Sets detection resolution, upscale, enhancer & swap steps. Other settings unchanged.
            {userDefaults
              ? ' “Reset” restores all Face Swap tab settings to the default you saved.'
              : ' “Save as default” stores the current Face Swap tab settings so “Reset” restores to them.'}
          </div>
        </Section>

        <Section title="Clip advisor">
          <button type="button" disabled={advisorBusy || !targets.length || progress.processing} onClick={runAdvisor}
            title="Samples the selected target (face sizes, count, detection coverage, motion, lighting) and recommends settings tuned to it. Nothing changes until you apply."
            className="w-full py-2 rounded-lg text-note font-bold bg-[var(--accent)]/10 border border-[var(--accent)]/30 text-[var(--accent)] hover:bg-[var(--accent)]/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2">
            {advisorBusy
              ? (<><span className="h-3 w-3 rounded-full border-2 border-[var(--accent)]/40 border-t-[var(--accent)] animate-spin" /> Analyzing target…</>)
              : 'Analyze target & recommend settings'}
          </button>
          {advice && (
            <div className="space-y-2 mt-2">
              <div className="text-micro text-white/45 leading-relaxed">
                {advice.stats.sampled_frames} frame{advice.stats.sampled_frames === 1 ? '' : 's'} sampled ·
                faces found on {advice.stats.detection_coverage}% ·
                face size {advice.stats.min_face_size_pct}–{advice.stats.max_face_size_pct}% ·
                brightness {advice.stats.brightness}
                {advice.is_video ? ` · motion ${advice.stats.motion}` : ''}
              </div>
              {advice.message && <div className="text-mini text-amber-300/80">{advice.message}</div>}
              {advice.recommendations.length === 0 && !advice.message ? (
                <div className="text-mini font-bold text-emerald-400">✓ Current settings already fit this clip</div>
              ) : advice.recommendations.length > 0 && (
                <>
                  {advice.recommendations.map((r) => (
                    <div key={r.key} className="rounded-lg bg-black/25 border border-white/5 px-2.5 py-1.5">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-mini font-bold text-white/85">{ADVISOR_LABELS[r.key] || r.key}</span>
                        <span className="text-mini font-mono shrink-0">
                          <span className="text-white/35">{fmtAdviceVal(p[r.key] ?? '—')}</span>
                          <span className="text-white/30"> → </span>
                          <span className="text-[var(--accent)] font-bold">{fmtAdviceVal(r.value)}</span>
                        </span>
                      </div>
                      <div className="text-micro text-white/45 leading-snug mt-0.5">{r.reason}</div>
                    </div>
                  ))}
                  <div className="flex gap-2">
                    <button type="button" onClick={applyAdvice}
                      className="flex-1 py-1.5 rounded-lg text-mini font-bold bg-[var(--accent)] text-white hover:opacity-90 transition-opacity">
                      ✓ Apply all {advice.recommendations.length}
                    </button>
                    <button type="button" onClick={() => setAdvice(null)}
                      className="px-3 py-1.5 rounded-lg text-mini font-bold bg-white/[0.04] border border-white/10 text-white/60 hover:text-white/90 transition-colors">
                      Dismiss
                    </button>
                  </div>
                </>
              )}
            </div>
          )}
        </Section>

        <Section title="Live camera" collapsible defaultOpen={false}>
          <div className="text-micro text-white/45 leading-relaxed -mt-1">
            Swap your webcam feed live using the loaded source face{liveObs ? ',' : ''} — optionally
            published as a system <b>virtual camera</b> for OBS / video calls.
          </div>
          <div className="flex gap-2">
            <TextInput label="Camera #" type="number" value={liveCamNum}
              onChange={(v) => setLiveCamNum(Math.max(0, parseInt(v, 10) || 0))} />
            <Select label="Resolution" value={liveRes} onChange={setLiveRes}
              options={['640x480', '1280x720', '1920x1080']} />
          </div>
          <Toggle label="Stream to virtual camera (OBS)" info="Publishes the swapped feed as a system camera device via pyvirtualcam — pick 'OBS Virtual Camera' in any app." checked={liveObs} onChange={setLiveObs} />
          {!liveActive ? (
            <button type="button" disabled={liveBusy || progress.processing} onClick={startLiveCam}
              className="w-full py-2 rounded-lg text-note font-bold bg-[var(--accent)]/10 border border-[var(--accent)]/30 text-[var(--accent)] hover:bg-[var(--accent)]/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2">
              {liveBusy
                ? (<><span className="h-3 w-3 rounded-full border-2 border-[var(--accent)]/40 border-t-[var(--accent)] animate-spin" /> Opening camera…</>)
                : 'Start live camera'}
            </button>
          ) : (
            <>
              <div className="relative rounded-xl overflow-hidden bg-black/50 border border-white/10 aspect-video">
                <img src={`${API}/api/livecam/frame?t=${liveTick}`} alt="Live camera"
                  className="w-full h-full object-contain" draggable={false} />
                <span className="absolute top-2 left-2 inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-black/70 text-micro font-bold text-white/90 border border-white/10">
                  <span className="h-1.5 w-1.5 rounded-full bg-red-500 animate-pulse" /> LIVE
                </span>
              </div>
              <button type="button" disabled={liveBusy} onClick={stopLiveCam}
                className="w-full py-2 rounded-lg text-note font-bold bg-white/[0.04] border border-white/10 text-white/70 hover:text-white hover:border-white/25 transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
                ⏹ Stop live camera
              </button>
            </>
          )}
          {sourceFaces.length === 0 && (
            <div className="text-micro text-amber-300/70">No source face loaded — the feed will pass through unswapped.</div>
          )}
        </Section>

        <div className="space-y-5">
          <Section title="Swap settings">
          <Select label="Swap model" info="realswap (default) pairs hyperswap and hififace and sends each face to ONE of them — it never blends the two, so nothing can double. Measured over a yaw sweep: below profile it is byte-identical to hyperswap, and at ±90° it cuts the frames where the swap lands the face somewhere the original was not from 80 to 35. Only one net runs per face, so it costs the same as hyperswap plus a pose solve. CAVEAT, worth knowing before you trust it: the pose test that decides the route was calibrated on studio stills and over-fires on real video — on a normal two-person clip 69% of faces took the hififace route, which carries noticeably less of the faceset's identity. If a render looks less like your source than hyperswap did, that is why; switch to hyperswap and compare. · inswapper 128 · reswapper/hyperswap(a/b/c)/ghost(1-3)/simswap/blendswap/uniface/hififace 256 · simswap_512 (each downloads on first use; ghost/simswap/hififace use their own alignment + identity converter). COST PER FACE, measured on an RTX 4070 TensorRT FP16 at the 256px subsample — note this is per FACE, not per inference, because a model smaller than the subsample size is tiled and run (subsample/size)² times: ghost_1 3.7ms, hyperswap a/c 5.3ms, hyperswap_1b 5.6ms, hififace 6.1ms, simswap 6.7ms, ghost_3 11.4ms, simswap_512 11.6ms, blendswap 13.2ms, reswapper 16.4ms, and inswapper 21.1ms — inswapper is 5.3ms per call but 128px, so at 256px subsample it runs FOUR times per face and is the most expensive option here despite looking near-cheapest. The whole spread is ~17ms against a swap stage reporting ~46ms per face, so unless you are on inswapper this is a quality choice, not a speed one." value={p.swap_model} onChange={(v) => set('swap_model', v)} options={meta.swap_models} />
          <Select label="Face selection" value={p.face_detection_mode} onChange={(v) => set('face_detection_mode', v)} options={meta.face_detection_modes} />
          <Select
            label="Detector engine"
            info="RetinaFace R50 is the safest pick: it was the only engine that held full recall as a face grew to fill the frame, where SCRFD (the stock default) and RetinaFace 10g both dropped to zero. Prefer it unless you need the speed. YOLOFace is often better on steep profiles and partially occluded faces. YuNet is the lightest. SCRFD is fast and accurate on frontal faces but the weakest on close-ups. All engines reuse the same identity/landmark models; alternates download a small model on first use."
            value={p.detector_engine || 'scrfd'}
            onChange={(v) => set('detector_engine', v)}
            options={meta.detector_engines || ['scrfd', 'yoloface', 'retinaface', 'retinaface_r50', 'yunet']}
          />
          <Select
            label="Face detection resolution"
            info="Higher resolution improves detection of small/distant faces, but runs slower. Measured on an RTX 4070 with retinaface_r50 over 240 frames of angled footage, production module set: 640px = 14.27ms per frame, 512px = 10.95ms — 1.30x faster AND slightly MORE accurate (99.4% recall against 98.5%), with landmarks shifting only 0.24-0.72px. 512 is the better default on 720p/1080p sources; the extra 640px pixels are mostly the black padding a 16:9 frame leaves in a square canvas. 320 starts losing hard poses. NOTE: only retinaface honours this — yoloface and scrfd are fixed 640x640 exports and ignore it."
            value={p.face_detector_size || '640'}
            onChange={(v) => {
              set('face_detector_size', v);
              set('default_det_size', v === '512' || v === '640' || v === '960' || v === '1280');
            }} 
            options={['320', '512', '640', '960', '1280']} 
          />
          <Slider
            label="Face detection threshold"
            info="Lower values (e.g. 0.40) detect angled, profile, or hard-to-see faces. Higher values avoid false detections. Default 0.50"
            min={0.10}
            max={0.90}
            step={0.05}
            value={num(p.face_detector_threshold, 0.50)}
            onChange={(v) => set('face_detector_threshold', v)}
          />
          <Slider
            label="Overlap NMS threshold"
            info="Lower values (e.g. 0.30) suppress close duplicates. Higher values (e.g. 0.60) allow detecting overlapping faces close to each other. Default 0.40"
            min={0.10}
            max={0.90}
            step={0.05}
            value={num(p.face_detector_nms, 0.40)}
            onChange={(v) => set('face_detector_nms', v)}
          />
          <Toggle label="Refine alignment (68-pt)" info="Derives the alignment keypoints from the 68-point landmark model instead of the detector's raw 5 points — more stable alignment on angled faces, less residual swap wobble. Small per-face cost." checked={!!p.refine_landmarks} onChange={(v) => set('refine_landmarks', v)} />
          <Slider label="🧩 Swap model's own face mask" info="Only hififace and hyperswap emit this — every other swapper ignores the setting, so it is safe to leave on. Those two nets output a face mask alongside the swapped image, saying where they actually synthesised a face, and the pipeline used to throw it away. That matters because the paste mask cannot know: it is an ellipse intersected with the convex hull of the 106 landmarks, whose forehead extension runs 60% above the eyebrows and therefore into the HAIR. Measured against hififace's own verdict, 15–27% of the paste mask is territory the model says is not face on a FRONTAL head, and 31% on a profile — and looking at where that excess lands, it is the band above the hairline frontally, and a wedge of hair plus background behind the head on a profile. That is very likely why eyes/nose/mouth read as distorted at angles: not the features themselves, but the invalid territory around them being composited. Not pose-gated, unlike the two angle controls above, because the net derives this from the actual image and its verdict is right at every angle. 100 = trust the model fully, 0 = off (previous behaviour), which is the default — this is new and has not been judged on real footage yet. It is the first thing to raise if a hififace or hyperswap paste reaches up into hair." min={0} max={100} step={5} value={num(p.swap_model_mask_strength, 0)} onChange={(v) => set('swap_model_mask_strength', v)} />
          <Toggle label="Rescue small faces" info="When a frame has no detected face, retries on a 2x upscale to catch tiny/distant faces — without raising the global detection resolution for every frame." checked={!!p.rescue_small_faces} onChange={(v) => set('rescue_small_faces', v)} />
          <Slider label="Swapping steps" info="more = more likeness" min={1} max={5} step={1} value={num(p.num_swap_steps, 1)} onChange={(v) => set('num_swap_steps', v)} />
          <Select label="Post-processing enhancer" info="The single most expensive stage in a render — about 36% of total time — so this choice costs more than any other on this panel. Measured on an RTX 4070, TensorRT FP16, one 512² face, isolated, engines warm: GPEN-256 3.0ms, GPEN-512 19.5ms, RestoreFormer++ 22.7ms, CodeFormer 23.9ms. GFPGAN v1.4 was listed here at 11.8ms; that figure was taken on a TensorRT FP16 engine that COLLAPSED — it returned a flat grey face (pixel std 16 against FP32's 65) and was fast because it was not doing the work. It now runs forced-FP32 at 41.7ms, the most expensive restorer here, not the cheapest. CodeFormer and RestoreFormer++ look tied but are not: interleaved A/B over 8 rounds put RestoreFormer++ ahead 8/8 by 5.4%, with under 1% spread within each. Because the swap phase runs the GPU at ~98%, those ratios carry straight through to wall clock against an enhance stage that is ~36% of a render: GPEN-512 is worth ~5% off the whole render and CodeFormer COSTS you ~2%. The old GFPGAN-is-half-the-cost-of-RestoreFormer++ claim is withdrawn — see above. That is a quality trade, not a free win — GFPGAN is the older restorer and tends to look smoother and less detailed, while RestoreFormer++ and CodeFormer hold more texture. Cost is measured; which one looks better on your footage is yours to judge. GPEN 1024/2048 are heavier still and not in this comparison. GPEN 256 goes the other way — a quarter of 512's pixels through the network, so it is the cheapest restorer here; it is meant for distant or small faces, comparison grids and preview scrubbing, where 512's extra detail is thrown away by the paste-back downscale anyway. Its output is resized back to the crop size, so it is a drop-in for any of the others." value={p.selected_enhancer} onChange={(v) => set('selected_enhancer', v)} options={meta.enhancers} />
          <Slider label="Max face distance" info="How far from your captured target face a detected face may sit and still be swapped. This is a DISTANCE, not a similarity — 0 means identical and HIGHER IS MORE PERMISSIVE, so raising it makes look-alikes and bystanders start getting swapped. Scale is scipy cosine distance (0–2). Measured on a hard clip: the SAME person stays under ~0.66 even on bad frames, while DIFFERENT people sit at ~0.93–1.07. The 0.75 default sits mid-gap. Lower it toward 0.66 if the swap jumps to the wrong person mid-shot; raise it toward 0.85 if the swap blinks off on profiles and motion blur. Run with ROOP_DEBUG_MATCH=1 to print the real per-frame distances." min={0.01} max={1} step={0.01} value={num(p.max_face_distance, 0.75)} onChange={(v) => set('max_face_distance', v)} />
          <Select label="Subsample upscale" value={p.subsample_upscale} onChange={(v) => set('subsample_upscale', v)} options={meta.upscale} />
          <Toggle label="AI upscale (after swap)" info="Runs an AI upscaler as the final step of the swap pass — each frame is swapped & enhanced first, then upscaled, producing a single output file (no second pass). Full-frame upscaling is heavy: ×4 on video is slow and VRAM-hungry." checked={!!p.upscale_after_swap} onChange={(v) => set('upscale_after_swap', v)} />
          {p.upscale_after_swap && (
            <Select label="AI upscale model" value={p.upscale_model_after} onChange={(v) => set('upscale_model_after', v)} options={AI_UPSCALE_MODELS} />
          )}
          <Select label="Frame interpolation (after swap)" info="Raises the output frame rate with motion-interpolated in-between frames as the final pass (after any upscale). RIFE = AI motion interpolation (recommended, fast); minterpolate = classical ffmpeg motion estimation (no model, much slower). Duration is unchanged — frame count and fps are multiplied together, audio untouched." value={p.interp_after_swap || 'off'} onChange={(v) => set('interp_after_swap', v)}
            options={[{ value: 'off', label: 'Off' }, { value: 'rife_2x', label: 'RIFE ×2 fps' }, { value: 'rife_4x', label: 'RIFE ×4 fps' }, { value: 'minterpolate_2x', label: 'ffmpeg minterpolate ×2' }]} />
          <Select label="Color/lighting match" info="Matches the swapped face's skin tone & lighting to the original scene. RCT = per-channel (fast, default). LCT = corrects hue casts. MKL = fullest match. IDT = matches the full non-Gaussian colour distribution, which the other three cannot: under mixed lighting (warm key, cool fill) skin is bimodal and no single linear map lands it. IDT is the quality ceiling here and also much the dearest: measured ~65 ms per 512² face against ~11 ms for RCT and ~17 ms for LCT/MKL, because it makes several passes over every pixel instead of one matrix multiply. Reach for it on hard lighting or stills, not by default. None = off." value={p.color_transfer_mode || 'rct'} onChange={(v) => set('color_transfer_mode', v)} options={meta.color_transfer_modes || ['none', 'rct', 'lct', 'mkl', 'idt']} />
          <Toggle label="Target-conditioned lighting" info="SOURCE supplies identity while TARGET controls exposure, white balance, scene color cast, spatial shadows, highlights and local contrast. It uses a low-frequency target field, never a wholesale texture paste. DARK/VERY DARK scenes reduce restoration and sharpening; VERY DARK avoids aggressive exposure correction. Temporal EMA prevents warm/neutral/blue flicker. Off preserves the legacy color-transfer path." checked={!!p.target_conditioned_appearance} onChange={(v) => set('target_conditioned_appearance', v)} />
          {p.target_conditioned_appearance && <>
            <Slider label="Appearance conditioning strength" min={0} max={1} step={0.05} value={num(p.target_conditioned_appearance_strength, 0.75)} onChange={(v) => set('target_conditioned_appearance_strength', v)} />
            <Slider label="Appearance temporal stability" min={0.05} max={1} step={0.05} value={num(p.target_conditioned_appearance_temporal_alpha, 0.30)} onChange={(v) => set('target_conditioned_appearance_temporal_alpha', v)} />
          </>}
          <Slider label="Original/Enhanced blend" min={0} max={1} step={0.01} value={num(p.blend_ratio, 0.8)} onChange={(v) => set('blend_ratio', v)} />
          <Slider label="Skin detail transfer" info="Adds the ORIGINAL footage's real high-frequency texture (pores, stubble, grain) onto the swapped face. The generator smooths skin and the enhancer fakes flickery pores; this uses genuine detail from the scene instead. 0 = off. Start ~0.3–0.5; too high reintroduces the target's skin identity." min={0} max={1} step={0.05} value={num(p.detail_transfer_strength, 0)} onChange={(v) => set('detail_transfer_strength', v)} />
          <Slider label="Expression restore" info="Puts the TARGET's own expression back onto the swapped face using LivePortrait. Swappers pull faces toward the average expression of their training data, so laughing, crying and grimacing come out flattened — this reads the expression off the original frame and re-applies it. 0 = off (bit-exact no-op). Try ~0.8–1.0; above 1 exaggerates past the original, which helps when the swap compressed an expression rather than removed it. Only the expression moves — head pose cannot drift by construction. Downloads ~537MB on first use and measured ~0.33s per face on TensorRT, so it roughly doubles a slow render; needs TensorRT, since onnxruntime's CUDA GridSample cannot run this model and the CPU fallback is ~1.9s per face." min={0} max={2} step={0.05} value={num(p.expression_restore_strength, 0)} onChange={(v) => set('expression_restore_strength', v)} />
          {num(p.expression_restore_strength, 0) > 0 && (
            <Select label="Expression region" info="Which part of the face the restored expression is applied to. 'lips' is the usual choice for speech and laughing; 'eyes' for blinks and squinting; 'all' transfers everything including brow and jaw." value={p.expression_restore_region || 'all'} onChange={(v) => set('expression_restore_region', v)} options={['all', 'lips', 'eyes']} />
          )}
        </Section>

        {/* DeepFaceLab's merger knobs. Grouped and collapsed by default: they
            are all neutral at 0 and most renders never need them, but when a
            swap looks "pasted" rather than wrong, the fix is usually in here
            rather than in a bigger model. All five run on the CPU over a crop
            already in memory, so they do not compete with the GPU stages. */}
        <Section title="Merger post-processing" collapsible defaultOpen={false}>
          <Slider label="Face size" info="Grows or shrinks the swapped face about its own centre. Every identity swapper keeps the TARGET's head size and bone structure, so when the source person's face is visibly narrower or broader there is otherwise no lever for it — the swap comes out the right identity at the wrong size. 0 = off. This is free: it changes the paste matrix rather than adding a pass, so it costs 0 ms. Note that 'Restore original mouth' re-composites using the target's untouched geometry, so large values here will not register with it." min={-0.2} max={0.2} step={0.01} value={num(p.output_face_scale, 0)} onChange={(v) => set('output_face_scale', v)} />
          <Slider label="Grain match" info="Adds noise matched to the original footage's own measured noise floor. This fixes the most common 'it looks pasted' failure and it is the opposite of what people expect: the face is too CLEAN. The generator emits smooth skin, so on grainy, high-ISO or heavily compressed footage the swap reads as fake precisely because nothing else in the frame is that clean. The level is measured from the plate (median absolute high-pass, so pores and hair do not inflate it), not guessed. 0 = off; try 0.5–1.0 on noisy source. Measured ~6.8 ms per 512² face on CPU." min={0} max={1} step={0.05} value={num(p.merger_grain_match, 0)} onChange={(v) => set('merger_grain_match', v)} />
          <Slider label="Sharpen / soften" info="Signed unsharp mask on the finished face. Positive sharpens; NEGATIVE softens, which is the denoise direction and useful when the enhancer has left crunchy over-detailed skin. 0 = off. Radius scales with crop size, so the effect looks the same at 256/512/1024. The cheapest control here at ~0.6 ms per face." min={-1} max={1} step={0.05} value={num(p.merger_sharpen, 0)} onChange={(v) => set('merger_sharpen', v)} />
          <Slider label="Motion blur" info="Blurs the face along the direction the camera actually smeared it. On fast movement a pin-sharp face on a motion-blurred head is a loud tell. You set the amount; the DIRECTION is measured off the original crop rather than asked for — motion blur destroys detail along the axis of travel, so the plate's own gradient structure points the kernel. 0 = off. Measured ~7.5 ms per 512² face — the dearest of these five." min={0} max={1} step={0.05} value={num(p.merger_motion_blur, 0)} onChange={(v) => set('merger_motion_blur', v)} />
          <Slider label="Histogram match" info="Matches the face's per-channel histogram to the original crop. A fourth colour family alongside the Color/lighting match above: RCT/LCT/MKL match statistical moments (mean, covariance), this matches the shape of the whole distribution — which is what catches a face that is subtly flatter or more contrasted than the plate rather than differently tinted. Works alongside the setting above, not instead of it. 0 = off. Measured ~5.0 ms per 512² face." min={0} max={1} step={0.05} value={num(p.merger_hist_match, 0)} onChange={(v) => set('merger_hist_match', v)} />
          <Slider label="Degrade" info="Bicubic down-and-up on the finished face, to match a soft or heavily compressed plate. A crisp 512px swap dropped into 480p-grade or bitrate-starved footage is sharper than anything around it. 0 = off; at 1 the crop makes a round trip through 40% scale. Measured ~0.8 ms per face." min={0} max={1} step={0.05} value={num(p.merger_degrade, 0)} onChange={(v) => set('merger_degrade', v)} />
        </Section>

        <Section title="Masking parameters" collapsible defaultOpen={false}>
          <Select label="Masking engine" info="Choose this one on quality, not speed — the models are all cheap and within a few ms of each other. Measured on an RTX 4070, TensorRT FP16, isolated: Face Parser 2.4ms, DFL XSeg 2.9ms, XSeg-3 4.7ms, Face Occluder 5.0ms, MobileSAM 5.8ms+decoder, FastSAM 5.8ms. The whole spread is ~3ms against a masking stage that costs ~42ms per face, because almost all of that stage is the CPU work around the model — the landmark hull, the mouth mask, the blurs and the non-frontal unwarp — not the model itself. So switching engines to go faster will not work; switch when one of them handles YOUR occlusions better. SAM2 (tracked) is not comparable here: it runs a whole-clip pre-pass instead of per-crop inference." value={p.mask_engine} onChange={(v) => set('mask_engine', v)} options={meta.mask_engines} />
          <Select label="Second masking engine (occlusion)" info="A second, independent occlusion engine, run after the first. They compose as a union of &quot;not face&quot;, so this can only ever restore MORE of the original footage — it cannot eat into the swap, and turning it off is exactly the previous behaviour. Why you would: when something comes in front of a face and the swap gets painted over it anyway, the usual reason is not that masking is off, it is that the one engine you picked was not trained on that particular object. The engines miss different things — XSeg and XSeg-3 come from DeepFaceLab&apos;s hand/object data, Face Occluder from a different set again, Face Parser segments facial regions rather than occluders, and the SAM variants segment anything at all. Running two costs one more model call per face, which is 2–6ms against a masking stage of ~42ms. Start with DFL XSeg + Face Occluder. &quot;None&quot; = one engine, as before." value={p.mask_engine_2 || 'None'} onChange={(v) => set('mask_engine_2', v)} options={meta.mask_engines} />
          {String(p.mask_engine || '').startsWith('Face Parser') && (
            <ParserRegions
              regions={p.parser_regions}
              grow={p.parser_region_grow}
              onChange={({ regions, grow }) => setMany({ parser_regions: regions, parser_region_grow: grow })}
            />
          )}
          {p.mask_engine === 'Clip2Seg' && (
            <TextInput label="Objects to mask & restore" value={p.mask_clip_text} onChange={(v) => set('mask_clip_text', v)} placeholder="cup,hands,hair" />
          )}
          {p.mask_engine === 'Segment Anything 2 (tracked)' && (
            <Select label="SAM2 checkpoint (speed ↔ quality)" value={p.sam2_model_size || 'tiny'} onChange={(v) => set('sam2_model_size', v)} options={meta.sam2_model_sizes || ['tiny', 'small', 'base_plus', 'large']} />
          )}
          <Toggle label="Show mask overlay in preview" checked={!!p.show_mask_offsets} onChange={(v) => set('show_mask_offsets', v)} />
          <Slider label="Offset face top" min={0} max={2} step={0.01} value={num(p.mask_top, 0)} onChange={(v) => set('mask_top', v)} />
          <Slider label="Offset face bottom" min={0} max={2} step={0.01} value={num(p.mask_bottom, 0)} onChange={(v) => set('mask_bottom', v)} />
          <Slider label="Offset face left" min={0} max={2} step={0.01} value={num(p.mask_left, 0)} onChange={(v) => set('mask_left', v)} />
          <Slider label="Offset face right" min={0} max={2} step={0.01} value={num(p.mask_right, 0)} onChange={(v) => set('mask_right', v)} />
          <Slider label="Face mask edge blend" min={0} max={200} step={1} value={num(p.face_mask_blend, 20)} onChange={(v) => set('face_mask_blend', v)} />
        </Section>

        <Section title="Mouth & Angle math" collapsible defaultOpen={false}>
          <Slider label="Mouth mask top" min={0} max={2} step={0.01} value={num(p.mouth_top_scale, 1)} onChange={(v) => set('mouth_top_scale', v)} />
          <Slider label="Mouth mask bottom" min={0} max={2} step={0.01} value={num(p.mouth_bottom_scale, 1)} onChange={(v) => set('mouth_bottom_scale', v)} />
          <Slider label="Mouth mask left" min={0} max={2} step={0.01} value={num(p.mouth_left_scale, 1)} onChange={(v) => set('mouth_left_scale', v)} />
          <Slider label="Mouth mask right" min={0} max={2} step={0.01} value={num(p.mouth_right_scale, 1)} onChange={(v) => set('mouth_right_scale', v)} />
          <Slider label="Mouth mask edge blend" min={0} max={200} step={1} value={num(p.mouth_mask_blend, 10)} onChange={(v) => set('mouth_mask_blend', v)} />
          <Toggle label="3D source pose matching" info="Only affects image-source swappers (BlendSwap / UniFace) — feeds them a pose-matched source crop. Has NO effect on inswapper/ghost/hyperswap/simswap (their identity vector is pose-invariant, so warping the source would only degrade it)." checked={!!p.use_3d_recon} onChange={(v) => set('use_3d_recon', v)} />
          <Toggle label="Multi-angle source bank" info="auto-pick best source per frame" checked={!!p.use_source_bank} onChange={(v) => set('use_source_bank', v)} />
          <Toggle label="↔️ Frontalize angled faces" info="Un-rotates steep profile/side (lateral) faces before swapping so they don't come out distorted/'alien', then restores the original angle." checked={!!p.use_frontalization} onChange={(v) => set('use_frontalization', v)} />
          {p.use_frontalization && (
            <Slider label="Frontalize above angle (°)" info="Frontalization kicks in when the face yaw/pitch exceeds this. Lower = frontalize more; higher = only the steepest." min={10} max={60} step={5} value={num(p.frontalization_threshold, 30)} onChange={(v) => set('frontalization_threshold', v)} />
          )}
          <Toggle label="Reshape jaw/chin to source" info="Identity swappers (inswapper/hyperswap/reswapper…) keep the TARGET's jaw & chin bone structure. This warps the swapped face's lower-silhouette toward your SOURCE person's jaw/chin shape after the swap (a smooth liquify — no re-swap, so swap quality is untouched). Best for moderate shape differences; very large changes can distort the neck/background near the jaw. Landmark jitter is smoothed when Temporal detection is on." checked={!!p.jaw_reshape} onChange={(v) => set('jaw_reshape', v)} />
          {p.jaw_reshape && (
            <Slider label="Jaw reshape strength" info="0 = off (target's jaw), 1 = full source jaw/chin shape. Start around 0.4–0.6 and back off if the chin looks distorted." min={0} max={1} step={0.05} value={num(p.jaw_reshape_strength, 0.5)} onChange={(v) => set('jaw_reshape_strength', v)} />
          )}
        </Section>

        <Section title="Video parameters">
          <Select label="Video method" value={p.video_swapping_method} onChange={(v) => set('video_swapping_method', v)} options={meta.video_methods} />
          <Select label="On no face detected" value={p.no_face_action} onChange={(v) => set('no_face_action', v)} options={meta.no_face_actions} />
          <Toggle label="Temporal detection (anti-flicker)" info="Video (In-Memory method): one tracked detection pre-pass over the clip. Short detection misses (≤10 frames) are gap-filled by interpolating the face's position, so the swap can't blink out; with 'Stabilize face' also on, keypoints AND mask/mouth landmarks are smoothed per person. The swap pass then skips per-frame detection and stays multi-threaded. Includes identity locking when that toggle is on." checked={!!p.temporal_detection} onChange={(v) => set('temporal_detection', v)} />
          <Toggle label="VR mode" checked={!!p.vr_mode} onChange={(v) => set('vr_mode', v)} />
          <Toggle label="Reduce enhancer flicker" info="Temporally blends the enhanced face. Runs multi-threaded (work-stealing) when the launcher's ROOP_STAB_PARALLEL is on (the Pinokio default) — otherwise it forces single-thread. Either way it costs some extra compute (blending + per-block warm-up), so it's somewhat slower, not free." checked={!!p.stabilize_enhancer} onChange={(v) => set('stabilize_enhancer', v)} />
          {p.stabilize_enhancer && (
            <Slider label="Flicker reduction strength" info="higher = smoother" min={0} max={1} step={0.05} value={num(p.stabilize_enhancer_strength, 0.5)} onChange={(v) => set('stabilize_enhancer_strength', v)} />
          )}
          <Toggle label="Reduce mask flicker" info="Temporally blends the mask edge (the boundary between swapped and original pixels). XSeg/Face Parser/RealityUX all recompute this edge from scratch every frame with no memory of the last one, so on a close-up with hair moving across an eye it can jitter — most visible as the swapped face's own features flickering in and out right at the edge. Runs on the same multi-threaded chunked pipeline as the other stabilizers; unlike enhancer flicker reduction it does not back off during head motion, since the mask lives in the already motion-compensated aligned crop." checked={!!p.stabilize_mask} onChange={(v) => set('stabilize_mask', v)} />
          {p.stabilize_mask && (
            <Slider label="Mask flicker reduction strength" info="higher = smoother" min={0} max={1} step={0.05} value={num(p.stabilize_mask_strength, 0.5)} onChange={(v) => set('stabilize_mask_strength', v)} />
          )}
        </Section>

        <div>
          <Section title="System options" collapsible defaultOpen={false}>
            <Toggle label="Auto rotate horizontal faces" checked={!!p.autorotate_faces} onChange={(v) => set('autorotate_faces', v)} />
            <Toggle label="Skip audio" checked={!!p.skip_audio} onChange={(v) => set('skip_audio', v)} />
            <Toggle label="Keep frames (when extracting)" checked={!!p.keep_frames} onChange={(v) => set('keep_frames', v)} />
            <Toggle label="Wait before creating video" checked={!!p.wait_after_extraction} onChange={(v) => set('wait_after_extraction', v)} />
          </Section>
        </div>

        <div>
          <Section title="Saved Profiles" collapsible defaultOpen={false}>
            <div className="flex flex-wrap gap-2 items-end">
              <div className="flex-1 min-w-[120px]">
                <TextInput label="Save active settings as:" value={newProfileName} onChange={setNewProfileName} placeholder="My Preset Name" />
              </div>
              <Button size="sm" onClick={saveProfile}>Save</Button>
            </div>
            {profiles.length > 0 && (
              <div className="space-y-2 mt-3">
                <span className="text-xs font-semibold text-white/50">Apply custom preset:</span>
                <div className="flex flex-wrap gap-2">
                  {profiles.map((pr) => (
                    <div key={pr.name} className="flex items-center gap-1 bg-white/5 hover:bg-white/10 border border-white/5 rounded-lg px-2.5 py-1 text-xs transition-colors">
                      <button onClick={() => loadProfile(pr.name)} className="text-white hover:text-[var(--accent)] font-semibold">{pr.name}</button>
                      <button onClick={() => deleteProfile(pr.name)} className="text-white/40 hover:text-red-400 ml-1.5" title="Delete preset" aria-label={`Delete preset ${pr.name}`}><Icon.close size={12} /></button>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <div className="flex flex-wrap gap-2 mt-4 pt-3 border-t border-white/5">
              <Button size="xs" variant="secondary" onClick={exportProfiles}>Export Presets</Button>
              <label className="inline-flex items-center justify-center px-2.5 py-1.5 rounded-lg text-xs font-bold bg-white/5 border border-white/10 text-white/80 hover:bg-white/10 hover:text-white cursor-pointer transition-all active:scale-95 focus-within:outline focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-[var(--accent)]">
                Import Presets
                <input type="file" accept=".json" onChange={importProfiles} className="sr-only" />
              </label>
            </div>
            <div className="flex flex-wrap gap-2 mt-2">
              <Button size="xs" variant="secondary" onClick={exportRecipe} className="!text-[var(--accent)]">Share Recipe</Button>
              <label className="inline-flex items-center justify-center px-2.5 py-1.5 rounded-lg text-xs font-bold bg-white/5 border border-white/10 text-white/80 hover:bg-white/10 hover:text-white cursor-pointer transition-all active:scale-95 focus-within:outline focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-[var(--accent)]">
                Load Recipe
                <input type="file" accept=".json" onChange={importRecipe} className="sr-only" />
              </label>
            </div>
            <p className="text-micro text-white/45 mt-1.5 leading-relaxed">A recipe captures every setting <span className="text-white/45">and</span> the person→source mapping, so anyone can reproduce this exact look.</p>
          </Section>
        </div>

        <div>
          <Section title="Live Telemetry & Diagnostics" collapsible defaultOpen={false}>
            {telemetry ? (
              <div className="space-y-4 text-xs font-mono">
                {/* GPU & VRAM */}
                <div className="bg-black/25 p-3 rounded-xl border border-white/5 space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-white/45 text-micro uppercase font-bold tracking-wider">GPU</span>
                    <span className="text-white font-semibold truncate max-w-[200px]">{telemetry.gpu}</span>
                  </div>
                  {telemetry.vram_total > 0 && (
                    <div className="space-y-1">
                      <div className="flex justify-between text-micro">
                        <span className="text-white/40">VRAM Usage</span>
                        <span className="text-emerald-400 font-bold">{telemetry.vram_used} GB / {telemetry.vram_total} GB</span>
                      </div>
                      <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
                        <div 
                          className="bg-emerald-500 h-full rounded-full transition-all duration-500" 
                          style={{ width: `${Math.min(100, (telemetry.vram_used / telemetry.vram_total) * 100)}%` }} 
                        />
                      </div>
                    </div>
                  )}
                </div>

                {/* CPU & Memory */}
                <div className="bg-black/25 p-3 rounded-xl border border-white/5 space-y-2.5">
                  <div className="space-y-1">
                    <div className="flex justify-between items-center text-micro">
                      <span className="text-white/40 uppercase font-bold tracking-wider">CPU Utilization</span>
                      <span className="text-orange-400 font-bold">{telemetry.cpu_percent}%</span>
                    </div>
                    <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
                      <div 
                        className="bg-orange-500 h-full rounded-full transition-all duration-500" 
                        style={{ width: `${Math.min(100, telemetry.cpu_percent)}%` }} 
                      />
                    </div>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between items-center text-micro">
                      <span className="text-white/40 uppercase font-bold tracking-wider">System RAM</span>
                      <span className="text-blue-300 font-bold">{telemetry.ram_used} GB / {telemetry.ram_total} GB</span>
                    </div>
                    {telemetry.ram_total > 0 && (
                      <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
                        <div 
                          className="bg-blue-500 h-full rounded-full transition-all duration-500" 
                          style={{ width: `${Math.min(100, (telemetry.ram_used / telemetry.ram_total) * 100)}%` }} 
                        />
                      </div>
                    )}
                  </div>
                </div>

                {/* Active threads info */}
                <div className="bg-black/25 px-3 py-2 rounded-xl border border-white/5 flex items-center justify-between">
                  <span className="text-micro text-white/45 uppercase font-bold tracking-wider">Active Python Threads</span>
                  <span className="text-pink-400 font-bold text-xs bg-pink-500/10 px-2 py-0.5 rounded-md border border-pink-500/20">{telemetry.threads}</span>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <Skeleton className="h-16 w-full" />
                <Skeleton className="h-20 w-full" />
                <Skeleton className="h-9 w-full" />
                <div className="text-micro text-white/45 italic text-center">Connecting to hardware diagnostics…</div>
              </div>
            )}
            <div className="mt-3 flex justify-between items-center">
              <Button size="sm" variant="secondary" onClick={() => setShowShortcutHUD(true)}>Keyboard Shortcuts Info</Button>
            </div>
          </Section>
        </div>
      </div>
    </div>

      {/* COLUMN 2 & 3 WRAPPER: Preview canvas leads (hero) with the media
          asset managers as a right rail (2xl:flex-row-reverse), so the live
          preview is the visual center instead of buried on the far right. */}
      <div className="flex-1 w-full min-w-0 space-y-6 flex flex-col 2xl:flex-row-reverse gap-6">

        {/* COLUMN 2: Media Asset Manager — right rail. Stays put during a run,
            as with column 1: the sources and targets a running job baked in are
            already its own, so editing these is setting up the NEXT job. */}
        <div className={`w-full 2xl:w-[360px] 3xl:w-[440px] 4xl:w-[500px] shrink-0 space-y-6 select-none ${!showRightPanel ? 'hidden' : ''}`}>
          <Section title="Target media" icon={Icon.faceswap}>
            <PersonGroups
              targetFaces={targetFaces}
              targetGroups={targetGroups}
              targetNames={targetNames}
              targetFacesInfo={targetFacesInfo}
              selTargetFace={selTargetFace}
              setSelTargetFace={setSelTargetFace}
              sourceFaces={sourceFaces}
              faceMapping={faceMapping}
              setFaceMapping={setFaceMapping}
              frame={frame}
              selTarget={selTarget}
              setTargetFaces={setTargetFaces}
              setTargetGroups={setTargetGroups}
              setTargetNames={setTargetNames}
              setTargetFacesInfo={setTargetFacesInfo}
              notify={notify}
              clearPreviewCache={clearPreviewCache}
            />
          </Section>

          <Section title="Enhancements">
            <Toggle label="Lock face identities (video)" info="For 'Selected face' mode on video: tracks each person across the clip and keeps them on one source, so identities don't flip frame-to-frame when faces cross or turn. Adds a short tracking pre-pass; the swap stays multi-threaded." checked={!!p.track_identities} onChange={(v) => set('track_identities', v)} />
            <Toggle label="Stabilize face (video)" info="Temporal keypoint smoothing — reduces swap wobble. Runs at Max Threads (2-pass) unless Enhancer Flicker is on." checked={!!p.stabilize_face} onChange={(v) => set('stabilize_face', v)} />
            {p.stabilize_face && (
              <>
                <Select label="Smoothing method" info="One Euro = adaptive (best jitter-vs-lag). EMA = simpler fixed smoothing." value={p.stabilize_method || 'one_euro'} onChange={(v) => set('stabilize_method', v)} options={['one_euro', 'ema']} />
                {p.stabilize_method !== 'ema' && (
                  <>
                    <Slider label="Smoothing (min cutoff)" info="lower = smoother, more lag" min={0.01} max={0.3} step={0.01} value={num(p.stabilize_min_cutoff, 0.05)} onChange={(v) => set('stabilize_min_cutoff', v)} />
                    <Slider label="Reactivity (beta)" info="higher = less lag on fast motion" min={0} max={0.2} step={0.01} value={num(p.stabilize_beta, 0.02)} onChange={(v) => set('stabilize_beta', v)} />
                  </>
                )}
              </>
            )}
            {!p.lipsync_enabled && (
              <Toggle label="Restore original mouth area" checked={!!p.restore_original_mouth} onChange={(v) => set('restore_original_mouth', v)} />
            )}
            <Toggle
              label="Lip-sync (MuseTalk)"
              info="Regenerates the mouth region to match a driving audio track, after everything else — same slot as 'Restore original mouth area', so the two are mutually exclusive (this one wins when both would apply). 'Original audio' re-syncs the mouth to this clip's own soundtrack, repairing drift the swap itself introduces; 'Uploaded track' dubs against a separate audio file instead."
              checked={!!p.lipsync_enabled}
              onChange={(v) => set('lipsync_enabled', v)}
            />
            {p.lipsync_enabled && (
              <>
                <Select
                  label="Lip-sync audio source"
                  value={p.lipsync_audio_source || 'original'}
                  onChange={(v) => set('lipsync_audio_source', v)}
                  options={[{ value: 'original', label: 'Original audio' }, { value: 'upload', label: 'Uploaded track' }]}
                />
                {p.lipsync_audio_source === 'upload' && (
                  <FileDrop
                    accept="audio/*"
                    label={p.lipsync_audio_path ? `Dub track: ${p.lipsync_audio_path.split(/[\\/]/).pop()}` : 'Add dub audio track'}
                    onFiles={onAddLipsyncAudio}
                    busy={uploadingLipsyncAudio}
                    hint="drop an audio file here"
                  />
                )}
              </>
            )}
            <Toggle
              label="Restore original eyes"
              info="Brings the TARGET's own eyes back over the swap. Every identity swapper here works from a 112–128px aligned crop, so the eyes — the smallest high-frequency detail in a face, and the first thing a viewer looks at — come back soft, with the gaze nudged toward wherever the source person was looking. This keeps the plate's gaze direction, catchlights and eyelash detail while everything else stays swapped. Built from the two eye keypoints, not the 106-point landmarks the mouth restore uses, so it works on every detector engine. It fades out past 25° of yaw or pitch and switches off at a true profile, because there the far eye is edge-on and the near one sits over the nose bridge — pasting the plate there doubles the socket. Costs one ellipse and a blur, so it is effectively free. Note that it undoes the swapper's eye SHAPE too, which is part of the identity: if the two people's eyes differ a lot, back the amount off rather than running it at 1."
              checked={!!p.restore_original_eyes}
              onChange={(v) => set('restore_original_eyes', v)}
            />
            {p.restore_original_eyes && (
              <>
                <Slider label="Eyes blend amount" info="How much of the original eyes comes back. 1 = fully the plate's eyes, 0.5 = half-way, which keeps some of the swapped eye shape while recovering the sharpness." min={0} max={1} step={0.05} value={num(p.eyes_blend_amount, 1)} onChange={(v) => set('eyes_blend_amount', v)} />
                <Slider label="Eyes feather" info="Edge softness, as a percentage of the eye radius rather than a pixel count — a 15px feather is most of a distant face's eye and a hairline on a close-up, so a pixel value would need retuning every shot. Raise it if you can see where the restore stops." min={0} max={100} step={1} value={num(p.eyes_feather_blend, 25)} onChange={(v) => set('eyes_feather_blend', v)} />
                <Slider label="Eyes region size" info="Scales both ellipses together. The default covers the eye and lashes; raise it to take in the lid and socket, lower it to restore the iris alone." min={0.3} max={2} step={0.05} value={num(p.eyes_size_factor, 1)} onChange={(v) => set('eyes_size_factor', v)} />
                <Slider label="Eyes width" info="Stretches the region horizontally only — useful for wide or almond eyes where the default ellipse clips the outer corner." min={0.3} max={2} step={0.05} value={num(p.eyes_radius_x, 1)} onChange={(v) => set('eyes_radius_x', v)} />
                <Slider label="Eyes height" info="Stretches the region vertically only. Raise it to include the brow line and lower lid; keep it low to avoid pulling the target's eyebrows back in." min={0.3} max={2} step={0.05} value={num(p.eyes_radius_y, 1)} onChange={(v) => set('eyes_radius_y', v)} />
              </>
            )}
            {/* startsWith, not ===: the fp16 tier is the same model at half
                precision and takes the same fidelity input, so gating on an
                exact name would have hidden the slider for it. */}
            {p.selected_enhancer && p.selected_enhancer.toLowerCase().startsWith('codeformer') && (
              <Slider
                label="CodeFormer fidelity weight"
                info="Balances restoration quality vs original identity. 0.1 = maximum sharpness (may shift face geometry), 0.9 = maximum similarity to source (but less restoration detail). Verified live against this export — many CodeFormer ONNX conversions bake the weight in as a constant and the slider silently does nothing; measured here, w=0 and w=1 differ by a mean of 0.20 in the model's -1..1 output range."
                min={0.1}
                max={0.9}
                step={0.05}
                value={num(p.codeformer_fidelity, 0.5)}
                onChange={(v) => set('codeformer_fidelity', v)}
              />
            )}
            {p.selected_enhancer && p.selected_enhancer !== 'None' && (
              <>
                <Toggle
                  label="Align crop for the enhancer"
                  info="Re-crops the face into the alignment the RESTORER was trained on before enhancing, then puts it back. CodeFormer, GFPGAN, RestoreFormer++ and GPEN all learned their priors from FFHQ-aligned 512 faces — but the crop they are handed is whatever the SWAPPER's template produced, because the enhancer reuses the swap crop. Measured against ffhq_512 at 512px: the inswapper family is 0.856 scale and 22.1px mean landmark error (the face arrives ~17% larger than the prior expects and the eyes sit 31px too high), ghost/simswap 0.828 / 16.3px, blendswap's source template 0.744 / 30.7px, hififace 0.875 / 13.2px. Only uniface and blendswap already match exactly, and for those this does nothing. It matters most for CodeFormer, whose codebook lookup is a discrete nearest neighbour into a learned dictionary — an off-distribution input retrieves the wrong entries. Costs two extra 512 warps per face (~2ms), and those are two extra resampling passes, so A/B it: better prior match against slightly softer resampling. OFF keeps every render bit-identical to before."
                  checked={!!p.enhancer_align}
                  onChange={(v) => set('enhancer_align', v)}
                />
                <Toggle
                  label="Re-match colour after enhancing"
                  info="Runs the colour/lighting match a second time, on the enhancer's output. The first pass runs on the SWAP, before the restorer — and the restorers regrade what they are given, because they were trained to produce a plausible face rather than to preserve the plate's exposure. So some of the tone match is spent by the time the crop is pasted back. This does not replace the first pass: that one also feeds the enhancer a better-exposed input, which is worth keeping. Uses whichever colour mode is selected above. Cheap (one more transfer on a crop already in memory), but how much it changes depends entirely on how far your restorer moves the grade, so judge it on your own footage."
                  checked={!!p.color_match_after_enhance}
                  onChange={(v) => set('color_match_after_enhance', v)}
                />
              </>
            )}
          </Section>

          <Section title="Target file(s)">
            <FileDrop accept="image/*,video/*,.webp" multiple label="Add target media" onFiles={onAddTarget} busy={uploadingTgt} hint="drop images or videos here"
                      progress={tgtProgress} onCancel={() => tgtAbortRef.current?.abort()}
                      onPaths={onAddTargetPaths} />
            {targets.length === 0 ? (
              <div className="h-24 flex items-center justify-center rounded-xl border border-dashed border-white/10 text-xs text-white/35 bg-black/10 select-none">No target media loaded</div>
            ) : (
              <div className="space-y-2 max-h-52 overflow-y-auto pr-1">
                {targets.map((t, i) => {
                  // Show the most advanced state any queued job for this file
                  // reached — with segments, one target can carry several jobs.
                  const jobs = queue.jobs.filter(j => j.target_name === t.name);
                  const job = jobs.find(j => j.status === 'running')
                    || jobs.find(j => j.status === 'failed' || j.status === 'stopped')
                    || jobs.find(j => j.status === 'finished');
                  let statusLabel = null;
                  let badgeColor = '';
                  if (job) {
                    if (job.status === 'running') {
                      statusLabel = 'Running';
                      badgeColor = 'text-yellow-400 border-yellow-500/30 bg-yellow-500/10 animate-pulse';
                    } else if (job.status === 'finished') {
                      statusLabel = 'Finished';
                      badgeColor = 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10';
                    } else {
                      statusLabel = job.status === 'stopped' ? 'Stopped' : 'Failed';
                      badgeColor = 'text-red-400 border-red-500/30 bg-red-500/10';
                    }
                  }
                  const isVideo = t.frames > 1;
                  const duration = isVideo && t.fps ? (t.frames / t.fps).toFixed(1) : null;
                  const TypeIcon = isVideo ? Icon.film : Icon.still;
                  return (
                    <div key={i}
                      className={`group flex items-center gap-3 px-3.5 py-3 rounded-xl text-sm border transition-all duration-200 cursor-pointer ${selTarget === i ? 'bg-[var(--accent)]/10 border-[var(--accent)]/40' : 'bg-white/[0.02] border-white/[0.06] hover:border-white/15 hover:bg-white/[0.04]'}`}
                      onClick={() => selectTarget(i)}>
                      {/* v=name busts the browser cache: the URL is index-based, so
                          after a removal the same index can point at another file. */}
                      <img
                        src={`${API}/api/target/preview?index=${i}&frame=1&width=96&v=${encodeURIComponent(t.name)}`}
                        alt=""
                        loading="lazy"
                        className="w-12 h-9 shrink-0 rounded-lg object-cover bg-black/40 border border-white/10"
                        onError={(e) => { e.target.style.display = 'none'; if (e.target.nextElementSibling) e.target.nextElementSibling.style.display = 'block'; }}
                      />
                      {/* Revealed by the <img>'s onError above (it sets
                          display:block on its nextElementSibling), so this is
                          the fallback when a thumbnail cannot be decoded. */}
                      <div className="shrink-0 opacity-75 hidden text-white/50"><TypeIcon size={16} /></div>
                      <div className="flex-1 min-w-0">
                        <span className="truncate block font-bold text-white/90 group-hover:text-white transition-colors">{t.name}</span>
                        <div className="flex items-center gap-2 mt-0.5 text-micro font-medium text-white/45">
                          {isVideo ? (
                            <span>{t.frames} frames · {t.fps} FPS{duration ? ` · ${duration}s` : ''}</span>
                          ) : (
                            <span>Static Image</span>
                          )}
                          {statusLabel && (
                            <span className={`text-nano uppercase tracking-wider px-1.5 py-0.5 rounded border font-semibold ${badgeColor}`}>
                              {statusLabel}
                            </span>
                          )}
                        </div>
                      </div>
                      <button type="button" title="Remove this target" aria-label="Remove this target"
                        onClick={(e) => { e.stopPropagation(); removeTarget(i); }}
                        className="h-6 w-6 shrink-0 rounded-full bg-black/50 text-white/60 opacity-0 group-hover:opacity-100 focus-visible:opacity-100 hover:bg-[var(--accent-hover)] hover:text-white transition-all flex items-center justify-center"><Icon.close size={12} /></button>
                    </div>
                  );
                })}
              </div>
            )}
            {targets.length > 0 && (
              <div className="pt-2 border-t border-white/5 flex justify-end">
                <Button size="sm" variant="stop" onClick={async () => { const r = await postJSON('/api/target/clear', {}); setTargets(r.targets); setTargetFaces([]); setTargetGroups([]); setTargetNames([]); setTargetFacesInfo([]); setFaceMapping({}); setPreviewSrc(''); setPreviewFor(''); }}>Clear targets</Button>
              </div>
            )}
          </Section>
          
          <Section title="Source images / facesets" icon={Icon.faces}>
          <FileDrop accept="image/*,.fsz" multiple label="Add source faces" onFiles={onAddSource} busy={uploadingSrc} hint="drop images or .fsz here"
                    progress={srcProgress} onCancel={() => srcAbortRef.current?.abort()} />
          <FacesetLibrary
            canSave={sourceFaces.length > 0}
            onLoaded={(r) => { setSourceFaces(r.source_faces || []); if (r.source_faces_info) setSourceFacesInfo(r.source_faces_info); }}
            notify={notify} />
          {pinnedIdentities.length > 0 && (
            <div className="p-2.5 rounded-xl bg-black/40 border border-white/10 space-y-1.5 animate-slide-up">
              <div className="flex items-center justify-between text-micro font-bold uppercase tracking-wider text-white/50">
                <span>⭐ Pinned Quick Identities ({pinnedIdentities.length})</span>
                <button type="button" onClick={() => { setPinnedIdentities([]); localStorage.removeItem('roop_pinned_identities'); }} className="text-white/30 hover:text-white">Clear</button>
              </div>
              <div className="flex items-center gap-2 overflow-x-auto py-1">
                {pinnedIdentities.map((pin, i) => (
                  <button
                    key={pin.id || i}
                    type="button"
                    onClick={() => {
                      const idx = sourceFaces.findIndex((f) => f === pin.data);
                      if (idx >= 0) selectSource(idx);
                      else notify?.('Face not in current source gallery', 'warning');
                    }}
                    className="relative group h-12 w-12 rounded-lg border border-white/20 overflow-hidden shrink-0 hover:border-[var(--accent)] transition-all"
                    title={pin.label}
                  >
                    <img src={pin.data} alt="Pin" className="w-full h-full object-cover" />
                  </button>
                ))}
              </div>
            </div>
          )}
          <FaceGallery title="Input faces" faces={sourceFaces} selected={selSource} onSelect={selectSource} draggable={true} large={true}
            onRemove={(i) => removeSource(i)} empty="Upload a face image" info={sourceFacesInfo} />
          {sourceFaces.length > 0 && (
            <div className="space-y-3">
              <div className="flex flex-wrap gap-2">
                <Button size="sm" variant="secondary" onClick={togglePinCurrentSource} title="Pin selected identity to Quick Bar">
                  ⭐ Pin Identity
                </Button>
                <Button size="sm" variant="secondary" onClick={() => sourceAction('/api/source/move', { index: selSource, direction: 'left' })}>⬅ Move</Button>
                <Button size="sm" variant="secondary" onClick={() => sourceAction('/api/source/move', { index: selSource, direction: 'right' })}>Move ➡</Button>
                <Button size="sm" variant="secondary" onClick={() => sourceAction('/api/source/remove', { index: selSource })}>Remove</Button>
                <Button size="sm" variant="secondary" title="Set each tile to the most frontal face in its set" onClick={() => sourceAction('/api/source/refresh_thumbs', {})}>Frontal thumb</Button>
                <Button size="sm" variant="stop" onClick={() => sourceAction('/api/source/clear', {})}>Clear all</Button>
              </div>
              
              {sourceFacesInfo[selSource] && (
                <div className="p-3.5 rounded-xl bg-black/45 border border-white/5 space-y-2 text-xs select-none">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-micro uppercase tracking-[0.14em] text-white/50">Selected source details</span>
                    <span className="px-2 py-0.5 rounded-full bg-[var(--accent)]/10 text-micro text-[var(--accent)] font-bold border border-[var(--accent)]/20">
                      {sourceFacesInfo[selSource].count > 1 ? `${sourceFacesInfo[selSource].count} Reference Faces` : 'Single Face'}
                    </span>
                  </div>
                  
                  <div className="space-y-1.5 pt-1">
                    {sourceFacesInfo[selSource].count > 1 ? (
                      <>
                        <div className="text-micro font-bold text-white/45 mb-1">Pose Coverage Breakdown:</div>
                        <div className="flex flex-wrap gap-1.5">
                          {Object.entries(
                            sourceFacesInfo[selSource].poses.reduce((acc, p) => {
                              acc[p] = (acc[p] || 0) + 1;
                              return acc;
                            }, {})
                          ).map(([pose, cnt]) => (
                            <span key={pose} className="px-2 py-1 rounded-lg bg-white/[0.03] border border-white/5 text-micro text-white/70">
                              {pose} <span className="text-[var(--accent)] font-extrabold">({cnt})</span>
                            </span>
                          ))}
                        </div>
                      </>
                    ) : (
                      <div className="flex items-center justify-between text-white/60">
                        <span>Detected Pose:</span>
                        <span className="font-bold text-white">{sourceFacesInfo[selSource].poses[0] || 'Front'}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </Section>
      </div>

        {/* COLUMN 3: Active Canvas, Timeline & Outputs */}
        <div className="flex-1 min-w-0 space-y-6">
          {/* run bar */}
          <div className="sticky top-20 z-30 pb-3 bg-[#0c0e14]/90 backdrop-blur-md">
            {/* The run bar keeps its shape whether or not something is
                rendering. A run has its own tab now, so this one no longer
                turns into a progress panel — it just says a job is in flight,
                and offers the way over to it. */}
            <div className="w-full space-y-4">
              <div className="rounded-2xl glass-panel p-6 flex flex-col md:flex-row items-center justify-between gap-6 shadow-2xl border border-white/5 w-full">
                <div className="flex items-center gap-3 w-full md:w-auto">
                  <Button variant="primary" size="lg" onClick={start} disabled={targets.length === 0 || sourceFaces.length === 0 || progress.processing} className="w-full md:w-auto justify-center">▶ Start Swapping</Button>
                  {maxFrames > 1 && (
                    <Button variant="secondary" size="lg" onClick={renderPreviewClip} disabled={targets.length === 0 || sourceFaces.length === 0 || isGeneratingPreviewClip || progress.processing} className="!text-orange-400 border border-orange-500/20 hover:bg-orange-500/10 w-full md:w-auto justify-center">
                      Render 5s Preview
                    </Button>
                  )}
                  {progress.processing && (
                    <Button variant="secondary" size="lg" onClick={onOpenProcessing} className="w-full md:w-auto justify-center">
                      View progress →
                    </Button>
                  )}
                </div>
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2.5 text-sm font-semibold text-[var(--text-muted)] max-w-xs truncate text-right">
                    {progress.processing ? (
                      <>
                        <span className={`h-2.5 w-2.5 rounded-full ${progress.paused ? 'bg-amber-400' : 'bg-[var(--accent)] animate-pulse shadow-[0_0_8px_var(--accent-glow)]'}`} />
                        <span className="tabular-nums">
                          {progress.paused ? 'Paused' : `Rendering ${Math.round(prog * 100)}%`}
                          {etaMs > 0 ? ` · ${fmtTime(etaMs)} left` : ''}
                        </span>
                      </>
                    ) : (
                      <>
                        <span className={`h-2.5 w-2.5 rounded-full ${targets.length > 0 && sourceFaces.length > 0 ? 'bg-emerald-500 animate-pulse shadow-[0_0_8px_#10b981]' : 'bg-red-500/50'}`} />
                        {targets.length === 0 ? 'No target media selected' : sourceFaces.length === 0 ? 'No source faces loaded' : 'Ready to swap'}
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* The preview holds a live image and its scrub/compare controls;
              nothing here should move under the cursor. This used to spell that
              out as three separate opt-outs, which is now just an elevation. */}
          <Section title="Preview" elevation="flat">
            <div className="space-y-4">
                {/* Media Session Tab Bar for multi-file workspace */}
                <MediaTabSessionBar
                  targets={targets}
                  selTarget={selTarget}
                  onSelectTarget={selectTarget}
                  onRemoveTarget={removeTarget}
                  onAddTarget={() => {
                    const inputEl = document.createElement('input');
                    inputEl.type = 'file';
                    inputEl.accept = 'image/*,video/*,.webp';
                    inputEl.multiple = true;
                    inputEl.onchange = (e) => {
                      if (e.target.files && e.target.files.length > 0) {
                        onAddTarget(e.target.files);
                      }
                    };
                    inputEl.click();
                  }}
                />

                {/* Live Slider Tracker Bar positioned directly above the preview box */}
                <SliderTrackerBar
                  params={p}
                  onSetParam={(key, val) => {
                    set(key, val);
                  }}
                  sliderEffectEnabled={sliderEffectEnabled}
                  onToggleSliderEffect={toggleSliderEffect}
                  onResetSliders={resetTrackerSliders}
                  onRefreshPreview={() => refreshPreview({ force: true })}
                />

                {previewSrc ? (
                  comparingEnhancers ? (() => {
                const activeList = selectedGridEnhancers.filter(e => meta.enhancers?.includes(e));
                const gridColsClass = activeList.length === 1 ? 'grid-cols-1' : 'grid-cols-2';
                return (
                  <div className="space-y-4">
                    {/* Enhancer selector row */}
                    <div className="p-3.5 rounded-xl bg-black/45 border border-white/5 space-y-2 select-none">
                      <span className="text-micro font-semibold uppercase tracking-[0.14em] text-white/45 block">Compare Enhancers (Select up to 4)</span>
                      <div className="flex flex-wrap gap-2">
                        {meta.enhancers?.map((enh) => {
                          const isSelected = selectedGridEnhancers.includes(enh);
                          return (
                            <button
                              key={enh}
                              type="button"
                              onClick={() => {
                                if (isSelected) {
                                  if (selectedGridEnhancers.length > 1) {
                                    setSelectedGridEnhancers(prev => prev.filter(x => x !== enh));
                                  }
                                } else {
                                  if (selectedGridEnhancers.length >= 4) {
                                    notify('You can select a maximum of 4 enhancers for grid comparison.', 'warning');
                                  } else {
                                    setSelectedGridEnhancers(prev => [...prev, enh]);
                                  }
                                }
                              }}
                              className={`px-3 py-1.5 rounded-lg text-mini font-semibold border transition-all duration-200 ${isSelected ? 'bg-[var(--accent)]/15 border-[var(--accent)]/40 text-white' : 'bg-white/[0.02] border-white/10 text-white/50 hover:border-white/20 hover:text-white/85'}`}
                            >
                              {enh}
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    <CompareGrid
                      items={activeList}
                      gridColsClass={gridColsClass}
                      previews={enhancerPreviews}
                      times={enhancerTimes}
                      timers={liveRenderingTimers}
                    />
                  </div>
                );
              })() : comparingMasks ? (() => {
                const activeMasks = selectedGridMasks.filter(m => meta.mask_engines?.includes(m));
                const gridColsClass = activeMasks.length === 1 ? 'grid-cols-1' : 'grid-cols-2';
                return (
                  <div className="space-y-4">
                    {/* Mask-engine selector row */}
                    <div className="p-3.5 rounded-xl bg-black/45 border border-white/5 space-y-2 select-none">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-micro font-semibold uppercase tracking-[0.14em] text-white/45 block">Compare Mask Engines (Select up to 4)</span>
                        <span className="text-micro text-white/45">Enhancer: <span className="text-white/55 font-semibold">{p.selected_enhancer || 'None'}</span></span>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {meta.mask_engines?.map((mE) => {
                          const isSelected = selectedGridMasks.includes(mE);
                          return (
                            <button
                              key={mE}
                              type="button"
                              onClick={() => {
                                if (isSelected) {
                                  if (selectedGridMasks.length > 1) {
                                    setSelectedGridMasks(prev => prev.filter(x => x !== mE));
                                  }
                                } else {
                                  if (selectedGridMasks.length >= 4) {
                                    notify('You can select a maximum of 4 mask engines for grid comparison.', 'warning');
                                  } else {
                                    setSelectedGridMasks(prev => [...prev, mE]);
                                  }
                                }
                              }}
                              className={`px-3 py-1.5 rounded-lg text-mini font-semibold border transition-all duration-200 ${isSelected ? 'bg-[var(--accent)]/15 border-[var(--accent)]/40 text-white' : 'bg-white/[0.02] border-white/10 text-white/50 hover:border-white/20 hover:text-white/85'}`}
                            >
                              {mE}
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    <CompareGrid
                      items={activeMasks}
                      gridColsClass={gridColsClass}
                      previews={maskPreviews}
                      times={maskTimes}
                      timers={maskRenderTimers}
                    />
                  </div>
                );
              })() : comparingSwappers ? (() => {
                const activeSwappers = selectedGridSwappers.filter(m => meta.swap_models?.includes(m));
                const gridColsClass = activeSwappers.length === 1 ? 'grid-cols-1' : 'grid-cols-2';
                return (
                  <div className="space-y-4">
                    {/* Swapper-model selector row */}
                    <div className="p-3.5 rounded-xl bg-black/45 border border-white/5 space-y-2 select-none">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-micro font-semibold uppercase tracking-[0.14em] text-white/45 block">Compare Swapper Models (Select up to 4)</span>
                        <span className="text-micro text-white/45">Enhancer: <span className="text-white/55 font-semibold">{p.selected_enhancer || 'None'}</span></span>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {meta.swap_models?.map((sM) => {
                          const isSelected = selectedGridSwappers.includes(sM);
                          return (
                            <button
                              key={sM}
                              type="button"
                              onClick={() => {
                                if (isSelected) {
                                  if (selectedGridSwappers.length > 1) {
                                    setSelectedGridSwappers(prev => prev.filter(x => x !== sM));
                                  }
                                } else {
                                  if (selectedGridSwappers.length >= 4) {
                                    notify('You can select a maximum of 4 swapper models for grid comparison.', 'warning');
                                  } else {
                                    setSelectedGridSwappers(prev => [...prev, sM]);
                                  }
                                }
                              }}
                              className={`px-3 py-1.5 rounded-lg text-mini font-semibold border transition-all duration-200 ${isSelected ? 'bg-[var(--accent)]/15 border-[var(--accent)]/40 text-white' : 'bg-white/[0.02] border-white/10 text-white/50 hover:border-white/20 hover:text-white/85'}`}
                            >
                              {sM}
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    <CompareGrid
                      items={activeSwappers}
                      gridColsClass={gridColsClass}
                      previews={swapperPreviews}
                      times={swapperTimes}
                      timers={swapperRenderTimers}
                    />
                  </div>
                );
              })() : comparingUpscalers ? (() => {
                const activeUpscalers = selectedGridUpscalers.filter(l => AI_UPSCALE_MODELS.some(m => m.label === l));
                const gridColsClass = activeUpscalers.length === 1 ? 'grid-cols-1' : 'grid-cols-2';
                return (
                  <div className="space-y-4">
                    {/* AI-upscaler selector row */}
                    <div className="p-3.5 rounded-xl bg-black/45 border border-white/5 space-y-2 select-none">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-micro font-semibold uppercase tracking-[0.14em] text-white/45 block">Compare AI Upscalers (Select up to 4)</span>
                        <span className="text-micro text-white/45">Swaps once, then upscales each</span>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {AI_UPSCALE_MODELS.map((m) => {
                          const isSelected = selectedGridUpscalers.includes(m.label);
                          return (
                            <button
                              key={m.value}
                              type="button"
                              onClick={() => {
                                if (isSelected) {
                                  if (selectedGridUpscalers.length > 1) {
                                    setSelectedGridUpscalers(prev => prev.filter(x => x !== m.label));
                                  }
                                } else {
                                  if (selectedGridUpscalers.length >= 4) {
                                    notify('You can select a maximum of 4 upscalers for grid comparison.', 'warning');
                                  } else {
                                    setSelectedGridUpscalers(prev => [...prev, m.label]);
                                  }
                                }
                              }}
                              className={`px-3 py-1.5 rounded-lg text-mini font-semibold border transition-all duration-200 ${isSelected ? 'bg-[var(--accent)]/15 border-[var(--accent)]/40 text-white' : 'bg-white/[0.02] border-white/10 text-white/50 hover:border-white/20 hover:text-white/85'}`}
                            >
                              {m.label}
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    <CompareGrid
                      items={activeUpscalers}
                      gridColsClass={gridColsClass}
                      previews={upscalePreviews}
                      times={upscaleTimes}
                      timers={upscaleRenderTimers}
                    />
                  </div>
                );
              })() : (
                <div className="relative rounded-2xl">
                  <AmbilightGlow
                    previewSrc={previewSrc || rawUrl}
                    enabled={ambilightEnabled}
                  />
                  <InteractivePreview
                    beforeSrc={(isPlaying && bufferedSrc) ? bufferedSrc : rawUrl}
                    afterSrc={stageAfterSrc}
                    scrubbing={scrubbingNow}
                    onMaskChange={applyManualMask}
                    maskApplied={!!manualMask}
                    faces={previewFaces}
                    kps={previewKps}
                    pose={previewPose}
                    personIds={previewPersonIds}
                    onSelectPerson={addPersonFromBox}
                    splitView={splitView}
                    compare={compare}
                    onToggleCompare={() => setCompare((v) => { const n = !v; if (n) { setComparingEnhancers(false); setComparingMasks(false); setComparingSwappers(false); setComparingUpscalers(false); } return n; })}
                    sliderEffectEnabled={sliderEffectEnabled}
                    onToggleSliderEffect={toggleSliderEffect}
                    frame={frame}
                    setFrame={setFrame}
                    maxFrames={maxFrames}
                    isPlaying={isPlaying}
                    previewing={previewing}
                    previewSecs={previewSecs}
                    setIsPlaying={setIsPlaying}
                  />
                </div>
              )
            ) : (
              <div className="relative aspect-video rounded-2xl overflow-hidden bg-gradient-to-br from-white/[0.03] to-black/20 border border-white/10 flex items-center justify-center select-none">
                <div className="absolute inset-0 pointer-events-none opacity-70" style={{ background: 'radial-gradient(circle at 50% 42%, var(--accent-glow), transparent 62%)' }} />
                {previewing ? (
                  <div className="relative flex flex-col items-center gap-3">
                    <div className="h-9 w-9 rounded-full border-2 border-white/10 border-t-[var(--accent)] animate-spin" />
                    <span className="text-sm font-medium text-white/50">Rendering preview…</span>
                  </div>
                ) : (() => {
                  const hasTarget = targets.length > 0;
                  const hasSource = sourceFaces.length > 0;
                  const ready = hasTarget && hasSource;
                  const steps = [
                    { done: hasTarget, label: 'Add a target image or video' },
                    { done: hasSource, label: 'Add a source face' },
                  ];
                  return (
                    <div className="relative flex flex-col items-center gap-4 text-center px-6 w-full max-w-sm">
                      <div className={`grid place-items-center h-14 w-14 rounded-2xl border ${ready ? 'bg-emerald-500/10 border-emerald-500/25 text-emerald-400/80' : 'bg-[var(--accent)]/10 border-[var(--accent)]/20 text-[var(--accent)]/80'}`}>
                        {ready ? <Icon.full size={24} /> : <Icon.faceswap size={24} />}
                      </div>
                      <div>
                        <div className="text-sm font-semibold text-white/85">{ready ? 'Ready to preview' : 'No preview yet'}</div>
                        <div className="text-xs text-white/40 mt-1 leading-relaxed">
                          {ready
                            ? 'Scrub the timeline or press ▶ Start Swapping to render.'
                            : 'Two quick steps and the live swap shows up right here.'}
                        </div>
                      </div>
                      {!ready && (
                        <div className="w-full space-y-1.5">
                          {steps.map((s, i) => (
                            <div key={i} className={`flex items-center gap-2.5 px-3 py-2 rounded-lg border text-xs font-medium transition-colors ${s.done ? 'border-emerald-500/25 bg-emerald-500/[0.06] text-emerald-300/90' : 'border-white/10 bg-white/[0.02] text-white/55'}`}>
                              <span className={`grid place-items-center h-4 w-4 rounded-full text-nano font-bold shrink-0 ${s.done ? 'bg-emerald-500 text-black' : 'bg-white/10 text-white/50'}`}>{s.done ? '✓' : i + 1}</span>
                              <span className={s.done ? 'line-through opacity-70' : ''}>{s.label}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })()}
              </div>
            )}
          </div>
            
            {/* Clip timeline. It used to go away for the duration of a run;
                it now stays, because this tab is where the next job is set up
                and the timeline is how a range is chosen. */}
            {maxFrames > 1 && showTimelineDeck && (
              <div className="pt-4 border-t border-[var(--border-color)]">
                <Timeline
                  fps={targets[selTarget]?.fps || 25}
                  maxFrames={maxFrames}
                  frame={frame}
                  setFrame={setFrame}
                  startFrame={startFrame}
                  endFrame={endFrame}
                  setFrameMarkerVal={setFrameMarkerVal}
                  timelineRef={timelineRef}
                  onPointerDown={handleTimelinePointerDown}
                  onPointerMove={handleTimelinePointerMove}
                  onPointerLeave={handleTimelinePointerLeave}
                  hoverFrame={hoverFrame}
                  isScrubbing={isScrubbing}
                  storyboardThumbs={storyboardThumbs}
                  view={view}
                  setView={setView}
                  isPlaying={isPlaying}
                  setIsPlaying={setIsPlaying}
                  buffering={playStalled}
                  isLooping={isLooping}
                  setIsLooping={setIsLooping}
                  playbackRate={playbackRate}
                  setPlaybackRate={setPlaybackRate}
                  thumbUrl={(f) => `${API}/api/target/preview?index=${selTarget}&frame=${f}&width=384`}
                  targetKey={targets[selTarget]?.name || String(selTarget)}
                  segments={segments.segments}
                  onSegmentClick={jumpToSegment}
                />
                <SegmentBar
                  segments={segments.segments}
                  fps={targets[selTarget]?.fps || 25}
                  maxFrames={maxFrames}
                  startFrame={startFrame}
                  endFrame={endFrame}
                  onAdd={segments.add}
                  onRemove={segments.remove}
                  onClear={segments.clear}
                  onJump={jumpToSegment}
                  onQueueAll={queueSegments}
                  onJoin={joinSegments}
                  joinable={joinableJobs.length}
                  coveredFrames={segments.coveredFrames()}
                  busy={progress.processing}
                />
              </div>
            )}

            {/* Preview controls. Each one drives a preview render, which does
                compete with a run for the GPU — but it takes a deliberate click
                to fire one, and the automatic refreshes still hold off until
                the run is over (see the deferred-preview effect above). */}
            <div className={`flex items-center flex-wrap gap-3 ${maxFrames > 1 ? 'pt-3 border-t border-white/5' : ''}`}>
              <Button size="sm" variant="secondary" title="Re-run the swap for this frame, ignoring the cached result" onClick={() => refreshPreview({ force: true })}>Refresh</Button>
              <Button size="sm" variant="primary" onClick={useFaceFromFrame}>Use face from frame</Button>
              {previewSrc && !comparingEnhancers && !comparingMasks && !comparingSwappers && !comparingUpscalers && (
                <Button size="sm" variant="secondary" disabled={upscaling} onClick={upscaleThisFrame}
                  title="AI-upscale just this frame to preview final quality">
                  {upscaling ? 'Upscaling…' : `Upscale this frame (${AI_UPSCALE_MODELS.find(m => m.value === (p.upscale_model_after || 'esrganx2'))?.label || 'AI'})`}
                </Button>
              )}
            </div>

            <div className="flex items-center flex-wrap gap-3">
              <Toggle label="Live Swap" checked={fakePreview} onChange={setFakePreview} />
              <Toggle label="Compare" checked={compare} onChange={(v) => { setCompare(v); if (v) { setComparingEnhancers(false); setComparingMasks(false); setComparingSwappers(false); setComparingUpscalers(false); } }} />
              {compare && <Toggle label="Split View" checked={splitView} onChange={setSplitView} />}
              <Toggle label="Enhancer Grid" checked={comparingEnhancers} onChange={(v) => { setComparingEnhancers(v); if (v) { setCompare(false); setComparingMasks(false); setComparingSwappers(false); setComparingUpscalers(false); } }} />
              <Toggle label="Mask Grid" checked={comparingMasks} onChange={(v) => { setComparingMasks(v); if (v) { setCompare(false); setComparingEnhancers(false); setComparingSwappers(false); setComparingUpscalers(false); } }} />
              <Toggle label="Swapper Grid" checked={comparingSwappers} onChange={(v) => { setComparingSwappers(v); if (v) { setCompare(false); setComparingEnhancers(false); setComparingMasks(false); setComparingUpscalers(false); } }} />
              <Toggle label="Upscale Grid" checked={comparingUpscalers} onChange={(v) => { setComparingUpscalers(v); if (v) { setCompare(false); setComparingEnhancers(false); setComparingMasks(false); setComparingSwappers(false); } }} />
            </div>
          </Section>

          {/* Single-frame AI-upscale spot-check modal — portaled to <body> so
              transformed ancestors (TiltCard / motion) can't clip the fixed
              overlay. */}
          {upscaledSrc && createPortal((
            <div
              className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-6"
              onClick={() => setUpscaledSrc('')}
            >
              <div
                className="relative flex flex-col max-w-[95vw] max-h-[92vh] rounded-2xl border border-white/10 bg-[var(--card-bg)] shadow-2xl overflow-hidden"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex items-center justify-between gap-4 px-4 py-2.5 border-b border-white/10">
                  <span className="text-xs font-semibold text-white/80">
                    Upscaled frame{upscaledDims ? ` · ${upscaledDims.w}×${upscaledDims.h}` : ''}
                  </span>
                  <div className="flex items-center gap-2">
                    <a
                      href={upscaledSrc}
                      download={`upscaled_frame_${frame}.png`}
                      className="px-3 py-1.5 rounded-lg text-mini font-semibold border border-white/10 bg-white/[0.04] text-white/80 hover:border-white/25 hover:text-white transition-colors"
                    >
                      ⬇ Download
                    </a>
                    <button
                      type="button"
                      onClick={() => setUpscaledSrc('')}
                      className="px-3 py-1.5 rounded-lg text-mini font-semibold border border-white/10 bg-white/[0.04] text-white/80 hover:border-white/25 hover:text-white transition-colors"
                    >
                      ✕ Close
                    </button>
                  </div>
                </div>
                <div className="overflow-auto flex-1 min-h-0">
                  <img src={upscaledSrc} alt="Upscaled frame" className="block max-w-none" />
                </div>
              </div>
            </div>
          ), document.body)}

          {targets.length > 0 && sourceFaces.length > 0 && estFrames > 1 && (
            <TiltCard className="rounded-2xl w-full" max={6}>
            <div className="rounded-2xl glass-panel p-5 shadow-2xl border border-white/5 w-full">
              <div className="flex items-center justify-between mb-4">
                <span className="text-mini uppercase tracking-[0.14em] text-white/45 font-semibold">Runtime estimation</span>
                <span className={`text-micro font-semibold uppercase tracking-wide px-2 py-0.5 rounded-full border ${estSourceClass}`}>{estSourceLabel}</span>
              </div>
              <div className="flex items-end gap-3 mb-4">
                <span className="text-3xl font-bold text-white/95 tabular-nums leading-none">~{fmtTime(estTotalMs)}</span>
                <span className="text-xs text-white/40 mb-0.5">{Math.round(estPerFrame)} ms/frame{heavyVram ? ' · high VRAM' : ''}</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-5 gap-y-2.5 text-mini">
                {[
                  ['Frames', estFrames.toLocaleString()],
                  ['Duration', estDurationS ? `${estDurationS.toFixed(1)}s @ ${estFps} fps` : '—'],
                  ['Faces / frame', `${previewFaces.length || '—'}${calibEst?.density_bucket ? ` (${calibEst.density_bucket})` : ''}`],
                  ['This combo', estLearned ? `${calibEst.samples} run${calibEst.samples > 1 ? 's' : ''}` : 'no data yet'],
                  ['GPU', calibEst?.gpu || '—'],
                  ['Threads', calibEst?.threads ?? '—'],
                  ['Precision', calibEst?.precision || '—'],
                  ['Learned combos', calibEst?.store?.entries ?? 0],
                  ['Total runs logged', calibEst?.store?.global_samples ?? 0],
                ].map(([k, v]) => (
                  <div key={k} className="flex flex-col leading-tight min-w-0">
                    <span className="text-white/35">{k}</span>
                    <span className="text-white/80 tabular-nums truncate" title={String(v)}>{v}</span>
                  </div>
                ))}
              </div>
              <div className="mt-3 pt-3 border-t border-white/5 text-micro text-white/45 leading-snug">
                {estLearned
                  ? 'Learned from your completed runs with these settings. Accuracy improves as you process more.'
                  : calibEst?.source === 'global'
                    ? 'No history for this exact settings + face-density combo yet — showing a blend of your overall average and the heuristic. Finish a run to calibrate it.'
                    : 'Heuristic estimate. Finish a run with these settings to start learning the real speed.'}
              </div>
            </div>
            </TiltCard>
          )}

          {/* Batch queue. It used to be hidden mid-run unless a queue was what
              was running — the queue-level Stop had to stay reachable, or the
              next job would start after stopping the current one. Nothing is
              hidden mid-run any more, so that special case is gone with it. */}
          <div>
            <QueuePanel
              q={queue}
              onAddCurrent={addToQueue}
              onLoadJobSettings={loadJobSettings}
              canAdd={targets.length > 0 && sourceFaces.length > 0}
              notify={notify}
            />
          </div>

          <Section title="Output settings & renders">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
              <div className="space-y-4">
                <Select label="Output method" value={p.output_method} onChange={(v) => set('output_method', v)} options={meta.output_methods} />
              </div>
              {out?.path && (
                <div className="space-y-2">
                  <div className="text-xs text-[var(--text-muted)]">Latest output</div>
                  {out.kind === 'video'
                    ? <video src={outUrl} controls className="w-full rounded-xl border border-white/5" />
                    : <img src={outUrl} alt="output" className="w-full rounded-xl border border-white/5" />}
                  <div className="flex flex-wrap gap-2">
                    <a href={outUrl} download
                      className="inline-block px-3 py-1.5 rounded-xl text-sm bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white font-bold transition-colors">⬇ Download</a>
                    <Button size="sm" variant="secondary" onClick={revealOutput}>Open folder</Button>
                  </div>
                  <QualityReport outputPath={out.path} notify={notify} />
                </div>
              )}
            </div>
          </Section>

        </div>
      </div>

      {/* Paste routing dialog */}
      {pastedFiles && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center animate-slide-up">
          <Card className="p-6 max-w-md w-full border border-white/10 shadow-2xl flex flex-col gap-4 text-center">
            <h3 className="text-lg font-bold text-white">Clipboard/Dropped File</h3>
            <p className="text-sm text-white/60">Would you like to load <span className="font-semibold text-white">{pastedFiles[0]?.name}</span> as a Source Face or Target Media?</p>
            <div className="flex gap-3 justify-center mt-2">
              <Button variant="primary" onClick={() => { onAddSource(pastedFiles); setPastedFiles(null); }}>Source Face</Button>
              <Button variant="secondary" onClick={() => { onAddTarget(pastedFiles); setPastedFiles(null); }}>Target Media</Button>
              <Button variant="stop" onClick={() => setPastedFiles(null)}>Cancel</Button>
            </div>
          </Card>
        </div>
      )}

      {/* Drag-n-Drop overlay removed (managed globally by App.jsx) */}

      {/* Keyboard Shortcuts HUD */}
      {showShortcutHUD && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-md z-50 flex items-center justify-center animate-slide-up" onClick={() => setShowShortcutHUD(false)}>
          <Card className="p-6 max-w-lg w-full border border-white/10 shadow-2xl flex flex-col gap-4" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">Pro Keyboard Shortcuts</h3>
              <Button size="sm" variant="ghost" onClick={() => setShowShortcutHUD(false)}>✕</Button>
            </div>
            <div className="grid grid-cols-2 gap-x-8 gap-y-5 py-2 text-sm text-white/80">
              <div className="space-y-2.5">
                <h4 className="font-bold text-[var(--accent)] text-xs uppercase tracking-wider">Playback & Nav</h4>
                <div className="flex items-center justify-between"><span className="text-white/60">Play / Pause</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Space</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Prev / Next Frame</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">← / →</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Step 10 Frames</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Shift + ← / →</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Jump to Start/End</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Home / End</kbd></div>
              </div>
              <div className="space-y-2.5">
                <h4 className="font-bold text-[var(--accent)] text-xs uppercase tracking-wider">Timeline Trimming</h4>
                <div className="flex items-center justify-between"><span className="text-white/60">Set Start Frame</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">[</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Set End Frame</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">]</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Reset Range</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">R</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Marker at playhead</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">M</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Zoom timeline</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Wheel on track</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Pan timeline</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Shift + Wheel</kbd></div>
              </div>
              <div className="space-y-2.5">
                <h4 className="font-bold text-[var(--accent)] text-xs uppercase tracking-wider">Compare & Zoom</h4>
                <div className="flex items-center justify-between"><span className="text-white/60">Zoom In / Out</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">+ / -</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Toggle Comparison</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">C</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Toggle Split View</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">S</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Blend / slider mode</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">O</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Split axis</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">X</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Auto-swipe</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">A</kbd></div>
              </div>
              <div className="space-y-2.5">
                <h4 className="font-bold text-[var(--accent)] text-xs uppercase tracking-wider">Preview Tools</h4>
                <div className="flex items-center justify-between"><span className="text-white/60">Magnifier lens</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">G</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Mask brush</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">B</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Keypoints &amp; pose</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">D</kbd></div>
              </div>
              <div className="space-y-2.5">
                <h4 className="font-bold text-[var(--accent)] text-xs uppercase tracking-wider">Queue & Process</h4>
                <div className="flex items-center justify-between"><span className="text-white/60">Add to Batch Queue</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Q</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Run Swapper / Queue</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Ctrl + Enter</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Toggle Shortcuts HUD</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">?</kbd></div>
              </div>
            </div>
            <p className="text-xs text-white/40 text-center mt-2">Click anywhere outside this modal or press Esc to close.</p>
          </Card>
        </div>
      )}

      {/* Floating Action Dock HUD */}
      <FloatingActionDock
        workspaceMode={workspaceMode}
        setWorkspaceMode={setWorkspaceMode}
        isRendering={!!progress.processing}
        onStartSwap={start}
        onCancelSwap={stop}
        progress={Math.round((progress.progress || 0) * 100)}
        onPreview={() => refreshPreview({ force: true })}
        previewing={previewing}
        ambilightEnabled={ambilightEnabled}
        setAmbilightEnabled={setAmbilightEnabled}
        onOpenPopout={() => popoutManager.openPopout(previewSrc || rawUrl)}
        onOpenPresetStudio={() => setShowPresetStudio(true)}
        drawers={drawers}
        setDrawers={setDrawers}
      />

      {/* Preset Studio & Recipe Manager Modal */}
      <PresetStudioModal
        isOpen={showPresetStudio}
        onClose={() => setShowPresetStudio(false)}
        activeParams={p}
        onApplyRecipe={(newParams) => {
          Object.entries(newParams).forEach(([k, v]) => set(k, v));
        }}
        onExportRecipe={exportRecipe}
        onImportRecipe={importRecipe}
        notify={notify}
      />

    </div>
  );
}
