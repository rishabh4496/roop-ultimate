import OutputVideoPlayer from './OutputVideoPlayer';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { API, timeoutFor } from '../api';
import { useApi } from '../useApi';
import { Section, Select, Slider, Toggle, TextInput, Button, FaceGallery, Card } from './ui';
import { confirmDialog } from './confirm';
import { Icon } from '../icons';
import PersonGroups from './PersonGroups';
import QualityReport from './QualityReport';
import FileDrop from './faceswap/FileDrop';
import MouthCustodyPanel from './faceswap/MouthCustodyPanel';
import useTargetAudio from './faceswap/useTargetAudio';
import { modeOf, overridePatch, requestFields, clampOffset, MODE_LABELS } from './faceswap/mouthCustody';
import PipelineControls from './faceswap/PipelineControls';
import {
  buildFaceMappingArray,
  buildTargetSelectionState,
  defaultSourceIndexForTarget,
  stableTargetSourceMapping,
  mappingObjectFromArray,
  remapSourceMappingAfterRemoval,
  remapSourceMappingAfterMove,
  targetPersonRecords,
} from './faceswap/faceMapping';
import { faceRole, excludePerson, includePerson } from './faceswap/faceRoles';
import {
  buildProcessingSelection as buildCanonicalSelection,
  classifyPreviewResponse,
  nextSelectionVersion,
  reconcileRestoredSelection,
  selectionIdentity,
} from './faceswap/processingSelection';
import ComparisonGridPanel from './faceswap/ComparisonGridPanel';
import ABReferencePanel from './faceswap/ABReferencePanel';
import ParserRegions from './faceswap/ParserRegions';
import InteractivePreview from './faceswap/InteractivePreview';
import useQueue, {
  ACTIVE_STATES, QUEUE_STATE_CLASS, QUEUE_STATE_LABEL, RETRYABLE_STATES, jobState,
} from './faceswap/useQueue';
import QueuePanel from './faceswap/QueuePanel';
import ProjectsPanel from './faceswap/ProjectsPanel';
import useSegments from './faceswap/useSegments';
import { useRenderLog, useSceneOverrides } from './faceswap/timeline/useSceneOverrides';
import { canBypass, overrideAt, overridePieces, patchPayload, patchSettings } from './faceswap/timeline/overrides';
import { fingerprint, sceneAt, scenesFromCuts } from './faceswap/timeline/laneModel';
import { nextWarmth } from './faceswap/faceContextModel';
import SegmentBar from './faceswap/SegmentBar';
import SliderTrackerBar from './faceswap/SliderTrackerBar';
import Timeline from './faceswap/Timeline';
import FacesetLibrary from './faceswap/FacesetLibrary';
import IdentityBlender from './faceswap/IdentityBlender';
import useIdentityKeyframes from './faceswap/useIdentityKeyframes';
import useParameterCurves from './faceswap/curves/useParameterCurves';
import { applyCurveValues, morphToIdentityKeyframes, patchSettingsWithCurves } from './faceswap/curves/curveModel';
import { bakeCurvePieces, describeBake, piecesHaveCurves } from './faceswap/curves/curvePieces';
import { EMPTY_BLEND, fetchIdentityBlend, normalizeBlend } from './faceswap/identityBlend';
import AmbilightGlow from './faceswap/AmbilightGlow';
import FloatingActionDock from './faceswap/FloatingActionDock';
import PipelinePill from './faceswap/PipelinePill';
import { ModularPipelineFlowStrip } from './faceswap/PipelinePill';
import ParameterSweepModal from './faceswap/ParameterSweepModal';
import MediaTabSessionBar from './faceswap/MediaTabSessionBar';
import PresetStudioModal from './faceswap/PresetStudioModal';
import { popoutManager } from './faceswap/PopoutPreviewManager';
import { setLastPreview } from './faceswap/lastPreview';
import { num, fmtTime } from './faceswap/utils';
import useProfiles from './faceswap/useProfiles';
import { useSystemTelemetryPoller } from './faceswap/useTelemetry';
import { FACESWAP_DEFAULTS } from './faceswap/defaults';
import { addPin, makePin, removePin, serializePins, trackLabel } from './faceswap/identityLock';
import { anchorsOf, serialize as serializeRotoZones } from './faceswap/rotoModel';
import InspectorGroup, { Disclosure } from './faceswap/InspectorGroup';
import RecipePresetBar from './faceswap/RecipePresetBar';
import LinkedPaddingControl from './faceswap/LinkedPaddingControl';
import EnhancerPipelineStack from './faceswap/EnhancerPipelineStack';
import TemporalModeControl from './faceswap/TemporalModeControl';
import { QUICK_RECIPES, activeRecipeId } from './faceswap/recipes';
import { temporalModeOf } from './faceswap/temporalModes';
import HeadPoseHud from './faceswap/HeadPoseHud';
import useThrottledFrameRequest from './faceswap/useThrottledFrameRequest';
import FrameUnavailable from './faceswap/FrameUnavailable';
import {
  dataUrlToOwnedBlobUrl, releaseOwner, revokeUrl, blobUrlToDataUrl,
} from './faceswap/objectUrls';
import useCompareGrid from './faceswap/useCompareGrid';
import useRenderLite from './faceswap/useRenderLite';
import useLiveCam from './faceswap/useLiveCam';
import useClipAdvisor, { ADVISOR_LABELS, fmtAdviceVal } from './faceswap/useClipAdvisor';
import useRuntimeEstimate from './faceswap/useRuntimeEstimate';
import useUserDefaults from './faceswap/useUserDefaults';
import useViewPersistence from './faceswap/useViewPersistence';
import { useCoalescedPreviewController } from './faceswap/coalescedPreviewController';
import usePlaybackBuffer from './faceswap/usePlaybackBuffer';
import useGridPreviewLoader from './faceswap/useGridPreviewLoader';
import useWorkspaceLayout from './faceswap/useWorkspaceLayout';
import { TRACKER_DEFAULT_VALUES, TRACKER_BYPASS_VALUES } from './faceswap/trackerConfig';
import { TiltCard } from '../motion';
import { LiveText } from './LiveTelemetry';
import { selectProg, etaMsOf, useTelemetryStore } from '../store/telemetryStore';
import { outputMediaUrl, outputSource } from './outputUrl';
import { useFrameSocketHold } from '../transport/useFrameSocket';
import { logFailure, reportFailure } from '../failureLog';
import { useHistoryStore } from '../store/historyStore';
import { useSlotStore, SLOT_IDS } from '../store/slotStore';
import { useHistoryChannel, MASK_CHANNEL, ROTO_CHANNEL, PINS_CHANNEL } from '../history/historyChannels';
import useSnapshotSlots from './faceswap/useSnapshotSlots';
import SnapshotSlots from './faceswap/SnapshotSlots';
import { decodeFrame } from './faceswap/frameDecoder';

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
  progress, setProgress, startTime, setStartTime, onOpenProcessing, onStopRun
}) {
  const { postJSON, getJSON, postFiles, postFile } = useApi();
  const [sourceFaces, setSourceFaces] = useState([]);
  const [sourceFacesInfo, setSourceFacesInfo] = useState([]);
  // Identity Blender recipe. Rides inside every preview/swap payload, so the
  // preview cache keys on it and a queued job freezes it. Rehydrated from the
  // backend's live global on mount (a Pinokio tab switch reloads this app).
  const [identityBlend, setIdentityBlend] = useState(EMPTY_BLEND);
  useEffect(() => {
    let live = true;
    // The keyframes are per clip and come from localStorage (useIdentityKeyframes), not from the backend's
    // live global (which holds whichever clip was rendered last), so the rehydrate keeps what is already here.
    fetchIdentityBlend().then((r) => { if (live && r) setIdentityBlend((prev) => ({ ...normalizeBlend(r), keyframes: prev.keyframes })); /* setIdentityBlend(normalizeBlend(r)) */ });
    return () => { live = false; };
  }, []);
  const [targetFaces, setTargetFaces] = useState([]);
  const [targetGroups, setTargetGroups] = useState([]);
  const [targetNames, setTargetNames] = useState([]);
  const [targetFacesInfo, setTargetFacesInfo] = useState([]);
  const [targetPersonIds, setTargetPersonIds] = useState([]);
  const [targetReferenceFaceIds, setTargetReferenceFaceIds] = useState([]);
  const [selectedTargetPersonId, setSelectedTargetPersonId] = useState(null);
  const [selectedReferenceFaceId, setSelectedReferenceFaceId] = useState(null);
  const [targets, setTargets] = useState([]);
  const [selSource, setSelSource] = useState(0);
  const [selTarget, setSelTarget] = useState(0);
  // Legacy state name retained for component compatibility. This is the
  // selected reference-angle index only; target person identity is carried by
  // selectedTargetPersonId and target-media identity by activeTargetMediaId.
  const [selTargetFace, setSelTargetFace] = useState(0);
  const [frame, setFrame] = useState(1);
  const [maxFrames, setMaxFrames] = useState(1);
  const [previewSrc, setPreviewSrc] = useState('');
  // The same value, readable synchronously. The revoke decision above has to
  // be made against what state currently HOLDS, not against the value this
  // render closed over — refreshPreview is async and re-entrant.
  const previewSrcRef = useRef('');
  // Which view `previewSrc` was actually rendered for, as `${index}_${frame}`.
  // A render lags the playhead by however long the swap takes, so without this
  // the stage had no way to tell "the swap for the frame you are looking at"
  // from "the swap for the frame you just left" and happily showed the latter.
  const [previewFor, setPreviewFor] = useState('');
  const [previewFaces, setPreviewFaces] = useState([]);
  const [previewPersonIds, setPreviewPersonIds] = useState([]);
  // A detection index is only valid for the exact target media/frame that
  // produced the overlay. It is intentionally not part of the target-person
  // context and is cleared whenever either input changes.
  const [selectedDetectedFace, setSelectedDetectedFace] = useState(null);
  const [previewKps, setPreviewKps] = useState([]);
  const [previewAnchors, setPreviewAnchors] = useState([]);   // [nose, chin] per face, for the Pen tool's pin
  // (yaw, pitch, roll) per face, solved server-side by the SAME solve_pose_5pt
  // the non-frontal mask router gates on — see the
  // preview endpoint for why it is not re-derived here.
  const [previewPose, setPreviewPose] = useState([]);
  const [previewDiagnostic, setPreviewDiagnostic] = useState(null);
  const [previewSar, setPreviewSar] = useState(1.0);
  const [previewDar, setPreviewDar] = useState(0.0);
  const [previewColorBadge, setPreviewColorBadge] = useState('');
  // Manual mask painted in the preview box (PNG data URL) + the face keypoints
  // it was painted against. maskVersion is a cheap token for the preview cache
  // key so the multi-KB data URL never has to be stringified per render.
  const [manualMask, setManualMask] = useState(null);
  const [maskRefKps, setMaskRefKps] = useState(null);
  const [maskVersion, setMaskVersion] = useState(0);
  const maskVersionSeq = useRef(0);
  // Single-frame AI-upscale spot-check (result shown in a modal).
  const [upscaledSrc, setUpscaledSrc] = useState('');
  const [upscaledDims, setUpscaledDims] = useState(null);
  const [upscaling, setUpscaling] = useState(false);
  const [fakePreview, setFakePreview] = useState(true);
  const [soloStage, setSoloStage] = useState(null);
  const soloStageRef = useRef(null);
  soloStageRef.current = soloStage;
  const [bypassedStages, setBypassedStages] = useState({});
  const [showFlowStrip, setShowFlowStrip] = useState(true);
  const [showSweepModal, setShowSweepModal] = useState(false);

  const handleSoloStage = useCallback((stage) => {
    const next = stage === soloStageRef.current ? null : stage;
    setSoloStage(next);
    soloStageRef.current = next;
    setTimeout(() => {
      if (refreshPreviewRef.current) {
        refreshPreviewRef.current({ force: true, solo_stage: next });
      }
    }, 0);
  }, []);

  const handleToggleBypass = useCallback((stageId) => {
    setBypassedStages((prev) => ({ ...prev, [stageId]: !prev[stageId] }));
  }, []);
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

  // The floating dock covers the bottom of the viewport on this tab. Tabbing to a
  // control, or scrollIntoView, scrolls it only to the viewport edge -- i.e. under
  // the dock -- so keep a bottom inset on the scroller while Face Swap is mounted
  // (WCAG 2.4.11, focus not obscured). Restored on unmount: other tabs have no dock.
  useEffect(() => {
    const root = document.documentElement;
    const prev = root.style.scrollPaddingBottom;
    root.style.scrollPaddingBottom = '104px';
    return () => { root.style.scrollPaddingBottom = prev; };
  }, []);

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
  // System telemetry: the poll runs for as long as this panel is mounted, but
  // the panel itself only subscribes to the thread count (for the estimate);
  // the HUD that shows GPU/VRAM/CPU is its own subscriber (SystemTelemetryHud),
  // so a 3-second reading no longer re-renders everything here.
  useSystemTelemetryPoller();
  const telemetryThreads = useTelemetryStore((s) => s.system?.threads);

  // Target-to-Source visual mapping state
  const [faceMapping, setFaceMapping] = useState({});
  const [sceneCuts, setSceneCuts] = useState([]);

  // Target media is an ordered view, not an identity. Keep the complete UI
  // context keyed by the backend-issued media id so A -> B -> A restores A's
  // people, angles, names, mapping, selected reference, frame, and preview.
  const targetContextsRef = useRef({});
  const targetIdAt = (index = selTarget) => targets[index]?.media_id
    || targets[index]?.id || null;
  const activeTargetMediaId = targetIdAt(selTarget);
  const selectedDetectedFaceIndex = selectedDetectedFace
    && selectedDetectedFace.mediaId === activeTargetMediaId
    && selectedDetectedFace.frame === frame
    ? selectedDetectedFace.index : null;

  useEffect(() => {
    setSelectedDetectedFace(null);
  }, [activeTargetMediaId, frame]);

  useEffect(() => {
    if (selectedDetectedFaceIndex !== null
        && selectedDetectedFaceIndex >= previewFaces.length) {
      setSelectedDetectedFace(null);
    }
  }, [previewFaces, selectedDetectedFaceIndex]);

  const selectDetectedFace = (faceIndex) => {
    if (!activeTargetMediaId || !Number.isInteger(faceIndex)
        || faceIndex < 0 || faceIndex >= previewFaces.length) return;
    setSelectedDetectedFace({
      mediaId: activeTargetMediaId,
      frame,
      index: faceIndex,
    });
  };

  const rememberTargetContext = (mediaId = activeTargetMediaId, patch = {}) => {
    if (!mediaId) return;
    targetContextsRef.current[mediaId] = {
      ...(targetContextsRef.current[mediaId] || {}),
      targetFaces,
      targetGroups,
      targetNames,
      targetFacesInfo,
      targetPersonIds,
      targetReferenceFaceIds,
      selectedTargetPersonId,
      selectedReferenceFaceId,
      faceMapping,
      selTargetFace,
      frame,
      previewSrc,
      previewFor,
      selSource,
      ...patch,
    };
  };

  const applyTargetContext = (res, mediaId = res?.target_media_id) => {
    if (!mediaId) return;
    if (Array.isArray(res?.scene_cuts)) {
      setSceneCuts(res.scene_cuts);
    }
    const saved = targetContextsRef.current[mediaId] || {};
    const faces = res?.target_faces ?? saved.targetFaces ?? [];
    const groups = res?.target_groups ?? saved.targetGroups ?? [];
    const names = res?.target_names ?? saved.targetNames ?? [];
    const facesInfo = res?.target_faces_info ?? saved.targetFacesInfo ?? [];
    const personIds = res?.target_person_ids ?? saved.targetPersonIds ?? [];
    const referenceIds = res?.target_reference_face_ids
      ?? saved.targetReferenceFaceIds ?? [];
    const persistedMapping = res?.face_mapping ?? saved.faceMapping ?? {};
    // Stage 14: whatever was remembered client-side is reconciled against the
    // ids the backend just returned. A person, angle or mapping entry that no
    // longer exists there is dropped — a reload can never resurrect it. The
    // backend's own selected ids win when they are valid; the client memory
    // is only a fallback for ids the backend still knows.
    const reconciled = reconcileRestoredSelection({
      savedPersonId: saved.selectedTargetPersonId,
      savedReferenceId: saved.selectedReferenceFaceId,
      savedMapping: Array.isArray(persistedMapping)
        ? mappingObjectFromArray(persistedMapping) : persistedMapping,
      personIds, referenceIds,
      sourceIdentityIds: res?.source_faces_info
        ? res.source_faces_info.map((info, index) => info?.id || `memory-slot-${index}`)
        : null,
      serverPersonId: res?.selected_target_person_id,
      serverReferenceId: res?.selected_reference_face_id,
    });
    const restoredMapping = reconciled.faceMapping;
    const fallbackFace = Number(res?.selected_target_face_index ?? saved.selTargetFace ?? 0);
    const selectedPerson = reconciled.selectedTargetPersonId
      ?? (personIds[fallbackFace] || null);
    const selectedReference = reconciled.selectedReferenceFaceId
      ?? (referenceIds[fallbackFace] || null);
    const selectedFace = Number.isInteger(res?.selected_target_face_index)
      ? res.selected_target_face_index
      : (Number.isInteger(saved.selTargetFace) ? saved.selTargetFace : 0);
    const boundedFace = Math.max(0, Math.min(selectedFace, Math.max(0, faces.length - 1)));
    const targetIndex = Number.isInteger(res?.selected_target_index)
      ? res.selected_target_index
      : targets.findIndex((target) => (target.media_id || target.id) === mediaId);
    const availableSourceInfo = Array.isArray(res?.source_faces_info)
      ? res.source_faces_info : sourceFacesInfo;
    const availableSourceCount = Math.max(sourceFaces.length, availableSourceInfo.length);
    const serverSourceIndex = res?.target_selected_source_id
      ? availableSourceInfo.findIndex((info) => String(info?.id || '')
        === String(res.target_selected_source_id)) : -1;
    const storedSourceIndex = Number.isInteger(saved.selSource) ? saved.selSource
      : serverSourceIndex >= 0 ? serverSourceIndex : undefined;
    const restoredSelSource = storedSourceIndex !== undefined
      && storedSourceIndex >= 0 && storedSourceIndex < availableSourceCount
      ? storedSourceIndex
      : defaultSourceIndexForTarget(targetIndex, availableSourceCount, selSource);
    setTargetFaces(faces);
    setTargetGroups(groups);
    setTargetNames(names);
    setTargetFacesInfo(facesInfo);
    setTargetPersonIds(personIds);
    setTargetReferenceFaceIds(referenceIds);
    setSelectedTargetPersonId(selectedPerson);
    setSelectedReferenceFaceId(selectedReference);
    setFaceMapping(restoredMapping || {});
    setSelTargetFace(boundedFace);
    const resultTarget = res?.targets?.[res?.selected_target_index ?? selTarget]
      || targets[res?.selected_target_index ?? selTarget];
    const max = resultTarget?.frames || maxFrames || 1;
    const restoredFrame = Number.isInteger(saved.frame) && saved.frame >= 1
      ? Math.min(saved.frame, max) : 1;
    setFrame(restoredFrame);
    setPreviewSrc(saved.previewSrc || '');
    setPreviewFor(saved.previewFor || '');
    setSelSource(restoredSelSource);
    // Keep the backend's target-scoped source selection in step with the
    // preview. This is also what makes a webview reload restore target 2's
    // faceset instead of falling back to the globally selected source.
    postJSON('/api/source/select', { index: restoredSelSource }).catch(logFailure('Saving the source selection for this target', { toast: false }));
    targetContextsRef.current[mediaId] = {
      ...saved,
      targetFaces: faces,
      targetGroups: groups,
      targetNames: names,
      targetFacesInfo: facesInfo,
      targetPersonIds: personIds,
      targetReferenceFaceIds: referenceIds,
      selectedTargetPersonId: selectedPerson,
      selectedReferenceFaceId: selectedReference,
      selTargetFace: boundedFace,
      frame: restoredFrame,
      faceMapping: restoredMapping || {},
      previewSrc: saved.previewSrc || '',
      previewFor: saved.previewFor || '',
      selSource: restoredSelSource,
    };
  };

  // Keep the ref current for changes made inside PersonGroups (mapping/name/
  // angle operations) before a target switch can serialize the context.
  // `rememberTargetContext` intentionally closes over the current context;
  // memoizing it would make the per-target snapshot lag one render behind.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!activeTargetMediaId) return;
    rememberTargetContext();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTargetMediaId, targetFaces, targetGroups, targetNames,
    targetFacesInfo, targetPersonIds, targetReferenceFaceIds,
    selectedTargetPersonId, selectedReferenceFaceId, faceMapping,
    selTargetFace, frame, previewSrc, previewFor, selSource]);

  // Single source of truth, shared with the PersonGroups dropdown so the row
  // the user reads and the payload the backend receives cannot disagree.
  const getFaceMappingArray = (params = p, selectedSource = selSource) => buildFaceMappingArray({
    targetGroups,
    faceMapping,
    sourceCount: sourceFaces.length,
    faceSelection: params.face_detection_mode,
    selTargetFace,
    selectedSource,
    targetPersonIds,
    selectedTargetPersonId,
    selectedReferenceFaceId,
    sourceIdentityIds: sourceFacesInfo.map((info, index) => info?.id || `memory-slot-${index}`),
  });

  const getStableTargetSourceMapping = (params = p, selectedSource = selSource) => stableTargetSourceMapping({
    targetGroups,
    targetPersonIds,
    faceMapping,
    sourceCount: sourceFaces.length,
    sourceIdentityIds: sourceFacesInfo.map((info, index) => info?.id || `memory-slot-${index}`),
    faceSelection: params.face_detection_mode,
    selectedTargetPersonId,
    selectedSource,
  });

  const getTargetSelectionState = (params = p, selectedSource = selSource,
    targetMediaIndex = selTarget) => buildTargetSelectionState({
    faceSelection: params.face_detection_mode,
    targetGroups,
    faceMapping,
    sourceCount: sourceFaces.length,
    selTargetFace,
    selectedSource,
    targetReferenceIndex: selTargetFace,
    targetMediaIndex,
    targetPersonIds,
    selectedTargetPersonId,
    selectedReferenceFaceId,
    sourceIdentityIds: sourceFacesInfo.map((info, index) => info?.id || `memory-slot-${index}`),
  });

  const sourceNameAt = (index) => sourceFacesInfo[index]?.name
    || (sourceFaces[index] ? `Face ${index + 1}` : null);
  const sourceIdAt = (index) => sourceFacesInfo[index]?.id
    || (sourceFaces[index] ? `memory-slot-${index}` : null);

  // ── Stage 14: the canonical processing selection ─────────────────────────
  // `selectionVersion` is bumped IN RENDER (useMemo, not an effect) whenever
  // any identity-bearing input changes, so a payload built in the same render
  // as the change already carries the new version. `commitTargetContext` bumps
  // it synchronously too, so a write fired from a click handler is ordered
  // before the preview that follows the re-render.
  const selectionVersionRef = useRef(0);
  const bumpSelectionVersion = () => {
    selectionVersionRef.current = nextSelectionVersion(selectionVersionRef.current);
    return selectionVersionRef.current;
  };
  // `settings` (the prop), not `p`: `p` is declared further down the body and
  // a dependency array is evaluated as the body runs — naming it here would be
  // a temporal-dead-zone throw on first render that neither the build nor
  // oxlint reports.
  const selectionVersion = useMemo(bumpSelectionVersion,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [activeTargetMediaId, selectedTargetPersonId, selectedReferenceFaceId, selSource,
      faceMapping, settings?.face_detection_mode, targetPersonIds, targetReferenceFaceIds,
      targetGroups, sourceFacesInfo]);

  // One object, built from one snapshot of the UI, sent by preview, swap and
  // queue alike. Nothing downstream may re-derive any of these fields.
  const buildProcessingSelection = (params = p, {
    requestId = null,
    selectedSource = selSource,
    targetMediaId = activeTargetMediaId,
    targetMediaIndex = selTarget,
  } = {}) => buildCanonicalSelection({
    targetMediaId,
    targetGroups, targetPersonIds, faceMapping,
    sourceCount: sourceFaces.length,
    sourceIdentityIds: sourceFacesInfo.map((info, index) => info?.id || `memory-slot-${index}`),
    faceSelection: params.face_detection_mode,
    selectedTargetPersonId, selectedReferenceFaceId,
    selectedSource,
    selectionState: getTargetSelectionState(params, selectedSource, targetMediaIndex),
    selectionVersion,
    requestId,
  });

  // The single write path for target-scoped identity state (selected person /
  // reference angle / mapping). Carries a fresh selection_version so the
  // backend can drop any OLDER write that lands after it — e.g. a preview built
  // before the click that finishes after this.
  const commitTargetContext = async (patch = {}, mediaId = activeTargetMediaId) => {
    if (!mediaId) return null;
    try {
      return await postJSON('/api/target/context', {
        target_media_id: mediaId,
        selection_version: bumpSelectionVersion(),
        ...patch,
      });
    } catch (e) {
      notify(e.message, 'error');
      return null;
    }
  };

  const getSourceMappingNames = (params = p, selectedSource = selSource) => getFaceMappingArray(params, selectedSource)
    .map((sourceIndex) => sourceIndex >= 0 ? sourceNameAt(sourceIndex) : null);
  const getSourceMappingIds = (params = p, selectedSource = selSource) => getFaceMappingArray(params, selectedSource)
    .map((sourceIndex) => sourceIndex >= 0 ? sourceIdAt(sourceIndex) : null);

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
  // The timeline's per-scene overrides and its record of finished renders, per clip. The overrides
  // reach a render the same way segments do (one queued job per piece); see timeline/overrides.js.
  const timelineKey = activeTargetMediaId || targets[selTarget]?.name || String(selTarget);
  const sceneOverrides = useSceneOverrides(timelineKey, maxFrames);
  const renderLog = useRenderLog(timelineKey);
  // Stage 25: parameter curves (keyframed settings), per clip. The preview reads them exactly; a render reads them as
  // pieces, the way it reads scene overrides. `curveOpen` is here, not in the Timeline, because the C key is owned by
  // this component's shortcut handler.
  const curves = useParameterCurves(timelineKey, maxFrames);
  const [curveOpen, setCurveOpen] = useState(false);
  // Stage 20: identity keyframes ride inside the Identity Blender recipe (so previews, the cache strip and queued
  // jobs all see them) and are stored per clip like the markers and overrides beside them.
  const identityKeyframes = useIdentityKeyframes(timelineKey, identityBlend, setIdentityBlend, maxFrames);
  const identityTrack = React.useMemo(() => ({
    ...identityKeyframes,
    recipe: identityBlend,
    components: identityBlend.components,
    labelOf: (id) => sourceFacesInfo.find((info) => info?.id === id)?.name || 'Not in gallery',
  }), [identityKeyframes, identityBlend, sourceFacesInfo]);

  // Pasted Files Dialog State
  const [pastedFiles, setPastedFiles] = useState(null);

  // Preview Cache Ref.
  //
  // Entries hold a blob URL, not the base64 data URL the backend sends. The
  // difference is not cosmetic: this cache is capped at 200 entries, and a
  // rendered HD/4K preview is 1-4 MB of base64 — so as data URLs it was a ref
  // holding up to ~800 MB of JavaScript STRING, re-read on every cache lookup
  // and copied by every consumer that touched it. As blob URLs the bytes live
  // once in the browser's blob store and each entry is a ~50-character token.
  //
  // The whole cost of that is that blob URLs must be revoked BY NAME or they
  // outlive the document — so eviction and clearing both go through
  // objectUrls.js, which owns them against `previewOwner` and can free the lot
  // in one call. An entry dropped without a revoke is a permanent leak that no
  // GC will ever collect, which is exactly the trade this makes explicit.
  const previewCacheRef = useRef({});
  const previewOwner = useRef({ tag: 'preview-cache' }).current;
  // Set when a preview refresh is skipped because a swap run owns the GPU.
  const previewDeferredRef = useRef(false);

  // Every blob URL that is CURRENTLY ON SCREEN — the stage, plus any comparison
  // grid cell. Recomputed each render, which is cheap (at most ~17 short
  // strings) and, crucially, always current at the moment the cache is cleared.
  const displayedUrlsRef = useRef(new Set());
  displayedUrlsRef.current = new Set([
    previewSrc,
    ...Object.values(enhancerPreviews || {}),
    ...Object.values(maskPreviews || {}),
    ...Object.values(swapperPreviews || {}),
    ...Object.values(upscalePreviews || {}),
  ].filter(Boolean));

  // Dropping the cache must free its blobs — but NOT the ones something is
  // still showing. `releaseOwner` would take the lot, and the caller of this is
  // a settings/faceset change that leaves the stage and any open grid up until
  // their replacements land: revoking underneath them blanks the picture with
  // no error anywhere, which is the hardest kind of regression to trace back to
  // a memory fix. What stays behind is bounded by what is visible, and the
  // unmount handler below sweeps it.
  const clearPreviewCache = () => {
    const cache = previewCacheRef.current;
    previewCacheRef.current = {};
    const displayed = displayedUrlsRef.current;
    for (const k in cache) {
      const url = cache[k]?.image;
      if (url && !displayed.has(url)) revokeUrl(url);
    }
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
    `${meta?.git_version || 'dev'}_${activeTargetMediaId || 'no-target'}_${sourceSig}_${targetSig}_${selSource}_${selTargetFace}`;

  const getCacheKey = (idx = selTarget, fr = frame, solo = soloStageRef.current) => {
    return `${targetIdAt(idx) || `legacy-index-${idx}`}_${fr}_${previewKey}_${cacheSuffix}${solo ? `_solo_${solo}` : ''}`;
  };

  // Is this url still referenced by a cache entry? Asked before freeing the
  // url that state is dropping — the cache and `previewSrc` routinely point at
  // the same blob, and revoking one out from under the other blanks the stage
  // on the next cache hit with no error anywhere.
  const cacheHoldsUrl = (url) => {
    if (!url) return false;
    const cache = previewCacheRef.current;
    for (const k in cache) if (cache[k]?.image === url) return true;
    return false;
  };

  const getCachedPreview = (idx = selTarget, fr = frame, solo = soloStageRef.current) => {
    const key = getCacheKey(idx, fr, solo);
    return previewCacheRef.current[key];
  };

  // Write under an explicit key — the key captured when a request was SENT —
  // so a response is always filed under the selection it was rendered for.
  const setCachedPreviewByKey = (key, data) => {
    const cache = previewCacheRef.current;
    const keys = Object.keys(cache);
    if (keys.length > 200) {
      // Evicting the entry is not enough — its blob URL would stay resident for
      // the life of the document. Free it here, where the reference is still in
      // hand; there is nowhere later that could.
      const victim = cache[keys[0]];
      if (victim?.image) revokeUrl(victim.image);
      delete cache[keys[0]];
    }
    // Replacing a key in place leaks the same way.
    if (cache[key] && cache[key].image && cache[key].image !== data.image) {
      revokeUrl(cache[key].image);
    }
    cache[key] = data;
  };

  // Last resort: the panel going away must not strand a cache full of blobs.
  // clearPreviewCache covers the in-app paths (a settings change, a new
  // faceset); this covers the tab switch, which is how this window usually
  // ends.
  useEffect(() => () => { releaseOwner(previewOwner); }, [previewOwner]);
  useEffect(() => () => {
    if (previewAbortCtrlRef.current) {
      try { previewAbortCtrlRef.current.abort(); } catch {}
      previewAbortCtrlRef.current = null;
    }
  }, []);

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
        target_person_source_mapping: getStableTargetSourceMapping(),
        target_person_ids: targetPersonIds,
        source_mapping_names: getSourceMappingNames(),
        source_mapping_ids: getSourceMappingIds(),
        selected_source_name: sourceNameAt(selSource),
        selected_source_id: sourceIdAt(selSource),
        selection_state: getTargetSelectionState(),
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

  const handleExportEdl = async () => {
    try {
      const currentTarget = targets[selTarget];
      const fps = currentTarget?.fps || 30.0;
      const totalFrames = currentTarget?.frames || 1800;
      const res = await postJSON('/api/timeline/export_edl', {
        document: {
          name: currentTarget?.name || 'Roop_Ultimate_Timeline',
          total_frames: totalFrames,
          timeline: {
            fps,
            frame_start: 0,
            frame_end: totalFrames,
            face_segments: [],
          },
        },
        fps,
      });
      if (res?.edl_text) {
        const blob = new Blob([res.edl_text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        const stem = (currentTarget?.name || 'roop_timeline').replace(/\.[^/.]+$/, '');
        link.download = `${stem}_TrackV2.edl`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        notify?.('Exported CMX3600 EDL timeline (Track V1 source, Track V2 face-swapped)');
      }
    } catch (err) {
      notify?.(`Failed to export EDL: ${err.message}`, 'error');
    }
  };

  const handleExportOtio = async () => {
    try {
      const currentTarget = targets[selTarget];
      const fps = currentTarget?.fps || 30.0;
      const totalFrames = currentTarget?.frames || 1800;
      const res = await postJSON('/api/timeline/export_otio', {
        document: {
          name: currentTarget?.name || 'Roop_Ultimate_Timeline',
          total_frames: totalFrames,
          timeline: {
            fps,
            frame_start: 0,
            frame_end: totalFrames,
            face_segments: [],
          },
        },
        fps,
      });
      if (res?.otio_json) {
        const blob = new Blob([res.otio_json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        const stem = (currentTarget?.name || 'roop_timeline').replace(/\.[^/.]+$/, '');
        link.download = `${stem}_TrackV2.otio.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        notify?.('Exported OpenTimelineIO JSON timeline (V1 source, V2 face-swap)');
      }
    } catch (err) {
      notify?.(`Failed to export OTIO: ${err.message}`, 'error');
    }
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
          // Preserve explicit skips and malformed entries as skips.  Importing
          // a null/invalid value must not make mapPerson fall back to source 0.
          const recipeNames = Array.isArray(r.source_mapping_names)
            ? r.source_mapping_names : [];
          const recipeIds = Array.isArray(r.source_mapping_ids)
            ? r.source_mapping_ids : [];
          const currentNames = sourceFacesInfo.map((info, index) =>
            info?.name || (sourceFaces[index] ? `Face ${index + 1}` : null));
          const currentIds = sourceFacesInfo.map((info, index) =>
            info?.id || (sourceFaces[index] ? `memory-slot-${index}` : null));
          const imported = r.face_mapping.map((value, index) => {
            const sourceId = recipeIds[index];
            if (recipeIds.length === r.face_mapping.length
                && sourceId != null && String(sourceId).trim() !== '') {
              return currentIds.findIndex((current) =>
                current && String(current).toLowerCase() === String(sourceId).toLowerCase());
            }
            const name = recipeNames[index];
            if (name == null || String(name).trim() === '') return -1;
            const match = currentNames.findIndex((current) =>
              current && String(current).toLowerCase() === String(name).toLowerCase());
            return match >= 0 ? match : -1;
          });
          const nextMapping = mappingObjectFromArray(
            (recipeIds.length === r.face_mapping.length
              || recipeNames.length === r.face_mapping.length)
              ? imported : r.face_mapping,
          );
          setFaceMapping(nextMapping);
          // Persist through the versioned write path, like a dropdown change,
          // so a reload restores the imported mapping rather than the old one.
          commitTargetContext({ face_mapping: nextMapping, target_person_source_mapping: nextMapping });
        }
        const savedSelection = r.selection_state;
        const savedReference = Number(savedSelection?.target_reference_index);
        if (Number.isInteger(savedReference) && savedReference >= 0
            && savedReference < targetFaces.length) {
          setSelTargetFace(savedReference);
        }
        if (r.selected_source_id || r.selected_source_name) {
          const matchById = r.selected_source_id
            ? sourceFacesInfo.findIndex((info, index) =>
              String(info?.id || `memory-slot-${index}`).toLowerCase()
                === String(r.selected_source_id).toLowerCase())
            : -1;
          const match = matchById >= 0 ? matchById : sourceFacesInfo.findIndex((info, index) =>
            String(info?.name || `Face ${index + 1}`).toLowerCase()
              === String(r.selected_source_name || '').toLowerCase());
          if (match >= 0) setSelSource(match);
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
  // `target_index` is deliberately absent: the queue resolves the target from
  // the immutable target_media_id at dispatch time, because a stored index goes
  // stale as soon as a target is removed.
  const buildSwapPayload = (params = p, {
    sourceIndex = selSource,
    targetMediaId = activeTargetMediaId,
    targetMediaIndex = selTarget,
  } = {}) => {
    const sp = withSliderBypass(params);
    const selectedSource = Number.isInteger(sourceIndex) ? sourceIndex : selSource;
    return {
      ...sp,
      enhancer: sp.selected_enhancer, adaptive_enhancer_profile: sp.adaptive_enhancer_profile || 'BALANCED', detection: sp.face_detection_mode,
      output_method: sp.output_method,
      upscale: sp.subsample_upscale, mask_engine: sp.mask_engine,
      mask_engine_2: sp.mask_engine_2, clip_text: sp.mask_clip_text,
      sam2_model_size: sp.sam2_model_size, track_identities: sp.track_identities,
      autorotate: sp.autorotate_faces,
      face_distance: num(sp.max_face_distance, 0.75), blend_ratio: num(sp.blend_ratio, 0.8),
      num_swap_steps: num(sp.num_swap_steps, 1),
      face_mapping: getFaceMappingArray(sp, selectedSource),
      target_person_source_mapping: getStableTargetSourceMapping(sp, selectedSource),
      target_person_ids: targetPersonIds,
      source_mapping_names: getSourceMappingNames(sp, selectedSource),
      source_mapping_ids: getSourceMappingIds(sp, selectedSource),
      selected_source_name: sourceNameAt(selectedSource),
      selected_source_id: sourceIdAt(selectedSource),
      target_media_id: targetMediaId,
      selection_state: getTargetSelectionState(sp, selectedSource, targetMediaIndex),
      processing_selection: buildProcessingSelection(sp, {
        selectedSource,
        targetMediaId,
        targetMediaIndex,
      }),
      selection_version: selectionVersion,
      imagemask: maskJson,
      identity_blend: identityBlend,
      roto_polygons: rotoPayload,
      identity_pins: pinsPayload,
    };
  };

  // Describe the current setup as a queue job. `extra` carries a segment
  // (frame_start/frame_end) when one range of several is being queued.
  const currentJob = (extra = {}) => {
    const payload = buildSwapPayload();
    return {
      target_name: targets[selTarget]?.name || '',
      target_media_id: activeTargetMediaId,
      source_index: selSource,
      source_name: sourceFacesInfo[selSource]?.name
        || (sourceFaces[selSource] ? `Face ${selSource + 1}` : 'Selected face'),
      source_id: sourceIdAt(selSource),
      payload,
      // Frozen at queue-creation time; the server stores it on the job and the
      // worker consumes exactly this (see app/routes_queue.py).
      processing_selection: payload.processing_selection,
      selection_version: payload.selection_version,
      request_id: payload.processing_selection?.request_id,
      ...extra,
    };
  };

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
    if (Number.isInteger(job.source_index) && job.source_index >= 0) {
      setSelSource(job.source_index);
    }
    const idx = job.target_media_id
      ? targets.findIndex((t) => (t.media_id || t.id) === job.target_media_id)
      : targets.findIndex((t) => t.name === job.target_name);
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
    const name = targets[selTarget]?.name || '';
    await queue.addMany(segments.segments.map((s, i) => currentJob({
      label: `${name} — segment ${i + 1}`,
      frame_start: s.start,
      frame_end: s.end,
    })));
    notify(`Queued ${segments.segments.length} segment${segments.segments.length === 1 ? '' : 's'} of "${name}"`);
  };

  // Finished segment jobs for THIS target, in timeline order — the join has to
  // concatenate them in clip order, not in the order they happened to render.
  const joinableJobs = React.useMemo(() => {
    const name = targets[selTarget]?.name;
    const mediaId = activeTargetMediaId;
    if (!name && !mediaId) return [];
    return queue.jobs
      .filter((j) => (mediaId ? j.target_media_id === mediaId : j.target_name === name)
                     && jobState(j) === 'COMPLETED'
                     && j.frame_start != null && (j.outputs || []).length > 0)
      .sort((a, b) => a.frame_start - b.frame_start);
  }, [queue.jobs, targets, selTarget, activeTargetMediaId]);

  const joinSegments = async () => {
    try {
      const res = await postJSON('/api/queue/join', { ids: joinableJobs.map((j) => j.id) }, { timeout: 0 });
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
    playStalled, playError, playbackSource, clearPlaybackFrame,
  } = usePlaybackBuffer({ frame, setFrame, selTarget, maxFrames, targets });
  // Playback stops itself after five failed chunks in a row (usePlaybackBuffer);
  // without a word the Play button would just pop back out.
  useEffect(() => {
    if (playError) notify('Playback stopped: the backend would not return frames. Press Play to try again.', 'error');
  }, [playError, notify]);
  // Keep /ws/frames connected while this panel is up, so pressing Play streams
  // over the socket from the first frame instead of first paying a handshake
  // (the buffer falls back to HTTP chunks whenever it is not open).
  useFrameSocketHold();

  const [isScrubbing, setIsScrubbing] = useState(false);
  const [dragType, setDragType] = useState('playhead'); // 'playhead', 'start', 'end'
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
  // Playhead drag, decoupled from this panel's render. This component owns ~4,000 lines of
  // JSX and every `setFrame` re-renders all of it, so committing the frame once per display
  // frame was ~92 React commits/s during a drag (react-ui/e2e/frame-budget.spec.js). The
  // playhead and its timecode are moved by the Timeline's imperative API at pointer rate
  // (playheadApiRef); `frame` itself is committed at most every SCRUB_COMMIT_MS, which is
  // all the preview can use anyway (useThrottledFrameRequest asks for a frame per 150 ms),
  // and once more, immediately, when the pointer is released.
  const playheadApiRef = useRef(null);
  const scrubCommitRef = useRef({ timer: null, pending: null, last: 0 });
  const [isDraggingScrub, setIsDraggingScrub] = useState(false);
  const scrubDebounceTimerRef = useRef(null);
  const [isGeneratingPreviewClip, setIsGeneratingPreviewClip] = useState(false);
  const [origStartEnd, setOrigStartEnd] = useState(null);


  const previewBusyRef = useRef(false);   // a /api/preview call is in flight
  const previewAbortCtrlRef = useRef(null); // active AbortController for in-flight preview
  const _coalescedPreviewCtrl = useCoalescedPreviewController();
  const previewPendingRef = useRef(null); // latest queued request while busy (coalesced)
  // The NEWEST render's refreshPreview. The coalesced request is dispatched
  // through this, never through the closure that happened to be in flight:
  // that closure still holds the person/source/settings of the render it was
  // created in, so it would rebuild the OLD request and the change the user
  // just made would never be rendered.
  const refreshPreviewRef = useRef(null);

  // p = the swap parameters, seeded from CFG (settings) and patched locally.
  const p = settings || {};
  // Stage 17: does the selected video have audio? Answers the panel's silent-video advisory and whether the timeline shows an
  // audio row at all. The envelope it asks for is the one the timeline's waveform already builds, so this costs nothing extra.
  const targetHasAudio = useTargetAudio(getJSON, selTarget, !!targets[selTarget] && targets[selTarget].frames > 1);
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
    // From a counter that only goes up, not from the current version: undo can put an older
    // version back, and `v + 1` from there would reuse a version a DIFFERENT mask already owns, so
    // the preview cache (keyed by `mask:<version>`) would serve that other mask's render.
    maskVersionSeq.current += 1;
    setMaskVersion(maskVersionSeq.current);
  };

  // ── Stage 11: Pen-tool zones and identity pins ───────────────────────────
  // Per CLIP, keyed by the target's media id: a zone drawn around a microphone in one video must never
  // ride into the next video's render (the brush mask above is not keyed and can; these are). The
  // request carries only the serialised form, and the backend CLEARS both lists on any request that
  // does not carry them, so what is sent is the whole truth for this render.
  const clipKey = activeTargetMediaId || `legacy-index-${selTarget}`;
  const [rotoByMedia, setRotoByMedia] = useState({});
  const [pinsByMedia, setPinsByMedia] = useState({});
  const [rotoSelectedId, setRotoSelectedId] = useState(null);

  // Stage 23: what this tab owns, reported to the undo history. The applied mask is three states
  // that change together, so it travels as one bundle (equal-by-fields, see MASK_CHANNEL).
  const maskBundle = useMemo(
    () => ({ dataUrl: manualMask, kps: maskRefKps, version: maskVersion }),
    [manualMask, maskRefKps, maskVersion],
  );
  useHistoryChannel('mask', maskBundle, (b) => {
    setManualMask(b.dataUrl);
    setMaskRefKps(b.kps);
    setMaskVersion(b.version);
  }, MASK_CHANNEL);
  useHistoryChannel('roto', rotoByMedia, setRotoByMedia, ROTO_CHANNEL);
  useHistoryChannel('pins', pinsByMedia, setPinsByMedia, PINS_CHANNEL);
  const NO_ITEMS = useMemo(() => [], []);
  const rotoZones = rotoByMedia[clipKey] || NO_ITEMS;
  const identityPins = pinsByMedia[clipKey] || NO_ITEMS;
  const setRotoZones = (next) => setRotoByMedia((all) => ({
    ...all, [clipKey]: typeof next === 'function' ? next(all[clipKey] || []) : next,
  }));
  const rotoPayload = useMemo(() => serializeRotoZones(rotoZones), [rotoZones]);
  const pinsPayload = useMemo(() => serializePins(identityPins), [identityPins]);

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
  const buildPreviewPayload = (params, {
    index, frame: fr, fake, target_media_id: requestedMediaId, ...overrides
  } = {}) => {
    const activeParams = withSliderBypass(params);
    // Respect the user's selected face detection mode. Do not coerce "Selected face"
    // to "All faces" when targetFaces is empty, as that causes unwanted automatic
    // swapping of random faces before the user chooses a target face.
    const previewDetection = activeParams.face_detection_mode;
    return {
      index, frame: fr, fake_preview: fake,
      enhancer: activeParams.selected_enhancer, adaptive_enhancer_profile: activeParams.adaptive_enhancer_profile || 'BALANCED', codeformer_fidelity: num(activeParams.codeformer_fidelity, 0.5), tile_diffusion_strength: num(activeParams.tile_diffusion_strength, 0.65), tile_diffusion_steps: num(activeParams.tile_diffusion_steps, 2), restoration_mode: activeParams.restoration_mode || "balanced",
      detection: previewDetection,
      face_distance: num(activeParams.max_face_distance, 0.75), blend_ratio: num(activeParams.blend_ratio, 0.8),
      identity_confidence_threshold: num(activeParams.identity_confidence_threshold, 0.0),
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
      ...requestFields(activeParams),
      lipsync_enabled: activeParams.lipsync_enabled,
      lipsync_model: activeParams.lipsync_model,
      lipsync_coarticulation_mode: activeParams.lipsync_coarticulation_mode,
      oral_cavity_restore: activeParams.oral_cavity_restore,
      jaw_coarticulation_strength: num(activeParams.jaw_coarticulation_strength, 0.6),
      lipsync_audio_source: activeParams.lipsync_audio_source,
      lipsync_audio_path: activeParams.lipsync_audio_path,
      restore_original_eyes: activeParams.restore_original_eyes,
      eyes_blend_amount: num(activeParams.eyes_blend_amount, 1),
      eyes_feather_blend: num(activeParams.eyes_feather_blend, 25),
      eyes_size_factor: num(activeParams.eyes_size_factor, 1),
      eyes_radius_x: num(activeParams.eyes_radius_x, 1),
      eyes_radius_y: num(activeParams.eyes_radius_y, 1),
      parser_regions: activeParams.parser_regions,
      glasses_frame_protect: activeParams.glasses_frame_protect,
      parser_region_grow: activeParams.parser_region_grow,
      merger_hist_match: num(activeParams.merger_hist_match, 0),
      merger_sharpen: num(activeParams.merger_sharpen, 0),
      merger_motion_blur: num(activeParams.merger_motion_blur, 0),
      merger_grain_match: num(activeParams.merger_grain_match, 0),
      merger_degrade: num(activeParams.merger_degrade, 0),
      merger_clarity: num(activeParams.merger_clarity, 0),
      merger_match_sharpness: activeParams.merger_match_sharpness !== false,
      merger_sharpness_headroom: num(activeParams.merger_sharpness_headroom, 0.7),
      output_face_scale: num(activeParams.output_face_scale, 0),
      expression_restore_strength: num(activeParams.expression_restore_strength, 0),
      expression_restore_region: activeParams.expression_restore_region || 'all',
      expression_gaze_follow: num(activeParams.expression_gaze_follow, 0),
      expression_blink_sync: !!activeParams.expression_blink_sync,
      pupil_gaze_retention: num(activeParams.pupil_gaze_retention, 0),
      micro_expression_restore: num(activeParams.micro_expression_restore, 0),
      inner_mouth_preserve: !!activeParams.inner_mouth_preserve,
      identity_lock: !!activeParams.identity_lock,
      identity_lock_margin: num(activeParams.identity_lock_margin, 0.08),
      identity_lock_hold: num(activeParams.identity_lock_hold, 3),
      swap_model: activeParams.swap_model, default_det_size: activeParams.default_det_size,
      face_detector_size: activeParams.face_detector_size, face_detector_threshold: activeParams.face_detector_threshold,
      face_detector_nms: activeParams.face_detector_nms,
      color_transfer_mode: activeParams.color_transfer_mode,
      skin_tone_warmth: num(activeParams.skin_tone_warmth, 0),
      saturation_match: num(activeParams.saturation_match, 0),
      specular_highlight_recovery: num(activeParams.specular_highlight_recovery, 0),
      ambient_shadow_match: num(activeParams.ambient_shadow_match, 0),
      subdermal_warmth_match: num(activeParams.subdermal_warmth_match, 0),
      target_conditioned_appearance: !!activeParams.target_conditioned_appearance,
      target_conditioned_appearance_strength: num(activeParams.target_conditioned_appearance_strength, 0.75),
      target_conditioned_appearance_temporal_alpha: num(activeParams.target_conditioned_appearance_temporal_alpha, 0.30),
      light_harmonizer: !!activeParams.light_harmonizer,
      light_harmonizer_key_intensity: num(activeParams.light_harmonizer_key_intensity, 1.0),
      light_harmonizer_ambient_bias: num(activeParams.light_harmonizer_ambient_bias, 0.0),
      light_harmonizer_eye_specular: num(activeParams.light_harmonizer_eye_specular, 0.8),
      light_harmonizer_shadow_occlusion: num(activeParams.light_harmonizer_shadow_occlusion, 0.6),
      sam2_model_size: activeParams.sam2_model_size,
      refine_landmarks: activeParams.refine_landmarks,
      swap_model_mask_strength: activeParams.swap_model_mask_strength,
      rescue_small_faces: activeParams.rescue_small_faces,
      detector_engine: activeParams.detector_engine,
      face_mapping: getFaceMappingArray(activeParams),
      target_person_source_mapping: getStableTargetSourceMapping(activeParams),
      target_person_ids: targetPersonIds,
      source_mapping_names: getSourceMappingNames(activeParams),
      selected_source_id: sourceIdAt(selSource),
      selected_source_name: sourceNameAt(selSource),
      target_media_id: requestedMediaId || targetIdAt(index) || activeTargetMediaId,
      selection_state: getTargetSelectionState(activeParams),
      processing_selection: buildProcessingSelection(activeParams),
      selection_version: selectionVersion,
      mask_top: activeParams.mask_top,
      mask_bottom: activeParams.mask_bottom,
      mask_left: activeParams.mask_left,
      mask_right: activeParams.mask_right,
      face_mask_blend: activeParams.face_mask_blend,
      mask_edge_mode: activeParams.mask_edge_mode,
      mask_flow_warp: !!activeParams.mask_flow_warp,
      mask_guided_filter: !!activeParams.mask_guided_filter,
      restore_ultra_frequency_blend: activeParams.restore_ultra_frequency_blend !== false,
      restore_ultra_detail_weight: num(activeParams.restore_ultra_detail_weight, 0.75),
      ultramax_swap_lowband: activeParams.ultramax_swap_lowband !== false,
      ultramax_detail_weight: num(activeParams.ultramax_detail_weight, 0.75),
      restore_ultra_inner_only: !!activeParams.restore_ultra_inner_only,
      mask_erode_dilate_radius: num(activeParams.mask_erode_dilate_radius, 0),
      boundary_illumination_strength: activeParams.boundary_illumination_strength,
      mouth_mask_blend: activeParams.mouth_mask_blend,
      mouth_top_scale: activeParams.mouth_top_scale,
      mouth_bottom_scale: activeParams.mouth_bottom_scale,
      mouth_left_scale: activeParams.mouth_left_scale,
      mouth_right_scale: activeParams.mouth_right_scale,
      // Manual brush mask (JSON string, '' when nothing is painted).
      imagemask: maskJson,
      identity_blend: identityBlend,
      // Stage 11: closed Pen-tool zones and the user's identity pins for THIS clip.
      roto_polygons: rotoPayload,
      identity_pins: pinsPayload,
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
  // The canonical selection is reduced to its identity fields: the request id
  // and version are per-request bookkeeping and would make every signature
  // unique, which would turn the cache into a no-op.
  const previewSignature = (params, fake, solo = soloStageRef.current) =>
    JSON.stringify(buildPreviewPayload(params, {
      index: 0, frame: 0, fake, imagemask: `mask:${maskVersion}`,
      processing_selection: selectionIdentity(buildProcessingSelection(params, { requestId: 'sig' })),
      selection_version: null,
      solo_stage: solo,
    }));

  // The settings IN FORCE at the playhead: the clip's, with the scene override covering this frame
  // (if any) written over them. Same object as `p` when no override applies, so nothing downstream
  // changes identity until the playhead crosses into one. Scrubbing into an override range
  // therefore previews the override; the key below changes with it, so a cached preview of the
  // un-overridden frame cannot be served for the overridden one.
  //
  // Stage 25: a parameter curve is written over that last, at the frame's own value (the curve is the finest control:
  // it can ramp across an override's edge, which an override alone could only do as a pop). Exact here; a render
  // approximates it in steps (curves/curvePieces.js).
  const settingsAtFrame = (fr) => patchSettingsWithCurves(
    patchSettings(p, overrideAt(sceneOverrides.activeOverrides, fr)), curves.activeTracks, fr);
  const pAtFrame = settingsAtFrame(frame);
  const previewKey = previewSignature(pAtFrame, fakePreview);

  // What the stage wants to show RIGHT NOW, readable from inside an async
  // response handler that closed over an older render. Updated every render.
  const wantedPreviewRef = useRef({ key: '', mediaId: null, frame: 1 });
  wantedPreviewRef.current = {
    key: getCacheKey(selTarget, frame, soloStage),
    mediaId: activeTargetMediaId,
    frame,
  };
  // Monotonic id of the newest preview request issued; a response for an
  // older one is logged as superseded.
  const previewSeqRef = useRef(0);

  // The four quick recipes (faceswap/recipes.js). Each is a block of real settings
  // written straight into the live params; a recipe the settings already match
  // is highlighted. Pure UI: no runtime cost, and nothing global (provider,
  // threads) is touched.
  const activeRecipe = activeRecipeId(p);
  const applyRecipe = (recipe) => {
    setSettings((s) => ({ ...s, ...recipe.patch }));
    notify(`Applied ${recipe.label}`, 'info');
  };

  // The restoration tier buttons in the Face Restorer stage: a restorer plus the
  // `restoration_mode` label the Tile Diffusion path reads. The lit tier comes from
  // the restorer that is actually selected (a mode label alone is not what runs).
  const RESTORATION_TIERS = [
    { id: 'fast', label: 'Fast', note: '25–60 FPS', hint: 'GPEN-512 / GFPGAN: realtime scrubbing.', patch: { restoration_mode: 'fast', selected_enhancer: 'GPEN' } },
    { id: 'balanced', label: 'Balanced', note: 'CodeFormer w 0.6', hint: 'CodeFormer at the optimal fidelity weight.', patch: { restoration_mode: 'balanced', selected_enhancer: 'Codeformer', codeformer_fidelity: 0.6 } },
    { id: 'vfx', label: 'VFX Master', note: '2–6 FPS', hint: 'Tile Diffusion: offline rendering only.', patch: { restoration_mode: 'vfx', selected_enhancer: 'Tile Diffusion Synthesizer' } },
  ];
  const restorationTierOf = (cur) => {
    const e = String(cur.selected_enhancer || '');
    if (e === 'GPEN' || e === 'GFPGAN') return 'fast';
    if (e.toLowerCase().startsWith('codeformer')) return 'balanced';
    if (e === 'Tile Diffusion Synthesizer' || e === 'Tile Diffusion') return 'vfx';
    return null;
  };
  const TEMPORAL_BADGE = { off: 'Off', smart_skip: 'Smart-Skip', balanced: 'Balanced', cinema: 'Cinema', custom: 'Custom' };

  // The user's own "Save as default" snapshot, and the resets that read it.
  // See faceswap/useUserDefaults.
  const { userDefaults, saveAsDefault, resetToDefaults, clearUserDefault } =
    useUserDefaults({ settings: p, setSettings, notify });

  // ── View state across a webview reload ───────────────────────────────────
  // Where you were looking (target, playhead, rendered frame), remembered
  // across the webview reload a Pinokio tab switch causes. Captured at mount
  // and read once by the rehydrate effect below — see
  // faceswap/useViewPersistence.
  const restoredViewRef = useViewPersistence({
    selTarget, targetMediaId: activeTargetMediaId, frame, previewSrc,
  });

  // Pull the backend's own view of the workspace — sources, targets, harvested
  // faces — into this component. Two callers, and the difference between them
  // is `restoreView`:
  //
  //   mount        restoreView: true  — the webview was reloaded under a
  //                                     running backend, so the playhead and
  //                                     rendered frame the user was looking at
  //                                     are still the truth.
  //   project load restoreView: false — the sources and target have just been
  //                                     REPLACED by a saved project's, so a
  //                                     playhead remembered from the previous
  //                                     media would point at the wrong frame of
  //                                     the wrong file.
  const hydrateWorkspaceFromState = React.useCallback(async ({ restoreView = false } = {}) => {
    const st = await getJSON('/api/state');
    checkDesync(st);
    setSourceFaces(st.source_faces || []);
    if (st.source_faces_info) setSourceFacesInfo(st.source_faces_info);
    // The selected SOURCE is restored by its stable id; the numeric index is
    // only a fallback for a backend that predates selected_source_id.
    if (Array.isArray(st.source_faces_info)) {
      const byId = st.selected_source_id
        ? st.source_faces_info.findIndex((info) => String(info?.id) === String(st.selected_source_id))
        : -1;
      const byIndex = Number.isInteger(st.selected_source_index)
        && st.selected_source_index >= 0
        && st.selected_source_index < st.source_faces_info.length
        ? st.selected_source_index : 0;
      setSelSource(byId >= 0 ? byId : byIndex);
    }
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
      const v = restoreView ? restoredViewRef.current : null;
      const mediaId = st.target_media_id || tg[sel]?.media_id || tg[sel]?.id;
      const sameView = v &&
        ((v.target_media_id && v.target_media_id === mediaId)
         || (!v.target_media_id && v.target === sel))
        && v.frame >= 1 && v.frame <= mf;
      setSelTarget(sel);
      setMaxFrames(mf);
      if (mediaId) {
        targetContextsRef.current[mediaId] = {
          ...(targetContextsRef.current[mediaId] || {}),
          frame: sameView ? v.frame : 1,
          previewSrc: sameView ? (v.image || '') : '',
          previewFor: sameView ? `${mediaId}_${v.frame}` : '',
        };
        applyTargetContext({
          ...st,
          selected_target_index: sel,
        }, mediaId);
      }
    } else {
      setTargetFaces([]);
      setTargetGroups([]);
      setTargetNames([]);
      setTargetFacesInfo([]);
      setTargetPersonIds([]);
      setTargetReferenceFaceIds([]);
      setSelectedTargetPersonId(null);
      setSelectedReferenceFaceId(null);
      setFaceMapping({});
      setSelTargetFace(0);
      setPreviewSrc('');
      setPreviewFor('');
    }
    return st;
  /* eslint-disable-next-line react-hooks/exhaustive-deps -- setters are stable; checkDesync and restoredViewRef are refs/stable by construction */
  }, []);

  // ── initial rehydrate ──
  // Pinokio reloads the webview whenever you switch the RUN/DEV/FILES tabs,
  // which remounts this component and wipes its React state. The backend keeps
  // running, so we restore both the faces/targets AND the live job state.
  useEffect(() => {
    hydrateWorkspaceFromState({ restoreView: true }).catch(logFailure('Restoring the workspace', { toast: false }));

    // Prefer the backend's own start time when in flight without extra network poll
    if (progress?.processing && !startTime) {
      setStartTime(progress.started_at ? progress.started_at * 1000 : Date.now());
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const refreshPreview = async (opts = {}) => {
    if (targets.length === 0) {
      setPreviewSrc('');
      setPreviewFor('');
      setPreviewDiagnostic(null);
      return;
    }

    const idx = opts.index ?? selTarget;
    const fr = opts.frame ?? frame;
    const fake = opts.fake ?? fakePreview;
    const requestedMediaId = opts.target_media_id || targetIdAt(idx);

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
    const effectiveSolo = opts.solo_stage !== undefined ? opts.solo_stage : soloStageRef.current;
    if (!opts.force) {
      const cached = getCachedPreview(idx, fr, effectiveSolo);
      if (cached) {
        setPreviewFaces(cached.faces);
        setPreviewPersonIds(cached.personIds || []);
        setPreviewKps(cached.kps || []);
        setPreviewAnchors(cached.anchors || []);
        setPreviewPose(cached.pose || []);
        setPreviewDiagnostic(cached.diagnostic || null);
        if (cached.sar !== undefined) setPreviewSar(cached.sar);
        if (cached.dar !== undefined) setPreviewDar(cached.dar);
        if (cached.colorBadge !== undefined) setPreviewColorBadge(cached.colorBadge);
        // No revoke of the outgoing url here: it is a cache entry, and the
        // cache is what owns it. Freeing it would blank a frame the user can
        // still step back onto.
        previewSrcRef.current = cached.image;
        setPreviewSrc(cached.image);
        setPreviewFor(`${requestedMediaId || `legacy-index-${idx}`}_${fr}`);
        return;
      }
    } else {
      delete previewCacheRef.current[getCacheKey(idx, fr, effectiveSolo)];
    }

    // Single-flight: the backend's live_swap shares one (non-thread-safe)
    // ProcessMgr on the GPU. Two overlapping /api/preview calls corrupt/hang
    // TensorRT/CUDA. So never run two at once — queue the latest request and
    // run it once the current one finishes.
    if (previewBusyRef.current) {
      // Coalesced: store newest opts and immediately abort the stale in-flight request
      previewPendingRef.current = { ...opts };
      if (previewAbortCtrlRef.current) {
        try {
          previewAbortCtrlRef.current.abort();
        } catch {
          // ignore
        }
        previewAbortCtrlRef.current = null;
      }
      return;
    }
    
    previewBusyRef.current = true;
    setPreviewing(true);
    // Safety net: the first run of a new model downloads it and builds a
    // TensorRT/CUDA engine (minutes). Abort after 15 min so a genuine hang can
    // never wedge the single-flight guard permanently.
    const ctrl = new AbortController();
    previewAbortCtrlRef.current = ctrl;
    // `killed` tells OUR deadline apart from any other abort. The request is also aborted
    // when this panel unmounts (abortOnUnmount): that is the user leaving, not a model
    // build that took too long, and must neither toast nor start the queued next preview.
    let killed = false;
    let abandoned = false;
    const killer = setTimeout(() => { killed = true; ctrl.abort(); }, 15 * 60 * 1000);
    // Everything the response will be judged against is captured NOW, from
    // this render's values — never re-read after the await, when the UI may
    // be on another person, source, target or frame.
    previewSeqRef.current += 1;
    const request = {
      seq: previewSeqRef.current,
      requestId: null,
      key: getCacheKey(idx, fr, effectiveSolo),
      mediaId: requestedMediaId,
      frame: fr,
      selection: null,
    };
    try {
      // The selection is built first so its request id is known before the
      // body leaves; the body itself still comes from buildPreviewPayload (the
      // only /api/preview request site, see test_settings_wiring).
      const selection = buildProcessingSelection(p);
      request.requestId = selection.request_id;
      request.selection = selection;
      const res = await postJSON('/api/preview', buildPreviewPayload(settingsAtFrame(fr), {
        index: idx, frame: fr, fake, target_media_id: requestedMediaId,
        processing_selection: selection,
        solo_stage: effectiveSolo,
        ...(opts.overrides || {}),
      }), { abortOnUnmount: true, timeout: 0, signal: ctrl.signal });
      if (res?.error) throw new Error(res.message || res.error || 'preview swap failed');
      const verdict = classifyPreviewResponse({
        request, response: res, wanted: wantedPreviewRef.current,
      });
      if (!verdict.accept) {
        // Never displayed; never written under the live selection's key.
        // A superseded-but-valid answer is filed under ITS OWN key so a step
        // back to that frame/selection is instant.
        if (verdict.cache && res.image) {
          const own = dataUrlToOwnedBlobUrl(res.image, previewOwner);
          setCachedPreviewByKey(request.key, {
            faces: res.faces || [], personIds: res.person_ids || [], kps: res.kps || [],
            anchors: res.anchors || [],
            pose: res.pose || [], image: own,
            diagnostic: res.selection_diagnostic || (res.target_required ? 'target_required' : null),
            signature: res.preview_signature || null, requestId: res.request_id || null,
            sar: res.sar_val ?? 1.0, dar: res.dar_val ?? 0.0,
          colorBadge: res.color_badge || '',
          });
        }
        if (import.meta.env?.DEV) {
          console.debug('[preview] stale response discarded', {
            reason: verdict.reason, seq: request.seq, newest: previewSeqRef.current,
            request_id: request.requestId, target_media_id: request.mediaId,
            target_person_id: request.selection?.target_person_id,
            source_identity_id: request.selection?.source_identity_id,
            selection_version: request.selection?.selection_version,
            preview_signature: res.preview_signature,
          });
        }
        return;
      }
      if (res.faces) setPreviewFaces(res.faces);
      setPreviewPersonIds(res.person_ids || []);
      setPreviewKps(res.kps || []);
      setPreviewAnchors(res.anchors || []);
      setPreviewPose(res.pose || []);
      if (res.sar_val !== undefined) setPreviewSar(res.sar_val);
      if (res.dar_val !== undefined) setPreviewDar(res.dar_val);
      if (res.color_badge !== undefined) setPreviewColorBadge(res.color_badge);
      const diagnostic = res.selection_diagnostic
        || (res.target_required ? 'target_required' : null);
      setPreviewDiagnostic(diagnostic);
      // The backend hands back a base64 data URL. It is converted here, ONCE,
      // at the only point it enters the client — so nothing downstream (React
      // state, the cache, the pop-out, localStorage) ever holds the megabytes.
      // See the note on previewCacheRef.
      const blobSrc = res.image ? dataUrlToOwnedBlobUrl(res.image, previewOwner) : '';
      // The previous frame's URL is about to stop being referenced by state. If
      // the cache is not holding it too, nothing else ever will, so free it.
      const prevSrc = previewSrcRef.current;
      if (prevSrc && prevSrc !== blobSrc && !cacheHoldsUrl(prevSrc)) revokeUrl(prevSrc);
      previewSrcRef.current = blobSrc;
      setPreviewSrc(blobSrc);
      setPreviewFor(blobSrc ? `${requestedMediaId || `legacy-index-${idx}`}_${fr}` : '');
      if (blobSrc) {
        setCachedPreviewByKey(request.key, {
          faces: res.faces || [], personIds: res.person_ids || [], kps: res.kps || [],
          anchors: res.anchors || [],
          pose: res.pose || [], image: blobSrc, diagnostic,
          signature: res.preview_signature || null, requestId: res.request_id || null,
          sar: res.sar_val ?? 1.0, dar: res.dar_val ?? 0.0,
        });
      }
    } catch (e) {
      if (e.name === 'AbortError') {
        if (killed) notify('Preview timed out (model build took too long)', 'error');
        else if (!previewPendingRef.current) abandoned = true;
      } else {
        notify(e.message, 'error');
      }
    }
    finally {
      clearTimeout(killer);
      if (previewAbortCtrlRef.current === ctrl) {
        previewAbortCtrlRef.current = null;
      }
      previewBusyRef.current = false;
      setPreviewing(false);
      if (previewPendingRef.current && !abandoned) {
        const next = previewPendingRef.current;
        previewPendingRef.current = null;
        requestAnimationFrame(() => {
          (refreshPreviewRef.current || refreshPreview)(next);
        });
      }
    }
  };
  refreshPreviewRef.current = refreshPreview;

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
    cacheSuffix, reloadKey: previewKey, owner: previewOwner,
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
      const baseRes = await postJSON('/api/preview', buildPreviewPayload(pAtFrame, { index: selTarget, frame, fake: true }), { abortOnUnmount: true, timeout: 0 });
      baseImage = baseRes.image || '';
    } catch (err) {
      reportFailure('Rendering the base frame for the upscale preview', err, { toast: false });
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

        const res = await postJSON('/api/preview_upscale', { image: baseImage, subtype }, { abortOnUnmount: true, timeout: 0 });

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
    // Keep the flag alive if the user is mid-scrub or playing — we cannot
    // refresh right now, and clearing the flag here would discard the
    // "needs refresh after run" intent permanently. Re-fire when scrubbing
    // ends (the [selTarget, frame, …] effect handles that).
    if (isScrubbing || isPlaying) return;
    previewDeferredRef.current = false;
    if (targets.length === 0) return;
    const t = setTimeout(() => refreshPreview(), 350);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [progress.processing, isScrubbing, isPlaying]);

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
      const res = checkDesync(await postFiles('/api/source/add', files, undefined, { timeout: 0,
        onProgress: setSrcProgress, signal: ctrl.signal,
      }));
      setSourceFaces(res.source_faces);
      if (res.source_faces_info) setSourceFacesInfo(res.source_faces_info);
      const added = res.source_faces.length - before;
      if (res.errors?.length) {
        // Structured per-file faceset failures (corrupt archive, checksum
        // mismatch, no detectable face). The reason, not "no face detected".
        res.errors.forEach((e) => notify(`${e.file}: ${e.message}`, 'error'));
      }
      if (res.unsupported?.length) {
        notify(
          `Unsupported file type(s) — only images (.png/.jpg/.jpeg/.webp) and .fsz facesets are accepted as source: ${res.unsupported.join(', ')}`,
          'error'
        );
      } else if (added > 0) notify(`Loaded ${added} face(s) — ${res.faceset_count} faceset(s) total`);
      else if (res.errors?.length) { /* already reported above */ }
      else notify('No face detected in the uploaded file(s)', 'error');
      return res;
    } catch (err) { reportUploadError(err); return null; }
    finally { setUploadingSrc(false); setSrcProgress(null); srcAbortRef.current = null; }
  };

  const onAddSourceFolder = async (files) => {
    if (!files || !files.length) return;
    const ctrl = new AbortController();
    srcAbortRef.current = ctrl;
    setUploadingSrc(true);
    setSrcProgress(null);
    try {
      const res = checkDesync(await postFiles('/api/source/add-folder', files, undefined, { timeout: 0,
        onProgress: setSrcProgress, signal: ctrl.signal,
      }));
      if (res.error) throw new Error(res.error);
      setSourceFaces(res.source_faces);
      if (res.source_faces_info) setSourceFacesInfo(res.source_faces_info);
      notify(`Built multi-angle faceset from ${files.length} reference image(s)`);
    } catch (err) { reportUploadError(err); }
    finally { setUploadingSrc(false); setSrcProgress(null); srcAbortRef.current = null; }
  };

  // Everything that happens once the backend has accepted new targets, however
  // they arrived — uploaded, or referenced by path. Shared so the two routes
  // cannot drift on the auto-queue rule or the preview refresh.
  const applyTargetAdd = async (res, beforeCount) => {
      const newTargetsList = res.targets || [];
      setTargets(newTargetsList);
      const selectedIndex = res.selected_target_index || 0;
      const mediaId = res.target_media_id || targetIdAt(selectedIndex)
        || newTargetsList[selectedIndex]?.media_id || newTargetsList[selectedIndex]?.id;
      setSelTarget(selectedIndex);
      const mf = newTargetsList[selectedIndex]?.frames || 1;
      setMaxFrames(mf); setFrame(1);
      applyTargetContext({ ...res, selected_target_index: selectedIndex }, mediaId);
      // No eager refreshPreview here: this closure still holds the PREVIOUS
      // target's people/mapping, so a request built now would pair the new
      // media with the old selection (and be discarded as stale on return).
      // The [selTarget, frame] effect issues the request from the re-rendered
      // context instead.
      notify(`Added ${newTargetsList.length - beforeCount} target(s)`);

      // Automatically add videos to batch queue if more than 1 video is uploaded
      const newTargets = newTargetsList.slice(beforeCount);
      const newVideos = newTargets.filter(t => t.frames > 1);
      if (newVideos.length > 1) {
        await queue.addMany(newVideos.map((t, idx) => {
          const srcIdx = sourceFaces.length > 1 ? (idx % sourceFaces.length) : selSource;
          const targetMediaId = t.media_id || t.id || null;
          const targetMediaIndex = newTargetsList.findIndex((candidate) =>
            (candidate.media_id || candidate.id) === targetMediaId);
          const targetPayload = buildSwapPayload(p, {
            sourceIndex: srcIdx,
            targetMediaId,
            targetMediaIndex: targetMediaIndex >= 0 ? targetMediaIndex : beforeCount + idx,
          });
          return {
            target_name: t.name || '',
            target_media_id: targetMediaId,
            source_index: srcIdx,
            source_name: sourceFacesInfo[srcIdx]?.name
              || (sourceFaces[srcIdx] ? `Face ${srcIdx + 1}` : 'Selected face'),
            source_id: sourceIdAt(srcIdx),
            payload: {
              ...targetPayload,
              source_index: srcIdx,
            },
          };
        }));
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
      await applyTargetAdd(await postFiles('/api/target/add', files, undefined, { timeout: 0,
        onProgress: setTgtProgress, signal: ctrl.signal,
      }), beforeCount);
    } catch (err) { reportUploadError(err); }
    finally { setUploadingTgt(false); setTgtProgress(null); tgtAbortRef.current = null; }
  };

  const onAddLipsyncAudio = async (files) => {
    if (!files || !files.length) return;
    setUploadingLipsyncAudio(true);
    try {
      const res = await postFile('/api/lipsync/audio/add', files[0], undefined, { timeout: 0 });
      setMany(overridePatch(res.path));
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
      const res = await postJSON('/api/target/add_path', { paths }, { timeout: 0 });
      // Report rejects individually: with several paths pasted at once, a
      // silent drop looks like the app ignored the whole thing.
      (res.rejected || []).forEach((r) => notify(`Skipped ${r.path} — ${r.why}`, 'error'));
      if ((res.added || []).length) await applyTargetAdd(res, beforeCount);
    } catch (err) { notify(err.message, 'error'); }
    finally { setUploadingTgt(false); }
  };

  const removeTarget = async (i) => {
    rememberTargetContext();
    const mediaId = targetIdAt(i);
    const res = await postJSON('/api/target/remove', { index: i, target_media_id: mediaId });
    setTargets(res.targets);
    const newSel = res.selected_target_index || 0;
    setSelTarget(newSel);
    const nextId = res.target_media_id || res.targets[newSel]?.media_id || res.targets[newSel]?.id;
    if (res.targets.length === 0) {
      setPreviewSrc(''); setPreviewFor(''); setMaxFrames(1);
      applyTargetContext(res, null);
    } else {
      setMaxFrames(res.targets[newSel]?.frames || 1);
      applyTargetContext({ ...res, selected_target_index: newSel }, nextId);
    }
  };

  const selectTarget = async (i) => {
    const mediaId = targetIdAt(i);
    if (!mediaId) return;
    rememberTargetContext();
    const res = await postJSON('/api/target/select', {
      index: i,
      target_media_id: mediaId,
      selected_target_face_index: selTargetFace,
    });
    setTargets(res.targets);
    const selectedIndex = res.selected_target_index ?? i;
    const selectedId = res.target_media_id || mediaId;
    setSelTarget(selectedIndex);
    const mf = res.targets[selectedIndex]?.frames || 1;
    setMaxFrames(mf);
    applyTargetContext({ ...res, selected_target_index: selectedIndex }, selectedId);
    // See applyTargetAdd: the refresh comes from the [selTarget, frame] effect,
    // once this render's stale closure has been replaced by B's context.
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
      const res = checkDesync(await postJSON(path, body, { timeout: timeoutFor('POST', path) }));
      if (res.source_faces) setSourceFaces(res.source_faces);
      if (res.source_faces_info) setSourceFacesInfo(res.source_faces_info);
      return res;
    } catch (e) {
      // Surface failures (e.g. a 404 when the backend hasn't been restarted to
      // pick up a new endpoint) instead of silently doing nothing.
      notify(`${path.split('/').pop()} failed: ${e.message}. If this is a new feature, restart the app server.`, 'error');
      return null;
    }
  };

  // Removing a source faceset shifts every later index down by one. The
  // person→source mapping stores raw indices, so without this it keeps pointing
  // at whatever slid into the removed slot (wrong face) or off the end of the
  // list (the backend then swaps in an empty faceset).
  const removeSource = async (i) => {
    if (i < 0 || i >= sourceFaces.length) return;
    const res = await sourceAction('/api/source/remove', { index: i });
    if (!res) return;
    // Rebase the EFFECTIVE mapping, not only explicit dropdown overrides. This
    // keeps default person→source bindings stable too: the removed source is a
    // deliberate skip, and later indices compact around it.
    setFaceMapping(mappingObjectFromArray(
      remapSourceMappingAfterRemoval(getFaceMappingArray(), i),
    ));
    const nextSel = selSource === i ? 0 : selSource > i ? selSource - 1 : selSource;
    if (nextSel !== selSource) {
      setSelSource(nextSel);
      try { await postJSON('/api/source/select', { index: nextSel }); } catch (err) { reportFailure('Saving the source selection', err); /* selection is best-effort */ }
    }
  };

  const moveSource = async (direction) => {
    const from = selSource;
    const to = direction === 'left' ? from - 1 : from + 1;
    if (from < 0 || from >= sourceFaces.length || to < 0 || to >= sourceFaces.length) return;
    const res = await sourceAction('/api/source/move', { index: from, direction });
    if (!res) return;
    // Rebase the effective person->source mapping in the same transaction as
    // the gallery move.  Otherwise the raw numeric source slots would make a
    // target person inherit whichever faceset slid into the old slot.
    setFaceMapping(mappingObjectFromArray(
      remapSourceMappingAfterMove(getFaceMappingArray(), from, to),
    ));
    setSelSource(to);
    try { await postJSON('/api/source/select', { index: to }); } catch (err) { reportFailure('Saving the source order', err); /* best effort */ }
  };
  const selectSource = async (i) => {
    setSelSource(i);
    if (activeTargetMediaId) {
      rememberTargetContext(activeTargetMediaId, { selSource: i });
    }
    await postJSON('/api/source/select', { index: i });
  };

  const captureTargetFaceFromFrame = async ({ faceIndex = null, captureAll = false } = {}) => {
    try {
      const res = await postJSON('/api/target/use_face', {
        index: selTarget,
        frame,
        target_media_id: activeTargetMediaId,
        ...(captureAll ? { capture_all: true } : { face_index: faceIndex }),
      }, { timeout: 0 });
      setTargetFaces(res.target_faces);
      setTargetGroups(res.target_groups || []);
      setTargetNames(res.target_names || []);
      setTargetFacesInfo(res.target_faces_info || []);
      setTargetPersonIds(res.target_person_ids || []);
      setTargetReferenceFaceIds(res.target_reference_face_ids || []);
      setSelectedTargetPersonId(res.selected_target_person_id || null);
      setSelectedReferenceFaceId(res.selected_reference_face_id || null);
      setSelTargetFace(Math.max(0, (res.target_faces || []).length - 1));
      setSelectedDetectedFace(null);
      applyTargetContext(res, res.target_media_id || activeTargetMediaId);
      // Capturing more than one target person must not leave the preview in
      // the single-person mode. That made "Capture all people" visibly add
      // several boxes while the next preview still swapped only the selected
      // one. Keep the explicit single-face behavior for one capture, but make
      // a multi-person capture immediately preview every captured person.
      const capturedPeople = new Set(
        (res.target_person_ids || []).filter((id) => id !== null && id !== undefined && id !== ''),
      );
      set('face_detection_mode', capturedPeople.size > 1 ? 'Selected people' : 'Selected face');
      notify(captureAll
        ? `Added ${res.count} target person(s)${capturedPeople.size > 1 ? ' — previewing all captured people' : ''}`
        : `Added Face ${faceIndex + 1} as a target person`);
      return res.selected_target_person_id || null;
    } catch (e) { notify(e.message, 'error'); return null; }
  };

  const useFaceFromFrame = async () => {
    if (selectedDetectedFaceIndex === null) {
      notify('Select one detected face in the preview first', 'error');
      return;
    }
    await captureTargetFaceFromFrame({ faceIndex: selectedDetectedFaceIndex });
  };

  // A click on ONE numbered box in the preview IS the explicit choice of a
  // target face, so it captures that face right away. It used to only mark
  // the box and wait for the "Use selected face" button below the stage --
  // which sat out of view, so nothing ever reached the target panel and
  // nothing swapped (the backend log showed no /api/target/use_face at all).
  //
  // A box the overlay already labels with a CAPTURED person (its person id is
  // below the captured count) selects that person instead of capturing it a
  // second time: every use_face creates a new person, and two people for one
  // face is exactly the duplicate this avoids.
  // What the render will do with each detected box, and the one-click lock-out.
  // `faceRole` shares its rule with the payload (stableTargetSourceMapping), so the
  // badge can never say "bystander" for a face the job will swap.
  const previewFaceRoles = previewFaces.map((_, faceIndex) => {
    const records = targetPersonRecords({ targetGroups, targetPersonIds });
    const rawRank = previewPersonIds[faceIndex];
    // Number(null) is 0: a missing id must not read as person 0.
    const rank = rawRank === null || rawRank === undefined || rawRank === '' ? NaN : Number(rawRank);
    const personId = Number.isInteger(rank) && rank >= 0 && rank < records.length
      ? records[rank].targetPersonId : null;
    return {
      personId,
      ...faceRole({
        personId,
        faceSelection: p.face_detection_mode,
        selectedTargetPersonId,
        faceMapping,
        targetGroups,
        targetPersonIds,
        sourceCount: sourceFaces.length,
        sourceIdentityIds: sourceFacesInfo.map((info, index) => info?.id || `memory-slot-${index}`),
        selectedSource: selSource,
      }),
    };
  });

  // Stage 11. Per detected face: the source it swaps from (the same mapping the payload carries, indexed by
  // person rank), and its nose/chin. The Pen tool pins to the former's face and the latter's landmarks.
  const mappingByRank = getFaceMappingArray();
  const faceSources = previewFaces.map((_, i) => {
    const rank = previewPersonIds[i] === null || previewPersonIds[i] === undefined || previewPersonIds[i] === ''
      ? NaN : Number(previewPersonIds[i]);
    const s = Number.isInteger(rank) ? mappingByRank[rank] : null;
    return Number.isInteger(s) && s >= 0 ? s : null;
  });
  const faceAnchors = previewFaces.map((bbox, i) => {
    const a = anchorsOf(previewAnchors, i);
    return a ? { source_index: faceSources[i], nose: a.nose, chin: a.chin, bbox } : null;
  }).filter(Boolean);

  const onPinIdentity = (faceIndex) => {
    const bbox = previewFaces[faceIndex];
    const pin = makePin({
      frame, bbox, personId: previewPersonIds[faceIndex], sourceIndex: faceSources[faceIndex], faceIndex,
    });
    if (!pin) { notify('This face has no source to lock: map it to a source first', 'error'); return; }
    setPinsByMedia((all) => ({ ...all, [clipKey]: addPin(all[clipKey] || [], pin) }));
    notify(`Pinned ${trackLabel(previewPersonIds[faceIndex])} to its source on frame ${frame}`, 'info');
  };
  const onUnpinIdentity = (id) => setPinsByMedia((all) => ({ ...all, [clipKey]: removePin(all[clipKey] || [], id) }));

  const onToggleFaceExclusion = (faceIndex) => {
    const info = previewFaceRoles[faceIndex];
    if (!info || !info.personId || !info.canExclude) return;
    const next = info.excluded
      ? includePerson(faceMapping, info.personId)
      : excludePerson(faceMapping, info.personId);
    setFaceMapping(next);
    // Same write the Person Groups dropdown makes; the versioned path keeps a
    // preview that was already in flight from landing on top of it.
    commitTargetContext({ face_mapping: next, target_person_source_mapping: next });
    clearPreviewCache();
    notify(info.excluded
      ? 'Person included again'
      : 'Excluded: this person is never swapped', 'info');
  };

  const onFaceBoxClick = async (faceIndex) => {
    if (!Number.isInteger(faceIndex) || faceIndex < 0 || faceIndex >= previewFaces.length) return;
    selectDetectedFace(faceIndex);
    const records = targetPersonRecords({
      targetGroups, targetPersonIds, targetReferenceFaceIds, targetNames,
    });
    // Two ways a box can already be somebody: the overlay labelled it with a
    // captured person (after the preview refreshed), or THIS box on THIS frame
    // was captured a moment ago and the refresh has not landed yet (a second
    // click, or a double-click). Both select; neither captures again.
    const boxKey = `${activeTargetMediaId}_${frame}_${faceIndex}`;
    const known = capturedBoxesRef.current[boxKey];
    let rank = Number(previewPersonIds[faceIndex]);
    if (known && records.some((r) => r.targetPersonId === known)) {
      rank = records.findIndex((r) => r.targetPersonId === known);
    }
    if (Number.isInteger(rank) && rank >= 0 && rank < records.length) {
      const record = records[rank];
      const firstFace = record.faceIndices[0] ?? 0;
      setSelectedTargetPersonId(record.targetPersonId);
      setSelectedReferenceFaceId(targetReferenceFaceIds[firstFace] || null);
      setSelTargetFace(firstFace);
      set('face_detection_mode', 'Selected face');
      commitTargetContext({
        selected_target_person_id: record.targetPersonId,
        ...(targetReferenceFaceIds[firstFace]
          ? { selected_reference_face_id: targetReferenceFaceIds[firstFace] } : {}),
        selected_target_face_index: firstFace,
      });
      notify(`Selected ${record.name || `Person ${rank + 1}`} as the target person`);
      return;
    }
    if (captureBusyRef.current) return;   // a capture for a box is already in flight
    captureBusyRef.current = true;
    try {
      const created = await captureTargetFaceFromFrame({ faceIndex });
      if (created) capturedBoxesRef.current[boxKey] = created;
    } finally {
      captureBusyRef.current = false;
    }
  };
  // box (media_frame_index) -> the target_person_id its capture produced.
  const capturedBoxesRef = useRef({});
  const captureBusyRef = useRef(false);

  const captureAllPeopleFromFrame = async () => {
    if (!previewFaces.length) {
      notify('No detected faces to capture', 'error');
      return;
    }
    await captureTargetFaceFromFrame({ captureAll: true });
  };

  // Spot-check the AI upscale on just the current preview frame — sends the
  // exact swapped image the preview shows to the backend, which upscales the
  // one frame and returns it for inspection in a modal.
  const upscaleThisFrame = async () => {
    if (!previewSrc) { notify('No preview to upscale', 'error'); return; }
    setUpscaling(true);
    try {
      // The backend wants the pixels, and previewSrc is now a blob URL — a
      // reference this process owns, meaningless to the server. Re-serialise
      // on demand: this is a click, not a render tick, so the cost is invisible
      // here and the saving is on every frame of every scrub.
      const imageBytes = await blobUrlToDataUrl(previewSrc);
      if (!imageBytes) throw new Error('preview frame is no longer available');
      const res = await postJSON('/api/preview_upscale', {
        image: imageBytes,
        subtype: p.upscale_model_after || 'esrganx2',
      }, { abortOnUnmount: true, timeout: 0 });
      if (!res.image) throw new Error(res.message || 'upscale failed');
      setUpscaledSrc(res.image);
      setUpscaledDims(res.width && res.height ? { w: res.width, h: res.height } : null);
    } catch (e) {
      notify(e.message, 'error');
    } finally {
      setUpscaling(false);
    }
  };

  const setFrameMarkerVal = async (which, val) => {
    if (!val || isNaN(val)) return;
    try {
      const res = await postJSON('/api/target/set_frame', {
        which, frame: val, target_media_id: activeTargetMediaId,
      });
      if (res.targets) setTargets(res.targets);
      notify(`Set ${which} frame = ${val}`);
    } catch (e) { notify(e.message, 'error'); }
  };

  // ── start / stop ──
  const start = async () => {
    // Scene overrides inside the In/Out range cannot ride in one payload (a run has one trim range
    // and one set of settings), so the range goes out as one job per piece and the queue joins
    // them. Without this a Start with overrides on the timeline would quietly render the clip's own
    // settings everywhere - the timeline would say one thing and the file another.
    if (overrideRangePlan().has) { await queueOverridePieces({ run: true }); return; }
    runCtxRef.current = { start: startFrame, end: endFrame, fp: fingerprint(buildSwapPayload()) };
    try {
      await postJSON('/api/settings', p);            // persist CFG
      await postJSON('/api/swap', buildSwapPayload(), { timeout: 0 });
      setStartTime(Date.now());
      // Ask for desktop-notification permission so we can ping on completion if
      // the tab is backgrounded (no-op if already decided).
      try { if ('Notification' in window && Notification.permission === 'default') Notification.requestPermission(); } catch { /* ignore */ }
      // Optimistically flag processing so the old "Latest output" clears
      // immediately, before the first poll tick (~1s) confirms it.
      setProgress((pr) => ({ ...pr, processing: true, paused: false, pause_requested: false, stop_requested: false, progress: 0, desc: 'Starting…' }));
      notify('Processing started');
    } catch (e) { runCtxRef.current = null; notify(e.message, 'error'); }
  };

  // Stop stays here because the floating dock offers it from this tab. Pause and
  // Resume moved to the Processing tab with the run bar that carried them.
  const stop = onStopRun || (async () => { await postJSON('/api/stop', {}); notify('Stopping…', 'info'); });

  // The run bar's one primary control. While a job is in flight it BECOMES the
  // cancel control, in the same place, so a double-click on Start would land on
  // Cancel -- hence the confirm (the header chip's Stop asks the same question).
  const confirmStop = async () => {
    if (await confirmDialog({ title: 'Stop job?', message: 'Stop the active job? The partial output so far is finalized and kept.', confirmLabel: 'Stop', danger: true })) {
      stop();
    }
  };

  // WHY Start is unavailable, in words. One derivation feeds both the button's
  // `disabled` and the sentence beside it, so they cannot disagree. Empty means
  // Start is available (or a run is already in flight, which the bar shows itself).
  const startBlockedReason = progress.processing ? ''
    : targets.length === 0 ? 'Add a target video or image to start.'
    : sourceFaces.length === 0 ? 'Add a source face to start.'
    : '';

  // Hide the previous "Latest output" while a job is running so a new upload +
  // run never shows a stale result. The poll keeps reporting the old _last_output
  // until the new job finishes, so we gate on `processing` rather than clearing
  // progress.output (which the next poll tick would just restore).
  const out = progress.processing ? null : progress.output;
  // Versioned by the file itself (see outputUrl.js). It used to carry
  // `?t=${progress.progress}`, which tied the URL of a finished file to a
  // number that has nothing to do with it.
  const outUrl = outputMediaUrl(out);

  // The render's "N% · time left" readout is a <LiveText> subscribed to the
  // telemetry store (see the header strip below): computing it here re-rendered
  // this whole panel on every telemetry frame.
  // While actively scrubbing the timeline (or playing back) each frame is a
  // fresh server fetch, so request a lightweight downscaled frame — a full-res
  // HD/4K JPEG per frame makes dragging feel sluggish. Snap back to full
  // resolution the instant the user settles on a frame.
  const scrubbingNow = isDraggingScrub || isPlaying;
  const rawReqUrl = targets.length > 0
    // `scrubbing=1` asks the backend for a proxy (<= 720 px tall, JPEG q85); when pointer movement
    // stops for 150 ms or releases, the settled frame is requested without it, at full size.
    // Keep `index=..&frame=..` first: rawIsCurrent reads them.
    ? `${API}/api/target/preview?index=${selTarget}&frame=${frame}${scrubbingNow ? '&width=960&scrubbing=1' : ''}`
    : '';
  // Load raw frames one at a time, latest wins (see useSequentialImage). Binding
  // an <img> directly to rawReqUrl issued a request per intermediate frame of a
  // drag, and since each one is a video seek on a single shared decoder, the
  // frame you stopped on arrived behind every frame you swept past.
  // Throttled to 150 ms, one request in flight, and — the part useSequentialImage
  // could not do — the superseded request is ABORTED rather than left to finish
  // into a frame nobody will look at. See useThrottledFrameRequest.
  const {
    frame: rawFrameBitmap, frameSrc: loadedRawUrl,
    error: rawFrameError, retry: retryRawFrame,
  } =
    // NOT while playing: playback frames come from the buffered player (see
    // usePlaybackBuffer). Left enabled, every played frame also queued a
    // random-access still request on the same single decoder the playback
    // stream reads sequentially — a seek per request, fighting the stream.
    useThrottledFrameRequest(rawReqUrl, { throttleMs: 150, enabled: !isPlaying, sequence: true });
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

  // Hand the stage back from the playback layer. After Pause the layer keeps
  // showing the frame playback stopped on — the still underneath is still the
  // frame playback STARTED on — and is cleared only once that still has
  // caught up to the playhead, so stopping never flashes back to an old frame.
  useEffect(() => {
    if (!isPlaying && rawIsCurrent) clearPlaybackFrame();
  }, [isPlaying, rawIsCurrent, clearPlaybackFrame]);

  // What the stage's "after" layer shows, in strict preference order. The rule
  // this encodes is that the picture may only ever move FORWARD onto the frame
  // the playhead is on — never back onto one already left. Stepping used to
  // show the previous frame's swap (previewSrc lags by a whole render) and then
  // jump forward once the new one landed, which read as the image flicking
  // between two frames.
  const stageAfterSrc = (() => {
    // The playback layer covers the stage while playing; hold the still
    // underneath rather than churning decodes nobody can see.
    if (isPlaying) return rawUrl;
    // Mid-drag there is no render for most frames and none is coming; the
    // freshest decoded raw frame is what keeps a scrub feeling continuous.
    if (scrubbingNow) return getCachedPreview(selTarget, frame)?.image || rawUrl;
    if (previewSrc && previewFor === `${activeTargetMediaId || `legacy-index-${selTarget}`}_${frame}`) return previewSrc;
    const cached = getCachedPreview(selTarget, frame)?.image;
    if (cached) return cached;
    if (rawIsCurrent) return loadedRawUrl;   // right frame, not swapped yet
    // The new frame has not decoded yet either: hold whatever is already up
    // rather than blanking or bouncing. It is replaced the moment either of
    // the two branches above can answer.
    return previewSrc || rawUrl;
  })();

  // The raw frame for the playhead failed every retry, so whatever the stage
  // shows is NOT the frame asked for -- say so rather than leave a stale picture
  // that looks current. A swapped preview (or a cached one) for THIS frame is the
  // right picture, and then there is nothing to apologise for.
  const frameUnavailable = !!rawFrameError && !rawIsCurrent
    && !(previewSrc && previewFor === `${activeTargetMediaId || `legacy-index-${selTarget}`}_${frame}`)
    && !getCachedPreview(selTarget, frame)?.image;

  // ── Stage 23: A/B/C/D snapshot slots ─────────────────────────────────────
  // A slot keeps the recipe state and, when this frame has a rendered preview, a decoded copy of
  // it. Recalling applies the recipe through the history (one undoable entry) and holds the kept
  // frame on the stage until the real render of that recipe lands, so the flip is instant.
  const frameKeyNow = `${activeTargetMediaId || `legacy-index-${selTarget}`}_${frame}`;
  const [heldSlot, setHeldSlot] = useState(null);
  const slotBank = useSnapshotSlots({
    bank: 'faceswap',
    notify,
    captureFrame: async () => {
      // Only a RENDERED preview of THIS frame is worth keeping: the raw frame is not "the look".
      if (!(previewSrc && previewFor === frameKeyNow)) return null;
      return { bitmap: await decodeFrame(previewSrc), frameKey: frameKeyNow };
    },
    onRecalled: (slot) => {
      setHeldSlot(slot.bitmap && slot.frameKey === frameKeyNow
        ? { id: slot.id, bitmap: slot.bitmap, previewSrc, frameKey: frameKeyNow, headId: useHistoryStore.getState().headId }
        : null);
      if (slot.frameKey && slot.frameKey !== frameKeyNow) {
        notify(`Slot ${slot.id} was captured on another frame: its recipe is applied, its picture is not shown`, 'info');
      }
    },
    onReference: (id) => {
      if (!id) return;
      // A reference only means something inside the compare wipe, so open it.
      setCompare(true);
      setComparingEnhancers(false); setComparingMasks(false); setComparingSwappers(false); setComparingUpscalers(false);
    },
  });
  const headIdNow = useHistoryStore((s) => s.headId);
  const heldStoreSlot = useSlotStore((s) => (heldSlot ? s.banks.faceswap[heldSlot.id] : null));
  const referenceId = useSlotStore((s) => s.reference.faceswap);
  const referenceSlot = useSlotStore((s) => (s.reference.faceswap ? s.banks.faceswap[s.reference.faceswap] : null));
  // The held picture is good only while NOTHING has moved on: same frame, same preview on the
  // stage, no edit since the recall, and the slot still owns that bitmap.
  const heldFrame = heldSlot && !isPlaying && !scrubbingNow
    && heldSlot.frameKey === frameKeyNow && heldSlot.previewSrc === previewSrc
    && heldSlot.headId === headIdNow && heldStoreSlot?.bitmap === heldSlot.bitmap
    ? heldSlot.bitmap : null;
  const referenceFrame = referenceSlot?.bitmap && referenceSlot.frameKey === frameKeyNow ? referenceSlot.bitmap : null;

  // Keep a detached pop-out monitor in sync. It used to receive exactly one
  // frame — the one it was opened with — because nothing ever called
  // sendUpdate() again, so it sat frozen on that image while the main window
  // carried on. Only broadcasts while a pop-out is actually open: a rendered
  // preview is a multi-MB data URI.
  useEffect(() => {
    if (!popoutManager.isOpen()) return undefined;
    const src = previewSrc || rawUrl;
    if (!src) return undefined;
    let cancelled = false;
    // A blob URL is a handle into THIS document's blob store. The pop-out is a
    // separate window with its own lifetime — it outlives this component, and
    // this component revokes what it owns on unmount — so it is sent the bytes.
    // http(s) frame URLs are already self-contained and pass straight through.
    (async () => {
      const payload = src.startsWith('blob:') ? await blobUrlToDataUrl(src) : src;
      if (cancelled || !payload) return;
      popoutManager.sendUpdate({ type: 'UPDATE_PREVIEW', src: payload });
    })();
    return () => { cancelled = true; };
  }, [previewSrc, rawUrl]);

  // Park the current still where the Processing tab can find it. That tab is
  // mounted only while this one is not, and its live-frame peek needs something
  // to show for the window before the pipeline publishes its first frame — see
  // faceswap/lastPreview.
  useEffect(() => {
    let cancelled = false;
    // Coordinates first, so the processing tab has the right frame numbers even
    // if the image re-serialisation loses a race with a tab switch.
    setLastPreview({ rawUrl, frame, maxFrames });
    (async () => {
      // Bytes, not the blob URL — see the header of lastPreview.js. This runs
      // once per RENDERED preview (a GPU swap costing hundreds of ms), never
      // per scrub tick, so the re-encode is not on any hot path.
      const bytes = previewSrc.startsWith('blob:')
        ? await blobUrlToDataUrl(previewSrc)
        : previewSrc;
      if (!cancelled) setLastPreview({ previewSrc: bytes });
    })();
    return () => { cancelled = true; };
  }, [previewSrc, rawUrl, frame, maxFrames]);

  const revealOutput = async () => {
    try { await postJSON('/api/reveal', { path: out?.path }); }
    catch (e) { notify(e.message, 'error'); }
  };

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
    // Mid-drag the hover line and thumbnail would sit exactly under the playhead the user is
    // already moving. Updating them anyway was a React commit of this whole panel per pointer
    // move, for a readout nobody can see.
    if (isScrubbing) return;
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

  // Commit the newest dragged-to frame into state now, if one is waiting.
  const SCRUB_COMMIT_MS = 100;
  const commitScrubFrame = () => {
    const c = scrubCommitRef.current;
    if (c.timer) { clearTimeout(c.timer); c.timer = null; }
    const f = c.pending;
    if (f === null) return;
    c.pending = null;
    c.last = performance.now();
    setFrame(f);
  };
  // Move the playhead to `f` at pointer rate and commit the frame at most every SCRUB_COMMIT_MS
  // (a leading commit, then a trailing one for whatever the pointer settles on).
  const scrubPlayheadTo = (f) => {
    playheadApiRef.current?.(f);
    const c = scrubCommitRef.current;
    c.pending = f;
    const wait = SCRUB_COMMIT_MS - (performance.now() - c.last);
    if (wait <= 0) commitScrubFrame();
    else if (!c.timer) c.timer = setTimeout(commitScrubFrame, wait);
  };
  useEffect(() => () => {
    const c = scrubCommitRef.current;
    if (c.timer) { clearTimeout(c.timer); c.timer = null; }
    if (scrubDebounceTimerRef.current) {
      clearTimeout(scrubDebounceTimerRef.current);
      scrubDebounceTimerRef.current = null;
    }
  }, []);

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
      if (dragType === 'playhead') {
        // Pointer rate: move the playhead only; the frame is committed on a slower clock (see scrubPlayheadTo).
        scrubPlayheadTo(Math.max(1, Math.min(targetFrame, maxFrames)));
        return;
      }
      updateTimelinePos(targetFrame, dragType);
    };
    const handlePointerMove = (e) => {
      const clientX = e.clientX ?? e.touches?.[0]?.clientX;
      if (clientX === undefined) return;
      pendingX = clientX;
      if (dragType === 'playhead') {
        setIsDraggingScrub((prev) => (prev ? prev : true));
        if (scrubDebounceTimerRef.current) clearTimeout(scrubDebounceTimerRef.current);
        scrubDebounceTimerRef.current = setTimeout(() => {
          setIsDraggingScrub(false);
        }, 150);
      }
      if (rafId === null) rafId = requestAnimationFrame(flush);
    };

    const handlePointerUp = () => {
      if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
      if (scrubDebounceTimerRef.current) {
        clearTimeout(scrubDebounceTimerRef.current);
        scrubDebounceTimerRef.current = null;
      }
      setIsDraggingScrub(false);
      // Land on the frame the pointer was released over: commit it in the same batch that ends the
      // scrub, so the full-resolution request goes out for THIS frame and not the last committed one.
      if (dragType === 'playhead') commitScrubFrame();
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
      if (scrubDebounceTimerRef.current) {
        clearTimeout(scrubDebounceTimerRef.current);
        scrubDebounceTimerRef.current = null;
      }
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, [isScrubbing, dragType, selTarget, maxFrames, targets]);
  /* eslint-enable react-hooks/exhaustive-deps */

  const handleTimelinePointerDown = (e) => {
    if (!timelineRef.current) return;
    // Primary button only: the lanes give right-click a menu and middle-click a pan, and neither
    // should also drag the playhead to wherever the pointer happened to be.
    if (e.button !== undefined && e.button !== 0) return;
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
    if (dragTarget === 'playhead') {
      // The click itself commits at once; the drag that follows is paced from here.
      scrubCommitRef.current.pending = null;
      scrubCommitRef.current.last = performance.now();
    }
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
      await postJSON('/api/target/set_frame', {
        which: 'start', frame: previewStart, target_media_id: activeTargetMediaId,
      });
      await postJSON('/api/target/set_frame', {
        which: 'end', frame: previewEnd, target_media_id: activeTargetMediaId,
      });

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
      }, { timeout: 0 });

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
          await postJSON('/api/target/set_frame', {
            which: 'start', frame: origStartEnd.start, target_media_id: activeTargetMediaId,
          });
          await postJSON('/api/target/set_frame', {
            which: 'end', frame: origStartEnd.end, target_media_id: activeTargetMediaId,
          });
          
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

      // Space ACTIVATES a focused button/link/tab -- it is the keyboard's click.
      // This handler used to claim it for play/pause unconditionally and call
      // preventDefault, which suppresses the activation: with focus on "Start
      // Swapping" (or any button on this tab) Space toggled playback and the
      // button never fired. Enter was unaffected, so mouse users never noticed.
      // A character-key shortcut must not override a focused control's own key
      // (WCAG 2.1.4); Space plays/pauses only when focus is on nothing in particular.
      if (e.key === ' ' && document.activeElement?.closest?.('button, a[href], summary, [role="button"], [role="tab"], [role="menuitem"], [role="switch"]')) return;

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

      // 'C' opens the timeline's Curve Editor (Stage 25) and Shift+C is the comparison view it displaced - this handler
      // stays the one owner of the bare C key (test_ui_shortcut_keys). A still image has no timeline to hang a curve
      // on, so there plain C keeps doing what it always did. 'S' toggles split view.
      if (e.key.toLowerCase() === 'c') {
        if (e.ctrlKey || e.metaKey || e.altKey) return;       // Ctrl/Cmd+C is copy
        e.preventDefault();
        if (!e.shiftKey && maxFrames > 1) { setCurveOpen((open) => !open); return; }
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
    maskOverlay: () => set('show_mask_offsets', !p.show_mask_offsets),
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

  // ── Timeline: finished renders, scene overrides, and the actions on them ──
  // A render's fingerprint names the settings it used. The cache strip calls a frame "rendered" only
  // when a finished render of it used the settings in force AT THAT FRAME now (the clip's own, or the
  // clip's with a scene override on top), so moving a slider turns the strip amber again instead of
  // leaving it green. Recomputed only when those inputs change, not on every playback tick.
  //
  // Stage 25: the pieces are the scene overrides' AND the parameter curves' (a curve is rendered as steps). The cut
  // itself depends only on the overrides, the curves and the range, so it is memoised apart from the fingerprints:
  // dragging a slider must not re-walk every frame of a long clip.
  const clipPlan = React.useMemo(
    () => bakeCurvePieces(overridePieces(sceneOverrides.activeOverrides, 1, Math.max(1, maxFrames)), curves.activeTracks),
    [sceneOverrides.activeOverrides, curves.activeTracks, maxFrames]);
  const rangeIsClip = startFrame <= 1 && endFrame >= maxFrames;
  const rangePlan = React.useMemo(
    () => (rangeIsClip ? clipPlan
      : bakeCurvePieces(overridePieces(sceneOverrides.activeOverrides, startFrame, endFrame), curves.activeTracks)),
    [rangeIsClip, clipPlan, sceneOverrides.activeOverrides, curves.activeTracks, startFrame, endFrame]);
  const pieceBase = (base, pc) => applyCurveValues(patchPayload(base, pc.override), pc.curves);
  const timelinePieces = React.useMemo(() => {
    const base = buildSwapPayload();
    return clipPlan.pieces.map((pc) => ({ start: pc.start, end: pc.end, fp: fingerprint(pieceBase(base, pc)) }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [p, selSource, activeTargetMediaId, selectionVersion, faceMapping, targetPersonIds, maskJson,
    identityBlend, clipPlan]);

  // The range an in-flight run is working through, for the strip filling in as frames land. A queued
  // piece reports its own range; a direct run reports the In/Out.
  const liveJob = progress.processing
    ? queue.jobs.find((j) => jobState(j) === 'PROCESSING' && j.frame_start != null && j.frame_end != null)
    : null;
  const liveRange = progress.processing ? (() => {
    const lo = liveJob ? liveJob.frame_start : startFrame;
    const hi = liveJob ? liveJob.frame_end : endFrame;
    const frac = Math.max(0, Math.min(1, Number(progress.progress) || 0));
    return { start: lo, end: hi, done: Math.round(frac * Math.max(0, hi - lo + 1)) };
  })() : null;

  // A direct run's range and settings, captured at Start, recorded when it completes cleanly (the
  // same "finished" test App.jsx uses for its completion chime).
  const runCtxRef = useRef(null);
  const wasProcessingRef = useRef(false);
  useEffect(() => {
    const was = wasProcessingRef.current;
    wasProcessingRef.current = !!progress.processing;
    if (!was || progress.processing) return;
    const ctx = runCtxRef.current;
    runCtxRef.current = null;
    if (ctx && !progress.error && (progress.progress || 0) >= 0.99) renderLog.record(ctx);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [progress.processing]);

  // Queued jobs that finished: each is a render of its frame range with its own payload.
  const loggedJobsRef = useRef(new Set());
  useEffect(() => {
    for (const j of queue.jobs) {
      if (jobState(j) !== 'COMPLETED' || j.frame_start == null || j.frame_end == null || !j.payload) continue;
      if (activeTargetMediaId && j.target_media_id !== activeTargetMediaId) continue;
      if (loggedJobsRef.current.has(j.id)) continue;
      loggedJobsRef.current.add(j.id);
      renderLog.record({ start: j.frame_start, end: j.frame_end, fp: fingerprint(j.payload) });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queue.jobs, activeTargetMediaId]);

  /**
   * The In/Out range cut into the pieces its scene overrides and parameter curves define. `has` = any piece differs
   * from the clip's own settings (a curve that sits exactly on its slider changes nothing and costs no extra job).
   */
  const overrideRangePlan = () => {
    const { pieces } = rangePlan;
    return { pieces, has: pieces.some((pc) => pc.override) || piecesHaveCurves(pieces, buildSwapPayload()) };
  };

  // The Curve Editor's view of all this. Identity Morph Weight is the one curve that is not a settings key: it is
  // compiled into the Identity Keyframe Track (the first two sources of the blender, 0 -> 1), which the backend
  // already renders per frame, with the curve's own easing.
  const morphSources = identityBlend.components || [];
  const morphKeys = curves.tracks.identity_morph || [];
  const curveEditor = {
    open: curveOpen,
    onToggle: () => setCurveOpen((open) => !open),
    api: curves,
    tracks: curves.tracks,
    enabled: curves.enabled,
    settings: p,
    planNote: describeBake(rangePlan),
    morph: {
      ready: morphSources.length >= 2,
      note: morphSources.length >= 2
        ? `Written into the Identity Keyframe Track as ${identityTrack.labelOf(morphSources[0].source_id)} (0) to ${identityTrack.labelOf(morphSources[1].source_id)} (1); the renderer follows it frame by frame.${
          identityKeyframes.keyframes.length ? ` This replaces the ${identityKeyframes.keyframes.length} identity keyframe${identityKeyframes.keyframes.length === 1 ? '' : 's'} on the clip now.` : ''}${
          identityBlend.enabled ? '' : ' The Identity Blender is off, so nothing applies until it is turned on.'}`
        : 'Needs an Identity Blender with two or more sources: the curve is written into the Identity Keyframe Track as a weight between the first two.',
      title: 'Compile this curve into the Identity Keyframe Track',
      onApply: async () => {
        const had = identityKeyframes.keyframes.length;
        if (had && !(await confirmDialog({
          title: 'Replace identity keyframes?',
          message: `The clip has ${had} identity keyframe${had === 1 ? '' : 's'}. Writing the morph curve replaces them with ${morphKeys.length}.`,
          confirmLabel: 'Replace', danger: true,
        }))) return;
        identityKeyframes.replace(morphToIdentityKeyframes(morphKeys, morphSources, maxFrames));
        notify(`Wrote ${morphKeys.length} identity keyframe${morphKeys.length === 1 ? '' : 's'} from the morph curve`, 'info');
      },
    },
  };

  // ── Stage 25: the right-click menu on a detected face ────────────────────────
  // Where a "bypass this face" override would go: the scene under the playhead, as the timeline draws it (its cuts
  // merge the scan's and the backend's); the page's own scan cuts when the timeline deck is not mounted.
  const sceneApiRef = useRef(null);
  const bypassRange = () => {
    const api = sceneApiRef.current;
    const own = scenesFromCuts(sceneCuts, maxFrames);
    const s = (api ? api.sceneAt(frame) : null)
      || (() => { const sc = sceneAt(own, frame); return sc ? { start: sc.start, end: sc.end, label: `Scene ${sc.index + 1}` } : null; })()
      || { start: 1, end: Math.max(1, maxFrames), label: 'Whole clip' };
    const known = api ? api.hasCuts : sceneCuts.length > 0;
    return {
      ...s,
      text: known ? `${s.label}: frames ${s.start}-${s.end}` : `Whole clip, frames ${s.start}-${s.end} (no scene cuts found yet)`,
    };
  };
  const bypassInfoFor = (faceIndex) => {
    const info = previewFaceRoles[faceIndex];
    if (!info || info.role === 'bystander') return { reason: 'This face is not swapped, so there is nothing to bypass', rangeText: '' };
    if (!canBypass(p.face_detection_mode)) {
      return { reason: 'Needs the "Selected face" detection mode: the other modes have no match threshold to close', rangeText: '' };
    }
    const swapped = previewFaceRoles.filter((r) => r && r.role !== 'bystander').length;
    if (swapped > 1) {
      return { reason: `${swapped} people are swapped on this frame and a bypass covers the whole scene - use Skip for this one person`, rangeText: '' };
    }
    return { reason: '', rangeText: bypassRange().text };
  };

  // Map one captured person to a source (or, with -1, to "never swap"). The same rules PersonGroups' dropdown follows:
  // see its setMapping for why the implicit "Selected face" person is written out first and why two mapped people
  // move the mode to "Selected people".
  const setPersonSource = (personId, sourceIndex) => {
    const sourceId = sourceIndex >= 0 ? sourceIdAt(sourceIndex) : null;
    const next = { ...faceMapping };
    if (p.face_detection_mode === 'Selected face' && selectedTargetPersonId && selectedTargetPersonId !== personId
        && !Object.prototype.hasOwnProperty.call(next, selectedTargetPersonId)) {
      const implicit = sourceIdAt(Number(selSource));
      if (implicit) next[selectedTargetPersonId] = implicit;
    }
    if (sourceId) next[personId] = sourceId;
    else Object.assign(next, excludePerson(next, personId));
    setFaceMapping(next);
    if (p.face_detection_mode === 'Selected face' && Object.keys(next).filter((k) => next[k]).length >= 2) {
      set('face_detection_mode', 'Selected people');
      notify('2+ people mapped - switched to "Selected people" so all of them swap', 'info');
    }
    commitTargetContext({ face_mapping: next, target_person_source_mapping: next });
    clearPreviewCache();
  };

  const faceMenu = {
    sourceNames: previewFaces.map((_, i) => (Number.isInteger(faceSources[i]) ? sourceNameAt(faceSources[i]) : null)),
    sources: sourceFaces.map((thumb, index) => ({ index, name: sourceNameAt(index) || `Face ${index + 1}`, thumb: typeof thumb === 'string' ? thumb : null })),
    bypassInfo: bypassInfoFor,
    assign: async (faceIndex, sourceIndex) => {
      // A box nobody has captured has no person to map yet: capture it first, as a click on the box does.
      let personId = previewFaceRoles[faceIndex]?.personId ?? null;
      if (!personId) personId = await captureTargetFaceFromFrame({ faceIndex });
      if (!personId) return;
      setPersonSource(personId, sourceIndex);
      notify(sourceIndex >= 0 ? `Face ${faceIndex + 1} now swaps from ${sourceNameAt(sourceIndex)}` : `Face ${faceIndex + 1} is never swapped`, 'info');
    },
    bypass: (faceIndex) => {
      const { reason } = bypassInfoFor(faceIndex);
      if (reason) { notify(reason, 'error'); return; }
      const r = bypassRange();
      const wasOff = sceneOverrides.enabled === false;
      if (wasOff) sceneOverrides.setEnabled(true);
      sceneOverrides.add({ start: r.start, end: r.end, bypass: true }, `Bypass ${trackLabel(previewPersonIds[faceIndex] ?? faceIndex)}`);
      notify(`Bypassing the swap on frames ${r.start}-${r.end} (Scene overrides lane)${wasOff ? ' - scene overrides were off and are on again' : ''}`);
    },
    // The crop goes through the same ingest as an uploaded image: the Face Manager detects the face in it, aligns it,
    // embeds it and adds it to its bank. Building the .fsz is still the Face Manager's "Build" step.
    extract: async (faceIndex, blob, meta) => {
      const file = new File([blob], `${(targets[selTarget]?.name || 'target').replace(/\.[^.]+$/, '')}_f${meta.frame}_face${faceIndex + 1}.png`, { type: 'image/png' });
      const res = await postFiles('/api/facemgr/add', [file], { detector: 'scrfd', restore: 'false' }, { timeout: 0 });
      const n = res?.faces?.length || 0;
      if (!n) throw new Error('the Face Manager found no face in that crop');
      notify(`Added Face ${faceIndex + 1} (frame ${meta.frame}, ${meta.rect.w}x${meta.rect.h}) to the Face Manager bank, now ${n} face${n === 1 ? '' : 's'}. Build the faceset in #/facemgr.`);
    },
    // m is faceContextModel.warmthDeltaFor: the chroma difference between the plate's skin and the swapped skin, fitted
    // with the renderer's own warmth gain. Warmth is relative to the slider, so a second run converges instead of
    // stacking: it measures what the CURRENT result still lacks.
    applySkin: (faceIndex, m) => {
      const measured = `a* ${m.da >= 0 ? '+' : ''}${m.da.toFixed(1)}, b* ${m.db >= 0 ? '+' : ''}${m.db.toFixed(1)}`;
      if (m.verdict === 'matched') { notify(`Skin tone already matches the plate (swapped skin is off by ${measured})`, 'info'); return; }
      if (m.verdict === 'tint') {
        notify(`Swapped skin is off by ${measured}, mostly a green / magenta tint that warmth cannot fix - try Saturation match or a different colour transfer mode`, 'info');
        return;
      }
      const current = num(p.skin_tone_warmth, 0);
      const next = nextWarmth(current, m.warmthDelta);
      if (next === current) { notify(`Skin tone warmth is already at its limit (${current}); swapped skin is still off by ${measured}`, 'info'); return; }
      set('skin_tone_warmth', next);
      notify(`Skin tone warmth ${current} to ${next} (swapped skin was off by ${measured}${m.residual > 2 ? `; ${m.residual} of it is tint warmth cannot move` : ''})`);
    },
  };

  // One queued job per piece, each with its own range and its own payload, stamped as one split
  // render so the queue joins the finished pieces into a single file by itself.
  const queueOverridePieces = async ({ run = false } = {}) => {
    if (targets.length === 0 || sourceFaces.length === 0) {
      notify('Load a target and a source face first', 'error');
      return false;
    }
    const { pieces, has } = overrideRangePlan();
    if (!has) { notify('No scene override or parameter curve changes anything inside the In/Out range', 'info'); return false; }
    if (pieces.some((pc) => pc.override?.bypass) && !canBypass(p.face_detection_mode)) {
      notify('Bypass swap needs the "Selected face" detection mode: in the all-faces modes there is no match threshold to close.', 'error');
      return false;
    }
    const base = buildSwapPayload();
    const name = targets[selTarget]?.name || '';
    const group = `ovr-${Date.now().toString(36)}`;
    await queue.addMany(pieces.map((pc, i) => currentJob({
      label: `${name} - ${pc.override ? pc.override.label : 'clip settings'}${pc.curves ? ' + curves' : ''} (${pc.start}-${pc.end})`,
      frame_start: pc.start,
      frame_end: pc.end,
      payload: pieceBase(base, pc),
      ...(pieces.length > 1 ? { segment: { group, index: i + 1, count: pieces.length } } : {}),
    })));
    notify(`Queued ${pieces.length} piece${pieces.length === 1 ? '' : 's'} of "${name}" with scene overrides`);
    if (run && !queue.running) await queue.start();
    return true;
  };

  // A pill on the presence lane is one person in one stretch: select that person (when they are one
  // of the target people) the way clicking their face box does. The timeline has already moved the
  // In/Out points and the playhead.
  const selectPresencePerson = (row) => {
    const records = targetPersonRecords({ targetGroups, targetPersonIds, targetReferenceFaceIds, targetNames });
    const record = records.find((r) => r.targetPersonId === row.id);
    if (!record) {
      notify(`${row.name} is not one of the target people yet - In/Out were set to their scene. Add them as a target face to swap them.`, 'info');
      return;
    }
    const firstFace = record.faceIndices[0] ?? 0;
    setSelectedTargetPersonId(record.targetPersonId);
    setSelectedReferenceFaceId(targetReferenceFaceIds[firstFace] || null);
    setSelTargetFace(firstFace);
    set('face_detection_mode', 'Selected face');
    commitTargetContext({
      selected_target_person_id: record.targetPersonId,
      ...(targetReferenceFaceIds[firstFace] ? { selected_reference_face_id: targetReferenceFaceIds[firstFace] } : {}),
      selected_target_face_index: firstFace,
    });
    notify(`Selected ${record.name || row.name} as the target person`);
  };

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
    threads: telemetryThreads,
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
    liveFps, setLiveFps,
    liveInterval, setLiveInterval,
    liveAudio, setLiveAudio,
    liveAudioInput, setLiveAudioInput,
    liveAudioOutput, setLiveAudioOutput,
    liveAudioDevices, liveStatus,
    startLiveCam, stopLiveCam,
  } = useLiveCam({ notify });

  // Derived values for the estimation box.
  const estFps = targets[selTarget]?.fps || 0;
  const estDurationS = estFps ? estFrames / estFps : 0;
  const estSourceLabel = estLearned ? 'Learned'
    : (calibEst?.source === 'global' ? 'Global avg' : 'Heuristic');
  const estSourceClass = estLearned ? 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10'
    : (calibEst?.source === 'global' ? 'text-amber-400 border-amber-500/30 bg-amber-500/10'
      : 'text-muted border-white/10 bg-white/5');

  return (
    // pb-28: the floating dock is fixed over the bottom ~80px of the viewport. Without
    // this room the last row of every column could be scrolled no higher than the
    // dock's own top edge, so whatever sat there (the final setting, a button) was
    // unreachable by mouse. See also the scroll-padding effect near the dock hook.
    <div className="flex flex-col lg:flex-row gap-4 items-start w-full pb-28">

      {/* COLUMN 1: the inspector — sticky sidebar on large viewports so it follows
          the scroll while scrolling a taller workspace.

          Five named groups instead of one 100-control column: Detection, Masking,
          Restoration & Color (the four-stage enhancer chain), Temporal, and the
          rest. The groups are InspectorGroup accordions that remember their open
          state; every control in them is the one that used to sit in the long
          column, moved, not rewritten.

          This keeps its layout during a run: watching a render and setting up the
          next job are two different tabs, so the run has the Processing tab to
          itself. */}
      <div
        data-testid="inspector"
        className={`w-full lg:w-[350px] 3xl:w-[400px] 4xl:w-[460px] shrink-0 space-y-3 select-none ${!showLeftPanel ? 'hidden' : ''}`}
      >
        <RecipePresetBar
          recipes={QUICK_RECIPES}
          activeId={activeRecipe}
          onApply={applyRecipe}
          profiles={profiles}
          newProfileName={newProfileName}
          onNameChange={setNewProfileName}
          onSaveProfile={saveProfile}
          onLoadProfile={loadProfile}
          onDeleteProfile={deleteProfile}
          userDefaults={userDefaults}
          onSaveDefault={saveAsDefault}
          onResetDefaults={resetToDefaults}
          onClearDefault={clearUserDefault}
          onExportProfiles={exportProfiles}
          onImportProfiles={importProfiles}
          onShareRecipe={exportRecipe}
          onLoadRecipe={importRecipe}
        />

        {/* The 3-stage pipeline's four levers, shared with #/settings: swapper, restorer mode, pre-pass stride K,
            Temporal Smart-Skip. The swapper select that sat here, and the auto-bypass toggle further down, are
            part of it now. There is no run button in it: the run bar stays the one control that starts a render. */}
        <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-3 space-y-3" role="group" aria-labelledby="pipeline-controls-heading">
          <div id="pipeline-controls-heading" className="text-nano font-bold uppercase tracking-[0.16em] text-white/60">3-stage pipeline</div>
          <PipelineControls values={p} models={meta.swap_models} onChange={set} onChangeMany={setMany} />
        </div>

        <InspectorGroup id="detection" title="Detection" icon={Icon.face} defaultOpen>
          <Select label="Face selection" value={p.face_detection_mode} onChange={(v) => set('face_detection_mode', v)} options={meta.face_detection_modes} />
          <Select
            label="Detector engine"
            info="RetinaFace R50 is the safest pick: it was the only engine that held full recall as a face grew to fill the frame, where SCRFD (the stock default) and RetinaFace 10g both dropped to zero. Prefer it unless you need the speed. YOLOFace is often better on steep profiles and partially occluded faces. YuNet is the lightest. SCRFD is fast and accurate on frontal faces but the weakest on close-ups. All engines reuse the same identity/landmark models; alternates download a small model on first use."
            value={p.detector_engine || 'scrfd'}
            onChange={(v) => set('detector_engine', v)}
            options={meta.detector_engines || ['scrfd', 'yoloface', 'retinaface', 'retinaface_r50', 'retinaface_r50_gpu', 'yunet']}
          />
          <Slider
            label="Face detection threshold"
            info="Lower values (e.g. 0.40) detect angled, profile, or hard-to-see faces. Higher values avoid false detections. Default 0.50"
            min={0.10}
            max={0.90}
            step={0.05}
            value={num(p.face_detector_threshold, 0.50)}
            onChange={(v) => set('face_detector_threshold', v)} defaultValue={FACESWAP_DEFAULTS.face_detector_threshold}
          />
          <Disclosure id="inspector.detector-tuning" title="Advanced Detector Tuning">
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
              label="Overlap NMS threshold"
              info="Lower values (e.g. 0.30) suppress close duplicates. Higher values (e.g. 0.60) allow detecting overlapping faces close to each other. Default 0.40"
              min={0.10}
              max={0.90}
              step={0.05}
              value={num(p.face_detector_nms, 0.40)}
              onChange={(v) => set('face_detector_nms', v)} defaultValue={FACESWAP_DEFAULTS.face_detector_nms}
            />
            <Toggle label="Refine alignment (68-pt)" info="Derives the alignment keypoints from the 68-point landmark model instead of the detector's raw 5 points — more stable alignment on angled faces, less residual swap wobble. Small per-face cost." checked={!!p.refine_landmarks} onChange={(v) => set('refine_landmarks', v)} />
            <Toggle label="Rescue small faces" info="When a frame has no detected face, retries on a 2x upscale to catch tiny/distant faces — without raising the global detection resolution for every frame." checked={!!p.rescue_small_faces} onChange={(v) => set('rescue_small_faces', v)} />
            <Select label="On no face detected" value={p.no_face_action} onChange={(v) => set('no_face_action', v)} options={meta.no_face_actions} />
          </Disclosure>
          <Disclosure id="inspector.identity-matching" title="Identity matching">
            <Slider label="Max face distance" info="How far from your captured target face a detected face may sit and still be swapped. This is a DISTANCE, not a similarity — 0 means identical and HIGHER IS MORE PERMISSIVE, so raising it makes look-alikes and bystanders start getting swapped. Scale is scipy cosine distance (0–2). Measured on a hard clip: the SAME person stays under ~0.66 even on bad frames, while DIFFERENT people sit at ~0.93–1.07. The 0.75 default sits mid-gap. Lower it toward 0.66 if the swap jumps to the wrong person mid-shot; raise it toward 0.85 if the swap blinks off on profiles and motion blur. Run with ROOP_DEBUG_MATCH=1 to print the real per-frame distances." min={0.01} max={1} step={0.01} value={num(p.max_face_distance, 0.75)} onChange={(v) => set('max_face_distance', v)} defaultValue={FACESWAP_DEFAULTS.max_face_distance} />
            <Toggle label="Lock face identities (video)" info="For 'Selected face' mode on video: tracks each person across the clip and keeps them on one source, so identities don't flip frame-to-frame when faces cross or turn. Adds a short tracking pre-pass; the swap stays multi-threaded." checked={!!p.track_identities} onChange={(v) => set('track_identities', v)} />
            <Slider label="Identity confidence threshold" info="Minimum cosine similarity required to swap a face. If similarity drops below this (due to extreme profile angle or motion blur), the swap is cleanly skipped rather than mapping onto an incorrect person. 0.0 = disabled." min={0.0} max={1.0} step={0.05} value={num(p.identity_confidence_threshold, 0.0)} onChange={(v) => set('identity_confidence_threshold', v)} defaultValue={FACESWAP_DEFAULTS.identity_confidence_threshold} />
          </Disclosure>
        </InspectorGroup>

        <InspectorGroup id="masking" title="Masking" icon={Icon.layout}>
          <Select label="Masking engine" info="Choose this one on quality, not speed — the models are all cheap and within a few ms of each other. Measured on an RTX 4070, TensorRT FP16, isolated: Face Parser 2.4ms, DFL XSeg 2.9ms, XSeg-3 4.7ms, Face Occluder 5.0ms, MobileSAM 5.8ms+decoder, FastSAM 5.8ms. The whole spread is ~3ms against a masking stage that costs ~42ms per face, because almost all of that stage is the CPU work around the model — the landmark hull, the mouth mask, the blurs and the non-frontal unwarp — not the model itself. So switching engines to go faster will not work; switch when one of them handles YOUR occlusions better. SAM2 (tracked) is not comparable here: it runs a whole-clip pre-pass instead of per-crop inference." value={p.mask_engine} onChange={(v) => set('mask_engine', v)} options={meta.mask_engines} />
          <Select label="Second masking engine (occlusion)" info="A second, independent occlusion engine, run after the first. They compose as a union of &quot;not face&quot;, so this can only ever restore MORE of the original footage — it cannot eat into the swap, and turning it off is exactly the previous behaviour. Why you would: when something comes in front of a face and the swap gets painted over it anyway, the usual reason is not that masking is off, it is that the one engine you picked was not trained on that particular object. The engines miss different things — XSeg and XSeg-3 come from DeepFaceLab&apos;s hand/object data, Face Occluder from a different set again, Face Parser segments facial regions rather than occluders, and the SAM variants segment anything at all. Running two costs one more model call per face, which is 2–6ms against a masking stage of ~42ms. Start with DFL XSeg + Face Occluder. &quot;None&quot; = one engine, as before." value={p.mask_engine_2 || 'None'} onChange={(v) => set('mask_engine_2', v)} options={meta.mask_engines} />
          {String(p.mask_engine || '').startsWith('Face Parser') && (
            <ParserRegions
              regions={p.parser_regions}
              grow={p.parser_region_grow}
              onChange={({ regions, grow }) => setMany({ parser_regions: regions, parser_region_grow: grow })}
            />
          )}
          {!String(p.mask_engine || '').startsWith('Face Parser') && (
            <p className="text-mini leading-snug text-muted">The face-region pills appear when the masking engine is Face Parser (BiSeNet).</p>
          )}
          {p.mask_engine === 'Clip2Seg' && (
            <TextInput label="Objects to mask & restore" value={p.mask_clip_text} onChange={(v) => set('mask_clip_text', v)} placeholder="cup,hands,hair" />
          )}
          {p.mask_engine === 'Segment Anything 2 (tracked)' && (
            <Select label="SAM2 checkpoint (speed ↔ quality)" value={p.sam2_model_size || 'tiny'} onChange={(v) => set('sam2_model_size', v)} options={meta.sam2_model_sizes || ['tiny', 'small', 'base_plus', 'large']} />
          )}
          {p.mask_engine === 'RealityUX' && (
            <div className="flex flex-col gap-2 p-3 my-2 rounded-xl bg-slate-900/60 border border-white/10">
              <div className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                <span>🛡️</span> RealityUX Surgical Governance
              </div>
              <Toggle
                label="Restore original mouth aperture (MAR gated)"
                info="Dynamically restores inner mouth aperture, teeth, and tongue when MAR exceeds threshold during speech, smiles, or yawning."
                checked={!!p.restore_original_mouth}
                onChange={(v) => set('restore_original_mouth', v)}
              />
              {p.restore_original_mouth && (
                <Slider
                  label="Mouth open threshold (MAR)"
                  info="Mouth Aspect Ratio trigger threshold (default 0.18)"
                  min={0.05}
                  max={0.5}
                  step={0.01}
                  value={num(p.mouth_open_threshold, 0.18)}
                  onChange={(v) => set('mouth_open_threshold', v)} defaultValue={FACESWAP_DEFAULTS.mouth_open_threshold}
                />
              )}
              <Toggle
                label="DFL XSeg foreign object occlusion"
                info="Isolates hands, microphones, fingers, and foreign objects with morphological erosion and feathering."
                checked={!!p.enable_xseg_occlusion}
                onChange={(v) => set('enable_xseg_occlusion', v)}
              />
              {p.enable_xseg_occlusion && (
                <Slider
                  label="Occlusion erosion radius"
                  info="Morphological erosion kernel size (1 - 5 px) to eliminate boundary halos."
                  min={1}
                  max={5}
                  step={1}
                  value={num(p.occlusion_erosion, 3)}
                  onChange={(v) => set('occlusion_erosion', v)} defaultValue={FACESWAP_DEFAULTS.occlusion_erosion}
                />
              )}
            </div>
          )}
          <Toggle label="Protect eyeglass frames" info="Keeps the rim, bridge and temple arms of a pair of glasses from being painted over by the swap, while leaving the lens directly over each eye swappable so the eyes stay the faceset&apos;s own rather than the original person&apos;s. Needs a parser-backed masking engine (RealityUX or Face Parser) — the classes come from BiSeNet. Measured over ~800k parsed glasses pixels on three subjects: ~71% of the frame was being painted over, and the RealityUX fusion was contributing exactly nothing there, because XSeg reads ~0.003 on glasses and the accessory gate scales with XSeg. Costs no extra model call — it reuses the parse already being run." checked={p.glasses_frame_protect !== false} onChange={(v) => set('glasses_frame_protect', v)} />
          <Toggle label="Show mask overlay in preview" checked={!!p.show_mask_offsets} onChange={(v) => set('show_mask_offsets', v)} />
          <LinkedPaddingControl settings={p} onChangeMany={setMany} defaults={FACESWAP_DEFAULTS} />
          <Slider label="Face mask edge blend" min={0} max={200} step={1} value={num(p.face_mask_blend, 20)} onChange={(v) => set('face_mask_blend', v)} defaultValue={FACESWAP_DEFAULTS.face_mask_blend} />
          <Select label="Mask edge falloff" info="Shape of the ramp between swapped and original pixels. 'Gaussian' is the shipped behaviour and what 'Face mask edge blend' was calibrated against. 'Distance (soft)' derives the ramp from a distance transform, so it is the SAME WIDTH everywhere along the boundary — a Gaussian narrows wherever the mask is locally convex or thin, which is why one setting can look soft along the cheek and hard at the temple in the same frame." value={p.mask_edge_mode || 'gaussian'} onChange={(v) => set('mask_edge_mode', v)} options={[{ value: 'gaussian', label: 'Gaussian (default)' }, { value: 'distance', label: 'Distance (soft, constant width)' }]} />
          <Slider label="Boundary illumination match" info="Grades only the LOW frequencies of the pasted face toward the original plate, and only at the rim — exactly zero where the mask is fully opaque, so the identity region is untouched and no target texture is copied in. Use when a seam is still visible after the edge is soft: that residual line is a brightness step, not a geometry problem. 0 disables it entirely." min={0} max={1} step={0.05} value={num(p.boundary_illumination_strength, 0)} onChange={(v) => set('boundary_illumination_strength', v)} defaultValue={FACESWAP_DEFAULTS.boundary_illumination_strength} />
          <Slider label="Mask erode / dilate radius" info="Signed frame-pixel adjustment to the facial landmark convex hull before Gaussian feathering. Negative values erode the swap to keep hair, hands, microphones, and accessories out. Positive values dilate it to recover clipped skin. 0 preserves the detected contour." min={-32} max={32} step={1} value={num(p.mask_erode_dilate_radius, 0)} onChange={(v) => set('mask_erode_dilate_radius', v)} defaultValue={FACESWAP_DEFAULTS.mask_erode_dilate_radius} />
          <Slider label="🧩 Swap model's own face mask" info="Only hififace and hyperswap emit this — every other swapper ignores the setting, so it is safe to leave on. Those two nets output a face mask alongside the swapped image, saying where they actually synthesised a face, and the pipeline used to throw it away. That matters because the paste mask cannot know: it is an ellipse intersected with the convex hull of the 106 landmarks, whose forehead extension runs 60% above the eyebrows and therefore into the HAIR. Measured against hififace's own verdict, 15–27% of the paste mask is territory the model says is not face on a FRONTAL head, and 31% on a profile — and looking at where that excess lands, it is the band above the hairline frontally, and a wedge of hair plus background behind the head on a profile. That is very likely why eyes/nose/mouth read as distorted at angles: not the features themselves, but the invalid territory around them being composited. Not pose-gated, unlike the two angle controls above, because the net derives this from the actual image and its verdict is right at every angle. 100 = trust the model fully, 0 = off (previous behaviour), which is the default — this is new and has not been judged on real footage yet. It is the first thing to raise if a hififace or hyperswap paste reaches up into hair." min={0} max={100} step={5} value={num(p.swap_model_mask_strength, 0)} onChange={(v) => set('swap_model_mask_strength', v)} defaultValue={FACESWAP_DEFAULTS.swap_model_mask_strength} />
          <Toggle label="XSeg edge refinement (guided filter)" info="DFL XSeg only. Re-fits XSeg's 256px mask boundary to the edges of the face crop with a guided filter. Measured on 10 clips against an independent face parser it did NOT move the boundary closer (placement unchanged, agreement worse on every clip); it only softens the edge. Off by default; kept for comparison. About 1 ms of CPU per face." checked={!!p.mask_guided_filter} onChange={(v) => set('mask_guided_filter', v)} />
          <Disclosure id="inspector.mouth-mask" title="Mouth mask">
            <Slider label="Mouth mask top" min={0} max={2} step={0.01} value={num(p.mouth_top_scale, 1)} onChange={(v) => set('mouth_top_scale', v)} defaultValue={FACESWAP_DEFAULTS.mouth_top_scale} />
            <Slider label="Mouth mask bottom" min={0} max={2} step={0.01} value={num(p.mouth_bottom_scale, 1)} onChange={(v) => set('mouth_bottom_scale', v)} defaultValue={FACESWAP_DEFAULTS.mouth_bottom_scale} />
            <Slider label="Mouth mask left" min={0} max={2} step={0.01} value={num(p.mouth_left_scale, 1)} onChange={(v) => set('mouth_left_scale', v)} defaultValue={FACESWAP_DEFAULTS.mouth_left_scale} />
            <Slider label="Mouth mask right" min={0} max={2} step={0.01} value={num(p.mouth_right_scale, 1)} onChange={(v) => set('mouth_right_scale', v)} defaultValue={FACESWAP_DEFAULTS.mouth_right_scale} />
            <Slider label="Mouth mask edge blend" min={0} max={200} step={1} value={num(p.mouth_mask_blend, 10)} onChange={(v) => set('mouth_mask_blend', v)} defaultValue={FACESWAP_DEFAULTS.mouth_mask_blend} />
          </Disclosure>
        </InspectorGroup>

        <InspectorGroup id="restoration" title="Restoration & Color" icon={Icon.wand} defaultOpen>
          <Select label="Subsample upscale" value={p.subsample_upscale} onChange={(v) => set('subsample_upscale', v)} options={meta.upscale} />
          <EnhancerPipelineStack
            settings={p}
            onChange={set}
            meta={meta}
            slots={{
              restorer: (
                <>
                  <div role="group" aria-label="Restoration tier" className="space-y-1.5">
                    <div className="text-xs font-medium text-white/70">Restoration tier</div>
                    <div className="grid grid-cols-3 gap-1.5">
                      {RESTORATION_TIERS.map((tier) => (
                        <button
                          key={tier.id}
                          type="button"
                          aria-pressed={restorationTierOf(p) === tier.id}
                          onClick={() => setMany(tier.patch)}
                          title={tier.hint}
                          className={`rounded-lg border px-1.5 py-1.5 text-left transition-colors ${
                            restorationTierOf(p) === tier.id
                              ? 'border-[var(--accent)]/50 bg-[var(--accent)]/15 text-white'
                              : 'border-white/10 bg-white/[0.04] text-white/70 hover:border-white/25 hover:text-white'
                          }`}
                        >
                          <div className="text-mini font-semibold leading-tight">{tier.label}</div>
                          <div className="text-mini leading-tight text-muted">{tier.note}</div>
                        </button>
                      ))}
                    </div>
                  </div>
            {p.selected_enhancer === 'Restore Ultra' && (
              <>
                <Toggle label="Restore Ultra: keep the swap's tone and shape" info="Takes lighting, skin tone and the face's large-scale shape from the SWAP and only the fine detail (pores, lash and lip edges) from the restorer, instead of the restorer's whole output. Measured on 72 real faces: identity to the source 0.573 -> 0.613 (better on 67/72), skin texture 1.58x -> 1.09x that of the real footage. Off = the restorer's full output, as before." checked={p.restore_ultra_frequency_blend !== false} onChange={(v) => set('restore_ultra_frequency_blend', v)} />
                {p.restore_ultra_frequency_blend !== false && (
                  <>
                    <Slider label="Restore Ultra detail fidelity" info="How much of the restorer's fine detail goes on top of the swap. 0 = the swap's own detail, 1 = all of the restorer's. Measured: 0.5 keeps the most identity but reads softer than real footage; 1.0 is as textured as the plain restorer." min={0} max={1} step={0.05} value={num(p.restore_ultra_detail_weight, 0.75)} onChange={(v) => set('restore_ultra_detail_weight', v)} defaultValue={FACESWAP_DEFAULTS.restore_ultra_detail_weight} />
                    <Toggle label="Restore only eyes, brows, nose and lips" info="Applies the restorer only to the facial features, feathered into the swap's own skin. Keeps the most identity (0.612) but the skin is the swap's, which is softer than real footage (0.84x)." checked={!!p.restore_ultra_inner_only} onChange={(v) => set('restore_ultra_inner_only', v)} />
                  </>
                )}
              </>
            )}
            {p.selected_enhancer === 'UltraMax' && (
              <>
                <Toggle label="UltraMax: keep the swap's tone and shape" info="Takes lighting, skin tone and the face's large-scale shape from the SWAP and only the fine detail from the restorer, instead of the restorer's whole output. UltraMax's restorer re-draws the face's shading and shape on every frame, which showed as shimmer: measured on a 420-frame dance clip this removes about half of the frame-to-frame flicker. Turn it off for the old full-restorer look." checked={p.ultramax_swap_lowband !== false} onChange={(v) => set('ultramax_swap_lowband', v)} />
                {p.ultramax_swap_lowband !== false && (
                  <Slider label="UltraMax detail" info="How much of the restorer's fine detail goes on top of the swap. 0 = the swap's own detail, 1 = all of the restorer's. Measured flicker (lower is steadier): 0.5 = 1.4, 0.75 = 2.1 (default), 1.0 = 3.0, old full-restorer paste = 4.6; with no enhancer = 1.5. Higher is crisper and grainier, lower is smoother." min={0} max={1} step={0.05} value={num(p.ultramax_detail_weight, 0.75)} onChange={(v) => set('ultramax_detail_weight', v)} defaultValue={FACESWAP_DEFAULTS.ultramax_detail_weight} />
                )}
              </>
            )}
            {p.selected_enhancer === 'Adaptive' && (
              <Select
                label="Adaptive enhancer profile"
                info="Selects one existing restorer per face/frame; it never chains restorers. FAST, BALANCED, REALISTIC, and MAX QUALITY change the candidate preference, while quality, blur, pose, illumination, occlusion, confidence, temporal stability, and prior output quality remain the gates. Extreme angles, severe occlusion, unstable tracking, and VERY DARK frames stay unenhanced. Manual enhancer selections remain available."
                value={p.adaptive_enhancer_profile || 'BALANCED'}
                onChange={(v) => set('adaptive_enhancer_profile', v)}
                options={['FAST', 'BALANCED', 'REALISTIC', 'MAX QUALITY']}
              />
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
                onChange={(v) => set('codeformer_fidelity', v)} defaultValue={FACESWAP_DEFAULTS.codeformer_fidelity}
              />
            )}
            {(p.selected_enhancer === 'Tile Diffusion Synthesizer' || p.selected_enhancer === 'Tile Diffusion') && (
              <div className="space-y-3 p-3 rounded-xl bg-purple-950/20 border border-purple-500/20 mt-2 mb-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-purple-300 flex items-center gap-1.5">
                    <span>✨</span> Tile Diffusion Synthesizer (VFX / Master Quality)
                  </span>
                  <span className="text-micro px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-200 border border-purple-500/30 font-mono">
                    VRAM &lt; 8 GB · 512² Tiled (64px margin)
                  </span>
                </div>
                <Slider
                  label="Diffusion detail injection strength"
                  info="Constrains diffusion strictly to the high-frequency residual domain: keeps low-frequency color and global geometry fixed from the swapper, while synthesizing micro-textures (skin pores, hair follicles, lip crevices, sclera vessels)."
                  min={0.0}
                  max={1.0}
                  step={0.05}
                  value={num(p.tile_diffusion_strength, 0.65)}
                  onChange={(v) => set('tile_diffusion_strength', v)} defaultValue={FACESWAP_DEFAULTS.tile_diffusion_strength}
                />
                <Slider
                  label="Few-step diffusion steps"
                  info="Number of reverse diffusion denoising trajectory steps (1–4). Step 1 = ultra-fast single-step turbo mode; steps 2–4 = balanced high-fidelity micro-texture generation."
                  min={1}
                  max={4}
                  step={1}
                  value={num(p.tile_diffusion_steps, 2)}
                  onChange={(v) => set('tile_diffusion_steps', v)} defaultValue={FACESWAP_DEFAULTS.tile_diffusion_steps}
                />
                <div className="text-mini text-white/50 leading-relaxed bg-black/20 p-2 rounded border border-white/5">
                  <strong className="text-white/70">Dynamic Tiling Engine:</strong> Automatically partitions 4K crops into 512×512 overlapping tiles with 64px Gaussian feathered margins for seamless boundary reconstruction and zero seam artifacts.
                </div>
              </div>
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
                </>
              ),
              texture: (
                <>
            <Slider label="Skin detail transfer" info="Adds the ORIGINAL footage's real high-frequency texture (pores, stubble, grain) onto the swapped face. The generator smooths skin and the enhancer fakes flickery pores; this uses genuine detail from the scene instead. 0 = off. Start ~0.3–0.5; too high reintroduces the target's skin identity." min={0} max={1} step={0.05} value={num(p.detail_transfer_strength, 0)} onChange={(v) => set('detail_transfer_strength', v)} defaultValue={FACESWAP_DEFAULTS.detail_transfer_strength} />
            <Slider label="Sharpen / soften" info="Signed unsharp mask on the finished face. Positive sharpens; NEGATIVE softens, which is the denoise direction and useful when the enhancer has left crunchy over-detailed skin. 0 = off. Radius scales with crop size, so the effect looks the same at 256/512/1024. The cheapest control here at ~0.6 ms per face." min={-1} max={1} step={0.05} value={num(p.merger_sharpen, 0)} onChange={(v) => set('merger_sharpen', v)} defaultValue={FACESWAP_DEFAULTS.merger_sharpen} />
            <Toggle label="Match sharpness to the footage" info="Scales Sharpen and the clarity lift by how detailed the footage is compared with the face (times the headroom below), never above 1. On soft, compressed or motion-blurred shots a fixed sharpen makes the face crisper than everything around it and amplifies the swap's frame-to-frame detail into shimmer. Measured on a soft 1500-frame clip: 30% less shimmer, face detail 2.9x to 2.1x the footage's. It only ever turns sharpening DOWN; footage clearly more detailed than the face leaves it at your setting." checked={p.merger_match_sharpness !== false} onChange={(v) => set('merger_match_sharpness', v)} />
            {p.merger_match_sharpness !== false && (
              <Slider label="Sharpness headroom" info="Multiplies the limit. At 1.0, sharpening is cut only when the footage is softer than the face; at 0.7 (default) it is also trimmed to 70% when they match, which measured steadiest without looking soft; at 1.5 the limit almost never applies. Lower = smoother, less shimmer." min={0.25} max={2} step={0.05} value={num(p.merger_sharpness_headroom, 0.7)} onChange={(v) => set('merger_sharpness_headroom', v)} defaultValue={FACESWAP_DEFAULTS.merger_sharpness_headroom} />
            )}
                  <Slider label="Clarity" info="LAB micro-clarity on luminance plus a soft-knee bound on chrominance: crispens iris rims, lid creases and lip margins without ringing, and holds skin tone back from the neon-orange drift a restorer can add. Works with any enhancer. 1.0 is the shipped value." min={0} max={1} step={0.05} value={num(p.merger_clarity, 1)} defaultValue={FACESWAP_DEFAULTS.merger_clarity} onChange={(v) => set('merger_clarity', v)} />
            <Slider label="Degrade" info="Bicubic down-and-up on the finished face, to match a soft or heavily compressed plate. A crisp 512px swap dropped into 480p-grade or bitrate-starved footage is sharper than anything around it. 0 = off; at 1 the crop makes a round trip through 40% scale. Measured ~0.8 ms per face." min={0} max={1} step={0.05} value={num(p.merger_degrade, 0)} onChange={(v) => set('merger_degrade', v)} defaultValue={FACESWAP_DEFAULTS.merger_degrade} />
                </>
              ),
              color: (
                <>
            <Slider label="Skin tone warmth" info="Fine LAB chroma correction applied only inside a conservative skin-region estimate. Negative values cool toward blue/green, positive values warm toward red/yellow. 0 is neutral." min={-100} max={100} step={1} value={num(p.skin_tone_warmth, 0)} onChange={(v) => set('skin_tone_warmth', v)} defaultValue={FACESWAP_DEFAULTS.skin_tone_warmth} />
            <Slider label="Saturation match" info="Matches swapped skin saturation to the original target skin region using robust HSV statistics. 0 is off, 1 is a full match. Hair, eyes, lips, glasses, and background are excluded from the estimate." min={0} max={1} step={0.05} value={num(p.saturation_match, 0)} onChange={(v) => set('saturation_match', v)} defaultValue={FACESWAP_DEFAULTS.saturation_match} />
            <Slider label="Specular highlight recovery" info="Extracts high-luminance specular peaks (sweat sheen, rim light, directional flash highlights) from the target face's L-channel in LAB space and composites them over the restored swap so skin reacts naturally to scene lighting. 0 = off." min={0} max={1} step={0.05} value={num(p.specular_highlight_recovery, 0)} onChange={(v) => set('specular_highlight_recovery', v)} defaultValue={FACESWAP_DEFAULTS.specular_highlight_recovery} />
            <Slider label="Ambient shadow & SH match" info="Matches directional shadow falloff across the left/right/top/bottom hemispheres of the face ROI so side-lit dramatic scenes retain natural 3D lighting shape. 0 = off." min={0} max={1} step={0.05} value={num(p.ambient_shadow_match, 0)} onChange={(v) => set('ambient_shadow_match', v)} defaultValue={FACESWAP_DEFAULTS.ambient_shadow_match} />
            <Slider label="Sub-dermal warmth & halation" info="Matches shadow, midtone, and highlight A*/B* chrominance balance between target and swap within the skin mask, recovering subsurface red scatter and edge halation. 0 = off." min={0} max={1} step={0.05} value={num(p.subdermal_warmth_match, 0)} onChange={(v) => set('subdermal_warmth_match', v)} defaultValue={FACESWAP_DEFAULTS.subdermal_warmth_match} />
                </>
              ),
            }}
          />
          <Disclosure id="inspector.lighting" title="Target lighting">
            <Toggle label="Target-conditioned lighting" info="SOURCE supplies identity while TARGET controls exposure, white balance, scene color cast, spatial shadows, highlights and local contrast. It uses a low-frequency target field, never a wholesale texture paste. DARK/VERY DARK scenes reduce restoration and sharpening; VERY DARK avoids aggressive exposure correction. Temporal EMA prevents warm/neutral/blue flicker. Off preserves the legacy color-transfer path." checked={!!p.target_conditioned_appearance} onChange={(v) => set('target_conditioned_appearance', v)} />
            {p.target_conditioned_appearance && <>
              <Slider label="Appearance conditioning strength" min={0} max={1} step={0.05} value={num(p.target_conditioned_appearance_strength, 0.75)} onChange={(v) => set('target_conditioned_appearance_strength', v)} defaultValue={FACESWAP_DEFAULTS.target_conditioned_appearance_strength} />
              <Slider label="Appearance temporal stability" min={0.05} max={1} step={0.05} value={num(p.target_conditioned_appearance_temporal_alpha, 0.30)} onChange={(v) => set('target_conditioned_appearance_temporal_alpha', v)} defaultValue={FACESWAP_DEFAULTS.target_conditioned_appearance_temporal_alpha} />
            </>}
            <Toggle
              label="Light Harmonizer"
              info="Neural environment re-lighting and normal harmonization: extracts 3D surface normals, diffuse albedo, and specular environment maps from the scene, re-shading the swapped face to match scene key/fill lighting, restoring corneal catchlights, and casting dynamic shadows."
              checked={!!p.light_harmonizer}
              onChange={(v) => set('light_harmonizer', v)}
            />
            {p.light_harmonizer && (
              <>
                <Slider
                  label="Key light intensity"
                  info="Intensity of the directional key light projected onto the swapped face normals. 1.0 matches scene lighting."
                  min={0}
                  max={2}
                  step={0.05}
                  value={num(p.light_harmonizer_key_intensity, 1.0)}
                  onChange={(v) => set('light_harmonizer_key_intensity', v)} defaultValue={FACESWAP_DEFAULTS.light_harmonizer_key_intensity}
                />
                <Slider
                  label="Ambient light bias"
                  info="Offset to the ambient omnidirectional fill illumination. Positive values brighten shadows, negative values deepen contrast."
                  min={-0.5}
                  max={0.5}
                  step={0.05}
                  value={num(p.light_harmonizer_ambient_bias, 0.0)}
                  onChange={(v) => set('light_harmonizer_ambient_bias', v)} defaultValue={FACESWAP_DEFAULTS.light_harmonizer_ambient_bias}
                />
                <Slider
                  label="Eye specular catchlight restoration"
                  info="Restores sharp, directional corneal catchlights and glints from the target eyes onto the swapped face, preserving lifelike gaze direction and ocular vitality."
                  min={0}
                  max={1}
                  step={0.05}
                  value={num(p.light_harmonizer_eye_specular, 0.8)}
                  onChange={(v) => set('light_harmonizer_eye_specular', v)} defaultValue={FACESWAP_DEFAULTS.light_harmonizer_eye_specular}
                />
                <Slider
                  label="Dynamic shadow casting"
                  info="Casts dynamic directional soft shadows across the face when foreground occluders (microphones, hair strands, hats, hands) intercept the key light path."
                  min={0}
                  max={1}
                  step={0.05}
                  value={num(p.light_harmonizer_shadow_occlusion, 0.6)}
                  onChange={(v) => set('light_harmonizer_shadow_occlusion', v)} defaultValue={FACESWAP_DEFAULTS.light_harmonizer_shadow_occlusion}
                />
              </>
            )}
          </Disclosure>
          <Disclosure id="inspector.after-swap" title="Upscale & frame rate (after swap)">
            <Toggle label="AI upscale (after swap)" info="Runs an AI upscaler as the final step of the swap pass — each frame is swapped & enhanced first, then upscaled, producing a single output file (no second pass). Full-frame upscaling is heavy: ×4 on video is slow and VRAM-hungry." checked={!!p.upscale_after_swap} onChange={(v) => set('upscale_after_swap', v)} />
            {p.upscale_after_swap && (
              <Select label="AI upscale model" value={p.upscale_model_after} onChange={(v) => set('upscale_model_after', v)} options={AI_UPSCALE_MODELS} />
            )}
            <Select label="Frame interpolation (after swap)" info="Raises the output frame rate with motion-interpolated in-between frames as the final pass (after any upscale). RIFE = AI motion interpolation (recommended, fast); minterpolate = classical ffmpeg motion estimation (no model, much slower). Duration is unchanged — frame count and fps are multiplied together, audio untouched." value={p.interp_after_swap || 'off'} onChange={(v) => set('interp_after_swap', v)}
              options={[{ value: 'off', label: 'Off' }, { value: 'rife_2x', label: 'RIFE ×2 fps' }, { value: 'rife_4x', label: 'RIFE ×4 fps' }, { value: 'minterpolate_2x', label: 'ffmpeg minterpolate ×2' }]} />
          </Disclosure>
          <Disclosure id="inspector.merger-geometry" title="Face size & motion blur">
            <Slider label="Face size" info="Grows or shrinks the swapped face about its own centre. Every identity swapper keeps the TARGET's head size and bone structure, so when the source person's face is visibly narrower or broader there is otherwise no lever for it — the swap comes out the right identity at the wrong size. 0 = off. This is free: it changes the paste matrix rather than adding a pass, so it costs 0 ms. Note that 'Restore original mouth' re-composites using the target's untouched geometry, so large values here will not register with it." min={-0.2} max={0.2} step={0.01} value={num(p.output_face_scale, 0)} onChange={(v) => set('output_face_scale', v)} defaultValue={FACESWAP_DEFAULTS.output_face_scale} />
            <Slider label="Motion blur" info="Blurs the face along the direction the camera actually smeared it. On fast movement a pin-sharp face on a motion-blurred head is a loud tell. You set the amount; the DIRECTION is measured off the original crop rather than asked for — motion blur destroys detail along the axis of travel, so the plate's own gradient structure points the kernel. 0 = off. Measured ~7.5 ms per 512² face — the dearest of these five." min={0} max={1} step={0.05} value={num(p.merger_motion_blur, 0)} onChange={(v) => set('merger_motion_blur', v)} defaultValue={FACESWAP_DEFAULTS.merger_motion_blur} />
          </Disclosure>
        </InspectorGroup>

        <InspectorGroup id="temporal" title="Temporal" icon={Icon.film} badge={TEMPORAL_BADGE[temporalModeOf(p)]}>
          <TemporalModeControl settings={p} onApply={setMany}>
            <Toggle label="Temporal detection (anti-flicker)" info="Video (In-Memory method): one tracked detection pre-pass over the clip. Short detection misses (≤10 frames) are gap-filled by interpolating the face's position, so the swap can't blink out; with 'Stabilize face' also on, keypoints AND mask/mouth landmarks are smoothed per person. The swap pass then skips per-frame detection and stays multi-threaded. Includes identity locking when that toggle is on." checked={!!p.temporal_detection} onChange={(v) => set('temporal_detection', v)} />
            <Toggle label="Stabilize face (video)" info="Temporal keypoint smoothing — reduces swap wobble. Runs at Max Threads (2-pass) unless Enhancer Flicker is on." checked={!!p.stabilize_face} onChange={(v) => set('stabilize_face', v)} />
            {p.stabilize_face && (
              <>
                <Select label="Smoothing method" info="One Euro = adaptive (default). Kalman = removes more high-frequency jitter on a still head (about 80% against 51%) with no lag on steady motion; measured no better on footage where the head moves and talks. EMA = simpler fixed smoothing." value={p.stabilize_method || 'one_euro'} onChange={(v) => set('stabilize_method', v)} options={['one_euro', 'kalman', 'ema']} />
                {(p.stabilize_method || 'one_euro') === 'one_euro' && (
                  <>
                    <Slider label="Smoothing (min cutoff)" info="lower = smoother, more lag" min={0.01} max={0.3} step={0.01} value={num(p.stabilize_min_cutoff, 0.05)} onChange={(v) => set('stabilize_min_cutoff', v)} defaultValue={FACESWAP_DEFAULTS.stabilize_min_cutoff} />
                    <Slider label="Reactivity (beta)" info="higher = less lag on fast motion" min={0} max={0.2} step={0.01} value={num(p.stabilize_beta, 0.02)} onChange={(v) => set('stabilize_beta', v)} defaultValue={FACESWAP_DEFAULTS.stabilize_beta} />
                  </>
                )}
              </>
            )}
            <Toggle label="Smooth face landmarks (boundary crawl)" info="Requires 'Stabilize face'. That filter smooths the 5 alignment keypoints only, leaving the 106-point landmarks raw — and those are what the paste mask's OUTLINE is drawn from, so the face sat still while its own edge crawled a pixel or two every frame along the jaw and hairline. This smooths both under one velocity-adaptive factor, so the crop and its outline cannot drift apart during motion: heavy smoothing when the head is nearly still, near-instant response on a fast turn." checked={p.stabilize_landmarks !== false} onChange={(v) => set('stabilize_landmarks', v)} />
            <Toggle label="Reduce enhancer flicker" info="Temporally blends the enhanced face. Runs multi-threaded (work-stealing) when the launcher's ROOP_STAB_PARALLEL is on (the Pinokio default) — otherwise it forces single-thread. Either way it costs some extra compute (blending + per-block warm-up), so it's somewhat slower, not free." checked={!!p.stabilize_enhancer} onChange={(v) => set('stabilize_enhancer', v)} />
            {p.stabilize_enhancer && (
              <Slider label="Flicker reduction strength" info="higher = smoother" min={0} max={1} step={0.05} value={num(p.stabilize_enhancer_strength, 0.5)} onChange={(v) => set('stabilize_enhancer_strength', v)} defaultValue={FACESWAP_DEFAULTS.stabilize_enhancer_strength} />
            )}
            <Toggle label="Reduce mask flicker" info="Temporally blends the mask edge (the boundary between swapped and original pixels). XSeg/Face Parser/RealityUX all recompute this edge from scratch every frame with no memory of the last one, so on a close-up with hair moving across an eye it can jitter — most visible as the swapped face's own features flickering in and out right at the edge. Runs on the same multi-threaded chunked pipeline as the other stabilizers; unlike enhancer flicker reduction it does not back off during head motion, since the mask lives in the already motion-compensated aligned crop." checked={!!p.stabilize_mask} onChange={(v) => set('stabilize_mask', v)} />
            {p.stabilize_mask && (
              <Slider label="Mask flicker reduction strength" info="higher = smoother" min={0} max={1} step={0.05} value={num(p.stabilize_mask_strength, 0.5)} onChange={(v) => set('stabilize_mask_strength', v)} defaultValue={FACESWAP_DEFAULTS.stabilize_mask_strength} />
            )}
            {p.stabilize_mask && (
              <Toggle label="Motion-compensated mask smoothing" info="Before blending last frame's mask into this one, moves it along the optical flow between the two face crops, so an edge that moves relative to the face (hair, a hand, an expression) is followed instead of leaving a fading trail. Measured on 10 clips against an independent face parser: less residual jitter than plain smoothing (0.040 vs 0.047) and closer to the parser boundary. Costs about 0.75 ms of CPU per face." checked={!!p.mask_flow_warp} onChange={(v) => set('mask_flow_warp', v)} />
            )}
            <Toggle label="Flow-warped texture carry (pore boiling)" info="Requires an enhancer. GPEN/CodeFormer/UltraMax invent pores and lashes from scratch on every frame, so nearly-still skin gets a different micro-texture 25 times a second. This warps the PREVIOUS frame's high-frequency layer along dense optical flow and carries a fraction of it forward, so only the invented noise is averaged — brightness, colour and expression still come wholly from the current frame and cannot ghost. Costs extra CPU per face." checked={!!p.stabilize_hf_texture} onChange={(v) => set('stabilize_hf_texture', v)} />
            {p.stabilize_hf_texture && (
              <Slider label="Texture carry amount" info="Fraction of the previous frame's detail retained. 0.15 gives the pore field about a 6-frame memory while leaving 85% of the current frame, so genuinely new texture — a turn revealing a new cheek — is not held back." min={0} max={0.9} step={0.05} value={num(p.stabilize_hf_texture_weight, 0.15)} onChange={(v) => set('stabilize_hf_texture_weight', v)} defaultValue={FACESWAP_DEFAULTS.stabilize_hf_texture_weight} />
            )}
          </TemporalModeControl>
        </InspectorGroup>

        <InspectorGroup id="mouth-custody" title="Audio & Mouth Custody" icon={Icon.preview} badge={modeOf(p) === 'off' ? undefined : MODE_LABELS[modeOf(p)]}>
          <MouthCustodyPanel
            params={p}
            onPatch={setMany}
            onAddAudio={onAddLipsyncAudio}
            uploading={uploadingLipsyncAudio}
            hasTargetAudio={targetHasAudio}
            isImage={!!targets[selTarget] && !(targets[selTarget].frames > 1)}
            defaults={FACESWAP_DEFAULTS}
          />
        </InspectorGroup>

        <InspectorGroup id="face-details" title="Eyes, mouth & expression" icon={Icon.preview}>
          <Toggle
            label="Restore original eyes"
            info="Brings the TARGET's own eyes back over the swap. Every identity swapper here works from a 112–128px aligned crop, so the eyes — the smallest high-frequency detail in a face, and the first thing a viewer looks at — come back soft, with the gaze nudged toward wherever the source person was looking. This keeps the plate's gaze direction, catchlights and eyelash detail while everything else stays swapped. Built from the two eye keypoints, not the 106-point landmarks the mouth restore uses, so it works on every detector engine. It fades out past 25° of yaw or pitch and switches off at a true profile, because there the far eye is edge-on and the near one sits over the nose bridge — pasting the plate there doubles the socket. Costs one ellipse and a blur, so it is effectively free. Note that it undoes the swapper's eye SHAPE too, which is part of the identity: if the two people's eyes differ a lot, back the amount off rather than running it at 1."
            checked={!!p.restore_original_eyes}
            onChange={(v) => set('restore_original_eyes', v)}
          />
          {p.restore_original_eyes && (
            <>
              <Slider label="Eyes blend amount" info="How much of the original eyes comes back. 1 = fully the plate's eyes, 0.5 = half-way, which keeps some of the swapped eye shape while recovering the sharpness." min={0} max={1} step={0.05} value={num(p.eyes_blend_amount, 1)} onChange={(v) => set('eyes_blend_amount', v)} defaultValue={FACESWAP_DEFAULTS.eyes_blend_amount} />
              <Slider label="Eyes feather" info="Edge softness, as a percentage of the eye radius rather than a pixel count — a 15px feather is most of a distant face's eye and a hairline on a close-up, so a pixel value would need retuning every shot. Raise it if you can see where the restore stops." min={0} max={100} step={1} value={num(p.eyes_feather_blend, 25)} onChange={(v) => set('eyes_feather_blend', v)} defaultValue={FACESWAP_DEFAULTS.eyes_feather_blend} />
              <Slider label="Eyes region size" info="Scales both ellipses together. The default covers the eye and lashes; raise it to take in the lid and socket, lower it to restore the iris alone." min={0.3} max={2} step={0.05} value={num(p.eyes_size_factor, 1)} onChange={(v) => set('eyes_size_factor', v)} defaultValue={FACESWAP_DEFAULTS.eyes_size_factor} />
              <Slider label="Eyes width" info="Stretches the region horizontally only — useful for wide or almond eyes where the default ellipse clips the outer corner." min={0.3} max={2} step={0.05} value={num(p.eyes_radius_x, 1)} onChange={(v) => set('eyes_radius_x', v)} defaultValue={FACESWAP_DEFAULTS.eyes_radius_x} />
              <Slider label="Eyes height" info="Stretches the region vertically only. Raise it to include the brow line and lower lid; keep it low to avoid pulling the target's eyebrows back in." min={0.3} max={2} step={0.05} value={num(p.eyes_radius_y, 1)} onChange={(v) => set('eyes_radius_y', v)} defaultValue={FACESWAP_DEFAULTS.eyes_radius_y} />
            </>
          )}
          <Slider label="Expression transfer strength" info="Puts the TARGET's own expression (mouth, brows, jaw) back onto the swapped face using LivePortrait. Swappers pull faces toward the average expression of their training data, so laughing, crying and grimacing come out flattened — this reads the expression off the original frame and re-applies it. 0 = off (bit-exact no-op); 1 adopts the target's expression; above 1 exaggerates past it. Head pose cannot drift by construction. The eyes are NOT moved by this slider — see Eye-gaze follow and Blink sync. Cost, measured 2026-09-25 on hyperswap + Restore Ultra over 1014 frames: identity to the source 0.596 → 0.512 at 1.0, because LivePortrait re-renders the whole face; ~34 ms per face on TensorRT. Downloads ~537MB on first use; needs TensorRT (the CPU fallback is ~1.9s per face)." min={0} max={2} step={0.05} value={num(p.expression_restore_strength, 0)} onChange={(v) => set('expression_restore_strength', v)} defaultValue={FACESWAP_DEFAULTS.expression_restore_strength} />
          {num(p.expression_restore_strength, 0) > 0 && (
            <Select label="Expression region" info="Which part of the face the expression strength applies to: 'all' (mouth, brows, jaw), 'lips' only, or 'eyes' (nothing — the eyes have their own controls below)." value={p.expression_restore_region || 'all'} onChange={(v) => set('expression_restore_region', v)} options={['all', 'lips', 'eyes']} />
          )}
          <Toggle label="Blink sync" info="Makes the swapped face blink when the target does, with LivePortrait's own eyelid-retargeting network: the lid opening is measured on the target and on the swapped face (LivePortrait's 203-point landmarker) and the lids are driven to the target's. Measured 2026-09-25 over 1014 frames with 66 closed-eye frames: the plain swap kept the eyes OPEN on 67% of the target's blinks; blink sync closes them on 97%, cuts lid error 0.044 → 0.019, closes eyes that should be open on 0.1% of frames, and costs identity 0.596 → 0.590. The best eye setting measured — use it without Eye-gaze follow. Works at Expression strength 0. Downloads ~115MB on first use." checked={!!p.expression_blink_sync} onChange={(v) => set('expression_blink_sync', v)} />
          <Slider label="Eye-gaze follow" info="Weight on LivePortrait's eye keypoints, which carry both where the eye looks and how open the lids are. 0 = the swap's own eyes. MEASURED 2026-09-25, 1014 frames: at 1.0 the eye-direction error got WORSE, not better (0.043 → 0.055, dark-iris position on wide-open eyes) — the swap already keeps the target's eye direction and re-rendering the eyes adds error. It does catch blinks (100%) but less precisely than Blink sync, and together with Blink sync the lids are worse than Blink sync alone (lid error 0.040 vs 0.019). Kept for footage where the swap's eye direction is visibly wrong; check the result." min={0} max={1} step={0.05} value={num(p.expression_gaze_follow, 0)} onChange={(v) => set('expression_gaze_follow', v)} defaultValue={FACESWAP_DEFAULTS.expression_gaze_follow} />
        </InspectorGroup>

        <InspectorGroup id="fidelity" title="Expression & Gaze Fidelity" icon={Icon.preview}>
          <Slider label="Pupil & eye-gaze retention" settingKey="pupil_gaze_retention" min={0} max={1} step={0.05}
                  value={num(p.pupil_gaze_retention, 0)} onChange={(v) => set('pupil_gaze_retention', v)}
                  defaultValue={FACESWAP_DEFAULTS.pupil_gaze_retention}
                  info="Keeps where the TARGET is looking. Finds the iris in the original frame and in the swap, slides the swapped iris (its colour and brightness intact) to where the original's is, erases it from where it was, matches its pupil / limbal contrast to the original's, and takes the sclera's lid shading. Unlike 'Restore original eyes' it does not bring back the target's lids, lashes or eye colour. A blink in the original (no iris found) leaves the swap alone, and so does an iris too pale for the detector to find. Fades out past 25° of head turn. 0 = off (bit-exact). Checked on synthetic eyes and one real clip: judge it on your footage." />
          <Slider label="Micro-expression & lip-sync tension" settingKey="micro_expression_restore" min={0} max={1} step={0.05}
                  value={num(p.micro_expression_restore, 0)} onChange={(v) => set('micro_expression_restore', v)}
                  defaultValue={FACESWAP_DEFAULTS.micro_expression_restore}
                  info="Puts back the creases a 128 px swap averages away: the nasolabial folds, the inner brows and the mouth corners. Inside soft regions built from the five keypoints it transplants the ORIGINAL's mid-frequency luminance (a difference of Gaussians scaled to the face); the swap's colour, tone and fine skin texture are left alone. It restores shading, not geometry: it does not warp the swapped face. 0 = off (bit-exact). Calibrated on a synthetic crease only: judge it on your footage." />
          <Toggle label="Inner-mouth & teeth preservation" settingKey="inner_mouth_preserve"
                  info="On a WIDE-open mouth (the gap between the inner lips is over a fifth of the mouth's width: a shout, a wide vowel) keep the target's real teeth and tongue instead of the swap's, whether or not the swap broke them. Off, the app only repairs a mouth it detects as clamped or blurry. CAUTION: restoring the target's cavity unconditionally was measured to give grey, mottled, doubled teeth when the two people's teeth differ, which is why this is opt-in and limited to wide-open frames. The 'wide' threshold is a starting value, not yet calibrated on footage."
                  checked={!!p.inner_mouth_preserve} onChange={(v) => set('inner_mouth_preserve', v)} />
          <Toggle label="Identity crossover lock" settingKey="identity_lock"
                  info="When two people cross, one face's crop is briefly half the other's and the per-frame check can say 'another person fits better' about someone who never changed, switching the swap off them. With the lock on, an established identity is not stripped by a short, weak challenge (the challenger must beat it by an extra margin, and by more again while the face is following its own predicted path); a challenge that persists is still honoured, after the hold. Pins always win. OFF by default until it has been A/B'd on a crossing clip."
                  checked={!!p.identity_lock} onChange={(v) => set('identity_lock', v)} />
          {p.identity_lock && (
            <>
              <Slider label="Lock margin" settingKey="identity_lock_margin" min={0} max={0.3} step={0.01} value={num(p.identity_lock_margin, 0.08)}
                      onChange={(v) => set('identity_lock_margin', v)} defaultValue={FACESWAP_DEFAULTS.identity_lock_margin}
                      info="Extra cosine distance a challenger must win by, on top of the matcher's own margin, before it counts." />
              <Slider label="Lock hold (frames)" settingKey="identity_lock_hold" min={1} max={12} step={1} value={num(p.identity_lock_hold, 3)}
                      onChange={(v) => set('identity_lock_hold', v)} defaultValue={FACESWAP_DEFAULTS.identity_lock_hold}
                      info="A challenge burst up to this many frames long is absorbed; the next frame of it is honoured." />
            </>
          )}
          {identityPins.length > 0 && (
            <div className="space-y-1" data-testid="pin-list">
              <div className="text-nano font-semibold uppercase tracking-[0.14em] text-muted">Identity pins ({identityPins.length})</div>
              {identityPins.map((pin) => (
                <div key={pin.id} className="flex items-center justify-between gap-2 rounded-lg bg-white/[0.04] px-2 py-1 text-note">
                  <span className="truncate">
                    📌 {trackLabel(pin.track_id !== null ? pin.track_id - 1 : null)} → source {pin.source_index + 1} · frame {pin.frame}
                  </span>
                  <button type="button" className="rounded px-1.5 text-muted hover:bg-white/10 hover:text-white"
                          aria-label={`Remove the pin on frame ${pin.frame}`} onClick={() => onUnpinIdentity(pin.id)}>×</button>
                </div>
              ))}
            </div>
          )}
        </InspectorGroup>

        <InspectorGroup id="pose" title="Pose, angle & system" icon={Icon.tune}>
          <Toggle label="3D source pose matching" info="Only affects image-source swappers (BlendSwap / UniFace) — feeds them a pose-matched source crop. Has NO effect on inswapper/ghost/hyperswap/simswap (their identity vector is pose-invariant, so warping the source would only degrade it)." checked={!!p.use_3d_recon} onChange={(v) => set('use_3d_recon', v)} />
          <Toggle label="Multi-angle source bank" info="auto-pick best source per frame" checked={!!p.use_source_bank} onChange={(v) => set('use_source_bank', v)} />
          <Toggle label="↔️ Frontalize angled faces" info="Un-rotates steep profile/side (lateral) faces before swapping, then restores the original angle. WARNING, measured 2026-09-25 (hyperswap, angle clips, 218 frames). Until that date it swapped the face upside down and lost the swap entirely; that is fixed, but a flat 2D warp cannot really turn a head, so it still costs identity at every angle: identity to the source 0.53 → 0.37 at 45-60°, 0.52 → 0.22 at 60-75°, 0.35 → 0.18 at 75-90°, with a visible seam on the far cheek. It does place the eyes, nose and mouth more accurately past 75°. Keep off unless you have checked the result on your footage." checked={!!p.use_frontalization} onChange={(v) => set('use_frontalization', v)} />
          {p.use_frontalization && (
            <Slider label="Frontalize above angle (°)" info="Frontalization kicks in when the face yaw/pitch exceeds this. Lower = frontalize more; higher = only the steepest." min={10} max={60} step={5} value={num(p.frontalization_threshold, 30)} onChange={(v) => set('frontalization_threshold', v)} defaultValue={FACESWAP_DEFAULTS.frontalization_threshold} />
          )}
          <Toggle label="Reshape jaw/chin to source" info="Identity swappers (inswapper/hyperswap/reswapper…) keep the TARGET's jaw & chin bone structure. This warps the swapped face's lower-silhouette toward your SOURCE person's jaw/chin shape after the swap (a smooth liquify — no re-swap, so swap quality is untouched). Best for moderate shape differences; very large changes can distort the neck/background near the jaw. Landmark jitter is smoothed when Temporal detection is on." checked={!!p.jaw_reshape} onChange={(v) => set('jaw_reshape', v)} />
          {p.jaw_reshape && (
            <Slider label="Jaw reshape strength" info="0 = off (target's jaw), 1 = full source jaw/chin shape. Start around 0.4–0.6 and back off if the chin looks distorted." min={0} max={1} step={0.05} value={num(p.jaw_reshape_strength, 0.5)} onChange={(v) => set('jaw_reshape_strength', v)} defaultValue={FACESWAP_DEFAULTS.jaw_reshape_strength} />
          )}
          <Toggle label="VR mode" checked={!!p.vr_mode} onChange={(v) => set('vr_mode', v)} />
          <Toggle label="Auto rotate horizontal faces" checked={!!p.autorotate_faces} onChange={(v) => set('autorotate_faces', v)} />
        </InspectorGroup>

        <InspectorGroup id="vfx-multipass" title="VFX Multi-Pass & NLE Export" icon={Icon.film}>
          <Toggle
            label="VFX multi-pass export"
            info="Simultaneously exports 10-bit Beauty, 10-bit grayscale Combined Matte Pass, and uncomposited aligned face crops with 3x3 inverse affine transform matrices."
            checked={!!p.vfx_multi_pass}
            onChange={(v) => set('vfx_multi_pass', v)}
          />
          {p.vfx_multi_pass && (
            <>
              <Toggle
                label="Export combined matte pass"
                info="10-bit grayscale matte merging Stage 7 XSeg 3 occlusion mask, vector roto zones, and edge feathering."
                checked={p.vfx_export_matte ?? true}
                onChange={(v) => set('vfx_export_matte', v)}
              />
              <Toggle
                label="Export uncomposited crops & JSON inverse affine transforms"
                info="Aligned face crop sequence accompanied by per-frame JSON 3x3 inverse affine matrices and corner pins for Nuke/Fusion."
                checked={p.vfx_export_crops ?? true}
                onChange={(v) => set('vfx_export_crops', v)}
              />
            </>
          )}
          <div className="pt-2 border-t border-white/10 flex gap-2">
            <button
              type="button"
              onClick={handleExportEdl}
              className="flex-1 py-1.5 px-2 rounded-lg text-micro font-bold bg-white/5 border border-white/10 hover:bg-white/10 text-white transition-colors"
              title="Export CMX3600 EDL (V1 untouched source, V2 face-swapped cuts)"
            >
              Export EDL (CMX3600)
            </button>
            <button
              type="button"
              onClick={handleExportOtio}
              className="flex-1 py-1.5 px-2 rounded-lg text-micro font-bold bg-white/5 border border-white/10 hover:bg-white/10 text-white transition-colors"
              title="Export OpenTimelineIO JSON (V1 source clip, V2 face-swapped clips & gaps)"
            >
              Export OTIO (.json)
            </button>
          </div>
        </InspectorGroup>

        <Section title="Clip advisor" collapsible defaultOpen={false}>
          <button type="button" disabled={advisorBusy || !targets.length || progress.processing} onClick={runAdvisor}
            title="Samples the selected target (face sizes, count, detection coverage, motion, lighting) and recommends settings tuned to it. Nothing changes until you apply."
            className="w-full py-2 rounded-lg text-note font-bold bg-[var(--accent)]/10 border border-[var(--accent)]/30 text-accent hover:bg-[var(--accent)]/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2">
            {advisorBusy
              ? (<><span className="h-3 w-3 rounded-full border-2 border-[var(--accent)]/40 border-t-[var(--accent)] animate-spin" /> Analyzing target…</>)
              : 'Analyze target & recommend settings'}
          </button>
          {advice && (
            <div className="space-y-2 mt-2">
              <div className="text-micro text-muted leading-relaxed">
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
                          <span className="text-muted">{fmtAdviceVal(p[r.key] ?? '—')}</span>
                          <span className="text-muted"> → </span>
                          <span className="text-accent font-bold">{fmtAdviceVal(r.value)}</span>
                        </span>
                      </div>
                      <div className="text-micro text-muted leading-snug mt-0.5">{r.reason}</div>
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
          <div className="text-micro text-muted leading-relaxed -mt-1">
            Swap your webcam feed live using the loaded source face{liveObs ? ',' : ''} — optionally
            published as a system <b>virtual camera</b> for OBS / video calls.
          </div>
          <div className="flex gap-2">
            <TextInput label="Camera #" type="number" value={liveCamNum}
              onChange={(v) => setLiveCamNum(Math.max(0, parseInt(v, 10) || 0))} />
            <Select label="Resolution" value={liveRes} onChange={setLiveRes}
              options={['640x480', '1280x720', '1920x1080']} />
          </div>
          <div className="flex gap-2">
            <Select label="Capture FPS" value={String(liveFps)} onChange={(v) => setLiveFps(parseInt(v, 10) || 30)}
              options={['30', '60']} />
            <Select label="Detect every" value={String(liveInterval)} onChange={(v) => setLiveInterval(parseInt(v, 10) || 6)}
              options={['3', '6', '9']} />
          </div>
          <Toggle label="Stream to virtual camera (OBS)" info="Publishes the swapped feed as a system camera device via pyvirtualcam — pick 'OBS Virtual Camera' in any app." checked={liveObs} onChange={setLiveObs} />
          <Toggle label="Microphone passthrough" info="Routes microphone audio through an in-memory delay line. Select a virtual audio cable output such as VB-CABLE for Zoom, Discord, or OBS." checked={liveAudio} onChange={setLiveAudio} />
          {liveAudio && (
            <div className="space-y-2">
              <Select label="Microphone input" value={String(liveAudioInput)} onChange={setLiveAudioInput}
                options={['', ...liveAudioDevices.filter((d) => d.max_input_channels > 0).map((d) => d.name)]} />
              <Select label="Virtual audio output" value={String(liveAudioOutput)} onChange={setLiveAudioOutput}
                options={['', ...liveAudioDevices.filter((d) => d.max_output_channels > 0).map((d) => d.name)]} />
              <div className="text-micro text-muted">Audio delay follows measured visual latency automatically.</div>
            </div>
          )}
          {!liveActive ? (
            <button type="button" disabled={liveBusy || progress.processing} onClick={startLiveCam}
              className="w-full py-2 rounded-lg text-note font-bold bg-[var(--accent)]/10 border border-[var(--accent)]/30 text-accent hover:bg-[var(--accent)]/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2">
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
              {liveStatus?.stats && (
                <div className="text-micro text-muted flex flex-wrap gap-x-3 gap-y-1">
                  <span>{Number(liveStatus.stats.processing_fps || 0).toFixed(1)} FPS</span>
                  <span>{Number(liveStatus.stats.average_latency_ms || 0).toFixed(1)} ms avg</span>
                  <span>{liveStatus.stats.tracked_frames || 0} flow frames</span>
                  <span>{liveStatus.stats.dropped_capture_frames || 0} dropped</span>
                </div>
              )}
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

        <Section title="Live Telemetry & Diagnostics" collapsible defaultOpen={false}>
            <div><HeadPoseHud /></div>
            <div className="mt-3 flex justify-between items-center">
              <Button size="sm" variant="secondary" onClick={() => setShowShortcutHUD(true)}>Keyboard Shortcuts Info</Button>
            </div>
        </Section>
      </div>


      {/* COLUMN 2 & 3 WRAPPER: Preview canvas leads (hero) with the media
          asset managers as a right rail (2xl:flex-row-reverse), so the live
          preview is the visual center instead of buried on the far right. */}
      <div className="flex-1 w-full min-w-0 space-y-3.5 flex flex-col xl:flex-row-reverse gap-3.5">

        {/* COLUMN 2: Media Asset Manager — right rail. Stays put during a run,
            as with column 1: the sources and targets a running job baked in are
            already its own, so editing these is setting up the NEXT job. */}
        <div className={`w-full xl:w-[330px] 3xl:w-[380px] 4xl:w-[440px] shrink-0 space-y-3.5 select-none ${!showRightPanel ? 'hidden' : ''}`}>
          <Section title="Target media" icon={Icon.faceswap}>
            <PersonGroups
              targetFaces={targetFaces}
              targetGroups={targetGroups}
              targetNames={targetNames}
              targetFacesInfo={targetFacesInfo}
              targetPersonIds={targetPersonIds}
              targetReferenceFaceIds={targetReferenceFaceIds}
              selectedTargetPersonId={selectedTargetPersonId}
              setSelectedTargetPersonId={setSelectedTargetPersonId}
              selectedReferenceFaceId={selectedReferenceFaceId}
              setSelectedReferenceFaceId={setSelectedReferenceFaceId}
              applyTargetContext={applyTargetContext}
              commitTargetContext={commitTargetContext}
              setTargetPersonIds={setTargetPersonIds}
              setTargetReferenceFaceIds={setTargetReferenceFaceIds}
              selTargetFace={selTargetFace}
              setSelTargetFace={setSelTargetFace}
              sourceFaces={sourceFaces}
              sourceFacesInfo={sourceFacesInfo}
              faceSelection={p.face_detection_mode}
              setFaceSelection={(v) => set('face_detection_mode', v)}
              selectedSource={selSource}
              faceMapping={faceMapping}
              setFaceMapping={setFaceMapping}
              frame={frame}
              selTarget={selTarget}
              targetMediaId={activeTargetMediaId}
              setTargetFaces={setTargetFaces}
              setTargetGroups={setTargetGroups}
              setTargetNames={setTargetNames}
              setTargetFacesInfo={setTargetFacesInfo}
              notify={notify}
              clearPreviewCache={clearPreviewCache}
            />
          </Section>

          <Section title="Target file(s)">
            <FileDrop accept="image/*,video/*,.webp" multiple label="Add target media" onFiles={onAddTarget} busy={uploadingTgt} hint="drop images or videos here"
                      progress={tgtProgress} onCancel={() => tgtAbortRef.current?.abort()}
                      onPaths={onAddTargetPaths} />
            {targets.length === 0 ? (
              <div className="h-24 flex items-center justify-center rounded-xl border border-dashed border-white/10 text-xs text-muted bg-black/10 select-none">No target media loaded</div>
            ) : (
              <div className="space-y-2 max-h-52 overflow-y-auto pr-1">
                {targets.map((t, i) => {
                  // Show the most advanced state any queued job for this file
                  // reached — with segments, one target can carry several jobs.
                  const mediaId = t.media_id || t.id;
                  const jobs = queue.jobs.filter(j => mediaId
                    ? j.target_media_id === mediaId
                    : j.target_name === t.name);
                  const job = jobs.find(j => ACTIVE_STATES.includes(jobState(j)))
                    || jobs.find(j => RETRYABLE_STATES.includes(jobState(j)))
                    || jobs.find(j => jobState(j) === 'RECOVERABLE')
                    || jobs.find(j => jobState(j) === 'COMPLETED');
                  const jobSt = job ? jobState(job) : null;
                  const statusLabel = jobSt ? (QUEUE_STATE_LABEL[jobSt] || jobSt) : null;
                  const badgeColor = jobSt ? QUEUE_STATE_CLASS[jobSt] : '';
                  const isVideo = t.frames > 1;
                  const duration = isVideo && t.fps ? (t.frames / t.fps).toFixed(1) : null;
                  const TypeIcon = isVideo ? Icon.film : Icon.still;
                  return (
                    <div key={mediaId || i}
                      className={`group flex items-center gap-3 px-3.5 py-3 rounded-xl text-sm border transition-all duration-200 cursor-pointer ${selTarget === i ? 'bg-[var(--accent)]/10 border-[var(--accent)]/40' : 'bg-white/[0.02] border-white/[0.06] hover:border-white/15 hover:bg-white/[0.04]'}`}
                      onClick={() => selectTarget(i)}>
                      {/* v=name busts the browser cache: the URL is index-based, so
                          after a removal the same index can point at another file. */}
                      <img
                        src={`${API}/api/target/preview?index=${i}&frame=1&width=96&v=${encodeURIComponent(mediaId || t.name)}`}
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
                        <div className="flex items-center gap-2 mt-0.5 text-note font-medium text-muted">
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
                <Button size="sm" variant="stop" onClick={async () => {
                  rememberTargetContext();
                  const r = await postJSON('/api/target/clear', {
                    target_media_id: activeTargetMediaId,
                  });
                  setTargets(r.targets || []);
                  setTargetFaces([]); setTargetGroups([]); setTargetNames([]);
                  setTargetFacesInfo([]); setTargetPersonIds([]);
                  setTargetReferenceFaceIds([]); setSelectedTargetPersonId(null);
                  setSelectedReferenceFaceId(null); setFaceMapping({}); setSelTargetFace(0);
                  setPreviewSrc(''); setPreviewFor('');
                }}>Clear targets</Button>
              </div>
            )}
          </Section>
          
          <Section title="Source images / facesets" icon={Icon.faces}>
          <FileDrop accept="image/*,.fsz" multiple label="Add source faces" onFiles={onAddSource} busy={uploadingSrc} hint="drop images or .fsz here"
                    progress={srcProgress} onCancel={() => srcAbortRef.current?.abort()} />
          <label className="mt-2 flex cursor-pointer items-center justify-center rounded-xl border border-dashed border-white/15 px-3 py-2 text-xs text-white/55 transition hover:border-[var(--accent)] hover:text-white">
            Build one multi-angle faceset from a folder
            <input type="file" accept="image/*" multiple webkitdirectory="" directory="" className="sr-only"
              onChange={(event) => { onAddSourceFolder(Array.from(event.target.files || [])); event.target.value = ''; }} />
          </label>
          <FacesetLibrary
            canSave={sourceFaces.length > 0}
            onLoaded={(r) => { setSourceFaces(r.source_faces || []); if (r.source_faces_info) setSourceFacesInfo(r.source_faces_info); }}
            notify={notify} />
          {pinnedIdentities.length > 0 && (
            <div className="p-2.5 rounded-xl bg-black/40 border border-white/10 space-y-1.5 animate-slide-up">
              <div className="flex items-center justify-between text-micro font-bold uppercase tracking-wider text-white/50">
                <span>⭐ Pinned Quick Identities ({pinnedIdentities.length})</span>
                <button type="button" onClick={() => { setPinnedIdentities([]); localStorage.removeItem('roop_pinned_identities'); }} className="text-muted hover:text-white">Clear</button>
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
                <Button size="sm" variant="secondary" onClick={() => moveSource('left')}>⬅ Move</Button>
                <Button size="sm" variant="secondary" onClick={() => moveSource('right')}>Move ➡</Button>
                <Button size="sm" variant="secondary" onClick={() => sourceAction('/api/source/remove', { index: selSource })}>Remove</Button>
                <Button size="sm" variant="secondary" title="Set each tile to the most frontal face in its set" onClick={() => sourceAction('/api/source/refresh_thumbs', {})}>Frontal thumb</Button>
                <Button size="sm" variant="stop" onClick={() => sourceAction('/api/source/clear', {})}>Clear all</Button>
              </div>
              
              {sourceFacesInfo[selSource] && (
                <div className="p-3.5 rounded-xl bg-black/45 border border-white/5 space-y-2 text-xs select-none">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-micro uppercase tracking-[0.14em] text-white/50">Selected source details</span>
                    <span className="px-2 py-0.5 rounded-full bg-[var(--accent)]/10 text-micro text-accent font-bold border border-[var(--accent)]/20">
                      {sourceFacesInfo[selSource].count > 1 ? `${sourceFacesInfo[selSource].count} Reference Faces` : 'Single Face'}
                    </span>
                  </div>
                  
                  <div className="space-y-1.5 pt-1">
                    {(sourceFacesInfo[selSource].count || 1) > 1 && Array.isArray(sourceFacesInfo[selSource].poses) && sourceFacesInfo[selSource].poses.length > 0 ? (
                      <>
                        <div className="text-micro font-bold text-muted mb-1">Pose Coverage Breakdown:</div>
                        <div className="flex flex-wrap gap-1.5">
                          {Object.entries(
                            (sourceFacesInfo[selSource].poses || []).reduce((acc, p) => {
                              acc[p] = (acc[p] || 0) + 1;
                              return acc;
                            }, {})
                          ).map(([pose, cnt]) => (
                            <span key={pose} className="px-2 py-1 rounded-lg bg-white/[0.03] border border-white/5 text-micro text-white/70">
                              {pose} <span className="text-accent font-extrabold">({cnt})</span>
                            </span>
                          ))}
                        </div>
                      </>
                    ) : (
                      <div className="flex items-center justify-between text-white/60">
                        <span>Detected Pose:</span>
                        <span className="font-bold text-white">{sourceFacesInfo[selSource].poses?.[0] || 'Front'}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </Section>
        <IdentityBlender sourceFaces={sourceFaces} sourceFacesInfo={sourceFacesInfo}
          value={identityBlend} onChange={setIdentityBlend} onUploadSource={onAddSource} notify={notify}
          playheadFrame={frame} />
      </div>

        {/* COLUMN 3: Active Canvas, Timeline & Outputs */}
        <div className="flex-1 min-w-0 space-y-6">
          {/* run bar */}
          <div className="sticky top-20 z-30 pb-3 bg-[#0c0e14]/90 backdrop-blur-md">
            {/* The run bar keeps its shape whether or not something is
                rendering. A run has its own tab now, so this one no longer
                turns into a progress panel — it just says a job is in flight,
                and offers the way over to it. */}
            <div className="w-full space-y-3">
              <div className="rounded-2xl glass-panel p-3.5 sm:p-4 flex flex-col md:flex-row items-center justify-between gap-3 sm:gap-4 shadow-xl border border-white/5 w-full">
                <div className="flex items-center gap-3 w-full md:w-auto">
                  {progress.processing ? (
                    <Button variant="stop" size="lg" onClick={confirmStop} aria-describedby="run-status" className="w-full md:w-auto justify-center">■ Cancel render</Button>
                  ) : (
                    <Button variant="primary" size="lg" onClick={start} disabled={!!startBlockedReason} aria-describedby="run-status" className="w-full md:w-auto justify-center">▶ Start Swapping</Button>
                  )}
                  {maxFrames > 1 && (
                    <Button variant="secondary" size="lg" onClick={renderPreviewClip} disabled={!!startBlockedReason || isGeneratingPreviewClip || progress.processing} aria-describedby="run-status" className="!text-orange-400 border border-orange-500/20 hover:bg-orange-500/10 w-full md:w-auto justify-center">
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
                  {/* No `truncate`: this is where the user is told WHY Start is
                      unavailable, so it wraps rather than being cut to "No tar…". */}
                  <div id="run-status" className="flex items-center gap-2.5 text-sm font-semibold text-[var(--text-muted)] max-w-sm text-right">
                    {progress.processing ? (
                      <>
                        <span className={`h-2.5 w-2.5 rounded-full ${progress.paused ? 'bg-amber-400' : 'bg-[var(--accent)] animate-pulse shadow-[0_0_8px_var(--accent-glow)]'}`} />
                        {progress.paused ? <span className="tabular-nums">Paused</span> : (
                          <LiveText
                            className="tabular-nums"
                            select={(s) => {
                              const eta = etaMsOf(s.run, startTime ? Date.now() - startTime : 0);
                              return `Rendering ${Math.round(selectProg(s) * 100)}%${eta > 0 ? ` · ${fmtTime(eta)} left` : ''}`;
                            }}
                          />
                        )}
                      </>
                    ) : (
                      <>
                        <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${startBlockedReason ? 'bg-amber-400' : 'bg-emerald-500 animate-pulse shadow-[0_0_8px_#10b981]'}`} />
                        <span className={startBlockedReason ? 'text-amber-300' : undefined}>{startBlockedReason || 'Ready to swap'}</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
              {progress.processing && <PipelinePill processing className="px-1" />}
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

                {/* Modular Pipeline Flow Strip & 1-Click Buffer Soloing */}
                {showFlowStrip && (
                  <div className="pt-1">
                    <div className="flex items-center justify-between mb-1 text-micro text-white/50 px-1">
                      <span className="font-semibold uppercase tracking-wider text-[11px] text-white/70">Modular Pipeline Flow</span>
                      <span className="text-[10px]">Alt+Click node to solo intermediate buffer &bull; ⏻ bypass</span>
                    </div>
                    <ModularPipelineFlowStrip
                      soloStage={soloStage}
                      onSoloStage={handleSoloStage}
                      bypassedStages={bypassedStages}
                      onToggleBypass={handleToggleBypass}
                    />
                  </div>
                )}

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

                {/* Stage 23: the four snapshot slots. Keys 1-4 over the stage do the same as these. */}
                <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
                  <SnapshotSlots api={slotBank} />
                  {referenceId && !referenceFrame && (
                    <span className="text-micro text-amber-300" role="status">
                      Split-wipe reference Slot {referenceId} is of another frame, so the wipe shows the original.
                    </span>
                  )}
                </div>

                {previewSrc ? (
                  comparingEnhancers ? (
                    <ComparisonGridPanel
                      title="Compare Enhancers (Select up to 4)"
                      availableItems={meta.enhancers || []}
                      selectedItems={selectedGridEnhancers}
                      setSelectedItems={setSelectedGridEnhancers}
                      previews={enhancerPreviews}
                      times={enhancerTimes}
                      timers={liveRenderingTimers}
                      notify={notify}
                      itemNoun="enhancers"
                    />
                  ) : comparingMasks ? (
                    <ComparisonGridPanel
                      title="Compare Mask Engines (Select up to 4)"
                      subtitle={<>Enhancer: <span className="text-white/55 font-semibold">{p.selected_enhancer || 'None'}</span></>}
                      availableItems={meta.mask_engines || []}
                      selectedItems={selectedGridMasks}
                      setSelectedItems={setSelectedGridMasks}
                      previews={maskPreviews}
                      times={maskTimes}
                      timers={maskRenderTimers}
                      notify={notify}
                      itemNoun="mask engines"
                    />
                  ) : comparingSwappers ? (
                    <ComparisonGridPanel
                      title="Compare Swapper Models (Select up to 4)"
                      subtitle={<>Enhancer: <span className="text-white/55 font-semibold">{p.selected_enhancer || 'None'}</span></>}
                      availableItems={meta.swap_models || []}
                      selectedItems={selectedGridSwappers}
                      setSelectedItems={setSelectedGridSwappers}
                      previews={swapperPreviews}
                      times={swapperTimes}
                      timers={swapperRenderTimers}
                      notify={notify}
                      itemNoun="swapper models"
                    />
                  ) : comparingUpscalers ? (
                    <ComparisonGridPanel
                      title="Compare AI Upscalers (Select up to 4)"
                      subtitle="Swaps once, then upscales each"
                      availableItems={AI_UPSCALE_MODELS.map((model) => model.label)}
                      selectedItems={selectedGridUpscalers}
                      setSelectedItems={setSelectedGridUpscalers}
                      previews={upscalePreviews}
                      times={upscaleTimes}
                      timers={upscaleRenderTimers}
                      notify={notify}
                      itemNoun="upscalers"
                    />
              ) : (
                <div className="relative rounded-2xl">
                  <AmbilightGlow
                    previewSrc={previewSrc || rawUrl}
                    enabled={ambilightEnabled}
                  />
                  {/* beforeFrame/afterFrame hand over the bitmap the scrub
                      hook has ALREADY decoded, so the stage does not fetch and
                      decode the same JPEG a second time — on 4K footage that
                      was ~30 ms of duplicated main-thread work per frame of
                      every drag. Passed only while the scrub hook is the one
                      that fetched the frame: during playback the buffered
                      player owns it and supplies a url instead. */}
                  <InteractivePreview
                    soloStage={soloStage}
                    onSoloStage={handleSoloStage}
                    beforeSrc={rawUrl}
                    afterSrc={stageAfterSrc}
                    beforeFrame={rawFrameBitmap && loadedRawUrl === rawUrl ? rawFrameBitmap : null}
                    playbackSource={playbackSource}
                    afterFrame={heldFrame || (scrubbingNow && !isPlaying
                      && stageAfterSrc === rawUrl
                      && loadedRawUrl === rawUrl ? rawFrameBitmap : null)}
                    referenceFrame={referenceFrame}
                    referenceLabel={referenceId ? `Slot ${referenceId}` : ''}
                    liveLabel={heldFrame ? `Slot ${heldSlot.id}` : 'Live result'}
                    onSlotHotkey={(n) => slotBank.activate(SLOT_IDS[n - 1])}
                    scrubbing={scrubbingNow}
                    onMaskChange={applyManualMask}
                    maskApplied={!!manualMask}
                    faces={previewFaces}
                    kps={previewKps}
                    pose={previewPose}
                    personIds={previewPersonIds}
                    selectedFaceIndex={selectedDetectedFaceIndex}
                    onSelectFace={onFaceBoxClick}
                    faceRoles={previewFaceRoles}
                    onToggleExclude={onToggleFaceExclusion}
                    rotoZones={rotoZones}
                    onRotoZones={setRotoZones}
                    rotoSelectedId={rotoSelectedId}
                    onRotoSelect={setRotoSelectedId}
                    faceAnchors={faceAnchors}
                    faceSources={faceSources}
                    pins={identityPins}
                    onPinIdentity={onPinIdentity}
                    onUnpinIdentity={onUnpinIdentity}
                    faceMenu={faceMenu}
                    notify={notify}
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
                    frameUnavailable={frameUnavailable}
                    onRetryFrame={retryRawFrame}
                    sarVal={previewSar}
                    darVal={previewDar}
                    colorBadge={previewColorBadge || targets[selTarget]?.color_badge || ''}
                  />
                </div>
              )
            ) : (
              <div className="relative aspect-video rounded-2xl overflow-hidden bg-gradient-to-br from-white/[0.03] to-black/20 border border-white/10 flex items-center justify-center select-none">
                <div className="absolute inset-0 pointer-events-none opacity-70" style={{ background: 'radial-gradient(circle at 50% 42%, var(--accent-glow), transparent 62%)' }} />
                {frameUnavailable && (
                  <FrameUnavailable onRetry={retryRawFrame} className="absolute inset-x-0 bottom-4 z-10" />
                )}
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
                        <div className="text-xs text-muted mt-1 leading-relaxed">
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
                  audioTrack={{
                    offsetMs: clampOffset(p.audio_offset_ms),
                    onOffsetChange: (ms) => set('audio_offset_ms', clampOffset(ms)),
                    trackPath: requestFields(p).override_audio_path,
                    hasTargetAudio: targetHasAudio,
                  }}
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
                  targetKey={activeTargetMediaId || targets[selTarget]?.name || String(selTarget)}
                  segments={segments.segments}
                  onSegmentClick={jumpToSegment}
                  sceneCuts={sceneCuts}
                  targetIndex={selTarget}
                  matchThreshold={num(p.max_face_distance, 0.75)}
                  running={!!progress.processing}
                  renderLog={renderLog.log}
                  fpPieces={timelinePieces}
                  liveRange={liveRange}
                  sceneOverrides={sceneOverrides}
                  identityKeys={identityTrack}
                  curveEditor={curveEditor}
                  sceneApiRef={sceneApiRef}
                  swapModels={meta.swap_models}
                  detectionMode={p.face_detection_mode}
                  baseValues={{
                    mask_erosion: num(p.mask_erode_dilate_radius, 0),
                    mask_feather: num(p.face_mask_blend, 20),
                    restore_blend: num(p.restore_ultra_detail_weight, 0.75),
                    swapper: p.swap_model,
                  }}
                  onPresenceSelect={selectPresencePerson}
                  onRenderRange={start}
                  onRenderOverrides={() => queueOverridePieces()}
                  renderBlockedReason={startBlockedReason}
                  playheadApiRef={playheadApiRef}
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

            {/* A previous render sent from Outputs, wiped against the live preview. Renders nothing until one is sent. */}
            <ABReferencePanel previewSrc={previewSrc} frame={frame} fps={targets[selTarget]?.fps || 25} startFrame={startFrame} />

            {/* Preview controls. Each one drives a preview render, which does
                compete with a run for the GPU — but it takes a deliberate click
                to fire one, and the automatic refreshes still hold off until
                the run is over (see the deferred-preview effect above). */}
            <div className={`flex items-center flex-wrap gap-3 ${maxFrames > 1 ? 'pt-3 border-t border-white/5' : ''}`}>
              <Button size="sm" variant="secondary" title="Re-run the swap for this frame, ignoring the cached result" onClick={() => refreshPreview({ force: true })}>Refresh</Button>
              <Button size="sm" variant="primary" disabled={selectedDetectedFaceIndex === null}
                title={selectedDetectedFaceIndex === null
                  ? 'Select a numbered detected face first'
                  : `Capture detected face ${selectedDetectedFaceIndex + 1}`} onClick={useFaceFromFrame}>
                {selectedDetectedFaceIndex === null
                  ? 'Select a face to capture'
                  : `Use selected face ${selectedDetectedFaceIndex + 1}`}
              </Button>
              <Button size="sm" variant="secondary" disabled={!previewFaces.length}
                title="Explicitly capture every detected face as a separate target person"
                onClick={captureAllPeopleFromFrame}>Capture all people</Button>
              <Button
                size="sm"
                variant="secondary"
                data-testid="faceswap-sweep-btn"
                title="Open 3x3 Parameter Sweep Contact Sheet"
                onClick={() => setShowSweepModal(true)}
              >
                Parameter Sweep (3×3 Grid)
              </Button>
              {previewSrc && !comparingEnhancers && !comparingMasks && !comparingSwappers && !comparingUpscalers && (
                <Button size="sm" variant="secondary" disabled={upscaling} onClick={upscaleThisFrame}
                  title="AI-upscale just this frame to preview final quality">
                  {upscaling ? 'Upscaling…' : `Upscale this frame (${AI_UPSCALE_MODELS.find(m => m.value === (p.upscale_model_after || 'esrganx2'))?.label || 'AI'})`}
                </Button>
              )}
            </div>

            {previewDiagnostic && (
              <div role="status" className="mt-2 rounded-lg border border-amber-400/25 bg-amber-400/[0.06] px-3 py-2 text-xs text-amber-200/90">
                {previewDiagnostic === 'target_required'
                  ? 'Capture a target face before using Selected face mode. The preview is showing the original frame.'
                  : previewDiagnostic === 'invalid_person_id'
                    ? 'The selected target person is no longer available. Select another target person. The preview is showing the original frame.'
                    : 'Select a target person before using this mode. The preview is showing the original frame.'}
              </div>
            )}

            <div className="flex items-center flex-wrap gap-3">
              <Toggle label="Live Swap" checked={fakePreview} onChange={setFakePreview} />
              <Toggle label="Compare" checked={compare} onChange={(v) => { setCompare(v); if (v) { setComparingEnhancers(false); setComparingMasks(false); setComparingSwappers(false); setComparingUpscalers(false); } }} />
              <Toggle label="Flow Nodes" checked={showFlowStrip} onChange={setShowFlowStrip} />
              {compare && <Toggle label="Split View" checked={splitView} onChange={setSplitView} />}
              <Toggle label="Enhancer Grid" checked={comparingEnhancers} onChange={(v) => { setComparingEnhancers(v); if (v) { setCompare(false); setComparingMasks(false); setComparingSwappers(false); setComparingUpscalers(false); } }} />
              <Toggle label="Mask Grid" checked={comparingMasks} onChange={(v) => { setComparingMasks(v); if (v) { setCompare(false); setComparingEnhancers(false); setComparingSwappers(false); setComparingUpscalers(false); } }} />
              <Toggle label="Swapper Grid" checked={comparingSwappers} onChange={(v) => { setComparingSwappers(v); if (v) { setCompare(false); setComparingEnhancers(false); setComparingMasks(false); setComparingUpscalers(false); } }} />
              <Toggle label="Upscale Grid" checked={comparingUpscalers} onChange={(v) => { setComparingUpscalers(v); if (v) { setCompare(false); setComparingEnhancers(false); setComparingMasks(false); setComparingSwappers(false); } }} />
            </div>
          </Section>

          {/* Stage 24 3x3 Parameter Sweep Contact Sheet Modal */}
          <ParameterSweepModal
            isOpen={showSweepModal}
            onClose={() => setShowSweepModal(false)}
            targetIndex={selTarget}
            frame={frame}
            currentSettings={p}
            onApplyRecipe={(recipe) => {
              setSettings((s) => ({ ...s, ...recipe }));
              refreshPreview({ force: true, overrides: recipe });
              notify?.('Applied sweep recipe parameters', 'info');
            }}
            notify={notify}
          />

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
                <span className="text-mini uppercase tracking-[0.14em] text-muted font-semibold">Runtime estimation</span>
                <span className={`text-micro font-semibold uppercase tracking-wide px-2 py-0.5 rounded-full border ${estSourceClass}`}>{estSourceLabel}</span>
              </div>
              <div className="flex items-end gap-3 mb-4">
                <span className="text-3xl font-bold text-white/95 tabular-nums leading-none">~{fmtTime(estTotalMs)}</span>
                <span className="text-xs text-muted mb-0.5">{Math.round(estPerFrame)} ms/frame{heavyVram ? ' · high VRAM' : ''}</span>
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
                    <span className="text-muted">{k}</span>
                    <span className="text-white/80 tabular-nums truncate" title={String(v)}>{v}</span>
                  </div>
                ))}
              </div>
              <div className="mt-3 pt-3 border-t border-white/5 text-note text-muted leading-snug">
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

          {/* Saved projects. The backend has always written one per render
              (api.py `_create_processing_project`) and kept its safe frame up
              to date; until now nothing could read them, so an interrupted
              render's checkpoint was unreachable from the UI. */}
          <div>
            <ProjectsPanel
              notify={notify}
              processing={progress.processing}
              onLoaded={() => hydrateWorkspaceFromState({ restoreView: false })}
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
                    ? <OutputVideoPlayer src={outUrl} renderKey={out?.path || out?.url} source={outputSource(out)} className="w-full rounded-xl border border-white/5" />
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
                <h4 className="font-bold text-accent text-xs uppercase tracking-wider">Playback & Nav</h4>
                <div className="flex items-center justify-between"><span className="text-white/60">Play / Pause</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Space</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Prev / Next Frame</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">← / →</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Step 10 Frames</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Shift + ← / →</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Jump to Start/End</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Home / End</kbd></div>
              </div>
              <div className="space-y-2.5">
                <h4 className="font-bold text-accent text-xs uppercase tracking-wider">Timeline Trimming</h4>
                <div className="flex items-center justify-between"><span className="text-white/60">Set Start Frame</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">[</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Set End Frame</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">]</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Reset Range</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">R</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Marker at playhead</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">M</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Set In / Out</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">I / O</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Shuttle back / pause / fwd</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">J / K / L</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Step 1 / 10 frames</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">, . / Shift</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Prev / next scene cut</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">↑ / ↓</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Zoom timeline</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Ctrl + Wheel</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Scroll timeline</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Shift + Wheel</kbd></div>
              </div>
              <div className="space-y-2.5">
                <h4 className="font-bold text-accent text-xs uppercase tracking-wider">Compare & Zoom</h4>
                <div className="flex items-center justify-between"><span className="text-white/60">Undo / Redo (all edits, masks, roto)</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Ctrl + Z / Y</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Snapshot slots A / B / C / D</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">1 / 2 / 3 / 4</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Zoom 1:1 / 200% / 400%</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Shift + 1 / 2 / 4</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Zoom In / Out</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">+ / -</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Toggle Comparison</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Shift + C</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Parameter Curve Editor (video)</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">C</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Toggle Split View</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">S</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Blend / slider mode</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">W</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Split axis</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">X</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Auto-swipe</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">A</kbd></div>
              </div>
              <div className="space-y-2.5">
                <h4 className="font-bold text-accent text-xs uppercase tracking-wider">Preview Tools</h4>
                <div className="flex items-center justify-between"><span className="text-white/60">Magnifier lens</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">G</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Mask brush</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">B</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Keypoints &amp; pose</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">D</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Lock to target face</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">T</kbd></div>
              </div>
              <div className="space-y-2.5">
                <h4 className="font-bold text-accent text-xs uppercase tracking-wider">Queue & Process</h4>
                <div className="flex items-center justify-between"><span className="text-white/60">Add to Batch Queue</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Q</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Run Swapper / Queue</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">Ctrl + Enter</kbd></div>
                <div className="flex items-center justify-between"><span className="text-white/60">Toggle Shortcuts HUD</span> <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs font-mono text-white">?</kbd></div>
              </div>
            </div>
            <p className="text-xs text-muted text-center mt-2">Click anywhere outside this modal or press Esc to close.</p>
          </Card>
        </div>
      )}

      {/* Floating Action Dock HUD */}
      <FloatingActionDock
        workspaceMode={workspaceMode}
        setWorkspaceMode={setWorkspaceMode}
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
