// ── Timeline Transient Zustand Store ───────────────────────────────────────
//
// High-performance transient store for the 60 FPS CinematicTimeline.
// Isolates frequent playhead updates (sub-frame scrubbing, shuttle velocity)
// from React component re-render loops.
// Subscribers can consume fine-grained updates or read directly from
// `useTimelineStore.getState()` inside the requestAnimationFrame render loop.

import { create } from 'zustand';

/** Formats frame into standard SMPTE timecode (HH:MM:SS:FF) */
export function formatSMPTE(frame, fps = 25) {
  const f = Math.max(1, Math.round(frame));
  const zeroIndexed = f - 1;
  const effectiveFps = Math.max(1, Math.round(fps));
  const totalSeconds = Math.floor(zeroIndexed / effectiveFps);
  const ff = zeroIndexed % effectiveFps;
  const hh = Math.floor(totalSeconds / 3600);
  const mm = Math.floor((totalSeconds % 3600) / 60);
  const ss = totalSeconds % 60;

  const hStr = String(hh).padStart(2, '0');
  const mStr = String(mm).padStart(2, '0');
  const sStr = String(ss).padStart(2, '0');
  const fStr = String(ff).padStart(2, '0');
  return `${hStr}:${mStr}:${sStr}:${fStr}`;
}

const MIN_ZOOM = 0.02; // pixels per frame (overview)
const MAX_ZOOM = 60.0; // pixels per frame (sub-frame inspection)
const SNAP_THRESHOLD_PX = 8; // Snap magnet distance in screen pixels

export const useTimelineStore = create((set, get) => ({
  // Core Time & Range
  frame: 1,
  maxFrames: 600,
  fps: 25,
  inPoint: 1,
  outPoint: 600,

  // Viewport Geometry
  zoom: 1.5, // pixels per frame
  scrollLeft: 0,
  canvasWidth: 1000,
  canvasHeight: 260,

  // Transport & Shuttle (J-K-L)
  isPlaying: false,
  shuttleSpeed: 0, // -8, -4, -2, -1, 0, 1, 2, 4, 8

  // Snapping & Guides
  snappingEnabled: true,
  activeSnapLine: null, // { frame, x }

  // Track 1: Scene Cuts detected by backend
  sceneCuts: [60, 180, 310, 475],

  // Track 3+: Identity Sub-tracks (clustered faces with active presence blocks)
  identityTracks: [
    {
      id: 'person_0',
      name: 'Person #1',
      color: '#38BDF8', // Cyan
      intervals: [
        [1, 140],
        [190, 310],
        [420, 580],
      ],
    },
    {
      id: 'person_1',
      name: 'Person #2',
      color: '#EC4899', // Pink
      intervals: [
        [45, 180],
        [250, 470],
      ],
    },
    {
      id: 'person_2',
      name: 'Person #3',
      color: '#10B981', // Emerald
      intervals: [
        [185, 340],
        [480, 600],
      ],
    },
  ],

  // Track 4: Parameter Keyframe Automation Curves
  curvesExpanded: true,
  activeCurveId: 'enhancer_fidelity',
  curves: [
    {
      id: 'enhancer_fidelity',
      name: 'Enhancer Fidelity',
      color: '#F59E0B', // Amber
      min: 0.0,
      max: 1.0,
      keyframes: [
        { frame: 1, value: 0.65, type: 'bezier' },
        { frame: 120, value: 0.9, type: 'bezier' },
        { frame: 280, value: 0.45, type: 'bezier' },
        { frame: 450, value: 0.85, type: 'linear' },
        { frame: 600, value: 0.7, type: 'bezier' },
      ],
    },
    {
      id: 'mask_feather',
      name: 'Mask Feathering',
      color: '#8B5CF6', // Purple
      min: 0.0,
      max: 50.0,
      keyframes: [
        { frame: 1, value: 12.0, type: 'bezier' },
        { frame: 200, value: 28.0, type: 'bezier' },
        { frame: 500, value: 15.0, type: 'bezier' },
      ],
    },
  ],

  // ── Actions ─────────────────────────────────────────────────────────────

  /**
   * Set playhead frame with optional magnetic snapping.
   * If bypassSnapping or Shift key is held, snaps are ignored for sub-frame precision.
   */
  setFrame: (rawFrame, bypassSnapping = false) => {
    const { maxFrames, snappingEnabled, zoom, sceneCuts, inPoint, outPoint, curves } = get();
    let target = Math.max(1, Math.min(maxFrames, rawFrame));
    let snappedLine = null;

    if (snappingEnabled && !bypassSnapping) {
      const snapThresholdFrames = SNAP_THRESHOLD_PX / zoom;
      let minDiff = Infinity;
      let bestSnap = null;

      // Candidates: Scene cuts, in/out clamps, and keyframe points
      const candidates = [...sceneCuts, inPoint, outPoint];
      for (const curve of curves) {
        for (const kf of curve.keyframes) {
          candidates.push(kf.frame);
        }
      }

      for (const cand of candidates) {
        const diff = Math.abs(cand - target);
        if (diff <= snapThresholdFrames && diff < minDiff) {
          minDiff = diff;
          bestSnap = cand;
        }
      }

      if (bestSnap !== null) {
        target = bestSnap;
        snappedLine = { frame: bestSnap };
      }
    }

    set({ frame: target, activeSnapLine: snappedLine });
  },

  setInPoint: (f) => {
    const { outPoint } = get();
    const clamped = Math.max(1, Math.min(outPoint, Math.round(f)));
    set({ inPoint: clamped });
  },

  setOutPoint: (f) => {
    const { inPoint, maxFrames } = get();
    const clamped = Math.max(inPoint, Math.min(maxFrames, Math.round(f)));
    set({ outPoint: clamped });
  },

  setInPointToCurrent: () => {
    const { frame } = get();
    get().setInPoint(frame);
  },

  setOutPointToCurrent: () => {
    const { frame } = get();
    get().setOutPoint(frame);
  },

  clearInOutRange: () => {
    const { maxFrames } = get();
    set({ inPoint: 1, outPoint: maxFrames });
  },

  setZoom: (newZoom, anchorClientX = null) => {
    const { zoom, scrollLeft, canvasWidth, maxFrames } = get();
    const clampedZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newZoom));
    if (clampedZoom === zoom) return;

    let newScrollLeft = scrollLeft;
    if (anchorClientX !== null) {
      // Zoom centered on the cursor position
      const frameAtCursor = 1 + (scrollLeft + anchorClientX) / zoom;
      newScrollLeft = (frameAtCursor - 1) * clampedZoom - anchorClientX;
    } else {
      // Zoom centered on canvas center
      const centerX = canvasWidth / 2;
      const frameAtCenter = 1 + (scrollLeft + centerX) / zoom;
      newScrollLeft = (frameAtCenter - 1) * clampedZoom - centerX;
    }

    const maxScroll = Math.max(0, (maxFrames - 1) * clampedZoom - canvasWidth + 100);
    newScrollLeft = Math.max(0, Math.min(maxScroll, newScrollLeft));

    set({ zoom: clampedZoom, scrollLeft: newScrollLeft });
  },

  setScrollLeft: (s) => {
    const { zoom, maxFrames, canvasWidth } = get();
    const maxScroll = Math.max(0, (maxFrames - 1) * zoom - canvasWidth + 100);
    set({ scrollLeft: Math.max(0, Math.min(maxScroll, s)) });
  },

  setDimensions: (width, height) => {
    set({ canvasWidth: Math.max(1, width), canvasHeight: Math.max(1, height) });
  },

  togglePlay: () => {
    const { isPlaying } = get();
    if (isPlaying) {
      set({ isPlaying: false, shuttleSpeed: 0 });
    } else {
      set({ isPlaying: true, shuttleSpeed: 1 });
    }
  },

  setPlaying: (isPlaying) => {
    set({ isPlaying, shuttleSpeed: isPlaying ? 1 : 0 });
  },

  /**
   * J-K-L Shuttling Engine:
   * J (reverse): -1x -> -2x -> -4x -> -8x
   * K (pause): stops playback
   * L (forward): 1x -> 2x -> 4x -> 8x
   */
  shuttle: (action) => {
    const { shuttleSpeed } = get();
    if (action === 'pause') {
      set({ isPlaying: false, shuttleSpeed: 0 });
      return;
    }
    if (action === 'forward') {
      let nextSpeed;
      if (shuttleSpeed <= 0) nextSpeed = 1;
      else if (shuttleSpeed === 1) nextSpeed = 2;
      else if (shuttleSpeed === 2) nextSpeed = 4;
      else nextSpeed = 8;
      set({ isPlaying: true, shuttleSpeed: nextSpeed });
      return;
    }
    if (action === 'reverse') {
      let nextSpeed;
      if (shuttleSpeed >= 0) nextSpeed = -1;
      else if (shuttleSpeed === -1) nextSpeed = -2;
      else if (shuttleSpeed === -2) nextSpeed = -4;
      else nextSpeed = -8;
      set({ isPlaying: true, shuttleSpeed: nextSpeed });
    }
  },

  stepFrame: (delta) => {
    const { frame, maxFrames } = get();
    const next = Math.max(1, Math.min(maxFrames, Math.round(frame + delta)));
    set({ frame: next, isPlaying: false, shuttleSpeed: 0, activeSnapLine: null });
  },

  toggleCurves: () => {
    set((state) => ({ curvesExpanded: !state.curvesExpanded }));
  },

  setActiveCurveId: (activeCurveId) => {
    set({ activeCurveId });
  },

  addKeyframe: (curveId, frame, value, type = 'bezier') => {
    set((state) => {
      const curves = state.curves.map((curve) => {
        if (curve.id !== curveId) return curve;
        const clampedVal = Math.max(curve.min, Math.min(curve.max, value));
        const filtered = curve.keyframes.filter((k) => k.frame !== frame);
        const nextKeyframes = [...filtered, { frame, value: clampedVal, type }].sort(
          (a, b) => a.frame - b.frame
        );
        return { ...curve, keyframes: nextKeyframes };
      });
      return { curves };
    });
  },

  updateKeyframe: (curveId, index, frame, value) => {
    set((state) => {
      const curves = state.curves.map((curve) => {
        if (curve.id !== curveId) return curve;
        const kfs = [...curve.keyframes];
        if (!kfs[index]) return curve;
        const clampedVal = Math.max(curve.min, Math.min(curve.max, value));
        kfs[index] = { ...kfs[index], frame: Math.round(frame), value: clampedVal };
        kfs.sort((a, b) => a.frame - b.frame);
        return { ...curve, keyframes: kfs };
      });
      return { curves };
    });
  },

  deleteKeyframe: (curveId, index) => {
    set((state) => {
      const curves = state.curves.map((curve) => {
        if (curve.id !== curveId) return curve;
        const kfs = curve.keyframes.filter((_, i) => i !== index);
        return { ...curve, keyframes: kfs };
      });
      return { curves };
    });
  },

  setSceneCuts: (sceneCuts) => {
    set({ sceneCuts: [...sceneCuts].sort((a, b) => a - b) });
  },

  setIdentityTracks: (identityTracks) => {
    set({ identityTracks });
  },
}));
