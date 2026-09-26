// ── Timeline Pointer & Keyboard Interactions ────────────────────────────────
//
// Manages:
//   - Pointer scrubbing & sub-frame scrubbing with magnetic snapping
//   - In/Out point clamp dragging
//   - Identity block click-to-jump
//   - Keyframe node dragging & curve manipulation
//   - Keyboard controls: Space (Play/Pause), J-K-L (Shuttle), Arrows (Step 1f), I/O (Trim)
//   - Shift-key bypass for snapping engine

import { useCallback, useRef, useState } from 'react';
import { useTimelineStore } from './timelineStore.js';
import { TRACK_CONFIG, xToFrame } from './timelineRenderer.js';

export function useTimelineInteractions({ canvasRef }) {
  const [isScrubbing, setIsScrubbing] = useState(false);
  const dragModeRef = useRef(null); // 'scrub' | 'inPoint' | 'outPoint' | 'pan' | 'keyframe'
  const activePointerIdRef = useRef(null);
  const lastPointerPosRef = useRef({ x: 0, y: 0 });
  const draggedKeyframeRef = useRef(null); // { curveId, index }

  // ── Hit Testing ──────────────────────────────────────────────────────────
  const hitTest = useCallback((canvasX, canvasY) => {
    const {
      zoom,
      scrollLeft,
      inPoint,
      outPoint,
      identityTracks,
      curvesExpanded,
      curves,
      activeCurveId,
    } = useTimelineStore.getState();

    const clickedFrame = xToFrame(canvasX, zoom, scrollLeft);

    let currentY = 0;

    // Track 1: Ruler
    const rulerH = TRACK_CONFIG.RULER_HEIGHT;
    if (canvasY >= currentY && canvasY < currentY + rulerH) {
      return { type: 'ruler', frame: clickedFrame };
    }
    currentY += rulerH + TRACK_CONFIG.TRACK_GAP;

    // Track 2: Master Video Track (Check In/Out clamp handles)
    const videoH = TRACK_CONFIG.VIDEO_HEIGHT;
    if (canvasY >= currentY && canvasY < currentY + videoH) {
      const inX = (inPoint - 1) * zoom - scrollLeft;
      const outX = (outPoint - 1) * zoom - scrollLeft;

      if (Math.abs(canvasX - inX) <= 8) {
        return { type: 'inHandle' };
      }
      if (Math.abs(canvasX - outX) <= 8) {
        return { type: 'outHandle' };
      }
      return { type: 'videoTrack', frame: clickedFrame };
    }
    currentY += videoH + TRACK_CONFIG.TRACK_GAP;

    // Track 3+: Identity Sub-tracks
    const idH = TRACK_CONFIG.IDENTITY_HEIGHT;
    for (let i = 0; i < identityTracks.length; i++) {
      const track = identityTracks[i];
      if (canvasY >= currentY && canvasY < currentY + idH) {
        // Check if user clicked inside an active presence block
        for (const [start, end] of track.intervals || []) {
          if (clickedFrame >= start && clickedFrame <= end) {
            return { type: 'identityBlock', trackId: track.id, jumpFrame: start };
          }
        }
        return { type: 'identityTrack', trackId: track.id, frame: clickedFrame };
      }
      currentY += idH + TRACK_CONFIG.TRACK_GAP;
    }

    // Track 4: Keyframe Automation Curves
    if (curvesExpanded && curves && curves.length > 0) {
      const curveH = TRACK_CONFIG.CURVES_HEIGHT;
      if (canvasY >= currentY && canvasY < currentY + curveH) {
        const activeCurve = curves.find((c) => c.id === activeCurveId) || curves[0];
        // Check if clicking near any keyframe node
        for (let k = 0; k < activeCurve.keyframes.length; k++) {
          const kf = activeCurve.keyframes[k];
          const kfX = (kf.frame - 1) * zoom - scrollLeft;
          const normVal = (kf.value - activeCurve.min) / (activeCurve.max - activeCurve.min);
          const kfY = currentY + curveH - 10 - normVal * (curveH - 20);

          const distSq = (canvasX - kfX) ** 2 + (canvasY - kfY) ** 2;
          if (distSq <= 64) { // 8px radius
            return {
              type: 'keyframeNode',
              curveId: activeCurve.id,
              index: k,
              frame: kf.frame,
              value: kf.value,
            };
          }
        }
        return {
          type: 'curveTrack',
          curveId: activeCurve.id,
          frame: clickedFrame,
          yRatio: 1 - (canvasY - currentY - 10) / (curveH - 20),
        };
      }
    }

    return { type: 'background', frame: clickedFrame };
  }, []);

  // ── Pointer Down ────────────────────────────────────────────────────────
  const handlePointerDown = useCallback(
    (e) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const canvasX = e.clientX - rect.left;
      const canvasY = e.clientY - rect.top;

      // Middle click or Alt+Drag -> Pan timeline
      if (e.button === 1 || (e.button === 0 && e.altKey)) {
        dragModeRef.current = 'pan';
        lastPointerPosRef.current = { x: e.clientX, y: e.clientY };
        activePointerIdRef.current = e.pointerId;
        canvas.setPointerCapture?.(e.pointerId);
        return;
      }

      if (e.button !== 0) return; // Left click only for actions

      const hit = hitTest(canvasX, canvasY);

      if (hit.type === 'inHandle') {
        dragModeRef.current = 'inPoint';
      } else if (hit.type === 'outHandle') {
        dragModeRef.current = 'outPoint';
      } else if (hit.type === 'identityBlock') {
        // Jump playhead directly to identity appearance start
        useTimelineStore.getState().setFrame(hit.jumpFrame, false);
        return;
      } else if (hit.type === 'keyframeNode') {
        dragModeRef.current = 'keyframe';
        draggedKeyframeRef.current = { curveId: hit.curveId, index: hit.index };
      } else if (hit.type === 'curveTrack' && e.detail === 2) {
        // Double-click on curve track creates a new keyframe
        const { curves } = useTimelineStore.getState();
        const curve = curves.find((c) => c.id === hit.curveId);
        if (curve) {
          const clampedVal = curve.min + Math.max(0, Math.min(1, hit.yRatio)) * (curve.max - curve.min);
          useTimelineStore.getState().addKeyframe(hit.curveId, Math.round(hit.frame), clampedVal);
        }
        return;
      } else {
        // Default to playhead scrub
        dragModeRef.current = 'scrub';
        const bypassSnap = e.shiftKey;
        useTimelineStore.getState().setFrame(hit.frame, bypassSnap);
        setIsScrubbing(true);
      }

      activePointerIdRef.current = e.pointerId;
      lastPointerPosRef.current = { x: e.clientX, y: e.clientY };
      try {
        canvas.setPointerCapture?.(e.pointerId);
      } catch {
        // Ignore
      }
    },
    [canvasRef, hitTest]
  );

  // ── Pointer Move ────────────────────────────────────────────────────────
  const handlePointerMove = useCallback(
    (e) => {
      const mode = dragModeRef.current;
      if (!mode) return;

      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const canvasX = e.clientX - rect.left;
      const canvasY = e.clientY - rect.top;

      const { zoom, scrollLeft, setScrollLeft, setFrame, setInPoint, setOutPoint, updateKeyframe, curves } =
        useTimelineStore.getState();

      const currentFrame = xToFrame(canvasX, zoom, scrollLeft);
      const bypassSnap = e.shiftKey;

      if (mode === 'scrub') {
        setFrame(currentFrame, bypassSnap);
      } else if (mode === 'inPoint') {
        setInPoint(currentFrame);
      } else if (mode === 'outPoint') {
        setOutPoint(currentFrame);
      } else if (mode === 'pan') {
        const dx = e.clientX - lastPointerPosRef.current.x;
        lastPointerPosRef.current = { x: e.clientX, y: e.clientY };
        setScrollLeft(scrollLeft - dx);
      } else if (mode === 'keyframe' && draggedKeyframeRef.current) {
        const { curveId, index } = draggedKeyframeRef.current;
        const curve = curves.find((c) => c.id === curveId);
        if (curve) {
          const curveY =
            TRACK_CONFIG.RULER_HEIGHT +
            TRACK_CONFIG.VIDEO_HEIGHT +
            TRACK_CONFIG.IDENTITY_HEIGHT * useTimelineStore.getState().identityTracks.length +
            TRACK_CONFIG.TRACK_GAP * 3;
          const curveH = TRACK_CONFIG.CURVES_HEIGHT;
          const normY = 1 - (canvasY - curveY - 10) / (curveH - 20);
          const newVal = curve.min + Math.max(0, Math.min(1, normY)) * (curve.max - curve.min);
          updateKeyframe(curveId, index, currentFrame, newVal);
        }
      }
    },
    [canvasRef]
  );

  // ── Pointer Up ──────────────────────────────────────────────────────────
  const handlePointerUp = useCallback(
    (e) => {
      if (activePointerIdRef.current === e.pointerId) {
        dragModeRef.current = null;
        draggedKeyframeRef.current = null;
        activePointerIdRef.current = null;
        setIsScrubbing(false);
        try {
          canvasRef.current?.releasePointerCapture?.(e.pointerId);
        } catch {
          // Ignore
        }
      }
    },
    [canvasRef]
  );

  // ── Wheel Zoom & Scroll ─────────────────────────────────────────────────
  const handleWheel = useCallback(
    (e) => {
      e.preventDefault();
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const clientX = e.clientX - rect.left;

      const { zoom, scrollLeft, setZoom, setScrollLeft } = useTimelineStore.getState();

      if (e.ctrlKey || e.metaKey || e.altKey) {
        // Zoom centered on cursor
        const factor = Math.exp(-e.deltaY * 0.002);
        setZoom(zoom * factor, clientX);
      } else {
        // Horizontal scroll
        const delta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
        setScrollLeft(scrollLeft + delta * 0.8);
      }
    },
    [canvasRef]
  );

  // ── Keyboard Controls (Space, J-K-L, Arrows, I/O) ───────────────────────
  const handleKeyDown = useCallback((e) => {
    // Ignore if user is typing into an input or textarea
    if (['INPUT', 'TEXTAREA'].includes(e.target?.tagName)) return;

    const store = useTimelineStore.getState();

    if (e.code === 'Space') {
      e.preventDefault();
      store.togglePlay();
    } else if (e.key === 'j' || e.key === 'J') {
      e.preventDefault();
      store.shuttle('reverse');
    } else if (e.key === 'k' || e.key === 'K') {
      e.preventDefault();
      store.shuttle('pause');
    } else if (e.key === 'l' || e.key === 'L') {
      e.preventDefault();
      store.shuttle('forward');
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      const step = e.shiftKey ? -10 : -1;
      store.stepFrame(step);
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      const step = e.shiftKey ? 10 : 1;
      store.stepFrame(step);
    } else if (e.key === 'i' || e.key === 'I') {
      e.preventDefault();
      store.setInPointToCurrent();
    } else if (e.key === 'o' || e.key === 'O') {
      e.preventDefault();
      store.setOutPointToCurrent();
    }
  }, []);

  return {
    isScrubbing,
    handlePointerDown,
    handlePointerMove,
    handlePointerUp,
    handleWheel,
    handleKeyDown,
  };
}
