import { useState, useRef, useEffect, useCallback } from 'react';

const MIN_ZOOM = 0.1;  // 10%
const MAX_ZOOM = 16.0; // 1600%
const LERP_SPEED = 0.25;

/**
 * Custom hook providing pan, zoom, and inspection controls with:
 * - Smooth wheel delta zoom centered on cursor
 * - Infinite canvas drag-to-pan
 * - Pixel inspection coordinates
 */
export function useCinematicControls({ canvasRef, onTransformChange, onCursorMove }) {
  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);

  // Mutable refs for high-frequency rAF animation without React re-render churn
  const zoomRef = useRef(1.0);
  const targetZoomRef = useRef(1.0);
  const panRef = useRef({ x: 0, y: 0 });
  const anchorRef = useRef(null); // { dx, dy } offset from viewport center
  const rAFRef = useRef(null);
  const isPointerDownRef = useRef(false);
  const activePointerIdRef = useRef(null);
  const lastPointerPosRef = useRef({ x: 0, y: 0 });

  // Update refs when props/state change
  useEffect(() => {
    zoomRef.current = zoom;
    targetZoomRef.current = zoom;
    panRef.current = pan;
  }, [zoom, pan]);

  // Smooth Zoom Animation Loop
  const tickAnimation = useCallback(() => {
    const curZoom = zoomRef.current;
    const tgtZoom = targetZoomRef.current;
    const diff = tgtZoom - curZoom;

    if (Math.abs(diff) > 0.0005) {
      const nextZoom = curZoom + diff * LERP_SPEED;
      const alpha = nextZoom / curZoom;

      // Adjust pan to preserve anchor under cursor
      if (anchorRef.current) {
        const { dx, dy } = anchorRef.current;
        panRef.current.x = dx - (dx - panRef.current.x) * alpha;
        panRef.current.y = dy - (dy - panRef.current.y) * alpha;
      }

      zoomRef.current = nextZoom;
      setZoom(nextZoom);
      setPan({ ...panRef.current });
      onTransformChange?.(panRef.current.x, panRef.current.y, nextZoom);

      rAFRef.current = requestAnimationFrame(tickAnimation);
    } else {
      // Snapped to target
      zoomRef.current = tgtZoom;
      setZoom(tgtZoom);
      setPan({ ...panRef.current });
      onTransformChange?.(panRef.current.x, panRef.current.y, tgtZoom);
      rAFRef.current = null;
      anchorRef.current = null;
    }
  }, [onTransformChange]);

  const scheduleZoom = useCallback((newTargetZoom, anchor) => {
    targetZoomRef.current = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newTargetZoom));
    anchorRef.current = anchor;
    if (!rAFRef.current) {
      rAFRef.current = requestAnimationFrame(tickAnimation);
    }
  }, [tickAnimation]);

  // Wheel handler for smooth zoom centered on mouse
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Viewport center offset
    const dx = mx - rect.width / 2;
    const dy = my - rect.height / 2;

    const deltaY = e.deltaY;
    const factor = e.deltaMode === 1 ? 0.04 : 0.0018;
    const multiplier = Math.exp(-deltaY * factor);

    const nextTgt = targetZoomRef.current * multiplier;
    scheduleZoom(nextTgt, { dx, dy });
  }, [canvasRef, scheduleZoom]);

  // Pointer Down (Drag-to-pan start)
  const handlePointerDown = useCallback((e) => {
    // Left click or middle click starts pan
    if (e.button !== 0 && e.button !== 1) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    isPointerDownRef.current = true;
    activePointerIdRef.current = e.pointerId;
    lastPointerPosRef.current = { x: e.clientX, y: e.clientY };

    try {
      canvas.setPointerCapture(e.pointerId);
    } catch {
      // Ignore if capture fails
    }

    setIsDragging(true);
  }, [canvasRef]);

  // Pointer Move (Infinite drag-to-pan and inspector tracking)
  const handlePointerMove = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;

    if (isPointerDownRef.current) {
      const dx = e.clientX - lastPointerPosRef.current.x;
      const dy = e.clientY - lastPointerPosRef.current.y;
      lastPointerPosRef.current = { x: e.clientX, y: e.clientY };

      panRef.current.x += dx;
      panRef.current.y += dy;
      setPan({ ...panRef.current });
      onTransformChange?.(panRef.current.x, panRef.current.y, zoomRef.current);
    }

    // Always report cursor position for pixel inspection
    onCursorMove?.({
      canvasX,
      canvasY,
      clientX: e.clientX,
      clientY: e.clientY,
      inBounds: canvasX >= 0 && canvasX <= rect.width && canvasY >= 0 && canvasY <= rect.height,
    });
  }, [canvasRef, onTransformChange, onCursorMove]);

  // Pointer Up / Cancel
  const handlePointerUp = useCallback((e) => {
    if (activePointerIdRef.current === e.pointerId) {
      isPointerDownRef.current = false;
      activePointerIdRef.current = null;
      setIsDragging(false);
      try {
        canvasRef.current?.releasePointerCapture(e.pointerId);
      } catch {
        // Ignore
      }
    }
  }, [canvasRef]);

  // Reset View to fit / default
  const resetView = useCallback(() => {
    if (rAFRef.current) {
      cancelAnimationFrame(rAFRef.current);
      rAFRef.current = null;
    }
    targetZoomRef.current = 1.0;
    zoomRef.current = 1.0;
    panRef.current = { x: 0, y: 0 };
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
    onTransformChange?.(0, 0, 1.0);
  }, [onTransformChange]);

  // Set explicit zoom percentage
  const setZoomLevel = useCallback((newZoom) => {
    scheduleZoom(newZoom, { dx: 0, dy: 0 });
  }, [scheduleZoom]);

  // Cleanup rAF on unmount
  useEffect(() => {
    return () => {
      if (rAFRef.current) {
        cancelAnimationFrame(rAFRef.current);
      }
    };
  }, []);

  return {
    zoom,
    pan,
    isDragging,
    handleWheel,
    handlePointerDown,
    handlePointerMove,
    handlePointerUp,
    resetView,
    setZoomLevel,
  };
}
