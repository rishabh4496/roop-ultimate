import React, {
  useCallback, useEffect, useRef, useState, useMemo,
} from 'react';
import { useGesture } from '@use-gesture/react';
import AIScannerOverlay from './AIScannerOverlay';
import PreviewCanvas from './PreviewCanvas';
import CompareSlider from './CompareSlider';
import FrameUnavailable from './FrameUnavailable';
import { brushSample } from './maskBrush';
import RotoLayer, { RotoToolbar } from './RotoLayer';
import { trackLabel } from './identityLock';
import FastCanvasPlayer from '../player/FastCanvasPlayer';
import RadarNavigator from './RadarNavigator';
import BroadcastScopesDock from './BroadcastScopesDock';
import ColorGradingPanel from './ColorGradingPanel';
import StereoVRViewport, { DEFAULT_VR_PARAMS } from './StereoVRViewport';
import { GRADING_DEFAULTS } from './webgl2GradingPipeline';
import QuadInspectionGrid from './QuadInspectionGrid';
import {
  createReferenceChannel,
  closeReferenceChannel,
  broadcastFrame,
  broadcastTransport,
  broadcastCompare,
  broadcastGrading,
  MESSAGE_TYPES,
} from '../monitor/referenceChannel';
import {
  ZOOM_MIN, ZOOM_MAX, clampPan, panAnchoredAt, panCenteringAt,
  transformFor, uiScale, wheelZoom, faceTrackingTarget,
} from './zoomPan';

export default function InteractivePreview({
  beforeSrc,
  afterSrc,
  beforeFrame = null,
  afterFrame = null,
  faces = [],
  kps = [],
  pose = [],
  personIds = [],
  selectedFaceIndex = null,
  onSelectFace,
  faceRoles = [],
  onToggleExclude,
  splitView = false,
  compare = false,
  onToggleCompare,
  sliderEffectEnabled: _sliderEffectEnabled = true,
  onToggleSliderEffect: _onToggleSliderEffect,
  frame = 1,
  setFrame,
  maxFrames = 1,
  isPlaying = false,
  setIsPlaying,
  previewing = false,
  previewSecs = 0,
  scrubbing = false,
  onMaskChange,
  maskApplied = false,
  // Stage 11: vector roto zones (Pen tool) and identity pins. `faceAnchors` is
  // [{ source_index, nose, chin, bbox }] for the faces on THIS preview frame.
  rotoZones = [],
  onRotoZones,
  rotoSelectedId = null,
  onRotoSelect,
  faceAnchors = [],
  faceSources = [],
  pins = [],
  onPinIdentity,
  onUnpinIdentity,
  playbackSource = null,
  frameUnavailable = false,
  onRetryFrame,
  sarVal = 1.0,
  darVal = 0.0,
  colorBadge = '',
  notify = null,
  onGradingChange = null,
}) {
  // ── State & Refs ──────────────────────────────────────────────────────────
  const [sliderPosition, setSliderPosition] = useState(50);
  const sliderPosRef = useRef(50);

  // Advanced comparison modes: 'slider' | 'blend' | 'quad' | 'heatmap' | 'diff'
  const [compareMode, setCompareMode] = useState('slider');
  const [compareDir, setCompareDir] = useState('vertical'); // 'vertical' | 'horizontal'
  const [wipeAngle, setWipeAngle] = useState(0);            // 0..360 vector wipe angle
  const [diffGain, setDiffGain] = useState(3.0);            // 1.0..10.0 amplified diff gain
  const [diffColorMap, setDiffColorMap] = useState('turbo');// 'turbo' | 'inferno'
  const [autoSwipe, setAutoSwipe] = useState(false);

  const containerRef = useRef(null);
  const stageViewportRef = useRef(null);
  const imageRef = useRef(null);
  const canvasApiRef = useRef(null);
  const compareApiRef = useRef(null);
  const maskCanvasRef = useRef(null);
  const maskExportRef = useRef(null);
  const lastPtRef = useRef(null);

  // ── Zero-Rerender Viewport Controller (0.1x to 16x) ──────────────────────
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const zoomRef = useRef(1);
  const panRef = useRef({ x: 0, y: 0 });
  const zoomReadoutRef = useRef(null);
  const isInteractingRef = useRef(false);
  const isSpaceDownRef = useRef(false);
  const inertiaRafRef = useRef(null);
  const settleTimeoutRef = useRef(null);

  const [, setIsPanning] = useState(false);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showDebug, setShowDebug] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isPeekingOriginal, setIsPeekingOriginal] = useState(false);
  const [lockTargetFace, setLockTargetFace] = useState(false);
  const lockTargetFaceRef = useRef(false);
  lockTargetFaceRef.current = lockTargetFace;
  const smoothedFaceCameraRef = useRef({ zoom: 1, pan: { x: 0, y: 0 } });

  // Feature: 3.5x–8x Floating Magnifier Loupe
  const [magnifierActive, setMagnifierActive] = useState(false);
  const [loupePower, setLoupePower] = useState(3.5); // 3.5x | 5.0x | 8.0x
  const [loupeSplit] = useState(true); // Split Before/After inside loupe
  const lensGlassRef = useRef(null);
  const lensBeforeImgRef = useRef(null);
  const lensAfterImgRef = useRef(null);
  const lensPendingRef = useRef(null);
  const lensRafRef = useRef(null);
  const lensBoxRef = useRef('');

  // Feature: Sub-Pixel Mask Brush & Dual Concentric Cursor
  const [maskBrushActive, setMaskBrushActive] = useState(false);
  const [brushMode, setBrushMode] = useState('paint'); // 'paint' | 'erase'
  const [brushSize, setBrushSize] = useState(30);
  const [brushFeather, setBrushFeather] = useState(0.35); // 0.0 (hard) .. 1.0 (soft)
  const [, setIsDrawingMask] = useState(false);
  const brushCursorRef = useRef(null);

  // Stage 11: the Pen tool (P). Mutually exclusive with the Mask Brush: both want the left button.
  const [rotoActive, setRotoActive] = useState(false);
  const [rotoNewMode, setRotoNewMode] = useState('exclude');
  const toggleBrush = useCallback(() => {
    setMaskBrushActive((b) => { if (!b) setRotoActive(false); return !b; });
  }, []);
  const toggleRoto = useCallback(() => {
    setRotoActive((a) => { if (!a) setMaskBrushActive(false); return !a; });
  }, []);

  // Feature: Broadcast Scopes & 3D LUT Color Grading
  const [scopeActive, setScopeActive] = useState(false);
  const [gradingActive, setGradingActive] = useState(false);
  const [vrActive, setVrActive] = useState(false);
  const [vrParams, setVrParams] = useState(DEFAULT_VR_PARAMS);
  const [gradingParams, setGradingParams] = useState(GRADING_DEFAULTS);
  const [_activeLut, setActiveLut] = useState(null);
  // Feature: False-Color Luminance & Skin-Tone Zebra Overlay
  const [zebraActive, setZebraActive] = useState(false);
  const [monitorActive, setMonitorActive] = useState(false);
  const refChannelRef = useRef(null);

  const openMonitor = useCallback(() => {
    setMonitorActive(true);
    if (typeof window !== 'undefined') {
      window.open(
        '#/monitor',
        'roop_clean_monitor',
        'width=1280,height=720,menubar=no,toolbar=no,location=no,status=no,resizable=yes'
      );
    }
  }, []);

  useEffect(() => {
    if (!monitorActive) return undefined;
    const channel = createReferenceChannel((data) => {
      if (data?.type === MESSAGE_TYPES.PING) {
        broadcastTransport(channel, {
          isPlaying,
          currentFrame: frame,
          totalFrames: maxFrames,
          fps: 30,
        });
        broadcastCompare(channel, {
          mode: compareMode,
          wipe: sliderPosRef.current ?? 50,
          compareDir,
          wipeAngle,
          diffGain,
        });
        broadcastGrading(channel, {
          ...gradingParams,
          zebraActive,
        });
      }
    });
    refChannelRef.current = channel;

    let animId = 0;
    let lastTs = 0;
    let isBroadcasting = false;

    const streamLoop = (ts) => {
      animId = requestAnimationFrame(streamLoop);
      if (ts - lastTs < 16) return; // ~60fps
      lastTs = ts;

      if (isBroadcasting) return;
      const cvs = canvasApiRef.current?.canvas?.();
      if (!cvs || cvs.width === 0 || cvs.height === 0) return;

      if (typeof createImageBitmap === 'function') {
        isBroadcasting = true;
        createImageBitmap(cvs)
          .then((bmp) => {
            broadcastFrame(channel, bmp, {
              width: cvs.width,
              height: cvs.height,
              frameIndex: frame,
              timestamp: Date.now(),
            });
          })
          .catch(() => {})
          .finally(() => {
            isBroadcasting = false;
          });
      }
    };
    animId = requestAnimationFrame(streamLoop);

    return () => {
      if (animId) cancelAnimationFrame(animId);
      closeReferenceChannel(channel);
      refChannelRef.current = null;
    };
  }, [monitorActive, frame, maxFrames, isPlaying, compareMode, compareDir, wipeAngle, diffGain, gradingParams, zebraActive]);

  useEffect(() => {
    if (!refChannelRef.current) return;
    broadcastTransport(refChannelRef.current, {
      isPlaying,
      currentFrame: frame,
      totalFrames: maxFrames,
      fps: 30,
    });
  }, [isPlaying, frame, maxFrames]);

  useEffect(() => {
    if (!refChannelRef.current) return;
    broadcastCompare(refChannelRef.current, {
      mode: compareMode,
      wipe: sliderPosition,
      compareDir,
      wipeAngle,
      diffGain,
    });
  }, [compareMode, sliderPosition, compareDir, wipeAngle, diffGain]);

  useEffect(() => {
    if (!refChannelRef.current) return;
    broadcastGrading(refChannelRef.current, {
      ...gradingParams,
      zebraActive,
    });
  }, [gradingParams, zebraActive]);

  const activeFaceBbox = useMemo(() => {
    if (!faces || faces.length === 0) return null;
    const idx = (typeof selectedFaceIndex === 'number' && selectedFaceIndex >= 0 && selectedFaceIndex < faces.length)
      ? selectedFaceIndex
      : 0;
    const f = faces[idx];
    if (!f) return null;
    return f.bbox || (Array.isArray(f) ? f : null);
  }, [faces, selectedFaceIndex]);

  const [imgDim, setImgDim] = useState(null);
  const splitMode = compare && (splitView || compareMode === 'quad');

  // Direct DOM Viewport Update (Zero-Rerender 60fps)
  const applyViewportDom = useCallback((z, p) => {
    zoomRef.current = z;
    panRef.current = p;
    if (stageViewportRef.current) {
      stageViewportRef.current.style.transform = `scale(${z}) translate(${p.x / z}px, ${p.y / z}px)`;
    }
    if (zoomReadoutRef.current) {
      zoomReadoutRef.current.textContent = `${Math.round(z * 100)}%`;
    }
  }, []);

  const scheduleSettle = useCallback((z, p) => {
    if (settleTimeoutRef.current) clearTimeout(settleTimeoutRef.current);
    settleTimeoutRef.current = setTimeout(() => {
      setZoom(z);
      setPan(p);
    }, 120);
  }, []);

  const updateViewport = useCallback((z, p, commit = false) => {
    applyViewportDom(z, p);
    if (commit) {
      setZoom(z);
      setPan(p);
    } else {
      scheduleSettle(z, p);
    }
  }, [applyViewportDom, scheduleSettle]);

  // Inertial momentum pan decay
  const stopInertia = useCallback(() => {
    if (inertiaRafRef.current) {
      cancelAnimationFrame(inertiaRafRef.current);
      inertiaRafRef.current = null;
    }
  }, []);

  useEffect(() => () => {
    if (inertiaRafRef.current) cancelAnimationFrame(inertiaRafRef.current);
    if (lensRafRef.current) cancelAnimationFrame(lensRafRef.current);
    if (settleTimeoutRef.current) clearTimeout(settleTimeoutRef.current);
  }, []);

  const startInertia = useCallback((initVx, initVy) => {
    stopInertia();
    let vx = initVx;
    let vy = initVy;
    const decay = 0.90;
    const step = () => {
      vx *= decay;
      vy *= decay;
      if (Math.abs(vx) < 0.25 && Math.abs(vy) < 0.25) {
        inertiaRafRef.current = null;
        updateViewport(zoomRef.current, panRef.current, true);
        return;
      }
      const stage = containerRef.current;
      const content = imageRef.current;
      const nextPan = clampPan({
        x: panRef.current.x + vx,
        y: panRef.current.y + vy,
      }, zoomRef.current, stage, content);
      updateViewport(zoomRef.current, nextPan, false);
      inertiaRafRef.current = requestAnimationFrame(step);
    };
    inertiaRafRef.current = requestAnimationFrame(step);
  }, [stopInertia, updateViewport]);

  // Lock-to-Face Auto-Follow Camera Engine
  useEffect(() => {
    if (!lockTargetFace || !imgDim || !faces.length) return;
    const targetIdx = (selectedFaceIndex !== null && faces[selectedFaceIndex]) ? selectedFaceIndex : 0;
    const bbox = faces[targetIdx];
    const faceKps = kps[targetIdx];
    const target = faceTrackingTarget(bbox, faceKps, imgDim, containerRef.current);
    if (!target) return;

    // Smoothed EMA interpolation so camera does not jitter across video frames
    const cur = smoothedFaceCameraRef.current;
    const alpha = 0.22;
    cur.zoom += (target.zoom - cur.zoom) * alpha;
    cur.pan.x += (target.pan.x - cur.pan.x) * alpha;
    cur.pan.y += (target.pan.y - cur.pan.y) * alpha;

    updateViewport(cur.zoom, cur.pan, false);
  }, [lockTargetFace, frame, faces, kps, selectedFaceIndex, imgDim, updateViewport]);

  // Quick Preset Actions
  const handlePresetFit = useCallback(() => {
    stopInertia();
    setLockTargetFace(false);
    updateViewport(1.0, { x: 0, y: 0 }, true);
  }, [stopInertia, updateViewport]);

  const handlePresetFill = useCallback(() => {
    stopInertia();
    setLockTargetFace(false);
    const stage = containerRef.current;
    if (!stage || !imgDim) return;
    const ss = uiScale(stage);
    const r = stage.getBoundingClientRect();
    const sw = r.width / ss;
    const sh = r.height / ss;
    const s = Math.max(sw / imgDim.w, sh / imgDim.h);
    const fitScale = Math.min(sw / imgDim.w, sh / imgDim.h);
    const fillZoom = fitScale > 0 ? Math.min(ZOOM_MAX, s / fitScale) : 1.25;
    updateViewport(fillZoom, { x: 0, y: 0 }, true);
  }, [imgDim, stopInertia, updateViewport]);

  const handlePreset1to1 = useCallback(() => {
    stopInertia();
    setLockTargetFace(false);
    const el = imageRef.current || containerRef.current;
    const w = el?.offsetWidth;
    if (!w || !imgDim) return;
    const z1 = Math.min(Math.max(imgDim.w / w, ZOOM_MIN), ZOOM_MAX);
    updateViewport(z1, { x: 0, y: 0 }, true);
  }, [imgDim, stopInertia, updateViewport]);

  const handlePreset200 = useCallback(() => {
    stopInertia();
    setLockTargetFace(false);
    updateViewport(2.0, { x: 0, y: 0 }, true);
  }, [stopInertia, updateViewport]);

  const handlePreset400 = useCallback(() => {
    stopInertia();
    setLockTargetFace(false);
    updateViewport(4.0, { x: 0, y: 0 }, true);
  }, [stopInertia, updateViewport]);

  const zoomBy = useCallback((factor) => {
    stopInertia();
    setLockTargetFace(false);
    const nz = Math.min(Math.max(ZOOM_MIN, zoomRef.current * factor), ZOOM_MAX);
    const np = nz === 1 ? { x: 0, y: 0 } : clampPan(panRef.current, nz, containerRef.current, imageRef.current);
    updateViewport(nz, np, true);
  }, [stopInertia, updateViewport]);

  // Double Click toggles Fit <-> 2.5x
  const handleDoubleClick = useCallback((e) => {
    if (maskBrushActive || rotoActive) return;
    stopInertia();
    setLockTargetFace(false);
    if (zoomRef.current > 1.05) {
      handlePresetFit();
    } else {
      const z = 2.5;
      const p = panCenteringAt(
        { x: e.clientX ?? e.touches?.[0]?.clientX, y: e.clientY ?? e.touches?.[0]?.clientY },
        containerRef.current, z, imageRef.current,
      );
      updateViewport(z, p, true);
    }
  }, [maskBrushActive, rotoActive, stopInertia, handlePresetFit, updateViewport]);

  // Imperative Slider Wipe Manipulation
  const handleSliderMove = useCallback((clientX, clientY) => {
    const target = imageRef.current || containerRef.current;
    if (!target) return;
    const rect = target.getBoundingClientRect();
    let percent = null;

    if (wipeAngle !== 0) {
      const rad = (wipeAngle * Math.PI) / 180;
      const nx = Math.cos(rad);
      const ny = Math.sin(rad);
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const proj = ((clientX - cx) / rect.width) * nx + ((clientY - cy) / rect.height) * ny;
      percent = Math.max(0, Math.min(100, (0.5 - proj) * 100));
    } else if (compareDir === 'horizontal' && clientY !== undefined) {
      const y = Math.max(0, Math.min(clientY - rect.top, rect.height));
      percent = Math.max(0, Math.min((y / rect.height) * 100, 100));
    } else if (clientX !== undefined) {
      const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
      percent = Math.max(0, Math.min((x / rect.width) * 100, 100));
    }

    if (percent === null) return;
    sliderPosRef.current = percent;
    if (compareMode === 'blend') canvasApiRef.current?.setWipe(percent);
    else compareApiRef.current?.set(100 - percent);
  }, [compareDir, compareMode, wipeAngle]);

  const commitSliderPosition = (percent) => setSliderPosition(percent);

  // Auto-Swipe Animation Loop
  const autoSwipeRuns = autoSwipe && compare && compareMode !== 'diff';
  useEffect(() => {
    if (!autoSwipeRuns) return undefined;
    let animId;
    let startTime;
    let phase = 0;
    let pos = 50;
    const animate = (time) => {
      if (document.hidden || document.documentElement.hasAttribute('data-render-lite')) {
        startTime = undefined;
        animId = requestAnimationFrame(animate);
        return;
      }
      if (startTime === undefined) startTime = time - phase * 1000;
      phase = (time - startTime) / 1000;
      pos = 50 + 40 * Math.sin(phase * (Math.PI * 2 / 2.5));
      if (compareMode === 'blend') canvasApiRef.current?.setWipe(pos);
      else compareApiRef.current?.set(100 - pos);
      animId = requestAnimationFrame(animate);
    };
    animId = requestAnimationFrame(animate);
    return () => {
      cancelAnimationFrame(animId);
      setSliderPosition(pos);
    };
  }, [autoSwipeRuns, compareMode]);

  // Ends any drag — slider, pan, peek or brush stroke.

  // ── Sub-Pixel Mask Brush Engine with Radial Feather Falloff ──────────────
  const drawMaskStroke = useCallback((e, isStart = false) => {
    const canvas = maskCanvasRef.current;
    const exportCanvas = maskExportRef.current;
    if (!canvas || !exportCanvas) return;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX ?? e.touches?.[0]?.clientX;
    const cy = e.clientY ?? e.touches?.[0]?.clientY;
    if (cx === undefined) return;

    const sample = brushSample(rect, canvas.width, canvas.height, cx, cy, brushSize, brushFeather);
    if (!sample) return;
    const { x, y, r, innerR } = sample;
    const prev = isStart ? null : lastPtRef.current;
    const erase = brushMode === 'erase';

    const tint = erase ? '#ffffff' : 'rgba(233, 69, 96, 0.45)';
    for (const [c, isExport] of [[canvas, false], [exportCanvas, true]]) {
      const ctx = c.getContext('2d');
      if (!ctx) continue;
      ctx.globalCompositeOperation = erase ? 'destination-out' : 'source-over';

      if (brushFeather > 0.05) {
        const grad = ctx.createRadialGradient(x, y, Math.max(0, innerR), x, y, r);
        if (isExport) {
          grad.addColorStop(0, '#ffffff');
          grad.addColorStop(1, 'rgba(255, 255, 255, 0)');
        } else if (erase) {
          grad.addColorStop(0, '#ffffff');
          grad.addColorStop(1, 'rgba(255, 255, 255, 0)');
        } else {
          grad.addColorStop(0, tint);
          grad.addColorStop(1, 'rgba(233, 69, 96, 0)');
        }
        ctx.fillStyle = grad;
        ctx.strokeStyle = grad;
      } else {
        const color = isExport || erase ? '#ffffff' : tint;
        ctx.fillStyle = color;
        ctx.strokeStyle = color;
      }

      ctx.lineWidth = r * 2;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      if (prev) {
        ctx.beginPath();
        ctx.moveTo(prev.x, prev.y);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
    lastPtRef.current = { x, y };
  }, [brushFeather, brushMode, brushSize]);

  const commitMask = useCallback(() => {
    if (!onMaskChange) return;
    const c = maskExportRef.current;
    if (!c) return;
    let painted = false;
    try {
      const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      for (let i = 3; i < d.length; i += 16) {
        if (d[i] > 8) { painted = true; break; }
      }
    } catch {
      painted = true;
    }
    onMaskChange(painted ? c.toDataURL('image/png') : null);
  }, [onMaskChange]);

  const clearMaskCanvas = useCallback(() => {
    for (const c of [maskCanvasRef.current, maskExportRef.current]) {
      const ctx = c?.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, c.width, c.height);
    }
    lastPtRef.current = null;
    if (onMaskChange) onMaskChange(null);
  }, [onMaskChange]);

  // ── Native Non-Passive Wheel Listener ────────────────────────────────────
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return undefined;
    const onWheel = (e) => {
      // Ctrl/Cmd + wheel belongs to the app zoom (App.jsx, on window). This
      // listener runs first, so without the bail one gesture zoomed the stage
      // AND the whole UI — the same split as the keyboard, where Ctrl +/- is
      // the app and bare +/- is the stage.
      if (e.ctrlKey || e.metaKey) return;

      // Alt + Scroll Wheel adjusts Mask Brush radius
      if (e.altKey && maskBrushActive) {
        e.preventDefault();
        const step = e.deltaY < 0 ? 3 : -3;
        setBrushSize((s) => Math.max(8, Math.min(120, s + step)));
        return;
      }

      // Smooth multiplicative cursor-centered zoom (0.1x to 16x)
      const next = wheelZoom(e.deltaY, zoomRef.current, ZOOM_MAX, ZOOM_MIN, 0.15);
      // At either end of the range the stage has nothing to do with this event,
      // so it must NOT swallow it — otherwise scrolling the settings column
      // stops dead the moment the pointer crosses the preview.
      if (next === null) return;
      e.preventDefault();
      stopInertia();
      setLockTargetFace(false);

      if (next === 1) {
        updateViewport(1, { x: 0, y: 0 }, false);
        return;
      }
      const stage = containerRef.current;
      const content = imageRef.current;
      const p = panAnchoredAt({ x: e.clientX, y: e.clientY }, stage, zoomRef.current, next, panRef.current);
      const clamped = clampPan(p, next, stage, content);
      updateViewport(next, clamped, false);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [maskBrushActive, updateViewport, stopInertia]);

  // ── Gesture Engine (@use-gesture/react) ──────────────────────────────────
  useGesture(
    {
      onPinch: ({ offset: [d], origin: [ox, oy], first, event }) => {
        event?.preventDefault?.();
        stopInertia();
        setLockTargetFace(false);
        if (first) return;
        const factor = 1 + d * 0.005;
        const next = Math.min(Math.max(ZOOM_MIN, zoomRef.current * factor), ZOOM_MAX);
        const stage = containerRef.current;
        const content = imageRef.current;
        const p = panAnchoredAt({ x: ox, y: oy }, stage, zoomRef.current, next, panRef.current);
        const clamped = clampPan(p, next, stage, content);
        updateViewport(next, clamped, false);
      },

      onDragStart: ({ event }) => {
        if (event.target !== event.currentTarget && event.target.closest?.('button, a, input, select, textarea, label, [role="button"]')) {
          return;
        }
        stopInertia();
        if (rotoActive) return;           // the Pen tool's layer handles its own pointer
        if (maskBrushActive) {
          setIsDrawingMask(true);
          drawMaskStroke(event, true);
        }
      },

      onDrag: ({ delta: [dx, dy], velocity: [vx, vy], direction: [dirX, dirY], event, down }) => {
        // While the Pen is active only a middle-button or Space drag may pan; a plain left drag is
        // the tool's, and must not also move the compare divider underneath it.
        if (rotoActive && !(event.button === 1 || event.buttons === 4 || isSpaceDownRef.current)) return;
        if (maskBrushActive && down) {
          drawMaskStroke(event, false);
          return;
        }
        // Pan condition: Spacebar + Left Drag OR Middle-Mouse Drag OR Left Drag when Zoomed
        const isMiddleMouse = event.button === 1 || event.buttons === 4;
        const isSpaceDrag = isSpaceDownRef.current && event.buttons === 1;
        const isZoomPan = zoomRef.current > 1.05 && event.buttons === 1 && !compare;

        if (isMiddleMouse || isSpaceDrag || isZoomPan) {
          event.preventDefault();
          isInteractingRef.current = true;
          setIsPanning(true);
          setLockTargetFace(false);

          const s = uiScale(containerRef.current);
          const nextPan = clampPan({
            x: panRef.current.x + dx / s,
            y: panRef.current.y + dy / s,
          }, zoomRef.current, containerRef.current, imageRef.current);
          updateViewport(zoomRef.current, nextPan, false);

          if (!down) {
            setIsPanning(false);
            isInteractingRef.current = false;
            const speed = Math.hypot(vx, vy);
            if (speed > 0.4) {
              startInertia(dirX * vx * 16, dirY * vy * 16);
            } else {
              updateViewport(zoomRef.current, panRef.current, true);
            }
          }
          return;
        }

        // Compare Slider dragging
        if (compare && compareMode === 'slider' && event.buttons === 1 && !splitMode) {
          handleSliderMove(event.clientX, event.clientY);
        }
      },

      onDragEnd: () => {
        setIsDrawingMask(false);
        setIsPanning(false);
        lastPtRef.current = null;
        if (compare) commitSliderPosition(sliderPosRef.current);
      },

      onMove: ({ event }) => {
        const cx = event.clientX;
        const cy = event.clientY;

        // Update Dual Concentric Mask Brush Cursor
        if (maskBrushActive && brushCursorRef.current && containerRef.current) {
          const rect = containerRef.current.getBoundingClientRect();
          const s = uiScale(containerRef.current);
          const x = (cx - rect.left) / s;
          const y = (cy - rect.top) / s;
          brushCursorRef.current.style.transform = `translate3d(${x}px, ${y}px, 0)`;
        }

        // Update Magnifier Loupe position
        if (magnifierActive && containerRef.current) {
          const rect = containerRef.current.getBoundingClientRect();
          const ir = imageRef.current?.getBoundingClientRect();
          const s = uiScale(containerRef.current);
          lensPendingRef.current = {
            x: (cx - rect.left) / s,
            y: (cy - rect.top) / s,
            imgW: (ir?.width || rect.width) / s,
            imgH: (ir?.height || rect.height) / s,
            imgX: ((ir?.left ?? rect.left) - rect.left) / s,
            imgY: ((ir?.top ?? rect.top) - rect.top) / s,
            visible: cx >= rect.left && cx <= rect.right && cy >= rect.top && cy <= rect.bottom,
          };
          if (!lensRafRef.current) {
            lensRafRef.current = requestAnimationFrame(() => {
              lensRafRef.current = null;
              const p = lensPendingRef.current;
              const glass = lensGlassRef.current;
              const bImg = lensBeforeImgRef.current;
              const aImg = lensAfterImgRef.current;
              if (!p || !glass) return;
              glass.style.display = p.visible ? '' : 'none';
              if (!p.visible) return;

              const LENS_R = 90;
              glass.style.transform = `translate3d(${p.x - LENS_R}px, ${p.y - LENS_R}px, 0)`;
              const box = `${p.imgW}x${p.imgH}`;
              if (box !== lensBoxRef.current) {
                lensBoxRef.current = box;
                if (bImg) {
                  bImg.style.width = `${p.imgW * loupePower}px`;
                  bImg.style.height = `${p.imgH * loupePower}px`;
                }
                if (aImg) {
                  aImg.style.width = `${p.imgW * loupePower}px`;
                  aImg.style.height = `${p.imgH * loupePower}px`;
                }
              }
              const tx = -((p.x - p.imgX) * loupePower - LENS_R);
              const ty = -((p.y - p.imgY) * loupePower - LENS_R);
              if (bImg) bImg.style.transform = `translate3d(${tx}px, ${ty}px, 0)`;
              if (aImg) aImg.style.transform = `translate3d(${tx}px, ${ty}px, 0)`;
            });
          }
        }
      },
    },
    {
      target: containerRef,
      eventOptions: { passive: false },
    }
  );

  // ── Keyboard Shortcuts (Coexisting & clash-free) ──────────────────────────
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
      if (document.querySelector('[role="dialog"]')) return;

      if (e.code === 'Space') {
        isSpaceDownRef.current = true;
      }

      // Hold Backslash (`\`) for instant Original peek
      if (e.key === '\\') {
        setIsPeekingOriginal(true);
        return;
      }

      // Shift + [ and Shift + ] for brush feather falloff
      if (e.shiftKey && maskBrushActive) {
        if (e.code === 'BracketLeft') {
          e.preventDefault();
          setBrushFeather((f) => Math.max(0, parseFloat((f - 0.1).toFixed(2))));
          return;
        }
        if (e.code === 'BracketRight') {
          e.preventDefault();
          setBrushFeather((f) => Math.min(1, parseFloat((f + 0.1).toFixed(2))));
          return;
        }
      }

      // Bare keys only from here
      if (e.ctrlKey || e.metaKey || e.altKey) return;

      if (e.key === '=' || e.key === '+') {
        e.preventDefault();
        zoomBy(1.4);
      } else if (e.key === '-' || e.key === '_') {
        e.preventDefault();
        zoomBy(1 / 1.4);
      } else if (e.key.toLowerCase() === 'f') {
        // Fit Preset (F)
        e.preventDefault();
        handlePresetFit();
      } else if (e.key === '1') {
        // 1:1 Pixel Preset (1)
        e.preventDefault();
        handlePreset1to1();
      } else if (e.key === '2') {
        // 200% Preset (2)
        e.preventDefault();
        handlePreset200();
      } else if (e.key === '4') {
        // 400% Preset (4)
        e.preventDefault();
        handlePreset400();
      } else if (e.key.toLowerCase() === 't') {
        // Lock to Target Face (T)
        e.preventDefault();
        setLockTargetFace((v) => !v);
      } else if (e.key.toLowerCase() === 'd') {
        // Debug Landmarks/Pose (D)
        setShowDebug((v) => !v);
      } else if ((e.key.toLowerCase() === 'g' || e.code === 'KeyM') && !splitMode) {
        // Loupe / Glass (G or M)
        setMagnifierActive((m) => !m);
      } else if (e.key.toLowerCase() === 'b' && !splitMode) {
        // Mask Brush (B)
        toggleBrush();
      } else if (e.key.toLowerCase() === 'p' && !splitMode) {
        // Pen tool: vector exclusion / inclusion zones (P)
        e.preventDefault();
        toggleRoto();
      } else if (e.key.toLowerCase() === 'x' && compare && !splitMode) {
        setCompareDir((d) => (d === 'vertical' ? 'horizontal' : 'vertical'));
      } else if (e.key.toLowerCase() === 'w' && compare && !splitMode) {
        setCompareMode((m) => (m === 'blend' ? 'slider' : 'blend'));
      } else if (e.key.toLowerCase() === 'a' && compare && !splitMode && compareMode !== 'diff') {
        setAutoSwipe((a) => !a);
      }
    };

    const handleKeyUp = (e) => {
      if (e.code === 'Space') {
        isSpaceDownRef.current = false;
      }
      if (e.key === '\\') {
        setIsPeekingOriginal(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [
    compare, compareMode, handlePreset1to1, handlePreset200, handlePreset400,
    handlePresetFit, maskBrushActive, splitMode, zoomBy, toggleBrush, toggleRoto,
  ]);

  useEffect(() => {
    const onFsChange = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', onFsChange);
    return () => document.removeEventListener('fullscreenchange', onFsChange);
  }, []);

  const triggerFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {});
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {});
    }
  };

  const stepFrame = (delta) => setFrame && setFrame((f) => Math.max(1, Math.min(maxFrames, f + delta)));
  const isVideo = maxFrames > 1;

  // ── Face Bounding Boxes Overlay ───────────────────────────────────────────
  const faceBoxes = useMemo(() => {
    if (!faces.length || !imgDim || !showBoxes) return null;
    const selectable = typeof onSelectFace === 'function';
    return faces.map((bbox, i) => {
      const [sx, sy, ex, ey] = bbox;
      const left = (sx / imgDim.w) * 100;
      const top = (sy / imgDim.h) * 100;
      const width = ((ex - sx) / imgDim.w) * 100;
      const height = ((ey - sy) / imgDim.h) * 100;
      const personLabel = (personIds[i] ?? i) + 1;
      const selected = selectedFaceIndex === i;
      const info = faceRoles[i] || null;
      const roleLabel = !info ? null
        : info.role === 'target' ? 'Target'
          : info.role === 'swapped' ? 'Swapped'
            : (info.excluded ? 'Excluded · never swapped' : 'Bystander · not swapped');
      const roleClass = !info ? ''
        : info.role === 'target' ? 'bg-emerald-600'
          : info.role === 'swapped' ? 'bg-[var(--accent)]'
            : 'bg-zinc-700';
      const togglable = typeof onToggleExclude === 'function' && info && info.canExclude;
      // Stage 11: pin this person's identity to the track passing through this box on this frame.
      const pinSource = Number.isInteger(faceSources[i]) && faceSources[i] >= 0 ? faceSources[i] : null;
      const pinned = pinSource !== null
        ? (pins.find((p) => p.source_index === pinSource && p.frame === frame) || null) : null;
      const pinnable = typeof onPinIdentity === 'function' && pinSource !== null;
      return (
        <div
          key={i}
          className={`absolute border-2 ${selected ? 'border-white bg-white/15 ring-2 ring-[var(--accent)]' : 'border-[var(--accent)]'} shadow-[0_0_10px_var(--accent-glow)] z-20 ${
            selectable
              ? 'pointer-events-auto cursor-pointer group/face hover:bg-[var(--accent)]/10 transition-colors'
              : 'pointer-events-none'
          }`}
          style={{ left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}
          data-face-index={i}
          data-face-role={info ? info.role : undefined}
          title={selectable ? `Click to use Face ${i + 1} (Person ${personLabel}) as target` : undefined}
          onPointerDown={selectable ? (e) => e.stopPropagation() : undefined}
          onClick={selectable ? (e) => { e.stopPropagation(); onSelectFace(i); } : undefined}
        >
          <span className="absolute -top-6 left-0 bg-[var(--accent)] text-white text-micro font-bold px-1.5 py-0.5 rounded whitespace-nowrap">
            Face {i + 1}
          </span>
          {roleLabel && (
            <span
              className={`absolute -top-6 left-14 ${roleClass} text-white text-micro font-bold px-1.5 py-0.5 rounded whitespace-nowrap pointer-events-none`}
            >
              {roleLabel}
            </span>
          )}
          {pinnable && (
            <button
              type="button"
              data-testid="pin-identity"
              aria-pressed={!!pinned}
              className={`absolute left-1 bottom-1 ${pinned ? 'bg-sky-500 text-black' : 'bg-black/80 text-white hover:bg-sky-600'} backdrop-blur text-nano font-bold px-1.5 py-0.5 rounded whitespace-nowrap pointer-events-auto cursor-pointer`}
              title={pinned
                ? 'Remove the pin: the renderer goes back to deciding who this is automatically'
                : `Tell the renderer this person is ${trackLabel(personIds[i] ?? i)} on this frame. The tracker's track through this box is then locked to this person's source and is never re-assigned.`}
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => { e.stopPropagation(); if (pinned) onUnpinIdentity?.(pinned.id); else onPinIdentity(i); }}
            >
              {pinned ? `📌 Pinned · ${trackLabel(personIds[i] ?? i)}` : `Pin Identity to ${trackLabel(personIds[i] ?? i)}`}
            </button>
          )}
          {togglable && (
            <button
              type="button"
              className="absolute left-1 -bottom-6 bg-black/80 backdrop-blur text-white text-nano font-bold px-1.5 py-0.5 rounded whitespace-nowrap pointer-events-auto cursor-pointer hover:bg-[var(--accent)]"
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => { e.stopPropagation(); onToggleExclude(i); }}
            >
              {info.excluded ? 'Include again' : 'Exclude (never swap)'}
            </button>
          )}
        </div>
      );
    });
  }, [faces, imgDim, showBoxes, personIds, selectedFaceIndex, onSelectFace, faceRoles, onToggleExclude,
    faceSources, pins, frame, onPinIdentity, onUnpinIdentity]);

  // ── Landmark Telemetry Debug Overlay ──────────────────────────────────────
  const debugOverlay = useMemo(() => {
    if (!showDebug || !imgDim || !kps.length) return null;
    const NAMES = ['eyeL', 'eyeR', 'nose', 'mouthL', 'mouthR'];
    const COLORS = ['#38bdf8', '#38bdf8', '#fbbf24', '#f472b6', '#f472b6'];
    const r = Math.max(imgDim.w, imgDim.h) / 320;
    return (
      <svg
        className="absolute inset-0 w-full h-full z-20 pointer-events-none"
        viewBox={`0 0 ${imgDim.w} ${imgDim.h}`}
        preserveAspectRatio="none"
        aria-hidden
      >
        {kps.map((k, i) => {
          if (!k || k.length < 5) return null;
          const [eyeL, eyeR, nose, mL, mR] = k;
          const p = pose[i];
          return (
            <g key={i}>
              <line x1={eyeL[0]} y1={eyeL[1]} x2={eyeR[0]} y2={eyeR[1]} stroke="#38bdf8" strokeWidth={r * 0.5} opacity="0.75" />
              <line x1={mL[0]} y1={mL[1]} x2={mR[0]} y2={mR[1]} stroke="#f472b6" strokeWidth={r * 0.5} opacity="0.75" />
              <line
                x1={(eyeL[0] + eyeR[0]) / 2} y1={(eyeL[1] + eyeR[1]) / 2}
                x2={(mL[0] + mR[0]) / 2} y2={(mL[1] + mR[1]) / 2}
                stroke="#a3e635" strokeWidth={r * 0.4} opacity="0.6" strokeDasharray={`${r * 2} ${r * 1.5}`}
              />
              {k.map(([x, y], j) => (
                <circle key={j} cx={x} cy={y} r={r} fill={COLORS[j]} stroke="#000" strokeWidth={r * 0.3} opacity="0.95">
                  <title>{NAMES[j]}</title>
                </circle>
              ))}
              {p && (
                <text
                  x={nose[0]} y={nose[1] - r * 4}
                  fill="#fff" fontSize={r * 5} fontWeight="700" textAnchor="middle" stroke="#000" strokeWidth={r * 1.2}
                >
                  {`y${p[0]}° p${p[1]}° r${p[2]}°`}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    );
  }, [showDebug, imgDim, kps, pose]);

  const displayRatio = useMemo(() => {
    if (!imgDim) return null;
    if (darVal && Number(darVal) > 0) return Number(darVal);
    const s = Number(sarVal) || 1.0;
    if (Math.abs(s - 1.0) > 0.005) return (imgDim.w * s) / imgDim.h;
    return imgDim.w / imgDim.h;
  }, [imgDim, sarVal, darVal]);

  const aspectStyle = {
    aspectRatio: displayRatio ? `${displayRatio}` : (imgDim ? `${imgDim.w}/${imgDim.h}` : '1'),
    maxHeight: '100%',
    maxWidth: '100%',
    display: 'flex',
  };

  // ── Bottom-Left Viewport HUD Floating Pill ────────────────────────────────
  const viewportPill = () => (
    <div className="absolute bottom-3 left-3 z-50 flex items-center gap-1.5 px-2 py-1 rounded-xl bg-black/85 backdrop-blur-md border border-white/20 shadow-2xl pointer-events-auto select-none">
      <span
        ref={zoomReadoutRef}
        className="font-mono text-micro font-bold tabular-nums text-accent px-1.5 py-0.5 rounded bg-[var(--accent)]/15 border border-[var(--accent)]/30"
      >
        {Math.round(zoom * 100)}%
      </span>

      <span className="w-px h-3.5 bg-white/15" />

      <button
        type="button"
        onClick={handlePresetFit}
        className="px-1.5 py-0.5 rounded text-nano font-mono font-bold bg-white/10 hover:bg-white/20 text-white/80 hover:text-white transition-colors"
        title="Fit whole frame to view (F)"
      >
        Fit (F)
      </button>

      <button
        type="button"
        onClick={handlePresetFill}
        className="px-1.5 py-0.5 rounded text-nano font-mono font-bold bg-white/10 hover:bg-white/20 text-white/80 hover:text-white transition-colors"
        title="Fill stage box"
      >
        Fill
      </button>

      <button
        type="button"
        onClick={handlePreset1to1}
        className="px-1.5 py-0.5 rounded text-nano font-mono font-bold bg-white/10 hover:bg-white/20 text-white/80 hover:text-white transition-colors"
        title="1:1 Native pixel resolution (1)"
        aria-label="1:1 Native pixel resolution (1)"
      >
        1:1 (1)
      </button>

      <button
        type="button"
        onClick={handlePreset200}
        className="px-1.5 py-0.5 rounded text-nano font-mono font-bold bg-white/10 hover:bg-white/20 text-white/80 hover:text-white transition-colors"
        title="200% Zoom (2)"
        aria-label="200% Zoom (2)"
      >
        200% (2)
      </button>

      <button
        type="button"
        onClick={handlePreset400}
        className="px-1.5 py-0.5 rounded text-nano font-mono font-bold bg-white/10 hover:bg-white/20 text-white/80 hover:text-white transition-colors"
        title="400% Zoom (4)"
        aria-label="400% Zoom (4)"
      >
        400% (4)
      </button>

      <span className="w-px h-3.5 bg-white/15" />

      {/* Lock Target Face Auto-Follow Toggle */}
      <button
        type="button"
        onClick={() => setLockTargetFace((v) => !v)}
        className={`px-2 py-0.5 rounded text-nano font-bold flex items-center gap-1 transition-all ${
          lockTargetFace
            ? 'bg-emerald-500 text-black font-extrabold shadow-[0_0_8px_rgba(16,185,129,0.5)]'
            : 'bg-white/10 hover:bg-white/20 text-white/80 hover:text-white'
        }`}
        title="Lock to Target Face Auto-Follow camera (T)"
        aria-label="Lock to Target Face Auto-Follow camera (T)"
      >
        <span>🎯 Lock Face (L)</span>
      </button>
    </div>
  );

  // ── Main Preview Box Toolbar (HUD) ─────────────────────────────────────────
  const hudBar = () => (
    <div className="absolute inset-x-0 bottom-0 z-50 flex justify-center p-3 pointer-events-none">
      <div className="spring-cluster pointer-events-auto flex flex-wrap justify-center max-w-full items-center gap-0.5 rounded-xl hud-glass p-1 opacity-90 translate-y-0.5 group-hover:opacity-100 group-hover:translate-y-0 focus-within:opacity-100 focus-within:translate-y-0 transition-all duration-300 shadow-2xl border border-white/10">
        <button
          type="button"
          onClick={() => zoomBy(1 / 1.4)}
          disabled={zoom <= ZOOM_MIN}
          className="grid place-items-center h-7 w-7 rounded-lg hud-glass-button text-sm font-bold disabled:opacity-40"
          title="Zoom out (−)"
          aria-label="Zoom out (−)"
        >
          −
        </button>

        <button
          type="button"
          onClick={handlePresetFit}
          className={`px-2 h-7 rounded-lg text-micro font-bold font-mono tabular-nums ${
            zoom > 1.05 ? 'text-accent bg-[var(--accent)]/10 border border-[var(--accent)]/20' : 'hud-glass-button'
          }`}
          title="Reset to Fit (F)"
        >
          {zoom > 1.05 ? `${zoom.toFixed(1)}×` : 'FIT'}
        </button>

        <button
          type="button"
          onClick={() => zoomBy(1.4)}
          disabled={zoom >= ZOOM_MAX}
          className="grid place-items-center h-7 w-7 rounded-lg hud-glass-button text-sm font-bold disabled:opacity-40"
          title="Zoom in (+)"
          aria-label="Zoom in (+)"
        >
          +
        </button>

        <span className="w-px h-5 bg-white/10 mx-1" />

        {/* Faces Overlays Toggle */}
        <button
          type="button"
          onClick={() => setShowBoxes((b) => !b)}
          title="Show detected face boxes"
          aria-pressed={showBoxes}
          className={`px-2 h-7 rounded-lg text-micro font-bold ${
            showBoxes ? 'text-accent bg-[var(--accent)]/10 border border-[var(--accent)]/20' : 'hud-glass-button'
          }`}
        >
          Faces
          {faces.length > 0 && <span className="ml-1 font-normal tabular-nums">{faces.length}</span>}
        </button>

        {/* Landmark / Pose Debug Overlay */}
        {kps.length > 0 && (
          <button
            type="button"
            onClick={() => setShowDebug((v) => !v)}
            title="Show keypoints and head pose (D)"
            aria-pressed={showDebug}
            className={`px-2 h-7 rounded-lg text-micro font-bold ${
              showDebug ? 'text-accent bg-[var(--accent)]/10 border border-[var(--accent)]/20' : 'hud-glass-button'
            }`}
          >
            Debug
          </button>
        )}

        {/* Compare Faces Toggle Button */}
        {onToggleCompare && (
          <button
            type="button"
            onClick={() => onToggleCompare()}
            title="Toggle Compare Modes (C)"
            className={`px-2.5 h-7 rounded-lg text-micro font-bold flex items-center gap-1.5 transition-all duration-200 ${
              compare
                ? 'text-white bg-[var(--accent)] shadow-[0_0_12px_var(--accent-glow)] border border-white/20'
                : 'hud-glass-button text-white/80 hover:text-white'
            }`}
          >
            <span>Compare Faces</span>
            {compare && <span className="text-nano bg-black/40 px-1 rounded font-mono">{Math.round(sliderPosition)}%</span>}
          </button>
        )}

        {/* Hold Peek Original Button / Shortcut `\\` */}
        <button
          type="button"
          onMouseDown={() => setIsPeekingOriginal(true)}
          onMouseUp={() => setIsPeekingOriginal(false)}
          onMouseLeave={() => setIsPeekingOriginal(false)}
          onTouchStart={() => setIsPeekingOriginal(true)}
          onTouchEnd={() => setIsPeekingOriginal(false)}
          title="Hold to view original target image (or hold \\)"
          className={`px-2 h-7 rounded-lg text-micro font-bold transition-all duration-200 select-none ${
            isPeekingOriginal
              ? 'bg-amber-500 text-black border border-amber-300 font-extrabold shadow-[0_0_10px_rgba(245,158,11,0.5)]'
              : 'hud-glass-button text-white/70'
          }`}
        >
          Hold Peek (\)
        </button>

        <span className="w-px h-5 bg-white/10 mx-1" />

        {/* Floating Magnifier Loupe Button */}
        {!splitMode && (
          <div className="flex items-center gap-0.5">
            <button
              type="button"
              onClick={() => setMagnifierActive((m) => !m)}
              title="Toggle 3.5×–8× Floating Magnifier Loupe (G or M)"
              className={`px-2 h-7 rounded-l-lg text-micro font-bold flex items-center gap-1 transition-colors ${
                magnifierActive
                  ? 'text-cyan-300 bg-cyan-500/20 border border-cyan-400/40 shadow-[0_0_10px_rgba(6,182,212,0.3)]'
                  : 'hud-glass-button text-white/70'
              }`}
            >
              Loupe ({loupePower}×)
            </button>
            {magnifierActive && (
              <button
                type="button"
                onClick={() => setLoupePower((p) => (p === 3.5 ? 5.0 : p === 5.0 ? 8.0 : 3.5))}
                className="px-1.5 h-7 rounded-r-lg text-nano font-mono font-bold bg-cyan-500/30 text-cyan-200 border-y border-r border-cyan-400/40"
                title="Cycle Loupe Magnification: 3.5x, 5.0x, 8.0x"
              >
                {loupePower}×
              </button>
            )}
          </div>
        )}

        {/* Face Mask Brush Tool Button */}
        {!splitMode && (
          <button
            type="button"
            onClick={toggleBrush}
            title="Toggle sub-pixel Mask Brush (B)"
            className={`px-2 h-7 rounded-lg text-micro font-bold flex items-center gap-1 transition-colors ${
              maskBrushActive
                ? 'text-pink-300 bg-pink-500/20 border border-pink-400/40 shadow-[0_0_10px_rgba(236,72,153,0.3)]'
                : 'hud-glass-button text-white/70'
            }`}
          >
            Mask Brush
          </button>
        )}

        {/* Pen tool: vector Bezier / polygon exclusion & inclusion zones */}
        {!splitMode && onRotoZones && (
          <button
            type="button"
            onClick={toggleRoto}
            aria-pressed={rotoActive}
            data-testid="roto-toggle"
            title="Pen tool (P): draw a Bezier / polygon zone around a microphone, hand, glasses rim or watermark to keep it, or to force the swap inside it"
            className={`px-2 h-7 rounded-lg text-micro font-bold flex items-center gap-1 transition-colors ${
              rotoActive
                ? 'text-rose-300 bg-rose-500/20 border border-rose-400/40 shadow-[0_0_10px_rgba(251,113,133,0.3)]'
                : 'hud-glass-button text-white/70'
            }`}
          >
            Pen{rotoZones.length ? ` (${rotoZones.length})` : ''}
          </button>
        )}

        {/* Broadcast Scopes Dock Toggle Button */}
        <button
          type="button"
          onClick={() => setScopeActive((s) => !s)}
          aria-pressed={scopeActive}
          title="Toggle Broadcast Scopes Dock (Waveform, RGB Parade, Vectorscope with I-Line)"
          className={`px-2 h-7 rounded-lg text-micro font-bold flex items-center gap-1 transition-colors ${
            scopeActive
              ? 'text-amber-300 bg-amber-500/20 border border-amber-400/40 shadow-[0_0_10px_rgba(245,158,11,0.3)]'
              : 'hud-glass-button text-white/70'
          }`}
        >
          Scopes
        </button>

        {/* 3D LUT Color Grade Toggle Button */}
        <button
          type="button"
          onClick={() => setGradingActive((g) => !g)}
          aria-pressed={gradingActive}
          title="Toggle WebGL2 3D .cube LUT Grading Panel"
          className={`px-2 h-7 rounded-lg text-micro font-bold flex items-center gap-1 transition-colors ${
            gradingActive
              ? 'text-indigo-300 bg-indigo-500/20 border border-indigo-400/40 shadow-[0_0_10px_rgba(99,102,241,0.3)]'
              : 'hud-glass-button text-white/70'
          }`}
        >
          Grade
        </button>

        {/* False-Color / Zebra Exposure Toggle Button */}
        <button
          type="button"
          onClick={() => setZebraActive((z) => !z)}
          aria-pressed={zebraActive}
          title="Toggle False-Color / Zebra Exposure overlay (>95 IRE clipped highlights, <5 IRE crushed shadows on face ROI)"
          className={`px-2 h-7 rounded-lg text-micro font-bold flex items-center gap-1 transition-colors ${
            zebraActive
              ? 'text-rose-300 bg-rose-500/20 border border-rose-400/40 shadow-[0_0_10px_rgba(244,63,94,0.3)]'
              : 'hud-glass-button text-white/70'
          }`}
        >
          Zebra
        </button>

                {/* Stage 21 3D / VR Equirectangular Viewport Button */}
        <button
          type="button"
          onClick={() => setVrActive((v) => !v)}
          aria-pressed={vrActive}
          title="Toggle WebGL2 3D / Equirectangular VR180/360 Viewport & Stereoscopic Inspector"
          className={`px-2 h-7 rounded-lg text-micro font-bold flex items-center gap-1 transition-colors ${
            vrActive
              ? 'text-teal-300 bg-teal-500/20 border border-teal-400/40 shadow-[0_0_10px_rgba(20,184,166,0.3)]'
              : 'hud-glass-button text-white/70'
          }`}
        >
          3D / VR
        </button>

        {/* Pop-Out Clean Feed Reference Monitor Button */}
        <button
          type="button"
          onClick={openMonitor}
          aria-pressed={monitorActive}
          title="Pop-Out Clean Feed Monitor (Open clean reference feed on second display or OBS Window Capture)"
          className={`px-2 h-7 rounded-lg text-micro font-bold flex items-center gap-1 transition-colors ${
            monitorActive
              ? 'text-cyan-300 bg-cyan-500/20 border border-cyan-400/40 shadow-[0_0_10px_rgba(6,182,212,0.3)]'
              : 'hud-glass-button text-white/70'
          }`}
        >
          Monitor ↗
        </button>

        {/* Video Frame controls */}
        {isVideo && (
          <>
            <span className="w-px h-5 bg-white/10 mx-1" />
            <button
              type="button"
              onClick={() => stepFrame(-1)}
              className="grid place-items-center h-7 w-7 rounded-lg hud-glass-button"
              title="Previous frame (←)"
              aria-label="Previous frame"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M6 6h2v12H6zM20 6v12l-9-6z"/></svg>
            </button>
            <button
              type="button"
              onClick={() => setIsPlaying && setIsPlaying((p) => !p)}
              className={`grid place-items-center h-7 w-7 rounded-lg ${
                isPlaying ? 'text-accent bg-[var(--accent)]/10 border border-[var(--accent)]/20' : 'hud-glass-button'
              }`}
              title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
              aria-label={isPlaying ? 'Pause' : 'Play'}
            >
              {isPlaying ? (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>
              ) : (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
              )}
            </button>
            <button
              type="button"
              onClick={() => stepFrame(1)}
              className="grid place-items-center h-7 w-7 rounded-lg hud-glass-button"
              title="Next frame (→)"
              aria-label="Next frame"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M16 6h2v12h-2zM4 6l9 6-9 6z"/></svg>
            </button>
            <span className="px-1 text-micro font-bold font-mono text-muted tabular-nums whitespace-nowrap">
              {frame.toLocaleString()}<span className="font-normal">/{maxFrames.toLocaleString()}</span>
            </span>
          </>
        )}

        <span className="w-px h-5 bg-white/10 mx-1" />

        {/* Fullscreen Button */}
        <button
          type="button"
          onClick={triggerFullscreen}
          className="grid place-items-center h-7 w-7 rounded-lg hud-glass-button"
          title="Toggle fullscreen"
          aria-label="Toggle fullscreen"
        >
          {isFullscreen ? (
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"/></svg>
          ) : (
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>
          )}
        </button>
      </div>
    </div>
  );

  // ── Top Information Badges ────────────────────────────────────────────────
  const stageInfo = () => (
    <div className="absolute top-3 left-3 z-40 flex items-center gap-2 pointer-events-none">
      {imgDim && (
        <span className="rounded-lg bg-black/75 backdrop-blur-md px-2.5 py-1 font-mono text-micro tabular-nums text-white/80 border border-white/10 shadow-lg">
          {imgDim.w}×{imgDim.h}
        </span>
      )}
      {colorBadge && (
        <span className="rounded-lg bg-indigo-500/20 text-indigo-300 backdrop-blur-md px-2.5 py-1 text-micro font-bold border border-indigo-400/30 shadow-lg">
          {colorBadge}
        </span>
      )}
      {zebraActive && (
        <span className="rounded-lg bg-rose-500/20 text-rose-300 backdrop-blur-md px-2.5 py-1 text-micro font-bold border border-rose-400/30 shadow-lg animate-pulse">
          Zebra (&gt;95 / &lt;5 IRE)
        </span>
      )}
      {compare && (
        <span className="rounded-lg bg-[var(--accent)] text-white backdrop-blur-md px-2.5 py-1 text-micro font-bold uppercase tracking-wider shadow-lg border border-white/20">
          Compare ({compareMode.toUpperCase()})
        </span>
      )}
      {lockTargetFace && (
        <span className="rounded-lg bg-emerald-600/90 text-white backdrop-blur-md px-2 py-1 text-nano font-bold uppercase tracking-wider shadow-lg border border-emerald-400/30 animate-pulse">
          🎯 Face Locked
        </span>
      )}
      {magnifierActive && !splitMode && (
        <span className="rounded-lg bg-cyan-500/20 text-cyan-300 backdrop-blur-md px-2 py-1 text-nano font-bold uppercase tracking-wider border border-cyan-400/30">
          Loupe ({loupePower}×)
        </span>
      )}
      {maskBrushActive && !splitMode && (
        <span className="rounded-lg bg-pink-500/20 text-pink-300 backdrop-blur-md px-2 py-1 text-nano font-bold uppercase tracking-wider border border-pink-400/30">
          Brush ({brushMode} · {brushSize}px · {Math.round(brushFeather * 100)}% feather)
        </span>
      )}
      {rotoActive && !splitMode && (
        <span className="rounded-lg bg-rose-500/20 text-rose-300 backdrop-blur-md px-2 py-1 text-nano font-bold uppercase tracking-wider border border-rose-400/30">
          Pen ({rotoNewMode})
        </span>
      )}
    </div>
  );

  // ── 4-Up Quad Inspection Grid Rendering ──────────────────────────────────
  if (compare && compareMode === 'quad') {
    return (
      <div
        ref={containerRef}
        className={`relative w-full max-h-[65vh] min-h-[260px] rounded-2xl overflow-hidden preview-stage group border border-white/10 shadow-2xl ${
          isFullscreen ? 'h-screen w-screen max-h-none' : ''
        } aspect-video`}
        onDoubleClick={handleDoubleClick}
        style={{ touchAction: 'none' }}
      >
        {stageInfo()}

        {/* 4-Up Quad View Component */}
        <QuadInspectionGrid
          beforeSrc={beforeSrc}
          afterSrc={afterSrc}
          beforeFrame={beforeFrame}
          afterFrame={afterFrame}
          faces={faces}
          kps={kps}
          pose={pose}
          selectedFaceIndex={selectedFaceIndex}
          zoom={zoom}
          pan={pan}
          imgDim={imgDim}
          playbackSource={playbackSource}
          faceBoxes={faceBoxes}
          debugOverlay={debugOverlay}
        />

        {/* Mini Radar Navigator (PiP) */}
        <RadarNavigator
          zoom={zoom}
          pan={pan}
          onPanChange={(p) => updateViewport(zoom, p, false)}
          imgDim={imgDim}
          thumbnailSrc={afterSrc || beforeSrc}
          stageRef={containerRef}
          contentRef={imageRef}
        />

        {/* Compare Mode Toolbar */}
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-black/85 backdrop-blur-md border border-white/15 shadow-2xl">
          {['slider', 'blend', 'quad', 'heatmap', 'diff'].map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setCompareMode(m)}
              className={`px-2.5 py-1 rounded-lg text-micro font-bold uppercase transition-all ${
                compareMode === m ? 'bg-[var(--accent)] text-white shadow-md' : 'bg-white/10 text-white/70 hover:text-white'
              }`}
            >
              {m}
            </button>
          ))}
        </div>

        {viewportPill()}
        {hudBar()}
      </div>
    );
  }

  // ── Standard & Comparison Stages ──────────────────────────────────────────
  const canvasMode = compare && !isPeekingOriginal ? (compareMode === 'heatmap' ? 'heatmap' : compareMode) : 'normal';
  const canvasWipe = (() => {
    if (isPeekingOriginal) return 0;
    if (!compare) return 100;
    if (compareMode === 'blend') return sliderPosition;
    if (compareMode === 'diff' || compareMode === 'heatmap') return 100;
    return 100 - sliderPosition;
  })();

  return (
    <div
      ref={containerRef}
      className={`relative w-full max-h-[65vh] min-h-[260px] rounded-2xl overflow-hidden preview-stage contain-panel select-none group border border-white/10 shadow-2xl ${
        isFullscreen ? 'h-screen w-screen max-h-none' : ''
      } ${displayRatio ? '' : 'aspect-video'}`}
      onDoubleClick={handleDoubleClick}
      style={{
        touchAction: 'none',
        ...(displayRatio && !isFullscreen ? {
          aspectRatio: `${displayRatio}`,
          maxWidth: displayRatio < 1.0 ? `min(100%, calc(65vh * ${displayRatio}))` : '100%',
          margin: '0 auto',
        } : {})
      }}
    >
      {/* Render Progress Strip */}
      {previewing && (
        <div className="absolute inset-x-0 top-0 h-[3px] z-50 overflow-hidden bg-white/[0.06]">
          <div className="h-full w-1/3 rounded-full bg-[var(--accent)] preview-indeterminate" />
        </div>
      )}
      {previewing && <AIScannerOverlay />}
      {previewing && (
        <div className="absolute top-4 right-3 z-50">
          <div className="inline-flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-black/80 backdrop-blur-md text-mini font-semibold text-white/90 tabular-nums border border-white/10 shadow-2xl">
            <span className="h-3 w-3 rounded-full border-2 border-white/25 border-t-[var(--accent)] animate-spin" />
            Rendering {previewSecs}s
          </div>
        </div>
      )}

      {frameUnavailable && (
        <FrameUnavailable onRetry={onRetryFrame} className="absolute inset-x-0 bottom-4 z-50" />
      )}

      {stageInfo()}

      {/* ── 5 Instant Comparison Toolbar Modes ─────────────────────────────── */}
      {compare && !splitMode && !maskBrushActive && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-black/85 backdrop-blur-md border border-white/15 shadow-2xl animate-fade-in">
          <button
            type="button"
            onClick={() => setCompareMode('slider')}
            className={`px-2.5 py-1 rounded-lg text-micro font-bold transition-all ${
              compareMode === 'slider' ? 'bg-[var(--accent)] text-white shadow-md' : 'bg-white/10 text-white/70 hover:text-white'
            }`}
            title="360° Vector Split Wipe"
          >
            Split 360°
          </button>
          <button
            type="button"
            onClick={() => setCompareMode('blend')}
            className={`px-2.5 py-1 rounded-lg text-micro font-bold transition-all ${
              compareMode === 'blend' ? 'bg-[var(--accent)] text-white shadow-md' : 'bg-white/10 text-white/70 hover:text-white'
            }`}
            title="Opacity Blend mode (W)"
          >
            Blend
          </button>
          <button
            type="button"
            onClick={() => setCompareMode('quad')}
            className={`px-2.5 py-1 rounded-lg text-micro font-bold transition-all ${
              compareMode === 'quad' ? 'bg-[var(--accent)] text-white shadow-md' : 'bg-white/10 text-white/70 hover:text-white'
            }`}
            title="4-Up Quad Inspection Grid"
          >
            4-Up Quad
          </button>
          <button
            type="button"
            onClick={() => setCompareMode('heatmap')}
            className={`px-2.5 py-1 rounded-lg text-micro font-bold transition-all ${
              compareMode === 'heatmap' ? 'bg-[var(--accent)] text-white shadow-md' : 'bg-white/10 text-white/70 hover:text-white'
            }`}
            title="Amplified Turbo Difference Heatmap"
          >
            Heatmap
          </button>
          <button
            type="button"
            onClick={() => setCompareMode('diff')}
            className={`px-2.5 py-1 rounded-lg text-micro font-bold transition-all ${
              compareMode === 'diff' ? 'bg-[var(--accent)] text-white shadow-md' : 'bg-white/10 text-white/70 hover:text-white'
            }`}
            title="Raw Difference Map"
          >
            Diff
          </button>

          {/* Mode-Specific Sub-Controls */}
          {compareMode === 'slider' && (
            <>
              <span className="w-px h-4 bg-white/15 mx-0.5" />
              <button
                type="button"
                onClick={() => setCompareDir((d) => (d === 'vertical' ? 'horizontal' : 'vertical'))}
                className="px-2 py-1 rounded-lg text-micro font-bold bg-white/10 hover:bg-white/20 text-white/80 transition-all flex items-center gap-1"
                title="Toggle Split Direction: Vertical vs Horizontal (X)"
              >
                {compareDir === 'vertical' ? '↔ Vert' : '↕ Horiz'}
              </button>
              {wipeAngle !== 0 && (
                <button
                  type="button"
                  onClick={() => setWipeAngle(0)}
                  className="px-1.5 py-0.5 rounded text-nano font-mono font-bold bg-accent/20 text-accent hover:bg-accent/30"
                  title="Reset rotation angle to 0°"
                >
                  {wipeAngle}° ↺
                </button>
              )}
            </>
          )}

          {compareMode === 'heatmap' && (
            <>
              <span className="w-px h-4 bg-white/15 mx-0.5" />
              <div className="flex items-center gap-1">
                <span className="text-nano font-mono text-white/50 uppercase">Gain:</span>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="0.5"
                  value={diffGain}
                  onChange={(e) => setDiffGain(parseFloat(e.target.value))}
                  className="w-14 h-1 accent-[var(--accent)] cursor-pointer"
                  title={`Amplified difference gain: ${diffGain}×`}
                />
                <span className="text-nano font-mono text-white/80">{diffGain}×</span>
              </div>
              <button
                type="button"
                onClick={() => setDiffColorMap((c) => (c === 'turbo' ? 'inferno' : 'turbo'))}
                className="px-1.5 py-0.5 rounded text-nano font-mono font-bold bg-white/10 hover:bg-white/20 text-white/80"
                title="Toggle false-color map: Turbo vs Inferno"
              >
                {diffColorMap.toUpperCase()}
              </button>
            </>
          )}

          {(compareMode === 'slider' || compareMode === 'blend') && (
            <>
              <span className="w-px h-4 bg-white/15 mx-0.5" />
              {[25, 50, 75].map((pct) => (
                <button
                  key={pct}
                  type="button"
                  onClick={() => {
                    sliderPosRef.current = pct;
                    setSliderPosition(pct);
                    if (compareMode === 'blend') canvasApiRef.current?.setWipe(pct);
                    else compareApiRef.current?.set(100 - pct);
                  }}
                  className={`px-1.5 py-1 rounded text-nano font-mono font-bold ${
                    sliderPosition === pct ? 'bg-white/20 text-white' : 'text-white/50 hover:text-white'
                  }`}
                >
                  {pct}%
                </button>
              ))}
              <span className="w-px h-4 bg-white/15 mx-0.5" />
              <button
                type="button"
                onClick={() => setAutoSwipe((a) => !a)}
                className={`px-2 py-1 rounded-lg text-micro font-bold transition-all flex items-center gap-1 ${
                  autoSwipe ? 'bg-emerald-500 text-black shadow-md font-extrabold animate-pulse' : 'bg-white/10 text-white/70 hover:text-white'
                }`}
                title="Toggle Auto-Swipe oscillation (A)"
              >
                {autoSwipe ? '⏸️ Auto' : '▶️ Auto'}
              </button>
            </>
          )}
        </div>
      )}

      {/* ── Pen tool toolbar ──────────────────────────────────────────────── */}
      {rotoActive && !splitMode && onRotoZones && (
        <RotoToolbar
          zones={rotoZones}
          selectedId={rotoSelectedId}
          onSelect={onRotoSelect || (() => {})}
          onChange={onRotoZones}
          newMode={rotoNewMode}
          setNewMode={setRotoNewMode}
          frame={frame}
          maxFrames={maxFrames}
          faces={faceAnchors}
          onDone={() => setRotoActive(false)}
        />
      )}

      {/* ── Sub-Pixel Mask Brush Toolbar ───────────────────────────────────── */}
      {maskBrushActive && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2 px-3 py-1.5 rounded-xl bg-black/85 backdrop-blur-md border border-white/15 shadow-2xl">
          <button
            type="button"
            onClick={() => setBrushMode('paint')}
            className={`px-2.5 py-1 rounded-lg text-micro font-bold transition-colors ${
              brushMode === 'paint' ? 'bg-pink-500 text-white' : 'bg-white/10 text-white/70'
            }`}
          >
            Paint
          </button>
          <button
            type="button"
            onClick={() => setBrushMode('erase')}
            className={`px-2.5 py-1 rounded-lg text-micro font-bold transition-colors ${
              brushMode === 'erase' ? 'bg-amber-500 text-black' : 'bg-white/10 text-white/70'
            }`}
          >
            Erase
          </button>
          <div className="flex items-center gap-1.5 px-2 border-l border-white/10">
            <span className="text-nano font-mono text-white/50 uppercase font-bold">Size:</span>
            <input
              type="range"
              min={8}
              max={120}
              value={brushSize}
              onChange={(e) => setBrushSize(parseInt(e.target.value, 10))}
              className="w-16 h-1 accent-pink-500 cursor-pointer"
            />
            <span className="text-micro font-mono font-bold text-white/80 tabular-nums">{brushSize}px</span>
          </div>

          <div className="flex items-center gap-1.5 px-2 border-l border-white/10">
            <span className="text-nano font-mono text-white/50 uppercase font-bold">Feather:</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={brushFeather}
              onChange={(e) => setBrushFeather(parseFloat(e.target.value))}
              className="w-14 h-1 accent-pink-400 cursor-pointer"
              title="Adjust edge feather falloff (Shift + [ / ])"
            />
            <span className="text-micro font-mono font-bold text-white/80 tabular-nums">{Math.round(brushFeather * 100)}%</span>
          </div>

          <button
            type="button"
            onClick={clearMaskCanvas}
            className="px-2 py-1 rounded-lg text-micro font-bold bg-white/10 hover:bg-white/20 text-white/70 hover:text-white transition-colors"
          >
            ↺ Clear
          </button>

          {onMaskChange && (
            <button
              type="button"
              onClick={commitMask}
              className={`px-2.5 py-1 rounded-lg text-micro font-bold transition-colors ${
                maskApplied ? 'bg-emerald-500 text-black' : 'bg-[var(--accent)] text-white hover:brightness-110'
              }`}
            >
              {maskApplied ? '✓ Applied' : 'Apply Mask'}
            </button>
          )}
        </div>
      )}

      {/* ── Viewport Stage Container ───────────────────────────────────────── */}
      <div
        ref={stageViewportRef}
        className="absolute inset-0 flex items-center justify-center pointer-events-none"
        style={transformFor(zoom, pan)}
      >
        <div className="relative z-10 pointer-events-auto" style={aspectStyle} ref={imageRef}>
          <PreviewCanvas
            ref={canvasApiRef}
            baseSrc={beforeSrc}
            baseFrame={beforeFrame}
            topSrc={afterSrc || beforeSrc}
            topFrame={afterFrame || (afterSrc ? null : beforeFrame)}
            wipe={canvasWipe}
            wipeDir={compareDir}
            wipeAngle={wipeAngle}
            mode={canvasMode}
            diffGain={diffGain}
            diffColorMap={diffColorMap}
            fadeMs={isPlaying ? 0 : scrubbing ? 60 : 200}
            zebra={zebraActive}
            faceBbox={activeFaceBbox}
            onDimensions={setImgDim}
            className="relative z-[1] w-full h-full object-contain pointer-events-none"
          />

          {playbackSource && (
            <FastCanvasPlayer
              source={playbackSource}
              label="Playback frame"
              className="absolute inset-0 z-[2] w-full h-full pointer-events-none"
            />
          )}

          <div className="absolute inset-0 pointer-events-none z-30">
            {faceBoxes}
            {debugOverlay}
          </div>

          {/* Sub-Pixel Mask Canvas Overlays */}
          {imgDim && (
            <>
              <canvas
                ref={maskCanvasRef}
                width={imgDim.w}
                height={imgDim.h}
                className={`absolute inset-0 w-full h-full object-contain z-[35] ${
                  maskBrushActive ? 'cursor-none pointer-events-auto' : 'pointer-events-none opacity-60'
                }`}
              />
              <canvas
                ref={maskExportRef}
                width={imgDim.w}
                height={imgDim.h}
                style={{ display: 'none' }}
                aria-hidden
              />
            </>
          )}

          {/* Pen tool zones (always drawn; interactive only while the tool is on) */}
          {imgDim && onRotoZones && (
            <RotoLayer
              imgDim={imgDim}
              zones={rotoZones}
              onChange={onRotoZones}
              selectedId={rotoSelectedId}
              onSelect={onRotoSelect || (() => {})}
              active={rotoActive && !splitMode}
              frame={frame}
              faces={faceAnchors}
              newMode={rotoNewMode}
              zoom={zoom}
            />
          )}

          {/* 360 Vector Wipe A/B Slider */}
          {compare && !isPeekingOriginal && compareMode === 'slider' && (
            <CompareSlider
              ref={compareApiRef}
              canvasRef={canvasApiRef}
              direction={compareDir}
              angle={wipeAngle}
              onAngleChange={setWipeAngle}
              defaultPosition={100 - sliderPosition}
              onCommit={(pct) => {
                sliderPosRef.current = 100 - pct;
                setSliderPosition(100 - pct);
              }}
              labelA="Target original"
              labelB="Swapped result"
              className="absolute inset-0 z-40 pointer-events-none"
            />
          )}
        </div>
      </div>

      {/* ── Dual Concentric Circle Mask Brush Cursor ───────────────────────── */}
      {maskBrushActive && (
        <div
          ref={brushCursorRef}
          className="absolute pointer-events-none z-50 -translate-x-1/2 -translate-y-1/2"
          style={{ width: brushSize, height: brushSize, willChange: 'transform' }}
        >
          {/* Outer dashed ring (radius) */}
          <div
            className="absolute inset-0 rounded-full border border-dashed border-pink-400 shadow-[0_0_6px_rgba(236,72,153,0.5)]"
          />
          {/* Inner solid ring (feather core) */}
          <div
            className="absolute rounded-full border border-pink-300 left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"
            style={{
              width: Math.max(2, brushSize * (1 - brushFeather)),
              height: Math.max(2, brushSize * (1 - brushFeather)),
            }}
          />
          {/* Center pinpoint */}
          <span className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-1 h-1 rounded-full bg-pink-200" />
        </div>
      )}

      {/* ── 3.5x–8x Floating Magnifier Loupe Overlay ───────────────────────── */}
      {magnifierActive && (
        <div
          ref={lensGlassRef}
          className="absolute rounded-full border-2 border-cyan-400 shadow-[0_0_35px_rgba(6,182,212,0.7)] overflow-hidden pointer-events-none z-50 bg-black"
          style={{ width: 180, height: 180, left: 0, top: 0, display: 'none', willChange: 'transform' }}
        >
          {/* Split Loupe: Left half Before (Original) | Right half After (Swapped) */}
          {loupeSplit ? (
            <div className="relative w-full h-full flex">
              <div className="w-1/2 h-full overflow-hidden relative border-r border-cyan-300/80">
                <img
                  ref={lensBeforeImgRef}
                  src={beforeSrc}
                  alt="Original loupe detail"
                  draggable={false}
                  style={{ position: 'absolute', left: 0, top: 0, maxWidth: 'none', objectFit: 'contain' }}
                />
                <span className="absolute bottom-1.5 left-2 px-1 rounded bg-black/80 text-nano font-mono text-cyan-200 font-bold">
                  ORIG
                </span>
              </div>
              <div className="w-1/2 h-full overflow-hidden relative">
                <img
                  ref={lensAfterImgRef}
                  src={afterSrc || beforeSrc}
                  alt="Swapped loupe detail"
                  draggable={false}
                  style={{ position: 'absolute', left: -90, top: 0, maxWidth: 'none', objectFit: 'contain' }}
                />
                <span className="absolute bottom-1.5 right-2 px-1 rounded bg-black/80 text-nano font-mono text-cyan-200 font-bold">
                  SWAP
                </span>
              </div>
            </div>
          ) : (
            <img
              ref={lensAfterImgRef}
              src={isPeekingOriginal ? beforeSrc : (afterSrc || beforeSrc)}
              alt="Magnifier lens view"
              draggable={false}
              style={{ position: 'absolute', left: 0, top: 0, maxWidth: 'none', objectFit: 'contain' }}
            />
          )}

          {/* Loupe Reticle Crosshair */}
          <span className="absolute left-1/2 top-1/2 w-4 h-px -translate-x-1/2 -translate-y-1/2 bg-cyan-300/80" />
          <span className="absolute left-1/2 top-1/2 h-4 w-px -translate-x-1/2 -translate-y-1/2 bg-cyan-300/80" />
          <span className="absolute top-1.5 left-1/2 -translate-x-1/2 px-1.5 py-0.5 rounded bg-black/80 backdrop-blur border border-cyan-400/40 text-nano font-mono font-bold text-cyan-300 tracking-wider">
            {loupePower}× LOUPE
          </span>
        </div>
      )}

      {/* ── Mini Radar Navigator (PiP) ─────────────────────────────────────── */}
      <RadarNavigator
        zoom={zoom}
        pan={pan}
        onPanChange={(p) => updateViewport(zoom, p, false)}
        imgDim={imgDim}
        thumbnailSrc={afterSrc || beforeSrc}
        stageRef={containerRef}
        contentRef={imageRef}
      />

      {/* ── Broadcast Scopes Dock (Waveform, RGB Parade, Vectorscope with I-Line) ── */}
      <BroadcastScopesDock
        active={scopeActive}
        onClose={() => setScopeActive(false)}
        sourceCanvas={canvasApiRef.current?.canvas?.()}
        renderedCanvas={canvasApiRef.current?.canvas?.()}
        faceBbox={faces[selectedFaceIndex ?? 0]}
        className="absolute bottom-16 right-3"
      />

            {/* ── Stage 21 3D / VR Viewport ── */}
      <StereoVRViewport
        active={vrActive}
        onClose={() => setVrActive(false)}
        mediaSource={afterSrc || beforeSrc || (imageRef.current?.querySelector('canvas') || imageRef.current?.querySelector('img') || imageRef.current?.querySelector('video'))}
        compareSource={beforeSrc || null}
        params={vrParams}
        onChange={setVrParams}
      />

      {/* ── WebGL2 3D .cube LUT Grading Panel ── */}
      {gradingActive && (
        <ColorGradingPanel
          params={gradingParams}
          onChange={(patch) => {
            setGradingParams((prev) => ({ ...prev, ...patch }));
            if (onGradingChange) onGradingChange(patch);
          }}
          onLutLoaded={setActiveLut}
          notify={notify}
          className="absolute bottom-16 left-3 w-[290px] z-40 max-h-[75vh] overflow-y-auto"
        />
      )}

      {/* ── Bottom Floating Pill & HUD ─────────────────────────────────────── */}
      {viewportPill()}
      {hudBar()}
    </div>
  );
}
