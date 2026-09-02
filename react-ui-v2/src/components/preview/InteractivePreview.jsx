import React, { useCallback, useEffect, useRef, useState } from 'react';
import { CrossfadeImage } from './CrossfadeImage';
import { TrackingOverlay } from './TrackingOverlay';
import { ComparisonWipe } from './ComparisonWipe';
import { MagnifierLoupe } from './MagnifierLoupe';
import { MaskBrushOverlay } from './MaskBrushOverlay';
import { PreviewControlsHUD } from './PreviewControlsHUD';
import {
  clampPan,
  panAnchoredAt,
  panCenteringAt,
  transformFor,
  uiScale,
  wheelZoom,
} from '../../adapters/zoomPan';

const ZOOM_MAX = 8;

/**
 * Master Interactive Preview Workstation Stage for React UI 2.0.
 * Orchestrates sub-pixel zoom/pan, face detection overlays, comparison wipes,
 * magnifying loupe, and interactive mask drawing.
 */
export function InteractivePreview({
  beforeSrc = '',
  afterSrc = '',
  faces = [],
  kps = [],
  pose = [],
  personIds = [],
  faceMapping = {},
  selectedFaceIndex = null,
  onSelectPerson,
  onMaskChange,
  _isProcessing = false,
  liveSrc = '',
  liveSeq = 0,
  className = '',
}) {
  const containerRef = useRef(null);
  const imageRef = useRef(null);
  const maskBrushRef = useRef(null);

  // Layout & Dimension state
  const [imgDim, setImgDim] = useState(null);

  // Zoom & Pan state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [startPan, setStartPan] = useState({ x: 0, y: 0 });

  // Comparison state
  const [compare, setCompare] = useState(false);
  const [compareMode, setCompareMode] = useState('slider'); // 'slider' | 'blend' | 'diff'
  const [compareDir, setCompareDir] = useState('vertical'); // 'vertical' | 'horizontal'
  const [sliderPosition, setSliderPosition] = useState(50);
  const [autoSwipe, setAutoSwipe] = useState(false);

  // Pro Tools state
  const [magnifierActive, setMagnifierActive] = useState(false);
  const [lensPos, setLensPos] = useState({ x: 0, y: 0, relX: 0, relY: 0, visible: false });
  const [maskBrushActive, setMaskBrushActive] = useState(false);
  const [brushMode] = useState('paint'); // 'paint' | 'erase'
  const [brushSize] = useState(30);
  const [isDrawingMask, setIsDrawingMask] = useState(false);

  // Overlays state
  const [showDetections, setShowDetections] = useState(true);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(1.0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Determine active visual frame
  const activeDisplaySrc = liveSrc || afterSrc || beforeSrc;
  const hasMedia = Boolean(activeDisplaySrc);
  const faceCount = faces?.length || 0;

  // Track natural image dimensions on load
  const handleImageLoad = (e) => {
    const { naturalWidth, naturalHeight } = e.target;
    if (naturalWidth && naturalHeight) {
      setImgDim({ w: naturalWidth, h: naturalHeight });
    }
  };

  // Zoom helpers
  const zoomRef = useRef(zoom);
  const panRef = useRef(pan);
  useEffect(() => {
    zoomRef.current = zoom;
  }, [zoom]);
  useEffect(() => {
    panRef.current = pan;
  }, [pan]);

  const zoomBy = useCallback((factor) => {
    const nz = Math.min(Math.max(1, zoomRef.current * factor), ZOOM_MAX);
    setZoom(nz);
    setPan(
      nz === 1
        ? { x: 0, y: 0 }
        : clampPan(panRef.current, nz, containerRef.current, imageRef.current),
    );
  }, []);

  const zoomToFit = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  const zoomToActual = useCallback(() => {
    const el = imageRef.current || containerRef.current;
    const w = el?.offsetWidth;
    if (!w || !imgDim) return;
    setZoom(Math.min(Math.max(imgDim.w / w, 1), ZOOM_MAX));
    setPan({ x: 0, y: 0 });
  }, [imgDim]);

  // Wheel Zoom Listener (Non-passive to cancel window scroll)
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return undefined;

    const onWheel = (e) => {
      if (e.ctrlKey || e.metaKey) return; // allow global zoom
      const next = wheelZoom(e.deltaY, zoomRef.current, ZOOM_MAX);
      if (next === null) return;
      e.preventDefault();

      if (next === 1) {
        setZoom(1);
        setPan({ x: 0, y: 0 });
        return;
      }

      const p = panAnchoredAt(
        { x: e.clientX, y: e.clientY },
        el,
        zoomRef.current,
        next,
        panRef.current,
      );
      setZoom(next);
      setPan(clampPan(p, next, el, imageRef.current));
    };

    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  // Keyboard Shortcuts Handler
  useEffect(() => {
    const onKeyDown = (e) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;

      const key = e.key.toLowerCase();
      if (key === '=' || key === '+') {
        e.preventDefault();
        zoomBy(1.3);
      } else if (key === '-' || key === '_') {
        e.preventDefault();
        zoomBy(1 / 1.3);
      } else if (key === '0') {
        e.preventDefault();
        zoomToFit();
      } else if (key === 'd') {
        setShowLandmarks((v) => !v);
      } else if (key === 'g') {
        setMagnifierActive((v) => !v);
      } else if (key === 'b') {
        setMaskBrushActive((v) => !v);
      } else if (key === 'x' && compare) {
        setCompareDir((d) => (d === 'vertical' ? 'horizontal' : 'vertical'));
      } else if (key === 'o' && compare) {
        setCompareMode((m) => (m === 'blend' ? 'slider' : 'blend'));
      } else if (key === 'a' && compare && compareMode !== 'diff') {
        setAutoSwipe((a) => !a);
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [compare, compareMode, zoomBy, zoomToFit]);

  // Pointer Handling for Pan, Magnifier, and Brush
  const handlePointerDown = (e) => {
    if (maskBrushActive) {
      setIsDrawingMask(true);
      maskBrushRef.current?.drawStroke(e, true);
      return;
    }
    if (zoom > 1) {
      setIsPanning(true);
      setStartPan({
        x: e.clientX,
        y: e.clientY,
        panX: pan.x,
        panY: pan.y,
        scale: uiScale(containerRef.current),
      });
      e.currentTarget.setPointerCapture?.(e.pointerId);
    }
  };

  const handlePointerMove = (e) => {
    const cx = e.clientX;
    const cy = e.clientY;

    // 1. Update Magnifier Loupe Position
    if (magnifierActive && containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const ir = imageRef.current?.getBoundingClientRect();
      const s = uiScale(containerRef.current);

      setLensPos({
        x: (cx - rect.left) / s,
        y: (cy - rect.top) / s,
        imgW: (ir?.width || rect.width) / s,
        imgH: (ir?.height || rect.height) / s,
        imgX: ((ir?.left ?? rect.left) - rect.left) / s,
        imgY: ((ir?.top ?? rect.top) - rect.top) / s,
        visible: cx >= rect.left && cx <= rect.right && cy >= rect.top && cy <= rect.bottom,
      });
    }

    // 2. Draw Mask Stroke
    if (maskBrushActive && isDrawingMask) {
      maskBrushRef.current?.drawStroke(e);
      return;
    }

    // 3. Pan Canvas
    if (isPanning && zoom > 1) {
      const s = startPan.scale || 1;
      setPan(
        clampPan(
          {
            x: startPan.panX + (cx - startPan.x) / s,
            y: startPan.panY + (cy - startPan.y) / s,
          },
          zoom,
          containerRef.current,
          imageRef.current,
        ),
      );
    }
  };

  const handlePointerUp = () => {
    setIsPanning(false);
    if (isDrawingMask) {
      setIsDrawingMask(false);
      maskBrushRef.current?.endStroke();
      maskBrushRef.current?.commit();
    }
  };

  const handleDoubleClick = (e) => {
    if (zoom > 1) {
      zoomToFit();
    } else {
      const z = 2.5;
      setZoom(z);
      setPan(
        panCenteringAt(
          { x: e.clientX, y: e.clientY },
          containerRef.current,
          z,
          imageRef.current,
        ),
      );
    }
  };

  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {});
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {});
    }
  };

  const transformStyle = {
    ...transformFor(zoom, pan),
    willChange: zoom > 1 || isPanning ? 'transform' : 'auto',
  };

  return (
    <div
      ref={containerRef}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      onDoubleClick={handleDoubleClick}
      className={`relative w-full h-full min-h-[420px] flex items-center justify-center overflow-hidden bg-[#07080c] select-none ${
        zoom > 1 ? (isPanning ? 'cursor-grabbing' : 'cursor-grab') : ''
      } ${className}`}
    >
      {/* Visual Status Badges (Top-Left) */}
      <div className="absolute top-3 left-3 z-30 flex items-center gap-1.5 pointer-events-none">
        {liveSrc ? (
          <span className="flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 text-[10px] font-mono font-bold backdrop-blur-md">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            LIVE SEQ #{liveSeq}
          </span>
        ) : afterSrc ? (
          <span className="px-2 py-0.5 rounded-md bg-[var(--accent-primary,#38bdf8)]/20 border border-[var(--accent-primary,#38bdf8)]/40 text-[var(--accent-primary,#38bdf8)] text-[10px] font-mono font-bold backdrop-blur-md">
            PREVIEW READY
          </span>
        ) : hasMedia ? (
          <span className="px-2 py-0.5 rounded-md bg-white/10 border border-white/15 text-white/70 text-[10px] font-mono font-semibold backdrop-blur-md">
            TARGET FRAME
          </span>
        ) : null}

        {faceCount > 0 ? (
          <span className="px-2 py-0.5 rounded-md bg-purple-500/20 border border-purple-500/40 text-purple-300 text-[10px] font-mono font-bold backdrop-blur-md">
            {faceCount} FACE{faceCount === 1 ? '' : 'S'} DETECTED
          </span>
        ) : hasMedia ? (
          <span className="px-2 py-0.5 rounded-md bg-amber-500/20 border border-amber-500/40 text-amber-300 text-[10px] font-mono font-semibold backdrop-blur-md">
            NO FACES DETECTED
          </span>
        ) : null}
      </div>

      {/* Main Canvas Presentation Viewport */}
      {hasMedia ? (
        <div
          ref={imageRef}
          className="relative max-w-full max-h-full flex items-center justify-center transition-transform duration-75"
          style={{
            aspectRatio: imgDim ? `${imgDim.w}/${imgDim.h}` : '16/9',
            ...transformStyle,
          }}
        >
          {compare && beforeSrc && (afterSrc || liveSrc) ? (
            <ComparisonWipe
              beforeSrc={beforeSrc}
              afterSrc={afterSrc || liveSrc}
              compareMode={compareMode}
              compareDir={compareDir}
              sliderPosition={sliderPosition}
              onSliderChange={setSliderPosition}
              isAutoSwiping={autoSwipe}
            />
          ) : (
            <CrossfadeImage
              src={activeDisplaySrc}
              alt="Target Preview"
              onLoad={handleImageLoad}
            />
          )}

          {/* Face Tracking & Landmarking Overlay Engine */}
          <TrackingOverlay
            faces={faces}
            kps={kps}
            pose={pose}
            personIds={personIds}
            selectedFaceIndex={selectedFaceIndex}
            faceMapping={faceMapping}
            imgDim={imgDim}
            showDetections={showDetections}
            showTracking={showTracking}
            showLandmarks={showLandmarks}
            showLabels={showLabels}
            overlayOpacity={overlayOpacity}
            onSelectPerson={onSelectPerson}
          />

          {/* Interactive Mask Brush Canvas */}
          <MaskBrushOverlay
            ref={maskBrushRef}
            active={maskBrushActive}
            brushMode={brushMode}
            brushSize={brushSize}
            imgDim={imgDim}
            onMaskChange={onMaskChange}
          />
        </div>
      ) : (
        /* Empty State */
        <div className="flex flex-col items-center justify-center gap-3 p-8 text-center select-none text-white/40">
          <div className="w-16 h-16 rounded-2xl bg-white/[0.03] border border-white/10 flex items-center justify-center text-2xl text-white/30 shadow-inner">
            ◇
          </div>
          <div className="space-y-1">
            <h3 className="text-sm font-semibold text-white/80">No Target Media Loaded</h3>
            <p className="text-xs text-white/40 max-w-[280px]">
              Drop an image or video onto the workspace or use the media intake rail to begin.
            </p>
          </div>
        </div>
      )}

      {/* 3.5x Magnifying Loupe Overlay */}
      <MagnifierLoupe
        src={activeDisplaySrc}
        lensPos={lensPos}
      />

      {/* Floating HUD Control Bar */}
      <PreviewControlsHUD
        zoom={zoom}
        onZoomIn={() => zoomBy(1.3)}
        onZoomOut={() => zoomBy(1 / 1.3)}
        onZoomReset={zoomToFit}
        onZoomActual={zoomToActual}
        showDetections={showDetections}
        onToggleDetections={() => setShowDetections((v) => !v)}
        showLandmarks={showLandmarks}
        onToggleLandmarks={() => setShowLandmarks((v) => !v)}
        showLabels={showLabels}
        onToggleLabels={() => setShowLabels((v) => !v)}
        overlayOpacity={overlayOpacity}
        onChangeOpacity={setOverlayOpacity}
        magnifierActive={magnifierActive}
        onToggleMagnifier={() => setMagnifierActive((v) => !v)}
        maskBrushActive={maskBrushActive}
        onToggleMaskBrush={() => setMaskBrushActive((v) => !v)}
        compare={compare}
        onToggleCompare={() => setCompare((v) => !v)}
        compareMode={compareMode}
        onChangeCompareMode={setCompareMode}
        compareDir={compareDir}
        onToggleCompareDir={() => setCompareDir((d) => (d === 'vertical' ? 'horizontal' : 'vertical'))}
        autoSwipe={autoSwipe}
        onToggleAutoSwipe={() => setAutoSwipe((v) => !v)}
        isFullscreen={isFullscreen}
        onToggleFullscreen={toggleFullscreen}
        hasMedia={hasMedia}
        faceCount={faceCount}
      />
    </div>
  );
}
