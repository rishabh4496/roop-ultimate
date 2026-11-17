import React, { useCallback, useEffect, useRef, useState } from 'react';
import { clampPan, panAnchoredAt, uiScale, wheelZoomDelta } from '../faceswap/zoomPan';

const MIN_ZOOM = 0.25;
const MAX_ZOOM = 8;

export default function ExtrasMediaViewport({ children, overlays, chrome, mediaSize, onMediaSize, mediaRef, scopes }) {
  const stageRef = useRef(null);
  const planeRef = useRef(null);
  const zoomRef = useRef(1);
  const panRef = useRef({ x: 0, y: 0 });
  const dragRef = useRef(null);
  const spaceDownRef = useRef(false);
  const settleRef = useRef(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [pixelGrid, setPixelGrid] = useState(false);
  const [stageBounds, setStageBounds] = useState({ width: 0, height: 0 });

  const paintTransform = useCallback((nextZoom, nextPan, commit = false) => {
    zoomRef.current = nextZoom;
    panRef.current = nextPan;
    if (planeRef.current) planeRef.current.style.transform = `scale(${nextZoom}) translate(${nextPan.x / nextZoom}px, ${nextPan.y / nextZoom}px)`;
    if (settleRef.current) clearTimeout(settleRef.current);
    if (commit) {
      setZoom(nextZoom);
      setPan(nextPan);
    } else {
      settleRef.current = setTimeout(() => {
        setZoom(nextZoom);
        setPan(nextPan);
      }, 100);
    }
  }, []);

  const fittedMediaBox = useCallback(() => {
    const stage = stageRef.current;
    if (!stage || !mediaSize?.width || !mediaSize?.height) return planeRef.current;
    const fit = Math.min(stage.clientWidth / mediaSize.width, stage.clientHeight / mediaSize.height);
    return { offsetWidth: mediaSize.width * fit, offsetHeight: mediaSize.height * fit };
  }, [mediaSize]);

  const fitZoom = useCallback(() => {
    paintTransform(1, { x: 0, y: 0 }, true);
  }, [paintTransform]);

  const fillZoom = useCallback(() => {
    const stage = stageRef.current;
    if (!stage || !mediaSize?.width || !mediaSize?.height) return;
    const rect = stage.getBoundingClientRect();
    const scale = uiScale(stage);
    const fit = Math.min(rect.width / scale / mediaSize.width, rect.height / scale / mediaSize.height);
    const fill = Math.max(rect.width / scale / mediaSize.width, rect.height / scale / mediaSize.height);
    paintTransform(Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, fill / Math.max(fit, 1e-6))), { x: 0, y: 0 }, true);
  }, [mediaSize, paintTransform]);

  const pixelZoom = useCallback((factor) => {
    const stage = stageRef.current;
    if (!stage || !mediaSize?.width || !mediaSize?.height) return;
    const rect = stage.getBoundingClientRect();
    const scale = uiScale(stage);
    const fit = Math.min(rect.width / scale / mediaSize.width, rect.height / scale / mediaSize.height);
    paintTransform(Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, factor / Math.max(fit, 1e-6))), { x: 0, y: 0 }, true);
  }, [mediaSize, paintTransform]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return undefined;
    const onWheel = (event) => {
      if (event.ctrlKey || event.metaKey) return;
      const next = wheelZoomDelta(event.deltaY, event.deltaMode, zoomRef.current, MAX_ZOOM, MIN_ZOOM);
      if (next == null) return;
      event.preventDefault();
      const anchored = panAnchoredAt({ x: event.clientX, y: event.clientY }, stage, zoomRef.current, next, panRef.current);
      paintTransform(next, clampPan(anchored, next, stage, fittedMediaBox()), false);
    };
    stage.addEventListener('wheel', onWheel, { passive: false });
    return () => stage.removeEventListener('wheel', onWheel);
  }, [paintTransform, fittedMediaBox]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return undefined;
    const measure = () => setStageBounds({ width: stage.clientWidth, height: stage.clientHeight });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(stage);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || zoomRef.current <= 1) return;
    const next = clampPan(panRef.current, zoomRef.current, stage, fittedMediaBox());
    if (Math.abs(next.x - panRef.current.x) > 0.5 || Math.abs(next.y - panRef.current.y) > 0.5) {
      paintTransform(zoomRef.current, next, true);
    }
  }, [stageBounds, fittedMediaBox, paintTransform]);

  useEffect(() => {
    const ignore = (target) => target?.closest?.('input,textarea,select,button,[contenteditable="true"]');
    const down = (event) => { if (event.code === 'Space' && !ignore(event.target)) spaceDownRef.current = true; };
    const up = (event) => { if (event.code === 'Space') spaceDownRef.current = false; };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', up); };
  }, []);

  useEffect(() => () => { if (settleRef.current) clearTimeout(settleRef.current); }, []);

  useEffect(() => {
    if (mediaSize?.width && mediaSize?.height) fitZoom();
  }, [fitZoom, mediaSize?.width, mediaSize?.height]);

  const beginPan = (event) => {
    if (event.button !== 1 && !(event.button === 0 && spaceDownRef.current)) return;
    event.preventDefault();
    dragRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY, pan: panRef.current };
    event.currentTarget.setPointerCapture?.(event.pointerId);
  };
  const movePan = (event) => {
    const drag = dragRef.current;
    if (!drag || drag.id !== event.pointerId) return;
    const scale = uiScale(stageRef.current);
    const next = clampPan({ x: drag.pan.x + (event.clientX - drag.x) / scale, y: drag.pan.y + (event.clientY - drag.y) / scale }, zoomRef.current, stageRef.current, fittedMediaBox());
    paintTransform(zoomRef.current, next, false);
  };
  const endPan = (event) => {
    if (!dragRef.current || dragRef.current.id !== event.pointerId) return;
    dragRef.current = null;
    event.currentTarget.releasePointerCapture?.(event.pointerId);
    setZoom(zoomRef.current);
    setPan(panRef.current);
  };

  const mediaFitScale = mediaSize?.width && mediaSize?.height && stageBounds.width && stageBounds.height
    ? Math.min(stageBounds.width / mediaSize.width, stageBounds.height / mediaSize.height)
    : 1;
  const gridWidth = (mediaSize?.width || stageBounds.width) * mediaFitScale;
  const gridHeight = (mediaSize?.height || stageBounds.height) * mediaFitScale;
  return (
    <div
      ref={stageRef}
      className="extras-media-viewport"
      tabIndex={0}
      aria-label="Media inspection viewport. Use the mouse wheel to zoom, or hold Space and drag to pan."
      onPointerDown={beginPan}
      onPointerMove={movePan}
      onPointerUp={endPan}
      onPointerCancel={endPan}
      onContextMenu={(event) => event.preventDefault()}
    >
      <div ref={planeRef} className={`extras-media-viewport__plane${zoom >= 3 ? ' is-pixelated' : ''}`}>
        {React.cloneElement(children, {
          ref: mediaRef,
          style: { ...children.props.style, imageRendering: zoom >= 3 ? 'pixelated' : 'auto' },
          onLoadedMetadata: (event) => {
            onMediaSize?.({ width: event.currentTarget.videoWidth, height: event.currentTarget.videoHeight });
            children.props.onLoadedMetadata?.(event);
          },
          onLoad: (event) => {
            onMediaSize?.({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight });
            children.props.onLoad?.(event);
          },
        })}
        {overlays}
        {zoom >= 3 && pixelGrid && <div className="extras-media-viewport__pixel-grid" style={{ width: `${gridWidth}px`, height: `${gridHeight}px`, '--pixel-grid-size': `${mediaFitScale}px` }} aria-hidden="true" />}
      </div>

      {chrome}
      <div className="extras-media-viewport__toolbar" role="toolbar" aria-label="Viewport controls">
        <span className="extras-media-viewport__zoom" aria-live="polite">{Math.round(zoom * 100)}%</span>
        <button type="button" onClick={fitZoom} title="Fit image to viewport (F)">Fit</button>
        <button type="button" onClick={fillZoom} title="Fill viewport; crop overflow">Fill</button>
        <button type="button" onClick={() => pixelZoom(1)} title="1:1 source pixels">100%</button>
        <button type="button" onClick={() => pixelZoom(2)} title="2:1 source pixels">200%</button>
        {zoom >= 3 && <button type="button" aria-pressed={pixelGrid} onClick={() => setPixelGrid((value) => !value)} title="Toggle pixel grid">Grid</button>}
        <span className="extras-media-viewport__hint">MMB / Space drag · scroll zoom</span>
      </div>
      {scopes}
    </div>
  );
}
