import React, {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from 'react';
import {
  Check,
  Copy,
  Flame,
  Grid,
  Layers,
  Maximize2,
  Minimize2,
  RotateCcw,
  Shield,
  Sliders,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { CinematicRenderer } from './cinematicRenderer';
import { useCinematicControls } from './useCinematicControls';

/**
 * <CinematicPreview />
 * High-performance WebGL 2.0 dual-buffered viewport with:
 *   - GLSL fragment shaders (Modes 0-3: Standard, Split Screen, Diff Heatmap, BiSeNet Mask)
 *   - Double-buffered GPU textures for zero-tear frame swapping
 *   - Binary WebSocket frame packet ingestion (RGBA buffers or WebP/JPEG blobs)
 *   - Cursor-anchored smooth wheel zoom (10% to 1600%) and infinite canvas drag-to-pan
 *   - gl.LINEAR (smooth) vs gl.NEAREST (pixel-peeping) inspection toggle
 *   - Top-right HUD ribbon with resolution, scale percentage, and live HEX/RGB color picker
 *   - Clean ResizeObserver viewport auto-scaling preserving aspect ratio
 */
export const CinematicPreview = forwardRef(function CinematicPreview(
  {
    targetSrc,
    swapSrc,
    maskSrc,
    targetFrame,
    swapFrame,
    maskFrame,
    wsUrl,
    mode: initialMode = 1, // Default to A/B Split Screen
    standardLayer: initialStandardLayer = 1, // 0: Target, 1: Swap
    splitRatio: initialSplitRatio = 0.5,
    splitOrientation: initialSplitOrientation = 'vertical', // 'vertical' | 'horizontal'
    diffGain: initialDiffGain = 3.0,
    colormap: initialColormap = 'turbo', // 'turbo' | 'inferno'
    maskOpacity: initialMaskOpacity = 0.55,
    filterMode: initialFilterMode = 'linear', // 'linear' | 'nearest'
    className = '',
    style = {},
  },
  ref
) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const rendererRef = useRef(null);
  const wsRef = useRef(null);

  // UI state
  const [mode, setMode] = useState(initialMode);
  const [standardLayer, setStandardLayer] = useState(initialStandardLayer);
  const [splitRatio, setSplitRatio] = useState(initialSplitRatio);
  const [splitOrientation, setSplitOrientation] = useState(initialSplitOrientation);
  const [diffGain, setDiffGain] = useState(initialDiffGain);
  const [colormap, setColormap] = useState(initialColormap);
  const [maskOpacity, setMaskOpacity] = useState(initialMaskOpacity);
  const [filterMode, setFilterMode] = useState(initialFilterMode);
  const [imgDims, setImgDims] = useState([0, 0]);
  const [cursorColor, setCursorColor] = useState(null);
  const [copied, setCopied] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showControlsDrawer, setShowControlsDrawer] = useState(true);
  const [isDraggingSplit, setIsDraggingSplit] = useState(false);

  // Transform change callback: update renderer and request draw
  const handleTransformChange = useCallback((panX, panY, z) => {
    if (!rendererRef.current) return;
    rendererRef.current.setPanZoom(panX, panY, z);
    rendererRef.current.draw();
  }, []);

  // Cursor move callback: update pixel inspector readout
  const handleCursorMove = useCallback(({ canvasX, canvasY, inBounds }) => {
    if (!inBounds || !rendererRef.current) {
      setCursorColor(null);
      return;
    }
    const dpr = Math.min(typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1, 2);
    const color = rendererRef.current.readPixelAt(canvasX, canvasY, dpr);
    if (color) {
      setCursorColor(color);
    }
  }, []);

  // Controls hook (Pan, Zoom, Anchor Math)
  const {
    zoom,
    pan,
    isDragging,
    handleWheel,
    handlePointerDown,
    handlePointerMove,
    handlePointerUp,
    resetView,
    setZoomLevel,
  } = useCinematicControls({
    canvasRef,
    onTransformChange: handleTransformChange,
    onCursorMove: handleCursorMove,
  });

  // Initialize WebGL 2.0 Renderer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const renderer = new CinematicRenderer(canvas);
    rendererRef.current = renderer;

    renderer.setMode(mode);
    renderer.setStandardLayer(standardLayer);
    renderer.setSplitRatio(splitRatio);
    renderer.setSplitOrientation(splitOrientation);
    renderer.setDiffGain(diffGain);
    renderer.setColormap(colormap);
    renderer.setMaskOpacity(maskOpacity);
    renderer.setFilterMode(filterMode);
    renderer.setPanZoom(pan.x, pan.y, zoom);

    return () => {
      renderer.destroy();
      rendererRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Run once on mount

  // ResizeObserver for clean canvas scaling without distortion
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    let rafId = null;
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const rect = entry.contentRect;
      if (rafId) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        rafId = null;
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = Math.max(1, Math.round(rect.width * dpr));
        const h = Math.max(1, Math.round(rect.height * dpr));

        if (canvas.width !== w || canvas.height !== h) {
          canvas.width = w;
          canvas.height = h;
          if (rendererRef.current) {
            rendererRef.current.draw();
          }
        }
      });
    });

    ro.observe(container);
    return () => {
      if (rafId) cancelAnimationFrame(rafId);
      ro.disconnect();
    };
  }, []);

  // Synchronize state props with renderer
  useEffect(() => {
    if (!rendererRef.current) return;
    rendererRef.current.setMode(mode);
    rendererRef.current.draw();
  }, [mode]);

  useEffect(() => {
    if (!rendererRef.current) return;
    rendererRef.current.setStandardLayer(standardLayer);
    rendererRef.current.draw();
  }, [standardLayer]);

  useEffect(() => {
    if (!rendererRef.current) return;
    rendererRef.current.setSplitRatio(splitRatio);
    rendererRef.current.draw();
  }, [splitRatio]);

  useEffect(() => {
    if (!rendererRef.current) return;
    rendererRef.current.setSplitOrientation(splitOrientation);
    rendererRef.current.draw();
  }, [splitOrientation]);

  useEffect(() => {
    if (!rendererRef.current) return;
    rendererRef.current.setDiffGain(diffGain);
    rendererRef.current.draw();
  }, [diffGain]);

  useEffect(() => {
    if (!rendererRef.current) return;
    rendererRef.current.setColormap(colormap);
    rendererRef.current.draw();
  }, [colormap]);

  useEffect(() => {
    if (!rendererRef.current) return;
    rendererRef.current.setMaskOpacity(maskOpacity);
    rendererRef.current.draw();
  }, [maskOpacity]);

  useEffect(() => {
    if (!rendererRef.current) return;
    rendererRef.current.setFilterMode(filterMode);
    rendererRef.current.draw();
  }, [filterMode]);

  // Helper to load image from URL into a layer
  const loadSourceIntoLayer = useCallback((layerIndex, src) => {
    if (!src) return;
    let active = true;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      if (!active || !rendererRef.current) return;
      rendererRef.current.uploadImage(layerIndex, img);
      rendererRef.current.draw();
      setImgDims(rendererRef.current.getImageDimensions());
    };
    img.src = src;
    return () => {
      active = false;
    };
  }, []);

  // Ingest source URLs
  useEffect(() => {
    if (targetSrc) return loadSourceIntoLayer(0, targetSrc);
  }, [targetSrc, loadSourceIntoLayer]);

  useEffect(() => {
    if (swapSrc) return loadSourceIntoLayer(1, swapSrc);
  }, [swapSrc, loadSourceIntoLayer]);

  useEffect(() => {
    if (maskSrc) return loadSourceIntoLayer(2, maskSrc);
  }, [maskSrc, loadSourceIntoLayer]);

  // Ingest direct frame inputs (ImageBitmap / Blob / ArrayBuffer)
  useEffect(() => {
    if (!targetFrame || !rendererRef.current) return;
    if (typeof ImageBitmap !== 'undefined' && targetFrame instanceof ImageBitmap) {
      rendererRef.current.uploadImage(0, targetFrame);
      rendererRef.current.draw();
      setImgDims(rendererRef.current.getImageDimensions());
    } else {
      rendererRef.current.ingestBinaryPacket(0, targetFrame).then(() => {
        rendererRef.current?.draw();
        setImgDims(rendererRef.current?.getImageDimensions() || [0, 0]);
      });
    }
  }, [targetFrame]);

  useEffect(() => {
    if (!swapFrame || !rendererRef.current) return;
    if (typeof ImageBitmap !== 'undefined' && swapFrame instanceof ImageBitmap) {
      rendererRef.current.uploadImage(1, swapFrame);
      rendererRef.current.draw();
      setImgDims(rendererRef.current.getImageDimensions());
    } else {
      rendererRef.current.ingestBinaryPacket(1, swapFrame).then(() => {
        rendererRef.current?.draw();
        setImgDims(rendererRef.current?.getImageDimensions() || [0, 0]);
      });
    }
  }, [swapFrame]);

  useEffect(() => {
    if (!maskFrame || !rendererRef.current) return;
    if (typeof ImageBitmap !== 'undefined' && maskFrame instanceof ImageBitmap) {
      rendererRef.current.uploadImage(2, maskFrame);
      rendererRef.current.draw();
    } else {
      rendererRef.current.ingestBinaryPacket(2, maskFrame).then(() => {
        rendererRef.current?.draw();
      });
    }
  }, [maskFrame]);

  // WebSocket Binary Frame Streaming
  useEffect(() => {
    if (!wsUrl) return;
    let ws = null;
    let reconnectTimer = null;
    let isDisposed = false;
    let retryDelay = 1000;

    function connect() {
      if (isDisposed) return;
      try {
        ws = new WebSocket(wsUrl);
        ws.binaryType = 'arraybuffer';
        wsRef.current = ws;

        ws.onopen = () => {
          retryDelay = 1000;
        };

        ws.onmessage = async (e) => {
          if (!rendererRef.current || !(e.data instanceof ArrayBuffer) || isDisposed) return;
          // Ingest into swap layer (1) by default or stream based on header
          await rendererRef.current.ingestBinaryPacket(1, e.data);
          if (isDisposed || !rendererRef.current) return;
          rendererRef.current.draw();
          setImgDims(rendererRef.current.getImageDimensions());
        };

        ws.onclose = () => {
          wsRef.current = null;
          if (!isDisposed) {
            reconnectTimer = setTimeout(() => {
              retryDelay = Math.min(retryDelay * 1.5, 10000);
              connect();
            }, retryDelay);
          }
        };

        ws.onerror = (err) => {
          console.warn('[CinematicPreview] WebSocket error:', err);
        };
      } catch (err) {
        console.warn('[CinematicPreview] Failed to open WebSocket:', err);
        if (!isDisposed) {
          reconnectTimer = setTimeout(connect, retryDelay);
        }
      }
    }

    connect();

    return () => {
      isDisposed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws) {
        try {
          ws.close();
        } catch {
          // Ignore
        }
      }
      wsRef.current = null;
    };
  }, [wsUrl]);

  // Interactive Split Screen Handle Dragging
  const handleSplitPointerDown = useCallback((e) => {
    e.stopPropagation();
    e.preventDefault();
    setIsDraggingSplit(true);
    e.target.setPointerCapture?.(e.pointerId);
  }, []);

  const handleSplitPointerMove = useCallback(
    (e) => {
      if (!isDraggingSplit) return;
      const container = containerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();

      let ratio;
      if (splitOrientation === 'vertical') {
        ratio = (e.clientX - rect.left) / rect.width;
      } else {
        ratio = (e.clientY - rect.top) / rect.height;
      }
      const clamped = Math.max(0.01, Math.min(0.99, ratio));
      setSplitRatio(clamped);
      if (rendererRef.current) {
        rendererRef.current.setSplitRatio(clamped);
        rendererRef.current.draw();
      }
    },
    [isDraggingSplit, splitOrientation]
  );

  const handleSplitPointerUp = useCallback(
    (e) => {
      if (isDraggingSplit) {
        setIsDraggingSplit(false);
        try {
          e.target.releasePointerCapture?.(e.pointerId);
        } catch {
          // Ignore
        }
      }
    },
    [isDraggingSplit]
  );

  // Copy Color to Clipboard
  const handleCopyColor = useCallback(() => {
    if (!cursorColor) return;
    if (navigator?.clipboard?.writeText) {
      navigator.clipboard.writeText(cursorColor.hex).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1600);
      });
    }
  }, [cursorColor]);

  // Fullscreen toggle
  const toggleFullscreen = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    if (!document.fullscreenElement) {
      el.requestFullscreen?.().then(() => setIsFullscreen(true)).catch(() => {});
    } else {
      document.exitFullscreen?.().then(() => setIsFullscreen(false)).catch(() => {});
    }
  }, []);

  // Expose Imperative API via forwardRef
  useImperativeHandle(ref, () => ({
    pushTargetFrame: (frame, meta) => {
      rendererRef.current?.ingestBinaryPacket(0, frame, meta).then(() => {
        rendererRef.current?.draw();
        setImgDims(rendererRef.current?.getImageDimensions() || [0, 0]);
      });
    },
    pushSwapFrame: (frame, meta) => {
      rendererRef.current?.ingestBinaryPacket(1, frame, meta).then(() => {
        rendererRef.current?.draw();
        setImgDims(rendererRef.current?.getImageDimensions() || [0, 0]);
      });
    },
    pushMaskFrame: (frame, meta) => {
      rendererRef.current?.ingestBinaryPacket(2, frame, meta).then(() => {
        rendererRef.current?.draw();
      });
    },
    uploadRgba: (layer, buf, w, h) => {
      rendererRef.current?.uploadRgba(layer, buf, w, h);
      rendererRef.current?.draw();
      setImgDims(rendererRef.current?.getImageDimensions() || [0, 0]);
    },
    setMode: (m) => setMode(m),
    setSplitRatio: (r) => setSplitRatio(r),
    setFilterMode: (f) => setFilterMode(f),
    resetView: () => resetView(),
    setZoomLevel: (z) => setZoomLevel(z),
    getColorAt: (x, y) => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      return rendererRef.current?.readPixelAt(x, y, dpr);
    },
    getRenderer: () => rendererRef.current,
  }));

  const [imgW, imgH] = imgDims;

  return (
    <div
      ref={containerRef}
      className={`relative w-full h-full min-h-[460px] bg-zinc-950 overflow-hidden select-none font-sans text-zinc-100 flex flex-col items-center justify-center ${className}`}
      style={style}
      onPointerMove={isDraggingSplit ? handleSplitPointerMove : undefined}
      onPointerUp={isDraggingSplit ? handleSplitPointerUp : undefined}
    >
      {/* WebGL 2.0 Canvas */}
      <canvas
        ref={canvasRef}
        className={`block w-full h-full outline-none touch-none ${
          isDragging ? 'cursor-grabbing' : 'cursor-grab'
        }`}
        onWheel={handleWheel}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
      />

      {/* Interactive Mode 1 Split Handle Overlay */}
      {mode === 1 && (
        <div
          className={`absolute pointer-events-auto z-20 flex items-center justify-center transition-transform ${
            splitOrientation === 'vertical'
              ? 'top-0 bottom-0 w-8 -ml-4 cursor-col-resize'
              : 'left-0 right-0 h-8 -mt-4 cursor-row-resize'
          }`}
          style={
            splitOrientation === 'vertical'
              ? { left: `${splitRatio * 100}%` }
              : { top: `${splitRatio * 100}%` }
          }
          onPointerDown={handleSplitPointerDown}
          title="Drag to adjust split divider"
        >
          <div className="flex items-center justify-center w-6 h-6 rounded-full bg-zinc-900/90 border border-zinc-600/80 shadow-2xl text-white hover:scale-110 active:scale-95 transition-transform backdrop-blur-md">
            <Sliders size={12} className="text-amber-400" />
          </div>
        </div>
      )}

      {/* Mini Top-Right Status Ribbon */}
      <div className="absolute top-3 right-3 z-30 flex items-center gap-2 p-1.5 px-3 rounded-lg bg-zinc-900/85 backdrop-blur-md border border-zinc-800 shadow-2xl text-xs text-zinc-300 pointer-events-auto">
        {/* Resolution Badge */}
        <div className="flex items-center gap-1.5 font-mono">
          <span className="text-zinc-500 text-micro font-bold tracking-wider">RES</span>
          <span className="text-zinc-200 font-medium">
            {imgW > 0 ? `${imgW}×${imgH}` : '—'}
          </span>
        </div>

        <div className="w-[1px] h-3.5 bg-zinc-800" />

        {/* Zoom & Scale Badge */}
        <div className="flex items-center gap-1.5 font-mono">
          <span className="text-zinc-500 text-micro font-bold tracking-wider">ZOOM</span>
          <span className="font-semibold text-amber-400">{Math.round(zoom * 100)}%</span>
        </div>

        <div className="w-[1px] h-3.5 bg-zinc-800" />

        {/* Color-Under-Cursor Picker */}
        {cursorColor ? (
          <div
            className="flex items-center gap-2 cursor-pointer group hover:bg-zinc-800/60 p-0.5 px-1.5 rounded transition-colors"
            onClick={handleCopyColor}
            title="Click to copy HEX color code"
          >
            <div
              className="w-3.5 h-3.5 rounded-sm border border-white/40 shadow-inner shrink-0"
              style={{ backgroundColor: cursorColor.hex }}
            />
            <div className="font-mono flex items-center gap-1.5">
              <span className="text-zinc-100 group-hover:text-amber-300 font-semibold tracking-wide">
                {cursorColor.hex}
              </span>
              <span className="text-zinc-400 text-micro">
                ({cursorColor.r},{cursorColor.g},{cursorColor.b})
              </span>
            </div>
            {copied ? (
              <span className="text-micro text-emerald-400 font-semibold flex items-center gap-0.5">
                <Check size={11} /> Copied!
              </span>
            ) : (
              <Copy size={11} className="text-zinc-500 group-hover:text-zinc-300 transition-colors" />
            )}
          </div>
        ) : (
          <div className="flex items-center gap-1.5 font-mono text-zinc-500">
            <div className="w-3.5 h-3.5 rounded-sm border border-zinc-700/60 bg-zinc-800/40 shrink-0" />
            <span>#------</span>
          </div>
        )}
      </div>

      {/* Floating Bottom Control Bar */}
      <div className="absolute bottom-4 z-30 flex flex-col items-center gap-2 pointer-events-auto max-w-[94%]">
        {/* Mode-Specific Parameters Drawer */}
        {showControlsDrawer && (
          <div className="flex flex-wrap items-center gap-3 p-2 px-3 rounded-lg bg-zinc-900/90 backdrop-blur-md border border-zinc-800 shadow-xl text-xs text-zinc-300 animate-in fade-in duration-200">
            {/* Mode 0 Controls: Target vs Swap */}
            {mode === 0 && (
              <div className="flex items-center gap-2">
                <span className="text-zinc-400 font-medium">Layer:</span>
                <div className="flex items-center rounded-md bg-zinc-950 p-0.5 border border-zinc-800">
                  <button
                    type="button"
                    onClick={() => setStandardLayer(0)}
                    className={`px-2 py-1 rounded text-xs font-medium transition-all ${
                      standardLayer === 0
                        ? 'bg-amber-500 text-zinc-950 font-semibold shadow'
                        : 'text-zinc-400 hover:text-white'
                    }`}
                  >
                    Original Target
                  </button>
                  <button
                    type="button"
                    onClick={() => setStandardLayer(1)}
                    className={`px-2 py-1 rounded text-xs font-medium transition-all ${
                      standardLayer === 1
                        ? 'bg-amber-500 text-zinc-950 font-semibold shadow'
                        : 'text-zinc-400 hover:text-white'
                    }`}
                  >
                    Swapped Face
                  </button>
                </div>
              </div>
            )}

            {/* Mode 1 Controls: Split Screen */}
            {mode === 1 && (
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5">
                  <span className="text-zinc-400">Orientation:</span>
                  <div className="flex items-center rounded-md bg-zinc-950 p-0.5 border border-zinc-800">
                    <button
                      type="button"
                      onClick={() => setSplitOrientation('vertical')}
                      className={`px-2 py-0.5 rounded text-xs transition-all ${
                        splitOrientation === 'vertical'
                          ? 'bg-zinc-700 text-white font-medium'
                          : 'text-zinc-400 hover:text-white'
                      }`}
                    >
                      Vertical
                    </button>
                    <button
                      type="button"
                      onClick={() => setSplitOrientation('horizontal')}
                      className={`px-2 py-0.5 rounded text-xs transition-all ${
                        splitOrientation === 'horizontal'
                          ? 'bg-zinc-700 text-white font-medium'
                          : 'text-zinc-400 hover:text-white'
                      }`}
                    >
                      Horizontal
                    </button>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-zinc-400">Split:</span>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={splitRatio}
                    onChange={(e) => setSplitRatio(parseFloat(e.target.value))}
                    className="w-24 accent-amber-500 cursor-pointer h-1 bg-zinc-700 rounded-lg appearance-none"
                  />
                  <span className="font-mono text-zinc-300 w-8">{Math.round(splitRatio * 100)}%</span>
                </div>
              </div>
            )}

            {/* Mode 2 Controls: Difference Heatmap */}
            {mode === 2 && (
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <span className="text-zinc-400">Gain:</span>
                  <input
                    type="range"
                    min="1.0"
                    max="10.0"
                    step="0.5"
                    value={diffGain}
                    onChange={(e) => setDiffGain(parseFloat(e.target.value))}
                    className="w-24 accent-amber-500 cursor-pointer h-1 bg-zinc-700 rounded-lg appearance-none"
                  />
                  <span className="font-mono text-amber-400 w-8">{diffGain.toFixed(1)}×</span>
                </div>

                <div className="w-[1px] h-3.5 bg-zinc-800" />

                <div className="flex items-center gap-1.5">
                  <span className="text-zinc-400">Colormap:</span>
                  <div className="flex items-center rounded-md bg-zinc-950 p-0.5 border border-zinc-800">
                    <button
                      type="button"
                      onClick={() => setColormap('turbo')}
                      className={`px-2 py-0.5 rounded text-xs transition-all flex items-center gap-1 ${
                        colormap === 'turbo'
                          ? 'bg-gradient-to-r from-blue-600 via-emerald-500 to-red-500 text-white font-semibold shadow'
                          : 'text-zinc-400 hover:text-white'
                      }`}
                    >
                      Turbo
                    </button>
                    <button
                      type="button"
                      onClick={() => setColormap('inferno')}
                      className={`px-2 py-0.5 rounded text-xs transition-all flex items-center gap-1 ${
                        colormap === 'inferno'
                          ? 'bg-gradient-to-r from-purple-900 via-red-600 to-yellow-400 text-white font-semibold shadow'
                          : 'text-zinc-400 hover:text-white'
                      }`}
                    >
                      Inferno
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Mode 3 Controls: Mask Overlay */}
            {mode === 3 && (
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <span className="text-zinc-400">Opacity:</span>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={maskOpacity}
                    onChange={(e) => setMaskOpacity(parseFloat(e.target.value))}
                    className="w-24 accent-emerald-500 cursor-pointer h-1 bg-zinc-700 rounded-lg appearance-none"
                  />
                  <span className="font-mono text-zinc-300 w-8">{Math.round(maskOpacity * 100)}%</span>
                </div>

                <div className="w-[1px] h-3.5 bg-zinc-800" />

                <div className="flex items-center gap-3 text-mini">
                  <span className="flex items-center gap-1.5 font-medium">
                    <span className="w-2.5 h-2.5 rounded-full bg-red-500 shadow-sm" />
                    <span>Swapped Skin</span>
                  </span>
                  <span className="flex items-center gap-1.5 font-medium">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-sm" />
                    <span>Preserved Hair / Occluders</span>
                  </span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Master Toolbar Dock */}
        <div className="flex items-center gap-1.5 p-1.5 px-2 rounded-xl bg-zinc-900/90 backdrop-blur-md border border-zinc-800 shadow-2xl">
          {/* Modes 0-3 Switcher */}
          <div className="flex items-center rounded-lg bg-zinc-950 p-0.5 border border-zinc-800/80">
            <button
              type="button"
              onClick={() => {
                setMode(0);
                setShowControlsDrawer(true);
              }}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-all ${
                mode === 0
                  ? 'bg-zinc-800 text-amber-400 font-semibold shadow-sm'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
              title="Mode 0: Standard Single Frame"
              aria-label="Standard Single Frame Mode"
            >
              <Layers size={13} />
              <span>Standard</span>
            </button>

            <button
              type="button"
              onClick={() => {
                setMode(1);
                setShowControlsDrawer(true);
              }}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-all ${
                mode === 1
                  ? 'bg-zinc-800 text-amber-400 font-semibold shadow-sm'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
              title="Mode 1: A/B Split Screen Wipe"
              aria-label="A/B Split Screen Mode"
            >
              <Sliders size={13} />
              <span>A/B Split</span>
            </button>

            <button
              type="button"
              onClick={() => {
                setMode(2);
                setShowControlsDrawer(true);
              }}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-all ${
                mode === 2
                  ? 'bg-zinc-800 text-amber-400 font-semibold shadow-sm'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
              title="Mode 2: Difference Heatmap"
              aria-label="Difference Heatmap Mode"
            >
              <Flame size={13} />
              <span>Heatmap</span>
            </button>

            <button
              type="button"
              onClick={() => {
                setMode(3);
                setShowControlsDrawer(true);
              }}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-all ${
                mode === 3
                  ? 'bg-zinc-800 text-amber-400 font-semibold shadow-sm'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
              title="Mode 3: BiSeNet Occlusion Mask Blend"
              aria-label="BiSeNet Mask Mode"
            >
              <Shield size={13} />
              <span>Mask</span>
            </button>
          </div>

          <div className="w-[1px] h-4 bg-zinc-800 mx-0.5" />

          {/* Texture Filtering Toggle: LINEAR vs NEAREST */}
          <button
            type="button"
            onClick={() => setFilterMode((f) => (f === 'linear' ? 'nearest' : 'linear'))}
            className={`flex items-center gap-1.5 px-2 py-1.5 rounded-lg text-xs font-medium transition-all border ${
              filterMode === 'nearest'
                ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                : 'bg-zinc-800/50 text-zinc-300 hover:text-white border-zinc-700/50'
            }`}
            title="Toggle texture filtering: Linear (Smooth Bilinear) vs Nearest (Pixel Peep)"
            aria-label="Toggle Texture Filter Mode"
          >
            <Grid size={13} />
            <span>{filterMode === 'nearest' ? 'Pixel Peep' : 'Smooth'}</span>
          </button>

          <div className="w-[1px] h-4 bg-zinc-800 mx-0.5" />

          {/* Zoom Controls */}
          <div className="flex items-center gap-0.5">
            <button
              type="button"
              onClick={() => setZoomLevel(zoom / 1.4)}
              className="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
              title="Zoom Out"
              aria-label="Zoom Out"
            >
              <ZoomOut size={14} />
            </button>

            <button
              type="button"
              onClick={resetView}
              className="px-2 py-1 rounded-md text-mini font-mono text-zinc-300 hover:text-white hover:bg-zinc-800 transition-colors"
              title="Click to reset zoom & pan (Fit to view)"
              aria-label="Fit to View"
            >
              Fit
            </button>

            <button
              type="button"
              onClick={() => setZoomLevel(1.0)}
              className="px-1.5 py-1 rounded-md text-mini font-mono text-zinc-300 hover:text-white hover:bg-zinc-800 transition-colors"
              title="100% 1:1 Pixel Scale"
              aria-label="100% Zoom"
            >
              1×
            </button>

            <button
              type="button"
              onClick={() => setZoomLevel(4.0)}
              className="px-1.5 py-1 rounded-md text-mini font-mono text-zinc-300 hover:text-white hover:bg-zinc-800 transition-colors"
              title="400% Zoom"
              aria-label="400% Zoom"
            >
              4×
            </button>

            <button
              type="button"
              onClick={() => setZoomLevel(zoom * 1.4)}
              className="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
              title="Zoom In"
              aria-label="Zoom In"
            >
              <ZoomIn size={14} />
            </button>

            <button
              type="button"
              onClick={resetView}
              className="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
              title="Reset Pan & Zoom"
              aria-label="Reset Pan and Zoom"
            >
              <RotateCcw size={13} />
            </button>

            <button
              type="button"
              onClick={toggleFullscreen}
              className="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors ml-0.5"
              title="Toggle Fullscreen"
              aria-label="Toggle Fullscreen"
            >
              {isFullscreen ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
});

export default CinematicPreview;
