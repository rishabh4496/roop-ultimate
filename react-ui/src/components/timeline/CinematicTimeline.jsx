import React, {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from 'react';
import {
  ChevronLeft,
  ChevronRight,
  FastForward,
  Magnet,
  Pause,
  Play,
  Rewind,
  RotateCcw,
  Scissors,
  Sliders,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { formatSMPTE, useTimelineStore } from './timelineStore.js';
import { TimelineRenderer, TRACK_CONFIG } from './timelineRenderer.js';
import { useTimelineInteractions } from './useTimelineInteractions.js';

/**
 * <CinematicTimeline />
 * High-performance 60 FPS non-linear timeline component:
 *   - Virtualized HTML5 canvas with zero DOM elements per frame
 *   - Track 1: Ruler & SMPTE timecode (HH:MM:SS:FF) + backend scene-cut flags
 *   - Track 2: Master video filmstrip with interactive [I]/[O] clamps and hatch dimming
 *   - Track 3+: Identity sub-tracks with clustered face appearance intervals
 *   - Track 4: Parameter keyframe automation curves (editable Bézier/linear nodes)
 *   - Sub-frame scrubbing with zero A/V de-sync
 *   - Zustand transient store isolating 60 FPS playhead motion from React diffing
 *   - Snapping engine for cuts, in/out clamps, and keyframe points (Shift to bypass)
 *   - Keyboard controls: Space (Play/Pause), J-K-L (Shuttle), Arrows (1f step), I/O (Trim)
 */
export const CinematicTimeline = forwardRef(function CinematicTimeline(
  {
    frame: externalFrame,
    maxFrames: externalMaxFrames,
    fps: externalFps,
    inPoint: externalInPoint,
    outPoint: externalOutPoint,
    sceneCuts: externalSceneCuts,
    identityTracks: externalIdentityTracks,
    curves: externalCurves,
    onFrameChange,
    onInPointChange,
    onOutPointChange,
    className = '',
    style = {},
  },
  ref
) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const rendererRef = useRef(null);
  const lastTimeRef = useRef(performance.now());
  const rAFRef = useRef(null);

  // Local React state for coarse toolbar displays (debounced / low frequency)
  const [displayFrame, setDisplayFrame] = useState(externalFrame || 1);
  const [isPlayingDisplay, setIsPlayingDisplay] = useState(false);
  const [shuttleDisplay, setShuttleDisplay] = useState(0);
  const [zoomDisplay, setZoomDisplay] = useState(1.5);
  const [inPointDisplay, setInPointDisplay] = useState(externalInPoint || 1);
  const [outPointDisplay, setOutPointDisplay] = useState(externalOutPoint || 600);
  const [curvesExpanded, setCurvesExpanded] = useState(true);
  const [activeCurveId, setActiveCurveId] = useState('enhancer_fidelity');
  const [snappingActive, setSnappingActive] = useState(true);

  // Sync external props into Zustand store
  useEffect(() => {
    if (typeof externalMaxFrames === 'number' && externalMaxFrames > 0) {
      useTimelineStore.setState({ maxFrames: externalMaxFrames });
    }
  }, [externalMaxFrames]);

  useEffect(() => {
    if (typeof externalFps === 'number' && externalFps > 0) {
      useTimelineStore.setState({ fps: externalFps });
    }
  }, [externalFps]);

  useEffect(() => {
    if (typeof externalInPoint === 'number') {
      useTimelineStore.setState({ inPoint: externalInPoint });
      setInPointDisplay(externalInPoint);
    }
  }, [externalInPoint]);

  useEffect(() => {
    if (typeof externalOutPoint === 'number') {
      useTimelineStore.setState({ outPoint: externalOutPoint });
      setOutPointDisplay(externalOutPoint);
    }
  }, [externalOutPoint]);

  useEffect(() => {
    if (Array.isArray(externalSceneCuts)) {
      useTimelineStore.getState().setSceneCuts(externalSceneCuts);
    }
  }, [externalSceneCuts]);

  useEffect(() => {
    if (Array.isArray(externalIdentityTracks)) {
      useTimelineStore.getState().setIdentityTracks(externalIdentityTracks);
    }
  }, [externalIdentityTracks]);

  useEffect(() => {
    if (Array.isArray(externalCurves)) {
      useTimelineStore.setState({ curves: externalCurves });
    }
  }, [externalCurves]);

  // Sync external frame into store if it moved externally
  useEffect(() => {
    if (typeof externalFrame === 'number') {
      const cur = useTimelineStore.getState().frame;
      if (Math.abs(cur - externalFrame) > 0.05) {
        useTimelineStore.getState().setFrame(externalFrame, true);
        setDisplayFrame(Math.round(externalFrame));
      }
    }
  }, [externalFrame]);

  // Pointer & keyboard interaction hook
  const {
    handlePointerDown,
    handlePointerMove,
    handlePointerUp,
    handleWheel,
    handleKeyDown,
  } = useTimelineInteractions({
    canvasRef,
  });

  // Calculate dynamic canvas height based on active tracks
  const computeCanvasHeight = useCallback(() => {
    const store = useTimelineStore.getState();
    let totalH = TRACK_CONFIG.RULER_HEIGHT + TRACK_CONFIG.TRACK_GAP + TRACK_CONFIG.VIDEO_HEIGHT;
    totalH += TRACK_CONFIG.TRACK_GAP + store.identityTracks.length * (TRACK_CONFIG.IDENTITY_HEIGHT + TRACK_CONFIG.TRACK_GAP);
    if (store.curvesExpanded) {
      totalH += TRACK_CONFIG.CURVES_HEIGHT;
    }
    return totalH;
  }, []);

  // Initialize Canvas Renderer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const renderer = new TimelineRenderer(canvas);
    rendererRef.current = renderer;

    return () => {
      renderer.destroy();
      rendererRef.current = null;
    };
  }, []);

  // 60 FPS RequestAnimationFrame Render & Playback Loop
  useEffect(() => {
    let lastReportedFrame = -1;
    let lastReportedIn = -1;
    let lastReportedOut = -1;
    let lastUiSyncTime = performance.now();

    const loop = (now) => {
      const dt = (now - lastTimeRef.current) / 1000;
      lastTimeRef.current = now;

      const store = useTimelineStore.getState();

      // Handle Playback & Shuttle Velocity
      if (store.isPlaying && store.shuttleSpeed !== 0) {
        const deltaFrames = dt * store.fps * store.shuttleSpeed;
        let nextFrame = store.frame + deltaFrames;

        // Loop playback within in/out region if playing forward
        if (store.shuttleSpeed > 0 && nextFrame > store.outPoint) {
          nextFrame = store.inPoint;
        } else if (store.shuttleSpeed < 0 && nextFrame < store.inPoint) {
          nextFrame = store.outPoint;
        }

        store.setFrame(nextFrame, true);
      }

      // Draw canvas slice
      if (rendererRef.current) {
        rendererRef.current.render(useTimelineStore.getState());
      }

      // Throttle React UI updates to ~15-30 Hz to eliminate React commit churn
      if (now - lastUiSyncTime > 40) {
        const cur = useTimelineStore.getState();
        const roundedF = Math.round(cur.frame);

        if (roundedF !== lastReportedFrame) {
          setDisplayFrame(roundedF);
          onFrameChange?.(roundedF);
          lastReportedFrame = roundedF;
        }

        if (cur.inPoint !== lastReportedIn) {
          setInPointDisplay(cur.inPoint);
          onInPointChange?.(cur.inPoint);
          lastReportedIn = cur.inPoint;
        }

        if (cur.outPoint !== lastReportedOut) {
          setOutPointDisplay(cur.outPoint);
          onOutPointChange?.(cur.outPoint);
          lastReportedOut = cur.outPoint;
        }

        setIsPlayingDisplay(cur.isPlaying);
        setShuttleDisplay(cur.shuttleSpeed);
        setZoomDisplay(cur.zoom);
        setCurvesExpanded(cur.curvesExpanded);
        setActiveCurveId(cur.activeCurveId);
        setSnappingActive(cur.snappingEnabled);
        lastUiSyncTime = now;
      }

      rAFRef.current = requestAnimationFrame(loop);
    };

    lastTimeRef.current = performance.now();
    rAFRef.current = requestAnimationFrame(loop);

    return () => {
      if (rAFRef.current) {
        cancelAnimationFrame(rAFRef.current);
      }
    };
  }, [onFrameChange, onInPointChange, onOutPointChange]);

  // ResizeObserver for canvas dimensions
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const rect = entry.contentRect;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);

      const targetH = computeCanvasHeight();
      const w = Math.max(1, Math.round(rect.width));
      const h = targetH;

      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;

      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);

      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.scale(dpr, dpr);
      }

      useTimelineStore.getState().setDimensions(w, h);
    });

    ro.observe(container);
    return () => ro.disconnect();
  }, [computeCanvasHeight]);

  // Zoom to Fit whole clip on screen
  const handleZoomFit = useCallback(() => {
    const { canvasWidth, maxFrames, setZoom, setScrollLeft } = useTimelineStore.getState();
    const fitZoom = (canvasWidth - 40) / Math.max(1, maxFrames);
    setZoom(fitZoom);
    setScrollLeft(0);
  }, []);

  // Expose Imperative API
  useImperativeHandle(ref, () => ({
    setFrame: (f) => useTimelineStore.getState().setFrame(f, true),
    setInPoint: (f) => useTimelineStore.getState().setInPoint(f),
    setOutPoint: (f) => useTimelineStore.getState().setOutPoint(f),
    play: () => useTimelineStore.getState().setPlaying(true),
    pause: () => useTimelineStore.getState().setPlaying(false),
    togglePlay: () => useTimelineStore.getState().togglePlay(),
    shuttle: (dir) => useTimelineStore.getState().shuttle(dir),
    stepFrame: (delta) => useTimelineStore.getState().stepFrame(delta),
    zoomFit: handleZoomFit,
    getStore: () => useTimelineStore.getState(),
  }));

  const storeState = useTimelineStore.getState();
  const maxF = storeState.maxFrames;
  const fps = storeState.fps;
  const durationFrames = outPointDisplay - inPointDisplay + 1;

  return (
    <div
      ref={containerRef}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      className={`flex flex-col w-full bg-zinc-950 border border-zinc-800 rounded-xl overflow-hidden shadow-2xl font-sans text-zinc-200 select-none outline-none focus-visible:ring-1 focus-visible:ring-amber-500/40 ${className}`}
      style={style}
    >
      {/* ── Top Header & Transport Toolbar ─────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-2.5 px-4 bg-zinc-900/90 border-b border-zinc-800/80 backdrop-blur-md">
        {/* Left Cluster: Timecode Readout & Frame Counter */}
        <div className="flex items-center gap-3">
          <div className="flex items-baseline gap-2 font-mono">
            <span className="text-zinc-400 text-micro font-bold tracking-wider">SMPTE</span>
            <span className="text-zinc-100 font-bold text-plain tracking-wider bg-zinc-950 px-2 py-0.5 rounded border border-zinc-800">
              {formatSMPTE(displayFrame, fps)}
            </span>
          </div>

          <div className="flex items-baseline gap-1 font-mono text-mini text-zinc-400">
            <span className="text-zinc-100 font-semibold">#{displayFrame}</span>
            <span className="text-zinc-600">/</span>
            <span>{maxF}f</span>
          </div>

          {/* Active Shuttle Speed Indicator */}
          {shuttleDisplay !== 0 && (
            <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30 text-micro font-mono font-bold tracking-wide">
              {shuttleDisplay > 0 ? (
                <>
                  <FastForward size={11} />
                  <span>+{shuttleDisplay}× FWD</span>
                </>
              ) : (
                <>
                  <Rewind size={11} />
                  <span>{shuttleDisplay}× REV</span>
                </>
              )}
            </div>
          )}
        </div>

        {/* Center Cluster: Transport Shuttle (J-K-L) & Step Controls */}
        <div className="flex items-center gap-1 bg-zinc-950 p-1 rounded-lg border border-zinc-800/80">
          {/* Shuttle Reverse [J] */}
          <button
            type="button"
            onClick={() => useTimelineStore.getState().shuttle('reverse')}
            className={`p-1.5 rounded-md transition-colors ${
              shuttleDisplay < 0
                ? 'bg-amber-500 text-zinc-950 font-bold'
                : 'text-zinc-400 hover:text-white hover:bg-zinc-800'
            }`}
            title="Reverse Shuttle (J) — press multiple times for 1x, 2x, 4x, 8x"
            aria-label="Reverse Shuttle"
          >
            <Rewind size={14} />
          </button>

          {/* Step Back 1 Frame [←] */}
          <button
            type="button"
            onClick={() => useTimelineStore.getState().stepFrame(-1)}
            className="p-1.5 rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
            title="Step Backward 1 Frame (Left Arrow, Shift+Left for 10f)"
            aria-label="Step Backward One Frame"
          >
            <ChevronLeft size={14} />
          </button>

          {/* Play / Pause [Space] or [K] */}
          <button
            type="button"
            onClick={() => useTimelineStore.getState().togglePlay()}
            className="flex items-center justify-center w-8 h-8 rounded-md bg-amber-500 hover:bg-amber-400 text-zinc-950 shadow-md font-bold transition-transform active:scale-95"
            title="Play / Pause (Space or K)"
            aria-label={isPlayingDisplay ? 'Pause' : 'Play'}
          >
            {isPlayingDisplay ? <Pause size={15} /> : <Play size={15} className="ml-0.5" />}
          </button>

          {/* Step Forward 1 Frame [→] */}
          <button
            type="button"
            onClick={() => useTimelineStore.getState().stepFrame(1)}
            className="p-1.5 rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
            title="Step Forward 1 Frame (Right Arrow, Shift+Right for 10f)"
            aria-label="Step Forward One Frame"
          >
            <ChevronRight size={14} />
          </button>

          {/* Shuttle Forward [L] */}
          <button
            type="button"
            onClick={() => useTimelineStore.getState().shuttle('forward')}
            className={`p-1.5 rounded-md transition-colors ${
              shuttleDisplay > 0
                ? 'bg-amber-500 text-zinc-950 font-bold'
                : 'text-zinc-400 hover:text-white hover:bg-zinc-800'
            }`}
            title="Forward Shuttle (L) — press multiple times for 1x, 2x, 4x, 8x"
            aria-label="Forward Shuttle"
          >
            <FastForward size={14} />
          </button>
        </div>

        {/* Right Cluster: In/Out Clamps, Snapping, Curve Toggles & Zoom */}
        <div className="flex items-center gap-3">
          {/* In / Out Range Indicators */}
          <div className="flex items-center gap-1.5 text-mini font-mono">
            <button
              type="button"
              onClick={() => useTimelineStore.getState().setInPointToCurrent()}
              className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-zinc-950 border border-zinc-800 hover:border-emerald-500/50 text-zinc-300 hover:text-emerald-400 transition-colors"
              title="Set In-Point to Current Frame (I)"
              aria-label="Set In Point"
            >
              <span className="text-emerald-400 font-bold">IN</span>
              <span>{inPointDisplay}</span>
            </button>

            <button
              type="button"
              onClick={() => useTimelineStore.getState().setOutPointToCurrent()}
              className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-zinc-950 border border-zinc-800 hover:border-red-500/50 text-zinc-300 hover:text-red-400 transition-colors"
              title="Set Out-Point to Current Frame (O)"
              aria-label="Set Out Point"
            >
              <span className="text-red-400 font-bold">OUT</span>
              <span>{outPointDisplay}</span>
            </button>

            <span className="text-zinc-500 text-micro">({durationFrames}f)</span>

            <button
              type="button"
              onClick={() => useTimelineStore.getState().clearInOutRange()}
              className="p-1 text-zinc-500 hover:text-zinc-300 transition-colors"
              title="Reset In/Out Clamps"
              aria-label="Reset In and Out Clamps"
            >
              <RotateCcw size={11} />
            </button>
          </div>

          <div className="w-[1px] h-4 bg-zinc-800" />

          {/* Magnetic Snapping Toggle */}
          <button
            type="button"
            onClick={() =>
              useTimelineStore.setState((s) => ({ snappingEnabled: !s.snappingEnabled }))
            }
            className={`p-1.5 rounded-md border transition-colors ${
              snappingActive
                ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                : 'bg-zinc-950 text-zinc-500 border-zinc-800 hover:text-zinc-300'
            }`}
            title="Toggle Snapping to Cuts & Keyframes (Hold Shift while scrubbing to bypass)"
            aria-label="Toggle Snapping Engine"
          >
            <Magnet size={14} />
          </button>

          {/* Automation Curves Lane Toggle */}
          <button
            type="button"
            onClick={() => useTimelineStore.getState().toggleCurves()}
            className={`flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium border transition-colors ${
              curvesExpanded
                ? 'bg-amber-500/15 text-amber-300 border-amber-500/30'
                : 'bg-zinc-950 text-zinc-400 border-zinc-800 hover:text-white'
            }`}
            title="Toggle Automation Curves Track"
            aria-label="Toggle Automation Curves Track"
          >
            <Sliders size={13} />
            <span>Curves</span>
          </button>

          {/* Active Curve Selector */}
          {curvesExpanded && storeState.curves?.length > 1 && (
            <select
              value={activeCurveId}
              onChange={(e) => {
                setActiveCurveId(e.target.value);
                useTimelineStore.getState().setActiveCurveId(e.target.value);
              }}
              className="bg-zinc-950 border border-zinc-800 rounded px-1.5 py-0.5 text-micro font-medium text-zinc-300 outline-none"
              aria-label="Active Automation Curve"
            >
              {storeState.curves.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
          )}

          {/* Zoom Slider & Fit */}
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => useTimelineStore.getState().setZoom(zoomDisplay / 1.4)}
              className="p-1 rounded text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
              title="Zoom Out"
              aria-label="Zoom Out"
            >
              <ZoomOut size={13} />
            </button>

            <button
              type="button"
              onClick={handleZoomFit}
              className="px-1.5 py-0.5 rounded text-micro font-mono bg-zinc-950 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 hover:text-white transition-colors"
              title="Fit Entire Timeline into View"
              aria-label="Fit Timeline View"
            >
              Fit
            </button>

            <button
              type="button"
              onClick={() => useTimelineStore.getState().setZoom(zoomDisplay * 1.4)}
              className="p-1 rounded text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
              title="Zoom In"
              aria-label="Zoom In"
            >
              <ZoomIn size={13} />
            </button>
          </div>
        </div>
      </div>

      {/* ── Virtualized 60 FPS Canvas Viewport ─────────────────────────────── */}
      <div className="relative w-full overflow-hidden bg-zinc-950">
        <canvas
          ref={canvasRef}
          tabIndex={0}
          className="block w-full cursor-pointer outline-none touch-none"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          onWheel={handleWheel}
          onKeyDown={handleKeyDown}
        />
      </div>

      {/* ── Footer Status & Track Legends ──────────────────────────────────── */}
      <div className="flex items-center justify-between px-4 py-1.5 bg-zinc-900/60 border-t border-zinc-800/60 text-micro text-zinc-400">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5">
            <Scissors size={12} className="text-red-400" />
            <span>Scene Cuts</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2 rounded-sm bg-sky-400/50 border border-sky-400" />
            <span>Identity Presence Blocks</span>
          </span>
          {curvesExpanded && (
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-0.5 bg-amber-400" />
              <span>Bézier Automation Curve</span>
            </span>
          )}
        </div>

        <div className="flex items-center gap-3 font-mono text-zinc-500">
          <span>Scrub: Click / Drag</span>
          <span>•</span>
          <span>Shuttle: J-K-L</span>
          <span>•</span>
          <span>Trim: [I] / [O]</span>
          <span>•</span>
          <span>Bypass Snap: Hold Shift</span>
        </div>
      </div>
    </div>
  );
});

export default CinematicTimeline;
