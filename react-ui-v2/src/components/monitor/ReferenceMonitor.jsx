import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  createReferenceChannel,
  closeReferenceChannel,
  MESSAGE_TYPES,
} from './referenceChannel.js';
import { drawZebraOverlay } from '../faceswap/falseColorZebra.js';
import { eventMatches } from '../../store/shortcutStore';

/**
 * Stage 19: Pop-Out Clean Feed Reference Monitor (#/monitor).
 *
 * A dedicated chrome-free popup display that renders live swapped frames,
 * split-wipe comparison coordinates, and Stage 12/18 LUT and Zebra overlays
 * synchronized across the BroadcastChannel bridge with zero REST requests.
 */
export default function ReferenceMonitor() {
  const canvasRef = useRef(null);
  const channelRef = useRef(null);
  const lastFrameRef = useRef(null);
  const hudTimerRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [hudVisible, setHudVisible] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);
  const [stats, setStats] = useState({
    width: 0,
    height: 0,
    fps: 0,
    frameIndex: 0,
    latencyMs: 0,
  });
  const [transport, setTransport] = useState({
    isPlaying: false,
    currentFrame: 0,
    totalFrames: 0,
  });
  const [compare, setCompare] = useState({
    mode: 'slider',
    wipe: 50,
    compareDir: 'vertical',
    wipeAngle: 0,
  });
  const [grading, setGrading] = useState({
    lut_opacity: 1.0,
    color_vibrance: 0.0,
    zebraActive: false,
  });

  const fpsCounterRef = useRef({ count: 0, lastTime: performance.now(), currentFps: 0 });

  // ── Canvas Paint Loop ───────────────────────────────────────────────────
  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const frame = lastFrameRef.current;
    const cw = canvas.width;
    const ch = canvas.height;

    ctx.fillStyle = '#05070a';
    ctx.fillRect(0, 0, cw, ch);

    if (!frame) {
      // Idle standby state
      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.font = '14px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(
        connected ? 'Awaiting clean feed broadcast...' : 'Reference Monitor Offline (Opening channel...)',
        cw / 2,
        ch / 2
      );
      return;
    }

    const fw = frame.width || frame.videoWidth || frame.naturalWidth || 1920;
    const fh = frame.height || frame.videoHeight || frame.naturalHeight || 1080;

    // Aspect contain calculation (letterbox/pillarbox)
    const scale = Math.min(cw / fw, ch / fh);
    const dw = fw * scale;
    const dh = fh * scale;
    const dx = (cw - dw) / 2;
    const dy = (ch - dh) / 2;

    try {
      ctx.drawImage(frame, dx, dy, dw, dh);
    } catch {
      // frame might be detached
    }

    // Split wipe overlay line if slider active
    if (compare.mode === 'slider' && compare.wipe > 0 && compare.wipe < 100) {
      ctx.save();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.75)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      if (compare.compareDir === 'vertical') {
        const lineX = dx + (dw * compare.wipe) / 100;
        ctx.moveTo(lineX, dy);
        ctx.lineTo(lineX, dy + dh);
      } else {
        const lineY = dy + (dh * compare.wipe) / 100;
        ctx.moveTo(dx, lineY);
        ctx.lineTo(dx + dw, lineY);
      }
      ctx.stroke();
      ctx.restore();
    }

    // Stage 12 / 18 Zebra Overlay
    if (grading.zebraActive) {
      try {
        drawZebraOverlay(ctx, cw, ch, null, 'zebra');
      } catch {
        // ignore
      }
    }
  }, [connected, grading.zebraActive, compare]);

  // Handle window resizing
  const updateCanvasSize = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const w = window.innerWidth;
    const h = window.innerHeight;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    paint();
  }, [paint]);

  useEffect(() => {
    updateCanvasSize();
    window.addEventListener('resize', updateCanvasSize);
    return () => window.removeEventListener('resize', updateCanvasSize);
  }, [updateCanvasSize]);

  // ── BroadcastChannel Subscription ───────────────────────────────────────
  useEffect(() => {
    const channel = createReferenceChannel((data) => {
      if (!data || !data.type) return;

      switch (data.type) {
        case MESSAGE_TYPES.FRAME: {
          const payload = data.payload || {};
          if (payload.frame) {
            lastFrameRef.current = payload.frame;
            setConnected(true);

            // FPS calculation
            const now = performance.now();
            const fc = fpsCounterRef.current;
            fc.count += 1;
            if (now - fc.lastTime >= 1000) {
              fc.currentFps = Math.round((fc.count * 1000) / (now - fc.lastTime));
              fc.count = 0;
              fc.lastTime = now;
            }

            const latency = payload.timestamp ? Math.max(0, Math.round(now - payload.timestamp)) : 0;
            setStats({
              width: payload.width || 0,
              height: payload.height || 0,
              fps: fc.currentFps,
              frameIndex: payload.frameIndex ?? -1,
              latencyMs: latency,
            });

            requestAnimationFrame(paint);
          }
          break;
        }
        case MESSAGE_TYPES.TRANSPORT: {
          setTransport(data.payload || {});
          break;
        }
        case MESSAGE_TYPES.COMPARE: {
          setCompare(data.payload || {});
          requestAnimationFrame(paint);
          break;
        }
        case MESSAGE_TYPES.GRADING: {
          setGrading(data.payload || {});
          requestAnimationFrame(paint);
          break;
        }
        case MESSAGE_TYPES.PING: {
          setConnected(true);
          try {
            channel.postMessage({ type: MESSAGE_TYPES.PONG });
          } catch {
            // ignore
          }
          break;
        }
        case MESSAGE_TYPES.PONG: {
          setConnected(true);
          break;
        }
        case MESSAGE_TYPES.TEARDOWN: {
          setConnected(false);
          lastFrameRef.current = null;
          requestAnimationFrame(paint);
          break;
        }
        default:
          break;
      }
    });

    channelRef.current = channel;
    try {
      channel.postMessage({ type: MESSAGE_TYPES.PING });
    } catch {
      // ignore
    }

    // Deterministic channel teardown on unmount
    return () => {
      closeReferenceChannel(channel);
      channelRef.current = null;
    };
  }, [paint]);

  // ── Auto-Hiding HUD ─────────────────────────────────────────────────────
  const showHudBriefly = useCallback(() => {
    setHudVisible(true);
    if (hudTimerRef.current) clearTimeout(hudTimerRef.current);
    hudTimerRef.current = setTimeout(() => {
      setHudVisible(false);
    }, 2500);
  }, []);

  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
      setFullscreen(true);
    } else {
      document.exitFullscreen().catch(() => {});
      setFullscreen(false);
    }
  }, []);

  useEffect(() => {
    const handleKey = (e) => {
      showHudBriefly();
      if (eventMatches(e, 'monitor.fullscreen')) {
        e.preventDefault();
        toggleFullscreen();
      }
    };
    const handleFullscreenChange = () => {
      setFullscreen(!!document.fullscreenElement);
    };
    window.addEventListener('keydown', handleKey);
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => {
      window.removeEventListener('keydown', handleKey);
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, [showHudBriefly, toggleFullscreen]);

  return (
    <div
      className="fixed inset-0 bg-black overflow-hidden select-none cursor-default"
      onMouseMove={showHudBriefly}
      onDoubleClick={toggleFullscreen}
    >
      {/* ── Direct Video Presentation Surface ── */}
      <canvas ref={canvasRef} className="block w-full h-full object-contain" />

      {/* ── Auto-Hiding Studio HUD Bar ── */}
      <div
        className={`absolute bottom-4 left-4 right-4 flex items-center justify-between pointer-events-none transition-opacity duration-300 ${
          hudVisible ? 'opacity-100' : 'opacity-0'
        }`}
      >
        {/* Left Status Pill */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-black/75 backdrop-blur-md border border-white/10 text-mini pointer-events-auto shadow-2xl">
          <span
            className={`w-2 h-2 rounded-full ${
              connected ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'
            }`}
          />
          <span className="font-bold text-white/90">
            {connected ? 'LIVE CLEAN FEED' : 'CONNECTING...'}
          </span>
          {stats.width > 0 && (
            <span className="text-white/40 font-mono text-nano">
              {stats.width}×{stats.height} @ {stats.fps} FPS
            </span>
          )}
          {stats.latencyMs > 0 && (
            <span className="text-amber-300/80 font-mono text-nano">
              {stats.latencyMs}ms
            </span>
          )}
          {transport.totalFrames > 0 && (
            <span className="text-white/50 font-mono text-nano">
              F#{transport.currentFrame}/{transport.totalFrames}
            </span>
          )}
          {grading.zebraActive && (
            <span className="px-1 py-0.5 rounded bg-rose-500/20 text-rose-300 text-micro font-bold border border-rose-500/30">
              ZEBRA
            </span>
          )}
        </div>

        {/* Right Floating Controls */}
        <div className="flex items-center gap-2 pointer-events-auto">
          <button
            type="button"
            onClick={toggleFullscreen}
            className="px-3 py-1.5 rounded-xl bg-black/75 backdrop-blur-md border border-white/10 hover:border-white/30 text-white/80 hover:text-white text-mini font-semibold transition-all shadow-xl"
            title="Toggle Fullscreen (F)"
            aria-label={fullscreen ? 'Exit Fullscreen' : 'Enter Fullscreen'}
          >
            {fullscreen ? 'Exit Fullscreen' : 'Fullscreen (F)'}
          </button>
          <button
            type="button"
            onClick={() => window.close()}
            className="px-3 py-1.5 rounded-xl bg-black/75 backdrop-blur-md border border-white/10 hover:border-rose-500/50 hover:bg-rose-500/20 text-white/60 hover:text-rose-300 text-mini font-semibold transition-all shadow-xl"
            title="Close Reference Monitor"
            aria-label="Close Reference Monitor"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
