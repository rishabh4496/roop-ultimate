import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

export type CompareMode = "split" | "side-by-side";

export interface MediaSource {
  kind: "video" | "image";
  src: string;
}

export interface DualCanvasPlayerProps {
  /** The original target. */
  left: MediaSource;
  /** Swapped output (video/image), or a single-frame preview image; null = nothing yet. */
  right: MediaSource | null;
  fps: number;
  frames: number;
  initialMode?: CompareMode;
  /**
   * Called on every scrub step while paused (the caller debounces: see
   * ``usePreview``, 100 ms, aborting stale requests).
   */
  onScrub?: (frame: number) => void;
  leftLabel?: string;
  rightLabel?: string;
  /** Shown over the top-right of the canvas, e.g. preview latency. */
  badge?: ReactNode;
  /** Dim the output side while a newer frame is being rendered. */
  pending?: boolean;
}

type Drawable = HTMLVideoElement | HTMLImageElement;

const ready = (el: Drawable | null): el is Drawable =>
  el != null && (el instanceof HTMLVideoElement ? el.readyState >= 2 : el.complete && el.naturalWidth > 0);

const sizeOf = (el: Drawable) =>
  el instanceof HTMLVideoElement
    ? { w: el.videoWidth, h: el.videoHeight }
    : { w: el.naturalWidth, h: el.naturalHeight };

function context(canvas: HTMLCanvasElement | null): CanvasRenderingContext2D | null {
  if (!canvas) return null;
  try {
    return canvas.getContext("2d"); // null in jsdom / when unsupported
  } catch {
    return null;
  }
}

/**
 * Dual-canvas comparison player: the original target on the left, the swapped
 * output on the right, compared through a draggable split (or side by side).
 * Both media elements stay hidden; frames are drawn to canvas so split mode can
 * clip the output at an arbitrary divider. The output video (when present) is
 * the clock; the original follows it and is re-synced whenever the two drift
 * more than 1.5 frames apart. While paused, the timeline scrubs the original
 * and reports each frame through ``onScrub`` so the caller can fetch a GPU
 * preview of it.
 */
export function DualCanvasPlayer(props: DualCanvasPlayerProps) {
  const { left, right, fps, frames } = props;
  const [mode, setMode] = useState<CompareMode>(props.initialMode ?? "split");
  const [divider, setDivider] = useState(0.5);
  const [playing, setPlaying] = useState(false);
  const [frame, setFrame] = useState(0);
  const leftRef = useRef<Drawable | null>(null);
  const rightRef = useRef<Drawable | null>(null);
  const splitCanvas = useRef<HTMLCanvasElement>(null);
  const leftCanvas = useRef<HTMLCanvasElement>(null);
  const rightCanvas = useRef<HTMLCanvasElement>(null);
  const dragging = useRef(false);
  const raf = useRef<number | null>(null);

  const leftIsVideo = left.kind === "video";
  const rightIsVideo = right?.kind === "video";
  const master = (): HTMLVideoElement | null => {
    if (rightIsVideo && rightRef.current instanceof HTMLVideoElement) return rightRef.current;
    if (leftIsVideo && leftRef.current instanceof HTMLVideoElement) return leftRef.current;
    return null;
  };
  const slave = (): HTMLVideoElement | null =>
    rightIsVideo && leftIsVideo && leftRef.current instanceof HTMLVideoElement ? leftRef.current : null;

  const draw = useCallback(() => {
    const l = leftRef.current;
    const r = rightRef.current;
    if (mode === "split") {
      const canvas = splitCanvas.current;
      const ctx = context(canvas);
      const base = ready(l) ? l : ready(r) ? r : null;
      if (!canvas || !ctx || !base) return;
      const { w, h } = sizeOf(base);
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      if (ready(l)) ctx.drawImage(l, 0, 0, w, h);
      if (ready(r)) {
        const x = Math.round(w * divider);
        ctx.save();
        ctx.beginPath();
        ctx.rect(x, 0, w - x, h);
        ctx.clip();
        ctx.drawImage(r, 0, 0, w, h);
        ctx.restore();
        ctx.fillStyle = "rgba(255,255,255,0.9)";
        ctx.fillRect(x - 1, 0, 2, h);
      }
      return;
    }
    for (const [el, canvas] of [
      [l, leftCanvas.current],
      [r, rightCanvas.current],
    ] as const) {
      const ctx = context(canvas);
      if (!canvas || !ctx || !ready(el)) continue;
      const { w, h } = sizeOf(el);
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      ctx.drawImage(el, 0, 0, w, h);
    }
  }, [mode, divider]);

  const sync = useCallback(() => {
    const m = master();
    const s = slave();
    if (m && s && Math.abs(s.currentTime - m.currentTime) > 1.5 / Math.max(fps, 1)) {
      s.currentTime = m.currentTime;
    }
    if (m) setFrame(Math.min(frames - 1, Math.round(m.currentTime * fps)));
  }, [fps, frames, rightIsVideo, leftIsVideo]);

  // Redraw on any visual change.
  useEffect(() => {
    draw();
  }, [draw, right?.src, left.src]);

  // Animation loop while playing.
  useEffect(() => {
    if (!playing) return;
    const tick = () => {
      sync();
      draw();
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current != null) cancelAnimationFrame(raf.current);
    };
  }, [playing, draw, sync]);

  const togglePlay = async () => {
    const m = master();
    if (!m) return;
    if (m.paused) {
      const s = slave();
      if (s) s.currentTime = m.currentTime;
      try {
        await Promise.all([m.play(), s?.play()]);
        setPlaying(true);
      } catch {
        setPlaying(false);
      }
    } else {
      m.pause();
      slave()?.pause();
      setPlaying(false);
      draw();
    }
  };

  const seek = (next: number) => {
    const f = Math.max(0, Math.min(frames - 1, next));
    setFrame(f);
    // Aim at the middle of the frame so decoders do not land on the previous one.
    const t = (f + 0.5) / Math.max(fps, 1);
    for (const el of [leftRef.current, rightRef.current]) {
      if (el instanceof HTMLVideoElement) el.currentTime = t;
    }
    if (!playing) props.onScrub?.(f);
  };

  const onPointer = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (mode !== "split" || (!dragging.current && e.type === "pointermove")) return;
    const box = e.currentTarget.getBoundingClientRect();
    if (box.width > 0) setDivider(Math.max(0, Math.min(1, (e.clientX - box.left) / box.width)));
  };

  const mediaHandlers = {
    onLoadedData: () => draw(),
    onLoad: () => draw(),
    onSeeked: () => draw(),
    onEnded: () => setPlaying(false),
  };
  const hasClock = leftIsVideo || rightIsVideo;
  const canvasClass = "w-full rounded bg-black";

  return (
    <section aria-label="Comparison player" className="flex flex-col gap-2">
      <div className="hidden">
        {left.kind === "video" ? (
          <video
            ref={(el) => {
              leftRef.current = el;
            }}
            src={left.src}
            muted
            playsInline
            preload="auto"
            crossOrigin="anonymous"
            data-testid="left-media"
            {...mediaHandlers}
          />
        ) : (
          <img
            ref={(el) => {
              leftRef.current = el;
            }}
            src={left.src}
            alt=""
            crossOrigin="anonymous"
            data-testid="left-media"
            {...mediaHandlers}
          />
        )}
        {right &&
          (right.kind === "video" ? (
            <video
              key={right.src}
              ref={(el) => {
                rightRef.current = el;
              }}
              src={right.src}
              playsInline
              preload="auto"
              crossOrigin="anonymous"
              data-testid="right-media"
              {...mediaHandlers}
            />
          ) : (
            <img
              key={right.src}
              ref={(el) => {
                rightRef.current = el;
              }}
              src={right.src}
              alt=""
              crossOrigin="anonymous"
              data-testid="right-media"
              {...mediaHandlers}
            />
          ))}
      </div>

      <div className="flex items-center justify-between text-xs text-zinc-400">
        <span>
          {props.leftLabel ?? "Original"} ◂ ▸ {right ? (props.rightLabel ?? "Output") : "(no output yet)"}
        </span>
        <div role="group" aria-label="Comparison mode" className="flex gap-1">
          {(["split", "side-by-side"] as const).map((m) => (
            <button
              key={m}
              type="button"
              aria-pressed={mode === m}
              onClick={() => setMode(m)}
              className={`rounded px-2 py-0.5 ${mode === m ? "bg-sky-700 text-white" : "bg-zinc-800 hover:bg-zinc-700"}`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <div className="relative">
      {props.badge && (
        <div className="pointer-events-none absolute right-2 top-2 z-10" data-testid="player-badge">
          {props.badge}
        </div>
      )}
      {mode === "split" ? (
        <canvas
          ref={splitCanvas}
          data-testid="split-canvas"
          className={`${canvasClass} cursor-ew-resize touch-none`}
          onPointerDown={(e) => {
            dragging.current = true;
            e.currentTarget.setPointerCapture?.(e.pointerId);
            onPointer(e);
          }}
          onPointerMove={onPointer}
          onPointerUp={() => {
            dragging.current = false;
          }}
        />
      ) : (
        <div className="grid grid-cols-2 gap-2">
          <canvas ref={leftCanvas} data-testid="left-canvas" className={canvasClass} />
          <canvas
            ref={rightCanvas}
            data-testid="right-canvas"
            className={`${canvasClass} ${props.pending ? "opacity-70" : ""}`}
          />
        </div>
      )}
      </div>

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={togglePlay}
          disabled={!hasClock}
          className="w-16 rounded bg-zinc-800 px-2 py-1 text-sm hover:bg-zinc-700 disabled:opacity-40"
        >
          {playing ? "Pause" : "Play"}
        </button>
        <input
          type="range"
          aria-label="Frame"
          min={0}
          max={Math.max(frames - 1, 0)}
          step={1}
          value={frame}
          disabled={frames <= 1}
          onChange={(e) => seek(Number(e.target.value))}
          className="flex-1 accent-sky-500"
        />
        <span className="w-24 text-right font-mono text-xs text-zinc-400" data-testid="frame-label">
          {frame + 1} / {Math.max(frames, 1)}
        </span>
      </div>
      {mode === "split" && right && (
        <label className="flex items-center gap-2 text-xs text-zinc-400">
          Split
          <input
            type="range"
            aria-label="Split position"
            min={0}
            max={100}
            value={Math.round(divider * 100)}
            onChange={(e) => setDivider(Number(e.target.value) / 100)}
            className="flex-1 accent-sky-500"
          />
        </label>
      )}
    </section>
  );
}
