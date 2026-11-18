/**
 * clientMedia.ts: the same lane inputs the server provides, built in the browser from a local File.
 *
 * Extras never uploads the clip until Apply, so there is no /api/target to ask for stills or peaks.
 * What the browser can do instead, and what each piece is honest about:
 *
 *  - Stills: a detached <video> seeks to each frame's time and a canvas copies it out. One seek at a
 *    time (a single decoder), so a strip fills in over a second or two rather than at once.
 *  - Waveform: Web Audio's decodeAudioData reads the audio stream and the peaks are max-pooled from
 *    the PCM. The whole file is read into memory, so a file over WAVEFORM_MAX_BYTES is not decoded
 *    (state 'none' with a reason) and a codec the browser cannot decode ends in state 'error'.
 *  - Frame rate: the browser does not expose a video's frame rate. requestVideoFrameCallback gives
 *    the presentation time of each frame as it plays, from which the rate is read and snapped to the
 *    nearest standard rate. Where that API is missing the rate is assumed 30 and flagged `assumed`.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { filmstripSlots, type FilmSlot } from './filmstripModel';
import type { FrameFetcher } from './useScrubThumbnail';

export const WAVEFORM_MAX_BYTES = 400 * 1024 * 1024;
export const STANDARD_RATES = [15, 23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 120] as const;

/** Snap a measured rate to the nearest standard one when it is within 1.5%; otherwise keep it rounded. */
export function snapFps(measured: number): number {
  if (!Number.isFinite(measured) || measured <= 0) return 30;
  let best: number = STANDARD_RATES[0];
  for (const r of STANDARD_RATES) if (Math.abs(r - measured) < Math.abs(best - measured)) best = r;
  return Math.abs(best - measured) / best <= 0.015 ? best : Math.round(measured * 100) / 100;
}

/**
 * Frame rate from the presentation timestamps of the frames a player showed.
 *
 * The browser does not promise to show EVERY frame (a 60 Hz display skips frames of fast playback, a busy
 * main thread drops some), so successive samples are one frame apart OR a whole number of frames apart. For
 * constant-rate video the frame interval is therefore the SMALLEST gap, and the others are multiples of it.
 * When most gaps are not multiples of the smallest one the clip is variable-rate and the median gap is the
 * honest "typical" rate. (Reading the median directly, as before, read a skipped-frame stream as half the rate.)
 */
export function fpsFromTimestamps(times: ReadonlyArray<number>): number | null {
  const gaps: number[] = [];
  for (let i = 1; i < times.length; i++) {
    const dt = times[i] - times[i - 1];
    if (dt > 0.002 && dt < 0.5) gaps.push(dt);
  }
  if (gaps.length < 4) return null;
  const smallest = Math.min(...gaps);
  let sumGap = 0;
  let sumK = 0;
  let wholeCount = 0;
  for (const g of gaps) {
    const k = Math.round(g / smallest);
    if (k >= 1 && Math.abs(g - k * smallest) <= 0.12 * smallest) {
      sumGap += g;
      sumK += k;
      wholeCount++;
    }
  }
  // Constant rate: the interval is the total time over the total number of frames it spanned. Averaging
  // (rather than reading the smallest gap) keeps a millisecond of timestamp jitter from skewing the rate.
  if (wholeCount >= 0.8 * gaps.length) return snapFps(sumK / sumGap);
  const sorted = [...gaps].sort((a, b) => a - b);
  return snapFps(1 / sorted[sorted.length >> 1]);
}

/**
 * Measure a clip's frame rate WITHOUT touching the visible player: a hidden 2 px element plays the same file
 * muted at 2x and reports each presented frame's media time. This answers within a fraction of a second whether
 * or not the user's player is playing or paused (the in-player estimate only runs while it plays, so pausing
 * early left the rate assumed and every frame count approximate). Resolves null when the API is missing, the
 * clip will not play, or too few consecutive frames were seen (the caller keeps its fallback).
 */
export function probeFps(url: string, { samples = 20, timeoutMs = 5000 }: { samples?: number; timeoutMs?: number } = {}): Promise<number | null> {
  return new Promise((resolve) => {
    if (typeof document === 'undefined') { resolve(null); return; }
    const video = document.createElement('video') as HTMLVideoElement & {
      requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number; presentedFrames: number }) => void) => number;
    };
    if (typeof video.requestVideoFrameCallback !== 'function') { resolve(null); return; }
    video.muted = true;
    video.playsInline = true;
    video.preload = 'auto';
    Object.assign(video.style, { position: 'fixed', left: '0', top: '0', width: '2px', height: '2px', opacity: '0.01', pointerEvents: 'none', zIndex: '-1' });
    video.setAttribute('aria-hidden', 'true');
    video.src = url;
    document.body.appendChild(video);

    const times: number[] = [];
    let done = false;
    const finish = (value: number | null) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      video.pause();
      video.removeAttribute('src');
      video.load();
      video.remove();
      resolve(value);
    };
    const timer = setTimeout(() => finish(null), timeoutMs);
    const onFrame = (_now: number, meta: { mediaTime: number; presentedFrames: number }) => {
      if (done) return;
      times.push(meta.mediaTime);
      if (times.length >= samples) {
        const est = fpsFromTimestamps(times);
        if (est) { finish(est); return; }
        times.length = 0;
      }
      video.requestVideoFrameCallback!(onFrame);
    };
    video.addEventListener('loadeddata', () => {
      // 1x: at faster rates a 60 Hz display shows only some frames (every other one at 2x), which is
      // survivable (see fpsFromTimestamps) but needs more samples; 1x shows every frame of a normal clip.
      video.playbackRate = 1;
      video.play().then(() => video.requestVideoFrameCallback!(onFrame)).catch(() => finish(null));
    }, { once: true });
    video.addEventListener('error', () => finish(null), { once: true });
  });
}

// ── frame grabber ─────────────────────────────────────────────────────────────

export interface FrameGrabber {
  grab: (frame: number, width: number, signal?: AbortSignal) => Promise<ImageBitmap | null>;
  dispose: () => void;
}

export function createFrameGrabber(url: string, getFps: () => number): FrameGrabber {
  const video = document.createElement('video');
  video.muted = true;
  video.preload = 'auto';
  video.playsInline = true;
  video.src = url;
  let chain: Promise<unknown> = Promise.resolve();
  let disposed = false;

  const seekTo = (t: number) =>
    new Promise<boolean>((resolve) => {
      const done = (ok: boolean) => {
        video.removeEventListener('seeked', onSeeked);
        clearTimeout(timer);
        resolve(ok);
      };
      const onSeeked = () => done(true);
      const timer = setTimeout(() => done(false), 4000);
      video.addEventListener('seeked', onSeeked);
      video.currentTime = t;
    });

  const grabOne = async (frame: number, width: number, signal?: AbortSignal): Promise<ImageBitmap | null> => {
    if (disposed || signal?.aborted) return null;
    if (video.readyState < 1) {
      await new Promise<void>((resolve) => {
        const ok = () => resolve();
        video.addEventListener('loadedmetadata', ok, { once: true });
        setTimeout(ok, 4000);
      });
    }
    if (!video.videoWidth) return null;
    const fps = Math.max(1, getFps());
    // the middle of the frame's display interval, so the decoder lands on THIS frame, not the previous one
    const t = Math.min(Math.max(0, (video.duration || 0) - 0.001), (frame + 0.5) / fps);
    if (!(await seekTo(t)) || disposed || signal?.aborted) return null;
    const w = Math.max(16, Math.round(width));
    const h = Math.max(9, Math.round((w * video.videoHeight) / video.videoWidth));
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, w, h);
    return createImageBitmap(canvas);
  };

  return {
    grab: (frame, width, signal) => {
      const next = chain.then(() => grabOne(frame, width, signal)).catch(() => null);
      chain = next;
      return next as Promise<ImageBitmap | null>;
    },
    dispose: () => {
      disposed = true;
      video.removeAttribute('src');
      video.load();
    },
  };
}

export function useFrameGrabber(url: string | null | undefined, fps: number): FrameGrabber | null {
  const fpsRef = useRef(fps);
  fpsRef.current = fps;
  const grabber = useMemo(() => (url ? createFrameGrabber(url, () => fpsRef.current) : null), [url]);
  useEffect(() => () => grabber?.dispose(), [grabber]);
  return grabber;
}

export function grabberFetcher(grabber: FrameGrabber | null): FrameFetcher | null {
  return grabber ? (frame, signal) => grabber.grab(frame, 240, signal) : null;
}

// ── filmstrip ─────────────────────────────────────────────────────────────────

export interface ClientFilmstripOptions {
  grabber: FrameGrabber | null;
  mediaKey: string;
  viewStart: number;
  viewEnd: number;
  maxFrames: number;
  trackWidth: number;
  enabled?: boolean;
}

export function useClientFilmstrip({ grabber, mediaKey, viewStart, viewEnd, maxFrames, trackWidth, enabled = true }: ClientFilmstripOptions) {
  const cacheRef = useRef<Map<number, ImageBitmap>>(new Map());
  const [version, setVersion] = useState(0);
  const slots = useMemo<FilmSlot[]>(() => filmstripSlots({ viewStart, viewEnd, widthPx: trackWidth, maxFrames }), [viewStart, viewEnd, trackWidth, maxFrames]);

  useEffect(() => {
    const cache = cacheRef.current;
    return () => {
      for (const b of cache.values()) b.close?.();
      cache.clear();
    };
  }, [mediaKey]);

  useEffect(() => {
    if (!enabled || !grabber || slots.length === 0) return undefined;
    const ctrl = new AbortController();
    const wanted = slots.map((s) => s.frame).filter((f) => !cacheRef.current.has(f));
    if (wanted.length === 0) return undefined;
    (async () => {
      for (const frame of wanted) {
        if (ctrl.signal.aborted) return;
        const bmp = await grabber.grab(frame, 160, ctrl.signal);
        if (!bmp || ctrl.signal.aborted) {
          bmp?.close?.();
          continue;
        }
        cacheRef.current.set(frame, bmp);
        while (cacheRef.current.size > 150) {
          const oldest = cacheRef.current.keys().next().value;
          if (oldest === undefined) break;
          cacheRef.current.get(oldest)?.close?.();
          cacheRef.current.delete(oldest);
        }
        setVersion((v) => v + 1);
      }
    })();
    return () => ctrl.abort();
  }, [slots, grabber, enabled]);

  const getThumbnail = useCallback((frame: number, reach = 10): ImageBitmap | null => {
    const exact = cacheRef.current.get(frame);
    if (exact) return exact;
    if (reach <= 0) return null;
    let best: ImageBitmap | null = null;
    let min = reach + 1;
    for (const [f, bmp] of cacheRef.current) {
      const d = Math.abs(f - frame);
      if (d < min) { min = d; best = bmp; }
    }
    return best;
  }, []);

  return { slots, getThumbnail, cacheSize: cacheRef.current.size, version };
}

// ── waveform ──────────────────────────────────────────────────────────────────

export type WaveState = 'idle' | 'loading' | 'ready' | 'none' | 'error';
export interface ClientWaveform {
  peaks: Uint8Array | number[];
  duration: number;
  state: WaveState;
  reason?: string;
}

/** Max-pool PCM into `bucketsPerSecond` magnitude buckets, 0..255, across all channels. */
export function pcmPeaks(channels: ReadonlyArray<Float32Array>, sampleRate: number, bucketsPerSecond: number): Uint8Array {
  const length = channels[0]?.length ?? 0;
  const buckets = Math.max(1, Math.ceil((length / sampleRate) * bucketsPerSecond));
  const out = new Uint8Array(buckets);
  const per = length / buckets;
  for (let b = 0; b < buckets; b++) {
    const a = Math.floor(b * per);
    const z = Math.min(length, Math.max(a + 1, Math.floor((b + 1) * per)));
    let peak = 0;
    for (const ch of channels) for (let i = a; i < z; i++) { const v = ch[i] < 0 ? -ch[i] : ch[i]; if (v > peak) peak = v; }
    out[b] = Math.min(255, Math.round(peak * 255));
  }
  return out;
}

export function useWebAudioWaveform(file: File | null | undefined, enabled = true): ClientWaveform {
  const [wave, setWave] = useState<ClientWaveform>({ peaks: [], duration: 0, state: 'idle' });
  useEffect(() => {
    if (!enabled || !file) {
      setWave({ peaks: [], duration: 0, state: 'idle' });
      return undefined;
    }
    if (file.size > WAVEFORM_MAX_BYTES) {
      setWave({ peaks: [], duration: 0, state: 'none', reason: `file is ${(file.size / 1048576).toFixed(0)} MB; too large to decode in the browser` });
      return undefined;
    }
    let alive = true;
    setWave({ peaks: [], duration: 0, state: 'loading' });
    (async () => {
      try {
        const Ctx = (window as any).OfflineAudioContext || (window as any).webkitOfflineAudioContext;
        if (!Ctx) throw new Error('Web Audio is not available');
        // a 1-sample offline context only decodes; it never needs a user gesture and makes no sound
        const ctx = new Ctx(1, 1, 44100);
        const audio: AudioBuffer = await ctx.decodeAudioData(await file.arrayBuffer());
        if (!alive) return;
        const channels = Array.from({ length: audio.numberOfChannels }, (_, i) => audio.getChannelData(i));
        const peaks = pcmPeaks(channels, audio.sampleRate, 200);
        setWave({ peaks, duration: audio.duration, state: 'ready' });
      } catch (err: any) {
        if (!alive) return;
        // decodeAudioData rejects for a clip with no audio stream and for codecs the browser lacks
        setWave({ peaks: [], duration: 0, state: 'none', reason: err?.message ? String(err.message) : 'no decodable audio stream' });
      }
    })();
    return () => { alive = false; };
  }, [file, enabled]);
  return wave;
}
