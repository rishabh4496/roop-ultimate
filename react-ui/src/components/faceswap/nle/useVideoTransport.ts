/**
 * useVideoTransport.ts: frame-accurate transport over a plain <video> element (the Extras editor).
 *
 * FaceSwap's timeline drives a server-decoded frame buffer; Extras plays a local file in the browser,
 * so the same NLE needs the same handful of operations on an HTMLVideoElement:
 *
 *   frame      floor(currentTime * fps): seeks land on the MIDDLE of a frame's interval ((f + 0.5) / fps),
 *              so the decoder shows frame f, not the one before it
 *   fps        estimated from presentation timestamps while the clip plays (requestVideoFrameCallback),
 *              else assumed 30 and flagged `fpsAssumed` so timecodes read as approximate
 *   loop       the In-Out region (the whole clip until trimmed). In / Out here are a PREVIEW region only:
 *              /api/extras/apply has no trim, so Apply still exports the whole clip
 */

import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from 'react';
import { fpsFromTimestamps, probeFps } from './clientMedia';
import { loopRegion, snapRate, stepTo } from './transportModel';

const ASSUMED_FPS = 30;
const FPS_SAMPLES = 24;

export interface VideoTransport {
  fps: number;
  fpsAssumed: boolean;
  maxFrames: number;
  frame: number;
  isPlaying: boolean;
  rate: number;
  loop: boolean;
  trimIn: number;
  trimOut: number;
  togglePlay: () => void;
  seekFrame: (frame: number) => void;
  step: (delta: number) => void;
  setRate: (rate: number) => void;
  setLoop: (loop: boolean) => void;
  setTrim: (nextIn: number, nextOut: number) => void;
  setTrimIn: (frame: number) => void;
  setTrimOut: (frame: number) => void;
}

export function useVideoTransport(videoRef: RefObject<HTMLVideoElement | null>, resetKey: string): VideoTransport {
  const [duration, setDuration] = useState(0);
  const [fps, setFps] = useState(ASSUMED_FPS);
  const [fpsAssumed, setFpsAssumed] = useState(true);
  const [frame, setFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [rate, setRateState] = useState(1);
  const [loop, setLoop] = useState(true); // the old <video loop> behaviour is the default
  const [trim, setTrimState] = useState<{ a: number; b: number } | null>(null);

  const fpsRef = useRef(fps);
  fpsRef.current = fps;
  const maxFrames = Math.max(1, Math.floor(duration * fps + 1e-6));
  const last = maxFrames - 1;
  const region = useMemo(() => loopRegion(trim?.a ?? 0, trim?.b ?? last, maxFrames), [trim, last, maxFrames]);
  const regionRef = useRef(region);
  regionRef.current = region;
  const loopRef = useRef(loop);
  loopRef.current = loop;

  // A different clip starts over.
  useEffect(() => {
    setDuration(0);
    setFps(ASSUMED_FPS);
    setFpsAssumed(true);
    setFrame(0);
    setIsPlaying(false);
    setTrimState(null);
  }, [resetKey]);

  const readFrame = useCallback((video: HTMLVideoElement) => {
    const f = Math.floor(video.currentTime * fpsRef.current + 1e-6);
    return Math.max(0, Math.min(Math.max(0, Math.floor(video.duration * fpsRef.current + 1e-6) - 1), f));
  }, []);

  // Track the element's own state. Re-attached when the clip changes (the viewport re-keys its <video>).
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return undefined;
    const onMeta = () => { if (Number.isFinite(video.duration)) setDuration(video.duration); };
    const sync = () => setFrame((prev) => { const f = readFrame(video); return f === prev ? prev : f; });
    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    const onEnded = () => {
      if (loopRef.current) {
        video.currentTime = (regionRef.current.in + 0.5) / fpsRef.current;
        void video.play().catch(() => undefined);
      } else {
        setIsPlaying(false);
      }
    };
    video.addEventListener('loadedmetadata', onMeta);
    video.addEventListener('durationchange', onMeta);
    video.addEventListener('seeked', sync);
    video.addEventListener('timeupdate', sync);
    video.addEventListener('play', onPlay);
    video.addEventListener('pause', onPause);
    video.addEventListener('ended', onEnded);
    onMeta();
    setIsPlaying(!video.paused);
    return () => {
      video.removeEventListener('loadedmetadata', onMeta);
      video.removeEventListener('durationchange', onMeta);
      video.removeEventListener('seeked', sync);
      video.removeEventListener('timeupdate', sync);
      video.removeEventListener('play', onPlay);
      video.removeEventListener('pause', onPause);
      video.removeEventListener('ended', onEnded);
    };
  }, [videoRef, resetKey, readFrame]);

  // While playing: follow the clock every animation frame, and wrap at Out when looping the region.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !isPlaying) return undefined;
    let raf = 0;
    const tick = () => {
      const r = regionRef.current;
      if (loopRef.current && video.currentTime >= (r.out + 1) / fpsRef.current) {
        video.currentTime = (r.in + 0.5) / fpsRef.current;
      }
      setFrame((prev) => { const f = readFrame(video); return f === prev ? prev : f; });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [videoRef, isPlaying, resetKey, readFrame]);

  // The frame rate is not in the DOM. Primary: a hidden probe plays the clip off-screen and reads the
  // presentation timestamps at once, independent of the visible player. Backup (below): the same reading
  // from the visible player while it plays, for browsers/clips where the probe cannot run.
  useEffect(() => {
    if (!resetKey) return undefined;
    let alive = true;
    void probeFps(resetKey).then((measured) => {
      if (alive && measured) { setFps(measured); setFpsAssumed(false); }
    });
    return () => { alive = false; };
  }, [resetKey]);

  useEffect(() => {
    const video = videoRef.current as (HTMLVideoElement & { requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number; presentedFrames: number }) => void) => number; cancelVideoFrameCallback?: (id: number) => void }) | null;
    if (!video || typeof video.requestVideoFrameCallback !== 'function' || !fpsAssumed) return undefined;
    const times: number[] = [];
    let handle = 0;
    let stopped = false;
    const onFrame = (_now: number, meta: { mediaTime: number; presentedFrames: number }) => {
      if (stopped) return;
      times.push(meta.mediaTime);
      if (times.length >= FPS_SAMPLES) {
        const est = fpsFromTimestamps(times);
        if (est) { setFps(est); setFpsAssumed(false); return; }
        times.length = 0;
      }
      handle = video.requestVideoFrameCallback!(onFrame);
    };
    handle = video.requestVideoFrameCallback(onFrame);
    return () => { stopped = true; video.cancelVideoFrameCallback?.(handle); };
  }, [videoRef, resetKey, fpsAssumed]);

  const seekFrame = useCallback((f: number) => {
    const video = videoRef.current;
    if (!video) return;
    const target = Math.max(0, Math.min(Math.max(0, Math.floor(video.duration * fpsRef.current + 1e-6) - 1), Math.round(f)));
    video.currentTime = (target + 0.5) / fpsRef.current;
    setFrame(target);
  }, [videoRef]);

  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) void video.play().catch(() => undefined);
    else video.pause();
  }, [videoRef]);

  const step = useCallback((delta: number) => {
    const video = videoRef.current;
    if (!video) return;
    video.pause();
    seekFrame(stepTo(readFrame(video), delta, Math.floor(video.duration * fpsRef.current + 1e-6)));
  }, [videoRef, seekFrame, readFrame]);

  const setRate = useCallback((r: number) => {
    const snapped = snapRate(r);
    setRateState(snapped);
    if (videoRef.current) videoRef.current.playbackRate = snapped;
  }, [videoRef]);

  // a new <video> element starts at 1x: carry the chosen rate onto it
  useEffect(() => { if (videoRef.current) videoRef.current.playbackRate = rate; }, [videoRef, resetKey, rate]);

  const setTrim = useCallback((a: number, b: number) => setTrimState({ a: Math.min(a, b), b: Math.max(a, b) }), []);
  const setTrimIn = useCallback((f: number) => setTrimState((t) => ({ a: Math.min(f, t?.b ?? last), b: t?.b ?? last })), [last]);
  const setTrimOut = useCallback((f: number) => setTrimState((t) => ({ a: t?.a ?? 0, b: Math.max(f, t?.a ?? 0) })), []);

  return {
    fps, fpsAssumed, maxFrames, frame, isPlaying, rate, loop,
    trimIn: region.in, trimOut: region.out,
    togglePlay, seekFrame, step, setRate, setLoop, setTrim, setTrimIn, setTrimOut,
  };
}
