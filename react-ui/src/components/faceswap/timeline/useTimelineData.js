import { useCallback, useEffect, useRef, useState } from 'react';
import { getJSON, postJSON, API } from '../../../api';
import { mergeIntervals, presenceRows } from './laneModel';

// Everything the lanes read from the backend, in one place, each with its own cache and its own
// "no data" answer. The lanes never invent data: an empty answer draws an empty lane with a line
// that says WHY it is empty (not scanned yet / no audio / no run yet).

const EMPTY_TRACKS = Object.freeze({
  loaded: false, sceneCuts: [], presence: [], presenceSource: 'none', sampleStride: 0,
  faceFree: [], reuse: [], renderRange: null, renderVideoMatches: false,
});

/**
 * Scene cuts, per-person presence, and the last run's Stage 1/2 maps for one target.
 *
 * `poll` makes it re-ask every few seconds (a run is in flight, so the maps are filling in);
 * otherwise it asks once per target and again on `refresh()`.
 */
export function useTimelineTracks({ targetIndex, targetKey, maxFrames, poll = false }) {
  const [tracks, setTracks] = useState(EMPTY_TRACKS);
  const [scanning, setScanning] = useState(false);
  const aliveRef = useRef(true);
  useEffect(() => { aliveRef.current = true; return () => { aliveRef.current = false; }; }, []);

  const load = useCallback(async () => {
    if (targetIndex === null || targetIndex === undefined || targetIndex < 0 || maxFrames <= 1) return;
    try {
      const res = await getJSON(`/api/target/timeline_tracks?index=${targetIndex}`);
      if (!aliveRef.current) return;
      setTracks({
        loaded: true,
        sceneCuts: (res.scene_cuts || []).map(Number).filter(Number.isFinite),
        presence: presenceRows(res.presence, maxFrames),
        presenceSource: res.presence_source || 'none',
        sampleStride: Number(res.sample_stride) || 0,
        faceFree: mergeIntervals(res.render?.face_free || [], 1, maxFrames),
        reuse: mergeIntervals(res.render?.reuse || [], 1, maxFrames),
        renderRange: res.render?.range || null,
        renderVideoMatches: !!res.render?.video_matches,
      });
    } catch {
      // The lanes are decoration over the clip: a failed fetch leaves what was there.
      if (aliveRef.current) setTracks((t) => (t.loaded ? t : { ...EMPTY_TRACKS, loaded: true }));
    }
  }, [targetIndex, maxFrames]);

  useEffect(() => { setTracks(EMPTY_TRACKS); load(); }, [targetKey, load]);
  useEffect(() => {
    if (!poll) return undefined;
    const id = setInterval(load, 4000);
    return () => clearInterval(id);
  }, [poll, load]);

  // One Face Bank scan fills the presence lane. apply:false so it does not replace the target
  // faces the user has already set up — it only populates the bank the tracks are read from.
  const scanPresence = useCallback(async () => {
    setScanning(true);
    try {
      await postJSON('/api/target/face_bank', { index: targetIndex, apply: false }, { timeout: 0 });
      await load();
    } finally {
      if (aliveRef.current) setScanning(false);
    }
  }, [targetIndex, load]);

  return { tracks, refresh: load, scanPresence, scanning };
}

/**
 * Audio amplitude peaks for the visible window.
 *
 * A coarse whole-clip envelope is fetched once per target; zooming in past its resolution fetches
 * the visible window at one bin per pixel (debounced, cached by window). The backend builds the
 * envelope in the background, so a first request can answer "not ready" - this retries quietly.
 *
 * Returns {coarse, fine, state}: peaks are Uint8Array 0..255; `fine` = {t0, t1, peaks} for the
 * window it was fetched for; state is 'loading' | 'ready' | 'none' (no audio) | 'error'.
 */
export function useWaveform({ targetIndex, targetKey, fps, maxFrames, view, trackW, enabled = true }) {
  const [coarse, setCoarse] = useState(null);
  const [fine, setFine] = useState(null);
  const [state, setState] = useState('loading');
  const fineKeyRef = useRef('');

  const duration = maxFrames / (fps || 25);

  useEffect(() => {
    setCoarse(null); setFine(null); setState('loading'); fineKeyRef.current = '';
    if (!enabled || targetIndex === null || targetIndex === undefined || targetIndex < 0 || maxFrames <= 1) return undefined;
    let cancelled = false;
    let timer = null;
    let tries = 0;
    const attempt = async () => {
      try {
        const res = await getJSON(`/api/target/waveform?index=${targetIndex}&bins=2048`);
        if (cancelled) return;
        if (res.state === 'none') { setState('none'); return; }
        if (!res.ready) {
          if (++tries > 40) { setState('error'); return; }
          timer = setTimeout(attempt, 1500);
          return;
        }
        setCoarse({ t0: 0, t1: res.duration, peaks: decodePeaks(res.peaks) });
        setState('ready');
      } catch {
        if (!cancelled) setState('error');
      }
    };
    attempt();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [targetKey, targetIndex, maxFrames, enabled]);

  // Fine window.
  useEffect(() => {
    if (state !== 'ready' || !coarse || !trackW) return undefined;
    const t0 = Math.max(0, (view.start - 1) / (fps || 25));
    const t1 = Math.min(duration, (view.end) / (fps || 25));
    const span = Math.max(1e-3, t1 - t0);
    // Only worth asking when the coarse envelope has fewer bins across the window than pixels.
    const coarseBinsAcross = (span / Math.max(1e-3, coarse.t1 - coarse.t0)) * coarse.peaks.length;
    if (coarseBinsAcross >= trackW * 0.75) { setFine(null); return undefined; }
    // Quantise the window so a small pan reuses the response instead of refetching.
    const pad = span * 0.25;
    const q = Math.max(0.05, span / 8);
    const a = Math.max(0, Math.floor((t0 - pad) / q) * q);
    const b = Math.min(duration, Math.ceil((t1 + pad) / q) * q);
    const bins = Math.min(4096, Math.max(64, Math.round(trackW * ((b - a) / span))));
    const key = `${targetKey}|${a.toFixed(3)}|${b.toFixed(3)}|${bins}`;
    if (fineKeyRef.current === key) return undefined;
    const ctrl = new AbortController();
    const id = setTimeout(async () => {
      try {
        const res = await getJSON(`/api/target/waveform?index=${targetIndex}&t0=${a}&t1=${b}&bins=${bins}`, { signal: ctrl.signal });
        if (!res.ready) return;
        fineKeyRef.current = key;
        setFine({ t0: a, t1: b, peaks: decodePeaks(res.peaks) });
      } catch { /* the coarse envelope keeps drawing */ }
    }, 220);
    return () => { clearTimeout(id); ctrl.abort(); };
  }, [state, coarse, view.start, view.end, trackW, fps, duration, targetKey, targetIndex]);

  return { coarse, fine, state };
}

function decodePeaks(b64) {
  if (!b64) return new Uint8Array(0);
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/**
 * Filmstrip stills as a tile cache.
 *
 * The old strip was twelve stills re-fetched whenever the window settled. This keeps every still
 * it has ever fetched for the clip (bounded, LRU) keyed by FRAME, so zooming in shows the nearest
 * cached still instantly and then refines as the intermediate frames arrive. Requests go through
 * the existing ordered `preview_grid` endpoint, one batch in flight, and a superseded batch is
 * aborted. `slots` is what thumbSlots() returned for the current window.
 */
export function useFilmstrip({ targetIndex, targetKey, slots, enabled, onTile }) {
  const cacheRef = useRef(new Map());        // frame -> ImageBitmap
  const inflightRef = useRef(null);          // {ctrl, frames:Set}
  const failedRef = useRef(new Map());       // frame -> retry-after timestamp
  const keyRef = useRef(targetKey);
  const [version, setTick] = useState(0);

  const dispose = useCallback(() => {
    for (const bmp of cacheRef.current.values()) { try { bmp.close?.(); } catch { /* already closed */ } }
    cacheRef.current = new Map();
    failedRef.current = new Map();
    inflightRef.current?.ctrl.abort();
    inflightRef.current = null;
  }, []);

  useEffect(() => {
    if (keyRef.current !== targetKey) { dispose(); keyRef.current = targetKey; setTick((t) => t + 1); }
  }, [targetKey, dispose]);
  useEffect(() => dispose, [dispose]);

  useEffect(() => {
    if (!enabled || targetIndex === null || targetIndex === undefined || targetIndex < 0 || !slots.length) return undefined;
    const now = Date.now();
    const wanted = [];
    for (const s of slots) {
      if (s.dup || cacheRef.current.has(s.frame) || (failedRef.current.get(s.frame) || 0) > now) continue;
      wanted.push(s.frame);
    }
    if (!wanted.length) return undefined;
    // Keep an in-flight batch only if it still covers most of what is wanted now.
    const flying = inflightRef.current;
    if (flying && wanted.every((f) => flying.frames.has(f))) return undefined;
    const id = setTimeout(async () => {
      flying?.ctrl.abort();
      const batch = [...new Set(wanted)].sort((a, b) => a - b).slice(0, 24);
      const ctrl = new AbortController();
      inflightRef.current = { ctrl, frames: new Set(batch) };
      try {
        const res = await fetch(`${API}/api/target/preview_grid?index=${targetIndex}&frames=${batch.join(',')}&width=200`, { signal: ctrl.signal });
        if (!res.ok) throw new Error('grid failed');
        const buf = new Uint8Array(await res.arrayBuffer());
        let off = 0;
        let i = 0;
        const decoded = [];
        while (off + 4 <= buf.length && i < batch.length) {
          const len = ((buf[off] << 24) | (buf[off + 1] << 16) | (buf[off + 2] << 8) | buf[off + 3]) >>> 0;
          off += 4;
          if (off + len > buf.length) break;
          const frame = batch[i++];
          if (len === 0) failedRef.current.set(frame, Date.now() + 15000);
          else decoded.push(createImageBitmap(new Blob([buf.slice(off, off + len)], { type: 'image/jpeg' }))
            .then((bmp) => ({ frame, bmp })).catch(() => null));
          off += len;
        }
        for (const hit of await Promise.all(decoded)) {
          if (!hit) continue;
          if (keyRef.current !== targetKey) { hit.bmp.close?.(); continue; }
          cacheRef.current.set(hit.frame, hit.bmp);
        }
        // LRU-ish bound: Map iterates in insertion order, so the oldest go first.
        while (cacheRef.current.size > 700) {
          const oldest = cacheRef.current.keys().next().value;
          cacheRef.current.get(oldest)?.close?.();
          cacheRef.current.delete(oldest);
        }
        if (inflightRef.current?.ctrl === ctrl) inflightRef.current = null;
        setTick((t) => t + 1);
        onTile?.();
      } catch (err) {
        if (err?.name !== 'AbortError') for (const f of batch) failedRef.current.set(f, Date.now() + 15000);
        if (inflightRef.current?.ctrl === ctrl) inflightRef.current = null;
      }
    }, 160);
    return () => clearTimeout(id);
  }, [slots, enabled, targetIndex, targetKey, onTile]);

  /** The cached still for `frame`, else the cached still nearest to it within `reach` frames. */
  const tileFor = useCallback((frame, reach) => {
    const hit = cacheRef.current.get(frame);
    if (hit) return hit;
    if (!(reach > 0)) return null;
    let best = null;
    let bestD = reach + 1;
    for (const [f, bmp] of cacheRef.current) {
      const d = Math.abs(f - frame);
      if (d < bestD) { best = bmp; bestD = d; }
    }
    return best;
  }, []);

  return { tileFor, version, size: () => cacheRef.current.size };
}
