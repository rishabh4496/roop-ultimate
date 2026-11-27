// Still thumbnails for video files, so a tile can show a picture without keeping a <video> mounted.
//
// Outputs and History used to mount one <video preload="metadata"> per visible tile purely to show its first frame.
// Every one of those holds a decoder, and a grid of dozens is dozens of decoders resident at once. Now a tile shows
// an <img> of the first frame and mounts a real <video> only while the pointer is over it (see HoverVideoThumb).
//
// The first frame comes from a DETACHED <video> that is torn down (src removed, load() called) the moment the frame
// is drawn, at most CONCURRENCY at a time, and for tiles that are still wanted: a tile that scrolls away before its
// turn cancels its job. Results are kept in a bounded cache of owned object URLs (objectUrls.js owns the revoke).
//
// No seek: a seek would be one more decode per tile and the first frame is what the tile showed before.

import { createOwnedUrl, revokeUrl } from '../faceswap/objectUrls';

export const THUMB_WIDTH = 320;
export const THUMB_CACHE_MAX = 300;
const CONCURRENCY = 2;
const TIMEOUT_MS = 8000;

const OWNER = { name: 'video-thumbs' };
const cache = new Map();            // src -> blob url, or null when the frame could not be read (LRU: oldest first)
const jobs = new Map();             // src -> job, while queued or running
const waiting = [];
let running = 0;
const stats = { captured: 0, failed: 0, cancelled: 0, evicted: 0, peakRunning: 0 };

/** Free a media element's decoder NOW rather than whenever the garbage collector gets to it. */
export function releaseVideoElement(el) {
  if (!el) return;
  try { el.pause(); } catch { /* not playing */ }
  try { el.removeAttribute('src'); el.load(); } catch { /* already detached */ }
}

const remember = (src, url) => {
  cache.delete(src);
  cache.set(src, url);
  while (cache.size > THUMB_CACHE_MAX) {
    const [oldSrc, oldUrl] = cache.entries().next().value;
    cache.delete(oldSrc);
    if (oldUrl) revokeUrl(oldUrl);
    stats.evicted += 1;
  }
};

function captureFirstFrame(src) {
  return new Promise((resolve) => {
    const v = document.createElement('video');
    let done = false;
    let timer = 0;
    const finish = (blob) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      v.onloadeddata = null; v.onerror = null;
      releaseVideoElement(v);
      resolve(blob);
    };
    v.muted = true;
    v.playsInline = true;
    v.preload = 'metadata';
    v.onerror = () => finish(null);
    v.onloadeddata = () => {
      try {
        const w = v.videoWidth || 0;
        const h = v.videoHeight || 0;
        if (!w || !h) { finish(null); return; }
        const canvas = document.createElement('canvas');
        canvas.width = Math.min(THUMB_WIDTH, w);
        canvas.height = Math.max(1, Math.round(canvas.width * (h / w)));
        canvas.getContext('2d').drawImage(v, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((b) => finish(b), 'image/jpeg', 0.72);
      } catch {
        finish(null);                  // a tainted canvas (cross-origin) or a lost context: show the placeholder
      }
    };
    timer = setTimeout(() => finish(null), TIMEOUT_MS);
    v.src = src;
  });
}

function pump() {
  while (running < CONCURRENCY && waiting.length) {
    const job = waiting.shift();
    if (job.refs <= 0) continue;       // everyone who wanted it has scrolled away
    job.state = 'running';
    running += 1;
    stats.peakRunning = Math.max(stats.peakRunning, running);
    captureFirstFrame(job.src).then((blob) => {
      running -= 1;
      jobs.delete(job.src);
      let url = null;
      if (blob) { url = createOwnedUrl(blob, OWNER); stats.captured += 1; } else stats.failed += 1;
      remember(job.src, url);
      job.resolve(url);
      pump();
    });
  }
}

/** A cached result: a blob url, null (known unreadable), or undefined (never asked). */
export const peekVideoThumb = (src) => {
  if (!cache.has(src)) return undefined;
  const url = cache.get(src);
  cache.delete(src);                   // touch: most recently used goes last
  cache.set(src, url);
  return url;
};

/**
 * Ask for a thumbnail. Returns { promise, release }; call release() when the tile no longer needs it so a queued job
 * that nobody wants any more is dropped instead of decoded.
 */
export function acquireVideoThumb(src) {
  const hit = peekVideoThumb(src);
  if (hit !== undefined) return { promise: Promise.resolve(hit), release() {} };
  let job = jobs.get(src);
  if (!job) {
    job = { src, refs: 0, state: 'queued', resolve: null, promise: null };
    job.promise = new Promise((res) => { job.resolve = res; });
    jobs.set(src, job);
    waiting.push(job);
    pump();
  }
  job.refs += 1;
  let released = false;
  return {
    promise: job.promise,
    release() {
      if (released) return;
      released = true;
      job.refs -= 1;
      if (job.refs <= 0 && job.state === 'queued') {
        const i = waiting.indexOf(job);
        if (i >= 0) waiting.splice(i, 1);
        jobs.delete(src);
        stats.cancelled += 1;
        job.resolve(undefined);
      }
    },
  };
}

/** Drop everything (tests, and a changed output folder). */
export function clearVideoThumbs() {
  for (const url of cache.values()) if (url) revokeUrl(url);
  cache.clear();
}

export const __thumbStats = () => ({ ...stats, cached: cache.size, queued: waiting.length, running });
if (typeof window !== 'undefined') window.__roopVideoThumbs = { __thumbStats };
