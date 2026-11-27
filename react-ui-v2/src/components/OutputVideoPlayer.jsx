import React, { useEffect, useRef, useState } from 'react';
import VideoCompareStage from './player/VideoCompareStage';

/**
 * OutputVideoPlayer — a finished render, and optionally the original beside it.
 *
 * RANGE REQUESTS. The <video> element seeks with HTTP Range requests and the
 * backend answers them properly (routes_output._stream_file_response: 206,
 * suffix ranges, 416, ETag / If-Range / 304). Nothing here re-implements that:
 * `preload="metadata"` fetches only the header until the user plays, and the
 * browser's media stack then streams ranges as it needs them.
 *
 * CACHE BUSTING by FILE, not by clock. The caller passes a URL versioned with
 * the file's size + mtime (outputUrl.js), so a re-render at the same path is a
 * new URL and a remount of the same file is not. This component used to stamp
 * `?t=${Date.now()}` itself, which re-downloaded a finished render on every
 * mount — every Pinokio tab switch — and, because the stamp was recomputed
 * whenever `src` changed identity, could remount the element mid-playback.
 * The server's `Cache-Control: no-cache` + ETag covers unversioned callers.
 *
 * BUFFER MANAGEMENT. A <video> that is merely dropped from the DOM keeps its
 * connection and decoded frames until garbage collection. On unmount (and on
 * every new file, which is a new element via `key`) the element is paused and
 * its source detached + reloaded, which aborts in-flight ranges and frees the
 * decoder immediately. A network error mid-play retries (up to twice) from the
 * same position instead of leaving a dead player.
 *
 * COMPARE. Given `source` (outputUrl.outputSource: the original target and its
 * time offset), a Split / Side-by-side switch shows the original and the result
 * together in one WebGL canvas (player/VideoCompareStage).
 */
const MAX_NETWORK_RETRIES = 2;

const VIEWS = [
  { id: 'single', label: 'Result' },
  { id: 'wipe', label: 'Split' },
  { id: 'side', label: 'Side by side' },
];

export default function OutputVideoPlayer({
  src,
  renderKey,
  source = null,
  className = 'w-full rounded-xl border border-white/5',
  style,
  onLoadedMetadata,
  onCanPlay,
  onError,
  ...props
}) {
  const [view, setView] = useState('single');
  const [failure, setFailure] = useState('');
  const videoRef = useRef(null);
  const retriesRef = useRef(0);
  const resumeAtRef = useRef(null);
  const videoUrl = src || '';
  const canCompare = !!source?.url && (source.kind || 'video') === 'video';
  const elementKey = `${renderKey || ''}|${videoUrl}`;

  useEffect(() => {
    retriesRef.current = 0;
    resumeAtRef.current = null;
    setFailure('');
  }, [elementKey]);

  // Release on unmount / file change. The element is keyed by the file, so the
  // node captured here is always the one being discarded.
  useEffect(() => {
    const v = videoRef.current;
    return () => {
      if (!v) return;
      try { v.pause(); v.removeAttribute('src'); v.load(); } catch { /* already gone */ }
    };
  }, [elementKey, view]);

  if (!videoUrl) return null;

  const handleError = (e) => {
    const v = e.currentTarget;
    const err = v.error;
    // MEDIA_ERR_NETWORK (2): the connection dropped (a backend restart, the
    // laptop sleeping). Reload from where it was rather than dying.
    if (err?.code === 2 && retriesRef.current < MAX_NETWORK_RETRIES) {
      retriesRef.current += 1;
      resumeAtRef.current = v.currentTime || 0;
      const delay = 500 * retriesRef.current;
      setTimeout(() => { try { v.load(); } catch { /* unmounted */ } }, delay);
      return;
    }
    console.error('[OutputVideoPlayer] playback failed', {
      src: videoUrl, code: err?.code, message: err?.message,
      networkState: v.networkState, readyState: v.readyState,
    });
    setFailure(err?.code === 4
      ? 'This browser cannot play this file (codec or container not supported). Download it or open the folder instead.'
      : 'The video could not be loaded.');
    onError?.(e);
  };

  const handleLoadedMetadata = (e) => {
    if (resumeAtRef.current != null) {
      e.currentTarget.currentTime = resumeAtRef.current;
      resumeAtRef.current = null;
    }
    onLoadedMetadata?.(e);
  };

  return (
    <div className="relative w-full space-y-2">
      {canCompare && (
        <div role="radiogroup" aria-label="Output view" className="inline-flex rounded-lg border border-white/10 bg-white/[0.03] p-0.5 text-xs">
          {VIEWS.map((v) => (
            <button
              key={v.id}
              type="button"
              role="radio"
              aria-checked={view === v.id}
              onClick={() => setView(v.id)}
              className={`px-2.5 py-1 rounded-md font-semibold transition-colors ${view === v.id
                ? 'bg-[var(--accent)] text-white' : 'text-white/60 hover:text-white'}`}
            >
              {v.label}
            </button>
          ))}
        </div>
      )}

      {view === 'single' || !canCompare ? (
        <div className="relative w-full overflow-hidden">
          <video
            key={elementKey}
            ref={videoRef}
            controls
            playsInline
            preload="metadata"
            src={videoUrl}
            className={className}
            style={style}
            onError={handleError}
            onLoadedMetadata={handleLoadedMetadata}
            onCanPlay={onCanPlay}
            {...props}
          />
        </div>
      ) : (
        <VideoCompareStage
          key={`${elementKey}|${source.url}`}
          outputUrl={videoUrl}
          sourceUrl={source.url}
          offsetS={source.offsetS || 0}
          mode={view}
          labelA="Original"
          labelB="Swapped"
          onError={() => setFailure('The original clip could not be loaded for comparison.')}
        />
      )}
      {failure && <div className="text-xs text-red-400" role="alert">{failure}</div>}
    </div>
  );
}
