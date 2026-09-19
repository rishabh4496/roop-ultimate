import React, { useMemo } from 'react';

/**
 * OutputVideoPlayer
 *
 * Dedicated video player component for rendered outputs that:
 * 1. Wraps the <video> element to force re-render when a new render completes.
 * 2. Ensures the src attribute includes a cache-busting timestamp (?t=${Date.now()})
 *    to prevent the browser from serving a cached or stale file.
 * 3. Attaches explicit event handlers for onError, onLoadedMetadata, and onCanPlay
 *    with console logging to capture playback failures.
 * 4. Explicitly defines standard attributes:
 *    <video controls playsInline preload="metadata" src={videoUrl} />
 */
export default function OutputVideoPlayer({
  src,
  renderKey,
  className = 'w-full rounded-xl border border-white/5',
  style,
  onLoadedMetadata,
  onCanPlay,
  onError,
  ...props
}) {
  // Ensure the src attribute includes a cache-busting timestamp
  const videoUrl = useMemo(() => {
    if (!src) return '';
    // If src already contains a query parameter t=, refresh it with current timestamp
    if (/[?&]t=/.test(src)) {
      return src.replace(/([?&]t=)[^&]*/, `$1${Date.now()}`);
    }
    const sep = src.includes('?') ? '&' : '?';
    const stamp = renderKey ? `${Date.now()}_${encodeURIComponent(renderKey)}` : Date.now();
    return `${src}${sep}t=${stamp}`;
  }, [src, renderKey]);

  // Dynamic key derived from renderKey forces complete DOM remount when a new render completes
  const elementKey = renderKey ? `${renderKey}_${videoUrl}` : videoUrl;

  const handleError = (e) => {
    const err = e.currentTarget.error;
    console.error('[OutputVideoPlayer Error]', {
      src: videoUrl,
      code: err?.code,
      message: err?.message,
      networkState: e.currentTarget.networkState,
      readyState: e.currentTarget.readyState,
    });
    onError?.(e);
  };

  const handleLoadedMetadata = (e) => {
    console.log('[OutputVideoPlayer LoadedMetadata]', {
      src: videoUrl,
      duration: e.currentTarget.duration,
      videoWidth: e.currentTarget.videoWidth,
      videoHeight: e.currentTarget.videoHeight,
    });
    onLoadedMetadata?.(e);
  };

  const handleCanPlay = (e) => {
    console.log('[OutputVideoPlayer CanPlay]', {
      src: videoUrl,
      readyState: e.currentTarget.readyState,
    });
    onCanPlay?.(e);
  };

  if (!src) return null;

  return (
    <div className="relative w-full overflow-hidden">
      <video
        key={elementKey}
        controls
        playsInline
        preload="metadata"
        src={videoUrl}
        className={className}
        style={style}
        onError={handleError}
        onLoadedMetadata={handleLoadedMetadata}
        onCanPlay={handleCanPlay}
        {...props}
      />
    </div>
  );
}
