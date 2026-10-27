import { useEffect, useState } from 'react';
import { acquireVideoThumb, peekVideoThumb } from './videoThumb';

/**
 * A still for a video tile. `url` is the first frame once captured, null when it could not be read (the tile
 * shows its placeholder), and undefined while it is still being fetched.
 */
export function useVideoThumb(src, enabled = true) {
  const [url, setUrl] = useState(() => (src ? peekVideoThumb(src) : null));
  useEffect(() => {
    if (!enabled || !src) return undefined;
    const hit = peekVideoThumb(src);
    if (hit !== undefined) { setUrl(hit); return undefined; }
    setUrl(undefined);
    const req = acquireVideoThumb(src);
    let live = true;
    req.promise.then((u) => { if (live && u !== undefined) setUrl(u); });
    return () => { live = false; req.release(); };
  }, [src, enabled]);
  return url;
}

/**
 * Hover (and keyboard focus) mounts the tile's real <video>; leaving unmounts it. Handlers are returned as props to
 * spread onto the hover surface, so a tile's whole preview box is the target rather than just the <video>.
 *
 * Focus counts too: the tile's action buttons appear on focus-within, so a keyboard user should get the same
 * preview a pointer user does.
 */
export function useHoverMount() {
  const [on, setOn] = useState(false);
  return {
    on,
    bind: {
      onPointerEnter: () => setOn(true),
      onPointerLeave: () => setOn(false),
      onFocus: () => setOn(true),
      onBlur: (e) => { if (!e.currentTarget.contains(e.relatedTarget)) setOn(false); },
    },
  };
}
