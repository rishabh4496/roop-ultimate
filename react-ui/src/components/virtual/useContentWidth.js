import { useLayoutEffect, useState } from 'react';

/**
 * [attachRef, width]: the element's content width in layout px, kept current by a ResizeObserver.
 * Layout px (clientWidth), not getBoundingClientRect, so it is in the same units as scrollTop and survives the app's
 * CSS zoom.
 */
export default function useContentWidth(initial = 0) {
  const [el, setEl] = useState(null);
  const [width, setWidth] = useState(initial);
  useLayoutEffect(() => {
    if (!el) return undefined;
    setWidth(el.clientWidth);
    if (typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, [el]);
  return [setEl, width];
}
