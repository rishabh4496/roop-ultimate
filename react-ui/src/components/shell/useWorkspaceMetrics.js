import { useLayoutEffect } from 'react';

// Publishes the two numbers the sticky parts of Face Swap have to agree on, as CSS variables on <html>:
//
//   --ws-header-h  the live height of the top header. It wraps to two rows in a narrow window, so
//                  "top: 5rem" (a guess) either left a strip of scrolling content between the header and
//                  the run bar, or hid the run bar's top edge under the header.
//   --ws-vh        the visible height of the window in LAYOUT pixels. The app zoom (Ctrl +/-) is the CSS
//                  `zoom` property on <html>, which scales layout pixels, so `100vh` is not the room the
//                  layout actually has; innerHeight divided by the zoom is.
//
// Layout pixels (offsetHeight, not getBoundingClientRect) on purpose: they are what `top:` and `height:`
// are measured in under the app zoom.
export function useWorkspaceMetrics(headerRef) {
  useLayoutEffect(() => {
    const root = document.documentElement;
    const apply = () => {
      const z = parseFloat(root.style.zoom) || 1;
      const h = headerRef.current?.offsetHeight;
      if (h) root.style.setProperty('--ws-header-h', `${h}px`);
      root.style.setProperty('--ws-vh', `${Math.round(window.innerHeight / z)}px`);
    };
    apply();
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(apply) : null;
    if (ro && headerRef.current) ro.observe(headerRef.current);
    // The zoom is applied as an inline style on <html>, which no resize event reports.
    const mo = typeof MutationObserver !== 'undefined' ? new MutationObserver(apply) : null;
    mo?.observe(root, { attributes: true, attributeFilter: ['style'] });
    window.addEventListener('resize', apply);
    return () => {
      ro?.disconnect();
      mo?.disconnect();
      window.removeEventListener('resize', apply);
    };
  }, [headerRef]);
}
