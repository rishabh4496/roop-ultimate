// Tab-change transitions for React UI 2.0.
//
// Every style ends on the IDENTITY: no transform, no clip-path, no opacity below 1. That is not a
// nicety. The tab panel holds dialogs and popovers positioned `fixed`, and a resting transform (or
// perspective, or filter) on an ancestor makes that ancestor their containing block - a modal would
// then be positioned against the panel, not the window, and a clip-path would cut it off. So:
//   * 3D depth comes from `transformPerspective` on the MOVING states only, and `transitionEnd`
//     resets the transform to 'none' when the enter animation finishes;
//   * the portal's clip-path ends at 'none' through `transitionEnd` too.
// The cost rules of 1.0's viewTransition still hold: opacity + transform (+ one clip-path) only, no
// animated filter on what is the largest layer in the app.

const EASE_OUT = [0.16, 1, 0.3, 1];
const EASE_IN = [0.4, 0, 1, 1];
const REST = { transform: 'none' };

export const TRANSITION_VARIANTS = {
  depth: {
    initial: { opacity: 0, y: 26, scale: 0.94, rotateX: 9, transformPerspective: 1400, originY: 0 },
    animate: {
      opacity: 1, y: 0, scale: 1, rotateX: 0,
      transition: { duration: 0.5, ease: EASE_OUT },
      transitionEnd: REST,
    },
    exit: { opacity: 0, y: -14, scale: 1.025, rotateX: -4, transformPerspective: 1400, transition: { duration: 0.16, ease: EASE_IN } },
  },
  flip: {
    initial: { opacity: 0, rotateY: -38, x: 40, transformPerspective: 1800, originX: 0 },
    animate: {
      opacity: 1, rotateY: 0, x: 0,
      transition: { duration: 0.55, ease: EASE_OUT },
      transitionEnd: REST,
    },
    exit: { opacity: 0, rotateY: 24, x: -30, transformPerspective: 1800, originX: 1, transition: { duration: 0.18, ease: EASE_IN } },
  },
  portal: {
    initial: { opacity: 0.4, clipPath: 'circle(0% at 0% 40%)' },
    animate: {
      opacity: 1, clipPath: 'circle(160% at 0% 40%)',
      transition: { duration: 0.6, ease: EASE_OUT },
      transitionEnd: { clipPath: 'none' },
    },
    exit: { opacity: 0, transition: { duration: 0.14, ease: EASE_IN } },
  },
  glitch: {
    initial: { opacity: 0, x: -10, skewX: 6 },
    animate: {
      opacity: [0, 1, 0.6, 1],
      x: [-10, 6, -3, 0],
      skewX: [6, -3, 1, 0],
      transition: { duration: 0.42, times: [0, 0.35, 0.6, 1], ease: 'linear' },
      transitionEnd: REST,
    },
    exit: { opacity: 0, x: 8, skewX: -4, transition: { duration: 0.12 } },
  },
  warp: {
    initial: { opacity: 0, x: 90, scaleX: 1.18, scaleY: 0.92, originX: 0 },
    animate: {
      opacity: 1, x: 0, scaleX: 1, scaleY: 1,
      transition: { type: 'spring', stiffness: 420, damping: 36, opacity: { duration: 0.2 } },
      transitionEnd: REST,
    },
    exit: { opacity: 0, x: -70, scaleX: 1.12, transition: { duration: 0.15, ease: EASE_IN } },
  },
  none: {
    initial: { opacity: 1 },
    animate: { opacity: 1 },
    exit: { opacity: 1, transition: { duration: 0 } },
  },
};

export const variantsFor = (style) => TRANSITION_VARIANTS[style] || TRANSITION_VARIANTS.depth;
