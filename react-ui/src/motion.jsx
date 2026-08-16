// ── Shared motion foundation ──────────────────────────────────────────────
// One vocabulary of spring physics + entrance variants so every animated
// surface across the app feels like the same instrument. The app's signature
// easing is cubic-bezier(0.16,1,0.3,1) ("settle without ringing"); these
// springs approximate it, with a couple of intentionally bouncier presets for
// discrete targets (buttons, toggles, thumbnails).
//
// Reduced motion is handled globally via <MotionConfig reducedMotion="user">
// in App.jsx, so transforms/layout collapse to instant when the OS asks — we
// don't need to gate every component by hand.
import React, { useRef } from 'react';
import { motion, AnimatePresence, useReducedMotion, useMotionValue, useSpring, useTransform, LayoutGroup, MotionConfig } from 'framer-motion';

export { motion, AnimatePresence, useReducedMotion, LayoutGroup, MotionConfig };

export const spring = {
  snappy: { type: 'spring', stiffness: 520, damping: 34, mass: 0.7 },
  smooth: { type: 'spring', stiffness: 260, damping: 30 },
  bouncy: { type: 'spring', stiffness: 460, damping: 20, mass: 0.8 },
  soft: { type: 'spring', stiffness: 170, damping: 22 },
};

// Cinematic entrance: rise + settle. Used by cards/panels on mount. No filter
// blur here on purpose — these panels already carry a backdrop-blur, and many
// animating a blur at once is GPU-costly. The de-blur is reserved for the
// single per-tab view transition below.
export const fadeUp = {
  hidden: { opacity: 0, y: 24, scale: 0.98 },
  show: {
    opacity: 1, y: 0, scale: 1,
    transition: { ...spring.smooth, opacity: { duration: 0.4 } },
  },
  exit: { opacity: 0, y: -12, scale: 0.985, transition: { duration: 0.2 } },
};

// Parent orchestrator: cascade children in with a staggered delay.
export const staggerParent = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06, delayChildren: 0.05 } },
  exit: { transition: { staggerChildren: 0.02, staggerDirection: -1 } },
};

// Full-view swap (tab changes): depth cross-slide.
//
// Deliberately compositor-only — opacity + transform, no animated `filter`.
// A tab panel is the single largest layer in the app (the Face Swap view is
// thousands of nodes deep, most of them already inside backdrop-blur glass
// panels); animating blur() on that wrapper forced the whole subtree to be
// re-rasterized every frame and made every tab change stutter on its first
// few frames. Opacity/translate/scale run entirely on the compositor, so the
// same depth read costs nothing.
//
// The exit is short and the enter travel small on purpose: AnimatePresence
// runs in mode="wait", so exit + enter are serialised and their durations add
// up into the perceived latency of a tab click.
export const viewTransition = {
  initial: { opacity: 0, y: 12, scale: 0.99 },
  animate: {
    opacity: 1, y: 0, scale: 1,
    transition: { ...spring.snappy, opacity: { duration: 0.22, ease: [0.16, 1, 0.3, 1] } },
  },
  exit: { opacity: 0, y: -8, scale: 0.995, transition: { duration: 0.13, ease: [0.4, 0, 1, 1] } },
};

// Drop-in staggered container. Children should be <Reveal> (or any motion
// element with the fadeUp variants and no own initial/animate).
export const Stagger = ({ children, className = '', ...rest }) => (
  <motion.div
    className={className}
    variants={staggerParent}
    initial="hidden"
    animate="show"
    {...rest}
  >
    {children}
  </motion.div>
);

// A single cascading reveal element. Inherits orchestration from a <Stagger>
// parent; also works standalone (animates on mount).
export const Reveal = ({ children, className = '', as = 'div', ...rest }) => {
  const Comp = motion[as] || motion.div;
  return (
    <Comp className={className} variants={fadeUp} {...rest}>
      {children}
    </Comp>
  );
};

// ── Mouse-follow tilt + cursor-tracking glare ─────────────────────────────
// The signature "premium" hover: the surface tilts a few degrees toward the
// cursor with spring physics while a soft accent glow tracks the pointer. All
// of this runs on Framer motion values (not React state), so hovering never
// triggers a re-render — cheap even when many cards mount it.
//
// Shared hook so both the punchy hero <TiltCard> and the gentle, form-safe
// tilt baked into the <Card> primitive draw from one implementation. We
// deliberately DON'T set transformStyle: preserve-3d — nothing inside needs a
// real 3D layer, and omitting it avoids z-stacking surprises with dropdowns
// and overlays inside cards.
export const useTilt = ({ max = 8, perspective = 1000 } = {}) => {
  const ref = useRef(null);
  const reduce = useReducedMotion();
  const px = useMotionValue(0.5);
  const py = useMotionValue(0.5);
  const rx = useSpring(useTransform(py, [0, 1], [max, -max]), spring.soft);
  const ry = useSpring(useTransform(px, [0, 1], [-max, max]), spring.soft);
  const gx = useTransform(px, [0, 1], ['0%', '100%']);
  const gy = useTransform(py, [0, 1], ['0%', '100%']);
  // Computed unconditionally (hook rules) — only rendered when glare is on.
  const glareBg = useTransform([gx, gy], ([x, y]) => `radial-gradient(600px circle at ${x} ${y}, var(--accent-glow), transparent 40%)`);

  const onMouseMove = (e) => {
    if (reduce || !ref.current) return;
    const r = ref.current.getBoundingClientRect();
    px.set((e.clientX - r.left) / r.width);
    py.set((e.clientY - r.top) / r.height);
  };
  const onMouseLeave = () => { px.set(0.5); py.set(0.5); };

  const style = reduce ? undefined : { rotateX: rx, rotateY: ry, transformPerspective: perspective };
  return { ref, reduce, onMouseMove, onMouseLeave, style, glareBg };
};

// Glare overlay element — shared between TiltCard and Card. Fades in on hover
// of the given group name so multiple nested groups don't cross-trigger.
// NOTE: the group-hover classes MUST be written as complete literal strings,
// or Tailwind's JIT won't generate them (a `group-hover/${group}` template
// would silently drop the rule and the glare would never show).
const GLARE_HOVER = {
  tilt: 'group-hover/tilt:opacity-100',
  spot: 'group-hover/spot:opacity-100',
};
export const TiltGlare = ({ glareBg, group = 'tilt' }) => (
  <motion.div
    aria-hidden
    className={`pointer-events-none absolute inset-0 rounded-[inherit] opacity-0 transition-opacity duration-300 ${GLARE_HOVER[group] || GLARE_HOVER.tilt}`}
    style={{ background: glareBg }}
  />
);

// Punchy hero tilt — for discrete display cards where the depth pays off.
export const TiltCard = ({ children, className = '', max = 8, glare = true, ...rest }) => {
  const t = useTilt({ max, perspective: 900 });
  return (
    <motion.div
      ref={t.ref}
      onMouseMove={t.onMouseMove}
      onMouseLeave={t.onMouseLeave}
      style={t.style}
      className={`group/tilt relative ${className}`}
      {...rest}
    >
      {children}
      {glare && !t.reduce && <TiltGlare glareBg={t.glareBg} group="tilt" />}
    </motion.div>
  );
};
