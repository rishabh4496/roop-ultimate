// Pointer-driven surface effects, installed ONCE on the document (no per-component listeners, so it
// reaches every panel of every 1.0 screen without editing them):
//
//   * spotlight - a soft light that follows the pointer inside `.glass-panel` / `.v2-spot` surfaces
//     (v2.css paints it from --mx / --my; a panel without the vars paints nothing);
//   * tilt      - `.v2-tilt` cards lean toward the pointer in 3D (--rx / --ry) and settle back on
//     leave. Only on small, self-contained cards: a transform makes an element the containing
//     block of `fixed` descendants, so it never goes on a panel that can hold a dialog.
//
// One rAF-throttled pointermove; the work per frame is two closest() lookups and four custom
// properties. Reduced motion turns tilt off (it moves); the spotlight is light, not motion, and
// stays. Returns an uninstaller.
import { useFxStore, prefersReducedMotion } from './fxStore';

const SPOT_SEL = '.glass-panel, .v2-spot';
const TILT_SEL = '.v2-tilt';
const MAX_TILT = 7; // degrees

export function installPointerFx() {
  let raf = 0;
  let last = null;
  let tilted = null;
  let lit = null;

  const resetTilt = (el) => {
    if (!el) return;
    el.style.setProperty('--rx', '0deg');
    el.style.setProperty('--ry', '0deg');
    el.classList.remove('is-tilting');
  };

  const frame = () => {
    raf = 0;
    const e = last;
    if (!e) return;
    const { spotlight, tilt } = useFxStore.getState();
    const target = e.target instanceof Element ? e.target : null;

    const spot = spotlight && target ? target.closest(SPOT_SEL) : null;
    if (lit && lit !== spot) lit.classList.remove('is-lit');
    if (spot) {
      const r = spot.getBoundingClientRect();
      spot.style.setProperty('--mx', `${e.clientX - r.left}px`);
      spot.style.setProperty('--my', `${e.clientY - r.top}px`);
      spot.classList.add('is-lit');
    }
    lit = spot;

    const card = tilt && !prefersReducedMotion() && target ? target.closest(TILT_SEL) : null;
    if (tilted && tilted !== card) resetTilt(tilted);
    if (card) {
      const r = card.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width - 0.5;
      const py = (e.clientY - r.top) / r.height - 0.5;
      card.style.setProperty('--rx', `${(-py * MAX_TILT).toFixed(2)}deg`);
      card.style.setProperty('--ry', `${(px * MAX_TILT).toFixed(2)}deg`);
      card.style.setProperty('--gx', `${((px + 0.5) * 100).toFixed(1)}%`);
      card.style.setProperty('--gy', `${((py + 0.5) * 100).toFixed(1)}%`);
      card.classList.add('is-tilting');
    }
    tilted = card;
  };

  const onMove = (e) => {
    last = e;
    if (!raf) raf = requestAnimationFrame(frame);
  };
  const onLeave = () => {
    resetTilt(tilted); tilted = null;
    if (lit) lit.classList.remove('is-lit');
    lit = null;
  };

  document.addEventListener('pointermove', onMove, { passive: true });
  document.documentElement.addEventListener('pointerleave', onLeave);
  return () => {
    cancelAnimationFrame(raf);
    document.removeEventListener('pointermove', onMove);
    document.documentElement.removeEventListener('pointerleave', onLeave);
  };
}
