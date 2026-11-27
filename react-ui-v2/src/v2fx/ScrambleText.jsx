import React, { useEffect, useRef, useState } from 'react';
import { prefersReducedMotion } from './fxStore';

// Decrypt-style text: when `text` changes, each character cycles through glyphs and locks in left to
// right. The final string is always what screen readers get (aria-label), and the effect is skipped
// under reduced motion.
const GLYPHS = '!<>-_\\/[]{}=+*^?#░▒▓01';

export default function ScrambleText({ text, className = '', duration = 520 }) {
  const [out, setOut] = useState(text);
  const raf = useRef(0);

  useEffect(() => {
    const target = String(text ?? '');
    if (prefersReducedMotion()) { setOut(target); return undefined; }
    const start = performance.now();
    const step = (now) => {
      const p = Math.min(1, (now - start) / duration);
      const locked = Math.floor(p * target.length);
      let s = target.slice(0, locked);
      for (let i = locked; i < target.length; i++) {
        s += target[i] === ' ' ? ' ' : GLYPHS[(Math.random() * GLYPHS.length) | 0];
      }
      setOut(s);
      if (p < 1) raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf.current);
  }, [text, duration]);

  return (
    <span className={className} aria-label={text}>
      <span aria-hidden="true">{out}</span>
    </span>
  );
}
