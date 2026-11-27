import React, { useEffect, useState } from 'react';
import { Film } from 'lucide-react';

// A target frame (/api/target/preview). While a render is analysing, the engine refuses frame decodes
// with 503 + retry_after_ms on purpose (they would share the GPU burst), so a frame can fail and be
// fine a few seconds later. Instead of a broken-image icon this shows a placeholder and retries with
// back-off (2 s, 4 s, 8 s ... capped at 30 s).
export default function FrameImg({ src, alt = '', className = '', ...rest }) {
  const [attempt, setAttempt] = useState(0);
  const [failed, setFailed] = useState(false);
  useEffect(() => { setAttempt(0); setFailed(false); }, [src]);
  useEffect(() => {
    if (!failed) return undefined;
    const h = setTimeout(() => { setFailed(false); setAttempt((a) => a + 1); }, Math.min(30000, 2000 * 2 ** Math.min(attempt, 4)));
    return () => clearTimeout(h);
  }, [failed, attempt]);
  if (!src) return null;
  if (failed) {
    return (
      <span className={`grid place-items-center bg-[repeating-linear-gradient(135deg,rgba(255,255,255,.03)_0_6px,transparent_6px_12px)] ${className}`} title="Frame not available yet (the engine may be busy) - retrying" aria-label={alt || 'frame loading'}>
        <Film size={14} className="text-[var(--fg-faint)]" />
      </span>
    );
  }
  const url = attempt ? `${src}${src.includes('?') ? '&' : '?'}r=${attempt}` : src;
  return <img src={url} alt={alt} className={className} onError={() => setFailed(true)} {...rest} />;
}
