import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useFxStore, TRANSITIONS, AMBIENT_QUALITY, PALETTES } from './fxStore';
import { termLog } from './termFeed';

// The header's FX button and its popover: every 2.0 effect, switchable, per browser. Nothing in
// here is an engine setting (see fxStore.js).

function Toggle({ k, label, hint }) {
  const on = useFxStore((s) => s[k]);
  const toggleFx = useFxStore((s) => s.toggleFx);
  return (
    <button type="button" role="switch" aria-checked={!!on} className="v2-fx-row"
      onClick={() => { toggleFx(k); termLog(`fx ${k} ${on ? 'off' : 'on'}`, 'ok', 'fx'); }}>
      <span className="min-w-0">
        <span className="v2-fx-label">{label}</span>
        {hint && <span className="v2-fx-hint">{hint}</span>}
      </span>
      <span className={`v2-switch ${on ? 'is-on' : ''}`} aria-hidden="true"><i /></span>
    </button>
  );
}

export default function FxPanel() {
  const [open, setOpen] = useState(false);
  const transition = useFxStore((s) => s.transition);
  const quality = useFxStore((s) => s.ambientQuality);
  const palette = useFxStore((s) => s.palette);
  const setFx = useFxStore((s) => s.setFx);
  const resetFx = useFxStore((s) => s.resetFx);
  const wrap = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => { if (wrap.current && !wrap.current.contains(e.target)) setOpen(false); };
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('pointerdown', onDown);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('pointerdown', onDown); document.removeEventListener('keydown', onKey); };
  }, [open]);

  return (
    <div ref={wrap} className="relative">
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} aria-haspopup="dialog"
        title="Visual effects (React UI 2.0)" aria-label="Visual effects"
        className={`v2-fx-btn ${open ? 'is-on' : ''}`} data-testid="v2-fx-toggle">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M12 3l1.9 5.8L20 10l-5 3.6L16.2 20 12 16.6 7.8 20 9 13.6 4 10l6.1-1.2z" />
        </svg>
        <span className="max-2xl:hidden">FX</span>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            role="dialog" aria-label="Visual effects"
            className="v2-fx-pop custom-scrollbar"
            initial={{ opacity: 0, y: -8, scale: 0.96, rotateX: -10, transformPerspective: 900 }}
            animate={{ opacity: 1, y: 0, scale: 1, rotateX: 0, transition: { duration: 0.28, ease: [0.16, 1, 0.3, 1] } }}
            exit={{ opacity: 0, y: -6, scale: 0.98, transition: { duration: 0.14 } }}
          >
            <div className="v2-fx-title">
              <span className="v2-sunray">visual effects</span>
              <button type="button" className="v2-chip" onClick={() => { resetFx(); termLog('fx reset to defaults', 'ok', 'fx'); }}>reset</button>
            </div>

            <div className="v2-fx-section">3D ambient</div>
            <Toggle k="ambient" label="Horizon" hint="WebGL planet limb, stars and atmosphere" />
            <div className="v2-fx-seg" role="radiogroup" aria-label="Horizon quality">
              {AMBIENT_QUALITY.map((q) => (
                <button key={q.id} type="button" role="radio" aria-checked={quality === q.id}
                  className={`v2-chip ${quality === q.id ? 'is-on' : ''}`} onClick={() => setFx({ ambientQuality: q.id })}>
                  {q.label.toLowerCase()} · {q.fps}fps
                </button>
              ))}
            </div>
            <Toggle k="ambientDuringRender" label="Animate during renders" hint="Off: one still frame while the engine has the GPU" />
            <Toggle k="parallax" label="Pointer parallax" />

            <div className="v2-fx-section">Transitions</div>
            <div className="v2-fx-grid" role="radiogroup" aria-label="Tab transition">
              {TRANSITIONS.map((t) => (
                <button key={t.id} type="button" role="radio" aria-checked={transition === t.id} title={t.hint}
                  className={`v2-fx-tile ${transition === t.id ? 'is-on' : ''}`}
                  onClick={() => { setFx({ transition: t.id }); termLog(`transition → ${t.label}`, 'ok', 'fx'); }}>
                  <span className={`v2-fx-demo demo-${t.id}`} aria-hidden="true"><i /></span>
                  {t.label}
                </button>
              ))}
            </div>

            <div className="v2-fx-section">Terminal</div>
            <Toggle k="terminal" label="Terminal dock" hint="Ctrl ` to expand" />
            <Toggle k="typewriter" label="Typewriter lines" />
            <Toggle k="crt" label="CRT scanlines" />
            <Toggle k="boot" label="Boot sequence" hint="Once per browser session" />

            <div className="v2-fx-section">Palette</div>
            <div className="v2-fx-seg" role="radiogroup" aria-label="Palette">
              {PALETTES.map((pl) => (
                <button key={pl.id} type="button" role="radio" aria-checked={palette === pl.id}
                  className={`v2-chip ${palette === pl.id ? 'is-on' : ''}`} onClick={() => setFx({ palette: pl.id })}>
                  <span aria-hidden="true" style={{ width: 10, height: 10, borderRadius: 9, background: `linear-gradient(135deg, ${pl.a}, ${pl.b})` }} />
                  {pl.label.toLowerCase()}
                </button>
              ))}
            </div>

            <div className="v2-fx-section">Surfaces</div>
            <Toggle k="sunray" label="Title light sweep" />
            <Toggle k="grain" label="Film grain" />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
