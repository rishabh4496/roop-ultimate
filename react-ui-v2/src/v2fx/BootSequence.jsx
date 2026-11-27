import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useFxStore, prefersReducedMotion } from './fxStore';
import { termLog } from './termFeed';

// ── Boot sequence ───────────────────────────────────────────────────────────
//
// The first load of a session opens on a terminal that prints what is ACTUALLY happening: the
// connection to the local engine, the engine's version from /api/meta, the settings arriving. A
// step is only marked [ ok ] when the thing it names has happened (App passes `meta`, `settings`
// and `error` straight through) - it is a progress display, not an animation pretending to be one.
//
// It never blocks: any key or click skips it, it gets out of the way the moment the app is ready,
// it gives up after 8 s whatever state the engine is in (App's own error/retry UI is underneath),
// and it does not run at all under prefers-reduced-motion, when switched off, or a second time in
// one browser session.

const SESSION_KEY = 'roop_v2_booted';
const MIN_MS = 1100;
const MAX_MS = 8000;

const alreadyBooted = () => {
  try { return window.sessionStorage.getItem(SESSION_KEY) === '1'; } catch { return false; }
};

function Step({ state, children }) {
  const tag = { ok: '[ ok ]', run: '[ .. ]', fail: '[fail]', info: '[info]' }[state];
  return (
    <div className={`v2-boot-step is-${state}`}>
      <span className="v2-boot-tag">{tag}</span> <span className="v2-boot-text">{children}</span>
    </div>
  );
}

export default function BootSequence({ meta, settings, error }) {
  const enabled = useFxStore((s) => s.boot);
  const [show, setShow] = useState(() => enabled && !alreadyBooted() && !prefersReducedMotion());
  const started = useRef(Date.now());
  const ready = !!(meta && settings);

  useEffect(() => {
    if (!show) return undefined;
    try { window.sessionStorage.setItem(SESSION_KEY, '1'); } catch { /* storage blocked: it just boots again */ }
    const skip = () => setShow(false);
    const cap = setTimeout(skip, MAX_MS);
    window.addEventListener('keydown', skip);
    return () => { clearTimeout(cap); window.removeEventListener('keydown', skip); };
  }, [show]);

  useEffect(() => {
    if (!show || !(ready || error)) return undefined;
    const wait = Math.max(0, MIN_MS - (Date.now() - started.current)) + (error ? 900 : 380);
    const h = setTimeout(() => setShow(false), wait);
    return () => clearTimeout(h);
  }, [show, ready, error]);

  // The same facts go to the terminal dock's feed, so the boot is on record after the overlay goes.
  const logged = useRef({});
  useEffect(() => {
    const once = (k, msg, lvl) => { if (!logged.current[k]) { logged.current[k] = true; termLog(msg, lvl, 'boot'); } };
    once('start', `react ui 2.0 · ${window.location.origin}`, 'info');
    if (meta) once('meta', `engine connected${meta.git_version ? ` · ${meta.git_version}` : ''}${meta.installed_commit?.short ? ` (${String(meta.installed_commit.short).slice(0, 7)})` : ''}`, 'ok');
    if (settings) once('settings', `settings loaded · ${Object.keys(settings).length} keys`, 'ok');
    if (error) once('error', `engine unreachable: ${error}`, 'error');
  }, [meta, settings, error]);

  const origin = typeof window !== 'undefined' ? window.location.host : '';
  const commit = meta?.installed_commit?.short ? String(meta.installed_commit.short).slice(0, 7) : null;

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          className="v2-boot"
          role="status"
          aria-live="polite"
          onClick={() => setShow(false)}
          initial={{ opacity: 1 }}
          exit={{ opacity: 0, scale: 1.04, transition: { duration: 0.45, ease: [0.4, 0, 0.2, 1] } }}
        >
          <motion.div
            className="v2-boot-card"
            initial={{ opacity: 0, y: 14, rotateX: 12 }}
            animate={{ opacity: 1, y: 0, rotateX: 0, transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] } }}
            exit={{ opacity: 0, y: -10, rotateX: -8, transition: { duration: 0.3 } }}
          >
            <div className="v2-boot-head">
              <span className="v2-term-dots" aria-hidden="true"><i /><i /><i /></span>
              <span>roop-ultimate — boot</span>
            </div>
            <div className="v2-boot-logo v2-sunray">ROOP//ULTIMATE</div>
            <div className="v2-boot-sub">react ui 2.0 · local engine at {origin}</div>
            <div className="v2-boot-log">
              <Step state="ok">interface bundle loaded</Step>
              <Step state={meta ? 'ok' : error ? 'fail' : 'run'}>
                {meta ? `engine online${meta.git_version ? ` · ${meta.git_version}` : ''}${commit ? ` (${commit})` : ''}` : error ? `engine unreachable — ${error}` : 'connecting to local engine…'}
              </Step>
              <Step state={settings ? 'ok' : error ? 'fail' : 'run'}>
                {settings ? `settings loaded · ${Object.keys(settings).length} keys` : 'reading settings…'}
              </Step>
              {ready && <Step state="ok">workstation ready</Step>}
              <div className="v2-boot-prompt">❯ <span className="v2-caret" aria-hidden="true" /></div>
            </div>
            <div className="v2-boot-skip">press any key to skip</div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
