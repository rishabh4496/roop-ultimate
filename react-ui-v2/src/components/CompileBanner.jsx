import React, { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion, spring } from '../motion';
import { RECEIPT_MS, STAGE_COMPILING_TRT, STAGE_RUNNING_CANARY, clock, receiptText } from '../pipelineStatus';
import { usePipelineStatus } from '../usePipelineStatus';

// The backend is building a TensorRT engine or running the fidelity canary that follows it. Both
// take minutes on a cold cache and have no frame counter, so without this the app looks hung and
// (before api.js learned to suspend its deadline) every request timed out. Non-blocking: a status
// strip in the corner, never a modal; the UI stays usable underneath.
//
// Each build row names the model, the precision of the engine that is actually compiling, the
// workspace the provider was given and the cache directory it builds into (all read from the
// session's own provider options, so a value the backend could not read is simply not shown).
// When a build ends the banner stays a few seconds as a RECEIPT: a cache hit (the engine was
// deserialised) or a miss (it was built, and how long that took).
//
// Fed by pipelineStatus.js, which the app shell starts once (it outlives route changes, so
// navigating between tabs never drops or re-opens the socket).

const LABEL = {
  [STAGE_COMPILING_TRT]: 'Compiling TensorRT engine',
  [STAGE_RUNNING_CANARY]: 'Verifying engine fidelity',
};


export default function CompileBanner() {
  const { busy, active, lastCompile } = usePipelineStatus();
  const [tick, setTick] = useState(0);
  const [receipt, setReceipt] = useState(null);
  const wasBusy = useRef(false);
  const shownAt = useRef(null);

  useEffect(() => {
    if (!busy) return undefined;
    const id = setInterval(() => setTick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, [busy]);

  // A receipt is for a build this page WATCHED end (busy -> idle), never for one that finished
  // before it connected: the backend keeps its last compile for the whole session.
  // The timer lives in a ref, not in this effect's cleanup: `lastCompile` is a new object on every
  // snapshot, so the effect re-runs while the receipt is up and a cleanup would cancel its own expiry.
  const expiry = useRef(null);
  useEffect(() => () => clearTimeout(expiry.current), []);
  useEffect(() => {
    if (busy) { wasBusy.current = true; return; }
    if (!wasBusy.current) return;
    wasBusy.current = false;
    if (!lastCompile || lastCompile.at === shownAt.current) return;
    shownAt.current = lastCompile.at;
    setReceipt(lastCompile);
    clearTimeout(expiry.current);
    expiry.current = setTimeout(() => setReceipt(null), RECEIPT_MS);
  }, [busy, lastCompile]);

  const showing = (busy && active.length > 0) || receipt;
  return (
    <AnimatePresence>
      {showing && (
        <motion.div
          key="compile-banner"
          initial={{ opacity: 0, y: 16, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 16, scale: 0.96 }}
          transition={spring.snappy}
          role="status"
          aria-atomic="true"
          data-testid="compile-banner"
          className="fixed bottom-20 left-6 z-50 max-w-[min(30rem,calc(100vw-3rem))] px-4 py-3 rounded-xl bg-[#0E0F15]/95 backdrop-blur-xl border border-cyan-400/25 shadow-2xl flex flex-col gap-1.5"
        >
          {busy && active.map((a) => {
            // `since` is the backend's wall clock; elapsed_s was true when the snapshot was taken.
            // Counting from `since` keeps the clock moving between snapshots.
            const elapsed = a.since ? (Date.now() - a.since) / 1000 : a.elapsedS;
            return (
              <div key={`${a.kind}:${a.model}`} className="flex flex-col gap-0.5 min-w-0" data-stage={a.kind}>
                <div className="flex items-center gap-3 min-w-0">
                  <span className="h-2 w-2 shrink-0 rounded-full bg-cyan-400 animate-ping" aria-hidden="true" />
                  <span className="text-xs font-semibold text-cyan-100 truncate">
                    {LABEL[a.kind] || 'Preparing engine'}
                    {': '}
                    <span data-field="model" className="font-mono">{a.model}</span>
                    {a.precision && (
                      <>
                        {' · '}
                        <span data-field="precision" className="font-mono rounded bg-cyan-400/15 px-1">{a.precision}</span>
                      </>
                    )}
                  </span>
                  {/* Ticking text inside a live region would be re-announced every second. */}
                  <span aria-hidden="true" data-tick={tick} className="text-mini font-mono text-slate-300 shrink-0">{clock(elapsed)}</span>
                </div>
                {(a.workspaceMb || a.cachePath) && (
                  <p className="pl-5 text-mini font-mono text-slate-300 truncate" title={a.cachePath || ''}>
                    {a.workspaceMb && <span data-field="workspace">workspace {a.workspaceMb} MB</span>}
                    {a.workspaceMb && a.cachePath && ' · '}
                    {a.cachePath && <span data-field="cache">cache {a.cachePath}</span>}
                  </p>
                )}
              </div>
            );
          })}
          {busy && active.length > 0 && (
            <p className="text-mini text-slate-300">
              First launch only. Requests keep waiting; nothing has failed.
            </p>
          )}
          {receipt && !busy && (
            <p data-testid="compile-receipt" data-cache-hit={String(receipt.cacheHit)}
              className={`text-xs font-semibold ${receipt.ok ? 'text-cyan-100' : 'text-red-300'}`}>
              <span className="font-mono">{receipt.model}</span>
              {' · '}
              <span className="font-mono">{receipt.precision}</span>
              {' — '}
              {receiptText(receipt)}
            </p>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
