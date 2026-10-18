import React, { useEffect, useState } from 'react';
import { AnimatePresence, motion, spring } from '../motion';
import { STAGE_COMPILING_TRT, STAGE_RUNNING_CANARY } from '../pipelineStatus';
import { usePipelineStatus } from '../usePipelineStatus';

// The backend is building a TensorRT engine or running the fidelity canary that follows it. Both
// take minutes on a cold cache and have no frame counter, so without this the app looks hung and
// (before api.js learned to suspend its deadline) every request timed out. Non-blocking: a status
// strip in the corner, never a modal; the UI stays usable underneath.
//
// Fed by pipelineStatus.js, which the app shell starts once (it outlives route changes, so
// navigating between tabs never drops or re-opens the socket).

const LABEL = {
  [STAGE_COMPILING_TRT]: 'Compiling TensorRT engine',
  [STAGE_RUNNING_CANARY]: 'Verifying engine fidelity',
};

const clock = (s) => {
  const t = Math.max(0, Math.round(s));
  return t >= 60 ? `${Math.floor(t / 60)}m ${String(t % 60).padStart(2, '0')}s` : `${t}s`;
};

export default function CompileBanner() {
  const { busy, active } = usePipelineStatus();
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!busy) return undefined;
    const id = setInterval(() => setTick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, [busy]);

  return (
    <AnimatePresence>
      {busy && active.length > 0 && (
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
          {active.map((a) => {
            // `since` is the backend's wall clock; elapsed_s was true when the snapshot was taken.
            // Counting from `since` keeps the clock moving between snapshots.
            const elapsed = a.since ? (Date.now() - a.since) / 1000 : a.elapsedS;
            return (
              <div key={`${a.kind}:${a.model}`} className="flex items-center gap-3 min-w-0" data-stage={a.kind}>
                <span className="h-2 w-2 shrink-0 rounded-full bg-cyan-400 animate-ping" aria-hidden="true" />
                <span className="text-xs font-semibold text-cyan-100 truncate">
                  {LABEL[a.kind] || 'Preparing engine'}
                  {': '}
                  <span data-field="model" className="font-mono">{a.model}</span>
                  {a.precision && (
                    <>
                      {' · '}
                      <span data-field="precision" className="font-mono">{a.precision}</span>
                    </>
                  )}
                </span>
                {/* Ticking text inside a live region would be re-announced every second. */}
                <span aria-hidden="true" data-tick={tick} className="text-mini font-mono text-slate-300 shrink-0">{clock(elapsed)}</span>
              </div>
            );
          })}
          <p className="text-mini text-slate-300">
            First launch only. Requests keep waiting; nothing has failed.
          </p>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
