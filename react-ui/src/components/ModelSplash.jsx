import React, { useEffect, useState } from 'react';
import { getJSON } from '../api';

// Startup model check (app/roop/model_integrity.py via GET /api/models/integrity).
//
// While the backend is hashing or downloading the manifest models this covers
// the app with per-file progress, because nothing behind it can work until the
// swapper, detector and restorers exist. When the check ends it gets out of the
// way: 'ready' hides it, 'degraded' leaves a dismissible notice naming what is
// missing so a model failure is not discovered later as a cryptic ONNX error.
// A backend without the route (404) or an unreachable one hides it for good.

const POLL_MS = 500;
const STATUS_LABEL = {
  pending: 'Waiting', verifying: 'Verifying', downloading: 'Downloading',
  ok: 'Verified', corrupt: 'Failed check', missing: 'Missing',
};

const mb = (bytes) => `${((bytes || 0) / 1048576).toFixed(1)} MB`;

export default function ModelSplash() {
  const [state, setState] = useState(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    let alive = true;
    let timer = null;
    let failures = 0;
    const poll = async () => {
      try {
        const snapshot = await getJSON('/api/models/integrity', { timeout: 4000 });
        if (!alive) return;
        failures = 0;
        setState(snapshot);
        // 'idle' = the check has not started yet (the API thread comes up
        // before core.run reaches it); keep polling until it has finished.
        if (snapshot.busy || snapshot.phase === 'idle') timer = setTimeout(poll, POLL_MS);
      } catch {
        failures += 1;
        if (alive && failures < 20) timer = setTimeout(poll, POLL_MS * 2);
      }
    };
    poll();
    return () => { alive = false; clearTimeout(timer); };
  }, []);

  if (!state) return null;

  if (state.phase === 'degraded' && !dismissed) {
    return (
      <div role="alert"
           className="fixed bottom-4 left-1/2 -translate-x-1/2 z-[1000] max-w-xl w-[calc(100%-2rem)] rounded-xl border border-[var(--warn)]/40 bg-black/85 backdrop-blur px-4 py-3 text-sm text-white/90 shadow-lg flex gap-3 items-start">
        <div className="flex-1">
          <div className="font-semibold text-[var(--warn)]">Model check found problems</div>
          <div className="text-white/70 mt-0.5">{state.message}. Features that use these models will fail until they are downloaded; restart with a connection to retry.</div>
        </div>
        <button type="button" onClick={() => setDismissed(true)}
                className="text-white/60 hover:text-white px-2" aria-label="Dismiss">✕</button>
      </div>
    );
  }

  if (!state.busy) return null;

  const pct = state.download_percent;
  return (
    <div className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/80 backdrop-blur-sm"
         role="dialog" aria-modal="true" aria-label="Preparing models">
      <div className="w-[min(34rem,calc(100%-2rem))] rounded-2xl border border-white/10 bg-[#0d0f14]/95 p-6 shadow-2xl text-white">
        <div className="text-lg font-semibold">Preparing models</div>
        <div className="text-sm text-white/60 mt-1">{state.message || 'Verifying model files'}</div>
        {pct != null && (
          <div className="mt-4">
            <div className="h-2 rounded-full bg-white/10 overflow-hidden">
              <div className="h-full bg-[var(--accent)] transition-[width] duration-300"
                   style={{ width: `${Math.min(100, pct).toFixed(1)}%` }} />
            </div>
            <div className="text-xs text-white/50 mt-1 tabular-nums">
              {mb(state.download_bytes_done)} of {mb(state.download_bytes_total)} ({pct.toFixed(1)}%)
            </div>
          </div>
        )}
        <ul className="mt-4 space-y-1.5 text-sm">
          {(state.files || []).map((f) => {
            const filePct = f.bytes_total ? (100 * (f.bytes_done || 0)) / f.bytes_total : 0;
            const tone = f.status === 'ok' ? 'text-[var(--ok)]'
              : (f.status === 'corrupt' || f.status === 'missing') ? 'text-[var(--danger)]'
              : 'text-white/60';
            return (
              <li key={f.file} className="flex items-center gap-3">
                <span className="flex-1 truncate font-mono text-xs text-white/80">{f.file}</span>
                {(f.status === 'downloading' || f.status === 'verifying') && (
                  <span className="text-xs text-white/50 tabular-nums">{filePct.toFixed(0)}%</span>
                )}
                <span className={`text-xs ${tone}`}>{STATUS_LABEL[f.status] || f.status}</span>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
