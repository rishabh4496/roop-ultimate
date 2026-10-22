import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useApi } from '../../useApi';
import { reportFailure } from '../../failureLog';
import { benchPhase } from './providerMatrix';
import ProviderMatrixView from './ProviderMatrixView';

// "Benchmark Pipeline Providers": one click measures each part of the pipeline on the CPU, on the primary
// provider (TensorRT when active) and on native OpenVINO when an Intel accelerator exists, plus the warp/blend
// and optical-flow backends. It drives the `providers` profile of roop/bench.py through the existing
// /api/settings/benchmark_* endpoints; that profile only REPORTS, it never writes the config.
//
// It needs an idle GPU, so the button is disabled while a render runs (the backend refuses with 409 as well).

const POLL_MS = 1000;

export default function ProviderBenchmark({ processing = false, className = '' }) {
  const { getJSON, postJSON } = useApi();
  const [state, setState] = useState(null);
  const [error, setError] = useState('');
  const pollRef = useRef(null);

  const stop = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  }, []);
  useEffect(() => stop, [stop]);

  const poll = useCallback(() => {
    stop();
    pollRef.current = setInterval(async () => {
      try {
        const s = await getJSON('/api/settings/benchmark_status', { timeout: 5000 });
        setState(s);
        if (!s.running) stop();
      } catch (err) {
        reportFailure('Polling the provider benchmark', err, { toast: false });
      }
    }, POLL_MS);
  }, [getJSON, stop]);

  const start = async () => {
    setError('');
    try {
      const resp = await postJSON('/api/settings/benchmark_threads', { profile: 'providers' });
      if (resp && resp.status === 'started') {
        setState({ running: true, profile: 'providers', progress: 0, status_msg: 'Loading models…', logs: [] });
        poll();
      } else if (resp && resp.status === 'running') {
        poll();
      } else {
        throw new Error((resp && resp.message) || 'Could not start the provider benchmark');
      }
    } catch (err) {
      setError(err.message || 'Could not start the provider benchmark');
    }
  };

  const cancel = async () => {
    try { await postJSON('/api/settings/benchmark_cancel', {}); } catch (err) {
      reportFailure('Cancelling the provider benchmark', err, { toast: false });
    }
  };

  const phase = benchPhase(state);
  const running = phase === 'running';
  const matrix = phase === 'done' ? state.result.provider_matrix : null;
  const lastLog = state && Array.isArray(state.logs) && state.logs.length ? state.logs[state.logs.length - 1] : '';

  return (
    <details className={`rounded-2xl border border-white/10 bg-white/[0.02] ${className}`} data-testid="provider-benchmark">
      <summary className="cursor-pointer list-none flex flex-wrap items-center gap-x-4 gap-y-1 px-3 py-2">
        <span className="text-nano font-bold uppercase tracking-[0.16em] text-white/60">Pipeline providers</span>
        <span className="text-micro text-muted">CPU vs TensorRT vs OpenVINO, per stage</span>
      </summary>
      <div className="px-3 pb-3">
        <div className="flex flex-wrap items-center gap-2">
          <button type="button" onClick={start} disabled={processing || running}
                  title={processing ? 'A render is in progress — the benchmark needs an idle GPU' : undefined}
                  className="rounded-lg border border-white/15 bg-white/[0.06] px-3 py-1.5 text-micro font-semibold text-white/90 hover:bg-white/10 disabled:opacity-50 disabled:cursor-not-allowed">
            {running ? 'Benchmarking…' : 'Benchmark Pipeline Providers'}
          </button>
          {running && (
            <button type="button" onClick={cancel}
                    className="rounded-lg border border-white/10 px-3 py-1.5 text-micro text-white/70 hover:bg-white/5">
              Cancel
            </button>
          )}
          {running && state && (
            <span className="text-micro font-mono text-muted" aria-live="polite">
              {state.status_msg || ''}{typeof state.progress === 'number' ? ` · ${state.progress}%` : ''}
            </span>
          )}
        </div>
        {running && lastLog && <div className="mt-1 text-nano font-mono text-muted truncate" title={lastLog}>{lastLog}</div>}
        {(error || phase === 'failed') && (
          <div role="alert" className="mt-2 text-micro text-red-300">{error || state.error}</div>
        )}
        {phase === 'cancelled' && <div className="mt-2 text-micro text-muted">Cancelled.</div>}
        {phase === 'idle' && !error && (
          <p className="mt-2 text-nano text-muted leading-snug">
            Measures this machine once, with the GPU idle (a few minutes). It reports only; no setting is changed.
          </p>
        )}
        {matrix && <ProviderMatrixView matrix={matrix} />}
      </div>
    </details>
  );
}
