// The render: its live state, and Start / Pause / Resume / Stop.
//
// Two feeds keep it current, as in React UI 1.0: the /ws/telemetry socket (per-frame numbers,
// into 1.0's telemetry store so its progress selectors work unchanged) and a /api/progress poll -
// every 2 s while rendering, every 6 s idle - which also carries the log tail and output, and is
// how this window notices a render it did not start (the batch queue, another window).
import { useCallback, useEffect, useRef } from 'react';
import { create } from 'zustand';
import { getJSON, postJSON, timeoutFor } from '@engine/api';
import useTelemetrySocket from '@engine/useTelemetrySocket';
import { setRunTelemetry, setPipelineTelemetry, setDashboardTelemetry, clearEngineTelemetry } from '@engine/store/telemetryStore';
import { usePipelineStatusSocket } from '@engine/usePipelineStatus';
import { useStudio } from './studio';
import { buildEnginePayload, readiness } from './payload';
import { notify } from '../ui/controls';
import { termLog } from '../fx/termFeed';

// Spelled out (not `/api/${action}`) so every endpoint this client calls can be found by name.
const CONTROL_PATH = { pause: '/api/pause', resume: '/api/resume', stop: '/api/stop' };

const IDLE = { processing: false, paused: false, pause_requested: false, stop_requested: false, desc: '', progress: 0, log: [], error: null, output: null, started_at: 0 };

export const useRun = create((set, get) => ({
  progress: IDLE,
  online: true,
  socket: false,
  busy: '',

  merge: (patch) => set((s) => ({ progress: { ...s.progress, ...patch } })),

  start: async () => {
    const st = useStudio.getState();
    const why = readiness(st);
    if (why) { notify(why, 'warn'); return false; }
    set({ busy: 'start' });
    try {
      await postJSON('/api/swap', buildEnginePayload(st), { timeout: 0 });
      get().merge({ processing: true, paused: false, pause_requested: false, stop_requested: false, error: null, desc: 'Starting…' });
      termLog(`render started · ${st.targets[st.selectedTarget]?.name || 'target'} · ${st.settings?.swap_model}`, 'ok', 'run');
      return true;
    } catch (e) { notify(e.message, 'error'); return false; } finally { set({ busy: '' }); }
  },
  control: async (action) => {
    if (get().busy || !CONTROL_PATH[action]) return;
    set({ busy: action });
    const optimistic = {
      pause: { pause_requested: true, desc: 'Pause requested…' },
      resume: { pause_requested: false, paused: false, desc: 'Resuming…' },
      stop: { stop_requested: true, desc: 'Stopping…' },
    }[action];
    get().merge(optimistic);
    try {
      const path = CONTROL_PATH[action];
      const res = await postJSON(path, {}, { timeout: timeoutFor('POST', path) });
      if (action === 'pause') {
        const p = res?.pause || {};
        get().merge({ pause_requested: p.requested !== undefined ? !!p.requested : true, paused: !!p.acknowledged, desc: p.acknowledged ? 'Paused' : 'Pause requested…' });
        notify('Pause requested - the engine holds at the next safe checkpoint', 'info');
      } else if (action === 'resume') {
        notify('Resumed');
      } else if (action === 'stop') {
        notify('Stopping - the output is finalized as a playable video', 'info');
      }
      termLog(`${action} requested`, 'info', 'run');
    } catch (e) {
      notify(e.message, 'error');
      get().merge({ stop_requested: false, pause_requested: false });
    } finally { set({ busy: '' }); }
  },
}));

/** Mount once (App): socket + poll. */
export function useRunSync() {
  const structural = useRef('');
  const onFrame = useCallback((frame) => {
    if (!frame || !['progress', 'hello', 'heartbeat'].includes(frame.event)) return;
    setRunTelemetry(frame);
    setPipelineTelemetry(frame);
    setDashboardTelemetry(frame);
    const patch = {
      processing: frame.processing, paused: frame.paused, error: frame.error, started_at: frame.started_at,
      ...(frame.pause_requested !== undefined ? { pause_requested: frame.pause_requested } : {}),
      ...(frame.stop_requested !== undefined ? { stop_requested: frame.stop_requested } : {}),
    };
    const key = JSON.stringify(patch);
    if (key !== structural.current) {
      structural.current = key;
      useRun.getState().merge({ ...patch, desc: frame.desc, progress: frame.progress });
    }
  }, []);
  const { connected } = useTelemetrySocket(onFrame, true);
  usePipelineStatusSocket();
  useEffect(() => {
    useRun.setState({ socket: connected });
    if (!connected) clearEngineTelemetry();
  }, [connected]);

  const processing = useRun((s) => s.progress.processing);
  useEffect(() => {
    let alive = true;
    let fails = 0;
    const tick = async () => {
      try {
        const pr = await getJSON('/api/progress', { timeout: 8000 });
        if (!alive) return;
        fails = 0;
        useRun.setState({ online: true });
        setRunTelemetry(pr.processing ? pr : { ...pr, current_frame: 0, total_frames: 0, fps: 0, fps_now: null });
        useRun.getState().merge(pr);
      } catch {
        fails += 1;
        if (fails >= 3) useRun.setState({ online: false });
      }
    };
    tick();
    const h = setInterval(tick, processing ? 2000 : 6000);
    const onVis = () => { if (!document.hidden) tick(); };
    document.addEventListener('visibilitychange', onVis);
    return () => { alive = false; clearInterval(h); document.removeEventListener('visibilitychange', onVis); };
  }, [processing]);

  // Edges, for the terminal and a toast when a render ends.
  const was = useRef(false);
  const err = useRun((s) => s.progress.error);
  const output = useRun((s) => s.progress.output);
  useEffect(() => {
    if (processing && !was.current) termLog('render running — ambient animation held while the engine has the GPU', 'ok', 'run');
    if (!processing && was.current) {
      if (err) { termLog(`render ended: ${err}`, 'error', 'run'); notify(`Render ended: ${err}`, 'error'); }
      else { termLog(`render finished${output ? ` → ${output}` : ''}`, 'ok', 'run'); notify('Render finished'); }
    }
    was.current = !!processing;
  }, [processing, err, output]);
}
