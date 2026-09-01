import { useCallback, useEffect, useState } from 'react';
import { getJSON, postJSON } from '../api';

const EMPTY = { schema_version: 2, job_states: [], jobs: [], running: false, paused: false, current: null };

export const JOB_STATE_LABELS = Object.freeze({
  QUEUED: 'Queued',
  PREPARING: 'Preparing',
  PROCESSING: 'Processing',
  PAUSE_REQUESTED: 'Pause requested',
  PAUSED: 'Paused',
  COMPLETED: 'Completed',
  FAILED: 'Failed',
  CANCELLED: 'Cancelled',
  INTERRUPTED: 'Interrupted',
  RECOVERABLE: 'Recoverable',
});

export function useQueue(notify) {
  const [queue, setQueue] = useState(EMPTY);
  const [busy, setBusy] = useState('');

  const refresh = useCallback(async () => {
    const next = await getJSON('/api/queue');
    setQueue(next || EMPTY);
    return next;
  }, []);

  useEffect(() => {
    refresh().catch(() => {});
  }, [refresh]);

  useEffect(() => {
    if (!queue.running && !queue.current) return undefined;
    const timer = window.setInterval(() => refresh().catch(() => {}), 1000);
    return () => window.clearInterval(timer);
  }, [queue.current, queue.running, refresh]);

  const act = useCallback(async (name, path, body = {}) => {
    setBusy(name);
    try {
      const next = await postJSON(path, body);
      setQueue(next || EMPTY);
      return next;
    } catch (cause) {
      notify?.(cause.message || `Queue ${name} failed`, 'danger');
      throw cause;
    } finally {
      setBusy('');
    }
  }, [notify]);

  return {
    ...queue,
    busy,
    refresh,
    add: (job) => act('add', '/api/queue/add', job),
    addBatch: (jobs) => act('addBatch', '/api/queue/add_batch', { jobs }),
    start: () => act('start', '/api/queue/start'),
    pause: () => act('pause', '/api/queue/pause'),
    resume: () => act('resume', '/api/queue/resume'),
    stop: () => act('stop', '/api/queue/stop'),
    cancel: (id) => act(`cancel:${id}`, '/api/queue/cancel', { id }),
    remove: (id) => act(`remove:${id}`, '/api/queue/remove', { id }),
    retry: (id) => act(`retry:${id || 'all'}`, '/api/queue/retry', id ? { id } : {}),
    reorder: (ids) => act('reorder', '/api/queue/reorder', { ids }),
  };
}
