import { useCallback, useEffect, useState } from 'react';
import { getHardwareProfile, getJSON, getRuntimeState, getSystemProfile } from '../api';

const EMPTY = {
  runtime: null,
  hardware: null,
  profile: null,
  meta: null,
  loading: true,
  error: '',
  updatedAt: 0,
};

export function useOperationsStatus(intervalMs = 5000) {
  const [status, setStatus] = useState(EMPTY);

  const refresh = useCallback(async () => {
    setStatus((current) => ({ ...current, loading: true }));
    const results = await Promise.allSettled([
      getRuntimeState(),
      getHardwareProfile(),
      getSystemProfile(),
      getJSON('/api/meta'),
    ]);
    const [runtime, hardware, profile, meta] = results.map((result) => (
      result.status === 'fulfilled' ? result.value : null
    ));
    const failures = results.filter((result) => result.status === 'rejected');
    const message = failures.length === results.length
      ? (failures[0]?.reason?.message || 'Backend unavailable')
      : failures.length
        ? 'Some runtime evidence is unavailable'
        : '';
    setStatus({ runtime, hardware, profile, meta, loading: false, error: message, updatedAt: Date.now() });
    return { runtime, hardware, profile, meta };
  }, []);

  useEffect(() => {
    refresh().catch(() => {});
    if (!intervalMs) return undefined;
    const timer = window.setInterval(() => refresh().catch(() => {}), intervalMs);
    return () => window.clearInterval(timer);
  }, [intervalMs, refresh]);

  return { ...status, refresh };
}
