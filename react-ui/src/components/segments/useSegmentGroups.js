import { useCallback, useEffect, useRef, useState } from 'react';
import { getJSON, postJSON } from '../../api';
import { isActive, visibleGroups } from './segmentModel';

// The server's view of every split render (`GET /api/queue/segments`): per slice its state, its file on
// disk, and the join with its verification. Kept apart from useQueue on purpose - the queue answers "what
// is each job doing"; this answers "what do I physically have", including whether a "completed" slice's
// file is still there.
//
// Polls only while something in a split is queued or rendering (or a run is in flight); settled, it is
// quiet and refreshes on mount and on the window regaining focus, like the queue does.

const POLL_MS = 2000;

export default function useSegmentGroups({ notify, running = false } = {}) {
  const [groups, setGroups] = useState([]);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState('');                 // 'rerender:<jobId>' | 'join:<group>' | ''
  const alive = useRef(true);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);

  const refresh = useCallback(async () => {
    try {
      const res = await getJSON('/api/queue/segments', { scope: 'job' });
      if (alive.current) { setGroups(visibleGroups(res.groups)); setLoaded(true); }
    } catch { if (alive.current) setLoaded(true); }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    const onFocus = () => refresh();
    window.addEventListener('focus', onFocus);
    return () => window.removeEventListener('focus', onFocus);
  }, [refresh]);

  const active = running || isActive(groups);
  useEffect(() => {
    if (!active) return undefined;
    const id = setInterval(refresh, POLL_MS);
    return () => clearInterval(id);
  }, [active, refresh]);

  const rerender = useCallback(async (jobId, label) => {
    setBusy(`rerender:${jobId}`);
    try {
      // start:true runs the queue if it is idle, so one click redoes the slice rather than parking it.
      const res = await postJSON('/api/queue/segments/rerender', { id: jobId, start: true });
      if (alive.current) setGroups(visibleGroups(res.groups));
      notify?.(`Re-rendering ${label || 'the chunk'}`);
    } catch (e) { notify?.(e.message, 'error'); }
    finally { if (alive.current) setBusy(''); }
  }, [notify]);

  const join = useCallback(async (group) => {
    setBusy(`join:${group}`);
    try {
      const res = await postJSON('/api/queue/segments/join', { group }, { timeout: 0 });
      if (alive.current) setGroups(visibleGroups(res.groups));
      const v = res.verify;
      if (v && v.ok === false) notify?.(`Joined ${res.name}, but ${v.reason || 'the duration does not match'}`, 'error');
      else notify?.(`Joined ${res.name}${v?.ok ? ' · duration verified' : ''}${res.audio === 'copy' ? ' · source audio copied' : ''}`, 'success');
      return res;
    } catch (e) { notify?.(e.message, 'error'); await refresh(); return null; }
    finally { if (alive.current) setBusy(''); }
  }, [notify, refresh]);

  return { groups, loaded, busy, refresh, rerender, join };
}
