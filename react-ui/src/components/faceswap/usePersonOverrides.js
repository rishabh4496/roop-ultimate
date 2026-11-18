import { useCallback, useEffect, useMemo, useState } from 'react';
import { withOverride, withoutOverride } from './routingModel';

// Per-person render overrides for ONE target clip: { <target person id>: { <wire name>: number } }.
//
// They are part of the request (FaceSwap sends them as `person_overrides` with every preview and render, and the backend
// applies them per face: app/roop/person_overrides.py), so a queued job carries the values it was queued with. Between
// sessions they live in this browser's localStorage, keyed by the clip - a convenience, not the system of record: a render
// started from another browser sends that browser's values. localStorage can throw (private window, blocked site data), so
// every read and write is wrapped and the hook works in memory without it.

const KEY = (mediaId) => `roop_person_overrides:${mediaId ?? 'none'}`;

function load(mediaId) {
  try {
    const raw = window.localStorage.getItem(KEY(mediaId));
    const parsed = raw ? JSON.parse(raw) : null;
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

export default function usePersonOverrides(mediaId) {
  const [state, setState] = useState(() => ({ mediaId, overrides: load(mediaId) }));

  // A different clip is a different set of people: swap in that clip's own overrides, never carry the last clip's over.
  useEffect(() => {
    setState((s) => (s.mediaId === mediaId ? s : { mediaId, overrides: load(mediaId) }));
  }, [mediaId]);

  const commit = useCallback((next) => {
    setState((s) => ({ ...s, overrides: next }));
    try {
      if (Object.keys(next).length) window.localStorage.setItem(KEY(mediaId), JSON.stringify(next));
      else window.localStorage.removeItem(KEY(mediaId));
    } catch { /* storage unavailable: the overrides still work for this session */ }
  }, [mediaId]);

  // `overrides` is read through the state so a burst of scrub updates composes instead of overwriting each other.
  const setOverride = useCallback((personId, key, value) => {
    setState((s) => {
      const next = withOverride(s.overrides, personId, key, value);
      try { window.localStorage.setItem(KEY(s.mediaId), JSON.stringify(next)); } catch { /* see above */ }
      return { ...s, overrides: next };
    });
  }, []);
  const clearOverride = useCallback((personId, key) => {
    setState((s) => {
      const next = withoutOverride(s.overrides, personId, key);
      try {
        if (Object.keys(next).length) window.localStorage.setItem(KEY(s.mediaId), JSON.stringify(next));
        else window.localStorage.removeItem(KEY(s.mediaId));
      } catch { /* see above */ }
      return { ...s, overrides: next };
    });
  }, []);
  const clearAll = useCallback(() => commit({}), [commit]);

  return useMemo(() => ({
    overrides: state.mediaId === mediaId ? state.overrides : {},
    setOverride, clearOverride, clearAll,
  }), [state, mediaId, setOverride, clearOverride, clearAll]);
}
