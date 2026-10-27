import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { addOverride, normalizeOverrides, removeOverride, updateOverride } from './overrides';
import { recordRender } from './laneModel';

// Per-clip persistence for the timeline's own state, in the same style as useSegments: a
// convenience layered over the clip, kept in localStorage, and safe to lose.

const overridesKey = (k) => `roop_overrides_${k || 'default'}`;
const enabledKey = (k) => `roop_overrides_enabled_${k || 'default'}`;
const NONE = Object.freeze([]);
const renderLogKey = (k) => `roop_renderlog_${k || 'default'}`;

const read = (key) => {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
};
const write = (key, value, empty) => {
  try {
    if (empty) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify(value));
  } catch { /* private mode / quota */ }
};

/** The clip's scene overrides, with the selected one. */
export function useSceneOverrides(targetKey, maxFrames) {
  const [overrides, setOverrides] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  // Master switch. Off, the overrides stay on the timeline but nothing reads them: the preview,
  // the cache strip and the render all see the clip's own settings.
  const [enabled, setEnabledState] = useState(true);
  const listRef = useRef(overrides);
  listRef.current = overrides;

  useEffect(() => {
    // maxFrames is deliberately not a dependency: it arrives a tick after the target changes, and
    // re-reading on it would clamp the stored overrides against the PREVIOUS clip's length and
    // write the damage back (same trap useSegments documents).
    setOverrides(normalizeOverrides(read(overridesKey(targetKey)) || [], 1e9));
    setEnabledState(read(enabledKey(targetKey)) !== false);
    setSelectedId(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetKey]);

  const setEnabled = useCallback((on) => {
    setEnabledState(!!on);
    write(enabledKey(targetKey), false, !!on);       // only the non-default (off) is stored
  }, [targetKey]);

  const activeOverrides = useMemo(() => (enabled ? overrides : NONE), [enabled, overrides]);

  const commit = useCallback((next) => {
    setOverrides(next);
    write(overridesKey(targetKey), next, next.length === 0);
  }, [targetKey]);

  const add = useCallback((range, label) => {
    const { list, created } = addOverride(listRef.current, { ...range, label }, maxFrames);
    commit(list);
    setSelectedId(created.id);
    return created;
  }, [commit, maxFrames]);

  const update = useCallback((id, patch) => commit(updateOverride(listRef.current, id, patch, maxFrames)), [commit, maxFrames]);
  const remove = useCallback((id) => {
    commit(removeOverride(listRef.current, id));
    setSelectedId((cur) => (cur === id ? null : cur));
  }, [commit]);
  const clear = useCallback(() => { commit([]); setSelectedId(null); }, [commit]);

  return { overrides, activeOverrides, enabled, setEnabled, selectedId, setSelectedId, add, update, remove, clear };
}

/** Finished renders of this clip: [{start, end, fp}], fp = fingerprint of the settings used. */
export function useRenderLog(targetKey) {
  const [log, setLog] = useState([]);
  const logRef = useRef(log);
  logRef.current = log;
  useEffect(() => { setLog(read(renderLogKey(targetKey)) || []); }, [targetKey]);
  const record = useCallback((entry) => {
    if (!entry || !Number.isFinite(entry.start) || !Number.isFinite(entry.end) || !entry.fp) return;
    const next = recordRender(logRef.current, entry);
    setLog(next);
    write(renderLogKey(targetKey), next, next.length === 0);
  }, [targetKey]);
  const clear = useCallback(() => { setLog([]); write(renderLogKey(targetKey), [], true); }, [targetKey]);
  return { log, record, clear };
}
