import { useCallback, useMemo, useRef, useState } from 'react';
import { useHistoryChannel, CURVES_CHANNEL } from '../../../history/historyChannels';
import {
  addKey, hasCurves, moveKey, normalizeTracks, paramById, removeKey, setInterp, setValue,
} from './curveModel';

// Parameter curves, per clip.
//
// Stored in localStorage like the markers, segments, scene overrides and identity keyframes beside them (a convenience
// layered on the clip, still there tomorrow, safe to lose). The undo history tracks `edited` - the clips touched this
// session - and a clip that was only LOADED is read straight from storage without entering that map, so opening a clip
// never records itself as an edit.
//
// Writes happen in `commit` and `restore` and nowhere else; an effect that persisted "whatever the state holds" would
// run on mount before the stored list had been read, and erase it.

const tracksKey = (k) => `roop_curves_${k || 'default'}`;
const enabledKey = (k) => `roop_curves_enabled_${k || 'default'}`;
const NONE = Object.freeze({});

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

/**
 * @param targetKey  stable id of the clip on the timeline
 * @param maxFrames  clip length in frames (where a keyframe may go)
 */
export default function useParameterCurves(targetKey, maxFrames = 1) {
  const key = targetKey || 'default';
  const [edited, setEdited] = useState(NONE);
  // The stored tracks are not clamped to maxFrames on load: it arrives a tick after the clip changes, and clamping
  // against the previous clip's length would write the damage back (useSceneOverrides documents the same trap).
  const stored = useMemo(() => normalizeTracks(read(tracksKey(key)), 1e9), [key]);
  const tracks = edited[key] ?? stored;
  const tracksRef = useRef(tracks);
  tracksRef.current = tracks;
  const editedRef = useRef(edited);
  editedRef.current = edited;

  // The master switch is per clip and only the non-default (off) is stored. Not part of the undo history.
  const [enabledByClip, setEnabledByClip] = useState({});
  const storedEnabled = useMemo(() => read(enabledKey(key)) !== false, [key]);
  const enabled = enabledByClip[key] ?? storedEnabled;
  const setEnabled = useCallback((on) => {
    setEnabledByClip((m) => ({ ...m, [key]: !!on }));
    write(enabledKey(key), false, !!on);
  }, [key]);

  const commit = useCallback((paramId, next) => {
    const t = { ...tracksRef.current };
    if (next.length) t[paramId] = next; else delete t[paramId];
    write(tracksKey(key), t, Object.keys(t).length === 0);
    setEdited((all) => ({ ...all, [key]: t }));
    return t;
  }, [key]);

  // Undo / redo / slot recall hand back a whole map. Every clip that is in either map is re-persisted, or a clip undone
  // to "no curves" would still read its old tracks from storage.
  const restore = useCallback((map) => {
    const next = map && typeof map === 'object' ? map : NONE;
    for (const clip of new Set([...Object.keys(editedRef.current), ...Object.keys(next)])) {
      const t = next[clip];
      write(tracksKey(clip), t, !t || Object.keys(t).length === 0);
    }
    setEdited(next);
  }, []);
  useHistoryChannel('curves', edited, restore, CURVES_CHANNEL);

  const trackOf = (paramId) => tracksRef.current[paramId] || [];

  const add = useCallback((paramId, frame, value, interp) => {
    const p = paramById(paramId);
    if (!p) return null;
    const out = addKey(trackOf(paramId), p, frame, value, maxFrames, interp);
    if (out.created) commit(paramId, out.list);
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [commit, maxFrames]);

  const move = useCallback((paramId, frame, toFrame, toValue) => {
    const p = paramById(paramId);
    if (!p) return false;
    const cur = trackOf(paramId);
    const next = moveKey(cur, p, frame, toFrame, toValue, maxFrames);
    if (next === cur) return false;
    commit(paramId, next);
    return true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [commit, maxFrames]);

  const remove = useCallback((paramId, frame) => commit(paramId, removeKey(trackOf(paramId), frame)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [commit]);
  const interp = useCallback((paramId, frame, kind) => commit(paramId, setInterp(trackOf(paramId), frame, kind)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [commit]);
  const setKeyValue = useCallback((paramId, frame, value) => {
    const p = paramById(paramId);
    if (p) commit(paramId, setValue(trackOf(paramId), p, frame, value));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [commit]);
  const clearParam = useCallback((paramId) => commit(paramId, []), [commit]);

  return useMemo(() => ({
    tracks,
    /** What the preview and the render read: the tracks, or nothing while the master switch is off. */
    activeTracks: enabled ? tracks : NONE,
    enabled,
    setEnabled,
    any: hasCurves(tracks),
    add,
    move,
    remove,
    interp,
    setValue: setKeyValue,
    clearParam,
  }), [tracks, enabled, setEnabled, add, move, remove, interp, setKeyValue, clearParam]);
}
