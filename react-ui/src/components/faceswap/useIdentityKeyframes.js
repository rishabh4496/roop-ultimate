import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  addKeyframe, frameIndexOf, moveKeyframe, normalizeKeyframes, removeKeyframe, seedWeightsAt, updateKeyframe,
} from './identityKeyframes';

// Identity keyframes, per clip.
//
// The keyframes live INSIDE the Identity Blender recipe (`recipe.keyframes`), so they travel in every preview / swap
// payload like the rest of it: the preview re-renders when one moves, the cache-strip fingerprint changes, and a
// queued job freezes them. But a keyframe is a frame number on ONE clip, so they are stored per clip, in
// localStorage, exactly like the chapter markers, segments and scene overrides beside them (a convenience layered on
// the clip, still there tomorrow), and swapped into the recipe when the clip changes.
//
// Writes happen in `commit` and nowhere else. An effect that persisted "whatever the recipe holds" would run on mount
// with the empty initial recipe, before the stored list had been applied, and erase it.

const storageKey = (k) => `roop_idkeys_${k || 'default'}`;

const read = (key) => {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
};
const write = (key, list) => {
  try {
    if (list.length) localStorage.setItem(key, JSON.stringify(list));
    else localStorage.removeItem(key);
  } catch { /* private mode / quota: keyframes are a convenience, not state we own */ }
};

/**
 * @param targetKey  stable id of the clip on the timeline
 * @param recipe     the Identity Blender recipe state (normalizeBlend output)
 * @param setRecipe  its setter (functional form)
 * @param maxFrames  clip length in frames (clamps where a keyframe may go)
 */
export default function useIdentityKeyframes(targetKey, recipe, setRecipe, maxFrames = 1) {
  const [selected, setSelected] = useState(null);            // the selected keyframe's frame_index
  const keyRef = useRef(targetKey);                          // the clip whose keyframes the recipe holds
  const listRef = useRef(recipe.keyframes);
  listRef.current = recipe.keyframes;
  const recipeRef = useRef(recipe);
  recipeRef.current = recipe;

  useEffect(() => {
    // maxFrames is deliberately not a dependency: it arrives a tick after the clip changes, and a re-read on it
    // would clamp this clip's keyframes against the PREVIOUS clip's length (useSegments documents the same trap).
    keyRef.current = targetKey;
    const stored = normalizeKeyframes(read(storageKey(targetKey)) || []);
    setRecipe((r) => (JSON.stringify(r.keyframes) === JSON.stringify(stored) ? r : { ...r, keyframes: stored }));
    setSelected(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetKey]);

  const commit = useCallback((next) => {
    write(storageKey(keyRef.current), next);
    setRecipe((r) => ({ ...r, keyframes: next }));
    return next;
  }, [setRecipe]);

  const components = recipe.components || [];
  // Keying needs two sources to move between; with fewer there is nothing for a keyframe to say.
  const canKey = components.length >= 2;

  const add = useCallback((uiFrame, curve = 'linear') => {
    const r = recipeRef.current;
    if ((r.components || []).length < 2) return null;
    const weights = seedWeightsAt(listRef.current, r.components, uiFrame);
    const out = addKeyframe(listRef.current, uiFrame, weights, maxFrames, curve);
    if (out.created) commit(out.list);
    setSelected(out.frame_index);
    return out;
  }, [commit, maxFrames]);

  const move = useCallback((frameIndex, toUiFrame) => {
    const next = moveKeyframe(listRef.current, frameIndex, toUiFrame, maxFrames);
    if (next === listRef.current) return false;
    commit(next);
    setSelected(frameIndexOf(toUiFrame));
    return true;
  }, [commit, maxFrames]);

  const remove = useCallback((frameIndex) => {
    commit(removeKeyframe(listRef.current, frameIndex));
    setSelected((s) => (s === frameIndex ? null : s));
  }, [commit]);

  const update = useCallback((frameIndex, patch) => commit(updateKeyframe(listRef.current, frameIndex, patch)), [commit]);
  const clear = useCallback(() => { commit([]); setSelected(null); }, [commit]);
  // Replace the whole track at once (the Curve Editor's Identity Morph Weight compiles into it). One commit, so the
  // preview re-renders once and a queued job sees the finished track, never a half-written one.
  const replace = useCallback((list) => { commit(normalizeKeyframes(list)); setSelected(null); }, [commit]);

  return useMemo(() => ({
    keyframes: recipe.keyframes || [],
    selected,
    setSelected,
    canKey,
    add,
    move,
    remove,
    update,
    clear,
    replace,
  }), [recipe.keyframes, selected, canKey, add, move, remove, update, clear, replace]);
}
