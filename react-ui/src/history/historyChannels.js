import { useEffect, useRef } from 'react';
import { useHistoryStore } from '../store/historyStore';
import {
  describeSettingsChange, describeRotoChange, describePinsChange, describeStrokesChange,
  describeAppliedMaskChange, settingsEqual,
} from './historyEngine';
import { LABELS } from '../components/settingsDiff';
import { SETTINGS_CATALOG } from '../components/settingsCatalog';

// The channels the app registers, and the hook an owner uses to report one.
//
// A channel is { apply, equals, describe, coalesce, fromSlot, scope }:
//   apply      put a value back into the owner's state (undo, redo, time-travel, slot recall)
//   equals     when the owner rebuilds an equal value with a new identity (a memo over three
//              useStates), without this an undo would immediately record itself as a new edit
//   describe   (before, after) -> the human label for the entry
//   fromSlot   (live, slotValue) -> what to apply when a SNAPSHOT SLOT is recalled
//   scope      'global' travels with every slot; 'faceswap' / 'extras' only with that route's slots

const CATALOG_LABEL = Object.fromEntries(SETTINGS_CATALOG.map((s) => [s.key, s.label]));

const humanize = (k) => String(k).replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());

export const settingLabel = (k) => LABELS[k] || CATALOG_LABEL[k] || humanize(k);

// A slot is a LOOK, not a machine. Recalling Slot B must not flip the theme, the execution provider,
// the server port or the thread count underneath someone comparing two lighting setups. These stay
// at their live value when a slot is applied (undo and redo still restore everything, since a
// history entry is exactly what the user did).
const MACHINE_KEY = /^(theme_|selected_theme$|custom_themes$|provider$|server_|max_threads$|trt_|faceset_library_path$|output_template$|clear_output$)/;

export const isRecipeKey = (k) => !MACHINE_KEY.test(k);

export function mergeRecipeSettings(liveSettings, slotSettings) {
  const out = { ...(liveSettings || {}) };
  for (const [k, v] of Object.entries(slotSettings || {})) if (isRecipeKey(k)) out[k] = v;
  return out;
}

export const shallowEqual = (a, b) => {
  if (a === b) return true;
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object') return false;
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  return ka.every((k) => Object.is(a[k], b[k]));
};

export const SETTINGS_CHANNEL = {
  scope: 'global',
  equals: settingsEqual,
  describe: (b, a) => describeSettingsChange(b, a, settingLabel),
  fromSlot: mergeRecipeSettings,
};

export const ROTO_CHANNEL = { scope: 'faceswap', describe: describeRotoChange, coalesce: false };
export const PINS_CHANNEL = { scope: 'faceswap', describe: describePinsChange, coalesce: false };
export const STROKES_CHANNEL = { scope: 'faceswap', describe: describeStrokesChange, coalesce: false };
export const MASK_CHANNEL = { scope: 'faceswap', equals: shallowEqual, describe: describeAppliedMaskChange, coalesce: false };

export const EXTRAS_OPTS_CHANNEL = {
  scope: 'extras',
  equals: shallowEqual,
  describe: (b, a) => describeSettingsChange(b, a, (k) => ({
    resolution: 'Resize width', rotation: 'Rotation', fps: 'Output FPS',
    crop_left: 'Crop left', crop_right: 'Crop right', crop_top: 'Crop top', crop_bottom: 'Crop bottom',
    audio_offset_ms: 'Audio offset',
  }[k] || humanize(k))),
};

/**
 * Report `value` as the current state of channel `id`. `apply` is read through a ref so a fresh
 * closure each render never re-registers the channel. `enabled` false keeps the channel out of the
 * history until the owner has real data (App's settings are null until the backend answers).
 */
export function useHistoryChannel(id, value, apply, def, enabled = true) {
  const applyRef = useRef(apply);
  applyRef.current = apply;
  const observe = useHistoryStore.getState().observe;

  useEffect(() => {
    if (!enabled) return undefined;
    return useHistoryStore.getState().register(id, { ...def, apply: (v) => applyRef.current(v) });
  }, [id, def, enabled]);

  // After the registering effect (declaration order), so the first observation finds the channel.
  useEffect(() => {
    if (enabled) observe(id, value);
  }, [id, value, enabled, observe]);
}
