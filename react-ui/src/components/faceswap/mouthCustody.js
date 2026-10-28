// Viseme Mouth Custody: the pure model behind the "Audio & Mouth Custody" panel (app/roop/mouth_custody.py is the
// backend half; `.render-check/render-mouth-custody.mjs` pins the two to the same cases).
//
//   off                    the swap's own mouth with the standard expression transfer (the legacy default)
//   source_mouth_restore   the target's articulation kept (the old "Restore original mouth area")
//   audio_driven_sync      lip-sync from the target's audio, or from a bound external track (the old lip-sync toggle)
//
// An old saved recipe has no `mouth_custody_mode`, only `lipsync_enabled` / `restore_original_mouth`. `modeOf` reads
// those exactly the way the backend's `derive_mode` does, so opening a recipe never silently turns lip-sync off, and
// `withMode` writes the mode AND the legacy keys it means so the two can never disagree in a stored recipe.

export const MODES = Object.freeze(['off', 'source_mouth_restore', 'audio_driven_sync']);
export const MODE_LABELS = Object.freeze({
  off: 'Off',
  source_mouth_restore: 'Source mouth restore',
  audio_driven_sync: 'Audio-driven sync',
});
export const MODE_HINTS = Object.freeze({
  off: 'The swap keeps its own mouth, shaped by the standard expression transfer from the target video.',
  source_mouth_restore: "The target's own mouth is kept (open-mouth gated, with occlusion and landmarks), so articulation stays as clear as the original actor's.",
  audio_driven_sync: "The lower face is driven by audio: the target video's own track, or an external .wav / .mp3 / .flac that also becomes the delivered audio.",
});
export const OFFSET_LIMIT_MS = 500;
/** The slip nudge buttons, in ms. */
export const SLIP_NUDGES = Object.freeze([-10, -1, 1, 10]);
export const AUDIO_ACCEPT = '.wav,.mp3,.flac,.m4a,.aac,.ogg,.opus,audio/*';

const finite = (v) => v !== null && v !== undefined && v !== '' && typeof v !== 'boolean' && Number.isFinite(Number(v));

/** The recipe's mode: the explicit one, else what the legacy switches said (lip-sync outranks restore). */
export function modeOf(params) {
  const p = params || {};
  if (MODES.includes(p.mouth_custody_mode)) return p.mouth_custody_mode;
  if (p.lipsync_enabled) return 'audio_driven_sync';
  if (p.restore_original_mouth) return 'source_mouth_restore';
  return 'off';
}

/** The audio slip in ms: finite, clamped to +-500, rounded to a whole ms. Garbage / null is 0. */
export function clampOffset(v) {
  if (!finite(v)) return 0;
  return Math.max(-OFFSET_LIMIT_MS, Math.min(OFFSET_LIMIT_MS, Math.round(Number(v))));
}

/** `+120 ms` / `-40 ms` / `0 ms`, with a true minus sign for a screen reader. */
export function offsetText(v) {
  const n = clampOffset(v);
  return `${n > 0 ? '+' : n < 0 ? '−' : ''}${Math.abs(n)} ms`;
}

/** What a slip means in plain words (the sign is the one thing people get backwards). */
export function offsetMeaning(v) {
  const n = clampOffset(v);
  if (n === 0) return 'Audio and video are in step.';
  return n > 0 ? `Audio plays ${n} ms later than the video.` : `Audio plays ${-n} ms earlier than the video.`;
}

/**
 * The recipe PATCH for choosing a mode: the mode plus the legacy keys it stands for, so a stored recipe can never hold a
 * mode and a legacy switch that disagree. Switching AWAY from audio keeps the bound track and the slip in the recipe
 * (switching back restores them); the backend ignores them outside audio mode except that a bound track still replaces
 * the delivered audio.
 */
export function modePatch(mode) {
  const m = MODES.includes(mode) ? mode : 'off';
  return { mouth_custody_mode: m, lipsync_enabled: m === 'audio_driven_sync', restore_original_mouth: m === 'source_mouth_restore' };
}
export const withMode = (params, mode) => ({ ...(params || {}), ...modePatch(mode) });

/**
 * The PATCH for binding (or clearing, with null) an external audio track: the one remux source plus, unless `legacy` is
 * false, the legacy pair it folds into. `empty` is what "no track" is written as ('' for a multipart form, where
 * FormData.append(k, null) would send the text "null").
 */
export function overridePatch(path, { legacy = true, empty = null } = {}) {
  const p = path || empty;
  return legacy
    ? { override_audio_path: p, lipsync_audio_path: p, lipsync_audio_source: path ? 'upload' : 'original' }
    : { override_audio_path: p };
}
export const withOverride = (params, path) => ({ ...(params || {}), ...overridePatch(path) });

/** The keys a request carries. Always includes the mode: omitting it would hand the decision back to legacy keys. */
export function requestFields(params) {
  const p = params || {};
  return {
    mouth_custody_mode: modeOf(p),
    override_audio_path: p.override_audio_path || (p.lipsync_audio_source === 'upload' ? p.lipsync_audio_path : null) || null,
    audio_offset_ms: clampOffset(p.audio_offset_ms),
  };
}

/**
 * What the panel should tell the user about the clip and the mode, or null when there is nothing to say.
 * `hasTargetAudio` is true / false / null (unknown yet, e.g. the envelope is still building).
 */
export function advisory(params, { hasTargetAudio = null, isImage = false } = {}) {
  const mode = modeOf(params);
  const { override_audio_path: track } = requestFields(params);
  if (mode !== 'audio_driven_sync') return null;
  if (isImage) return { tone: 'warn', text: 'A still image has no audio to drive a mouth: it will use Source mouth restore.' };
  if (!track && hasTargetAudio === false) {
    return { tone: 'warn', text: 'This video is silent, so there is nothing to drive the mouth: it will fall back to Source mouth restore. Bind an external audio track to dub it.' };
  }
  return null;
}

/** How far the audio track is drawn from the video (seconds, + = right/later): what the render will do. */
export function trackShiftSeconds(params) {
  return clampOffset((params || {}).audio_offset_ms) / 1000;
}
