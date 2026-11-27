// The four temporal stability modes as DATA: which real settings each one sets,
// and how the current settings are read back into a mode.
//
// This is a summary over settings that already exist and are already read by
// the render (roop/ProcessOptions, ProcessMgr, procmgr_tracking):
//
//   temporal_detection     one tracked detection pre-pass; short misses gap-filled
//   temporal_smart_skip    reuse the previous frame's face ROI when little moved
//   stabilize_face         keypoint smoothing, with stabilize_method one_euro|ema|kalman
//   stabilize_landmarks    smooth the 106-pt landmarks the paste outline is drawn from
//   stabilize_enhancer     blend the restorer's output across frames
//   stabilize_mask         blend the mask edge across frames (+ mask_flow_warp)
//   stabilize_hf_texture   carry the previous frame's pore layer along optical flow
//
// "Cut-aware" is a property of the ENGINE, not a switch: every stabiliser resets
// at a detected scene cut whichever mode is chosen (the pre-pass records the cuts
// in the face bank). So "Cinema Cut-Aware" names the full stack that benefits
// most from it; there is no `scene_cut` setting to flip and none is invented here.
//
// A mode sets toggles and the smoothing method. It deliberately leaves the
// strength sliders alone: those are tuned by eye per machine, and a mode switch
// that re-tuned them would undo that work.
//
// The reading is strict. Anything that is not exactly one of the four is
// `custom`, shown as such, so the control never claims a mode the settings do
// not hold.

export const TEMPORAL_MODES = [
  {
    id: 'off',
    label: 'Off',
    hint: 'No temporal processing: every frame is detected and swapped on its own. Fastest; can flicker.',
    patch: {
      temporal_detection: false,
      temporal_smart_skip: false,
      stabilize_face: false,
      stabilize_landmarks: false,
      stabilize_enhancer: false,
      stabilize_mask: false,
      stabilize_hf_texture: false,
    },
  },
  {
    id: 'smart_skip',
    label: 'Smart-Skip ROI',
    hint: 'Tracked detection, and the previous frame\'s face region is reused when little moved. Saves compute on locked-off shots; no smoothing.',
    patch: {
      temporal_detection: true,
      temporal_smart_skip: true,
      stabilize_face: false,
      stabilize_landmarks: false,
      stabilize_enhancer: false,
      stabilize_mask: false,
      stabilize_hf_texture: false,
    },
  },
  {
    id: 'balanced',
    label: 'Balanced EMA',
    hint: 'Tracked detection with fixed-rate (EMA) keypoint smoothing, plus restorer and mask edge blending.',
    patch: {
      temporal_detection: true,
      stabilize_face: true,
      stabilize_method: 'ema',
      stabilize_landmarks: true,
      stabilize_enhancer: true,
      stabilize_mask: true,
      stabilize_hf_texture: false,
    },
  },
  {
    id: 'cinema',
    label: 'Cinema Cut-Aware',
    hint: 'The full stack (the shipped default): adaptive One Euro smoothing, restorer and mask blending, pore carry. Resets at every scene cut. Costs the most.',
    patch: {
      temporal_detection: true,
      track_identities: true,
      stabilize_face: true,
      stabilize_method: 'one_euro',
      stabilize_landmarks: true,
      stabilize_enhancer: true,
      stabilize_mask: true,
      stabilize_hf_texture: true,
    },
  },
];

const on = (v) => v === true;
// Backend defaults for the toggles, for a settings object that has not got the key yet.
const get = (s, k, d) => (s[k] === undefined || s[k] === null ? d : s[k]);

/**
 * The mode the settings are in: 'off' | 'smart_skip' | 'balanced' | 'cinema', or 'custom'.
 *
 * Smart-Skip is not part of balanced/cinema (a mode that leaves it alone), so it does not
 * disqualify them; it IS the whole point of 'smart_skip' and 'off' pins it.
 */
export function temporalModeOf(settings) {
  const s = settings || {};
  const td = on(get(s, 'temporal_detection', true));
  const skip = on(get(s, 'temporal_smart_skip', false));
  const sf = on(get(s, 'stabilize_face', true));
  const sl = on(get(s, 'stabilize_landmarks', true));
  const se = on(get(s, 'stabilize_enhancer', true));
  const sm = on(get(s, 'stabilize_mask', true));
  const sh = on(get(s, 'stabilize_hf_texture', true));
  const method = get(s, 'stabilize_method', 'one_euro');

  if (!td && !skip && !sf && !sl && !se && !sm && !sh) return 'off';
  if (td && skip && !sf && !sl && !se && !sm && !sh) return 'smart_skip';
  if (td && sf && sl && se && sm && !sh && method === 'ema') return 'balanced';
  if (td && sf && sl && se && sm && sh && method === 'one_euro') return 'cinema';
  return 'custom';
}

/** The settings patch for `id`, or {} for 'custom' (a description of the current state, never a target). */
export function temporalPatch(id) {
  const m = TEMPORAL_MODES.find((x) => x.id === id);
  return m ? { ...m.patch } : {};
}

/** One line under the control: what the current mode means, or what 'custom' is made of. */
export function temporalNote(settings) {
  const id = temporalModeOf(settings);
  const m = TEMPORAL_MODES.find((x) => x.id === id);
  if (m) return m.hint;
  const s = settings || {};
  const parts = [];
  if (on(get(s, 'temporal_detection', true))) parts.push('tracked detection');
  if (on(get(s, 'temporal_smart_skip', false))) parts.push('Smart-Skip');
  if (on(get(s, 'stabilize_face', true))) parts.push(`${get(s, 'stabilize_method', 'one_euro').replace('_', ' ')} keypoints`);
  if (on(get(s, 'stabilize_enhancer', true))) parts.push('restorer blend');
  if (on(get(s, 'stabilize_mask', true))) parts.push('mask blend');
  if (on(get(s, 'stabilize_hf_texture', true))) parts.push('pore carry');
  return parts.length ? `Custom: ${parts.join(', ')}.` : 'Custom: nothing temporal is on.';
}
