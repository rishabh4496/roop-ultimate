import { useState } from 'react';
import { postJSON } from '../../api';

// ── Clip advisor ──────────────────────────────────────────────────────────
// Samples the selected target (face sizes, count, detection coverage, motion,
// lighting) and asks the backend for settings tuned to it. Nothing is applied
// until the user says so, which is the whole point — the recommendations are
// shown as a before → after diff first.

// Module scope, not rebuilt per render: these are constants that happened to be
// declared inside the component body.
export const ADVISOR_LABELS = {
  temporal_detection: 'Temporal detection',
  detector_engine: 'Detector engine',
  rescue_small_faces: 'Rescue small faces',
  face_detector_size: 'Detection resolution',
  subsample_upscale: 'Subsample upscale',
  face_detection_mode: 'Face selection',
  track_identities: 'Track identities',
  face_detector_threshold: 'Detection threshold',
  stabilize_face: 'Stabilize face',
};

export const fmtAdviceVal = (v) => (v === true ? 'On' : v === false ? 'Off' : String(v));

/**
 * @param targets     the loaded target list (only its length is read here)
 * @param selTarget   index of the target to analyse
 * @param settings    the current Face Swap params, sent so the backend can
 *                    return only actual CHANGES rather than a full config
 * @param set         (key, value) writer used when applying a recommendation
 */
export default function useClipAdvisor({ targets, selTarget, settings, set, notify }) {
  const [advice, setAdvice] = useState(null);
  const [advisorBusy, setAdvisorBusy] = useState(false);

  const runAdvisor = async () => {
    if (!targets.length) { notify('Load a target first', 'error'); return; }
    setAdvisorBusy(true);
    setAdvice(null);
    try {
      const res = await postJSON('/api/advisor', { index: selTarget, settings });
      setAdvice(res);
      if (res.recommendations?.length === 0 && !res.message) notify('Settings already fit this clip ✓');
    } catch (e) { notify(e.message, 'error'); } finally { setAdvisorBusy(false); }
  };

  const applyAdvice = () => {
    if (!advice?.recommendations?.length) return;
    advice.recommendations.forEach((r) => set(r.key, r.value));
    notify(`Applied ${advice.recommendations.length} recommended setting${advice.recommendations.length === 1 ? '' : 's'}`);
    setAdvice(null);
  };

  return { advice, setAdvice, advisorBusy, runAdvisor, applyAdvice };
}
