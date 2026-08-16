import { useEffect, useState } from 'react';
import { postJSON } from '../../api';
import { num } from './utils';

// ── Pre-run estimate (idle only) ──────────────────────────────────────────
// A heuristic baseline, refined by the ms/frame the backend has actually
// MEASURED for the current settings across past completed runs
// (roop.runtime_calib). See [[learned-runtime-estimate]] for why the estimate
// is keyed on a settings signature rather than a single global average.

// The fields sent to /api/runtime_estimate. This list is not cosmetic: it has
// to match runtime_calib._SIG_FIELDS, because the backend looks up past runs by
// hashing exactly these. A field omitted here produces a signature that can
// never match the one a completed run records, so the estimate silently falls
// back to the global average forever.
const sigPayload = (p) => ({
  swap_model: p.swap_model,
  selected_enhancer: p.selected_enhancer,
  face_detection_mode: p.face_detection_mode,
  face_detector_size: p.face_detector_size,
  detector_engine: p.detector_engine,
  num_swap_steps: num(p.num_swap_steps, 1),
  subsample_upscale: p.subsample_upscale,
  track_identities: p.track_identities,
  temporal_detection: p.temporal_detection,
  mask_engine: p.mask_engine,
  stabilize_face: p.stabilize_face,
  stabilize_enhancer: p.stabilize_enhancer,
  stabilize_mask: p.stabilize_mask,
  // Both are whole GPU stages and part of the calibration signature.
  expression_restore_strength: num(p.expression_restore_strength, 0),
  upscale_after_swap: p.upscale_after_swap,
  // The merger post-ops cost up to 15 ms/face, so runtime_calib folds a COUNT
  // of the active ones into the signature. They have to be sent here as well:
  // the RECORD side builds its signature from the full swap payload, so if the
  // estimate request omitted them the two would key different buckets and the
  // estimate would never find the run it just measured. output_face_scale is
  // deliberately absent — it costs nothing and does not move the signature.
  merger_hist_match: num(p.merger_hist_match, 0),
  merger_sharpen: num(p.merger_sharpen, 0),
  merger_motion_blur: num(p.merger_motion_blur, 0),
  merger_grain_match: num(p.merger_grain_match, 0),
  merger_degrade: num(p.merger_degrade, 0),
});

// Rough wall-clock cost of a frame from the settings alone, used before there
// is anything measured to go on (and blended in when the measurement is thin).
const heuristicMsPerFrame = (p, threads) => {
  let ms = 45;
  if (p.selected_enhancer && p.selected_enhancer !== 'None') ms += 70;
  const det = parseInt(p.face_detector_size || '640', 10) || 640;
  ms += (det / 640) * 15;
  ms += (num(p.num_swap_steps, 1) - 1) * 25;
  if (p.track_identities) ms += 8;
  const parallel = Math.max(1.5, Math.min(4, (threads || 3) * 0.6));
  return ms / parallel;   // comparable to the measured wall-clock value
};

/**
 * @param settings   current Face Swap params
 * @param estFrames  frames the run would cover
 * @param faceCount  faces in the current preview frame — a density hint
 * @param processing whether a run is in flight (the estimate is idle-only)
 * @param hasTargets whether anything is loaded to estimate
 * @param threads    telemetry thread count, for the heuristic's parallelism
 */
export default function useRuntimeEstimate({
  settings: p, estFrames, faceCount, processing, hasTargets, threads,
}) {
  const [calibEst, setCalibEst] = useState(null);

  /* eslint-disable react-hooks/exhaustive-deps -- intentional: the settings
     object is a new reference on every keystroke anywhere in the panel, and
     this effect fires a network request. Depending on `p` would re-request the
     estimate when an unrelated setting changed; the fields that actually affect
     it are listed individually below. */
  useEffect(() => {
    if (processing || estFrames <= 1 || !hasTargets) { setCalibEst(null); return undefined; }
    let cancelled = false;
    const t = setTimeout(async () => {
      try {
        const res = await postJSON('/api/runtime_estimate', {
          frames: estFrames,
          face_count: faceCount,
          ...sigPayload(p),
        });
        if (!cancelled) setCalibEst(res || null);
      } catch { if (!cancelled) setCalibEst(null); }
    }, 500);
    return () => { cancelled = true; clearTimeout(t); };
  }, [processing, estFrames, faceCount, hasTargets,
      p.swap_model, p.selected_enhancer, p.face_detection_mode, p.face_detector_size,
      p.detector_engine, p.num_swap_steps, p.subsample_upscale, p.track_identities,
      p.temporal_detection, p.mask_engine, p.stabilize_face, p.stabilize_enhancer,
      p.stabilize_mask, p.expression_restore_strength, p.upscale_after_swap,
      p.merger_hist_match, p.merger_sharpen, p.merger_motion_blur,
      p.merger_grain_match, p.merger_degrade]);
  /* eslint-enable react-hooks/exhaustive-deps */

  const heuristicPerFrame = heuristicMsPerFrame(p, threads);

  // Prefer the measured ms/frame. Blend 50/50 with the heuristic when the data
  // is thin (a single sample, or the cross-settings global fallback).
  const estPerFrame = (() => {
    if (calibEst && calibEst.ms_per_frame) {
      const thin = calibEst.source !== 'measured' || (calibEst.samples || 0) < 2;
      return thin ? (calibEst.ms_per_frame + heuristicPerFrame) / 2 : calibEst.ms_per_frame;
    }
    return heuristicPerFrame;
  })();

  return {
    calibEst,
    estPerFrame,
    estTotalMs: estFrames * estPerFrame,
    estLearned: !!(calibEst && calibEst.source === 'measured' && (calibEst.samples || 0) >= 1),
  };
}
