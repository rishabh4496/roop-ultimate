import { getJSON, postJSON } from './apiClient';

export const swapAdapter = {
  buildSwapPayload: (settings = {}, extra = {}) => {
    const p = settings;
    return {
      ...p,
      index: extra.index ?? 0,
      frame: extra.frame ?? 1,
      fake_preview: extra.fake ?? false,
      mask_data_url: extra.manualMask || null,
      mask_ref_kps: extra.maskRefKps || null,
      face_mapping: extra.faceMapping || [],
      enhancer: p.selected_enhancer,
      detection: p.face_detection_mode,
      output_method: p.output_method,
      video_method: p.video_swapping_method,
      upscale: p.subsample_upscale,
      mask_engine: p.mask_engine,
      mask_engine_2: p.mask_engine_2,
      clip_text: p.mask_clip_text,
      sam2_model_size: p.sam2_model_size,
      track_identities: p.track_identities,
      autorotate: p.autorotate_faces,
      face_distance: p.max_face_distance,
      blend_ratio: p.blend_ratio,
      num_swap_steps: p.num_swap_steps,
      color_transfer_mode: p.color_transfer_mode,
      face_detector_threshold: p.face_detector_threshold,
      face_detector_nms: p.face_detector_nms,
    };
  },

  startSwap: (payload) => postJSON('/api/swap', payload),
  stopSwap: () => postJSON('/api/stop', {}),
  pauseSwap: () => postJSON('/api/pause', {}),
  resumeSwap: () => postJSON('/api/resume', {}),
  getProgress: (opts) => getJSON('/api/progress', opts),
  getState: () => getJSON('/api/state'),
  getMeta: () => getJSON('/api/meta'),
  getQualityAnalysis: (outputPath) => postJSON('/api/quality/analyze', { path: outputPath }),
};
