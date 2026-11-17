import schema from '../../../../app/settings.schema.json';

// Names/defaults come from Settings. Wire aliases share the existing saved key.
export const STUDIO_CONTROLS = [
  { wire: 'detector_model', key: 'detector_engine', label: 'Detector model', choices: ['gnkps', 'scrfd', 'retinaface', 'retinaface_r50', 'retinaface_r50_gpu', 'yoloface', 'yunet'] },
  { wire: 'multi_angle_detect', key: 'autorotate_faces', label: 'Multi-angle detection', toggle: true },
  { wire: 'stabilization_strength', key: 'stabilization_strength', label: 'Stabilization strength', min: 0, max: 1 },
  { wire: 'pose_adaptive_faceset', key: 'pose_adaptive_faceset', label: 'Pose-adaptive faceset', toggle: true },
  { wire: 'mask_engine', key: 'mask_engine', label: 'Mask engine', choices: ['bisenet+xseg', 'RealityUX', 'Face Parser', 'DFL XSeg', 'None'] },
  { wire: 'yaw_erosion_bias', key: 'yaw_erosion_bias', label: 'Yaw erosion bias', min: -1, max: 1 },
  { wire: 'expression_restorer_weight', key: 'expression_restore_strength', label: 'Expression restorer weight', min: 0, max: 1 },
  { wire: 'preserve_inner_mouth', key: 'oral_cavity_restore', label: 'Preserve inner mouth', toggle: true },
  { wire: 'pixel_boost', key: 'subsample_upscale', label: 'Pixel Boost', choices: [256, 512, 768] },
  { wire: 'ultramax_ramp', key: 'ultramax_ramp', label: 'Restorer bypass ramp end', min: 256, max: 384, step: 1, unit: 'px' },
  { wire: 'temporal_restore_alpha', key: 'temporal_enhancer_blend', label: 'Temporal restore alpha', min: 0, max: 0.9 },
  { wire: 'restore_identity_weight', key: 'codeformer_fidelity', label: 'Restore identity weight', min: 0, max: 1 },
  { wire: 'color_transfer_mode', key: 'color_transfer_mode', label: 'Color transfer mode', choices: ['lab', 'cdf', 'adain', 'none', 'rct', 'lct', 'mkl', 'idt'] },
  { wire: 'grain_match_strength', key: 'grain_match_strength', label: 'Grain match strength', min: 0, max: 1 },
  { wire: 'motion_blur_match', key: 'merger_motion_blur', label: 'Motion blur match', min: 0, max: 1 },
].map((c) => ({ ...c, defaultValue: schema.settings[c.key]?.default }));

export function studioValue(settings, control) {
  const raw = settings?.[control.key] ?? control.defaultValue;
  return control.wire === 'pixel_boost' ? Number.parseInt(raw, 10) : raw;
}

export function studioPayload(settings) {
  return Object.fromEntries(STUDIO_CONTROLS.map((c) => [c.wire, studioValue(settings, c)]));
}

export function studioSetting(control, value) {
  return [control.key, control.wire === 'pixel_boost' ? `${value}px` : value];
}
