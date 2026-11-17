/**
 * parameterKeyframes.ts: Model and interpolation engine for NLE Keyframe Lane ('KF').
 * Supports: 'Swap Opacity', 'Mask Feather', 'Yaw Erosion', 'Color Match'.
 * Curves: 'Linear', 'Ease In/Out', 'Hold'.
 * 100% Offline, exact integer frame indices.
 */

import { clampFrameIdx } from '../../../utils/ptsFrameMap';
import { STUDIO_CONTROLS } from '../studioControls';

export type TimelineParamType =
  | 'Swap Opacity'
  | 'Mask Feather'
  | 'Yaw Erosion'
  | 'Color Match'
  | 'detector_model' | 'multi_angle_detect' | 'stabilization_strength'
  | 'pose_adaptive_faceset' | 'mask_engine' | 'yaw_erosion_bias'
  | 'expression_restorer_weight' | 'preserve_inner_mouth' | 'pixel_boost'
  | 'ultramax_ramp' | 'temporal_restore_alpha' | 'restore_identity_weight'
  | 'color_transfer_mode' | 'grain_match_strength' | 'motion_blur_match';

export type TimelineCurveType = 'Linear' | 'Ease In/Out' | 'Hold';

export interface ParameterConfig {
  id: TimelineParamType;
  label: string;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
  unit: string;
  color: string;
  wire?: string;
  choices?: readonly (string | number | boolean)[];
}

export const PARAMETER_CONFIGS = {
  'Swap Opacity': {
    id: 'Swap Opacity',
    label: 'Swap Opacity',
    min: 0,
    max: 1,
    step: 0.01,
    defaultValue: 1.0,
    unit: '',
    color: '#38bdf8', // Sky 400
  },
  'Mask Feather': {
    id: 'Mask Feather',
    label: 'Mask Feather',
    min: 0,
    max: 100,
    step: 1,
    defaultValue: 20,
    unit: 'px',
    color: '#fbbf24', // Amber 400
  },
  'Yaw Erosion': {
    id: 'Yaw Erosion',
    label: 'Yaw Erosion',
    min: -1,
    max: 1,
    step: 0.01,
    defaultValue: 0.0,
    unit: '',
    color: '#f43f5e', // Rose 500
  },
  'Color Match': {
    id: 'Color Match',
    label: 'Color Match',
    min: 0,
    max: 1,
    step: 0.01,
    defaultValue: 1.0,
    unit: '',
    color: '#34d399', // Emerald 400
  },
} as Record<TimelineParamType, ParameterConfig>;

for (const control of STUDIO_CONTROLS) {
  const choices: readonly (string | number | boolean)[] | undefined = control.toggle ? [false, true] : control.choices;
  const rawDefault = control.wire === 'pixel_boost' ? Number.parseInt(String(control.defaultValue), 10) : control.defaultValue;
  PARAMETER_CONFIGS[control.wire as TimelineParamType] = {
    id: control.wire as TimelineParamType, wire: control.wire, label: control.label,
    min: choices ? 0 : control.min!, max: choices ? choices.length - 1 : control.max!,
    step: choices ? 1 : control.step || 0.01,
    defaultValue: choices ? Math.max(0, choices.indexOf(rawDefault as string | number | boolean)) : Number(rawDefault),
    choices, unit: control.unit || '', color: '#a78bfa',
  };
}

const WIRE_KEYS = { 'Swap Opacity': 'swap_opacity', 'Mask Feather': 'mask_feather', 'Yaw Erosion': 'yaw_erosion_bias', 'Color Match': 'color_match_strength' };
const CURVES = { Linear: 'linear', 'Ease In/Out': 'ease_in_out', Hold: 'hold' };

/** Group keys by absolute decoder frame, retaining a curve for each parameter. */
export function serializeParameterKeyframes(keyframes: NleKeyframe[]) {
  const frames = new Map<number, Record<string, unknown>>();
  for (const keyframe of sortAndNormalizeKeyframes(keyframes)) {
    const cfg = PARAMETER_CONFIGS[keyframe.param];
    if (!cfg) continue;
    const wire = cfg.wire || WIRE_KEYS[keyframe.param as keyof typeof WIRE_KEYS];
    if (!wire) continue;
    const frame = frames.get(keyframe.frame_idx) || { frame: keyframe.frame_idx, interpolations: {} };
    frame[wire] = cfg.choices ? cfg.choices[Math.max(0, Math.min(cfg.choices.length - 1, Math.round(keyframe.value)))] : keyframe.value;
    (frame.interpolations as Record<string, string>)[wire] = cfg.choices ? 'hold' : CURVES[keyframe.curve];
    frames.set(keyframe.frame_idx, frame);
  }
  return [...frames.values()].sort((a, b) => Number(a.frame) - Number(b.frame));
}

export interface NleKeyframe {
  id: string;
  frame_idx: number; // exact integer frame index
  param: TimelineParamType;
  value: number;
  curve: TimelineCurveType;
  selected?: boolean;
}

/**
 * Standard cubic ease-in-out formula.
 */
export function easeInOut(t: number): number {
  const clamped = Math.max(0, Math.min(1, t));
  return clamped * clamped * (3 - 2 * clamped);
}

/**
 * Create a new parameter keyframe with clamped bounds.
 */
export function createNleKeyframe(
  param: TimelineParamType,
  frame_idx: number,
  value?: number,
  curve: TimelineCurveType = 'Ease In/Out',
  maxFrames = Infinity
): NleKeyframe {
  const cfg = PARAMETER_CONFIGS[param] || PARAMETER_CONFIGS['Swap Opacity'];
  const rawVal = value !== undefined ? value : cfg.defaultValue;
  const bounded = Math.max(cfg.min, Math.min(cfg.max, rawVal));
  const clampedVal = cfg.choices ? Math.round(bounded) : bounded;

  return {
    id: `kf_${param.replace(/\s+/g, '_').toLowerCase()}_${frame_idx}_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    frame_idx: clampFrameIdx(frame_idx, 0, maxFrames),
    param,
    value: Number(clampedVal.toFixed(2)),
    curve,
    selected: false,
  };
}

/**
 * Sort keyframes and ensure single keyframe per frame for the same parameter.
 */
export function sortAndNormalizeKeyframes(
  keyframes: NleKeyframe[],
  maxFrames = Infinity
): NleKeyframe[] {
  const map = new Map<string, NleKeyframe>();

  for (const kf of keyframes || []) {
    if (!kf || !kf.param) continue;
    const clamped = {
      ...kf,
      frame_idx: clampFrameIdx(kf.frame_idx, 0, maxFrames),
    };
    const key = `${clamped.param}_${clamped.frame_idx}`;
    map.set(key, clamped);
  }

  return [...map.values()].sort((a, b) => {
    if (a.param !== b.param) return a.param.localeCompare(b.param);
    return a.frame_idx - b.frame_idx;
  });
}

/**
 * Evaluate the interpolated value of a parameter at any arbitrary frame.
 */
export function evaluateKeyframeValue(
  keyframes: NleKeyframe[],
  param: TimelineParamType,
  frame_idx: number
): number {
  const cfg = PARAMETER_CONFIGS[param] || PARAMETER_CONFIGS['Swap Opacity'];
  const filtered = (keyframes || [])
    .filter((k) => k.param === param)
    .sort((a, b) => a.frame_idx - b.frame_idx);

  if (filtered.length === 0) {
    return cfg.defaultValue;
  }

  const target = Math.round(frame_idx);

  // Before first keyframe: hold first value
  if (target <= filtered[0].frame_idx) {
    return filtered[0].value;
  }

  // After last keyframe: hold last value
  const last = filtered[filtered.length - 1];
  if (target >= last.frame_idx) {
    return last.value;
  }

  // Find surrounding segment
  for (let i = 0; i < filtered.length - 1; i++) {
    const kf0 = filtered[i];
    const kf1 = filtered[i + 1];

    if (target >= kf0.frame_idx && target <= kf1.frame_idx) {
      const span = kf1.frame_idx - kf0.frame_idx;
      if (span <= 0) return kf0.value;

      const t = (target - kf0.frame_idx) / span;

      if (kf0.curve === 'Hold' || cfg.choices) {
        return kf0.value;
      }
      if (kf0.curve === 'Linear') {
        return kf0.value + (kf1.value - kf0.value) * t;
      }
      // Ease In/Out
      return kf0.value + (kf1.value - kf0.value) * easeInOut(t);
    }
  }

  return last.value;
}

/**
 * Duplicate / copy keyframe offset by delta frames.
 */
export function duplicateKeyframe(
  kf: NleKeyframe,
  offsetFrames = 10,
  maxFrames = Infinity
): NleKeyframe {
  return {
    ...kf,
    id: `kf_${kf.param.replace(/\s+/g, '_').toLowerCase()}_${kf.frame_idx + offsetFrames}_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    frame_idx: clampFrameIdx(kf.frame_idx + offsetFrames, 0, maxFrames),
    selected: true,
  };
}
