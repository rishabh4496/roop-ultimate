/**
 * parameterKeyframes.ts: Model and interpolation engine for NLE Keyframe Lane ('KF').
 * Supports: 'Swap Opacity', 'Mask Feather', 'Yaw Erosion', 'Color Match'.
 * Curves: 'Linear', 'Ease In/Out', 'Hold'.
 * 100% Offline, exact integer frame indices.
 */

import { clampFrameIdx } from '../../../utils/ptsFrameMap';

export type TimelineParamType =
  | 'Swap Opacity'
  | 'Mask Feather'
  | 'Yaw Erosion'
  | 'Color Match';

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
}

export const PARAMETER_CONFIGS: Record<TimelineParamType, ParameterConfig> = {
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
    min: 0,
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
};

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
  return clamped < 0.5
    ? 4 * clamped * clamped * clamped
    : 1 - Math.pow(-2 * clamped + 2, 3) / 2;
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
  const clampedVal = Math.max(cfg.min, Math.min(cfg.max, rawVal));

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

      if (kf0.curve === 'Hold') {
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
