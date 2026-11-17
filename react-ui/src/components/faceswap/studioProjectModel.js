import { serializeParameterKeyframes } from './timeline/parameterKeyframes';

export function serializeStudioTimeline(project) {
  const bindings = (project.bindings || []).map((binding) => {
    const intervals = (project.intervals || []).filter((range) => range.track_id === binding.track_id);
    return intervals.length ? { ...binding,
      enabled: binding.enabled !== false && intervals.some((range) => range.locked !== false),
      active_ranges: intervals.filter((range) => range.locked !== false).map((range) => [range.frame_start, range.frame_end]),
    } : binding;
  });
  return { version: 1, keyframes: serializeParameterKeyframes(project.keyframes || []), bindings };
}

export function restoreAutomationProjects(current, restored) {
  const next = { ...current };
  for (const key of new Set([...Object.keys(current), ...Object.keys(restored)])) {
    next[key] = { ...current[key], keyframes: [], bindings: [], intervals: [], strokes: [], trim: null, ...restored[key] };
  }
  return next;
}
