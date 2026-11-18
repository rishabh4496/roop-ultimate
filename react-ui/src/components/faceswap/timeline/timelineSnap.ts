/**
 * timelineSnap.ts: Frame math, magnetic snapping, and zoom calculations for NLE Timeline.
 * 100% Offline, frame-accurate using exact integer frame indices.
 */

import {
  clampFrameIdx,
  frameToTimecode,
} from '../../../utils/ptsFrameMap';

export interface ChapterMarker {
  id: string;
  frame: number; // exact integer frame_idx
  label: string;
  color?: string;
  /** free-text note shown in the marker's popover (first line doubles as the label) */
  note?: string;
  createdAt?: number;
}

export interface SnapTarget {
  frame: number;
  type: 'playhead' | 'trim_in' | 'trim_out' | 'marker' | 'keyframe' | 'range_edge';
  label?: string;
}

export interface SnapResult {
  frame: number;
  snapped: boolean;
  snappedTo?: SnapTarget;
}

/**
 * Magnetically snap a candidate frame to nearby snap targets if within threshold.
 *
 * @param candidateFrame Raw candidate frame from pointer/drag
 * @param targets Array of snap targets (playhead, markers, in/out, keyframes)
 * @param thresholdFrames Max frame distance for snapping (default: 3 frames)
 */
export function snapToFrame(
  candidateFrame: number,
  targets: SnapTarget[],
  thresholdFrames = 3
): SnapResult {
  const f = Math.round(candidateFrame);
  if (!targets || targets.length === 0 || thresholdFrames <= 0) {
    return { frame: f, snapped: false };
  }

  let closestDist = Infinity;
  let bestTarget: SnapTarget | undefined;

  for (const t of targets) {
    const dist = Math.abs(t.frame - f);
    if (dist <= thresholdFrames && dist < closestDist) {
      closestDist = dist;
      bestTarget = t;
    }
  }

  if (bestTarget !== undefined) {
    return {
      frame: bestTarget.frame,
      snapped: true,
      snappedTo: bestTarget,
    };
  }

  return { frame: f, snapped: false };
}

/**
 * Frame-accurate zoom to mouse cursor.
 * Guarantees that the frame under the pointer remains at the exact same client position.
 *
 * @param viewRange Current visible [viewStart, viewEnd]
 * @param zoomFactor < 1 zooms in, > 1 zooms out
 * @param cursorFraction Relative cursor position across timeline runway [0..1]
 * @param maxFrames Total clip frame count
 * @param minSpan Minimum frames in view (default: 4 frames)
 */
export function calculateZoomToCursor(
  viewRange: [number, number],
  zoomFactor: number,
  cursorFraction: number,
  maxFrames: number,
  minSpan = 4
): [number, number] {
  const [start, end] = viewRange;
  const currentSpan = Math.max(1, end - start);
  const clampedMax = Math.max(1, Math.round(maxFrames));
  const targetSpan = Math.max(minSpan, Math.min(clampedMax - 1, Math.round(currentSpan * zoomFactor)));

  const ratio = Math.max(0, Math.min(1, cursorFraction));
  const anchorFrame = start + ratio * currentSpan;

  let newStart = Math.round(anchorFrame - targetSpan * ratio);
  let newEnd = newStart + targetSpan;

  if (newStart < 0) {
    newEnd = Math.min(clampedMax - 1, newEnd - newStart);
    newStart = 0;
  }
  if (newEnd >= clampedMax) {
    newStart = Math.max(0, newStart - (newEnd - (clampedMax - 1)));
    newEnd = clampedMax - 1;
  }

  return [
    clampFrameIdx(newStart, 0, clampedMax),
    clampFrameIdx(newEnd, 0, clampedMax),
  ];
}

/**
 * Generate adaptive ruler tick divisions based on visible frame span.
 */
export interface RulerTick {
  frame: number;
  label: string;
  major: boolean;
  timecode: string;
}

export function generateRulerTicks(
  viewStart: number,
  viewEnd: number,
  fps = 30
): RulerTick[] {
  const span = Math.max(1, viewEnd - viewStart);
  let step = 1;
  let majorInterval = 5;

  if (span > 5000) {
    step = 500;
    majorInterval = 5;
  } else if (span > 2000) {
    step = 250;
    majorInterval = 4;
  } else if (span > 1000) {
    step = 100;
    majorInterval = 5;
  } else if (span > 500) {
    step = 50;
    majorInterval = 5;
  } else if (span > 200) {
    step = 25;
    majorInterval = 4;
  } else if (span > 80) {
    step = 10;
    majorInterval = 5;
  } else if (span > 30) {
    step = 5;
    majorInterval = 5;
  } else if (span > 15) {
    step = 2;
    majorInterval = 5;
  } else {
    step = 1;
    majorInterval = 5;
  }

  const ticks: RulerTick[] = [];
  const firstTick = Math.ceil(viewStart / step) * step;

  for (let f = firstTick; f <= viewEnd; f += step) {
    const isMajor = f % (step * majorInterval) === 0 || f === 0;
    const tc = frameToTimecode(f, fps);
    let label = '';

    if (span <= 40) {
      label = `#${f}`;
    } else if (span <= 400) {
      label = tc.slice(3); // MM:SS:FF
    } else {
      label = tc; // HH:MM:SS:FF
    }

    ticks.push({
      frame: f,
      label,
      major: isMajor,
      timecode: tc,
    });
  }

  return ticks;
}
