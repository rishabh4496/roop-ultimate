/**
 * rulerModel.ts: pixel-aware tick layout for the NLE ruler.
 *
 * The old generator picked its step from the visible FRAME span alone, so the same span drawn into
 * a 600 px window and a 3000 px window had the same tick count: crowded labels in one, sparse in
 * the other. Here the step comes from pixels per frame: the smallest step that keeps minor ticks
 * `minMinorPx` apart, and a major step (a whole multiple of it) that keeps labels `minMajorPx`
 * apart. Steps come from a ladder aligned to the frame rate, so major ticks land on whole seconds
 * once a second is wide enough to label.
 *
 * Every major tick carries BOTH labels the brief asks for: SMPTE `HH:MM:SS:FF` and the exact frame
 * number `#00412`. (SMPTE here is non-drop-frame, like the rest of the app: frameToTimecode rounds
 * the rate, so 29.97 fps timecode runs ~3.6 s/hour behind wall clock. The frame number is exact.)
 */

import { frameToTimecode } from '../../../utils/ptsFrameMap';

export interface RulerTick {
  frame: number;
  /** 0..1 across the visible span */
  ratio: number;
  major: boolean;
  /** HH:MM:SS:FF */
  timecode: string;
  /** #00412 */
  frameLabel: string;
}

export interface RulerLayout {
  ticks: RulerTick[];
  minorStep: number;
  majorStep: number;
  pxPerFrame: number;
}

export interface RulerOptions {
  viewStart: number;
  viewEnd: number;
  widthPx: number;
  fps: number;
  minMinorPx?: number;
  minMajorPx?: number;
  /** hard cap on emitted ticks, a guard against a zero-width window asking for millions */
  maxTicks?: number;
}

/** Exact, zero-padded frame number: `#00412`. Wider numbers keep their digits. */
export function formatFrameLabel(frame: number): string {
  const f = Math.max(0, Math.round(frame));
  return `#${String(f).padStart(5, '0')}`;
}

/** Candidate steps in frames, ascending: sub-second counts, then whole-second multiples. */
export function stepLadder(fps: number): number[] {
  const fr = Math.max(1, Math.round(fps));
  const set = new Set<number>();
  for (const f of [1, 2, 5, 10]) set.add(f);
  if (fr % 2 === 0 && fr / 2 > 5) set.add(fr / 2);
  for (const s of [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 18000, 36000]) set.add(s * fr);
  return [...set].sort((a, b) => a - b);
}

export function buildRuler({
  viewStart,
  viewEnd,
  widthPx,
  fps,
  minMinorPx = 7,
  minMajorPx = 104,
  maxTicks = 1500,
}: RulerOptions): RulerLayout {
  const span = Math.max(1, viewEnd - viewStart);
  const width = Math.max(1, widthPx);
  const pxPerFrame = width / span;
  const ladder = stepLadder(fps);

  const minorStep = ladder.find((s) => s * pxPerFrame >= minMinorPx) ?? ladder[ladder.length - 1];
  let majorStep = ladder.find((s) => s >= minorStep && s % minorStep === 0 && s * pxPerFrame >= minMajorPx);
  if (majorStep === undefined) {
    majorStep = minorStep * Math.max(1, Math.ceil(minMajorPx / Math.max(1e-9, minorStep * pxPerFrame)));
  }

  const ticks: RulerTick[] = [];
  const first = Math.ceil(viewStart / minorStep) * minorStep;
  for (let f = first; f <= viewEnd && ticks.length < maxTicks; f += minorStep) {
    const major = f % majorStep === 0;
    ticks.push({
      frame: f,
      ratio: (f - viewStart) / span,
      major,
      timecode: major ? frameToTimecode(f, fps) : '',
      frameLabel: major ? formatFrameLabel(f) : '',
    });
  }
  return { ticks, minorStep, majorStep, pxPerFrame };
}
