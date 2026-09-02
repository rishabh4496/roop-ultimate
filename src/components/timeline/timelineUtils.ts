export interface FaceTrackingRange {
  start: number;
  end: number;
  faceId: string;
  label?: string;
  confidence?: number;
}

export interface TimelineDropoutRange {
  start: number;
  end: number;
}

/**
 * Converts a frame index to SMPTE timecode string: HH:MM:SS:FF
 */
export function frameToTimecode(frame: number, fps = 30): string {
  const f = Math.max(0, Math.round(frame));
  const safeFps = Math.max(1, Math.round(fps));
  const ff = f % safeFps;
  const totalSeconds = Math.floor(f / safeFps);
  const ss = totalSeconds % 60;
  const mm = Math.floor(totalSeconds / 60) % 60;
  const hh = Math.floor(totalSeconds / 3600);
  const p = (n: number) => n.toString().padStart(2, '0');
  return `${p(hh)}:${p(mm)}:${p(ss)}:${p(ff)}`;
}

/**
 * Converts SMPTE timecode string (HH:MM:SS:FF) back to frame index
 */
export function timecodeToFrame(timecode: string, fps = 30): number {
  const parts = timecode.split(':').map((p) => parseInt(p, 10) || 0);
  const safeFps = Math.max(1, Math.round(fps));
  if (parts.length === 4) {
    const [hh, mm, ss, ff] = parts;
    return (hh * 3600 + mm * 60 + ss) * safeFps + ff;
  }
  return 0;
}

/**
 * Computes complementary dropout ranges where no face tracking is present.
 */
export function calculateDropoutRanges(
  totalFrames: number,
  ranges: FaceTrackingRange[]
): TimelineDropoutRange[] {
  if (totalFrames <= 0) return [];
  if (!ranges || ranges.length === 0) {
    return [{ start: 0, end: totalFrames }];
  }

  // Sort ranges by start frame ascending
  const sorted = [...ranges].sort((a, b) => a.start - b.start);
  const dropouts: TimelineDropoutRange[] = [];

  let currentPointer = 0;
  for (const r of sorted) {
    const rStart = Math.max(0, Math.min(totalFrames, r.start));
    const rEnd = Math.max(0, Math.min(totalFrames, r.end));

    if (rStart > currentPointer) {
      dropouts.push({ start: currentPointer, end: rStart });
    }
    currentPointer = Math.max(currentPointer, rEnd);
  }

  if (currentPointer < totalFrames) {
    dropouts.push({ start: currentPointer, end: totalFrames });
  }

  return dropouts;
}

/**
 * Computes face tracking coverage percentage across the total duration.
 */
export function calculateCoveragePercentage(
  totalFrames: number,
  ranges: FaceTrackingRange[]
): { trackedPercent: number; dropoutPercent: number } {
  if (totalFrames <= 0) return { trackedPercent: 0, dropoutPercent: 0 };

  const mask = new Uint8Array(totalFrames);
  ranges.forEach((r) => {
    const s = Math.max(0, Math.min(totalFrames - 1, r.start));
    const e = Math.max(0, Math.min(totalFrames, r.end));
    for (let i = s; i < e; i++) {
      mask[i] = 1;
    }
  });

  let trackedCount = 0;
  for (let i = 0; i < totalFrames; i++) {
    if (mask[i] === 1) trackedCount++;
  }

  const trackedPercent = parseFloat(((trackedCount / totalFrames) * 100).toFixed(1));
  const dropoutPercent = parseFloat((100 - trackedPercent).toFixed(1));

  return { trackedPercent, dropoutPercent };
}
