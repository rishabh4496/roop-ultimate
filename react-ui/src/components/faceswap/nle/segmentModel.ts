/**
 * segmentModel.ts: blocks for the Face Swap Segment / Override lane.
 *
 * A block is a named frame range with a rule attached: a queue segment (the piece of the clip a
 * separate render job covers), a per-range recipe override, a bypass (the swap is switched off there).
 * Frames are 0-based and inclusive here; FaceSwap converts from the 1-based ranges its stores keep.
 *
 * Overlap is normal (an override sits inside a segment), so blocks are packed into sub-rows instead of
 * painting over one another: the first row whose last block ends before this one starts.
 */

export type SegmentKind = 'segment' | 'override' | 'bypass' | 'range';

export interface SegmentBlock {
  id: string;
  start: number;
  end: number;
  label: string;
  kind: SegmentKind;
  color?: string;
}

export const SEGMENT_KIND_COLORS: Record<SegmentKind, string> = {
  segment: '#38bdf8',
  override: '#a78bfa',
  bypass: '#94a3b8',
  range: '#34d399',
};

export const SEGMENT_KIND_LABELS: Record<SegmentKind, string> = {
  segment: 'Segment',
  override: 'Override',
  bypass: 'Bypass (swap off)',
  range: 'Range',
};

export interface PackedBlock extends SegmentBlock {
  row: number;
}

export function normalizeBlocks(blocks: ReadonlyArray<SegmentBlock> | null | undefined, maxFrames: number): SegmentBlock[] {
  const last = Math.max(0, Math.floor(maxFrames) - 1);
  const out: SegmentBlock[] = [];
  for (const b of blocks || []) {
    if (!b || !Number.isFinite(b.start) || !Number.isFinite(b.end)) continue;
    const start = Math.max(0, Math.min(last, Math.round(Math.min(b.start, b.end))));
    const end = Math.max(0, Math.min(last, Math.round(Math.max(b.start, b.end))));
    out.push({ ...b, start, end });
  }
  return out.sort((a, b) => a.start - b.start || a.end - b.end);
}

export function packRows(blocks: ReadonlyArray<SegmentBlock>): { packed: PackedBlock[]; rows: number } {
  const rowEnds: number[] = [];
  const packed: PackedBlock[] = [];
  for (const b of blocks) {
    let row = rowEnds.findIndex((end) => end < b.start);
    if (row < 0) {
      row = rowEnds.length;
      rowEnds.push(b.end);
    } else {
      rowEnds[row] = b.end;
    }
    packed.push({ ...b, row });
  }
  return { packed, rows: Math.max(1, rowEnds.length) };
}

/** Top-most (last drawn) block under `frame` in `row`, or null. */
export function blockAt(packed: ReadonlyArray<PackedBlock>, frame: number, row?: number): PackedBlock | null {
  for (let i = packed.length - 1; i >= 0; i--) {
    const b = packed[i];
    if (frame >= b.start && frame <= b.end && (row === undefined || b.row === row)) return b;
  }
  return null;
}
