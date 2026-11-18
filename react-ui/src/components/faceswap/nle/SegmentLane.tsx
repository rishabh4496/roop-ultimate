/**
 * SegmentLane: Lane 3. Blocks for the swap rules / segment splits across time, and the draggable
 * chapter markers (colour tag + note) on a row above them.
 *
 * Markers drag on the lane with pointer capture, snap through the timeline's own snap function, and
 * commit once on release (a live drag does not write storage per pointermove). A click without a drag
 * seeks to the marker and opens its popover. When the lane is locked, markers do not drag and their
 * popover is read-only; blocks still select and seek.
 */

import React, { useMemo, useRef, useState } from 'react';
import type { ChapterMarker } from '../timeline/timelineSnap';
import { clientXToFrame } from './nleGeometry';
import { SEGMENT_KIND_COLORS, SEGMENT_KIND_LABELS, normalizeBlocks, packRows, type PackedBlock, type SegmentBlock } from './segmentModel';
import { useElementSize } from './useElementSize';

export interface SegmentLaneProps {
  blocks: ReadonlyArray<SegmentBlock>;
  markers: ReadonlyArray<ChapterMarker>;
  viewStart: number;
  viewEnd: number;
  maxFrames: number;
  locked: boolean;
  selectedMarkerId: string | null;
  onSeek: (frame: number) => void;
  onBlockClick?: (block: SegmentBlock) => void;
  onMarkerSelect: (marker: ChapterMarker, anchor: { left: number; top: number }) => void;
  onMarkerMove: (marker: ChapterMarker, frame: number) => void;
  /** snap a raw frame (the timeline's magnetic snapping), returns the frame to use */
  snap: (frame: number) => number;
  onScrubDown: (e: React.PointerEvent<HTMLDivElement>) => void;
  emptyHint?: string;
}

const FLAG_ROW = 15;
const DRAG_THRESHOLD_PX = 3;

function SegmentLaneImpl({
  blocks, markers, viewStart, viewEnd, maxFrames, locked, selectedMarkerId,
  onSeek, onBlockClick, onMarkerSelect, onMarkerMove, snap, onScrubDown, emptyHint,
}: SegmentLaneProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const { width, height } = useElementSize(wrapRef);
  const span = Math.max(1, viewEnd - viewStart);
  const pct = (f: number) => ((f - viewStart) / span) * 100;
  const [drag, setDrag] = useState<{ id: string; frame: number } | null>(null);
  const dragRef = useRef<{ marker: ChapterMarker; startX: number; moved: boolean; frame: number } | null>(null);

  const { packed, rows } = useMemo(() => packRows(normalizeBlocks(blocks, maxFrames)), [blocks, maxFrames]);
  const blockTop = FLAG_ROW;
  const blockAreaH = Math.max(8, height - FLAG_ROW - 2);
  const rowH = Math.max(8, Math.min(26, Math.floor(blockAreaH / rows)));

  const frameFromX = (clientX: number) => {
    const rect = wrapRef.current?.getBoundingClientRect();
    return clientXToFrame(clientX, rect ? { left: rect.left, width: rect.width } : null, viewStart, viewEnd, maxFrames);
  };

  const onFlagDown = (e: React.PointerEvent<HTMLDivElement>, marker: ChapterMarker) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    e.currentTarget.setPointerCapture?.(e.pointerId);
    dragRef.current = { marker, startX: e.clientX, moved: false, frame: marker.frame };
  };
  const onFlagMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const d = dragRef.current;
    if (!d || locked) return;
    if (!d.moved && Math.abs(e.clientX - d.startX) < DRAG_THRESHOLD_PX) return;
    d.moved = true;
    d.frame = snap(frameFromX(e.clientX));
    setDrag({ id: d.marker.id, frame: d.frame });
  };
  const onFlagUp = (e: React.PointerEvent<HTMLDivElement>) => {
    const d = dragRef.current;
    dragRef.current = null;
    e.currentTarget.releasePointerCapture?.(e.pointerId);
    setDrag(null);
    if (!d) return;
    if (d.moved && !locked) {
      if (d.frame !== d.marker.frame) onMarkerMove(d.marker, d.frame);
      return;
    }
    const rect = e.currentTarget.getBoundingClientRect();
    onSeek(d.marker.frame);
    onMarkerSelect(d.marker, { left: rect.left + rect.width / 2, top: rect.top });
  };

  const visibleMarkers = markers.filter((m) => {
    const f = drag?.id === m.id ? drag.frame : m.frame;
    return f >= viewStart - 1 && f <= viewEnd + 1;
  });

  const blockClick = (b: PackedBlock) => {
    onSeek(b.start);
    onBlockClick?.(b);
  };

  return (
    <div ref={wrapRef} className="absolute inset-0 overflow-hidden" onPointerDown={onScrubDown} data-testid="nle-segment-surface">
      {packed.length === 0 && markers.length === 0 && emptyHint ? (
        <div className="absolute inset-0 flex items-end px-3 pb-0.5 text-mini text-neutral-500 italic pointer-events-none">{emptyHint}</div>
      ) : null}

      {packed.map((b) => {
        if (b.end < viewStart || b.start > viewEnd) return null;
        const left = Math.max(0, pct(b.start));
        const right = Math.min(100, pct(b.end + 1));
        const color = b.color || SEGMENT_KIND_COLORS[b.kind];
        const pxW = ((right - left) / 100) * width;
        const label = `${b.label} · ${SEGMENT_KIND_LABELS[b.kind]} · frames ${b.start}–${b.end}`;
        return (
          <div
            key={b.id}
            role="button"
            tabIndex={0}
            aria-label={label}
            title={label}
            data-testid={`segment-block-${b.kind}`}
            data-start={b.start}
            data-end={b.end}
            onPointerDown={(e) => e.stopPropagation()}
            onClick={() => blockClick(b)}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); blockClick(b); } }}
            className="absolute rounded-[4px] border overflow-hidden cursor-pointer transition-[filter] hover:brightness-125 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-cyan,#00F0FF)]"
            style={{
              left: `${left}%`,
              width: `${Math.max(0.3, right - left)}%`,
              top: blockTop + b.row * rowH,
              height: rowH - 2,
              borderColor: color,
              background: b.kind === 'bypass'
                ? `repeating-linear-gradient(135deg, ${color}55 0 6px, ${color}22 6px 12px)`
                : `linear-gradient(90deg, ${color}66, ${color}33)`,
            }}
          >
            {pxW > 46 && rowH >= 14 ? (
              <span className="px-1.5 text-mini font-semibold text-white leading-none whitespace-nowrap flex items-center h-full">{b.label}</span>
            ) : null}
          </div>
        );
      })}

      {visibleMarkers.map((m) => {
        const f = drag?.id === m.id ? drag.frame : m.frame;
        const color = m.color || '#38bdf8';
        const selected = selectedMarkerId === m.id;
        return (
          <div
            key={m.id}
            role="button"
            tabIndex={0}
            aria-label={`Marker ${m.label} at frame ${f}${locked ? ' (locked)' : '. Drag to move'}`}
            title={`${m.label} · #${String(f).padStart(5, '0')}${m.note ? `\n${m.note}` : ''}`}
            data-testid="nle-marker-flag"
            data-marker-id={m.id}
            data-frame={f}
            onPointerDown={(e) => onFlagDown(e, m)}
            onPointerMove={onFlagMove}
            onPointerUp={onFlagUp}
            onPointerCancel={onFlagUp}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                e.stopPropagation();
                const r = e.currentTarget.getBoundingClientRect();
                onSeek(m.frame);
                onMarkerSelect(m, { left: r.left + r.width / 2, top: r.top });
              }
            }}
            className={`absolute top-0 bottom-0 z-20 touch-none focus:outline-none ${locked ? 'cursor-pointer' : 'cursor-ew-resize'}`}
            style={{ left: `${pct(f)}%`, width: 14, marginLeft: -1 }}
          >
            <div className="absolute top-0 bottom-0 w-px" style={{ left: 1, backgroundColor: color, opacity: selected ? 1 : 0.7 }} />
            <div
              className={`absolute top-0 left-px h-[13px] min-w-[12px] pl-1 pr-1.5 rounded-r-[3px] flex items-center text-black font-bold shadow ${selected ? 'ring-2 ring-white' : ''}`}
              style={{ backgroundColor: color }}
            >
              {m.note ? <span className="max-w-[86px] truncate text-micro leading-none">{m.label}</span> : null}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export const SegmentLane = React.memo(SegmentLaneImpl);
export default SegmentLane;
