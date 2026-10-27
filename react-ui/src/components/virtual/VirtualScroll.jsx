import React, { useCallback, useRef } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import useContentWidth from './useContentWidth';

// ── Windowed lists and grids ────────────────────────────────────────────────
//
// Outputs, History, the Face Manager's face grid and the terminal all used to render every row they were
// given. A 2,000-file output folder was 2,000 cards (and, with the video tiles, 2,000 media elements); a
// 500-crop faceset was 500 <img> with a base64 payload each; both re-rendered in full on every state change.
// These two components mount only the rows near the scroll window.
//
// WHY AN INNER SCROLLER, NOT THE WINDOW. App.jsx applies the user's UI zoom as `document.documentElement.style.zoom`,
// and the page scrolls on `window`. A window virtualizer measures with getBoundingClientRect, whose units under CSS
// zoom are not the units scrollY is in, so row positions drift by the zoom factor. A bounded element scroller
// measures with the element's own offset/client sizes and scrollTop, which are all in the same (unzoomed, local) space.
// e2e/stage8-virtualization.spec.js scrolls to the last row at zoom 1.5 to hold that.
//
// Rows are placed with `transform: translate3d(0, y, 0)` rather than `top`, so scrolling moves compositor layers
// instead of re-running layout for every mounted row.

const DEFAULT_SCROLLER = 'max-h-[72vh]';

/**
 * Windowed rows with measured (variable) heights.
 *
 * `renderRow(index)` returns the row's content; the wrapper owns positioning and measurement, so a row that grows
 * (an expanded report, a wrapped log line) re-measures through its own ResizeObserver and the rows below move.
 *
 * The scroller's height is its content's, capped by `className` (default 72vh): a short list is not padded out to a
 * tall empty box, a long one scrolls.
 */
export function VirtualRows({
  count, estimateSize, renderRow, getItemKey, overscan = 6, gap = 0,
  className = DEFAULT_SCROLLER, style, label, role = 'region', scrollerRef, onScroll,
  innerRef, children, testId, ...rest
}) {
  const localRef = useRef(null);
  const setRef = useCallback((el) => {
    localRef.current = el;
    if (typeof scrollerRef === 'function') scrollerRef(el);
    else if (scrollerRef) scrollerRef.current = el;
  }, [scrollerRef]);

  const virtualizer = useVirtualizer({
    count,
    getScrollElement: () => localRef.current,
    estimateSize: () => estimateSize,
    overscan,
    gap,
    getItemKey,
    // Something to mount on the very first render, before the ResizeObserver reports the real height.
    initialRect: { width: 0, height: 720 },
  });

  const items = virtualizer.getVirtualItems();
  const total = virtualizer.getTotalSize();
  // The wrappers that position rows are not part of the accessible structure: inside a table's rowgroup or a list they
  // must not sit between the container and its rows.
  const wrapperRole = role === 'region' ? undefined : 'presentation';

  return (
    <div
      ref={setRef}
      role={role}
      aria-label={label}
      // A scroll container must be reachable without a pointer (axe: scrollable-region-focusable).
      tabIndex={0}
      onScroll={onScroll}
      data-testid={testId}
      data-virtual-total={count}
      data-virtual-mounted={items.length}
      className={`overflow-y-auto custom-scrollbar ${className}`}
      // Paint and layout of a row cannot leak out of the scroller, so a row that changes never re-flows the page.
      style={{ contain: 'layout style paint', overscrollBehavior: 'contain', ...style }}
      {...rest}
    >
      <div ref={innerRef} role={wrapperRole} style={{ height: total, width: '100%', position: 'relative' }}>
        {items.map((v) => (
          <div
            key={v.key}
            role={wrapperRole}
            data-index={v.index}
            ref={virtualizer.measureElement}
            style={{ position: 'absolute', top: 0, left: 0, width: '100%', transform: `translate3d(0, ${v.start}px, 0)` }}
          >
            {renderRow(v.index)}
          </div>
        ))}
      </div>
      {children}
    </div>
  );
}

/**
 * A windowed grid: `items` laid out in as many `minCell`-wide columns as fit, one virtual row per grid row.
 *
 * `cellHeight(cellWidth)` is the ESTIMATE for a row (cards whose content decides their height are re-measured, so
 * it only has to be close). `renderCell(item, index)` renders one cell; the wrapper supplies the CSS grid.
 */
export function VirtualGrid({
  items, minCell = 240, gap = 12, cellHeight, renderCell, getKey, overscan = 3,
  className, label, testId, scrollerRef, maxCols = 8, ...rest
}) {
  const [setWidthEl, width] = useContentWidth(0);
  const scroller = useRef(null);
  const attach = useCallback((el) => {
    scroller.current = el;
    setWidthEl(el);
    if (typeof scrollerRef === 'function') scrollerRef(el);
    else if (scrollerRef) scrollerRef.current = el;
  }, [setWidthEl, scrollerRef]);

  const usable = width || 960;           // before the first measure; replaced a frame later
  const cols = Math.max(1, Math.min(maxCols, Math.floor((usable + gap) / (minCell + gap))));
  const cellWidth = (usable - gap * (cols - 1)) / cols;
  const rowCount = Math.ceil(items.length / cols);

  return (
    <VirtualRows
      scrollerRef={attach}
      count={rowCount}
      estimateSize={cellHeight(cellWidth) + gap}
      // Rows are keyed by their first item, so a filter that keeps a card keeps its DOM (and its hover state).
      getItemKey={(r) => (getKey ? getKey(items[r * cols], r * cols) : r)}
      overscan={overscan}
      className={className}
      label={label}
      testId={testId}
      renderRow={(r) => {
        const start = r * cols;
        const slice = items.slice(start, start + cols);
        return (
          <div
            className="grid"
            style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, gap, paddingBottom: gap }}
          >
            {slice.map((item, i) => (
              <div key={getKey ? getKey(item, start + i) : start + i} className="min-w-0">
                {renderCell(item, start + i)}
              </div>
            ))}
          </div>
        );
      }}
      {...rest}
    />
  );
}
