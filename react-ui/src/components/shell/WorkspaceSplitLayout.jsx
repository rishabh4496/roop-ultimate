import React, { useLayoutEffect, useRef, useState } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { DEFAULT_PANEL_SIZES, useWorkspaceStore } from '../../store/workspaceStore';
import PanelResizeDivider from '../faceswap/PanelResizeDivider';
/** Slot-based docking shell. Each pane owns its scroll. */
export default function WorkspaceSplitLayout({ left, children, right, bottom, queue,
  leftOpen = true, rightOpen = true, onToggleLeft, onToggleRight }) {
  const root = useRef(null);
  const [bounds, setBounds] = useState({ width: 1920, height: 1000 });
  const { mode, sizes, resize, epoch } = useWorkspaceStore(useShallow((s) => ({
    mode: s.workspaceMode, sizes: s.panelSizes, resize: s.setPanelSize, epoch: s.layoutEpoch,
  })));
  useLayoutEffect(() => {
    const el = root.current;
    if (!el) return;
    const update = () => setBounds({ width: el.clientWidth, height: el.clientHeight });
    update(); const observer = new ResizeObserver(update); observer.observe(el);
    return () => observer.disconnect();
  }, []);
  const [expanded, setExpanded] = useState({ left: false, right: false });
  const wide = mode === 'dual' && bounds.width > 2100 && !!queue;
  const showLeft = leftOpen || expanded.left;
  const showRight = rightOpen || expanded.right;
  const budget = Math.max(0, bounds.width - 360 - (wide ? sizes.queueWidth + 1 : 0) - 2);
  const wanted = (showLeft ? sizes.leftWidth : 32) + (showRight ? sizes.rightWidth : 32);
  const scale = Math.min(1, budget / Math.max(1, wanted));
  const leftWidth = showLeft ? Math.max(100, Math.floor(sizes.leftWidth * scale)) : 32;
  const rightWidth = showRight ? Math.max(100, Math.floor(sizes.rightWidth * scale)) : 32;
  const toggle = (side) => {
    const handler = side === 'left' ? onToggleLeft : onToggleRight;
    if (handler && mode !== 'cinema' && mode !== 'timeline') handler();
    else setExpanded((v) => ({ ...v, [side]: !v[side] }));
  };
  useLayoutEffect(() => setExpanded({ left: false, right: false }), [epoch]);
  const divider = (panel, value, side, max) => <PanelResizeDivider
    currentSize={value} side={side} min={140} max={max}
    onResize={(v) => resize(panel, v)} onReset={() => resize(panel, DEFAULT_PANEL_SIZES[panel])}
    title={`Resize ${panel === 'leftWidth' ? 'Assets' : panel === 'rightWidth' ? 'Inspector' : 'Live Queue'} panel`} />;
  const pane = (side, label, open, content) => <section className={`workspace-pane workspace-${side} ${open ? '' : 'is-collapsed'}`} aria-label={label}>
    <div className="workspace-pane-heading"><button type="button" aria-label={`${open ? 'Collapse' : 'Expand'} ${label}`} aria-expanded={open} onClick={() => toggle(side)} title={label}>{side === 'left' ? '\u25e7' : '\u25e8'}</button>{open && <span>{label}</span>}</div>
    <div className="workspace-pane-scroll" hidden={!open}>{content}{side === 'right' && !wide && queue}</div>
  </section>;
  return <div ref={root} className="workspace-split" data-testid="workspace-split" data-layout={mode}
    style={{ gridTemplateColumns: `${leftWidth}px 1px minmax(0, 1fr) 1px ${rightWidth}px${wide ? ` 1px ${sizes.queueWidth}px` : ''}` }}>
    {pane('left', 'Assets / Facesets', showLeft, left)}
    {divider('leftWidth', leftWidth, 'left', Math.max(240, bounds.width - rightWidth - 360))}
    <section className="workspace-center" aria-label="Center stage" style={{ gridTemplateRows: bottom ? `minmax(0, 1fr) 1px minmax(0, ${Math.min(sizes.bottomHeight, bounds.height * 0.48)}px)` : 'minmax(0, 1fr)' }}>
      <div className="workspace-stage">{children}</div>
      {bottom && <><PanelResizeDivider direction="horizontal" side="bottom" currentSize={Math.min(sizes.bottomHeight, bounds.height * 0.48)} min={140} max={Math.min(600, bounds.height * 0.48)} onResize={(v) => resize('bottomHeight', v)} onReset={() => resize('bottomHeight', DEFAULT_PANEL_SIZES.bottomHeight)} title="Resize Timeline panel" /><section className="workspace-timeline" aria-label="Timeline">{bottom}</section></>}
    </section>
    {divider('rightWidth', rightWidth, 'right', Math.max(240, bounds.width - leftWidth - 360))}
    {pane('right', 'Parameters / Diagnostics', showRight, right)}
    {wide && <>{divider('queueWidth', sizes.queueWidth, 'right', 500)}<section className="workspace-pane workspace-queue" aria-label="Live Queue"><div className="workspace-pane-heading">Live Queue</div><div className="workspace-pane-scroll">{queue}</div></section></>}
  </div>;
}
