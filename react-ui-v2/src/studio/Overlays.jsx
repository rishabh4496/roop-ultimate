import React, { useEffect, useRef } from 'react';
import { flatten, newZone, addPoint, closeZone, MODE_COLOR, activeAt } from '@engine/components/faceswap/rotoModel.js';
import { useStudio } from '../engine/studio';
import { clipFxOf } from '../engine/payload';

// Viewport overlays for the tools. Both sit inside the viewport's stage (the box the frame is drawn
// in, zoom and pan included), so a pointer maps to FRAME PIXELS as
//   x = (clientX - box.left) / box.width * frameWidth
// which is the space the engine reads masks and roto zones in.

const toFrame = (e, el, nat) => {
  const r = el.getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * nat.w, ((e.clientY - r.top) / r.height) * nat.h];
};

// ── Mask brush ───────────────────────────────────────────────────────────────
// Two canvases at frame size: the EXPORT one (white on transparent - ProcessMgr decodes it as
// grayscale, painted = 255) and the visible one (the same strokes in red). Painted areas KEEP THE
// ORIGINAL face (exclude), as in React UI 1.0.
const exportRef = { current: null };

/** The painted mask as a PNG data URL, or null when nothing is painted. */
export function exportMask() {
  const c = exportRef.current;
  if (!c) return null;
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  for (let i = 3; i < d.length; i += 16) if (d[i] > 8) return c.toDataURL('image/png');
  return null;
}

export function MaskCanvas({ nat, editing }) {
  const visible = useRef(null);
  const brush = useStudio((s) => s.brush);
  const mask = useStudio((s) => s.mask);
  const last = useRef(null);

  // (Re)size to the frame and repaint the committed mask into both canvases.
  useEffect(() => {
    if (!nat.w || !nat.h) return;
    const exp = exportRef.current?.width === nat.w && exportRef.current?.height === nat.h ? exportRef.current : document.createElement('canvas');
    exp.width = nat.w; exp.height = nat.h;
    exportRef.current = exp;
    const vis = visible.current;
    vis.width = nat.w; vis.height = nat.h;
    const ex = exp.getContext('2d'); const vx = vis.getContext('2d');
    ex.clearRect(0, 0, nat.w, nat.h); vx.clearRect(0, 0, nat.w, nat.h);
    if (!mask.dataUrl) return;
    const img = new Image();
    img.onload = () => {
      ex.drawImage(img, 0, 0, nat.w, nat.h);
      vx.drawImage(img, 0, 0, nat.w, nat.h);
      vx.globalCompositeOperation = 'source-in';
      vx.fillStyle = '#fb7185';
      vx.fillRect(0, 0, nat.w, nat.h);
      vx.globalCompositeOperation = 'source-over';
    };
    img.src = mask.dataUrl;
  }, [nat.w, nat.h, mask.version, mask.dataUrl]);

  const stroke = (from, to, radius) => {
    for (const [c, color] of [[exportRef.current, '#ffffff'], [visible.current, '#fb7185']]) {
      const g = c.getContext('2d');
      g.globalCompositeOperation = brush.erase ? 'destination-out' : 'source-over';
      g.strokeStyle = color; g.fillStyle = color; g.lineWidth = radius * 2; g.lineCap = 'round';
      g.beginPath(); g.moveTo(from[0], from[1]); g.lineTo(to[0], to[1]); g.stroke();
      g.beginPath(); g.arc(to[0], to[1], radius, 0, Math.PI * 2); g.fill();
      g.globalCompositeOperation = 'source-over';
    }
  };
  const down = (e) => {
    if (!editing || e.button !== 0) return;
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    const p = toFrame(e, e.currentTarget, nat);
    const radius = (brush.size / 2) * (nat.w / e.currentTarget.getBoundingClientRect().width);
    last.current = { p, radius };
    stroke(p, p, radius);
  };
  const move = (e) => {
    if (!last.current) return;
    const p = toFrame(e, e.currentTarget, nat);
    stroke(last.current.p, p, last.current.radius);
    last.current.p = p;
  };
  const up = () => { last.current = null; };
  return (
    <canvas ref={visible} data-face className="absolute inset-0 h-full w-full" aria-label="Mask brush"
      style={{ opacity: editing ? 0.55 : 0.28, cursor: editing ? 'crosshair' : 'default', pointerEvents: editing ? 'auto' : 'none', zIndex: 4 }}
      onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up} data-testid="mask-canvas" />
  );
}

// ── Roto zones ───────────────────────────────────────────────────────────────
export function RotoLayer({ nat, editing, frame }) {
  const s = useStudio();
  const zones = clipFxOf(s).roto;
  const update = s.updateClipFx;
  const open = zones.find((z) => !z.closed);
  const click = (e) => {
    if (!editing || e.button !== 0) return;
    e.stopPropagation();
    const [x, y] = toFrame(e, e.currentTarget, nat);
    update((fx) => {
      const list = fx.roto;
      const cur = list.find((z) => !z.closed);
      if (!cur) return { roto: [...list, addPoint(newZone({ mode: 'exclude' }), x, y)] };
      const first = cur.points[0];
      const tol = 12 * (nat.w / e.currentTarget.getBoundingClientRect().width);
      if (first && cur.points.length >= 3 && Math.hypot(first.x - x, first.y - y) <= tol) {
        return { roto: list.map((z) => (z.id === cur.id ? closeZone(z) : z)) };
      }
      return { roto: list.map((z) => (z.id === cur.id ? addPoint(z, x, y) : z)) };
    });
  };
  const dbl = (e) => {
    if (!editing) return;
    e.stopPropagation();
    update((fx) => ({ roto: fx.roto.map((z) => (!z.closed ? closeZone(z) : z)) }));
  };
  if (!nat.w || (!zones.length && !editing)) return null;
  return (
    <svg className="absolute inset-0 h-full w-full" viewBox={`0 0 ${nat.w} ${nat.h}`} preserveAspectRatio="none"
      style={{ pointerEvents: editing ? 'auto' : 'none', cursor: editing ? 'crosshair' : 'default', zIndex: 5 }}
      onPointerDown={click} onDoubleClick={dbl} data-face data-testid="roto-layer">
      {zones.map((z) => {
        const pts = z.closed ? flatten(z.points, 16, true) : z.points.map((p) => [p.x, p.y]);
        const d = pts.length ? `M${pts.map(([x, y]) => `${x},${y}`).join(' L')}${z.closed ? ' Z' : ''}` : '';
        const color = MODE_COLOR[z.mode] || '#fb7185';
        const live = !z.closed || activeAt(z, frame);
        return (
          <g key={z.id} opacity={live ? 1 : 0.3}>
            {d && <path d={d} fill={z.closed ? color : 'none'} fillOpacity={0.18} stroke={color} strokeWidth={2} vectorEffect="non-scaling-stroke" strokeDasharray={z.closed ? undefined : '6 4'} />}
            {(editing || !z.closed) && z.points.map((p, i) => (
              <circle key={i} cx={p.x} cy={p.y} r={Math.max(3, nat.w / 260)} fill={i === 0 && !z.closed ? '#fff' : color} stroke="#000" strokeWidth={1} vectorEffect="non-scaling-stroke" />
            ))}
          </g>
        );
      })}
      {editing && open && <text x={12} y={nat.h - 14} fontSize={Math.max(12, nat.w / 70)} fill="#fff" opacity="0.8">click the first point (or double-click) to close the zone</text>}
    </svg>
  );
}
