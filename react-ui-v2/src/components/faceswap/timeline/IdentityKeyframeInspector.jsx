import React, { useEffect, useRef, useState } from 'react';
import { useApi } from '../../../useApi';
import { logFailure } from '../../../failureLog';
import { CURVES, CURVE_LABELS, DEFAULT_BEZIER, curvePath, uiFrameOf } from '../identityKeyframes';

// Edit one identity keyframe: which curve the segment to the NEXT keyframe follows (with a draggable bezier when it
// is a Bezier), and the share of each blended source at this keyframe. The numbers under it are the backend's own
// reading of the whole track (POST /api/identity/keyframes/evaluate): the angle between this keyframe's identity and
// the next, and the largest frame-to-frame step - what "smooth" means on the identity sphere.

const BOX = 132;
const PAD = 14;
const PRESETS = [
  ['Ease in-out', [0.42, 0, 0.58, 1]],
  ['Ease in', [0.42, 0, 1, 1]],
  ['Ease out', [0, 0, 0.58, 1]],
  ['Fast start', [0.1, 0.9, 0.3, 1]],
];

const pctText = (v) => `${Math.round(v)}%`;

function BezierEditor({ bezier, onChange }) {
  const ref = useRef(null);
  const [x1, y1, x2, y2] = bezier;
  const toBox = (x, y) => [PAD + x * (BOX - 2 * PAD), PAD + (1 - y) * (BOX - 2 * PAD)];
  const fromEvent = (e) => {
    const rect = ref.current.getBoundingClientRect();
    const sx = (e.clientX - rect.left) * (BOX / rect.width);
    const sy = (e.clientY - rect.top) * (BOX / rect.height);
    return [Math.max(0, Math.min(1, (sx - PAD) / (BOX - 2 * PAD))), Math.max(0, Math.min(1, 1 - (sy - PAD) / (BOX - 2 * PAD)))];
  };
  const handle = (which) => ({
    onPointerDown: (e) => { e.preventDefault(); e.currentTarget.setPointerCapture?.(e.pointerId); },
    onPointerMove: (e) => {
      if (!(e.buttons & 1)) return;
      const [x, y] = fromEvent(e);
      onChange(which === 1 ? [x, y, x2, y2] : [x1, y1, x, y]);
    },
    onKeyDown: (e) => {
      const d = { ArrowLeft: [-0.02, 0], ArrowRight: [0.02, 0], ArrowUp: [0, 0.02], ArrowDown: [0, -0.02] }[e.key];
      if (!d) return;
      e.preventDefault();
      const c = (n) => Math.max(0, Math.min(1, n));
      onChange(which === 1 ? [c(x1 + d[0]), c(y1 + d[1]), x2, y2] : [x1, y1, c(x2 + d[0]), c(y2 + d[1])]);
    },
  });
  const [p0x, p0y] = toBox(0, 0);
  const [p3x, p3y] = toBox(1, 1);
  const [h1x, h1y] = toBox(x1, y1);
  const [h2x, h2y] = toBox(x2, y2);
  return (
    <svg ref={ref} width={BOX} height={BOX} viewBox={`0 0 ${BOX} ${BOX}`} data-testid="identity-bezier-editor"
         className="touch-none select-none rounded-lg border border-white/10 bg-black/25" role="group"
         aria-label={`Bezier easing, cubic-bezier(${bezier.map((n) => n.toFixed(2)).join(', ')}). Drag the two handles, or focus one and use the arrow keys.`}>
      <line x1={p0x} y1={p0y} x2={h1x} y2={h1y} stroke="rgba(255,255,255,0.3)" />
      <line x1={p3x} y1={p3y} x2={h2x} y2={h2y} stroke="rgba(255,255,255,0.3)" />
      <path d={curvePath('cubic_bezier', bezier, BOX, BOX, PAD)} fill="none" stroke="var(--accent)" strokeWidth="2" />
      {[[1, h1x, h1y], [2, h2x, h2y]].map(([which, cx, cy]) => (
        <circle key={which} cx={cx} cy={cy} r="6" tabIndex={0} role="slider"
                aria-label={`Bezier handle ${which}`} aria-valuemin={0} aria-valuemax={1} aria-valuenow={Number((which === 1 ? x1 : x2).toFixed(2))}
                aria-valuetext={`x ${(which === 1 ? x1 : x2).toFixed(2)}, y ${(which === 1 ? y1 : y2).toFixed(2)}`}
                data-testid={`identity-bezier-handle-${which}`}
                fill="#fff" stroke="rgba(0,0,0,0.7)" strokeWidth="1.5" style={{ cursor: 'grab' }}
                className="outline-none focus-visible:stroke-[var(--accent)]" {...handle(which)} />
      ))}
    </svg>
  );
}

export default function IdentityKeyframeInspector({
  keyframe, nextKeyframe, components, recipe, labelOf, onUpdate, onRemove, onClose,
}) {
  const { postJSON } = useApi();
  const [readout, setReadout] = useState(null);

  // The backend's reading of the track, debounced, re-read when the track or the sources change.
  const sig = JSON.stringify(recipe.keyframes) + '|' + components.map((c) => `${c.source_id}:${c.weight}`).join(',') + '|' + recipe.mode;
  useEffect(() => {
    const t = setTimeout(() => {
      postJSON('/api/identity/keyframes/evaluate', { recipe: { ...recipe, enabled: true }, samples: 64 }, { abortOnUnmount: true })
        .then(setReadout)
        .catch((err) => { setReadout(null); logFailure('Reading the identity keyframe track', { toast: false })(err); });
    }, 300);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig]);

  const entry = readout?.keyframes?.find((k) => k.frame_index === keyframe.frame_index);
  const shares = components.map((c) => Math.max(0, Number(keyframe.weights?.[c.source_id]) || 0));
  const total = shares.reduce((a, b) => a + b, 0);
  const sm = readout?.smoothness;

  return (
    <div className="mt-1.5 space-y-2.5 rounded-xl border border-white/10 bg-black/30 p-2.5" data-testid="identity-keyframe-inspector">
      <div className="flex items-center gap-2">
        <span className="text-mini font-semibold text-white/80">Identity keyframe · frame {uiFrameOf(keyframe).toLocaleString()}</span>
        <span className="flex-1" />
        <button type="button" onClick={() => onRemove(keyframe.frame_index)} data-testid="identity-keyframe-delete"
                className="rounded-md px-1.5 py-0.5 text-note font-semibold text-rose-300 hover:bg-white/[0.06]">Delete</button>
        <button type="button" onClick={onClose} aria-label="Close keyframe editor"
                className="rounded-md px-1.5 py-0.5 text-note text-[var(--text-muted)] hover:bg-white/[0.06]">✕</button>
      </div>

      <div className="flex flex-wrap gap-4">
        <div className="space-y-1.5">
          <div className="text-nano font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
            {nextKeyframe ? `Curve to frame ${uiFrameOf(nextKeyframe).toLocaleString()}` : 'Curve (no next keyframe: unused)'}
          </div>
          <div className="flex gap-1" role="group" aria-label="Interpolation curve">
            {CURVES.map((c) => (
              <button key={c} type="button" aria-pressed={keyframe.curve === c} data-testid={`identity-curve-${c}`}
                      onClick={() => onUpdate(keyframe.frame_index, { curve: c })}
                      className={`rounded-md border px-2 py-0.5 text-note font-semibold transition-colors ${
                        keyframe.curve === c ? 'border-[var(--accent)] bg-[var(--accent)]/15 text-white' : 'border-white/10 text-[var(--text-muted)] hover:border-white/25'}`}>
                {CURVE_LABELS[c]}
              </button>
            ))}
          </div>
          {keyframe.curve === 'cubic_bezier' && (
            <div className="flex items-start gap-2">
              <BezierEditor bezier={keyframe.bezier} onChange={(b) => onUpdate(keyframe.frame_index, { bezier: b })} />
              <div className="space-y-1">
                {PRESETS.map(([name, b]) => (
                  <button key={name} type="button" onClick={() => onUpdate(keyframe.frame_index, { bezier: b })}
                          className="block w-full rounded-md border border-white/10 px-1.5 py-0.5 text-left text-note text-[var(--text-muted)] hover:border-white/25">{name}</button>
                ))}
                <button type="button" onClick={() => onUpdate(keyframe.frame_index, { bezier: [...DEFAULT_BEZIER] })}
                        className="block w-full rounded-md px-1.5 py-0.5 text-left text-note text-[var(--text-muted)] hover:bg-white/[0.06]">Reset</button>
                <div className="font-mono text-nano tabular-nums text-[var(--text-muted)]">
                  ({keyframe.bezier.map((n) => n.toFixed(2)).join(', ')})
                </div>
              </div>
            </div>
          )}
          {keyframe.curve === 'hold' && (
            <p className="m-0 max-w-[220px] text-note text-[var(--text-muted)]">
              Keeps this identity until the next keyframe, then cuts to it - no in-between faces.
            </p>
          )}
        </div>

        <div className="min-w-[200px] flex-1 space-y-1.5">
          <div className="text-nano font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Shares at this keyframe</div>
          {components.map((c, i) => (
            <label key={c.source_id} className="block" data-testid={`identity-keyframe-share-${i}`}>
              <span className="flex items-center justify-between text-note text-white/75">
                <span className="truncate" title={labelOf(c.source_id)}>{labelOf(c.source_id)}</span>
                <span className="font-mono tabular-nums text-[var(--text-muted)]">{total > 0 ? pctText((shares[i] / total) * 100) : '-'}</span>
              </span>
              <input type="range" min={0} max={100} step={1} value={Math.round(shares[i])}
                     aria-label={`Share of ${labelOf(c.source_id)} at frame ${uiFrameOf(keyframe)}`}
                     onChange={(e) => onUpdate(keyframe.frame_index, { weights: { [c.source_id]: Number(e.target.value) } })}
                     className="h-1.5 w-full accent-[var(--accent)]" />
            </label>
          ))}
          {total <= 0 && <p className="m-0 text-note text-amber-300/80">All shares are 0: this keyframe is ignored until one is above 0.</p>}
        </div>
      </div>

      <div className="font-mono text-nano tabular-nums text-[var(--text-muted)]" data-testid="identity-keyframe-readout">
        {readout ? (
          <>
            {entry?.angle_to_next_deg != null ? `${entry.angle_to_next_deg.toFixed(1)}° to the next keyframe` : 'last keyframe'}
            {sm ? ` · largest step ${((sm.max_step * 180) / Math.PI).toFixed(2)}°/frame` : ''}
            {sm && sm.cuts > 0 ? ` · ${sm.cuts} hold cut${sm.cuts === 1 ? '' : 's'}${sm.max_cut ? ` (${((sm.max_cut * 180) / Math.PI).toFixed(1)}°)` : ''}` : ''}
            {readout.skipped ? ` · ${readout.skipped} keyframe${readout.skipped === 1 ? '' : 's'} skipped (source not loaded)` : ''}
          </>
        ) : 'reading the track…'}
      </div>
    </div>
  );
}
