import React, { useMemo } from 'react';
import { buildProfile, isModelled, DEFAULT_CROP_PX, DEFAULT_SPAN_FRAC } from './maskEdgeModel';

// A cross-section of the paste matte's edge, drawn from the same numbers the render uses (maskEdgeModel.js, checked sample
// for sample against the production feather by .render-check/mask-edge-check.mjs). It sits beside the feather and erode
// sliders: drag either and the curve moves with it, so "how soft is the seam" is a shape and a number instead of a guess.
//
//   solid cyan   a frontal face - the default
//   amber dashed the FAR cheek of a face turned 60 degrees: pulled inward and softer
//   violet dashed the NEAR cheek of the same face: tighter
//   vertical line  the detected face contour; where the alpha reaches 0 is the erode / dilate radius
//
// Axes are crop pixels (x, outward from the contour) and alpha (y, 1 = the swap, 0 = the original). The crop size and the
// matte span are not known to the browser, so the numbers state what they assume; only the 18% cap reads the span.

const W = 248;
const H = 76;
const PAD = { l: 6, r: 6, t: 6, b: 10 };

const px = (x, lo, hi) => PAD.l + ((x - lo) / (hi - lo)) * (W - PAD.l - PAD.r);
const py = (a) => PAD.t + (1 - a) * (H - PAD.t - PAD.b);
const path = (xs, ys, lo, hi) => xs.map((x, i) => `${i ? 'L' : 'M'}${px(x, lo, hi).toFixed(1)} ${py(ys[i]).toFixed(1)}`).join(' ');

export default function MaskEdgeProfile({ blend, erode = 0, settings }) {
  const modelled = isModelled(settings);
  const prof = useMemo(
    () => (modelled ? buildProfile({ blend, erode }) : null),
    [modelled, blend, erode],
  );

  if (!modelled) {
    return (
      <div data-testid="mask-edge-profile" data-modelled="false" className="rounded-lg border border-white/10 bg-black/20 px-3 py-2 text-note text-muted">
        Legacy Gaussian edge is selected (semantic paste matte off). Its width depends on the matte size in a space this preview
        does not see, so no curve is drawn rather than a wrong one.
      </div>
    );
  }

  const { x, alpha, far, near, lo, hi } = prof;
  const zeroX = px(0, lo, hi);
  const radiusX = px(prof.radiusPx, lo, hi);
  const f = (v, d = 0) => Number(v).toFixed(d);
  const summary = `Mask edge: ${f(prof.widthPx)} px ramp, 10 to 90 percent over ${f(prof.width1090Px)} px, steepest ${f(prof.maxSlopePerPx * 100, 1)} percent per pixel`;
  const assume = `Per a ${DEFAULT_CROP_PX}px pasted crop whose matte spans ${Math.round(DEFAULT_SPAN_FRAC * 100)}% of it. The feather is `
    + `crop × blend / 200, shaped by the production smoothstep over a distance transform (roop/mask_composer.py), `
    + `capped at 18% of the face span. Dashed: the far and near cheek of a 60° turned face.`;

  return (
    <div data-testid="mask-edge-profile" data-modelled="true" data-width-px={f(prof.widthPx, 2)} data-capped={prof.capped ? 'true' : 'false'}
      className="rounded-lg border border-white/10 bg-black/25 px-2.5 py-2" title={assume}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={summary} className="block w-full h-[76px]">
        {/* alpha guides */}
        <line x1={PAD.l} x2={W - PAD.r} y1={py(1)} y2={py(1)} stroke="rgba(255,255,255,0.10)" strokeWidth="1" />
        <line x1={PAD.l} x2={W - PAD.r} y1={py(0)} y2={py(0)} stroke="rgba(255,255,255,0.10)" strokeWidth="1" />
        <line x1={PAD.l} x2={W - PAD.r} y1={py(0.5)} y2={py(0.5)} stroke="rgba(255,255,255,0.14)" strokeWidth="1" strokeDasharray="2 3" />
        {/* the detected contour */}
        <line x1={zeroX} x2={zeroX} y1={PAD.t} y2={H - PAD.b} stroke="rgba(255,255,255,0.35)" strokeWidth="1" />
        {prof.radiusPx !== 0 && (
          <line x1={radiusX} x2={radiusX} y1={PAD.t} y2={H - PAD.b} stroke="var(--accent-amber)" strokeWidth="1" strokeDasharray="3 2" />
        )}
        {/* the turned-face extremes, then the frontal curve on top */}
        <path d={path(x, far, lo, hi)} fill="none" stroke="var(--accent-amber)" strokeWidth="1.1" strokeDasharray="3 2" opacity="0.8" />
        <path d={path(x, near, lo, hi)} fill="none" stroke="var(--accent-violet)" strokeWidth="1.1" strokeDasharray="3 2" opacity="0.85" />
        <path d={`${path(x, alpha, lo, hi)} L${px(hi, lo, hi).toFixed(1)} ${py(0)} L${px(lo, lo, hi).toFixed(1)} ${py(0)} Z`}
          fill="color-mix(in srgb, var(--accent-cyan) 14%, transparent)" stroke="none" />
        <path d={path(x, alpha, lo, hi)} fill="none" stroke="var(--accent-cyan)" strokeWidth="1.8" strokeLinejoin="round" />
        {/* the 50% point */}
        <circle cx={px(prof.midPx, lo, hi)} cy={py(0.5)} r="2.6" fill="var(--accent-cyan)" />
      </svg>
      <div className="mt-1 flex flex-wrap items-center justify-between gap-x-3 gap-y-0.5 text-note text-muted tabular-nums">
        <span data-testid="mask-edge-ramp">ramp {f(prof.widthPx)} px</span>
        <span>10–90% {f(prof.width1090Px)} px</span>
        <span>slope {f(prof.maxSlopePerPx * 100, 1)} %/px</span>
        {prof.radiusPx !== 0 && <span className="text-accent-amber">contour {prof.radiusPx > 0 ? '+' : ''}{prof.radiusPx} px</span>}
      </div>
      {prof.capped && (
        <div className="mt-0.5 text-note text-accent-amber">
          Capped: the slider asks for {f(prof.featherPx)} px but a ramp never exceeds 18% of the face.
        </div>
      )}
    </div>
  );
}
