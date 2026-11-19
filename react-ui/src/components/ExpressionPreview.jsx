import React, { useMemo } from 'react';
import { previewExpression } from '../numeric/mathExpr';

// The pill under a numeric field that is being typed into (Stage 15): `512/2 → 256`, `*2 → 80`, `min(v*2, 64) → 64 (max)`.
//
// It shows what Enter WOULD commit before Enter is pressed. The number comes from previewExpression, which calls the same
// settleTypedValue the field commits with, so the pill cannot say 256 and then store 255. A plain number gets no pill (it
// would only repeat what is in the field); a mistake gets an amber one that says why, instead of nothing happening on Enter.
//
// It is an aria-live status, so a screen-reader user hears the result as they type, and it is pointer-transparent so it never
// covers the control it describes. Positioned by `.expr-pill` (studio-controls.css) above the field's right edge.

export default function ExpressionPreview({ text, current, min, max, step, className = '' }) {
  const preview = useMemo(
    () => previewExpression(text, { current, min, max, step }),
    [text, current, min, max, step],
  );
  if (!preview) return null;
  return (
    <span
      role="status"
      data-testid="expr-preview"
      data-state={preview.ok ? (preview.clamped ? 'clamped' : 'ok') : 'error'}
      className={`expr-pill ${preview.ok ? '' : 'is-error'} ${className}`}
    >
      <span className="expr-pill__formula">{preview.formula}</span>
      <span aria-hidden="true" className="expr-pill__arrow">→</span>
      <span className="sr-only"> evaluates to </span>
      <span className="expr-pill__result">{preview.ok ? preview.text : preview.error}</span>
    </span>
  );
}
