import React from 'react';

/** Compact, honest 512-D ArcFace vector readout for the identity tools. */
export default function EmbeddingReadout({ embedding, testId = 'arcface-vector' }) {
  const values = Array.isArray(embedding) ? embedding.map(Number).filter(Number.isFinite) : [];
  if (!values.length) return null;
  const maxAbs = Math.max(1e-8, ...values.slice(0, 64).map(Math.abs));
  const norm = Math.sqrt(values.reduce((sum, v) => sum + v * v, 0));
  return (
    <div className="rounded-xl border border-cyan-400/15 bg-cyan-950/15 p-2.5" data-testid={testId}>
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <div className="text-nano font-semibold uppercase tracking-[0.14em] text-cyan-200/80">ArcFace embedding</div>
        <div className="font-mono text-nano tabular-nums text-white/55">{values.length}D · ‖z‖ {norm.toFixed(3)}</div>
      </div>
      <div className="flex h-10 items-center gap-px overflow-hidden" role="img" aria-label={`First ${Math.min(values.length, 64)} ArcFace embedding values`}
        title="The first 64 dimensions of the server-computed normalized blend vector; the complete vector has 512 dimensions.">
        {values.slice(0, 64).map((v, i) => (
          <span key={i} className={`min-w-0 flex-1 rounded-sm ${v >= 0 ? 'bg-cyan-300/75' : 'bg-fuchsia-300/75'}`}
            style={{ height: `${Math.max(2, Math.abs(v) / maxAbs * 46)}%`, opacity: 0.35 + 0.65 * Math.abs(v) / maxAbs }} />
        ))}
      </div>
      <p className="mb-0 mt-1 text-micro text-white/40">Centroid preview after blend, dials and privacy guard. The render resolves pose-aware vectors per face.</p>
    </div>
  );
}
