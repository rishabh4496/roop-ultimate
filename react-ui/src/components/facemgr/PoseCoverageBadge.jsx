import React from 'react';
import { coverageBadge } from './poseCoverage';

// Three pips, Left / Front / Right. A lit pip means the faceset holds at least one reference on that side; an amber
// outline means that side is missing; all-dashed means the archive carries no pose data (unknown, not "front only").
// Compact on purpose: it sits on a library row beside the face count, so the user sees a missing side without opening
// the faceset. The sentence is on `title` and the accessible name, not in the layout.
export default function PoseCoverageBadge({ entry, className = '' }) {
  const b = coverageBadge(entry);
  const pipClass = (pip) => {
    if (b.state === 'unknown') return 'border-dashed border-white/20 text-white/30';
    if (pip.lit) return 'border-[var(--accent)]/50 bg-[var(--accent)]/20 text-white/90';
    return 'border-amber-400/60 bg-amber-400/10 text-amber-300';
  };
  return (
    <span
      role="img"
      aria-label={b.title}
      title={b.title}
      data-testid="pose-coverage-badge"
      data-state={b.state}
      className={`inline-flex items-center gap-0.5 ${className}`}
    >
      {b.pips.map((pip) => (
        <span
          key={pip.side}
          data-side={pip.side}
          data-lit={pip.lit ? 'true' : 'false'}
          aria-hidden="true"
          className={`inline-flex h-3.5 min-w-[0.9rem] items-center justify-center rounded border px-0.5 text-[9px] font-bold leading-none ${pipClass(pip)}`}
        >
          {pip.short}
        </span>
      ))}
    </span>
  );
}
