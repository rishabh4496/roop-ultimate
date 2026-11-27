import React from 'react';
import { formatCombo } from '../../shortcuts/combo';

// The keys an action is on, as <kbd> chips. "unbound" when the user cleared it.
export default function KeyChips({ combos, className = '' }) {
  if (!combos || combos.length === 0) {
    return <span className={`text-micro italic text-white/40 ${className}`}>unbound</span>;
  }
  return (
    <span className={`inline-flex flex-wrap items-center justify-end gap-1 ${className}`}>
      {combos.map((c) => (
        <kbd
          key={c}
          className="font-mono text-micro bg-white/10 px-2 py-0.5 rounded border border-white/10 text-white whitespace-nowrap"
        >
          {formatCombo(c)}
        </kbd>
      ))}
    </span>
  );
}
