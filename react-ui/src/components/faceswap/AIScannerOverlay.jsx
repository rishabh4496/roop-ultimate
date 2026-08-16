import React from 'react';

export default function AIScannerOverlay() {
  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden z-20">
      {/* A single soft scanner sweep signals "working" — calm, no HUD clutter. */}
      <div className="absolute w-full h-px bg-gradient-to-r from-transparent via-[var(--accent)]/70 to-transparent shadow-[0_0_10px_var(--accent-glow)] animate-scanner-sweep" />
    </div>
  );
}
