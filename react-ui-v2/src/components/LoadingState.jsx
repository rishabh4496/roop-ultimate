import React from 'react';

export function LoadingState({ label = 'Loading' }) {
  return <div className="v2-loading" role="status" aria-live="polite"><span className="v2-spinner" aria-hidden="true" /><span>{label}</span></div>;
}

export function Skeleton({ className = '' }) {
  return <span className={`v2-skeleton ${className}`} aria-hidden="true" />;
}
