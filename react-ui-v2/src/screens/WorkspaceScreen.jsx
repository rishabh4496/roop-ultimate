import React from 'react';
import { Card, Notice } from '../components/primitives';
import { LoadingState, Skeleton } from '../components/LoadingState';

export default function WorkspaceScreen() {
  return <div className="v2-screen"><div className="v2-page-heading"><div><span className="v2-eyebrow">Reserved route</span><h2>Workspace foundation</h2><p>Responsive composition primitives are in place for future feature slices.</p></div></div><Notice title="Feature wiring is intentionally deferred">No source, target, model, provider, queue, processing, or output controls are connected in Stage 4A.</Notice><div className="v2-grid v2-grid-two"><Card><span className="v2-eyebrow">Loading state</span><h3>Async surfaces have a home</h3><LoadingState label="Example loading state" /></Card><Card><span className="v2-eyebrow">Skeleton state</span><h3>Content can arrive progressively</h3><Skeleton className="v2-skeleton-wide" /><Skeleton className="v2-skeleton-medium" /><Skeleton className="v2-skeleton-short" /></Card></div></div>;
}
