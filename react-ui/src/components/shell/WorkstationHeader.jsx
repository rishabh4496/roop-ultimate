import React, { useRef } from 'react';
import { Icon } from '../../icons';
import { useWorkspaceMetrics } from './useWorkspaceMetrics';
import LayoutSwitcher from './LayoutSwitcher';
import TelemetryPill from './TelemetryPill';
import HistoryPopover from './HistoryPopover';
import { LiveBar } from '../LiveTelemetry';
import { selectProg } from '../../store/telemetryStore';

// ── The top workstation header ──────────────────────────────────────────────
//
// A single 38px row spans both the navigation rail and route surface. Checkpoints,
// hash-route tabs, layouts, queue and live hardware readings stay in view; less
// frequent tools live in the disclosure to keep the row from wrapping.
const BTN = 'flex h-8 items-center gap-1.5 rounded-lg px-2 text-mini font-semibold transition-colors';
const IDLE = 'text-white/65 hover:bg-white/10 hover:text-white';

function WorkstationHeaderImpl({
  current, tab, onNavigate, runChip, engine, onOpenDiagnostics,
  zoom, onBumpZoom, onResetZoom,
  profileName, onOpenProfiles, onOpenSnapshots, onOpenPalette, pipOn, onTogglePip,
  tabs = [], warmTab, snapshots = [], checkpointId, dirty, onSelectCheckpoint, queueCount = 0,
}) {
  const headerRef = useRef(null);
  useWorkspaceMetrics(headerRef);
  return (
    <header ref={headerRef} className="studio-header">
      <h1 className="sr-only">{current?.label || 'Roop Ultimate'}</h1>
      <div className="studio-checkpoint">
        <span role="img" className={`studio-dirty ${dirty ? 'is-dirty' : ''}`} title={dirty ? 'Changes since checkpoint' : 'Checkpoint saved'} aria-label={dirty ? 'Unsaved checkpoint changes' : 'Checkpoint saved'} />
        <select aria-label="Project checkpoint" value={checkpointId || ''} onChange={(e) => onSelectCheckpoint(e.target.value)}>
          <option value="">Working session</option>
          {snapshots.map((s) => <option key={s.id} value={String(s.id)}>{s.name}</option>)}
          <option value="manage">Save / manage checkpoints…</option>
        </select>
      </div>
      <nav className="studio-routes" aria-label="Quick route switcher">
        {tabs.map((t) => { const Ico = t.icon; return <a key={t.id} href={`#/${t.id}`} aria-label={t.label} title={t.label}
          aria-current={t.id === tab ? 'page' : undefined} onPointerEnter={() => warmTab?.(t.id)} onFocus={() => warmTab?.(t.id)}>
          <Ico size={14} /><span>{t.label}</span>
        </a>; })}
      </nav>
      <LayoutSwitcher tab={tab} onNavigate={onNavigate} />
      <div className="studio-queue-pill">{runChip || <button type="button" onClick={() => onNavigate('processing')} aria-label="Global Render Queue">Queue · {queueCount || 'Idle'}</button>}<LiveBar select={selectProg} className="studio-queue-progress" /></div>
      <TelemetryPill engine={engine} onOpenDiagnostics={onOpenDiagnostics} />
      <details className="studio-tools"><summary aria-label="Studio tools" title="Studio tools">•••</summary>
      <div className="studio-tools-popover">
        <HistoryPopover />

        <div role="group" aria-label="Workspace tools" className="flex items-center gap-0.5 rounded-xl border border-white/10 bg-white/[0.03] p-0.5">
          <button
            type="button"
            onClick={onOpenProfiles}
            title="Quality profiles and custom presets"
            aria-label={`Quality profile: ${profileName}`}
            className={`${BTN} ${IDLE}`}
          >
            <Icon.brand size={14} className="text-accent" />
            <span className="max-2xl:hidden">{profileName}</span>
          </button>
          <button type="button" onClick={onOpenSnapshots} title="Workspace session snapshots" aria-label="Session snapshots" className={`${BTN} ${IDLE}`}>
            <Icon.snapshot size={14} />
          </button>
          <button
            type="button"
            onClick={onTogglePip}
            aria-pressed={!!pipOn}
            title="Floating preview: keeps the Face Swap picture in a corner while you work in Face Manager, Batch or Settings"
            aria-label="Floating preview"
            data-testid="pip-toggle"
            className={`${BTN} ${pipOn ? 'bg-[var(--accent)]/20 text-white' : IDLE}`}
          >
            <Icon.pip size={14} />
          </button>
          <button type="button" onClick={onOpenPalette} title="Command palette (Ctrl/⌘ K)" aria-label="Command palette (Ctrl K)" className={`${BTN} ${IDLE}`}>
            <Icon.search size={14} />
            <kbd className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5 font-mono text-nano max-2xl:hidden">Ctrl K</kbd>
          </button>
          <span aria-hidden="true" className="mx-0.5 h-4 w-px bg-white/10" />
          <div role="group" aria-label="UI zoom" title="UI zoom (Ctrl + / − / 0)" className="flex items-center">
            <button type="button" onClick={() => onBumpZoom(-0.05)} title="Zoom out (Ctrl −)" aria-label="Zoom out" className="grid h-7 w-6 place-items-center rounded-lg text-base leading-none text-white/50 transition-colors hover:bg-white/10 hover:text-white">−</button>
            <button type="button" onClick={onResetZoom} title="Reset zoom (Ctrl 0)" aria-label={`Reset zoom, ${Math.round(zoom * 100)}%`} className="min-w-[44px] text-mini font-semibold tabular-nums text-white/60 transition-colors hover:text-white">{Math.round(zoom * 100)}%</button>
            <button type="button" onClick={() => onBumpZoom(0.05)} title="Zoom in (Ctrl +)" aria-label="Zoom in" className="grid h-7 w-6 place-items-center rounded-lg text-base leading-none text-white/50 transition-colors hover:bg-white/10 hover:text-white">+</button>
          </div>
        </div>
      </div>
      </details>
    </header>
  );
}

export default React.memo(WorkstationHeaderImpl);
