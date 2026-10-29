import React from 'react';
import { Icon } from '../../icons';
import LayoutSwitcher from './LayoutSwitcher';
import TelemetryPill from './TelemetryPill';
import HistoryPopover from './HistoryPopover';

// ── The top workstation header ──────────────────────────────────────────────
//
// One slim bar above every route: where you are, the run chip while a job
// runs, the workspace layout switcher, the hardware pill and the workspace
// tools. Navigation moved to the rail (WorkstationRail), which is why the
// header no longer needs the density logic, the "More" menu or its tooltip
// system: it has nine fewer things to fit. It still wraps rather than clips if
// the window is narrow.
const BTN = 'flex h-8 items-center gap-1.5 rounded-lg px-2 text-mini font-semibold transition-colors';
const IDLE = 'text-white/65 hover:bg-white/10 hover:text-white';

export default function WorkstationHeader({
  current, tab, onNavigate, runChip, engine, onOpenDiagnostics,
  zoom, onBumpZoom, onResetZoom,
  profileName, onOpenProfiles, onOpenSnapshots, onOpenPalette, pipOn, onTogglePip,
}) {
  const Ico = current?.icon;
  return (
    <header className="sticky top-0 z-30 flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-white/[0.07] bg-black/35 px-4 py-2 backdrop-blur-xl">
      <div className="flex min-w-0 items-center gap-3">
        <h1 className="flex items-center gap-2 text-lead font-bold tracking-tight text-white/95">
          {Ico && <Ico size={16} className="text-accent" />}
          {current?.label || 'Roop Ultimate'}
        </h1>
        {runChip}
      </div>

      <div className="ml-auto flex flex-wrap items-center gap-2">
        <LayoutSwitcher tab={tab} onNavigate={onNavigate} />
        <HistoryPopover />
        <TelemetryPill engine={engine} onOpenDiagnostics={onOpenDiagnostics} />

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
    </header>
  );
}
