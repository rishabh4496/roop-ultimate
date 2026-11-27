import React, { useCallback, useEffect, useState } from 'react';
import { AnimatePresence, motion, MotionConfig } from 'framer-motion';
import { Clapperboard, Layers, Library as LibraryIcon, Boxes, FolderOpen, Settings2, TerminalSquare, ExternalLink, Zap, WifiOff, RefreshCw } from 'lucide-react';
import { setFailureToast } from '@engine/failureLog';
import { useStudio } from './engine/studio';
import { useRun, useRunSync } from './engine/run';
import Studio from './studio/Studio';
import { RunControls } from './studio/RunDeck';
import Batch from './screens/Batch';
import Library from './screens/Library';
import Models from './screens/Models';
import Outputs from './screens/Outputs';
import Settings from './screens/Settings';
import HorizonCanvas from './fx/HorizonCanvas';
import TerminalDock from './fx/TerminalDock';
import BootSequence from './fx/BootSequence';
import FxPanel from './fx/FxPanel';
import { Dock, SplitText } from './fx/effects';
import { useFxStore } from './fx/fxStore';
import { variantsFor } from './fx/transitions';
import { termLog } from './fx/termFeed';
import { Toasts, notify } from './ui/controls';

// React UI 2.0 - the shell. Hash routes, the boot sequence, the top bar (engine state + the render's
// Pause/Stop, reachable from every screen), the magnifying dock, the terminal and the transitions.

const SCREENS = [
  { id: 'studio', label: 'Studio', icon: Clapperboard, kbd: 'Alt 1', render: (p) => <Studio {...p} /> },
  { id: 'batch', label: 'Batch', icon: Layers, kbd: 'Alt 2', render: () => <Batch /> },
  { id: 'library', label: 'Library', icon: LibraryIcon, kbd: 'Alt 3', render: () => <Library /> },
  { id: 'models', label: 'Models', icon: Boxes, kbd: 'Alt 4', render: () => <Models /> },
  { id: 'outputs', label: 'Outputs', icon: FolderOpen, kbd: 'Alt 5', render: () => <Outputs /> },
  { id: 'settings', label: 'Settings', icon: Settings2, kbd: 'Alt 6', render: () => <Settings /> },
];
const ids = SCREENS.map((s) => s.id);
const fromHash = () => {
  const id = String(window.location.hash || '').replace(/^#\/?/, '').split(/[/?]/)[0];
  return ids.includes(id) ? id : 'studio';
};

function EngineChip() {
  const online = useRun((r) => r.online);
  const socket = useRun((r) => r.socket);
  const meta = useStudio((s) => s.meta);
  if (!online) return <span className="badge badge-danger"><WifiOff size={12} /> engine unreachable</span>;
  return (
    <span className="chip hidden md:inline-flex" title={socket ? 'Live telemetry socket connected' : 'Polling (socket not connected)'}>
      <span className={socket ? 'live-dot' : 'h-[7px] w-[7px] rounded-full bg-[var(--warn)]'} />
      <Zap size={12} /> {meta?.active_provider || '…'}{meta?.tensorrt_active ? ' · trt' : ''}
    </span>
  );
}

export default function App() {
  const [screen, setScreen] = useState(fromHash);
  const [error, setError] = useState(null);
  const booted = useStudio((s) => s.booted);
  const meta = useStudio((s) => s.meta);
  const settings = useStudio((s) => s.settings);
  const processing = useRun((r) => r.progress.processing);
  const progress = useRun((r) => r.progress);
  const transition = useFxStore((s) => s.transition);
  const palette = useFxStore((s) => s.palette);
  const grain = useFxStore((s) => s.grain);
  const sunray = useFxStore((s) => s.sunray);
  const terminal = useFxStore((s) => s.terminal);
  const toggleFx = useFxStore((s) => s.toggleFx);

  useRunSync();
  useEffect(() => setFailureToast((msg, type) => notify(msg, type === 'warning' ? 'warn' : type)), []);
  useEffect(() => { document.documentElement.dataset.palette = palette; }, [palette]);
  useEffect(() => { document.documentElement.classList.toggle('no-sunray', !sunray); }, [sunray]);

  const boot = useCallback(async () => {
    setError(null);
    try { await useStudio.getState().boot(); } catch (e) { setError(e.message || String(e)); }
  }, []);
  useEffect(() => { boot(); }, [boot]);

  const go = useCallback((id) => {
    if (!ids.includes(id)) return;
    if (window.location.hash !== `#/${id}`) window.location.hash = `#/${id}`;
    setScreen(id);
  }, []);
  useEffect(() => {
    if (!window.location.hash) window.history.replaceState(null, '', `#/${screen}`);
    const onHash = () => setScreen(fromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { termLog(`goto ${screen}`, 'info', 'nav'); }, [screen]);
  useEffect(() => {
    const onKey = (e) => {
      if (!e.altKey || e.ctrlKey || e.metaKey) return;
      const n = Number(e.key);
      if (n >= 1 && n <= SCREENS.length) { e.preventDefault(); go(SCREENS[n - 1].id); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [go]);

  const current = SCREENS.find((s) => s.id === screen) || SCREENS[0];
  const dockItems = [
    ...SCREENS.map((s) => ({ id: s.id, label: s.label, icon: s.icon, kbd: s.kbd })),
    { sep: true, key: 'sep' },
    { id: 'terminal', label: 'Terminal', icon: TerminalSquare, kbd: 'Ctrl `' },
    { id: 'classic', label: 'Editor, History & more in 1.0', icon: ExternalLink },
  ];
  const onDock = (id) => {
    if (id === 'terminal') { if (!useFxStore.getState().terminal) toggleFx('terminal'); toggleFx('terminalOpen'); return; }
    if (id === 'classic') { window.open('/#/extras', '_blank', 'noopener'); return; }
    go(id);
  };

  return (
    <MotionConfig reducedMotion="user">
      <HorizonCanvas busy={!!processing} />
      {grain && <div className="v2-grain" aria-hidden="true" />}
      <BootSequence meta={meta} settings={settings} error={error} />

      <div className="relative z-10 flex h-full flex-col">
        <header className="topbar">
          <a href="#/studio" className="flex items-center gap-2.5" aria-label="Roop Ultimate 2.0, Studio">
            <span className="brand-mark"><Zap size={15} /></span>
            <span className="hidden leading-tight sm:block">
              <span className="block text-[13.5px] font-semibold tracking-tight"><span className="sunray">Roop Ultimate</span> <span className="font-mono text-[11px] text-[var(--accent-2)]">2.0</span></span>
              <span className="block font-mono text-[10px] text-[var(--fg-faint)]">{meta?.git_version || 'engine'}{meta?.installed_commit?.short ? ` · ${String(meta.installed_commit.short).slice(0, 7)}` : ''}</span>
            </span>
          </a>
          <span className="mx-1 hidden h-6 w-px bg-white/10 md:block" aria-hidden="true" />
          <div className="min-w-0">
            <div className="font-mono text-[10.5px] text-[var(--fg-faint)]">~/workstation/{screen}</div>
            <h1 className="truncate text-[15px] font-semibold leading-tight"><SplitText key={screen} text={current.label} /></h1>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <EngineChip />
            <RunControls compact />
            <FxPanel />
            <a className="btn btn-ghost btn-sm hidden lg:inline-flex" href="/" title="Open React UI 1.0">1.0 <ExternalLink size={12} /></a>
          </div>
        </header>

        <main className="relative min-h-0 flex-1 overflow-y-auto overflow-x-hidden px-3 pt-3"
          style={{ paddingBottom: 'calc(var(--term-reserve, 33px) + 86px)' }}>
          {error && !booted ? (
            <div className="mx-auto mt-16 max-w-lg panel p-6 text-center" role="alert">
              <WifiOff size={28} className="mx-auto text-[var(--danger)]" />
              <div className="mt-2 text-lg font-semibold">Can't reach the engine</div>
              <div className="mt-1 text-[13px] text-[var(--fg-dim)] selectable">{error}</div>
              <button type="button" className="btn btn-primary mt-4" onClick={boot}><RefreshCw size={15} /> Retry</button>
            </div>
          ) : !booted ? (
            <div className="grid h-[60vh] place-items-center"><div className="bar indeterminate w-64"><i /></div></div>
          ) : (
            <AnimatePresence mode="wait">
              <motion.div key={screen} variants={variantsFor(transition)} initial="initial" animate="animate" exit="exit"
                data-transition={transition} className={screen === 'studio' ? 'xl:h-full xl:min-h-[640px]' : ''}>
                {current.render({ onOpenLibrary: () => go('library') })}
              </motion.div>
            </AnimatePresence>
          )}
        </main>
      </div>

      <Dock items={dockItems} active={screen} onSelect={onDock} />
      {terminal && <TerminalDock progress={progress} tabs={SCREENS} onNavigate={go} />}
      <Toasts />
    </MotionConfig>
  );
}
