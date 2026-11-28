import React, { useCallback, useEffect, useState } from 'react';
import { AnimatePresence, motion, MotionConfig } from 'framer-motion';
import { Clapperboard, Layers, Library as LibraryIcon, Boxes, FolderOpen, Settings2, TerminalSquare, Zap, WifiOff, RefreshCw, ScanFace, SlidersHorizontal, History as HistoryIcon } from 'lucide-react';
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
import Editor from './screens/Editor';
import History from './screens/History';
import FaceManager from './screens/FaceManager';
import HorizonCanvas from './fx/HorizonCanvas';
import TerminalDock from './fx/TerminalDock';
import BootSequence from './fx/BootSequence';
import FxPanel from './fx/FxPanel';
import { Dock, SplitText } from './fx/effects';
import { useFxStore } from './fx/fxStore';
import { variantsFor } from './fx/transitions';
import { termLog } from './fx/termFeed';
import { Toasts, notify } from './ui/controls';
import TermsGate, { useTerms } from './ui/TermsGate';

// React UI 2.0 - the shell. Hash routes, the boot sequence, the top bar (engine state + the render's
// Pause/Stop, reachable from every screen), the magnifying dock, the terminal and the transitions.

const SCREENS = [
  { id: 'studio', label: 'Studio', icon: Clapperboard, kbd: 'Alt 1', render: (p) => <Studio {...p} /> },
  { id: 'batch', label: 'Batch', icon: Layers, kbd: 'Alt 2', render: () => <Batch /> },
  { id: 'library', label: 'Library', icon: LibraryIcon, kbd: 'Alt 3', render: () => <Library /> },
  { id: 'faces', label: 'Face Manager', icon: ScanFace, kbd: 'Alt 4', render: (p) => <FaceManager {...p} /> },
  { id: 'models', label: 'Models', icon: Boxes, kbd: 'Alt 5', render: () => <Models /> },
  { id: 'editor', label: 'Editor', icon: SlidersHorizontal, kbd: 'Alt 6', render: (p) => <Editor {...p} /> },
  { id: 'outputs', label: 'Outputs', icon: FolderOpen, kbd: 'Alt 7', render: () => <Outputs /> },
  { id: 'history', label: 'History', icon: HistoryIcon, kbd: 'Alt 8', render: () => <History /> },
  { id: 'settings', label: 'Settings', icon: Settings2, kbd: 'Alt 9', render: () => <Settings /> },
];
const ids = SCREENS.map((s) => s.id);
const fromHash = () => {
  const id = String(window.location.hash || '').replace(/^#\/?/, '').split(/[/?]/)[0];
  return ids.includes(id) ? id : 'studio';
};

// One screen's render error must not blank the shell (the top bar's Pause/Stop stay reachable).
class ScreenBoundary extends React.Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(error) { return { error }; }
  componentDidCatch(error) { termLog(`screen ${this.props.name} crashed: ${error?.message || error}`, 'error', 'ui'); }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="mx-auto mt-16 max-w-lg panel p-6 text-center" role="alert" data-testid="screen-error">
        <div className="text-lg font-semibold">This screen hit an error</div>
        <div className="mt-1 font-mono text-[12px] text-[var(--fg-dim)] selectable">{String(this.state.error?.message || this.state.error)}</div>
        <button type="button" className="btn btn-primary mt-4" onClick={() => this.setState({ error: null })}><RefreshCw size={15} /> Reload screen</button>
      </div>
    );
  }
}

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

  const termsState = useTerms(booted);
  const gated = !!termsState.terms && !termsState.terms.acknowledged;

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
  ];
  const onDock = (id) => {
    if (id === 'terminal') { if (!useFxStore.getState().terminal) toggleFx('terminal'); toggleFx('terminalOpen'); return; }
    go(id);
  };

  return (
    <MotionConfig reducedMotion="user">
      <HorizonCanvas busy={!!processing} />
      {grain && <div className="v2-grain" aria-hidden="true" />}
      <BootSequence meta={meta} settings={settings} error={error} />

      <TermsGate {...termsState} />
      <div className="relative z-10 flex h-full flex-col" inert={gated || undefined} aria-hidden={gated || undefined}>
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
                <ScreenBoundary key={screen} name={screen}>{current.render({ onOpenLibrary: () => go('library'), meta })}</ScreenBoundary>
              </motion.div>
            </AnimatePresence>
          )}
        </main>
      </div>

      {!gated && <Dock items={dockItems} active={screen} onSelect={onDock} />}
      {terminal && !gated && <TerminalDock progress={progress} tabs={SCREENS} onNavigate={go} />}
      <Toasts />
    </MotionConfig>
  );
}
