import React from 'react';
import { Icon } from '../../icons';

// ── The workstation rail ────────────────────────────────────────────────────
//
// The app's navigation: one vertical rail down the left edge, icon + label per
// route, collapsible to icons only. It replaces the header's tab strip + "More"
// menu (which existed because nine labelled tabs did not fit one header row; a
// rail has the whole window height, so nothing hides behind a disclosure).
//
// What stays the same as the strip it replaced:
//   * the tab list, order, labels, ids and chunk preloaders come from App's
//     ALL_TABS, and every item is still the `#/<id>` hash route;
//   * `warm` runs on pointer/focus, so the panel's chunk is usually parsed by
//     the time the click lands;
//   * Processing is transient (it exists for a run), so it appears in the rail
//     while one is running, jobs are queued, or you are on it — and then
//     carries a live count.
//
// Below 768px the rail is icon-only whatever the stored preference says.

function RailItem({ tab, active, collapsed, badge, runActive, onSelect, warm }) {
  const Ico = tab.icon;
  const live = tab.id === 'processing' && runActive;
  const name = badge > 0
    ? `${tab.label}, ${badge} ${badge === 1 ? 'job' : 'jobs'} running or queued`
    : tab.label;
  return (
    <li>
      <button
        type="button"
        onClick={() => onSelect(tab.id)}
        onPointerEnter={warm}
        onFocus={warm}
        aria-current={active ? 'page' : undefined}
        aria-label={name}
        title={collapsed ? name : undefined}
        className={`group/rail relative flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-compact font-semibold tracking-wide transition-colors max-md:justify-center ${
          active
            ? 'bg-white/[0.08] text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.06)]'
            : 'text-muted hover:bg-white/[0.05] hover:text-white/90'
        } ${collapsed ? 'justify-center' : ''}`}
      >
        {/* The accent bar is the selected state; colour on the icon alone would
            be the only cue otherwise. */}
        <span
          aria-hidden="true"
          className={`absolute left-0 top-1.5 bottom-1.5 w-[3px] rounded-r-full bg-[var(--accent)] transition-opacity ${active ? 'opacity-100' : 'opacity-0'}`}
        />
        <span className="relative grid place-items-center">
          <Ico size={17} className={active ? 'text-accent' : undefined} />
          {live && !badge && (
            <span aria-hidden="true" className="absolute -right-1 -top-1 h-1.5 w-1.5 rounded-full bg-[var(--accent)] animate-pulse" />
          )}
          {badge > 0 && collapsed && (
            <span aria-hidden="true" className="absolute -right-2 -top-2 min-w-[15px] rounded-full bg-[var(--accent)] px-1 text-center text-nano font-bold leading-[15px] text-white">
              {badge > 9 ? '9+' : badge}
            </span>
          )}
        </span>
        <span className={`min-w-0 flex-1 truncate ${collapsed ? 'hidden' : 'max-md:hidden'}`}>{tab.label}</span>
        {badge > 0 && !collapsed && (
          <span aria-hidden="true" className="max-md:hidden min-w-[18px] rounded-full bg-[var(--accent)] px-1.5 text-center text-micro font-bold leading-[18px] text-white">
            {badge > 99 ? '99+' : badge}
          </span>
        )}
      </button>
    </li>
  );
}

export default function WorkstationRail({
  tabs, tab, onSelect, warmTab, collapsed, onToggleCollapse, badges, runActive, build,
}) {
  return (
    <nav
      aria-label="Main"
      className={`sticky top-0 z-40 flex h-screen shrink-0 flex-col self-start border-r border-white/[0.07] bg-black/30 backdrop-blur-xl py-3 transition-[width] duration-200 max-md:w-[60px] ${collapsed ? 'w-[60px]' : 'w-[200px]'}`}
    >
      <div className={`mb-3 flex items-center gap-2.5 px-3.5 ${collapsed ? 'justify-center px-0' : ''} max-md:justify-center max-md:px-0`}>
        <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg border border-[var(--accent)]/30 bg-[var(--accent)]/15 text-accent">
          <Icon.brand size={16} />
        </span>
        {!collapsed && (
          <div className="min-w-0 max-md:hidden">
            <div className="truncate text-compact font-bold leading-tight tracking-tight text-white/95">Roop Ultimate</div>
            {build && (
              <div
                data-small-ok="build stamp: the full commit is in the title"
                title={build.title}
                className="truncate font-mono text-nano tracking-wider text-muted"
              >
                {build.text}
              </div>
            )}
          </div>
        )}
      </div>

      <ul className="flex flex-1 flex-col gap-0.5 overflow-y-auto px-2">
        {tabs.map((t) => (
          <RailItem
            key={t.id}
            tab={t}
            active={tab === t.id}
            collapsed={collapsed}
            badge={badges?.[t.id] || 0}
            runActive={runActive}
            onSelect={onSelect}
            warm={() => warmTab(t.id)}
          />
        ))}
      </ul>

      <div className="px-2 pt-2 max-md:hidden">
        <button
          type="button"
          onClick={onToggleCollapse}
          aria-expanded={!collapsed}
          aria-label={collapsed ? 'Expand navigation' : 'Collapse navigation'}
          title={collapsed ? 'Expand navigation' : 'Collapse navigation'}
          className={`flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-mini font-semibold text-muted transition-colors hover:bg-white/[0.05] hover:text-white/90 ${collapsed ? 'justify-center' : ''}`}
        >
          <Icon.panelLeft size={16} className={collapsed ? 'rotate-180' : undefined} />
          {!collapsed && <span>Collapse</span>}
        </button>
      </div>
    </nav>
  );
}
