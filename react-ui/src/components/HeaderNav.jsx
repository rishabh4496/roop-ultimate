import React, { useCallback, useEffect, useRef, useState } from 'react';
import { motion, spring } from '../motion';
import { Icon } from '../icons';

// ── The header's right-hand side: utility cluster + tab strip ────────────────
//
// This used to be five loose buttons, a zoom stepper and a nine-tab strip, all
// in one `overflow-x-auto` row. Nine labelled tabs plus the tools beside them
// are wider than the header at anything under ~1500px, so tabs were silently
// scrolled out of reach (e2e/allowlist.json recorded which, per width).
//
// The fix is structural rather than a smaller font:
//
//   * Five PRIMARY tabs stay in the strip; every other tab lives in a "More"
//     disclosure, one click away. Which tabs are primary is decided here and
//     nowhere else; the order, labels, ids and chunk loaders still come from
//     App's ALL_TABS, so hash routing and warmTab are exactly what they were.
//   * Below 1280px the tabs drop their text and become icon-only, with a
//     tooltip. Below 1600px the utility cluster is icon-only too, because at
//     1280 the labelled tabs alone already take most of the row.
//   * Profile / HUD / Snapshots / Search / Zoom are one cluster.

// Layout width, not window width: the app zooms with CSS `zoom` on <html>, which
// media queries cannot see. At 125% a 1440px window lays out like 1152px, and a
// `min-width` class would keep the labels on in a header that no longer fits
// them. `innerWidth` is the media-query width, so dividing by the zoom gives the
// width the layout really has (and is identical to it at 100%).
const ICONS_BELOW = 1280;
// Measured, not guessed (e2e sweep with a run chip showing and a tab under More
// current -- the widest the header gets): the labelled cluster costs ~310px on
// top of the labelled tabs, and the run chip another ~186px. 1760 is where that
// still leaves ~40px spare; at 1600 it wrapped the header onto a second row.
const LABELLED_UTILITIES_FROM = 1760;

const densityFor = (innerWidth, zoom = 1) => {
  const w = innerWidth / (zoom || 1);
  if (w < ICONS_BELOW) return 'icons';
  return w < LABELLED_UTILITIES_FROM ? 'tabs' : 'full';
};

function useHeaderDensity(zoom) {
  const [density, setDensity] = useState(() => densityFor(window.innerWidth, zoom));
  useEffect(() => {
    const on = () => setDensity(densityFor(window.innerWidth, zoom));
    on();
    window.addEventListener('resize', on);
    return () => window.removeEventListener('resize', on);
  }, [zoom]);
  return density;
}

// Tabs that stay in the strip. Everything else in the tab list goes under More.
const PRIMARY_TAB_IDS =['home', 'faceswap', 'batch', 'gallery', 'settings'];

// ── Tooltip ──────────────────────────────────────────────────────────────────
// One shared tooltip rather than a CSS one per button: the tab strip is its own
// scroll container, which would clip a tooltip drawn from inside it. This one
// is positioned against the controls wrapper instead. Hover and keyboard focus
// (:focus-visible, so a mouse click does not leave one stuck) show it; leaving,
// blurring, pressing, scrolling or Escape hide it. It is aria-hidden because the
// button already carries the same text as its aria-label.
function useHeaderTip(zoom) {
  const rootRef = useRef(null);
  const [tip, setTip] = useState(null);
  const hide = useCallback(() => setTip(null), []);

  const show = useCallback((el, text) => {
    const root = rootRef.current;
    if (!root || !el) return;
    const z = zoom || 1;
    const r = el.getBoundingClientRect();
    const b = root.getBoundingClientRect();
    // Rects are visual pixels and `left`/`top` are layout pixels, so divide the
    // CSS zoom back out. Keep the tip on screen: the last tab sits against the
    // right edge and a centred tip would be cut off by the page.
    const half = (text.length * 6.6 + 20) / 2;
    const max = window.innerWidth / z;
    const cx = Math.min(Math.max((r.left + r.width / 2) / z, half + 8), max - half - 8);
    setTip({ text, x: cx - b.left / z, y: (r.bottom - b.top) / z });
  }, [zoom]);

  useEffect(() => {
    if (!tip) return undefined;
    window.addEventListener('scroll', hide, true);
    window.addEventListener('resize', hide);
    return () => {
      window.removeEventListener('scroll', hide, true);
      window.removeEventListener('resize', hide);
    };
  }, [tip, hide]);

  // `warm` is the caller's own pointer/focus work (warmTab); it runs whether or
  // not the tooltip is on, so the prefetch is the same in every density.
  const bind = useCallback((text, enabled, warm) => ({
    onPointerEnter: (e) => {
      warm?.();
      if (enabled && e.pointerType !== 'touch') show(e.currentTarget, text);
    },
    onPointerLeave: () => { if (enabled) hide(); },
    onPointerDown: () => { if (enabled) hide(); },
    onFocus: (e) => {
      warm?.();
      if (enabled && e.currentTarget.matches(':focus-visible')) show(e.currentTarget, text);
    },
    onBlur: () => { if (enabled) hide(); },
    // On the button, not on window: test_ui_shortcut_keys.py forbids two
    // components claiming a bare key at window level (FaceSwap owns Escape
    // there), so Escape dismisses the tooltip of the control that has focus. A
    // hover-only tooltip goes when the pointer moves, presses or the page scrolls.
    onKeyDown: (e) => { if (enabled && e.key === 'Escape') hide(); },
  }), [show, hide]);

  const node = tip && (
    <span
      aria-hidden="true"
      className="pointer-events-none absolute z-50 -translate-x-1/2 whitespace-nowrap rounded-md border border-white/10 bg-[#121216] px-2 py-1 text-mini font-semibold text-white shadow-xl"
      style={{ left: tip.x, top: tip.y + 8 }}
    >
      {tip.text}
    </span>
  );

  return { rootRef, bind, node };
}

// ── Tab strip ────────────────────────────────────────────────────────────────
function TabButton({ active, label, icon: TabIcon, iconsOnly, onClick, handlers, trailing, children, ...rest }) {
  return (
    <motion.button
      type="button"
      onClick={onClick}
      {...handlers}
      {...rest}
      aria-current={active ? 'page' : undefined}
      aria-label={iconsOnly ? (rest['aria-label'] || label) : rest['aria-label']}
      whileTap={{ scale: 0.94 }}
      transition={spring.snappy}
      className={`relative ${iconsOnly ? 'px-3 py-2' : 'px-3.5 py-2'} rounded-lg text-note font-semibold tracking-wide whitespace-nowrap flex items-center gap-1.5 transition-colors duration-200 ${
        active ? 'text-white' : 'text-muted hover:text-white/90'
      }`}
    >
      {active && (
        <motion.span
          layoutId="tab-pill"
          className="absolute inset-0 rounded-lg bg-white/[0.08] border border-white/10 shadow-[inset_0_1px_0_rgba(255,255,255,0.08)]"
          transition={spring.snappy}
        />
      )}
      {/* The icon takes the accent only while the tab is active, so the
          selected tab is legible from colour and from the pill behind it, not
          from colour alone. */}
      <span className="relative z-10 flex items-center gap-1.5">
        <TabIcon size={14} className={active ? 'text-accent' : undefined} />
        {!iconsOnly && label}
        {trailing}
      </span>
      {children}
    </motion.button>
  );
}

function TabNav({ tabs, tab, onSelect, warmTab, runActive, iconsOnly, tipBind }) {
  const primary = tabs.filter((t) => PRIMARY_TAB_IDS.includes(t.id));
  const more = tabs.filter((t) => !PRIMARY_TAB_IDS.includes(t.id));
  const current = more.find((t) => t.id === tab);
  const hasRunTab = more.some((t) => t.id === 'processing');

  const scrollRef = useRef(null);
  const wrapRef = useRef(null);
  const triggerRef = useRef(null);
  const panelRef = useRef(null);
  const [open, setOpen] = useState(false);

  // Keep the active tab inside the strip. The strip only scrolls when the row is
  // narrower than its tabs (a phone, a huge font), but there it would otherwise
  // leave the current tab out of sight after a hash change or Back. offsetLeft is
  // relative to the strip (it is the offsetParent), so no zoom maths is involved.
  useEffect(() => {
    const sc = scrollRef.current;
    const el = sc?.querySelector('[aria-current="page"]');
    if (!sc || !el) return;
    const pad = 8;
    const left = el.offsetLeft;
    const right = left + el.offsetWidth;
    if (left < sc.scrollLeft + pad) sc.scrollLeft = Math.max(0, left - pad);
    else if (right > sc.scrollLeft + sc.clientWidth - pad) sc.scrollLeft = right - sc.clientWidth + pad;
  }, [tab, iconsOnly, primary.length]);

  // Back/Forward or a pasted hash changes the tab without going through the menu.
  useEffect(() => { setOpen(false); }, [tab]);

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => { if (!wrapRef.current?.contains(e.target)) setOpen(false); };
    document.addEventListener('pointerdown', onDown);
    return () => document.removeEventListener('pointerdown', onDown);
  }, [open]);

  const items = () => [...(panelRef.current?.querySelectorAll('button') || [])];

  // A keyboard activation (click with detail 0) moves into the list; a mouse
  // click leaves focus where it is so no ring appears under the pointer. The
  // focus happens in an effect, once the list exists: React flushes it before the
  // next input event, so a fast Enter + ArrowDown cannot overtake it (a
  // requestAnimationFrame did exactly that).
  const focusListOnOpen = useRef(false);
  useEffect(() => {
    if (!open || !focusListOnOpen.current) return;
    focusListOnOpen.current = false;
    const list = [...(panelRef.current?.querySelectorAll('button') || [])];
    (list.find((b) => b.getAttribute('aria-current') === 'page') || list[0])?.focus();
  }, [open]);

  const toggle = (e) => {
    focusListOnOpen.current = !open && e.detail === 0;
    setOpen(!open);
  };

  const onKeyDown = (e) => {
    if (!open) return;
    if (e.key === 'Escape') {
      e.stopPropagation();
      setOpen(false);
      triggerRef.current?.focus();
      return;
    }
    if (!panelRef.current?.contains(document.activeElement)) {
      if (e.key === 'ArrowDown') { e.preventDefault(); items()[0]?.focus(); }
      return;
    }
    const list = items();
    const i = list.indexOf(document.activeElement);
    const to = { ArrowDown: (i + 1) % list.length, ArrowUp: (i - 1 + list.length) % list.length, Home: 0, End: list.length - 1 }[e.key];
    if (to !== undefined) { e.preventDefault(); list[to]?.focus(); }
  };

  // When the current tab lives under More, the trigger BECOMES that tab (its
  // icon and name) so the strip still says where you are; the chevron is what
  // keeps it reading as a menu. Otherwise it is a plain "More".
  const runDot = runActive && hasRunTab && tab !== 'processing';
  const moreName = current ? current.label : 'More';
  const moreLabel = `${current ? `${current.label}, more tabs` : 'More tabs'}${runDot ? ', run in progress' : ''}`;

  return (
    <nav
      aria-label="Main"
      className="flex items-center bg-black/25 rounded-xl border border-white/[0.06] w-full md:w-auto min-w-0"
    >
      <div
        ref={scrollRef}
        className="relative flex gap-0.5 p-1 pr-0.5 min-w-0 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
      >
        {primary.map((t) => (
          <TabButton
            key={t.id}
            active={tab === t.id}
            label={t.label}
            icon={t.icon}
            iconsOnly={iconsOnly}
            onClick={() => onSelect(t.id)}
            // Start fetching the panel's chunk the instant the pointer or
            // keyboard focus lands on the tab — by the time the click registers
            // the module is usually already parsed, so the view swap is a pure
            // animation with no loading state in between.
            handlers={tipBind(t.label, iconsOnly, () => warmTab(t.id))}
          />
        ))}
      </div>

      {more.length > 0 && (
        // role="presentation": this div only delegates Escape/arrow keys and
        // focus-out for the trigger + list inside it; it is not itself a control.
        <div ref={wrapRef} role="presentation" className="relative shrink-0 p-1 pl-0.5" onKeyDown={onKeyDown}
             onBlur={(e) => { if (open && e.relatedTarget && !wrapRef.current?.contains(e.relatedTarget)) setOpen(false); }}>
          <TabButton
            ref={triggerRef}
            active={!!current}
            label={moreName}
            icon={current ? current.icon : Icon.more}
            iconsOnly={iconsOnly}
            onClick={toggle}
            handlers={tipBind(current ? `${current.label} (more tabs)` : 'More tabs', iconsOnly && !open)}
            aria-label={moreLabel}
            aria-expanded={open}
            aria-controls="header-more-tabs"
            trailing={<Icon.expand size={11} className={`opacity-70 transition-transform ${open ? '-rotate-90' : 'rotate-90'}`} />}
          >
            {runDot && (
              <span aria-hidden="true" className="absolute top-1 right-1 z-10 h-1.5 w-1.5 rounded-full bg-[var(--accent)] animate-pulse" />
            )}
          </TabButton>
          {open && (
            <ul
              id="header-more-tabs"
              ref={panelRef}
              // .glass-panel's --card-bg is translucent, which is right for a
              // card but not for a list laid over page content: the chips behind
              // it showed through the labels. The same tint over the theme's
              // opaque --bg-base keeps the look and stops the bleed.
              style={{ background: 'linear-gradient(var(--card-bg), var(--card-bg)), var(--bg-base)' }}
              className="glass-panel animate-slide-up absolute right-0 top-full z-50 mt-2 min-w-[11rem] origin-top-right rounded-xl p-1 shadow-xl"
            >
              {more.map((t) => {
                const active = tab === t.id;
                return (
                  <li key={t.id}>
                    <button
                      type="button"
                      onClick={() => { onSelect(t.id); setOpen(false); triggerRef.current?.focus(); }}
                      onPointerEnter={() => warmTab(t.id)}
                      onFocus={() => warmTab(t.id)}
                      aria-current={active ? 'page' : undefined}
                      className={`flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-note font-semibold tracking-wide whitespace-nowrap transition-colors ${
                        active ? 'bg-white/[0.08] text-white' : 'text-muted hover:bg-white/[0.06] hover:text-white'
                      }`}
                    >
                      <t.icon size={14} className={active ? 'text-accent' : undefined} />
                      {t.label}
                      {t.id === 'processing' && runActive && (
                        <span aria-hidden="true" className="ml-auto h-1.5 w-1.5 rounded-full bg-[var(--accent)] animate-pulse" />
                      )}
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}
    </nav>
  );
}

// ── Utility cluster ──────────────────────────────────────────────────────────
// No display utility in here: the Profile button is `hidden lg:flex`, and two
// display classes on one element resolve by stylesheet order, not by intent.
const UTIL_BTN = 'h-7 items-center gap-1.5 rounded-lg transition-colors text-xs font-medium';
const UTIL_IDLE = 'text-white/60 hover:text-white hover:bg-white/10';

function Utilities({
  labelled, tipBind, zoom, onBumpZoom, onResetZoom,
  profileName, onOpenProfiles, showHud, onToggleHud, onOpenSnapshots, onOpenPalette,
}) {
  const pad = labelled ? 'px-2.5' : 'w-7 justify-center';
  const divider = <span aria-hidden="true" className="mx-1 h-4 w-px bg-white/10" />;
  const tip = !labelled;
  return (
    <div
      role="group"
      aria-label="Workspace tools"
      className="hidden md:flex items-center gap-0.5 p-1 rounded-xl bg-white/[0.03] border border-white/10 shrink-0"
    >
      <button
        type="button"
        onClick={onOpenProfiles}
        title={labelled ? 'Open Quality Profiles & Custom Presets Manager' : undefined}
        aria-label={labelled ? undefined : `Quality profile: ${profileName}`}
        {...tipBind(`Quality profile: ${profileName}`, tip)}
        className={`hidden lg:flex ${UTIL_BTN} ${UTIL_IDLE} ${pad}`}
      >
        <Icon.brand size={14} className="text-accent" />
        {labelled && <span>Profile: <strong className="text-white font-bold">{profileName}</strong></span>}
      </button>
      <span className="hidden lg:contents">{divider}</span>

      <button
        type="button"
        onClick={onToggleHud}
        aria-pressed={showHud}
        title={labelled ? 'Toggle Hardware Telemetry HUD' : undefined}
        aria-label={labelled ? undefined : 'Hardware telemetry HUD'}
        {...tipBind('Hardware telemetry HUD', tip)}
        className={`flex ${UTIL_BTN} ${pad} ${showHud ? 'bg-[var(--accent)]/20 text-white' : UTIL_IDLE}`}
      >
        <Icon.cpu size={14} className={showHud ? 'text-accent' : undefined} />
        {labelled && 'HUD'}
      </button>

      <button
        type="button"
        onClick={onOpenSnapshots}
        title={labelled ? 'Manage Workspace Session Snapshots' : undefined}
        aria-label={labelled ? undefined : 'Session snapshots'}
        {...tipBind('Session snapshots', tip)}
        className={`flex ${UTIL_BTN} ${UTIL_IDLE} ${pad}`}
      >
        <Icon.snapshot size={14} />
        {labelled && 'Snapshots'}
      </button>

      <button
        type="button"
        onClick={onOpenPalette}
        title={labelled ? 'Command palette (Ctrl/⌘ + K)' : undefined}
        aria-label={labelled ? undefined : 'Command palette (Ctrl K)'}
        {...tipBind('Command palette (Ctrl K)', tip)}
        className={`flex ${UTIL_BTN} ${UTIL_IDLE} ${pad} ${labelled ? 'gap-2' : ''}`}
      >
        <Icon.search size={14} />
        {labelled && (
          <>
            Search
            <kbd className="text-nano font-mono bg-white/5 px-1.5 py-0.5 rounded border border-white/10">Ctrl K</kbd>
          </>
        )}
      </button>

      {divider}
      <div role="group" aria-label="UI zoom" title="UI zoom (Ctrl + / − / 0)" className="flex items-center">
        <button type="button" onClick={() => onBumpZoom(-0.05)} title="Zoom out (Ctrl −)" aria-label="Zoom out" className="h-6 w-6 grid place-items-center rounded-lg text-white/50 hover:text-white hover:bg-white/10 text-base leading-none transition-colors">−</button>
        <button type="button" onClick={onResetZoom} title="Reset zoom (Ctrl 0)" aria-label={`Reset zoom, ${Math.round(zoom * 100)}%`} className="min-w-[44px] text-mini font-semibold text-white/60 hover:text-white tabular-nums transition-colors">{Math.round(zoom * 100)}%</button>
        <button type="button" onClick={() => onBumpZoom(0.05)} title="Zoom in (Ctrl +)" aria-label="Zoom in" className="h-6 w-6 grid place-items-center rounded-lg text-white/50 hover:text-white hover:bg-white/10 text-base leading-none transition-colors">+</button>
      </div>
    </div>
  );
}

export default function HeaderControls({
  tabs, tab, onSelect, warmTab, runActive,
  zoom, onBumpZoom, onResetZoom,
  profileName, onOpenProfiles, showHud, onToggleHud, onOpenSnapshots, onOpenPalette,
}) {
  const density = useHeaderDensity(zoom);
  const { rootRef, bind, node } = useHeaderTip(zoom);
  return (
    // md:ml-auto: if the row ever runs out of room (the run chip beside the brand
    // is the usual cause) the header wraps rather than clipping, and the controls
    // should then sit at the right of their own row, not hug the left.
    <div ref={rootRef} className="relative flex items-center gap-2 w-full md:w-auto md:ml-auto">
      <Utilities
        labelled={density === 'full'}
        tipBind={bind}
        zoom={zoom}
        onBumpZoom={onBumpZoom}
        onResetZoom={onResetZoom}
        profileName={profileName}
        onOpenProfiles={onOpenProfiles}
        showHud={showHud}
        onToggleHud={onToggleHud}
        onOpenSnapshots={onOpenSnapshots}
        onOpenPalette={onOpenPalette}
      />
      <TabNav
        tabs={tabs}
        tab={tab}
        onSelect={onSelect}
        warmTab={warmTab}
        runActive={runActive}
        iconsOnly={density === 'icons'}
        tipBind={bind}
      />
      {node}
    </div>
  );
}
