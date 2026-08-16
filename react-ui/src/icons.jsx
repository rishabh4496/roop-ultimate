// ── Icon vocabulary ───────────────────────────────────────────────────────
// One place that decides what every recurring concept in the app LOOKS like,
// built on lucide-react (already a dependency, and tree-shaken per icon).
//
// Why this exists rather than importing lucide directly at each call site:
//
//  * Emoji were the previous icon language, and they are not ours to design.
//    Every platform ships its own artwork, so the same glyph is a flat
//    pictogram on one machine and a glossy 3-D cartoon on another; they carry
//    their own colour, which fights a themed UI that has exactly one accent;
//    they sit on a different baseline from the text beside them; and they
//    render at wildly different optical weights, so a row of them never lines
//    up. None of that is fixable from here.
//  * Naming icons by ROLE, not by picture, means a concept can be redrawn once
//    and change everywhere. Call sites ask for `Icon.stop`, not for a square.
//
// Sizing convention: icons inherit `currentColor` and take their size in px so
// they can sit against the named type scale (a 10px label wants a 12px icon,
// not a 16px one). `strokeWidth` is pinned slightly above lucide's default —
// at these small sizes the default hairline dissolves against the app's dark
// glass surfaces.
import React from 'react';
import {
  Drama, Users, WandSparkles, FolderOpen, History, Settings2, Layers,
  Zap, Search, Play, Pause, Square, Plus, RefreshCw, Keyboard, Palette,
  Download, Upload, Trash2, X, Check, TriangleAlert, Sparkles, Feather,
  Columns2, SquareSplitVertical, Inbox, Clock, Gauge, Cpu, Star,
  CircleCheck, CircleX, CircleAlert, Info, Unplug, Bell, Radio, Image,
  Pencil, Film, PanelLeft, PanelRight, PanelBottom, LayoutGrid, FolderOpen as Folder,
  Eye, ExternalLink, ChevronRight,
  House, RotateCcw, Sun, Moon, Monitor,
} from 'lucide-react';

// Small sizes need a heavier stroke to stay legible; large ones need less or
// they read as bold. One function so the ramp is consistent everywhere.
const strokeFor = (size) => (size <= 12 ? 2.4 : size <= 16 ? 2.1 : 1.9);

const make = (Glyph, displayName) => {
  const Wrapped = ({ size = 16, className = '', strokeWidth, ...rest }) => (
    <Glyph
      size={size}
      strokeWidth={strokeWidth ?? strokeFor(size)}
      className={`shrink-0 ${className}`}
      // Icons here are decoration beside a text label. Where an icon is the
      // ONLY content of a control, the control carries the aria-label — see
      // test_ui_accessibility.py, which fails the build if one does not.
      aria-hidden="true"
      focusable="false"
      {...rest}
    />
  );
  Wrapped.displayName = `Icon.${displayName}`;
  return Wrapped;
};

export const Icon = {
  // Navigation — one per tab.
  home: make(House, 'home'),
  faceswap: make(Drama, 'faceswap'),
  batch: make(Layers, 'batch'),
  faces: make(Users, 'faces'),
  editor: make(WandSparkles, 'editor'),
  wand: make(WandSparkles, 'wand'),
  outputs: make(FolderOpen, 'outputs'),
  history: make(History, 'history'),
  settings: make(Settings2, 'settings'),

  // Identity + status.
  brand: make(Zap, 'brand'),
  search: make(Search, 'search'),
  warning: make(TriangleAlert, 'warning'),
  done: make(Check, 'done'),
  close: make(X, 'close'),
  elapsed: make(Clock, 'elapsed'),

  // Transport — the run controls in the header and the palette.
  play: make(Play, 'play'),
  pause: make(Pause, 'pause'),
  stop: make(Square, 'stop'),

  // Actions.
  queue: make(Plus, 'queue'),
  add: make(Plus, 'add'),
  plus: make(Plus, 'plus'),
  star: make(Star, 'star'),
  // Revert a control to its default. Deliberately counter-clockwise, so it
  // never reads as `refresh` (which re-fetches rather than undoes).
  reset: make(RotateCcw, 'reset'),
  refresh: make(RefreshCw, 'refresh'),
  compare: make(Columns2, 'compare'),
  split: make(SquareSplitVertical, 'split'),
  shortcuts: make(Keyboard, 'shortcuts'),
  theme: make(Palette, 'theme'),
  // The three states of the light/dark pairing control.
  light: make(Sun, 'light'),
  dark: make(Moon, 'dark'),
  system: make(Monitor, 'system'),
  download: make(Download, 'download'),
  upload: make(Upload, 'upload'),
  drop: make(Inbox, 'drop'),
  trash: make(Trash2, 'trash'),

  // Render-quality modes shown on the processing dock.
  full: make(Sparkles, 'full'),
  lite: make(Feather, 'lite'),
  meter: make(Gauge, 'meter'),
  cpu: make(Cpu, 'cpu'),
  bell: make(Bell, 'bell'),

  // Toast / dialog severity. These read as a SET — same family, same optical
  // weight — which is the part emoji could not give us, since the error,
  // info and success glyphs were drawn independently of one another.
  success: make(CircleCheck, 'success'),
  error: make(CircleX, 'error'),
  info: make(Info, 'info'),
  question: make(CircleAlert, 'question'),
  disconnected: make(Unplug, 'disconnected'),

  // Live-preview state.
  live: make(Radio, 'live'),
  still: make(Image, 'still'),

  // Item actions + workspace panels.
  rename: make(Pencil, 'rename'),
  reveal: make(Folder, 'reveal'),
  film: make(Film, 'film'),
  preview: make(Eye, 'preview'),
  popout: make(ExternalLink, 'popout'),
  // Disclosure caret. Call sites rotate it 90° when open, so it must point
  // right at rest — that rotation is why this is a chevron and not a caret
  // glyph, which would have needed a second character for the open state.
  expand: make(ChevronRight, 'expand'),
  panelLeft: make(PanelLeft, 'panelLeft'),
  panelRight: make(PanelRight, 'panelRight'),
  panelBottom: make(PanelBottom, 'panelBottom'),
  layout: make(LayoutGrid, 'layout'),
};

// oxlint's react(only-export-components) flags this file, as it already flags
// motion.jsx seven times: `Icon` is a registry OBJECT, not a component, so Vite
// cannot hot-swap it as granularly. That is inherent to a named icon set and
// the trade is worth it — a role-keyed registry is the whole point, since it
// lets a concept be redrawn in one place. Nothing exports a default here; the
// named import is the only entry point.
