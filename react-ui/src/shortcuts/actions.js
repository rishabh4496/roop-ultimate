// Every rebindable keyboard action in the workstation, with the keys it ships on.
//
// This table is the ONE place a default key is written. The handlers (App, FaceSwap, InteractivePreview,
// Timeline, Extras, history, the reference monitor) ask `eventMatches(e, 'action.id')` instead of testing
// `e.key === 'x'`, so a remap in Settings reaches the code that acts on it. Before this, nine window listeners each
// hard-coded their own keys and a "remapper" could only have rewritten a cheat-sheet.
//
// Fields
//   id        stable; persisted in localStorage next to the user's overrides
//   label     what the cheat-sheet and the remapper call it
//   category  the cheat-sheet section
//   scope     'global' (live on every tab), 'faceswap' (live on the Face Swap tab, and the Editor tab for the
//             snapshot slots) or 'monitor' (the pop-out reference monitor's own window). Two actions may share a key
//             only if their scopes cannot both be live - see scopesOverlap in combo.js.
//   keys      default combos (combo.js spelling). Several keys may do one thing.
//   hold      true: the action lasts while the key is down (it also reads keyup)
//   modInText true: its Mod+ combos still fire while a text field has focus (the palette, help). A BARE key never does -
//             typing "?" into a field must not open a dialog.
//
// KEEP ONE ACTION PER LINE with `id:` first: app/tests/test_ui_shortcut_keys.py reads this file to prove no two
// actions that can be live together share a default key.
//
// A key that types a different character when shifted ('?' is Shift+/, '>' is Shift+.) is listed under BOTH spellings: the
// character, for a keyboard that makes it, and Shift+<key>, for an event that reports the plain key with Shift held (a layout
// where Shift+. is not '>', or a synthetic press). Handlers used to honour `e.shiftKey` there; the aliases keep that.
//
// Not here, on purpose: Escape (closes dialogs, cancels drags - reserved), Space-drag to pan (a pointer gesture
// that needs Space held, not a command), and the keys inside a focused text field.

export const CATEGORIES = ['Workspace', 'Transport', 'Viewport & Scopes', 'Mask & Roto', 'Timeline', 'Snapshots'];

export const ACTIONS = [
  // ── Workspace ────────────────────────────────────────────────────────────
  { id: 'app.palette', label: 'Command palette', category: 'Workspace', scope: 'global', keys: ['Mod+K'], modInText: true },
  { id: 'app.help', label: 'Keyboard shortcuts', category: 'Workspace', scope: 'global', keys: ['?', 'Shift+/', 'Mod+/'], modInText: true },
  { id: 'app.toggleRail', label: 'Collapse / expand the side rail', category: 'Workspace', scope: 'global', keys: ['Mod+B'] },
  { id: 'app.zoomIn', label: 'Zoom the interface in', category: 'Workspace', scope: 'global', keys: ['Mod+=', 'Mod++'] },
  { id: 'app.zoomOut', label: 'Zoom the interface out', category: 'Workspace', scope: 'global', keys: ['Mod+-', 'Mod+_'] },
  { id: 'app.zoomReset', label: 'Reset interface zoom', category: 'Workspace', scope: 'global', keys: ['Mod+0'] },
  { id: 'edit.undo', label: 'Undo (edits, masks, roto)', category: 'Workspace', scope: 'global', keys: ['Mod+Z'] },
  { id: 'edit.redo', label: 'Redo', category: 'Workspace', scope: 'global', keys: ['Mod+Y', 'Mod+Shift+Z'] },
  { id: 'transport.run', label: 'Run the swap / the queue', category: 'Workspace', scope: 'faceswap', keys: ['Mod+Enter'] },
  { id: 'transport.addToQueue', label: 'Add to the batch queue', category: 'Workspace', scope: 'faceswap', keys: ['Q'] },

  // ── Transport ────────────────────────────────────────────────────────────
  { id: 'transport.playPause', label: 'Play / pause', category: 'Transport', scope: 'faceswap', keys: ['Space'] },
  { id: 'transport.stepBack', label: 'Previous frame', category: 'Transport', scope: 'faceswap', keys: ['ArrowLeft'] },
  { id: 'transport.stepForward', label: 'Next frame', category: 'Transport', scope: 'faceswap', keys: ['ArrowRight'] },
  { id: 'transport.stepBack10', label: 'Back 10 frames', category: 'Transport', scope: 'faceswap', keys: ['Shift+ArrowLeft'] },
  { id: 'transport.stepForward10', label: 'Forward 10 frames', category: 'Transport', scope: 'faceswap', keys: ['Shift+ArrowRight'] },
  { id: 'transport.toStart', label: 'Jump to the first frame', category: 'Transport', scope: 'faceswap', keys: ['Home'] },
  { id: 'transport.toEnd', label: 'Jump to the last frame', category: 'Transport', scope: 'faceswap', keys: ['End'] },
  { id: 'transport.shuttleBack', label: 'Shuttle backward', category: 'Transport', scope: 'faceswap', keys: ['J'] },
  { id: 'transport.shuttlePause', label: 'Shuttle: pause', category: 'Transport', scope: 'faceswap', keys: ['K'] },
  { id: 'transport.shuttleForward', label: 'Shuttle forward', category: 'Transport', scope: 'faceswap', keys: ['L'] },

  // ── Viewport & Scopes ────────────────────────────────────────────────────
  { id: 'viewport.compare', label: 'Compare before / after', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['Shift+C'] },
  { id: 'viewport.splitView', label: 'Split view', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['S'] },
  { id: 'viewport.compareMode', label: 'Compare: blend / slider', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['W'] },
  { id: 'viewport.compareAxis', label: 'Compare: split axis', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['X'] },
  { id: 'viewport.autoSwipe', label: 'Compare: auto-swipe', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['A'] },
  { id: 'viewport.zoomIn', label: 'Zoom the preview in', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['=', '+'] },
  { id: 'viewport.zoomOut', label: 'Zoom the preview out', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['-', '_'] },
  { id: 'viewport.fit', label: 'Fit the preview', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['F'] },
  { id: 'viewport.zoom100', label: 'Preview at 100%', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['Shift+1'] },
  { id: 'viewport.zoom200', label: 'Preview at 200%', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['Shift+2'] },
  { id: 'viewport.zoom400', label: 'Preview at 400%', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['Shift+4'] },
  { id: 'viewport.peekOriginal', label: 'Peek the original (hold)', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['\\'], hold: true },
  { id: 'viewport.loupe', label: 'Magnifier lens', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['G'] },
  { id: 'viewport.debugOverlay', label: 'Keypoints and pose overlay', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['D'] },
  { id: 'viewport.lockTarget', label: 'Lock to the target face', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['T'] },
  { id: 'monitor.fullscreen', label: 'Reference monitor: full screen', category: 'Viewport & Scopes', scope: 'monitor', keys: ['F', 'F11'] },

  // ── Mask & Roto ──────────────────────────────────────────────────────────
  { id: 'mask.brush', label: 'Mask brush', category: 'Mask & Roto', scope: 'faceswap', keys: ['B'] },
  { id: 'mask.pen', label: 'Roto pen (exclusion / inclusion zones)', category: 'Mask & Roto', scope: 'faceswap', keys: ['P'] },
  { id: 'mask.featherDown', label: 'Brush feather: softer', category: 'Mask & Roto', scope: 'faceswap', keys: ['{', 'Shift+['] },
  { id: 'mask.featherUp', label: 'Brush feather: harder', category: 'Mask & Roto', scope: 'faceswap', keys: ['}', 'Shift+]'] },

  // ── Timeline ─────────────────────────────────────────────────────────────
  { id: 'timeline.marker', label: 'Marker at the playhead', category: 'Timeline', scope: 'faceswap', keys: ['M'] },
  { id: 'timeline.setIn', label: 'Set in point', category: 'Timeline', scope: 'faceswap', keys: ['I'] },
  { id: 'timeline.setOut', label: 'Set out point', category: 'Timeline', scope: 'faceswap', keys: ['O'] },
  { id: 'timeline.trimStart', label: 'Set the start frame of the range', category: 'Timeline', scope: 'faceswap', keys: ['['] },
  { id: 'timeline.trimEnd', label: 'Set the end frame of the range', category: 'Timeline', scope: 'faceswap', keys: [']'] },
  { id: 'timeline.resetRange', label: 'Reset the range to the whole clip', category: 'Timeline', scope: 'faceswap', keys: ['R'] },
  { id: 'timeline.frameBack', label: 'Back 1 frame', category: 'Timeline', scope: 'faceswap', keys: [','] },
  { id: 'timeline.frameForward', label: 'Forward 1 frame', category: 'Timeline', scope: 'faceswap', keys: ['.'] },
  { id: 'timeline.frameBack10', label: 'Back 10 frames', category: 'Timeline', scope: 'faceswap', keys: ['<', 'Shift+,'] },
  { id: 'timeline.frameForward10', label: 'Forward 10 frames', category: 'Timeline', scope: 'faceswap', keys: ['>', 'Shift+.'] },
  { id: 'timeline.prevCut', label: 'Previous scene cut', category: 'Timeline', scope: 'faceswap', keys: ['ArrowUp'] },
  { id: 'timeline.nextCut', label: 'Next scene cut', category: 'Timeline', scope: 'faceswap', keys: ['ArrowDown'] },
  { id: 'timeline.curveEditor', label: 'Parameter curve editor (video)', category: 'Timeline', scope: 'faceswap', keys: ['C'] },

  // ── Snapshots ────────────────────────────────────────────────────────────
  { id: 'snapshot.slotA', label: 'Snapshot slot A', category: 'Snapshots', scope: 'faceswap', keys: ['1'] },
  { id: 'snapshot.slotB', label: 'Snapshot slot B', category: 'Snapshots', scope: 'faceswap', keys: ['2'] },
  { id: 'snapshot.slotC', label: 'Snapshot slot C', category: 'Snapshots', scope: 'faceswap', keys: ['3'] },
  { id: 'snapshot.slotD', label: 'Snapshot slot D', category: 'Snapshots', scope: 'faceswap', keys: ['4'] },
];

export const ACTION_BY_ID = Object.freeze(Object.fromEntries(ACTIONS.map((a) => [a.id, a])));

/** The action ids for the four snapshot slots, in slot order. */
export const SLOT_ACTION_IDS = ['snapshot.slotA', 'snapshot.slotB', 'snapshot.slotC', 'snapshot.slotD'];
