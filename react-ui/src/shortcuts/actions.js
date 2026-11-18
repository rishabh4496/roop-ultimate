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
//             snapshot slots), 'monitor' (the pop-out reference monitor's own window), 'flow' (the pipeline flow bar:
//             live only while focus is inside it) or 'sweep' (the 3x3 contact sheet: live only while it is open). Two
//             actions may share a key only if their scopes cannot both be live - see scopesOverlap in combo.js.
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
  { id: 'workspace.cinema', label: 'Cinema mode: collapse the side panels (with the preview focused)', category: 'Workspace', scope: 'global', keys: ['Tab'] },
  { id: 'facemgr.exclude', label: 'Face Manager: exclude the selected faces', category: 'Workspace', scope: 'facemgr', keys: ['Delete', 'Backspace'] },
  { id: 'facemgr.selectAll', label: 'Face Manager: select every face', category: 'Workspace', scope: 'facemgr', keys: ['Mod+A'] },
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
  { id: 'transport.shuttleSlowBack', label: 'Shuttle: slower backward (down to 0.25×)', category: 'Transport', scope: 'faceswap', keys: ['Shift+J'] },
  { id: 'transport.shuttleSlowForward', label: 'Shuttle: slower forward (down to 0.25×)', category: 'Transport', scope: 'faceswap', keys: ['Shift+L'] },

  // ── Viewport & Scopes ────────────────────────────────────────────────────
  { id: 'viewport.compare', label: 'Compare before / after', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['Shift+C'] },
  { id: 'viewport.splitView', label: 'Split view', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['S'] },
  { id: 'viewport.compareMode', label: 'Compare: blend / slider', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['W'] },
  { id: 'viewport.compareCycle', label: 'Cycle comparison: Wipe → Crossfade → Diff → Side-by-side', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['C'] },
  { id: 'viewport.compareAxis', label: 'Compare: split axis', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['X'] },
  { id: 'viewport.autoSwipe', label: 'Compare: auto-swipe', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['A'] },
  { id: 'viewport.zoomIn', label: 'Zoom the preview in', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['=', '+'] },
  { id: 'viewport.zoomOut', label: 'Zoom the preview out', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['-', '_'] },
  { id: 'viewport.fit', label: 'Fit the preview', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['F'] },
  { id: 'viewport.zoom100', label: 'Preview at 100%', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['Shift+1'] },
  { id: 'viewport.zoom200', label: 'Preview at 200%', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['Shift+2'] },
  { id: 'viewport.zoom400', label: 'Preview at 400%', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['Shift+4'] },
  { id: 'viewport.peekOriginal', label: 'Peek the original (hold)', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['\\'], hold: true },
  { id: 'viewport.loupe', label: 'Magnifier loupe (3.5×)', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['Z', 'G'] },
  { id: 'viewport.debugOverlay', label: 'Keypoints and pose overlay', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['D'] },
  { id: 'viewport.lockTarget', label: 'Lock to the target face', category: 'Viewport & Scopes', scope: 'faceswap', keys: ['T'] },
  { id: 'monitor.fullscreen', label: 'Reference monitor: full screen', category: 'Viewport & Scopes', scope: 'monitor', keys: ['F', 'F11'] },

  // ── Mask & Roto ──────────────────────────────────────────────────────────
  { id: 'mask.brush', label: 'Mask paint brush (toggle)', category: 'Mask & Roto', scope: 'faceswap', keys: ['B'] },
  { id: 'mask.erase', label: 'Mask erase brush (toggle)', category: 'Mask & Roto', scope: 'faceswap', keys: ['E'] },
  { id: 'mask.brushSmaller', label: 'Brush radius: smaller', category: 'Mask & Roto', scope: 'faceswap', keys: ['['] },
  { id: 'mask.brushLarger', label: 'Brush radius: larger', category: 'Mask & Roto', scope: 'faceswap', keys: [']'] },
  { id: 'mask.pen', label: 'Roto pen (exclusion / inclusion zones)', category: 'Mask & Roto', scope: 'faceswap', keys: ['P'] },
  { id: 'mask.featherDown', label: 'Brush feather: softer', category: 'Mask & Roto', scope: 'faceswap', keys: ['{', 'Shift+['] },
  { id: 'mask.featherUp', label: 'Brush feather: harder', category: 'Mask & Roto', scope: 'faceswap', keys: ['}', 'Shift+]'] },

  // ── Timeline ─────────────────────────────────────────────────────────────
  { id: 'timeline.marker', label: 'Marker at the playhead', category: 'Timeline', scope: 'faceswap', keys: ['M'] },
  { id: 'timeline.setIn', label: 'Set in point', category: 'Timeline', scope: 'faceswap', keys: ['I'] },
  { id: 'timeline.setOut', label: 'Set out point', category: 'Timeline', scope: 'faceswap', keys: ['O'] },
  // Bare [ and ] resize the mask brush (mask.brushSmaller / Larger), so the range trims moved to Alt.
  { id: 'timeline.trimStart', label: 'Set the start frame of the range', category: 'Timeline', scope: 'faceswap', keys: ['Alt+['] },
  { id: 'timeline.trimEnd', label: 'Set the end frame of the range', category: 'Timeline', scope: 'faceswap', keys: ['Alt+]'] },
  { id: 'timeline.resetRange', label: 'Reset the range to the whole clip', category: 'Timeline', scope: 'faceswap', keys: ['R'] },
  { id: 'timeline.frameBack', label: 'Back 1 frame', category: 'Timeline', scope: 'faceswap', keys: [','] },
  { id: 'timeline.frameForward', label: 'Forward 1 frame', category: 'Timeline', scope: 'faceswap', keys: ['.'] },
  { id: 'timeline.frameBack10', label: 'Back 10 frames', category: 'Timeline', scope: 'faceswap', keys: ['<', 'Shift+,'] },
  { id: 'timeline.frameForward10', label: 'Forward 10 frames', category: 'Timeline', scope: 'faceswap', keys: ['>', 'Shift+.'] },
  { id: 'timeline.prevCut', label: 'Previous scene cut', category: 'Timeline', scope: 'faceswap', keys: ['ArrowUp'] },
  { id: 'timeline.nextCut', label: 'Next scene cut', category: 'Timeline', scope: 'faceswap', keys: ['ArrowDown'] },
  // Bare C cycles the comparison modes (viewport.compareCycle), so the curve editor moved to Alt+C.
  { id: 'timeline.curveEditor', label: 'Parameter curve editor (video)', category: 'Timeline', scope: 'faceswap', keys: ['Alt+C'] },

  // ── Pipeline flow bar (Stage 11) ─────────────────────────────────────────
  // Live only while focus is INSIDE the flow bar (hover a node and the bar takes focus when nothing else has it), so 1-4 stay
  // the snapshot slots everywhere else. Scope 'flow' overlaps only 'global', which is why the same digits can be listed twice.
  { id: 'flow.solo1', label: 'Flow bar: solo the Detector buffer', category: 'Viewport & Scopes', scope: 'flow', keys: ['1'] },
  { id: 'flow.solo2', label: 'Flow bar: solo the raw Swapper buffer', category: 'Viewport & Scopes', scope: 'flow', keys: ['2'] },
  { id: 'flow.solo3', label: 'Flow bar: solo Gaze & Mouth', category: 'Viewport & Scopes', scope: 'flow', keys: ['3'] },
  { id: 'flow.solo4', label: 'Flow bar: solo the XSeg 3 matte', category: 'Viewport & Scopes', scope: 'flow', keys: ['4'] },
  { id: 'flow.solo5', label: 'Flow bar: solo the Restorer buffer', category: 'Viewport & Scopes', scope: 'flow', keys: ['5'] },
  { id: 'flow.solo6', label: 'Flow bar: solo Colour / LUT', category: 'Viewport & Scopes', scope: 'flow', keys: ['6'] },
  { id: 'flow.soloClear', label: 'Flow bar: back to the live result', category: 'Viewport & Scopes', scope: 'flow', keys: ['0'] },
  { id: 'flow.solo', label: 'Flow bar: solo / un-solo the focused node', category: 'Viewport & Scopes', scope: 'flow', keys: ['S'] },
  { id: 'flow.diff', label: 'Flow bar: difference pass of the focused node', category: 'Viewport & Scopes', scope: 'flow', keys: ['D'] },
  { id: 'flow.bypass', label: 'Flow bar: bypass the focused node', category: 'Viewport & Scopes', scope: 'flow', keys: ['B'] },

  // ── 3x3 Parameter Sweep contact sheet (Stage 11) ─────────────────────────
  // Live only while the sheet is open (it is a dialog, and every other window handler yields to an open dialog).
  { id: 'sweep.cell', label: 'Contact sheet: apply cell 1-9 to the active recipe', category: 'Viewport & Scopes', scope: 'sweep', keys: ['1', '2', '3', '4', '5', '6', '7', '8', '9'] },
  { id: 'sweep.apply', label: 'Contact sheet: apply the selected cell', category: 'Viewport & Scopes', scope: 'sweep', keys: ['Enter'] },
  { id: 'sweep.slotA', label: 'Contact sheet: assign the selected cell to Slot A', category: 'Viewport & Scopes', scope: 'sweep', keys: ['A'] },
  { id: 'sweep.slotB', label: 'Contact sheet: assign the selected cell to Slot B', category: 'Viewport & Scopes', scope: 'sweep', keys: ['B'] },
  { id: 'sweep.slotC', label: 'Contact sheet: assign the selected cell to Slot C', category: 'Viewport & Scopes', scope: 'sweep', keys: ['C'] },
  { id: 'sweep.slotD', label: 'Contact sheet: assign the selected cell to Slot D', category: 'Viewport & Scopes', scope: 'sweep', keys: ['D'] },
  { id: 'sweep.loupe', label: 'Contact sheet: synchronized 3.5x loupe', category: 'Viewport & Scopes', scope: 'sweep', keys: ['G'] },
  { id: 'sweep.fit', label: 'Contact sheet: fit all nine cells', category: 'Viewport & Scopes', scope: 'sweep', keys: ['F'] },

  // ── Snapshots ────────────────────────────────────────────────────────────
  { id: 'snapshot.slotA', label: 'Snapshot slot A', category: 'Snapshots', scope: 'faceswap', keys: ['1'] },
  { id: 'snapshot.slotB', label: 'Snapshot slot B', category: 'Snapshots', scope: 'faceswap', keys: ['2'] },
  { id: 'snapshot.slotC', label: 'Snapshot slot C', category: 'Snapshots', scope: 'faceswap', keys: ['3'] },
  { id: 'snapshot.slotD', label: 'Snapshot slot D', category: 'Snapshots', scope: 'faceswap', keys: ['4'] },
];

export const ACTION_BY_ID = Object.freeze(Object.fromEntries(ACTIONS.map((a) => [a.id, a])));

/** The action ids for the four snapshot slots, in slot order. */
export const SLOT_ACTION_IDS = ['snapshot.slotA', 'snapshot.slotB', 'snapshot.slotC', 'snapshot.slotD'];
