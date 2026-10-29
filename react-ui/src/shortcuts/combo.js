// Key combos as strings: "Mod+Shift+Z", "ArrowLeft", "?", "Mod++".
//
// ONE canonical form per physical gesture, so a stored binding and a live KeyboardEvent can be compared with ===.
//
//   Mod      Ctrl OR Cmd. The app has always treated them as one (App.jsx tested `metaKey || ctrlKey`), and a
//            binding that only worked with Ctrl would be a Mac-only bug.
//   Alt      as is.
//   Shift    kept for letters, digits and named keys (Shift+C is not C; Shift+ArrowLeft is not ArrowLeft) and
//            DROPPED for a symbol the keyboard already produced by shifting - `?`, `<`, `{`, `+`. The character IS
//            the identity: on a US layout Shift+/ is `?`, on a German one it is Shift+ß, and every handler in the
//            app has always matched `e.key === '?'`, not a physical key. Keeping Shift there would make "?" and
//            "Shift+?" two different bindings for one keypress.
//
//   Letters  come from e.key, folded to upper case (so Caps Lock changes nothing), falling back to e.code on a
//            layout whose letters are not Latin.
//   Digits   come from e.code (Digit1), because Shift+1 reports e.key '!'. The existing zoom presets
//            (Shift+1/2/4) depend on that.
//   Space    is "Space", never " ".
//
// Order is always Mod, Alt, Shift, then the key. Pure functions only - no DOM, no React - so .render-check runs them.

export const MODS = ['Mod', 'Alt', 'Shift'];

const IGNORED_KEYS = new Set(['Shift', 'Control', 'Alt', 'AltGraph', 'Meta', 'OS', 'CapsLock', 'NumLock', 'ScrollLock', 'Dead', 'Unidentified', 'Process', 'Compose']);

const isLetter = (s) => /^[A-Za-z]$/.test(s);
const isDigit = (s) => /^[0-9]$/.test(s);

/** A single printable character that is neither a letter, a digit nor a space: its Shift is part of the character. */
export const isSymbolKey = (token) => typeof token === 'string' && [...token].length === 1 && !isLetter(token) && !isDigit(token) && token !== ' ';

/** The key part of an event's combo, or null for a bare modifier / dead key. */
export function keyToken(e) {
  if (!e) return null;
  const key = e.key;
  if (typeof key !== 'string' || IGNORED_KEYS.has(key)) return null;
  const code = typeof e.code === 'string' ? e.code : '';
  const digit = /^Digit([0-9])$/.exec(code);
  if (digit) return digit[1];
  if (key === ' ' || key === 'Spacebar') return 'Space';
  if (isLetter(key)) return key.toUpperCase();
  const keyCode = /^Key([A-Z])$/.exec(code);
  if (keyCode && [...key].length === 1 && !isDigit(key)) return keyCode[1];     // non-Latin layout: the letter the key is
  if (isDigit(key)) return key;                                                   // numpad
  if (key === 'Esc') return 'Escape';
  return key;
}

// The character each punctuation key types WITHOUT Shift on the layout the defaults are written for.
const UNSHIFTED = {
  Comma: ',', Period: '.', Slash: '/', Semicolon: ';', Quote: "'", BracketLeft: '[', BracketRight: ']',
  Backslash: '\\', Minus: '-', Equal: '=', Backquote: '`',
};

/** Canonical combo for a KeyboardEvent, or null when the event is only a modifier. */
export function comboFromEvent(e) {
  const token = keyToken(e);
  if (token === null) return null;
  const parts = [];
  if (e.ctrlKey || e.metaKey) parts.push('Mod');
  if (e.altKey) parts.push('Alt');
  // Shift is part of a symbol's identity only when the layout CHANGED the character ('>' from Shift+.). When the key came
  // through as the unshifted character of its own physical key - a synthetic event, a layout where Shift+. is not '>' - the
  // press is still a Shift press, and handlers have always honoured `e.shiftKey` there (Timeline: Shift+. = 10 frames).
  const keptShift = e.shiftKey && (!isSymbolKey(token) || UNSHIFTED[e.code] === token);
  if (keptShift) parts.push('Shift');
  parts.push(token);
  return parts.join('+');
}

const COMBO = /^((?:(?:Mod|Alt|Shift)\+)*)(.+)$/;

/** { mods: ['Mod', ...], key } or null. `Mod++` parses as Mod + the `+` key. */
export function parseCombo(combo) {
  if (typeof combo !== 'string') return null;
  const m = COMBO.exec(combo);
  if (!m) return null;
  const mods = m[1] ? m[1].split('+').filter(Boolean) : [];
  const key = m[2];
  // One character, or a key name (ArrowLeft, F11). "Mod+" or a bare "Shift" is not a key.
  const named = /^[A-Za-z][A-Za-z0-9]*$/.test(key) && !MODS.includes(key);
  if ([...key].length !== 1 && !named) return null;
  // Canonical order, no duplicates.
  const ordered = MODS.filter((x) => mods.includes(x));
  if (ordered.length !== mods.length) return null;
  return { mods: ordered, key };
}

/** The canonical spelling of a combo string, or null if it does not parse. */
export function normalizeCombo(combo) {
  const p = parseCombo(combo);
  if (!p) return null;
  return [...p.mods, p.key].join('+');
}

export const hasMod = (combo) => !!parseCombo(combo)?.mods.includes('Mod');

const isMac = () => typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || '');

const KEY_LABEL = {
  ArrowLeft: '←', ArrowRight: '→', ArrowUp: '↑', ArrowDown: '↓',
  Escape: 'Esc', Backspace: '⌫', Delete: 'Del', Enter: 'Enter', Space: 'Space', PageUp: 'PgUp', PageDown: 'PgDn',
};

/** Human spelling: "Ctrl + Shift + Z" (or the Mac glyphs), "←", "?". */
export function formatCombo(combo, { mac = isMac() } = {}) {
  const p = parseCombo(combo);
  if (!p) return String(combo ?? '');
  const mods = p.mods.map((m) => {
    if (m === 'Mod') return mac ? '⌘' : 'Ctrl';
    if (m === 'Alt') return mac ? '⌥' : 'Alt';
    return mac ? '⇧' : 'Shift';
  });
  return [...mods, KEY_LABEL[p.key] || p.key].join(mac ? '' : ' + ');
}

// Combos the browser or the app's own dialogs own. A binding to one of these could never fire (the browser takes
// it first) or would fight a control that has to keep working.
const RESERVED = new Map([
  ['Escape', 'Esc closes dialogs and cancels drags'],
  ['Tab', 'Tab moves focus'],
  ['Enter', 'Enter activates the focused control'],
  ['Mod+W', 'the browser closes the tab'],
  ['Mod+T', 'the browser opens a tab'],
  ['Mod+N', 'the browser opens a window'],
  ['Mod+R', 'the browser reloads'],
  ['Mod+L', 'the browser focuses the address bar'],
  ['Mod+Q', 'the browser quits'],
  ['F5', 'the browser reloads'],
  ['F12', 'the browser opens its developer tools'],
]);

/** A reason this combo cannot be bound, or null. */
export function reservedReason(combo) {
  return RESERVED.get(combo) || null;
}

/** Two actions can fire from the same keypress only if their scopes can be live at once. 'global' is live everywhere. */
export const scopesOverlap = (a, b) => a === b || a === 'global' || b === 'global';
