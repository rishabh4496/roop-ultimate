import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ACTIONS, ACTION_BY_ID, CATEGORIES } from '../../shortcuts/actions';
import { comboFromEvent, formatCombo } from '../../shortcuts/combo';
import {
  allConflicts, bindingsOf, checkBinding, exportProfile, isOverridden, parseProfile, profileFileName,
} from '../../shortcuts/keymap';
import { actionsOnKey, analyzeKeymap, describeConflicts } from '../../shortcuts/ansiLayout';
import { useShortcutStore } from '../../store/shortcutStore';
import { Icon } from '../../icons';
import { Section } from '../ui';
import KeyChips from './KeyChips';
import AnsiKeyboard from './AnsiKeyboard';

// Settings -> Keyboard Shortcuts: click an action, press the new combination.
//
//   * Change   replaces all of an action's keys with the one pressed.
//   * +        adds a second key (J/K/L-style alternates, or Ctrl+= beside Ctrl++).
//   * a key already used by another action that could be live at the same time is a CONFLICT: nothing is bound until
//     the user chooses "Replace" (the other action loses that key) or cancels. Two actions on one key would both fire.
//   * keys the browser or a dialog owns (Esc, Tab, Enter, Ctrl+W ...) are refused, with the reason.
//   * Esc cancels a recording. Reset puts one action back; Reset all clears every remap.
//
// While recording, a CAPTURE-phase listener on window swallows the keypress before any handler (the palette, undo,
// Play/Pause) can act on it, so binding Ctrl+Z does not undo something.

export default function KeyboardShortcutsTab() {
  const overrides = useShortcutStore((s) => s.overrides);
  const setBindings = useShortcutStore((s) => s.setBindings);
  const resetBindings = useShortcutStore((s) => s.resetBindings);
  const resetAll = useShortcutStore((s) => s.resetAll);
  const setRecording = useShortcutStore((s) => s.setRecording);

  const [query, setQuery] = useState('');
  const [rec, setRec] = useState(null);                // { id, mode: 'replace' | 'add' }
  const [problem, setProblem] = useState('');          // why the last key was refused, while recording
  const [pending, setPending] = useState(null);        // { id, mode, combo, conflicts } waiting for Replace / Cancel
  const overridesRef = useRef(overrides);
  overridesRef.current = overrides;

  const replaceAll = useShortcutStore((s) => s.replaceAll);
  const [layerId, setLayerId] = useState('plain');
  const [selectedKey, setSelectedKey] = useState(null);       // a key clicked on the drawing: the list narrows to its actions
  const [profileName, setProfileName] = useState('My keymap');
  const [notice, setNotice] = useState(null);                 // { kind: 'ok' | 'error', text } after an import / export
  const fileRef = useRef(null);

  const analysis = useMemo(() => analyzeKeymap(overrides), [overrides]);
  const conflictLines = useMemo(() => describeConflicts(analysis), [analysis]);
  const onSelectedKey = useMemo(() => (selectedKey ? new Set(actionsOnKey(selectedKey, analysis).map((a) => a.id)) : null), [selectedKey, analysis]);
  const clashing = useMemo(() => new Set(allConflicts(overrides).flatMap((c) => c.actions)), [overrides]);

  // Export: the user's remaps as a file. A Blob URL is local to this page; nothing is uploaded anywhere.
  const exportKeymap = () => {
    const profile = exportProfile(overrides, profileName);
    const url = URL.createObjectURL(new Blob([`${JSON.stringify(profile, null, 2)}\n`], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = profileFileName(profile.name);
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    const n = Object.keys(profile.overrides).length;
    setNotice({ kind: 'ok', text: `Exported “${profile.name}”: ${n === 0 ? 'every shortcut is on its default' : `${n} changed action${n === 1 ? '' : 's'}`}.` });
  };

  // Import REPLACES the keymap (a profile is a whole keymap), then says what clashes - the keyboard has already lit it.
  const importKeymap = async (file) => {
    if (!file) return;
    try {
      const parsed = parseProfile(await file.text());
      if (!parsed.ok) { setNotice({ kind: 'error', text: parsed.error }); return; }
      replaceAll(parsed.overrides);
      setPending(null);
      setSelectedKey(null);
      const n = Object.keys(parsed.overrides).length;
      const clashes = allConflicts(parsed.overrides).length;
      setProfileName(parsed.name);
      setNotice({
        kind: clashes ? 'error' : 'ok',
        text: `Imported “${parsed.name}”: ${n} changed action${n === 1 ? '' : 's'}${parsed.ignored ? `, ${parsed.ignored} this version does not know ignored` : ''}${clashes ? `. ${clashes} key${clashes === 1 ? '' : 's'} now clash — see below.` : '. No conflicts.'}`,
      });
    } catch {
      setNotice({ kind: 'error', text: 'Could not read that file' });
    } finally {
      if (fileRef.current) fileRef.current.value = '';
    }
  };
  const changed = useMemo(() => ACTIONS.filter((a) => isOverridden(a.id, overrides)).length, [overrides]);

  const apply = (id, mode, combo) => {
    const cur = bindingsOf(id, overridesRef.current);
    setBindings(id, mode === 'add' ? [...cur.filter((c) => c !== combo), combo] : [combo]);
  };

  useEffect(() => {
    if (!rec) return undefined;
    setRecording(true);
    setProblem('');
    const stop = () => { setRec(null); setProblem(''); };
    const onKey = (e) => {
      // Nothing behind this listener may see the key.
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation?.();
      if (e.key === 'Escape' && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) { stop(); return; }
      const combo = comboFromEvent(e);
      if (!combo) return;                              // only a modifier so far: keep waiting for the key
      const check = checkBinding(rec.id, combo, overridesRef.current);
      if (check.ok) {
        apply(rec.id, rec.mode, check.combo);
        stop();
      } else if (check.conflicts) {
        setPending({ id: rec.id, mode: rec.mode, combo: check.combo, conflicts: check.conflicts });
        stop();
      } else {
        setProblem(check.reason);
      }
    };
    // The key's release must not click the button that has focus (Space on a button fires on keyup).
    const onUp = (e) => { e.preventDefault(); e.stopPropagation(); };
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('keyup', onUp, true);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      // Leave the keyup swallow in place for one more tick so the release that ended the recording is eaten too.
      setTimeout(() => window.removeEventListener('keyup', onUp, true), 250);
      setRecording(false);
    };
  }, [rec]); // eslint-disable-line react-hooks/exhaustive-deps -- apply/setBindings are stable; overrides are read through a ref

  const replace = () => {
    if (!pending) return;
    // The other actions give the key up (an emptied action becomes "unbound", it does not regrow its default).
    for (const other of pending.conflicts) {
      setBindings(other.id, bindingsOf(other.id, overridesRef.current).filter((c) => c !== pending.combo));
    }
    apply(pending.id, pending.mode, pending.combo);
    setPending(null);
  };

  const q = query.trim().toLowerCase();
  const sections = CATEGORIES.map((cat) => ({
    cat,
    rows: ACTIONS.filter((a) => a.category === cat).filter((a) => {
      if (onSelectedKey && !onSelectedKey.has(a.id)) return false;
      if (!q) return true;
      const keys = bindingsOf(a.id, overrides).map((c) => `${c} ${formatCombo(c)}`).join(' ').toLowerCase();
      return `${a.label} ${cat}`.toLowerCase().includes(q) || keys.includes(q);
    }),
  })).filter((s) => s.rows.length > 0);

  return (
    <div className="space-y-4" data-testid="keyboard-shortcuts-tab">
      <div className="flex flex-wrap items-center gap-3">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search actions or keys…"
          aria-label="Search keyboard shortcuts"
          className="flex-1 min-w-[12rem] px-3 py-2 rounded-xl glass-input text-compact text-white focus:outline-none"
        />
        <span className="text-mini text-muted" data-testid="shortcut-changed-count">{changed === 0 ? 'All defaults' : `${changed} changed`}</span>
        <button
          type="button"
          disabled={changed === 0}
          onClick={() => { resetAll(); setPending(null); }}
          className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-mini font-semibold bg-white/[0.04] border border-white/10 text-white/60 hover:text-white hover:border-white/20 apple-transition disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <Icon.reset size={12} /> Reset all
        </button>
      </div>

      <p className="text-mini text-muted">
        Click <b>Change</b>, then press the new key combination. <b>+</b> adds a second key. Esc cancels.
        Stored in this browser only. Space-drag to pan and Esc to close are fixed.
      </p>

      {/* Stage 15: the keyboard map, live conflicts, and hotkey profiles as files. */}
      <AnsiKeyboard
        overrides={overrides}
        layerId={layerId}
        onLayerChange={setLayerId}
        selected={selectedKey}
        onSelect={setSelectedKey}
      />
      {selectedKey && (
        <div className="flex items-center gap-2 text-mini text-muted" data-testid="selected-key-filter">
          <span>Showing the actions on <kbd className="rounded bg-white/10 px-1.5 font-mono text-white">{selectedKey}</kbd> (any modifier)</span>
          <button type="button" onClick={() => setSelectedKey(null)} className="rounded-md border border-white/10 px-2 py-0.5 text-white/70 hover:text-white">Show all</button>
        </div>
      )}

      {conflictLines.length > 0 && (
        <div role="alert" data-testid="keymap-conflicts" className="rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-xs space-y-1">
          <div className="font-semibold text-red-200">
            {conflictLines.length} key{conflictLines.length === 1 ? '' : 's'} would fire two actions at once
          </div>
          <ul className="m-0 list-none space-y-0.5 p-0">
            {conflictLines.map((c) => (
              <li key={c.combo} data-conflict-combo={c.combo} className="text-white/80">
                <kbd className="font-mono bg-white/10 px-1.5 rounded">{formatCombo(c.combo)}</kbd> — {c.labels.join(' · ')}
              </li>
            ))}
          </ul>
          <div className="text-white/50">Change one of them, or Reset it, and the red goes away.</div>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2" data-testid="keymap-profile">
        <label className="flex items-center gap-1.5 text-mini text-muted">
          Profile
          <input
            type="text"
            value={profileName}
            maxLength={60}
            onChange={(e) => setProfileName(e.target.value)}
            aria-label="Hotkey profile name"
            className="w-40 px-2 py-1 rounded-lg glass-input text-compact text-white focus:outline-none"
          />
        </label>
        <button
          type="button"
          onClick={exportKeymap}
          data-testid="keymap-export"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-mini font-semibold bg-white/[0.04] border border-white/10 text-white/70 hover:text-white hover:border-white/20 apple-transition"
        >
          <Icon.download size={12} /> Export profile
        </button>
        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          data-testid="keymap-import"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-mini font-semibold bg-white/[0.04] border border-white/10 text-white/70 hover:text-white hover:border-white/20 apple-transition"
        >
          <Icon.upload size={12} /> Import profile
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="application/json,.json"
          className="sr-only"
          tabIndex={-1}
          aria-label="Import a hotkey profile file"
          data-testid="keymap-import-file"
          onChange={(e) => importKeymap(e.target.files?.[0])}
        />
        {notice && (
          <span role="status" data-testid="keymap-notice" className={`text-mini ${notice.kind === 'error' ? 'text-amber-300' : 'text-emerald-300'}`}>{notice.text}</span>
        )}
      </div>

      {pending && (
        <div role="alert" data-testid="shortcut-conflict" className="rounded-xl border border-amber-400/40 bg-amber-400/10 px-4 py-3 text-xs flex flex-wrap items-center gap-3">
          <span className="min-w-0">
            <kbd className="font-mono bg-white/10 px-1.5 rounded">{formatCombo(pending.combo)}</kbd> is already used by{' '}
            <b>{pending.conflicts.map((a) => a.label).join(', ')}</b>. Both would fire together.
          </span>
          <span className="flex gap-2 ml-auto">
            <button type="button" onClick={replace} data-testid="shortcut-conflict-replace" className="px-3 py-1 rounded-lg bg-[var(--accent)] text-white font-semibold">
              Use it for “{ACTION_BY_ID[pending.id].label}”
            </button>
            <button type="button" onClick={() => setPending(null)} className="px-3 py-1 rounded-lg border border-white/15 text-white/80">Cancel</button>
          </span>
        </div>
      )}

      {sections.length === 0 ? (
        <p className="py-8 text-center text-xs text-muted">No shortcut matches “{query}”.</p>
      ) : (
        <div className="columns-1 xl:columns-2 gap-4 [column-fill:_balance]">
          {sections.map(({ cat, rows }) => (
            <Section key={cat} title={cat} icon={Icon.shortcuts} className="break-inside-avoid mb-4">
              <ul className="space-y-1">
                {rows.map((a) => {
                  const keys = bindingsOf(a.id, overrides);
                  const recording = rec?.id === a.id;
                  const modified = isOverridden(a.id, overrides);
                  return (
                    <li key={a.id} data-action={a.id} className="flex items-center gap-2 py-1 border-b border-white/5 last:border-0 text-xs">
                      <span className="flex-1 min-w-0 flex items-center gap-1.5">
                        <span className="text-white/75 truncate">{a.label}</span>
                        {modified && <span title="Changed from the default" className="h-1.5 w-1.5 rounded-full shrink-0 bg-[var(--mod-dot,#22d3ee)]" />}
                        {clashing.has(a.id) && <span className="text-nano font-bold uppercase text-amber-300 shrink-0" title="Shares its key with another action that can be live at the same time">clash</span>}
                      </span>
                      {recording ? (
                        <span className="text-micro text-[var(--accent)] font-semibold" data-testid="shortcut-recording" aria-live="polite">
                          {problem || (rec.mode === 'add' ? 'Press the extra key…' : 'Press the new keys…')}
                        </span>
                      ) : (
                        <KeyChips combos={keys} />
                      )}
                      <button
                        type="button"
                        onClick={() => { setPending(null); setRec(recording && rec.mode === 'replace' ? null : { id: a.id, mode: 'replace' }); }}
                        aria-pressed={recording && rec.mode === 'replace'}
                        aria-label={`Change shortcut for ${a.label}`}
                        className="px-2 py-0.5 rounded-md border border-white/10 text-white/70 hover:text-white hover:border-white/25 apple-transition"
                      >
                        {recording && rec.mode === 'replace' ? 'Cancel' : 'Change'}
                      </button>
                      <button
                        type="button"
                        onClick={() => { setPending(null); setRec(recording && rec.mode === 'add' ? null : { id: a.id, mode: 'add' }); }}
                        aria-label={`Add another shortcut for ${a.label}`}
                        title="Add a second key"
                        className="w-6 py-0.5 rounded-md border border-white/10 text-white/60 hover:text-white hover:border-white/25 apple-transition"
                      >
                        +
                      </button>
                      <button
                        type="button"
                        disabled={!modified}
                        onClick={() => { setPending(null); resetBindings(a.id); }}
                        aria-label={`Reset shortcut for ${a.label}`}
                        title="Back to the default"
                        className="w-6 py-0.5 rounded-md border border-white/10 text-white/60 hover:text-white hover:border-white/25 apple-transition disabled:opacity-30 disabled:cursor-not-allowed grid place-items-center"
                      >
                        <Icon.reset size={11} />
                      </button>
                    </li>
                  );
                })}
              </ul>
            </Section>
          ))}
        </div>
      )}
    </div>
  );
}
