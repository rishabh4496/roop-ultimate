import React, { useEffect, useMemo, useState } from 'react';
import { Button } from '../ui';
import { checkpointDiff, restorePatch } from '../settingsDiff';

// Compare Checkpoint vs. Current Settings.
//
// A saved run on the left, what is set RIGHT NOW on the right, one row per parameter that differs
// ("CodeFormer fidelity  0.50 → 0.80", "Mask feather  12px → 24px"). Tick the rows to take back from the
// checkpoint and press Restore: only those parameters are written to the current settings, nothing is
// rendered, and the rows you restored disappear because they now match. A parameter the checkpoint never
// recorded cannot be restored (there is no value to restore it TO); it is listed so the difference is not
// hidden, with its box disabled.

export default function CheckpointDiff({ entry, current, onRestore, onClose }) {
  const rows = useMemo(() => checkpointDiff(entry?.settings, current), [entry, current]);
  const [picked, setPicked] = useState(() => new Set());
  const [onlyKnown, setOnlyKnown] = useState(false);

  // A restore removes rows; drop ticks for rows that no longer exist so the count never lies.
  useEffect(() => {
    setPicked((p) => {
      const live = new Set(rows.map((r) => r.k));
      const next = new Set([...p].filter((k) => live.has(k)));
      return next.size === p.size ? p : next;
    });
  }, [rows]);

  const shown = onlyKnown ? rows.filter((r) => r.known) : rows;
  const restorable = shown.filter((r) => r.restorable);
  const toggle = (k) => setPicked((p) => { const n = new Set(p); if (n.has(k)) n.delete(k); else n.add(k); return n; });
  const restore = (keys) => {
    const patch = restorePatch(rows, keys);
    if (!Object.keys(patch).length) return;
    onRestore(patch, rows.filter((r) => r.k in patch));
  };

  const when = entry?.time ? new Date(entry.time * 1000).toLocaleString() : 'unknown time';

  return (
    <section className="rounded-2xl glass-panel border border-white/10 p-4 space-y-3" aria-label="Checkpoint compared with current settings" data-testid="checkpoint-diff">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="m-0 text-base font-bold text-white">Checkpoint vs. current settings</h3>
          <p className="m-0 text-mini text-muted">
            Run of {when}{entry?.settings?.swap_model ? ` · ${entry.settings.swap_model}` : ''} → what is set now.
            {' '}<span data-testid="diff-count">{rows.length === 0 ? 'No differences.' : `${rows.length} parameter${rows.length === 1 ? '' : 's'} differ.`}</span>
          </p>
        </div>
        <Button size="sm" variant="ghost" onClick={onClose} data-testid="diff-close">Close</Button>
      </div>

      {rows.length === 0 ? (
        <p className="m-0 text-sm text-emerald-300" role="status">The current settings match this checkpoint.</p>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <label className="flex items-center gap-1.5 text-mini text-muted">
              <input type="checkbox" checked={onlyKnown} onChange={(e) => setOnlyKnown(e.target.checked)} className="accent-[var(--accent)]" />
              Only the main parameters
            </label>
            <span className="flex-1" />
            <Button size="sm" variant="ghost" onClick={() => setPicked(new Set(restorable.map((r) => r.k)))} data-testid="diff-select-all">Select all</Button>
            <Button size="sm" variant="ghost" onClick={() => setPicked(new Set())}>None</Button>
          </div>

          <div className="overflow-x-auto rounded-lg border border-white/10">
            <table className="w-full border-collapse text-left text-mini">
              <thead>
                <tr className="bg-black/30 text-nano uppercase tracking-wider text-muted">
                  <th className="w-8 p-2"><span className="sr-only">Restore</span></th>
                  <th className="p-2">Parameter</th>
                  <th className="p-2">Checkpoint</th>
                  <th className="p-2">Current</th>
                  <th className="w-px p-2"><span className="sr-only">Restore this parameter</span></th>
                </tr>
              </thead>
              <tbody>
                {shown.map((r) => (
                  <tr key={r.k} className="border-t border-white/5" data-testid={`diff-row-${r.k}`} data-kind={r.kind}>
                    <td className="p-2">
                      <input
                        type="checkbox" checked={picked.has(r.k)} disabled={!r.restorable}
                        onChange={() => toggle(r.k)} aria-label={`Restore ${r.label}`}
                        title={r.restorable ? undefined : 'The checkpoint never recorded this, so there is nothing to restore it to'}
                        className="accent-[var(--accent)] disabled:opacity-30"
                      />
                    </td>
                    <td className="p-2 text-white/85" title={r.k}>{r.label}</td>
                    <td className="p-2 font-mono tabular-nums text-sky-300">{r.fromText}</td>
                    <td className="p-2 font-mono tabular-nums text-pink-300">{r.toText}</td>
                    <td className="p-2 text-right">
                      <button
                        type="button" onClick={() => restore([r.k])} disabled={!r.restorable}
                        className="rounded px-1.5 py-0.5 text-nano font-bold text-accent hover:bg-white/[0.08] disabled:opacity-30 disabled:hover:bg-transparent"
                        aria-label={`Restore ${r.label} to ${r.fromText}`}
                        title={r.restorable ? `Set ${r.label} back to ${r.fromText}` : 'Nothing to restore'}
                      >
                        ↺ Restore
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button size="sm" variant="primary" onClick={() => restore([...picked])} disabled={picked.size === 0} data-testid="diff-restore-selected">
              Restore selected ({picked.size})
            </Button>
            <Button size="sm" variant="secondary" onClick={() => restore(restorable.map((r) => r.k))} disabled={restorable.length === 0} data-testid="diff-restore-all">
              Restore all {restorable.length}
            </Button>
            <span className="text-nano text-muted">Writes the checkpoint's values into your current settings. Nothing is rendered.</span>
          </div>
        </>
      )}
    </section>
  );
}
