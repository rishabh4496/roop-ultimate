import React, { useEffect, useState } from 'react';
import { postFiles } from '../../api';
import { Section } from '../ui';
import FileDrop from '../faceswap/FileDrop';
import { Icon } from '../../icons';

// ── Quick launch ────────────────────────────────────────────────────────────
//
// Drop a source face and a target here and go: each lands in the backend's own
// workspace through the same endpoints the Face Swap tab uses
// (/api/source/add, /api/target/add), and once BOTH are loaded the app opens
// Face Swap, which rehydrates its sources and targets from /api/state on mount.
//
// Two explicit slots rather than one drop target that guesses: an image can be
// either a source face or a target still, and guessing wrong would swap the
// wrong way round with nothing to say so.
export default function QuickLaunch({ setTab, notify }) {
  const [source, setSource] = useState(null);
  const [target, setTarget] = useState(null);
  const [busy, setBusy] = useState('');

  const add = async (kind, files) => {
    const list = Array.from(files || []);
    if (!list.length) return;
    setBusy(kind);
    try {
      await postFiles(kind === 'source' ? '/api/source/add' : '/api/target/add', list, undefined, { timeout: 0 });
      const name = list.length > 1 ? `${list[0].name} +${list.length - 1} more` : list[0].name;
      (kind === 'source' ? setSource : setTarget)(name);
    } catch (e) {
      notify(e.message || `Could not load the ${kind}`, 'error');
    } finally {
      setBusy('');
    }
  };

  // Both slots filled: open the workspace. A beat first, so the second chip is seen.
  useEffect(() => {
    if (!source || !target) return undefined;
    notify('Source and target loaded. Opening Face Swap.', 'success');
    const t = setTimeout(() => setTab('faceswap'), 700);
    return () => clearTimeout(t);
    // `notify` and `setTab` are stable App callbacks; re-running on them would re-announce.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source, target]);

  const slot = ({ kind, title, accept, name, hint }) => (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-white/70">{title}</span>
        {name && (
          <span className="flex min-w-0 items-center gap-1 rounded-full border border-emerald-400/30 bg-emerald-400/10 px-2 py-0.5 text-mini font-semibold text-emerald-300">
            <Icon.done size={11} />
            <span className="truncate">{name}</span>
          </span>
        )}
      </div>
      <FileDrop accept={accept} multiple={kind === 'source'} label={name ? 'Replace' : `Add ${kind}`} hint={hint} busy={busy === kind} onFiles={(f) => add(kind, f)} />
    </div>
  );

  return (
    <div data-testid="quick-launch">
    <Section title="Quick launch" icon={Icon.drop}>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {slot({ kind: "source", title: "Source face", accept: "image/*", name: source, hint: "drop one or more face photos" })}
        {slot({ kind: "target", title: "Target media", accept: "image/*,video/*,.webp", name: target, hint: "drop a video or an image" })}
      </div>
    </Section>
    </div>
  );
}
