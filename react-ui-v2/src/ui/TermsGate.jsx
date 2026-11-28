import React, { useEffect, useState } from 'react';
import { ShieldCheck, Loader2 } from 'lucide-react';
import { getJSON, postJSON } from '@engine/api';

// First run: NOTICE.md's intended-use terms (GET /api/terms) must be accepted once per install, and
// again whenever their text changes (the version is a hash of it). Until then the engine refuses
// /api/swap with 403 terms_required, so without this screen a fresh install could never render from
// 2.0. The backend is the gate; this is the screen that lets someone pass it.
//
// Engine unreachable: render nothing. The shell's own "can't reach the engine" state covers that.

/** { terms, accept, busy, error } - terms is null until known; acknowledged once accepted. */
export function useTerms(enabled) {
  const [terms, setTerms] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    if (!enabled) return undefined;
    let alive = true;
    getJSON('/api/terms', { timeout: 8000 })
      .then((t) => { if (alive) setTerms(t || { acknowledged: true }); })
      .catch(() => { if (alive) setTerms({ acknowledged: true, unreachable: true }); });
    return () => { alive = false; };
  }, [enabled]);
  const accept = async () => {
    setBusy(true); setError('');
    try {
      await postJSON('/api/terms/acknowledge', { version: terms.version, accept: true });
      setTerms({ ...terms, acknowledged: true });
    } catch (e) {
      // 409: the text changed since it was shown. Load the new text; the user reads it again.
      setError(e?.message || 'Could not record your acceptance.');
      getJSON('/api/terms').then((t) => t && setTerms(t)).catch(() => {});
    } finally { setBusy(false); }
  };
  return { terms, accept, busy, error };
}

export default function TermsGate({ terms, accept, busy, error }) {
  const [ticked, setTicked] = useState(false);
  if (!terms || terms.acknowledged) return null;
  return (
    <div className="fixed inset-0 z-[200] grid place-items-center bg-black/85 p-4 backdrop-blur-sm"
      role="dialog" aria-modal="true" aria-labelledby="terms-title" data-testid="terms-gate">
      <div className="panel w-full max-w-2xl p-6">
        <div className="flex items-center gap-2.5">
          <ShieldCheck size={20} className="text-[var(--accent-2)]" />
          <h2 id="terms-title" className="text-lg font-semibold">Before you start: intended use</h2>
        </div>
        <p className="mt-1 text-[12.5px] text-[var(--fg-dim)]">
          These are the intended-use terms from <code>NOTICE.md</code>. The engine will not render until they are accepted on this install.
        </p>
        <pre className="selectable mt-4 max-h-[42vh] overflow-auto whitespace-pre-wrap rounded-xl border border-[var(--rule)] bg-black/40 p-4 text-[12.5px] leading-relaxed" data-testid="terms-text">{terms.text}</pre>
        <p className="mt-3 text-[12px] text-[var(--fg-dim)]">
          Output is tagged as synthetic media in its metadata by default (Settings, Output), and a visible watermark is available there. A tag is a label, not a signature: it can be stripped.
        </p>
        <label className="mt-4 flex cursor-pointer items-start gap-3 text-[13px]">
          <input type="checkbox" className="mt-1" checked={ticked} onChange={(e) => setTicked(e.target.checked)} data-testid="terms-tick" />
          <span>I have the right to use the material I process here and the informed consent of the people whose likenesses are involved, and I will label output as synthetic where that is required.</span>
        </label>
        {error && <div className="mt-3 text-[12.5px] text-[var(--danger)]" role="alert">{error}</div>}
        <div className="mt-5 flex justify-end">
          <button type="button" className="btn btn-primary" disabled={!ticked || busy} onClick={accept} data-testid="terms-accept">
            {busy && <Loader2 size={14} className="animate-spin" />} {busy ? 'Saving…' : 'I accept'}
          </button>
        </div>
      </div>
    </div>
  );
}
