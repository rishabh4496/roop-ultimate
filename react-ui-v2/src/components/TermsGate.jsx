import React, { useEffect, useState } from 'react';
import { getJSON, postJSON } from '../api';

// First-run screen: NOTICE.md's intended-use terms, served by the backend
// (`GET /api/terms`), must be accepted once per install (and again whenever
// the text changes -- the version is a hash of it). The backend refuses
// /api/swap until then, so this screen is the courteous half of a gate, not
// the whole of it. Nothing else is rendered while it is up.
//
// Backend unreachable: shows nothing. The app's own "reconnecting" state
// covers that; stacking a terms screen on top would hide it.
export default function TermsGate({ children }) {
  const [terms, setTerms] = useState(null);     // {text, version, acknowledged}
  const [ticked, setTicked] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const t = await getJSON('/api/terms', { timeout: 8000 });
        if (alive) setTerms(t);
      } catch {
        if (alive) setTerms({ acknowledged: true, unreachable: true });
      }
    })();
    return () => { alive = false; };
  }, []);

  if (!terms || terms.acknowledged) return children;

  const accept = async () => {
    setBusy(true); setError('');
    try {
      await postJSON('/api/terms/acknowledge', { version: terms.version, accept: true });
      setTerms({ ...terms, acknowledged: true });
    } catch (e) {
      setError(e?.message || 'Could not record your acceptance.');
    } finally { setBusy(false); }
  };

  return (
    <div className="fixed inset-0 z-[1000] flex items-center justify-center bg-[#0b0d12] text-[var(--text,#e8e8ee)] p-6"
         role="dialog" aria-modal="true" aria-labelledby="terms-title">
      <div className="w-full max-w-2xl rounded-2xl border border-white/10 bg-[#141821] shadow-2xl p-8 space-y-5">
        <h1 id="terms-title" className="text-xl font-semibold">Before you start: intended use</h1>
        <p className="text-sm text-white/60">
          These terms are the <code>Intended use</code> section of <code>NOTICE.md</code>. Rendering is
          disabled until they are accepted on this install.
        </p>
        <pre className="whitespace-pre-wrap text-sm leading-relaxed bg-black/30 rounded-xl p-4 max-h-[40vh] overflow-auto select-text">
          {terms.text}
        </pre>
        <p className="text-sm text-white/60">
          Output is tagged as synthetic media in its metadata by default (Settings → Output), and a
          visible watermark is available there. A tag is a label, not a signature: it can be stripped.
        </p>
        <label className="flex items-start gap-3 text-sm cursor-pointer">
          <input type="checkbox" className="mt-1" checked={ticked} onChange={(e) => setTicked(e.target.checked)} />
          <span>I have the right to use the material I process here and the informed consent of the people
            whose likenesses are involved, and I will label output as synthetic where that is required.</span>
        </label>
        {error && <div className="text-sm text-red-400">{error}</div>}
        <div className="flex justify-end gap-3">
          <button type="button" disabled={!ticked || busy} onClick={accept}
                  className="px-4 py-2 rounded-lg bg-[var(--accent,#3b82f6)] text-white disabled:opacity-40">
            {busy ? 'Saving…' : 'I accept'}
          </button>
        </div>
      </div>
    </div>
  );
}
