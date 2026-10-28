import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { getJSON, postJSON } from '../../api';
import { Section, Slider, Toggle, Button } from '../ui';
import { VirtualGrid } from '../virtual/VirtualScroll';
import {
  MAX_IDENTITIES, MIN_IDENTITIES, clampToPad, evenWeights, padVertices, pointAtWeights, setWeight, toPercents,
  togglePick, weightsAtPoint,
} from './morphPad';

// Embedding Mixer & Morph Lab.
//
// Pick 2-5 identities (Face Swap source faces, saved .fsz identities, or the Face Manager's own set),
// then blend them: by sliders (auto-normalised to 100%, or free-weighted) or by dragging a cursor on the
// pad - a line for two, a triangle for three (true barycentric weights), a square for four, a pentagon for
// five. Both edit the same weights. The blend is a Slerp / spherical mean on the ArcFace hypersphere (or the
// older normalised sum). The server reports how far the blend sits from each input (cosine), the pairwise
// cosine between the inputs, and warns when it has drifted off from all of them.
//
// Optionally the result can be de-identified: pushed away from reference identities, only as far as needed
// for its cosine to each of them to be at or below a threshold (a synthetic, privacy-safe character).
//
// "Preview" loads the blend as a Face Swap source and opens Face Swap, whose preview (the Stage 7 coalesced,
// single-flight one) renders the active frame with it. "Save" writes a V2 .fsz whose cached identity IS the
// blend, into the faceset library. Blending over TIME (identity keyframes) lives on the Face Swap timeline.

const COLORS = ['#38bdf8', '#f472b6', '#a3e635', '#fb923c', '#c084fc'];
const MODES = [['slerp', 'Slerp'], ['linear', 'Linear']];
const MODE_HINT = {
  slerp: 'A share is the fraction of the great-circle arc between the identities (exact Slerp for two, the spherical mean for 3-5).',
  linear: 'The normalised weighted sum of the vectors: the same path as Slerp for two, but not constant speed along it.',
};
const PRIVACY_DEFAULT = 0.3;
const PAD = 260;

function Pad({ n, weights, onChange, labels, thumbs }) {
  const ref = useRef(null);
  const verts = padVertices(n);
  const cursor = pointAtWeights(weights, n);

  const move = useCallback((e) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect || !rect.width) return;
    const p = clampToPad([(e.clientX - rect.left) / rect.width, (e.clientY - rect.top) / rect.height], n);
    onChange(weightsAtPoint(p, n).map((w) => w * 100));
  }, [n, onChange]);

  const down = (e) => {
    e.preventDefault();
    e.currentTarget.setPointerCapture?.(e.pointerId);
    move(e);
  };
  const dragging = (e) => { if (e.buttons & 1) move(e); };

  const key = (e) => {
    const step = e.shiftKey ? 0.05 : 0.01;
    const d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[e.key];
    if (!d) return;
    e.preventDefault();
    const p = clampToPad([cursor[0] + d[0], cursor[1] + d[1]], n);
    onChange(weightsAtPoint(p, n).map((w) => w * 100));
  };

  return (
    <div
      ref={ref}
      role="slider"
      tabIndex={0}
      aria-label="Identity pad. Arrow keys move the blend; hold Shift for larger steps."
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(cursor[0] * 100)}
      aria-valuetext={labels.map((l, i) => `${l} ${Math.round(weights[i])}%`).join(', ')}
      data-testid="morph-pad"
      onPointerDown={down}
      onPointerMove={dragging}
      onKeyDown={key}
      className="relative touch-none select-none rounded-xl border border-white/10 bg-black/25 outline-none focus-visible:border-[var(--accent)] cursor-crosshair"
      style={{ width: PAD, height: PAD }}
    >
      <svg viewBox="0 0 1 1" width={PAD} height={PAD} className="absolute inset-0 pointer-events-none">
        <polygon points={verts.map(([x, y]) => `${x},${y}`).join(' ')} fill="rgba(255,255,255,0.03)"
                 stroke="rgba(255,255,255,0.25)" strokeWidth="0.004" />
        {/* a spoke from each corner to the cursor, in that identity's colour: shorter = larger share */}
        {n >= 3 && verts.map((v, i) => (
          <line key={i} x1={v[0]} y1={v[1]} x2={cursor[0]} y2={cursor[1]} stroke={COLORS[i]} strokeOpacity="0.5" strokeWidth="0.004" />
        ))}
        <circle cx={cursor[0]} cy={cursor[1]} r="0.026" fill="#fff" stroke="rgba(0,0,0,0.7)" strokeWidth="0.006" data-testid="morph-cursor" />
      </svg>
      {verts.map((v, i) => (
        <div key={i} className="absolute -translate-x-1/2 -translate-y-1/2 pointer-events-none text-center"
             style={{ left: `${v[0] * 100}%`, top: `${v[1] * 100}%` }}>
          {thumbs[i]
            ? <img src={thumbs[i]} alt="" className="h-9 w-9 rounded-full object-cover border-2" style={{ borderColor: COLORS[i] }} />
            : <div className="h-9 w-9 rounded-full border-2 bg-white/10" style={{ borderColor: COLORS[i] }} />}
          <div className="mt-0.5 max-w-[72px] truncate text-nano font-semibold" style={{ color: COLORS[i] }}>{labels[i]}</div>
        </div>
      ))}
    </div>
  );
}

export default function MorphLab({ notify, setTab }) {
  const [candidates, setCandidates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [picked, setPicked] = useState([]);
  const [weights, setWeights] = useState([]);
  const [mode, setMode] = useState('auto');              // 'auto' = totals 100, 'free' = independent
  const [blendMode, setBlendMode] = useState('slerp');   // how the identities combine: 'slerp' | 'linear'
  const [deid, setDeid] = useState({ enabled: false, references: [], max_cosine: PRIVACY_DEFAULT });
  const [summary, setSummary] = useState(null);
  const [name, setName] = useState('Blended identity');
  const [busy, setBusy] = useState('');

  const refreshCandidates = useCallback(async () => {
    setLoading(true);
    try { setCandidates((await getJSON('/api/facemgr/morph/candidates')).candidates || []); }
    catch (e) { notify?.(e.message, 'error'); }
    finally { setLoading(false); }
  }, [notify]);
  useEffect(() => { refreshCandidates(); }, [refreshCandidates]);

  const byId = useMemo(() => new Map(candidates.map((c) => [c.id, c])), [candidates]);
  const chosen = picked.map((id) => byId.get(id)).filter(Boolean);
  const n = chosen.length;
  const labels = chosen.map((c) => c.label);
  const thumbs = chosen.map((c) => c.thumb);
  const percents = toPercents(weights);
  const usable = n >= MIN_IDENTITIES && n <= MAX_IDENTITIES && weights.length === n;

  const pick = (id) => {
    const next = togglePick(picked, id);
    if (next.length === picked.length && !picked.includes(id)) {
      notify?.(`A blend holds at most ${MAX_IDENTITIES} identities`, 'info');
      return;
    }
    setPicked(next);
    setWeights(next.length ? evenWeights(next.length) : []);
    setSummary(null);
  };

  const onSlider = (i, v) => setWeights((w) => setWeight(w, i, v, mode));
  const onMode = (auto) => {
    const m = auto ? 'auto' : 'free';
    setMode(m);
    if (auto) setWeights((w) => { const t = w.reduce((a, b) => a + b, 0) || 1; return w.map((x) => (x / t) * 100); });
  };

  // What the server is asked for every blend / preview / save: the components, how they combine, and the optional
  // de-identification (only sent when it is on and names a reference).
  const body = () => ({
    components: picked.map((p, i) => ({ id: p, weight: weights[i] })),
    mode: blendMode,
    deidentify: deid.enabled && deid.references.length ? deid : { enabled: false, references: [] },
  });
  const deidSig = JSON.stringify(deid);

  // Ask the server how the blend sits, debounced so a drag is one request per pause, not per pixel.
  const reqRef = useRef(0);
  useEffect(() => {
    if (!usable) return undefined;
    const id = ++reqRef.current;
    const t = setTimeout(async () => {
      try {
        const res = await postJSON('/api/facemgr/morph/blend', body());
        if (reqRef.current === id) setSummary(res);
      } catch (e) {
        if (reqRef.current === id) setSummary({ error: e.message });
      }
    }, 220);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [picked, weights, usable, blendMode, deidSig]);

  const toggleReference = (id) => setDeid((d) => ({
    ...d,
    references: d.references.includes(id) ? d.references.filter((r) => r !== id) : [...d.references, id].slice(0, MAX_IDENTITIES),
  }));

  const preview = async () => {
    setBusy('preview');
    try {
      // The server loads the blend as a source and SELECTS it; Face Swap rehydrates its sources and
      // selection from /api/state when it mounts, so opening it is all it takes to see the blend.
      await postJSON('/api/facemgr/morph/preview', body(), { timeout: 0 });
      notify?.('Blend loaded as a source. Showing it on the active frame in Face Swap');
      setTab?.('faceswap');
    } catch (e) { notify?.(e.message, 'error'); }
    finally { setBusy(''); }
  };
  const save = async () => {
    setBusy('save');
    try {
      const res = await postJSON('/api/facemgr/morph/save', { ...body(), name });
      notify?.(`Saved "${res.saved}.fsz" to the faceset library`, 'success');
      refreshCandidates();
    } catch (e) { notify?.(e.message, 'error'); }
    finally { setBusy(''); }
  };

  return (
    <div className="space-y-4" data-testid="morph-lab">
      <Section>
        <h2 className="text-lg font-bold">Embedding Mixer &amp; Morph Lab</h2>
        <p className="text-sm text-white/50">
          Blend {MIN_IDENTITIES} to {MAX_IDENTITIES} identities into a new one. Drag the cursor on the pad, or set each share with a
          slider. The result is a vector on the identity sphere: it keeps what the inputs share and averages what they do not.
        </p>
      </Section>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Section title={`1 · Pick identities (${n}/${MAX_IDENTITIES})`}>
          {loading ? <p className="text-mini text-muted">Loading identities…</p>
            : candidates.length === 0 ? (
              <p className="text-mini text-muted" data-testid="morph-none">
                Nothing to blend yet. Add source faces in Face Swap, save a faceset to the library, or build one here in the Faceset builder tab.
              </p>
            ) : (
              // Windowed: the library can hold hundreds of .fsz banks, and each card carries a thumbnail.
              <VirtualGrid
                items={candidates}
                minCell={104}
                gap={8}
                cellHeight={(w) => Math.round(w) + 52}
                getKey={(c) => c?.id}
                maxCols={6}
                overscan={3}
                className="max-h-[56vh] p-1"
                label="Identities"
                testId="morph-candidates"
                renderCell={(c) => {
                  const idx = picked.indexOf(c.id);
                  const on = idx >= 0;
                  return (
                    <div>
                      <button
                        type="button" onClick={() => pick(c.id)} disabled={!c.available}
                        aria-pressed={on} data-testid={`morph-pick-${c.id}`}
                        title={c.available ? `${c.label}${c.faces ? ` · ${c.faces} faces` : ''}` : `${c.label}: ${c.reason}`}
                        className={`relative w-full rounded-lg border p-1 text-left transition-colors disabled:opacity-40 ${
                          on ? 'border-[var(--accent)] bg-[var(--accent)]/10' : 'border-white/10 hover:border-white/25'}`}
                      >
                        {c.thumb ? <img src={c.thumb} alt="" className="aspect-square w-full rounded object-cover" />
                          : <div className="aspect-square w-full rounded bg-white/5" />}
                        <span className="mt-1 block truncate text-nano text-white/80">{c.label}</span>
                        <span className="block truncate text-nano text-muted">{{ source: 'source face', library: '.fsz', facemgr: 'builder set' }[c.kind]}</span>
                        {on && (
                          <span className="absolute left-1 top-1 grid h-5 w-5 place-items-center rounded-full text-nano font-bold text-black"
                                style={{ background: COLORS[idx] }}>{idx + 1}</span>
                        )}
                      </button>
                    </div>
                  );
                }}
              />
            )}
        </Section>

        <Section title="2 · Blend">
          {!usable ? (
            <p className="text-mini text-muted" data-testid="morph-need-more">
              Pick at least {MIN_IDENTITIES} identities to open the pad.
            </p>
          ) : (
            <div className="flex flex-wrap gap-5">
              <Pad n={n} weights={weights} onChange={setWeights} labels={labels} thumbs={thumbs} />
              <div className="min-w-[220px] flex-1 space-y-3">
                <Toggle label="Auto-normalise to 100%" checked={mode === 'auto'} onChange={onMode}
                        info="On: moving one share rescales the others. Off: each share is independent and the blend uses their ratios." />
                {chosen.map((c, i) => (
                  <div key={c.id} data-testid={`morph-weight-${i}`}>
                    <Slider label={c.label} info={`${percents[i]}%`} min={0} max={100} step={1}
                            value={Math.round(weights[i])} onChange={(v) => onSlider(i, v)} />
                  </div>
                ))}
                <p className="m-0 text-mini text-muted" data-testid="morph-shares">
                  {mode === 'free' ? 'Effective shares: ' : 'Shares: '}
                  {chosen.map((c, i) => `${c.label} ${percents[i]}%`).join(' · ')}
                </p>
                <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Blend mode">
                  <span className="text-mini text-white/60">Mode</span>
                  {MODES.map(([m, label]) => (
                    <button key={m} type="button" aria-pressed={blendMode === m} data-testid={`morph-mode-${m}`}
                      onClick={() => setBlendMode(m)} title={MODE_HINT[m]}
                      className={`rounded-md border px-2 py-0.5 text-mini font-semibold transition-colors ${
                        blendMode === m ? 'border-[var(--accent)] bg-[var(--accent)]/15 text-white' : 'border-white/10 text-white/50 hover:border-white/25'}`}>
                      {label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </Section>

        {usable && (
          <Section title="De-identify (optional)">
            <div className="space-y-2" data-testid="morph-deid">
              <Toggle label="Push the result away from reference identities" checked={deid.enabled}
                      onChange={(v) => setDeid((d) => ({ ...d, enabled: v }))}
                      info="Moves the blend along the great circle toward the point orthogonal to the references, only as far as needed for its cosine to each of them to be at or below the threshold." />
              {deid.enabled && (
                <>
                  <p className="m-0 text-mini text-muted">Pick the identities to move away from (any of the ones listed on the left, including the ones you are blending).</p>
                  <div className="flex flex-wrap gap-1.5" role="group" aria-label="De-identification references">
                    {candidates.filter((c) => c.available).map((c) => {
                      const on = deid.references.includes(c.id);
                      return (
                        <button key={c.id} type="button" aria-pressed={on} title={`${c.label} as a reference`}
                                data-testid={`morph-deid-ref-${c.id}`} onClick={() => toggleReference(c.id)}
                                className={`flex items-center gap-1.5 rounded-lg border px-1.5 py-1 text-nano ${
                                  on ? 'border-[var(--accent)] bg-[var(--accent)]/10 text-white' : 'border-white/10 text-white/60 hover:border-white/25'}`}>
                          {c.thumb ? <img src={c.thumb} alt="" className="h-6 w-6 rounded object-cover" /> : <span className="h-6 w-6 rounded bg-white/10" />}
                          <span className="max-w-[88px] truncate">{c.label}</span>
                        </button>
                      );
                    })}
                  </div>
                  <Slider label="Privacy threshold (max cosine to the references)" min={0} max={0.95} step={0.01}
                          value={deid.max_cosine} info={deid.max_cosine.toFixed(2)}
                          onChange={(v) => setDeid((d) => ({ ...d, max_cosine: v }))} />
                  {deid.references.length === 0 && <p className="m-0 text-mini text-muted">Pick at least one reference to apply it.</p>}
                </>
              )}
            </div>
          </Section>
        )}
      </div>

      {usable && (
        <Section title="3 · Result">
          <div className="space-y-3">
            {summary?.error ? <p className="text-sm text-red-400 m-0" role="alert">{summary.error}</p> : summary && (
              <>
                <ul className="m-0 grid list-none gap-1 p-0 sm:grid-cols-2" data-testid="morph-cosines">
                  {summary.components.map((c, i) => (
                    <li key={c.id} className="flex items-center justify-between rounded-md bg-white/[0.03] px-2 py-1 text-mini">
                      <span className="truncate" style={{ color: COLORS[i] }}>{c.label}</span>
                      <span className="font-mono tabular-nums text-white/80" title="cosine between the blend and this identity (1 = identical)">
                        {summary.cosine_to_each[i].toFixed(3)}
                      </span>
                    </li>
                  ))}
                </ul>
                {summary.pairwise_cosine?.length >= 2 && (
                  <div data-testid="morph-pairwise">
                    <div className="mb-1 text-nano font-semibold uppercase tracking-[0.14em] text-muted">Pairwise cosine between the inputs</div>
                    <table className="border-separate border-spacing-0.5 text-center text-nano tabular-nums">
                      <thead>
                        <tr>
                          <th aria-hidden="true" />
                          {summary.components.map((c, j) => (
                            <th key={c.id} scope="col" title={c.label} style={{ color: COLORS[j] }} className="px-1.5 font-semibold">{j + 1}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {summary.pairwise_cosine.map((row, i) => (
                          <tr key={summary.components[i].id}>
                            <th scope="row" title={summary.components[i].label} style={{ color: COLORS[i] }} className="px-1.5 font-semibold">{i + 1}</th>
                            {row.map((c, j) => (
                              <td key={j} title={`${summary.components[i].label} vs ${summary.components[j].label}`}
                                  className={`rounded px-1.5 py-0.5 ${i === j ? 'text-white/25' : 'bg-white/[0.04] text-white/80'}`}>
                                {i === j ? '·' : c.toFixed(2)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p className="m-0 mt-1 text-nano text-muted">1 = the same identity, about 0 = strangers. Numbers follow the pick order.</p>
                  </div>
                )}
                {summary.deidentify && (
                  <p className={`m-0 text-mini ${summary.deidentify.satisfied ? 'text-white/70' : 'text-amber-300'}`} role="status" data-testid="morph-deid-result">
                    De-identified: cosine to the references {summary.deidentify.cosine_before.map((c) => c.toFixed(2)).join(' / ')} → {summary.deidentify.cosine_after.map((c) => c.toFixed(2)).join(' / ')}
                    {' '}(threshold {summary.deidentify.max_cosine.toFixed(2)}
                    {summary.deidentify.applied ? `, ${summary.deidentify.cosine_to_blend.toFixed(2)} cosine to the un-edited blend` : ''}).
                    {summary.deidentify.reason === 'already_below' ? ' Already below the threshold, nothing moved.' : ''}
                    {summary.deidentify.reason === 'degenerate' ? ' The blend lies inside the references\' span, so it cannot be separated from them.' : ''}
                  </p>
                )}
                {summary.warning && <p className="m-0 text-mini text-amber-300" role="status" data-testid="morph-warning">{summary.warning}</p>}
              </>
            )}
            <div className="flex flex-wrap items-end gap-3">
              <Button variant="secondary" onClick={preview} disabled={!!busy || !summary || !!summary.error} data-testid="morph-preview">
                {busy === 'preview' ? 'Loading…' : 'Preview Blended Swap on Active Frame'}
              </Button>
              <label className="block">
                <span className="text-xs font-semibold uppercase tracking-wider text-muted">Name</span>
                <input value={name} onChange={(e) => setName(e.target.value)} maxLength={80} aria-label="Blended identity name"
                       className="mt-1 block w-56 rounded-lg px-3 py-2 glass-input text-white text-sm focus:outline-none" />
              </label>
              <Button variant="primary" onClick={save} disabled={!!busy || !summary || !!summary.error || !name.trim()} data-testid="morph-save">
                {busy === 'save' ? 'Saving…' : 'Save Blended Identity as .fsz'}
              </Button>
            </div>
            <p className="m-0 text-nano text-muted">
              Preview adds the blend to your Face Swap sources (not to the library). Save keeps it in the faceset library, anchored to a photo of the
              identity with the largest share.
            </p>
          </div>
        </Section>
      )}
    </div>
  );
}
