import React, { useCallback, useEffect, useState } from 'react';
import { getJSON } from '../api';
import { Section } from './ui';
import { Icon } from '../icons';

// ── Environment health & update compatibility ────────────────────────────────
//
// Two questions this app could not answer while idle:
//
//   "is this machine set up correctly right now?"  The structured runtime
//   report already answers it in full — but only from inside the Processing
//   tab, which means only during or just after a render. Before you start one,
//   or after a restart, there was nothing.
//
//   "is there an update, and is it safe?"  `app/update_manager.py` has always
//   known: it gates on a manifest, snapshots the environment, health-checks the
//   result and rolls back on failure. It is reachable only from Pinokio's
//   Update action, so its verdict was invisible until you had already run it.
//
// Both panels below report BACKEND MEASUREMENTS. Nothing here is inferred in
// the browser, nothing is filled in with a plausible default, and a value the
// backend does not have is shown as UNKNOWN rather than as zero.

const show = (v, suffix = '') => {
  if (v === null || v === undefined || v === '' || v === 'UNKNOWN'
      || v === 'NOT AVAILABLE' || v === 'NOT APPLICABLE') return 'UNKNOWN';
  return `${v}${suffix}`;
};

const Row = ({ label, value, tone = '' }) => (
  <div className="flex items-baseline justify-between gap-3 py-1 border-b border-white/[0.04] last:border-0">
    <span className="text-micro text-white/45 shrink-0">{label}</span>
    <span className={`text-micro font-mono text-right truncate ${tone || 'text-white/85'}`} title={String(value)}>
      {value}
    </span>
  </div>
);

// SAFE is the only classification that means "this can be taken".
// REQUIRES REVIEW and UNVERIFIED are NOT failures — they are the honest answer
// when the candidate changes something the manifest cannot vouch for, or when
// the remote could not be reached (which is what offline looks like).
const CLASS_TONE = {
  SAFE: 'text-emerald-400',
  'REQUIRES REVIEW': 'text-amber-300',
  UNVERIFIED: 'text-white/60',
};

export default function EnvironmentHealth({ notify }) {
  const [hw, setHw] = useState(null);
  const [runtime, setRuntime] = useState(null);
  const [tele, setTele] = useState(null);
  const [err, setErr] = useState('');
  const [loading, setLoading] = useState(true);

  const [upd, setUpd] = useState(null);
  const [updBusy, setUpdBusy] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    // allSettled, not all: one probe being unavailable must not blank the
    // others. Each panel says UNKNOWN for its own missing source.
    const [h, r, t] = await Promise.allSettled([
      getJSON('/api/system/hardware', { timeout: 15000 }),
      getJSON('/api/runtime/state', { timeout: 15000 }),
      getJSON('/api/system/telemetry', { timeout: 15000 }),
    ]);
    setHw(h.status === 'fulfilled' ? h.value : null);
    setRuntime(r.status === 'fulfilled' ? r.value : null);
    setTele(t.status === 'fulfilled' ? t.value : null);
    const failed = [h, r, t].filter((x) => x.status === 'rejected');
    setErr(failed.length === 3
      ? (failed[0].reason?.message || 'the backend did not answer')
      : failed.length ? 'some diagnostics are unavailable' : '');
    setLoading(false);
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // Deliberately NOT polled and NOT run on mount: it reaches the network
  // (`git ls-remote`), and a background check nobody asked for is exactly the
  // kind of thing that makes a local app feel like it needs the internet.
  const checkUpdate = async (refreshRemote) => {
    setUpdBusy(true);
    try {
      const res = await getJSON(`/api/update/check${refreshRemote ? '?refresh=true' : ''}`,
        { timeout: 60000 });
      setUpd(res);
      if (res.classification === 'SAFE' && res.available) {
        notify?.('A compatible update is available — apply it from Pinokio’s Update action', 'success');
      }
    } catch (e) {
      setUpd({ classification: 'UNVERIFIED', available: false,
        reasons: [e.message || 'the compatibility check could not be run'] });
    } finally {
      setUpdBusy(false);
    }
  };

  const sec = runtime?.sections || {};
  const vram = sec.HARDWARE?.values?.vram || {};
  const available = hw?.available !== false;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Section
        title="Environment health"
        icon={Icon.meter}
        action={(
          <button
            type="button"
            onClick={refresh}
            disabled={loading}
            className="px-2.5 py-1 rounded-lg text-micro font-semibold bg-white/5 border border-white/10 text-white/70 hover:bg-white/10 hover:text-white transition-colors disabled:opacity-40"
          >
            {loading ? 'Checking…' : 'Re-check'}
          </button>
        )}
      >
        <div className="text-micro text-white/45 mb-2">
          Measured by the backend on this machine, right now. A value it does not have reads UNKNOWN —
          nothing here is guessed in the browser.
        </div>
        {err && (
          <div className="mb-2 px-3 py-2 rounded-xl text-micro bg-amber-500/10 border border-amber-500/30 text-amber-200">
            {err}
          </div>
        )}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
          <div>
            <Row label="GPU" value={show(hw?.gpu_name || hw?.gpu)}
              tone={available ? 'text-emerald-400' : 'text-amber-300'} />
            <Row label="Driver" value={show(hw?.driver_version || hw?.driver)} />
            <Row label="CUDA" value={show(hw?.cuda_version || hw?.cuda)} />
            <Row label="TensorRT" value={show(hw?.tensorrt_version || hw?.tensorrt)} />
            <Row label="ONNX Runtime" value={show(hw?.onnxruntime_version || hw?.onnxruntime)} />
            <Row label="Provider (effective)" value={show(sec.PROVIDER?.values?.effective || runtime?.provider)}
              tone="text-emerald-400" />
            <Row label="Precision" value={show(sec.PRECISION?.values?.effective
              ?? sec.PRECISION?.values?.configured)} />
          </div>
          <div>
            <Row label="VRAM" value={vram.total_gb != null && vram.total_gb !== 'UNKNOWN'
              ? `${show(vram.used_gb)} / ${show(vram.total_gb)} GB` : 'UNKNOWN'} />
            <Row label="System RAM" value={tele?.ram_total
              ? `${tele.ram_used} / ${tele.ram_total} GB` : 'UNKNOWN'} />
            <Row label="This process (RSS)" value={tele?.process_rss ? `${tele.process_rss} GB` : 'UNKNOWN'} />
            <Row label="CPU" value={show(hw?.cpu_name)} />
            {/* The runtime report's POOLING section is populated by a RUN, so
                while idle it is legitimately all UNKNOWN. /api/system/telemetry
                asks session_pool directly and answers at any time, so it is the
                right source for a standing panel. Both are backend
                measurements — neither is inferred here. */}
            <Row label="Worker threads" value={show(tele?.threads
              ?? sec.POOLING?.values?.workers?.configured)} />
            <Row label="Pools (trt/detmask/detector)" value={tele?.pools
              ? `${tele.pools.trt} / ${tele.pools.detmask} / ${tele.pools.detector}` : 'UNKNOWN'} />
            <Row label="NVDEC / NVENC" value={hw ? `${hw.nvdec_available ? 'yes' : 'no'} / ${hw.nvenc_available ? 'yes' : 'no'}` : 'UNKNOWN'} />
            <Row label="Runtime status" value={show(runtime?.status?.code)} />
          </div>
        </div>
        {(sec.WARNINGS?.values?.count > 0 || sec.ERRORS?.values?.count > 0) && (
          <div className="mt-2 text-micro text-amber-300/90">
            {sec.ERRORS?.values?.count || 0} error(s), {sec.WARNINGS?.values?.count || 0} warning(s) in the
            last run — the Processing tab’s log has them in full.
          </div>
        )}
      </Section>

      <Section title="Updates" icon={Icon.settings}>
        <div className="text-micro text-white/45 mb-3">
          This checks whether a newer commit exists <em>and</em> whether it is safe to take. It never
          installs anything: applying an update is Pinokio’s <strong className="text-white/70">Update</strong>{' '}
          action, which runs the manifest-gated updater, snapshots the environment first, health-checks the
          result and rolls back if that fails. Python packages, CUDA, TensorRT, ONNX Runtime, FFmpeg,
          drivers and models are never changed by a browser action.
        </div>

        <div className="flex flex-wrap items-center gap-2 mb-3">
          <button
            type="button"
            onClick={() => checkUpdate(true)}
            disabled={updBusy}
            className="px-3 py-1.5 rounded-lg text-mini font-semibold bg-white/5 border border-white/10 text-white/80 hover:bg-white/10 hover:text-white transition-colors disabled:opacity-40"
          >
            {updBusy ? 'Checking…' : 'Check compatibility'}
          </button>
          {upd && (
            <span className={`text-mini font-bold uppercase tracking-wider ${CLASS_TONE[upd.classification] || 'text-white/60'}`}>
              {upd.classification}
            </span>
          )}
        </div>

        {upd ? (
          <>
            <Row label="Newer commit available" value={upd.available ? 'yes' : 'no'} />
            <Row label="Installed" value={show(upd.current?.version)} />
            {upd.candidate_sha && <Row label="Candidate" value={upd.candidate_sha.slice(0, 12)} />}
            <Row label="Applied by" value="Pinokio · Update" />
            {upd.reasons?.length > 0 && (
              <ul className="mt-2 ml-4 list-disc text-micro text-white/55 space-y-0.5">
                {upd.reasons.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            )}
            {upd.classification !== 'SAFE' && upd.available && (
              <div className="mt-2 px-3 py-2 rounded-xl text-micro bg-amber-500/10 border border-amber-500/30 text-amber-200">
                A newer commit exists but is not manifest-gated as safe for this environment. It is not
                offered as a one-click update; review the reasons above first.
              </div>
            )}
          </>
        ) : (
          <div className="text-micro text-white/35">
            Not checked. This is the only part of the app that reaches the internet, so it runs only when
            you ask — offline it simply reports UNVERIFIED and nothing else changes.
          </div>
        )}
      </Section>
    </div>
  );
}
