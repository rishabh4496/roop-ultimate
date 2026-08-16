import React, { useState, useEffect } from 'react';
import { postJSON } from '../api';
import { AnimatedNumber } from './ui';
import { Icon } from '../icons';

const scoreColor = (s) => (s >= 80 ? '#10b981' : s >= 60 ? 'var(--accent)' : s >= 40 ? '#f59e0b' : '#ef4444');
const gradeColor = (g) => ({ A: '#10b981', B: '#34d399', C: '#f59e0b', D: '#f97316', E: '#ef4444' }[g] || '#9ca3af');

function MetricBar({ label, score, detail }) {
  const color = scoreColor(score);
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-mini">
        <span className="font-bold text-white/70">{label}</span>
        <span className="font-black tabular-nums" style={{ color }}>
          <AnimatedNumber value={score} decimals={0} /><span className="text-white/30 font-medium"> / 100{detail ? ` · ${detail}` : ''}</span>
        </span>
      </div>
      <div className="h-2 w-full rounded-full bg-white/10 overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700 ease-out" style={{ width: `${score}%`, background: color, boxShadow: `0 0 8px ${color}55` }} />
      </div>
    </div>
  );
}

export default function QualityReport({ outputPath, notify }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');

  // Reset when a new output appears.
  useEffect(() => { setData(null); setErr(''); }, [outputPath]);

  const analyze = async () => {
    setLoading(true); setErr('');
    try {
      const res = await postJSON('/api/quality/analyze', { path: outputPath });
      if (res.ok) setData(res);
      else { setErr(res.message || 'No face detected in the output'); }
    } catch (e) {
      setErr(e.message); if (notify) notify(e.message, 'error');
    } finally { setLoading(false); }
  };

  if (!data && !loading && !err) {
    return (
      <button type="button" onClick={analyze}
        className="w-full mt-1 py-2.5 rounded-xl text-xs font-bold bg-white/[0.03] border border-white/10 text-white/70 hover:border-[var(--accent)]/40 hover:text-white transition-colors">
        Analyze quality of this result
      </button>
    );
  }

  if (loading) {
    return (
      <div className="mt-1 py-4 flex items-center justify-center gap-3 rounded-xl bg-black/30 border border-white/5">
        <div className="h-5 w-5 rounded-full border-2 border-white/10 border-t-[var(--accent)] animate-spin" />
        <span className="text-xs text-white/50 font-medium">Analyzing likeness, sharpness & stability…</span>
      </div>
    );
  }

  if (err) {
    return (
      <div className="mt-1 p-3 rounded-xl bg-black/30 border border-white/5 flex items-center justify-between gap-2">
        <span className="text-xs text-white/50 flex items-center gap-1.5"><Icon.warning size={13} className="text-amber-400/80" /> {err}</span>
        <button type="button" onClick={analyze} className="text-mini font-bold text-[var(--accent)] hover:underline shrink-0">Retry</button>
      </div>
    );
  }

  const m = data.metrics;
  const g = m.grade;
  const gc = gradeColor(g);
  const radius = 26;
  const circ = radius * 2 * Math.PI;
  const off = circ - (m.overall_score / 100) * circ;

  return (
    <div className="mt-1 p-4 rounded-xl bg-black/30 border border-white/5 space-y-4">
      <div className="flex items-center gap-4">
        {/* Grade ring */}
        <div className="relative h-20 w-20 shrink-0 flex items-center justify-center">
          <svg className="transform -rotate-90 w-20 h-20" viewBox="0 0 64 64">
            <circle cx="32" cy="32" r={radius} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="5" />
            <circle cx="32" cy="32" r={radius} fill="none" stroke={gc} strokeWidth="5" strokeLinecap="round"
              strokeDasharray={`${circ} ${circ}`} style={{ strokeDashoffset: off, transition: 'stroke-dashoffset 0.8s ease-out', filter: `drop-shadow(0 0 4px ${gc}66)` }} />
          </svg>
          <div className="absolute flex flex-col items-center leading-none">
            <span className="text-2xl font-black" style={{ color: gc }}>{g}</span>
            <span className="text-nano font-bold text-white/45 tabular-nums"><AnimatedNumber value={m.overall_score} decimals={0} /></span>
          </div>
        </div>
        <div className="min-w-0">
          <div className="text-micro font-semibold uppercase tracking-[0.14em] text-white/45">Quality Report</div>
          <div className="text-sm font-bold text-white/90">
            {g === 'A' ? 'Excellent result' : g === 'B' ? 'Good result' : g === 'C' ? 'Acceptable' : g === 'D' ? 'Needs work' : 'Poor — review settings'}
          </div>
          <div className="text-micro text-white/45 mt-0.5">
            {data.is_video ? `Sampled ${data.sampled} frames` : 'Single image'}
            {!data.has_source && ' · no source loaded (likeness skipped)'}
          </div>
        </div>
      </div>

      <div className="space-y-2.5">
        {m.identity_score !== undefined && (
          <MetricBar label="Identity likeness" score={m.identity_score} detail={`cos ${m.identity_similarity}`} />
        )}
        {m.temporal_stability !== undefined && (
          <MetricBar label="Temporal stability" score={m.temporal_stability} />
        )}
        <MetricBar label="Sharpness" score={m.sharpness_score} />
        {m.detection_rate !== undefined && data.is_video && (
          <MetricBar label="Face detection coverage" score={m.detection_rate} />
        )}
      </div>

      <div className="flex items-center justify-between pt-1">
        <span className="text-nano text-white/45 leading-tight max-w-[70%]">
          Heuristic scores from re-detecting faces in the output. Load the source for likeness.
        </span>
        <button type="button" onClick={analyze} className="text-mini font-bold text-white/50 hover:text-white shrink-0">Re-analyze</button>
      </div>
    </div>
  );
}
