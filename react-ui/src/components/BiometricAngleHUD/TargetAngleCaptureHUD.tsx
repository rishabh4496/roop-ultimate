// Biometric Angle HUD: automatic multi-angle capture for the selected target
// person, shown before the swap so the angle bank can be inspected, corrected
// and committed.
//
// What it drives (app/routes_angle_scan.py):
//   /ws/angle-scan          scan -> quality -> export, streamed progress
//   /api/angle-scan/*       live thresholds, per-bin override, apply
//
// Nothing here changes the swap until "Add to angle bank" runs: that POST puts
// the chosen frames into the person's angle bank (the same list auto-capture
// and hand-capture fill), and the returned target payload is handed to Face
// Swap through `onBankUpdated`, exactly like the auto-capture button.
import React, { useEffect, useState } from 'react';
import AngleCoverageRadar from './AngleCoverageRadar';
import AngleMatrixGallery from './AngleMatrixGallery';
import useAutoAngleCapture from './useAutoAngleCapture';
import {
  BIN_BY_NAME, coverageWarnings, fmtAngle, qualityTier, toTimelineFrame, WARNING_TEXT,
  type AngleBinName, type ScanPhase,
} from './angleHudModel';

const AUTO_KEY = 'roop.angleHud.autoScan';

const PHASE_LABEL: Record<ScanPhase, string> = {
  scan: 'Scanning timeline',
  quality: 'Measuring pose & quality',
  export: 'Exporting references',
};

type Notify = (message: string, type?: 'success' | 'warning' | 'error' | 'info') => void;

export interface TargetAngleCaptureHUDProps {
  targetPersonId: string | null | undefined;
  targetMediaId: string | null | undefined;
  targetIndex: number;
  isVideo: boolean;
  personLabel?: string;
  currentFrame: number;                               // Face Swap timeline, 1-based
  onJumpToFrame: (timelineFrame: number) => void;
  onBankUpdated?: (payload: Record<string, unknown>) => void;
  notify?: Notify;
}

function readAuto(): boolean {
  try {
    const v = window.localStorage.getItem(AUTO_KEY);
    return v === null ? true : v === '1';
  } catch {
    return true;
  }
}

function Metric({ label, value, title }: { label: string; value: React.ReactNode; title?: string }) {
  return (
    <div className="flex justify-between gap-2" title={title}>
      <span className="text-slate-400/80">{label}</span>
      <span className="text-cyan-100 tabular-nums">{value}</span>
    </div>
  );
}

export default function TargetAngleCaptureHUD(props: TargetAngleCaptureHUDProps) {
  const {
    targetPersonId, targetMediaId, targetIndex, isVideo, personLabel,
    currentFrame, onJumpToFrame, onBankUpdated, notify,
  } = props;
  const [autoScan, setAutoScan] = useState<boolean>(readAuto);
  useEffect(() => {
    try { window.localStorage.setItem(AUTO_KEY, autoScan ? '1' : '0'); } catch { /* storage blocked */ }
  }, [autoScan]);

  const hud = useAutoAngleCapture({ targetPersonId, targetMediaId, targetIndex, isVideo, autoScan });
  const { session, portfolio, stats } = hud;
  const current = hud.sessionIsCurrent;
  const warnings = current ? coverageWarnings(session) : [];
  const filled = Object.keys(portfolio).length;
  const ref = hud.selectedReferenceBin;
  const refEntry = current ? portfolio[ref] : undefined;

  const assign = async (bin: AngleBinName, frame: number): Promise<void> => {
    const res = await hud.assignOverride(bin, frame);
    if (!res || !notify) return;
    const problems = res.warnings.map((w) => WARNING_TEXT[w] || w);
    // An override is kept whatever its pose; say so when the frame's measured
    // pose belongs to a different bin (or to none).
    const natural = res.entry?.source_bin;
    if (natural !== bin) {
      problems.push(natural
        ? `its pose reads as ${BIN_BY_NAME[natural].label.toLowerCase()}`
        : 'its pose falls between bins');
    }
    notify(problems.length
      ? `${BIN_BY_NAME[bin].label} set to frame ${frame}, but: ${problems.join(', ')}`
      : `${BIN_BY_NAME[bin].label} set to frame ${frame}`,
    problems.length ? 'warning' : 'success');
  };

  const apply = async (): Promise<void> => {
    const res = await hud.applyToPerson();
    if (!res) return;
    onBankUpdated?.(res);
    const added = Number(res.added || 0);
    const skipped = Array.isArray(res.skipped) ? (res.skipped as { bin: string; reason: string }[]) : [];
    if (notify) {
      notify(added
        ? `Added ${added} angle${added === 1 ? '' : 's'} to ${personLabel || 'this person'}'s angle bank`
          + (skipped.length ? ` (${skipped.length} skipped: ${[...new Set(skipped.map((s) => s.reason))].join(', ')})` : '')
        : `Nothing added: ${[...new Set(skipped.map((s) => s.reason))].join(', ') || 'no frames'}`,
      added ? 'success' : 'warning');
    }
  };

  if (!isVideo) {
    return (
      <div className="rounded-xl border border-slate-700/60 bg-slate-950/60 p-3 font-mono text-[11px] text-slate-400">
        Angle capture needs a video target.
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-cyan-400/15 bg-slate-950/70 p-3 space-y-3 text-slate-200"
         data-testid="angle-hud">
      {/* Header */}
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-cyan-300/80">Biometric angle capture</div>
          <div className="text-[11px] text-slate-400 truncate">
            {targetPersonId
              ? `${personLabel || 'Selected person'}${current ? ` · ${filled}/9 bins` : ''}`
              : 'Select a target person to begin'}
          </div>
        </div>
        <label className="flex items-center gap-1.5 font-mono text-[10px] text-slate-400 shrink-0 cursor-pointer">
          <input type="checkbox" checked={autoScan} onChange={(e) => setAutoScan(e.target.checked)}
                 className="accent-cyan-400" />
          AUTO
        </label>
      </div>

      {/* Progress */}
      {hud.isScanning && (
        <div className="space-y-1" data-testid="angle-progress" aria-live="polite">
          <div className="flex justify-between font-mono text-[10px] text-cyan-200/90">
            <span>{hud.phase ? PHASE_LABEL[hud.phase] : 'Starting'}</span>
            <span className="tabular-nums">{Math.round(hud.progress * 100)}%</span>
          </div>
          <div className="h-1 rounded-full bg-slate-800 overflow-hidden"
               role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(hud.progress * 100)}>
            <div className="h-full bg-cyan-400 transition-[width] duration-300" style={{ width: `${hud.progress * 100}%` }} />
          </div>
          <div className="grid grid-cols-3 gap-1 font-mono text-[10px] text-slate-400 tabular-nums">
            <span title="Sampled frames analysed per second">{stats?.scan_fps != null ? `${stats.scan_fps} fps` : '— fps'}</span>
            <span title="Detections in the largest tracklet that matches this person so far" className="text-center">
              trk {stats?.tracklet_size ?? 0}
            </span>
            <span className="text-right" title="Frame reached / frames in clip">
              {stats?.phase === 'quality'
                ? `${stats.done ?? 0}/${stats.candidates ?? 0}`
                : `${stats?.frame_idx != null ? stats.frame_idx + 1 : 0}/${stats?.frame_total ?? '—'}`}
            </span>
          </div>
          <button type="button" onClick={hud.cancelScan}
                  className="w-full rounded border border-slate-700 hover:border-slate-500 font-mono text-[10px] py-0.5 text-slate-300">
            Cancel scan
          </button>
        </div>
      )}

      {/* Errors / notices */}
      {(hud.error || hud.notice) && (
        <div role={hud.error ? 'alert' : 'status'}
             className={`flex items-start justify-between gap-2 rounded-lg border px-2 py-1.5 text-[11px] ${hud.error
               ? 'border-rose-400/40 bg-rose-500/10 text-rose-100' : 'border-slate-600/50 bg-slate-800/50 text-slate-300'}`}>
          <span>{hud.error ? hud.error.message : hud.notice}</span>
          <button type="button" onClick={hud.dismissMessage} aria-label="Dismiss" className="text-slate-400 hover:text-white">×</button>
        </div>
      )}

      {/* Coverage warnings */}
      {!hud.isScanning && warnings.map((w) => (
        <div key={w.bins.join()} role="alert" data-testid="angle-warning"
             className="rounded-lg border border-amber-400/40 bg-amber-500/10 px-2 py-1.5 text-[11px] text-amber-100">
          <span className="font-mono text-[10px] uppercase tracking-wider text-amber-300 mr-1">Coverage</span>
          {w.message}
        </div>
      ))}

      {current && session && session.tracklets.length === 0 && !hud.isScanning && (
        <div className="rounded-lg border border-amber-400/40 bg-amber-500/10 px-2 py-1.5 text-[11px] text-amber-100">
          No part of the clip matched this person
          {session.best_rejected_score != null
            ? ` (closest other face scored ${session.best_rejected_score.toFixed(2)} against a ${session.scan.similarity_threshold ?? 0.65} threshold)`
            : ''}. Capture a clearer frame of them and rescan.
        </div>
      )}

      {/* Radar + reference detail */}
      <div className="grid grid-cols-1 gap-2">
        <AngleCoverageRadar portfolio={current ? portfolio : {}} selected={ref}
                            scanning={hud.isScanning} onSelect={hud.selectReferenceBin} />
        {refEntry && (
          <div className="flex gap-2 rounded-lg border border-cyan-400/15 bg-slate-900/60 p-2">
            {refEntry.url && (
              <img src={refEntry.url} alt={`${BIN_BY_NAME[ref].label} reference`} loading="lazy" decoding="async"
                   className="w-20 h-20 rounded object-cover border border-cyan-400/20 shrink-0" />
            )}
            <div className="flex-1 min-w-0 font-mono text-[10px] space-y-0.5">
              <div className="text-cyan-300/90 uppercase tracking-wider truncate">
                {BIN_BY_NAME[ref].label} · {qualityTier(refEntry) || '—'}
              </div>
              <Metric label="yaw / pitch" value={`${fmtAngle(refEntry.yaw)} / ${fmtAngle(refEntry.pitch_up)}`}
                      title="Yaw + = toward the viewer's right; pitch + = looking up" />
              <Metric label="roll" value={fmtAngle(refEntry.roll)} />
              <Metric label="sharpness" value={refEntry.sharpness != null ? Math.round(refEntry.sharpness) : '—'} />
              <Metric label="identity" value={refEntry.id_similarity != null ? refEntry.id_similarity.toFixed(3) : '—'} />
              <Metric label="eyes (EAR)" value={refEntry.ear != null ? refEntry.ear.toFixed(2) : 'unmeasured'} />
              <Metric label="frame" value={refEntry.frame_idx != null
                ? `${toTimelineFrame(refEntry.frame_idx)}${refEntry.time_s != null ? ` · ${refEntry.time_s.toFixed(2)}s` : ''}` : '—'} />
              {refEntry.status === 'nearest' && (
                <div className="text-amber-200/80">borrowed {refEntry.distance_deg.toFixed(1)}° from outside the bin</div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Matrix */}
      {hud.overrideBin && (
        <div className="rounded border border-violet-400/40 bg-violet-500/10 px-2 py-1 text-[11px] text-violet-100">
          Scrub the timeline to a frame showing the <b>{BIN_BY_NAME[hud.overrideBin].label.toLowerCase()}</b> angle,
          then press Assign.
        </div>
      )}
      <AngleMatrixGallery
        portfolio={current ? portfolio : {}}
        selectedBin={ref}
        overrideBin={hud.overrideBin}
        currentFrame={currentFrame}
        busy={hud.pending || hud.isScanning || !current}
        onSelect={hud.selectReferenceBin}
        onJump={onJumpToFrame}
        onBeginOverride={hud.beginOverride}
        onAssign={assign}
        onCancelOverride={hud.cancelOverride}
        onClearOverride={hud.clearOverride}
      />

      {/* Thresholds */}
      <div className="space-y-2 rounded-lg border border-slate-700/50 p-2 font-mono text-[10px]">
        <label className="block space-y-0.5">
          <span className="flex justify-between text-slate-400">
            <span title="Faces smaller than this frontal-equivalent eye distance are refused">min eye distance</span>
            <span className="text-cyan-200 tabular-nums">{Math.round(hud.thresholds.minIod)} px</span>
          </span>
          <input type="range" min={20} max={120} step={1} value={hud.thresholds.minIod}
                 onChange={(e) => hud.setMinIod(Number(e.target.value))}
                 className="w-full accent-cyan-400" aria-label="Minimum eye distance in pixels" />
        </label>
        <label className="block space-y-0.5">
          <span className="flex justify-between text-slate-400">
            <span title="Refuse frames sharper than less than this fraction of the clip's median; 0 turns the blur gate off">blur strictness</span>
            <span className="text-cyan-200 tabular-nums">{hud.thresholds.blurFrac.toFixed(2)}</span>
          </span>
          <input type="range" min={0} max={0.9} step={0.05} value={hud.thresholds.blurFrac}
                 onChange={(e) => hud.setBlurFrac(Number(e.target.value))}
                 className="w-full accent-cyan-400" aria-label="Blur strictness" />
        </label>
        {current && session && (
          <div className="text-slate-500 tabular-nums">
            {session.candidates.valid}/{session.candidates.evaluated} candidates pass
            {session.scan.frames_scanned != null ? ` · ${session.scan.frames_scanned} frames scanned` : ''}
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex gap-1.5">
        <button type="button" onClick={hud.startScan} disabled={hud.isScanning || !targetPersonId}
                className="flex-1 rounded-lg border border-cyan-400/30 hover:border-cyan-300/60 hover:bg-cyan-400/10 font-mono text-[10px] uppercase tracking-wider py-1.5 text-cyan-200 disabled:opacity-40">
          {current ? 'Rescan' : 'Scan'}
        </button>
        <button type="button" onClick={apply} disabled={!current || !filled || hud.pending || hud.isScanning}
                className="flex-[1.6] rounded-lg bg-cyan-400/15 hover:bg-cyan-400/25 border border-cyan-300/40 font-mono text-[10px] uppercase tracking-wider py-1.5 text-cyan-100 disabled:opacity-40"
                title="Add the chosen frames to this person's angle bank, so the swap matches them at these angles">
          Add to angle bank ({current ? filled : 0})
        </button>
      </div>
    </div>
  );
}
