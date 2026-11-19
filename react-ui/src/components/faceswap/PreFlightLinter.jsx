import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { postJSON } from '../../api';
import { Icon } from '../../icons';

const EMPTY_ARRAY = Object.freeze([]);
const EMPTY_SUMMARY = Object.freeze({ expected: 0, missing: 0, partial: 0 });
const EMPTY_SETTINGS = Object.freeze({});

const REQUIRED_CHECKS = ['disk', 'encoder', 'provider', 'trt', 'resolution', 'vram', 'lut', 'output', 'assignments'];
const LABELS = {
  disk: 'Disk headroom',
  encoder: 'Video encoder',
  provider: 'Inference provider',
  trt: 'TensorRT engine cache',
  resolution: 'Face resolution',
  vram: 'VRAM headroom',
  fsz: 'Faceset angle coverage',
  segments: 'Segment retention',
  lut: 'LUT file',
  output: 'Output compatibility',
  assignments: 'Face assignments',
  probe: 'Backend preflight',
};

function severityOf(check) {
  if (check?.severity) return check.severity;
  return check?.ok === false ? 'warning' : 'ok';
}

function statusClasses(severity) {
  if (severity === 'blocker') return 'border-rose-400/35 bg-rose-950/35 text-rose-200';
  if (severity === 'warning' || severity === 'unknown') return 'border-amber-300/30 bg-amber-950/20 text-amber-100';
  return 'border-emerald-400/20 bg-emerald-950/15 text-emerald-100';
}

function checkValue(name, check) {
  if (!check) return 'Not reported';
  if (name === 'disk') return check.available === false ? 'Unavailable' : `${check.free_gb} GB free · ${check.ratio ?? '—'}× buffer`;
  if (name === 'encoder') return `${check.current_codec || 'unknown'}${check.has_nvidia ? ' · NVIDIA' : ''}`;
  if (name === 'provider') return `${check.active_provider || 'unknown'} active${check.requested_provider ? ` · ${check.requested_provider} requested` : ''}`;
  if (name === 'trt') return check.is_trt ? (check.status || 'TensorRT') : 'Not selected';
  if (name === 'resolution') return check.max_face_px ? `${check.max_face_px}px max face` : 'No face size reported';
  if (name === 'vram') return check.available ? `${check.free_gb} GB free / ${check.total_gb} GB` : 'Unavailable';
  if (name === 'fsz') return check.severity === 'unknown' ? 'Not measured' : check.missing_angles?.length ? `Missing ${check.missing_angles.join(', ')}` : 'Covered';
  if (name === 'segments') return check.keep_segments ? 'Retained' : 'Retention disabled';
  if (name === 'lut') return check.active ? (check.path || 'No LUT selected') : 'LUT bake off';
  if (name === 'output') return `${check.width && check.height ? `${check.width}×${check.height} · ` : ''}${check.format || 'unknown'} / ${check.codec || 'unknown'}`;
  if (name === 'assignments') return check.expected > 0 ? `${check.expected - check.missing} / ${check.expected} assigned` : 'No mapped faces';
  return check.message || check.warning || 'Ready';
}

/** Hardware and recipe readiness gate shared by single renders and staged batch dispatch. */
export default function PreFlightLinter({
  target = null,
  targets = EMPTY_ARRAY,
  targetDimensions = EMPTY_ARRAY,
  jobPayloads = EMPTY_ARRAY,
  targetFacesInfo = EMPTY_ARRAY,
  faceAssignments = EMPTY_ARRAY,
  assignmentSummary = EMPTY_SUMMARY,
  settings = EMPTY_SETTINGS,
  onAutoFix = null,
  onReadinessChange = null,
  batchCount = 1,
  className = '',
  compact = false,
}) {
  const [probeData, setProbeData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const probeSequence = useRef(0);

  const request = useMemo(() => {
    const stagedPayload = jobPayloads[0] || {};
    const dimensions = targetDimensions.map((item) => ({
      width: Number(item?.width || item?.video_width || 0),
      height: Number(item?.height || item?.video_height || 0),
      name: item?.name || '',
    }));
    const facePixels = targetFacesInfo.map((face) => {
      const box = face?.bbox;
      const boxWidth = Array.isArray(box) && box.length >= 4 ? Math.abs(Number(box[2]) - Number(box[0])) : 0;
      const boxHeight = Array.isArray(box) && box.length >= 4 ? Math.abs(Number(box[3]) - Number(box[1])) : 0;
      return Math.max(0, ...[face?.width, face?.height, boxWidth, boxHeight]
        .map((value) => Number(value || 0)).filter(Number.isFinite));
    });
    const selectedEnhancers = jobPayloads.map((payload) =>
      payload?.enhancer || payload?.selected_enhancer || settings?.selected_enhancer || '',
    );
    const resolutionFallbackEnhancer = selectedEnhancers.find((name) => name && name !== 'Restore Ultra')
      || selectedEnhancers[0]
      || settings?.selected_enhancer
      || '';
    return {
      target_size: Number(target?.size || 0),
      max_frames: Number(target?.frames || targets.reduce((sum, item) => sum + Number(item?.frames || 0), 0) || 300),
      target_width: Number(target?.width || target?.video_width || 0),
      target_height: Number(target?.height || target?.video_height || 0),
      target_dimensions: dimensions,
      batch_count: Math.max(1, Number(batchCount || 1)),
      codec: stagedPayload.output_video_codec || settings?.output_video_codec || settings?.video_encoder || 'libx264',
      video_format: stagedPayload.output_video_format || settings?.output_video_format || 'mp4',
      selected_enhancer: resolutionFallbackEnhancer,
      selected_enhancers: selectedEnhancers,
      max_face_px: Math.max(0, ...facePixels),
      face_dimensions_known: facePixels.length > 0 && facePixels.every((value) => Number.isFinite(value) && value > 0),
      pixel_boost: Number(stagedPayload.pixel_boost || settings?.pixel_boost || settings?.subsample_size || 256),
      ultramax_batch_size: Math.max(1, Number(stagedPayload.ultramax_batch_size || settings?.ultramax_batch_size || 1)),
      faceset_file: stagedPayload.faceset_file || settings?.faceset_file || settings?.selected_faceset || '',
      keep_segments: stagedPayload.keep_segments ?? settings?.keep_segments !== false,
      segment_retention_path: settings?.segment_retention_path || settings?.segment_dir || '.segments/',
      provider: stagedPayload.provider || settings?.provider || 'auto',
      lut_path: stagedPayload.lut_path || settings?.lut_path || '',
      lut_bake: stagedPayload.lut_bake ?? !!settings?.lut_bake,
      face_selection: stagedPayload.detection || settings?.face_detection_mode || 'All faces',
      face_assignments: Array.isArray(faceAssignments) ? faceAssignments : [],
      assignment_summary: assignmentSummary || { expected: 0, missing: 0, partial: 0 },
    };
  }, [target, targets, targetDimensions, jobPayloads, batchCount, settings, targetFacesInfo, faceAssignments, assignmentSummary]);
  const requestKey = useMemo(() => JSON.stringify(request), [request]);

  const runProbe = useCallback(async () => {
    const sequence = ++probeSequence.current;
    setLoading(true);
    try {
      const response = await postJSON('/api/render/preflight', request);
      const checks = response?.checks || {};
      const missing = REQUIRED_CHECKS.filter((name) => !checks[name]);
      if (missing.length) {
        throw new Error(`Backend preflight response is missing ${missing.join(', ')} checks.`);
      }
      if (sequence === probeSequence.current) setProbeData({ ...response, requestKey });
    } catch (error) {
      const message = `Readiness could not be verified: ${error?.message || 'backend probe failed'}`;
      if (sequence === probeSequence.current) setProbeData({
        requestKey,
        status: 'blocked',
        checks: { probe: { ok: false, severity: 'blocker', message, warning: message } },
        warnings: [{ check: 'probe', severity: 'blocker', message, auto_fix: 'recheck', auto_fix_label: 'Retry check' }],
        blockers: [{ check: 'probe', severity: 'blocker', message }],
        auto_fixes: [{ action: 'recheck', label: 'Retry check' }],
      });
    } finally {
      if (sequence === probeSequence.current) setLoading(false);
    }
  }, [request, requestKey]);

  useEffect(() => { runProbe(); }, [runProbe]);

  const currentProbeData = probeData?.requestKey === requestKey ? probeData : null;
  const currentLoading = loading || !currentProbeData;
  const checks = currentProbeData?.checks || {};
  const warnings = currentProbeData?.warnings || [];
  const blockers = currentProbeData?.blockers || warnings.filter((item) => item.severity === 'blocker');
  const status = currentLoading ? 'checking' : (currentProbeData?.status || 'checking');
  const statusLabel = status === 'ready' ? 'Optimal'
    : status === 'blocked' ? `${blockers.length || 1} blocker${(blockers.length || 1) === 1 ? '' : 's'}`
      : status === 'warning' ? `${warnings.length} warning${warnings.length === 1 ? '' : 's'}`
        : 'Checking';
  const summaryTone = status === 'blocked' ? 'border-rose-400/35 bg-rose-950/20'
    : status === 'warning' ? 'border-amber-300/30 bg-amber-950/15'
      : status === 'ready' ? 'border-emerald-400/20 bg-emerald-950/10'
        : 'border-cyan-400/20 bg-cyan-950/10';
  const fixByAction = useMemo(() => new Map((currentProbeData?.auto_fixes || []).map((fix) => [fix.action, fix])), [currentProbeData]);

  useLayoutEffect(() => {
    onReadinessChange?.({
      loading: currentLoading,
      status,
      blockers,
      blockerMessage: blockers[0]?.message || '',
    });
  }, [currentLoading, status, blockers, onReadinessChange]);

  const handleFix = (warning) => {
    const action = warning.auto_fix;
    if (action === 'recheck') { runProbe(); return; }
    const fix = fixByAction.get(action) || { action, label: warning.auto_fix_label || 'Review' };
    onAutoFix?.(action, fix);
  };

  const checkItems = Object.entries(checks).filter(([name]) => name !== 'probe');

  return (
    <section
      aria-label="Pre-flight render readiness diagnostics"
      data-testid="preflight-linter"
      className={`relative rounded-xl border text-xs backdrop-blur-md ${summaryTone} ${className}`}
    >
      <div className={`flex flex-wrap items-center justify-between gap-x-3 ${compact ? 'gap-y-1 px-3 py-1.5' : 'gap-y-2 px-3 py-2'}`}>
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${loading ? 'animate-pulse bg-cyan-300' : status === 'blocked' ? 'animate-pulse bg-rose-400 shadow-[0_0_8px_rgba(251,113,133,0.75)]' : status === 'warning' ? 'bg-amber-300 shadow-[0_0_8px_rgba(252,211,77,0.65)]' : 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.65)]'}`} />
          <div className="min-w-0 truncate font-semibold text-white/90">
            <span>Pre-Flight Render Linter · </span>
            <span className={status === 'blocked' ? 'text-rose-300' : status === 'warning' ? 'text-amber-200' : status === 'ready' ? 'text-emerald-300' : 'text-cyan-200'}>{statusLabel}</span>
          </div>
          {batchCount > 1 && <span className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-micro text-white/65">Matrix · {batchCount} jobs</span>}
          {checks.vram?.available && (
            <span className={`rounded border px-1.5 py-0.5 font-mono text-micro ${statusClasses(checks.vram.severity)}`} title="Current device VRAM snapshot; free/total rather than a fixed card-size assumption">
              VRAM {checks.vram.free_gb} / {checks.vram.total_gb} GB free
            </span>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <span className={`rounded-full border px-2 py-0.5 font-mono text-micro font-bold ${statusClasses(checks.trt?.severity || 'ok')}`} title={checks.trt?.warning || 'TensorRT engine cache readiness'}>
            {checks.trt?.status || 'TRT · N/A'}
          </span>
          <button type="button" onClick={() => runProbe()} disabled={loading} className="rounded px-1.5 py-0.5 text-micro font-medium text-white/55 hover:bg-white/5 hover:text-white disabled:opacity-40" title="Re-read current hardware and recipe state">
            Recheck
          </button>
          <button type="button" onClick={() => setExpanded((value) => !value)} aria-expanded={expanded} className="flex items-center gap-1 rounded px-1.5 py-0.5 text-micro font-medium text-white/55 hover:bg-white/5 hover:text-white">
            {expanded ? 'Hide details' : 'Details'} <Icon.down size={12} className={`transition-transform ${expanded ? 'rotate-180' : ''}`} />
          </button>
        </div>
      </div>

      {warnings.length > 0 && (
        <div className={`space-y-1.5 border-t px-3 py-2 ${status === 'blocked' ? 'border-rose-400/20' : 'border-amber-300/15'}`}>
          {warnings.map((warning, index) => {
            const fix = warning.auto_fix ? (fixByAction.get(warning.auto_fix) || { label: warning.auto_fix_label }) : null;
            const severity = warning.severity || 'warning';
            const actionLabel = warning.auto_fix
              ? (['review_assignments', 'open_facemgr'].includes(warning.auto_fix) ? 'Review' : 'Auto-Fix')
              : 'Recheck';
            return (
              <div key={`${warning.check}-${index}`} className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1">
                <div className={`flex min-w-0 flex-1 items-start gap-1.5 leading-snug ${severity === 'blocker' ? 'text-rose-200' : 'text-amber-100/90'}`}>
                  <span className="mt-0.5 shrink-0 font-bold">{severity === 'blocker' ? 'BLOCK' : 'WARN'}</span>
                  <span>{warning.message}</span>
                </div>
                <button type="button" onClick={() => warning.auto_fix ? handleFix(warning) : runProbe()} disabled={currentLoading} className={`inline-flex shrink-0 items-center gap-1 rounded-lg border px-2 py-1 text-micro font-bold transition-colors disabled:cursor-wait disabled:opacity-50 ${severity === 'blocker' ? 'border-rose-200/25 bg-rose-300/15 text-rose-100 hover:bg-rose-300/25' : 'border-amber-200/25 bg-amber-300/10 text-amber-100 hover:bg-amber-300/20'}`} title={warning.auto_fix ? 'Apply the suggested safe configuration change' : 'Recheck this condition after reviewing the details'}>
                    <span>{actionLabel}</span><span className="opacity-80">{fix?.label || 'Inspect details'}</span>
                  </button>
              </div>
            );
          })}
        </div>
      )}

      {expanded && (
        <div className={`grid grid-cols-1 gap-1.5 p-2 sm:grid-cols-2 xl:grid-cols-4 ${compact ? 'absolute inset-x-0 top-full z-40 mt-1.5 rounded-xl border border-white/15 bg-[#0c0e14] shadow-2xl' : 'border-t border-white/5'}`}>
          {checkItems.map(([name, check]) => {
            const severity = severityOf(check);
            return (
              <div key={name} className={`min-w-0 rounded-lg border p-2 ${statusClasses(severity)}`}>
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate font-sans text-nano font-bold uppercase tracking-wide text-white/70">{LABELS[name] || name}</span>
                  <span className="shrink-0 font-mono text-[9px] uppercase">{severity}</span>
                </div>
                <div className="mt-1 truncate font-mono text-mini font-bold" title={check.path || check.warning || checkValue(name, check)}>{checkValue(name, check)}</div>
                {check.warning && <div className="mt-1 font-sans text-micro leading-snug text-white/65">{check.warning}</div>}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
