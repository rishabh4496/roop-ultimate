// The per-job 3-stage + finalization report, as rows a panel can print.
//
// The backend builds ONE report when a render ends (`roop/stage_telemetry.build_summary`) and sends it on
// /ws/telemetry (`summary`), in /api/runtime/state (PERFORMANCE.pipeline_report) and in each history entry
// (`report`). Everything here is a pure function of that object, so it can be checked without a render and
// so the Processing HUD, the Settings panel and the History row cannot disagree about what a number means.
//
// A metric the backend did not measure arrives as null and is shown as an em dash, never as 0: a stage
// that was skipped or restored from a checkpoint has no wall clock, and "0.0 s" would claim it was instant.

export const STAGE_1 = 'STAGE_1_SPARSE_PRESCAN';
export const STAGE_2 = 'STAGE_2_ACTIVE_PREPASS';
export const STAGE_3 = 'STAGE_3_MAIN_SWAP';
export const STAGE_4 = 'STAGE_4_FINALIZE';
export const STAGE_ORDER = [STAGE_1, STAGE_2, STAGE_3, STAGE_4];

export const STAGE_LABEL = {
  [STAGE_1]: 'Stage 1 · Sparse pre-scan',
  [STAGE_2]: 'Stage 2 · Active-range pre-pass',
  [STAGE_3]: 'Stage 3 · Main swap',
  [STAGE_4]: 'Stage 4 · Finalize',
};

const DASH = '—';
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

export const fmtSecs = (v) => {
  if (!isNum(v)) return DASH;
  if (v >= 60) return `${Math.floor(v / 60)}m ${String(Math.round(v % 60)).padStart(2, '0')}s`;
  return `${v.toFixed(v >= 10 ? 1 : 2)}s`;
};
export const fmtCount = (v) => (isNum(v) ? Math.round(v).toLocaleString('en-US') : DASH);
export const fmtPct = (v) => (isNum(v) ? `${v.toFixed(1)}%` : DASH);
export const fmtGb = (v) => (isNum(v) ? `${v.toFixed(2)} GB` : DASH);
export const fmtFps = (v) => (isNum(v) ? `${v.toFixed(1)} fps` : DASH);

/** done / running / failed / skipped / restored / pending, for a state chip. */
export const stateTone = (state) => {
  switch (state) {
    case 'DONE': return 'text-emerald-400';
    case 'RUNNING': return 'text-[var(--accent)]';
    case 'FAILED': return 'text-red-300';
    case 'RESTORED': return 'text-sky-300';
    case 'SKIPPED': return 'text-muted';
    default: return 'text-muted';
  }
};

/** Speedup multiplier as the backend's own words allow: always marked as an estimate. */
export const speedupText = (speedup) => {
  if (!speedup || !isNum(speedup.multiplier)) return DASH;
  return `≈ ${speedup.multiplier.toFixed(2)}×${speedup.estimated ? ' (estimate)' : ''}`;
};

const row = (label, value, tone) => ({ label, value, tone: tone || '' });

/**
 * Sections of label/value rows for a finished report. Returns [] for a missing / empty report so a caller
 * can render "no finished run yet" instead of a table of dashes.
 */
export function reportSections(report) {
  if (!report || typeof report !== 'object' || !report.stages) return [];
  const s = report.stages;
  const s1 = s[STAGE_1] || {};
  const s2 = s[STAGE_2] || {};
  const s3 = s[STAGE_3] || {};
  const s4 = s[STAGE_4] || null;
  const out = [];

  out.push({
    key: STAGE_1, label: STAGE_LABEL[STAGE_1], state: s1.state || 'PENDING',
    rows: [
      row('Wall clock', fmtSecs(s1.wall_s)),
      row('Sampled frames', isNum(s1.sampled_frames) && isNum(s1.samples_total)
        ? `${fmtCount(s1.sampled_frames)} / ${fmtCount(s1.samples_total)}` : fmtCount(s1.sampled_frames)),
      row('Worker pool utilization', isNum(s1.worker_utilization_pct) && isNum(s1.workers)
        ? `${fmtPct(s1.worker_utilization_pct)} of ${s1.workers} workers` : fmtPct(s1.worker_utilization_pct)),
      row('Active ranges', fmtCount(s1.active_ranges_count)),
      row('Frames skipped (no target)', fmtPct(s1.skipped_frames_pct)),
    ],
  });

  out.push({
    key: STAGE_2, label: STAGE_LABEL[STAGE_2], state: s2.state || 'PENDING',
    rows: [
      row('Wall clock', fmtSecs(s2.wall_s)),
      row('Coalesced active ranges', fmtCount(s2.active_ranges_count)),
      row('cap.set() seeks', fmtCount(s2.set_calls)),
      row('cap.grab() frames walked', fmtCount(s2.grab_calls)),
      row('Verified target frames', fmtCount(s2.verified_target_frames)),
      // What the pre-pass actually ran: absent (a dash) when Stage 2 was restored from a checkpoint.
      row('Detector calls', isNum(s2.detector_calls) && isNum(s2.scan_frames)
        ? `${fmtCount(s2.detector_calls)} of ${fmtCount(s2.scan_frames)} frames${isNum(s2.detect_interval) && s2.detect_interval > 1 ? ` (every ${s2.detect_interval}th)` : ''}`
        : '—'),
    ],
  });

  out.push({
    key: STAGE_3, label: STAGE_LABEL[STAGE_3], state: s3.state || 'PENDING',
    rows: [
      row('Wall clock', fmtSecs(s3.wall_s)),
      row('Zero-GPU passthrough frames', fmtCount(s3.passthrough_frames)),
      row('GPU frames (swap / enhance)', fmtCount(s3.gpu_frames)),
      row('Frames with a swapped face', fmtCount(s3.swapped_frames)),
      row('Faces swapped', fmtCount(s3.faces_swapped)),
      row('Peak VRAM', fmtGb(s3.peak_vram_gb)),
    ],
  });

  if (s4) {
    const ok = !!s4.ok;
    const color = s4.color || {};
    const rows = [
      row('Verification', ok ? 'passed' : 'FAILED', ok ? 'text-emerald-400' : 'text-red-300'),
      row('Frames delivered', isNum(s4.expected_frames)
        ? `${fmtCount(s4.frames)} of ${fmtCount(s4.expected_frames)} expected` : fmtCount(s4.frames)),
      row('Audio start vs video', isNum(s4.audio_start_offset_ms)
        ? `${s4.audio_start_offset_ms > 0 ? '+' : ''}${s4.audio_start_offset_ms.toFixed(1)} ms`
        : (s4.audio_present === false ? 'no audio track' : DASH)),
      row('Colour tags', color.color_primaries
        ? `${color.color_primaries} · ${color.color_space || '?'}${s4.color_preserved === false ? ' (CHANGED)' : ''}`
        : DASH, s4.color_preserved === false ? 'text-red-300' : ''),
      row('Mux + verify', `${fmtSecs(s4.mux_s)} + ${fmtSecs(s4.wall_s)} total`),
      row('Temp files', s4.orphans && s4.orphans.length ? `${s4.orphans.length} left behind` : 'none left',
        s4.orphans && s4.orphans.length ? 'text-amber-300' : ''),
    ];
    (s4.problems || []).forEach((p) => rows.push(row('Problem', p, 'text-red-300')));
    out.push({ key: STAGE_4, label: STAGE_LABEL[STAGE_4], state: s4.state || (ok ? 'DONE' : 'FAILED'), rows });
  }
  return out;
}

/** The one-line summary a collapsed card or a history row shows. */
export function headline(report) {
  if (!report || typeof report !== 'object' || !report.stages) return null;
  return {
    ok: report.ok !== false,
    fps: fmtFps(report.effective_fps),
    wall: fmtSecs(report.total_wall_s),
    speedup: speedupText(report.speedup),
    vram: fmtGb(report.vram && report.vram.peak_gb),
    frames: fmtCount(report.total_frames),
  };
}
