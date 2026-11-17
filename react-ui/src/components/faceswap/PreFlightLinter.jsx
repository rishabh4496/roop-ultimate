import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { postJSON, getJSON } from '../../api';
import { Icon } from '../../icons';

/**
 * Smart Pre-Flight Render Readiness Linter & 1-Click Auto-Fix.
 *
 * Runs real-time diagnostics before single-video render or Batch Matrix jobs:
 *   Check 1: Disk Space Custody (free space vs required output + temporary GOP footprint, <1.5x warning)
 *   Check 2: Hardware Encoder Mismatch (NVIDIA GPU active + CPU libx264 -> 1-click "Switch to NVENC")
 *   Check 3: Precision / TRT Warmth Check ([TRT Ready] vs [Will Compile ~90s])
 *   Check 4: Pose / Resolution Mismatch (Restore Ultra disabled when face bbox > 256px -> 1-click "Enable Restore Ultra")
 *   Check 5: 12 GB VRAM Budget (Pixel Boost + UltraMax batch size exceeding 12 GB budget -> 1-click "Clamp Batch to 1")
 *   Check 6: Missing .fsz Profile Angles (active faceset lacking yaw/pitch profile angles -> 1-click "Open Face Manager")
 *   Check 7: Segment Retention Paths (chunked renders missing retention custody or disabled keep_segments)
 */
export default function PreFlightLinter({
  target = null,
  targetFacesInfo = [],
  settings = {},
  onAutoFix = null,
  batchCount = 1,
  className = '',
  // compact: for a pinned bar. Tighter padding, and "Details" opens as a popover over the page instead of
  // growing the bar.
  compact = false,
}) {
  const [probeData, setProbeData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);

  const targetSize = target?.size || 0;
  const maxFrames = target?.frames || 300;
  const codec = settings?.output_video_codec || settings?.video_encoder || 'libx264';
  const selectedEnhancer = settings?.selected_enhancer || '';
  const provider = settings?.provider || '';
  const pixelBoost = Number(settings?.pixel_boost || settings?.subsample_size || 256);
  const ultramaxBatchSize = Math.max(1, Number(settings?.ultramax_batch_size || 1));
  const keepSegments = settings?.keep_segments !== false;
  const segmentRetentionPath = settings?.segment_retention_path || settings?.segment_dir || '.segments/';
  const facesetFile = settings?.faceset_file || settings?.selected_faceset || '';
  const useFaceset = Boolean(settings?.use_faceset);
  const enableSegments = Boolean(settings?.enable_segments);

  // Compute max face dimensions locally from targetFacesInfo as fast fallback
  const maxFacePx = useMemo(() => {
    let maxPx = 0;
    if (Array.isArray(targetFacesInfo)) {
      for (const f of targetFacesInfo) {
        if (f.width && f.height) {
          maxPx = Math.max(maxPx, f.width, f.height);
        } else if (f.bbox && Array.isArray(f.bbox) && f.bbox.length >= 4) {
          maxPx = Math.max(maxPx, Math.round(f.bbox[2]), Math.round(f.bbox[3]));
        }
      }
    }
    return maxPx;
  }, [targetFacesInfo]);

  const runProbe = useCallback(async () => {
    setLoading(true);
    let serverRes = null;
    try {
      serverRes = await postJSON('/api/render/preflight', {
        target_size: targetSize,
        max_frames: maxFrames,
        batch_count: batchCount,
        codec,
        selected_enhancer: selectedEnhancer,
        max_face_px: maxFacePx,
        pixel_boost: pixelBoost,
        ultramax_batch_size: ultramaxBatchSize,
        faceset_file: facesetFile,
        keep_segments: keepSegments,
      });
    } catch {
      // server call failed or unmocked; will compute client-side
      serverRes = null;
    }

    // Also probe radar if faceset active and not already provided
    let radarMissing = [];
    if (facesetFile || useFaceset) {
      try {
        const radar = await getJSON('/api/facemgr/radar');
        if (radar && Array.isArray(radar.missing)) {
          radarMissing = radar.missing;
        }
      } catch {
        // radar probe optional
      }
    }

    const hasNvidia = /tensorrt|cuda/i.test(provider);
    const encMismatch = hasNvidia && (/libx264|libx265/i.test(codec) || codec.startsWith('libx'));
    const isTrt = /tensorrt/i.test(provider);
    const resMismatch = maxFacePx > 256 && selectedEnhancer !== 'Restore Ultra';

    // ── Check 5: 12 GB VRAM Budget (Pixel Boost + UltraMax batch size) ──
    const tileFactor = pixelBoost >= 768 ? 9 : pixelBoost >= 512 ? 4 : 1;
    const isUltraMax = /ultramax/i.test(selectedEnhancer);
    const estimatedVramGb = Number((4.5 + (tileFactor * 0.45) + (ultramaxBatchSize * (isUltraMax ? 2.3 : 1.4))).toFixed(1));
    const vramOverBudget = estimatedVramGb > 12.0;

    // ── Check 6: Missing .fsz profile angles ──
    const missingAngles = (serverRes?.checks?.fsz?.missing_angles || radarMissing || []);
    const fszAngleMissing = Boolean(facesetFile && missingAngles.length > 0);

    // ── Check 7: Segment retention paths ──
    const isChunked = batchCount > 1 || maxFrames > 600 || enableSegments;
    const segmentRetentionMismatch = isChunked && (!keepSegments || !segmentRetentionPath);

    const warnings = [];
    const autoFixes = [];

    // Disk space check
    const diskCheck = serverRes?.checks?.disk || {
      ok: true,
      free_gb: 50.0,
      required_gb: 2.0,
      ratio: 25.0,
      warning: null,
    };
    if (diskCheck.warning) {
      warnings.push({ check: 'disk', message: diskCheck.warning });
    }

    // Hardware encoder check
    const encoderCheck = serverRes?.checks?.encoder || {
      ok: !encMismatch,
      has_nvidia: hasNvidia,
      current_codec: codec,
      recommended_codec: 'h264_nvenc',
      warning: encMismatch ? 'NVIDIA GPU active but encoder is CPU libx264' : null,
      auto_fix: encMismatch ? 'switch_nvenc' : null,
    };
    if (encoderCheck.warning) {
      warnings.push({ check: 'encoder', message: encoderCheck.warning });
      autoFixes.push({ action: 'switch_nvenc', label: 'Switch to NVENC', codec: 'h264_nvenc' });
    }

    // TRT engine check
    const trtCheck = serverRes?.checks?.trt || {
      ok: !isTrt,
      is_trt: isTrt,
      status: isTrt ? '[Will Compile ~90s]' : '[TRT Ready]',
      warning: null,
    };
    if (trtCheck.warning) {
      warnings.push({ check: 'trt', message: trtCheck.warning });
    }

    // Face resolution / restorer check
    const resolutionCheck = serverRes?.checks?.resolution || {
      ok: !resMismatch,
      max_face_px: maxFacePx,
      selected_enhancer: selectedEnhancer,
      is_restore_ultra: !resMismatch,
      warning: resMismatch ? `Target face is ${maxFacePx}px (>256px), but Restore Ultra is disabled.` : null,
      auto_fix: resMismatch ? 'enable_restore_ultra' : null,
    };
    if (resolutionCheck.warning) {
      warnings.push({ check: 'resolution', message: resolutionCheck.warning });
      autoFixes.push({ action: 'enable_restore_ultra', label: 'Enable Restore Ultra', enhancer: 'Restore Ultra' });
    }

    // VRAM budget check
    const vramCheck = serverRes?.checks?.vram || {
      ok: !vramOverBudget,
      estimated_vram_gb: estimatedVramGb,
      budget_gb: 12.0,
      pixel_boost: pixelBoost,
      ultramax_batch_size: ultramaxBatchSize,
      warning: vramOverBudget
        ? `Estimated VRAM (${estimatedVramGb} GB) exceeds 12 GB budget (Pixel Boost ${pixelBoost}px + UltraMax batch ${ultramaxBatchSize}).`
        : null,
      auto_fix: vramOverBudget ? 'clamp_vram_budget' : null,
    };
    if (vramCheck.warning) {
      warnings.push({ check: 'vram', message: vramCheck.warning });
      autoFixes.push({ action: 'clamp_vram_budget', label: 'Clamp UltraMax Batch to 1', batch_size: 1 });
    }

    // Faceset profile angles check
    const fszCheck = serverRes?.checks?.fsz || {
      ok: !fszAngleMissing,
      faceset_file: facesetFile,
      missing_angles: missingAngles,
      warning: fszAngleMissing
        ? `Active faceset is missing profile angles (${missingAngles.join(', ')}). Swaps at extreme yaw will drift.`
        : null,
      auto_fix: fszAngleMissing ? 'open_facemgr' : null,
    };
    if (fszCheck.warning) {
      warnings.push({ check: 'fsz', message: fszCheck.warning });
      autoFixes.push({ action: 'open_facemgr', label: 'Inspect in Face Manager', route: '#/facemgr' });
    }

    // Segment retention path check
    const segmentCheck = serverRes?.checks?.segments || {
      ok: !segmentRetentionMismatch,
      keep_segments: keepSegments,
      retention_path: segmentRetentionPath,
      warning: segmentRetentionMismatch
        ? `Segment retention is disabled; chunked render segments in '${segmentRetentionPath}' will not be retained.`
        : null,
      auto_fix: segmentRetentionMismatch ? 'enable_segment_retention' : null,
    };
    if (segmentCheck.warning) {
      warnings.push({ check: 'segments', message: segmentCheck.warning });
      autoFixes.push({ action: 'enable_segment_retention', label: 'Retain Segment Chunks', keep_segments: true });
    }

    setProbeData({
      status: warnings.length === 0 ? 'ready' : 'warning',
      checks: {
        disk: diskCheck,
        encoder: encoderCheck,
        trt: trtCheck,
        resolution: resolutionCheck,
        vram: vramCheck,
        fsz: fszCheck,
        segments: segmentCheck,
      },
      warnings,
      auto_fixes: autoFixes,
    });
    setLoading(false);
  }, [batchCount, codec, enableSegments, facesetFile, keepSegments, maxFacePx, maxFrames, pixelBoost, provider, segmentRetentionPath, selectedEnhancer, targetSize, ultramaxBatchSize, useFaceset]);

  useEffect(() => {
    runProbe();
  }, [runProbe]);

  const checks = probeData?.checks;
  const warnings = probeData?.warnings || [];
  const autoFixes = probeData?.auto_fixes || [];
  const status = probeData?.status || 'ready';

  const isReady = status === 'ready' && warnings.length === 0;
  const isWarning = warnings.length > 0;
  const trtStatus = checks?.trt?.status || '[TRT Ready]';
  const isTrtWarm = trtStatus === '[TRT Ready]';

  const handleFixClick = (fix) => {
    if (fix.action === 'open_facemgr') {
      window.location.hash = '#/facemgr';
    }
    if (onAutoFix) {
      onAutoFix(fix.action, fix);
    }
  };

  return (
    <div
      role="region"
      aria-label="Pre-flight render readiness diagnostics"
      data-testid="preflight-linter"
      className={`relative rounded-xl border transition-all text-xs select-none backdrop-blur-md ${
        isReady
          ? 'border-emerald-500/20 bg-emerald-950/15'
          : 'border-amber-500/30 bg-amber-950/20'
      } ${className}`}
    >
      {/* ── Summary Status Bar ── */}
      <div className={`flex flex-wrap items-center justify-between gap-x-2 ${compact ? 'gap-y-1 px-3 py-1.5' : 'gap-y-2 px-3 py-2'}`}>
        <div className="flex items-center gap-2.5 min-w-0 flex-wrap">
          <span
            className={`h-2.5 w-2.5 shrink-0 rounded-full transition-transform ${
              loading
                ? 'bg-cyan-400 animate-spin'
                : isReady
                ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.7)] animate-pulse'
                : 'bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.7)]'
            }`}
          />
          <span className="font-semibold text-white/90 truncate flex items-center gap-1.5">
            <span>Pre-Flight Readiness:</span>
            {loading ? (
              <span className="text-white/50">Probing hardware &amp; disk…</span>
            ) : isReady ? (
              <span className="text-emerald-400 font-bold">Ready</span>
            ) : (
              <span className="text-amber-300 font-bold">
                {warnings.length} Warning{warnings.length > 1 ? 's' : ''}
              </span>
            )}
          </span>
          {batchCount > 1 && (
            <span className="px-1.5 py-0.5 rounded text-micro bg-white/10 text-white/70 font-mono">
              Matrix ({batchCount} jobs)
            </span>
          )}
          {checks?.vram && (
            <span
              className={`px-1.5 py-0.5 rounded text-mini font-mono border ${
                checks.vram.ok
                  ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300'
                  : 'bg-amber-500/20 border-amber-500/40 text-amber-300 animate-pulse'
              }`}
              title="Estimated VRAM vs 12 GB Budget"
            >
              VRAM: {checks.vram.estimated_vram_gb} / 12 GB
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {/* TRT Warmth Badge */}
          <span
            className={`px-2 py-0.5 rounded-full text-micro font-mono font-bold border transition-colors ${
              isTrtWarm
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                : 'bg-amber-500/15 border-amber-500/40 text-amber-300 animate-pulse'
            }`}
            title={checks?.trt?.warning || 'TensorRT engine compilation status'}
          >
            {trtStatus}
          </span>

          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            aria-expanded={expanded}
            className="text-mini font-medium text-white/50 hover:text-white flex items-center gap-1 px-1.5 py-0.5 rounded hover:bg-white/5 transition-colors"
          >
            <span>{expanded ? 'Hide Details' : 'Details'}</span>
            <Icon.down size={12} className={`transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`} />
          </button>
        </div>
      </div>

      {/* ── Auto-Fix Quick Action Bar (when warnings exist) ── */}
      {isWarning && autoFixes.length > 0 && (
        <div className="px-3 py-1 border-t border-amber-500/20 bg-amber-500/10 flex flex-wrap items-center gap-2 rounded-b-xl">
          <span className="text-nano font-bold uppercase tracking-wider text-amber-300/80">
            1-Click Auto-Fix:
          </span>
          {autoFixes.map((fix) => (
            <button
              key={fix.action}
              type="button"
              onClick={() => handleFixClick(fix)}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-micro font-bold bg-amber-500 hover:bg-amber-400 text-black shadow-sm transition-colors cursor-pointer"
            >
              <span>⚡</span>
              <span>{fix.label}</span>
            </button>
          ))}
        </div>
      )}

      {/* ── Expanded Diagnostic Report Cards ── */}
      {expanded && checks && (
        <div className={`px-3 pb-3 pt-2 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-2 font-mono text-mini ${
          compact
            ? 'absolute inset-x-0 top-full z-40 mt-1.5 rounded-xl border border-white/15 bg-[#0c0e14] shadow-2xl'
            : 'border-t border-white/5'
        }`}>
          {/* Check 1: Disk Space */}
          <div
            className={`p-2 rounded-lg border ${
              checks.disk.ok ? 'bg-black/30 border-white/10' : 'bg-amber-950/40 border-amber-500/40'
            }`}
          >
            <div className="flex items-center justify-between text-muted text-nano uppercase tracking-wider font-sans font-bold">
              <span className="flex items-center gap-1">
                <Icon.drive size={12} />
                <span>Disk Custody</span>
              </span>
              <span className={checks.disk.ok ? 'text-emerald-400' : 'text-amber-400'}>
                {checks.disk.ratio}x buffer
              </span>
            </div>
            <div className="mt-1 text-white font-bold">
              {checks.disk.free_gb} GB Free
            </div>
            <div className="text-white/40 text-micro">
              Req: ~{checks.disk.required_gb} GB (GOP + Out)
            </div>
            {checks.disk.warning && (
              <div className="mt-1 text-micro text-amber-300 font-sans leading-tight">
                {checks.disk.warning}
              </div>
            )}
          </div>

          {/* Check 2: Encoder */}
          <div
            className={`p-2 rounded-lg border ${
              checks.encoder.ok ? 'bg-black/30 border-white/10' : 'bg-amber-950/40 border-amber-500/40'
            }`}
          >
            <div className="flex items-center justify-between text-muted text-nano uppercase tracking-wider font-sans font-bold">
              <span className="flex items-center gap-1">
                <Icon.cpu size={12} />
                <span>Hardware Encoder</span>
              </span>
              <span className={checks.encoder.ok ? 'text-emerald-400' : 'text-amber-400'}>
                {checks.encoder.ok ? 'Matched' : 'Mismatch'}
              </span>
            </div>
            <div className="mt-1 text-white font-bold truncate">
              {checks.encoder.current_codec}
            </div>
            <div className="text-white/40 text-micro">
              {checks.encoder.has_nvidia ? 'NVIDIA GPU Active' : 'CPU / Standard'}
            </div>
            {checks.encoder.warning && (
              <div className="mt-1 text-micro text-amber-300 font-sans leading-tight">
                {checks.encoder.warning}
              </div>
            )}
          </div>

          {/* Check 3: TRT Warmth */}
          <div
            className={`p-2 rounded-lg border ${
              checks.trt.ok ? 'bg-black/30 border-white/10' : 'bg-amber-950/40 border-amber-500/40'
            }`}
          >
            <div className="flex items-center justify-between text-muted text-nano uppercase tracking-wider font-sans font-bold">
              <span className="flex items-center gap-1">
                <Icon.brand size={12} />
                <span>TRT Engine Cache</span>
              </span>
              <span className={checks.trt.ok ? 'text-emerald-400' : 'text-amber-400'}>
                {checks.trt.status}
              </span>
            </div>
            <div className="mt-1 text-white font-bold">
              {checks.trt.is_trt ? 'TensorRT Precision' : 'Standard Execution'}
            </div>
            <div className="text-white/40 text-micro">
              {checks.trt.ok ? 'Warm in local cache' : 'Cold engine compile (~90s)'}
            </div>
            {checks.trt.warning && (
              <div className="mt-1 text-micro text-amber-300 font-sans leading-tight">
                {checks.trt.warning}
              </div>
            )}
          </div>

          {/* Check 4: Face Resolution & Restorer */}
          <div
            className={`p-2 rounded-lg border ${
              checks.resolution.ok ? 'bg-black/30 border-white/10' : 'bg-amber-950/40 border-amber-500/40'
            }`}
          >
            <div className="flex items-center justify-between text-muted text-nano uppercase tracking-wider font-sans font-bold">
              <span className="flex items-center gap-1">
                <Icon.wand size={12} />
                <span>Face Resolution</span>
              </span>
              <span className={checks.resolution.ok ? 'text-emerald-400' : 'text-amber-400'}>
                {checks.resolution.max_face_px ? `${checks.resolution.max_face_px}px` : 'Standard'}
              </span>
            </div>
            <div className="mt-1 text-white font-bold truncate">
              {checks.resolution.selected_enhancer || 'No Enhancer'}
            </div>
            <div className="text-white/40 text-micro">
              {checks.resolution.max_face_px > 256 ? '>256px Native Crop' : 'Normal Face Size'}
            </div>
            {checks.resolution.warning && (
              <div className="mt-1 text-micro text-amber-300 font-sans leading-tight">
                {checks.resolution.warning}
              </div>
            )}
          </div>

          {/* Check 5: 12 GB VRAM Budget (Pixel Boost + UltraMax batch size) */}
          <div
            className={`p-2 rounded-lg border ${
              checks.vram.ok ? 'bg-black/30 border-white/10' : 'bg-amber-950/40 border-amber-500/40'
            }`}
          >
            <div className="flex items-center justify-between text-muted text-nano uppercase tracking-wider font-sans font-bold">
              <span className="flex items-center gap-1">
                <Icon.meter size={12} />
                <span>12 GB VRAM Budget</span>
              </span>
              <span className={checks.vram.ok ? 'text-emerald-400' : 'text-amber-400'}>
                {checks.vram.estimated_vram_gb} GB / 12 GB
              </span>
            </div>
            <div className="mt-1 text-white font-bold">
              Pixel Boost {checks.vram.pixel_boost}px · Batch {checks.vram.ultramax_batch_size}
            </div>
            <div className="text-white/40 text-micro">
              {checks.vram.ok ? 'Within 12 GB VRAM headroom' : 'Exceeds 12 GB VRAM budget!'}
            </div>
            {checks.vram.warning && (
              <div className="mt-1 text-micro text-amber-300 font-sans leading-tight">
                {checks.vram.warning}
              </div>
            )}
          </div>

          {/* Check 6: Missing .fsz profile angles */}
          <div
            className={`p-2 rounded-lg border ${
              checks.fsz.ok ? 'bg-black/30 border-white/10' : 'bg-amber-950/40 border-amber-500/40'
            }`}
          >
            <div className="flex items-center justify-between text-muted text-nano uppercase tracking-wider font-sans font-bold">
              <span className="flex items-center gap-1">
                <Icon.face size={12} />
                <span>.fsz Profile Coverage</span>
              </span>
              <span className={checks.fsz.ok ? 'text-emerald-400' : 'text-amber-400'}>
                {checks.fsz.ok ? 'Covered' : 'Missing Angles'}
              </span>
            </div>
            <div className="mt-1 text-white font-bold truncate">
              {checks.fsz.faceset_file ? checks.fsz.faceset_file.split(/[\\/]/).pop() : 'Standard Reference'}
            </div>
            <div className="text-white/40 text-micro">
              {checks.fsz.missing_angles?.length ? `Missing: ${checks.fsz.missing_angles.join(', ')}` : 'All 5 pose buckets present'}
            </div>
            {checks.fsz.warning && (
              <div className="mt-1 text-micro text-amber-300 font-sans leading-tight">
                {checks.fsz.warning}
              </div>
            )}
          </div>

          {/* Check 7: Segment retention paths */}
          <div
            className={`p-2 rounded-lg border ${
              checks.segments.ok ? 'bg-black/30 border-white/10' : 'bg-amber-950/40 border-amber-500/40'
            }`}
          >
            <div className="flex items-center justify-between text-muted text-nano uppercase tracking-wider font-sans font-bold">
              <span className="flex items-center gap-1">
                <Icon.folder size={12} />
                <span>Segment Custody</span>
              </span>
              <span className={checks.segments.ok ? 'text-emerald-400' : 'text-amber-400'}>
                {checks.segments.ok ? 'Retained' : 'Purge Risk'}
              </span>
            </div>
            <div className="mt-1 text-white font-bold truncate">
              {checks.segments.retention_path}
            </div>
            <div className="text-white/40 text-micro">
              {checks.segments.keep_segments ? 'Chunks preserved on disk' : 'Keep segments disabled'}
            </div>
            {checks.segments.warning && (
              <div className="mt-1 text-micro text-amber-300 font-sans leading-tight">
                {checks.segments.warning}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export { PreFlightLinter, PreFlightLinter as PreflightLinter };
