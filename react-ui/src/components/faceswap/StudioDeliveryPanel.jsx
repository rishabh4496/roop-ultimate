import React, { useEffect, useMemo, useState } from 'react';
import { useApi } from '../../useApi';
import { Section, Toggle } from '../ui';

function saveText(text, filename, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function frameNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(0, Math.floor(n)) : fallback;
}

export default function StudioDeliveryPanel({
  target,
  output,
  fps = 25,
  markers = [],
  trimIn = 0,
  trimOut = 0,
  segmentBlocks = [],
  settings = {},
  onSettingsChange,
  previewMaskDebug = null,
  onPreviewMattes,
  previewBusy = false,
  notify,
}) {
  const { getJSON, postJSON } = useApi();
  const [busy, setBusy] = useState('');
  const [outputPath, setOutputPath] = useState('');
  const [verification, setVerification] = useState(null);
  const [receipt, setReceipt] = useState(null);

  useEffect(() => {
    let live = true;
    getJSON('/api/provenance/receipt', { abortOnUnmount: true })
      .then((result) => { if (live) setReceipt(result); })
      .catch(() => { if (live) setReceipt(null); });
    return () => { live = false; };
  }, [getJSON]);

  const timelineDocument = useMemo(() => {
    const totalFrames = Math.max(1, frameNumber(target?.frames, trimOut + 1));
    const start = Math.min(frameNumber(trimIn, 0), totalFrames - 1);
    const end = Math.min(totalFrames, Math.max(start + 1, frameNumber(trimOut, totalFrames - 1) + 1));
    const activeMarkers = markers.map(({ id, frame, label, color, note }) => ({
      id, frame: frameNumber(frame, 0), label: label || 'Marker', color: color || '', note: note || '',
    })).filter((marker) => marker.frame >= start && marker.frame < end);
    const segments = segmentBlocks
      .filter((block) => block?.kind === 'segment')
      .map((block, index) => {
        const inFrame = Math.max(start, frameNumber(block.start, start));
        const outFrame = Math.min(end, frameNumber(block.end, inFrame) + 1);
        return outFrame > inFrame ? {
          in: inFrame,
          out: outFrame,
          face_id: block.label || `segment_${index + 1}`,
          speaker_cluster_id: block.label || `segment_${index + 1}`,
          clip_name: target?.name || 'Face-swapped render',
        } : null;
      })
      .filter(Boolean);
    return {
      name: (target?.name || 'Roop_Ultimate_Timeline').replace(/\.[^/.]+$/, ''),
      source_media: target?.path || target?.filename || target?.name || 'source_reel.mp4',
      segments_use_source: true,
      total_frames: end,
      fps,
      markers: activeMarkers,
      segments,
      timeline: { fps, frame_start: start, frame_end: end, face_segments: segments,
        markers: activeMarkers },
    };
  }, [target, fps, markers, trimIn, trimOut, segmentBlocks]);

  const exportInterchange = async (format) => {
    if (!target) { notify?.('Load a target video before exporting the timeline', 'error'); return; }
    setBusy(format);
    try {
      const body = { document: timelineDocument, target_media_id: target.media_id || target.id, fps };
      const result = format === 'edl' ? await postJSON('/api/timeline/export_edl', body)
        : await postJSON('/api/timeline/export_otio', body);
      const text = format === 'edl' ? result.edl : result.otio;
      if (!text) throw new Error(`The backend returned no ${format.toUpperCase()} document`);
      const stem = timelineDocument.name || 'roop_timeline';
      saveText(text, result.filename || `${stem}.${format === 'edl' ? 'edl' : 'otio'}`, format === 'edl' ? 'text/plain' : 'application/json');
      notify?.(`Exported ${format === 'edl' ? 'CMX3600 EDL' : 'OpenTimelineIO'} with ${timelineDocument.markers.length} markers, trim range and ${timelineDocument.segments.length} segment cuts`);
    } catch (error) {
      notify?.(error.message, 'error');
    } finally { setBusy(''); }
  };

  const verifyOutput = async () => {
    const path = outputPath.trim() || output?.absolute_path || output?.path;
    if (!path) { notify?.('Enter the rendered output path to verify', 'error'); return; }
    setBusy('verify');
    try {
      const result = await postJSON('/api/provenance/verify', { file_path: path }, { timeout: 0, abortOnUnmount: true });
      setVerification(result);
      notify?.(result.verified ? 'Provenance signature verified' : `Provenance check failed: ${result.reason}`, result.verified ? 'success' : 'error');
    } catch (error) {
      setVerification({ verified: false, reason: error.message });
      notify?.(error.message, 'error');
    } finally { setBusy(''); }
  };

  const matteCards = [
    ['xseg3', 'XSeg 3 occlusion matte', settings.vfx_export_xseg3 !== false, 'xseg3_matte'],
    ['face_parser', 'Face parsing matte', settings.vfx_export_face_parser !== false, 'parsing_matte'],
    ['difference', 'Mask difference matte', !!settings.vfx_export_difference, 'difference_matte'],
  ];
  const patchSetting = (key, value) => onSettingsChange?.({ [key]: value });
  const manifest = verification?.manifest;

  return (
    <Section title="Studio delivery & interchange" collapsible defaultOpen={false}>
      <div className="space-y-4">
        <div className="rounded-xl border border-white/10 bg-black/20 p-3" data-testid="vfx-matte-export">
          <div className="mb-2 flex flex-wrap items-start justify-between gap-2">
            <div>
              <div className="text-mini font-bold text-white/85">Standalone VFX matte pass</div>
              <p className="mb-0 mt-1 text-micro text-muted">Export selected masks as grayscale PNG sequences beside the beauty render.</p>
            </div>
            <Toggle label="Enable matte export" checked={!!settings.vfx_multi_pass}
              onChange={(value) => patchSetting('vfx_multi_pass', value)} />
          </div>
          {settings.vfx_multi_pass && (
            <>
              <Toggle label="Export matte sequences" checked={settings.vfx_export_matte !== false}
                onChange={(value) => patchSetting('vfx_export_matte', value)} />
              <button type="button" onClick={onPreviewMattes} disabled={!target || previewBusy}
                className="mt-2 rounded-lg border border-white/15 px-2.5 py-1.5 text-micro font-bold text-white/80 hover:bg-white/10 disabled:opacity-40">
                {previewBusy ? 'Preview busy' : 'Preview current frame mattes'}
              </button>
              <div className="mt-2 grid gap-2 md:grid-cols-3">
                {matteCards.map(([key, label, enabled, channel]) => (
                  <div key={key} className={`overflow-hidden rounded-lg border ${enabled ? 'border-cyan-300/20 bg-cyan-950/10' : 'border-white/10 bg-white/[0.02]'}`}>
                    <div className="aspect-video bg-black/40">
                      {channel && previewMaskDebug?.[channel]
                        ? <img src={previewMaskDebug[channel]} alt={`${label} current-frame mask preview`} className="h-full w-full object-contain" />
                        : <div className="flex h-full items-center justify-center px-3 text-center text-micro text-white/35">{key === 'difference' ? 'Preview appears when XSeg 3 and Face Parser channels are both active.' : 'Enable mask channel diagnostics to preview the current frame.'}</div>}
                    </div>
                    <div className="flex items-center justify-between gap-2 p-2">
                      <span className="text-micro font-semibold text-white/75">{label}</span>
                      <input type="checkbox" checked={enabled} aria-label={`Export ${label}`}
                        onChange={(event) => patchSetting(`vfx_export_${key === 'face_parser' ? 'face_parser' : key}`, event.target.checked)}
                        className="h-3.5 w-3.5 accent-cyan-300" />
                    </div>
                  </div>
                ))}
              </div>
              <p className="mb-0 mt-2 text-micro text-white/45">The difference pass shows where XSeg 3 and Face Parser disagree. The manifest lists channels the render could produce.</p>
            </>
          )}
        </div>

        <div className="rounded-xl border border-white/10 bg-black/20 p-3" data-testid="nle-interchange">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <div>
              <div className="text-mini font-bold text-white/85">NLE timeline interchange</div>
              <div className="text-micro text-muted">{timelineDocument.markers.length} markers · frames {timelineDocument.timeline.frame_start}–{timelineDocument.timeline.frame_end} · {timelineDocument.segments.length} segment cuts</div>
            </div>
            <div className="flex gap-2">
              <button type="button" onClick={() => exportInterchange('edl')} disabled={!!busy || !target}
                className="rounded-lg border border-white/15 bg-white/5 px-3 py-1.5 text-micro font-bold text-white/80 hover:bg-white/10 disabled:opacity-40">{busy === 'edl' ? 'Exporting…' : 'Export EDL'}</button>
              <button type="button" onClick={() => exportInterchange('otio')} disabled={!!busy || !target}
                className="rounded-lg border border-cyan-300/20 bg-cyan-400/10 px-3 py-1.5 text-micro font-bold text-cyan-100 hover:bg-cyan-400/20 disabled:opacity-40">{busy === 'otio' ? 'Exporting…' : 'Export OTIO'}</button>
            </div>
          </div>
          <p className="mb-0 text-micro text-white/40">Exports CMX3600 and OpenTimelineIO with the active in/out range, chapter markers, and queued segment cuts.</p>
        </div>

        <div className="rounded-xl border border-white/10 bg-black/20 p-3" data-testid="provenance-inspector">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <div>
              <div className="text-mini font-bold text-white/85">Cryptographic provenance</div>
              <div className={`mt-1 inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-micro font-bold ${receipt?.enabled ? 'border-emerald-400/20 bg-emerald-400/10 text-emerald-200' : 'border-amber-300/20 bg-amber-300/10 text-amber-100'}`}>
                <span className={`h-1.5 w-1.5 rounded-full ${receipt?.enabled ? 'bg-emerald-300' : 'bg-amber-300'}`} />
                {receipt?.enabled ? 'Local signer enabled' : 'Signer disabled'}
              </div>
            </div>
            <button type="button" disabled={busy === 'verify'} onClick={verifyOutput}
              className="rounded-lg border border-white/15 bg-white/5 px-3 py-1.5 text-micro font-bold text-white/80 hover:bg-white/10 disabled:opacity-40">{busy === 'verify' ? 'Verifying…' : 'Verify output'}</button>
          </div>
          <label className="block text-micro text-white/50">Rendered output path
            <input value={outputPath} onChange={(event) => { setOutputPath(event.target.value); setVerification(null); }} placeholder={output?.absolute_path || output?.path || 'Paste a path from the render folder'}
              className="mt-1 w-full rounded-lg border border-white/10 bg-black/30 px-2.5 py-2 font-mono text-micro text-white/80 outline-none focus:border-cyan-300/40" />
          </label>
          <div className="mt-2 grid gap-1 text-micro text-white/45 sm:grid-cols-2">
            <div>Schema <span className="font-mono text-white/75">{receipt?.schema || '—'}</span></div>
            <div>Hardware receipt <span className="font-mono text-white/75">{receipt?.receipt_hash?.slice(0, 18) || '—'}{receipt?.receipt_hash ? '…' : ''}</span></div>
          </div>
          {verification && (
            <div className={`mt-2 rounded-lg border p-2 ${verification.verified ? 'border-emerald-300/20 bg-emerald-400/5' : 'border-rose-300/20 bg-rose-400/5'}`} role="status">
              <div className={`text-mini font-bold ${verification.verified ? 'text-emerald-200' : 'text-rose-200'}`}>{verification.file_verified ? 'Output hash and signature verified' : verification.verified ? 'Embedded signature verified · file hash unchecked' : `Verification failed · ${verification.reason || 'unverified output'}`}</div>
              {manifest && <dl className="mt-1 grid gap-x-3 gap-y-0.5 text-micro sm:grid-cols-[auto_1fr]">
                <dt className="text-white/40">Payload SHA-256</dt><dd className="break-all font-mono text-white/70">{manifest.payload_sha256}</dd>
                <dt className="text-white/40">Signature</dt><dd className="break-all font-mono text-white/70">{manifest.signature}</dd>
                <dt className="text-white/40">Timestamp</dt><dd className="font-mono text-white/70">{manifest.timestamp}</dd>
                <dt className="text-white/40">Disclosure</dt><dd className="text-white/70">{manifest.synthetic_media_disclosure}</dd>
              </dl>}
            </div>
          )}
        </div>
      </div>
    </Section>
  );
}
