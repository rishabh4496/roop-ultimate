import React, { useEffect, useRef, useState } from 'react';
import { AlertTriangle, CheckCircle2, Loader2, Wand2, RefreshCw } from 'lucide-react';
import { postJSON } from '@engine/api';
import { useStudio } from '../engine/studio';
import { inOutOf } from '../engine/payload';
import { notify } from '../ui/controls';

// Pre-flight: what the engine itself says about this render before it starts (POST /api/render/preflight):
// free disk vs the output's footprint, a CPU encoder on an NVIDIA GPU, TensorRT engines that will
// compile first (~90 s), and a >256 px face with Restore Ultra off. Advisory: Start is never blocked.
// Two warnings come with a one-click fix; each sets the same settings React UI 1.0's fix does.

const FIXES = {
  switch_nvenc: (fix) => ({ output_video_codec: fix.codec || 'h264_nvenc' }),
  enable_restore_ultra: (fix) => ({ selected_enhancer: fix.enhancer || 'Restore Ultra', restore_ultra_frequency_blend: true, auto_bypass_restorer_native_swap: false }),
};
const FIX_FOR = { encoder: 'switch_nvenc', resolution: 'enable_restore_ultra' };

export default function Preflight() {
  const target = useStudio((s) => s.targets[s.selectedTarget]);
  const codec = useStudio((s) => s.settings?.output_video_codec);
  const enhancer = useStudio((s) => s.settings?.selected_enhancer);
  const faces = useStudio((s) => s.targetFacesInfo);   // re-check after a capture changes the faces
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [nonce, setNonce] = useState(0);
  const seq = useRef(0);
  const { a, b } = inOutOf(target);
  const frames = target ? b - a + 1 : 0;

  useEffect(() => {
    if (!target) { setRes(null); return undefined; }
    const my = ++seq.current;
    const t = setTimeout(async () => {
      setBusy(true);
      try {
        const r = await postJSON('/api/render/preflight', {
          // The engine measures the captured faces itself (_target_faces_info), so no face size is sent.
          max_frames: frames, batch_count: 1, codec, selected_enhancer: enhancer,
        });
        if (my === seq.current) setRes(r);
      } catch { if (my === seq.current) setRes(null); } finally { if (my === seq.current) setBusy(false); }
    }, 500);
    return () => clearTimeout(t);
  }, [target?.media_id, target?.id, frames, codec, enhancer, faces, nonce]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!target || (!res && !busy)) return null;
  const warnings = Array.isArray(res?.warnings) ? res.warnings : [];
  const fixes = Array.isArray(res?.auto_fixes) ? res.auto_fixes : [];
  const apply = (fix) => {
    const patch = FIXES[fix.action]?.(fix);
    if (!patch) return;
    useStudio.getState().setSettings(patch);
    notify(fix.label || 'Fixed');
  };

  return (
    <div className="mb-3 rounded-xl border border-[var(--rule)] bg-black/25 p-2.5" data-testid="preflight">
      <div className="flex items-center gap-2 font-mono text-[10.5px] uppercase tracking-wide text-[var(--fg-faint)]">
        Pre-flight
        {busy ? <Loader2 size={12} className="animate-spin" /> : warnings.length ? <span style={{ color: 'var(--warn)' }}>{warnings.length} note{warnings.length === 1 ? '' : 's'}</span>
          : <span className="flex items-center gap-1" style={{ color: 'var(--ok)' }}><CheckCircle2 size={12} /> ready</span>}
        <button type="button" className="btn btn-ghost btn-sm btn-icon ml-auto !h-6 !w-6" aria-label="Re-run the pre-flight check" title="Re-run" onClick={() => setNonce((n) => n + 1)}><RefreshCw size={12} /></button>
      </div>
      {warnings.length > 0 && (
        <ul className="mt-1.5 space-y-1.5">
          {warnings.map((w, i) => {
            const fix = fixes.find((f) => f.action === FIX_FOR[w.check]);
            return (
              <li key={`${w.check}-${i}`} className="flex items-start gap-2 text-[12px] text-[var(--fg-dim)]" data-testid={`preflight-${w.check}`}>
                <AlertTriangle size={13} className="mt-0.5 shrink-0 text-[var(--warn)]" />
                <span className="min-w-0 flex-1">{w.message}</span>
                {fix && FIXES[fix.action] && (
                  <button type="button" className="btn btn-sm shrink-0" onClick={() => apply(fix)} data-testid={`preflight-fix-${fix.action}`}><Wand2 size={12} /> {fix.label}</button>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
