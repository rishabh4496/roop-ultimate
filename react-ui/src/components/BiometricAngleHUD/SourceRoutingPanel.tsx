// Pose-adaptive source routing for the source mapped to the selected person.
//
// Builds the SOURCE person's angle portfolio from their own faceset (never the
// target's: conditioning the swap on the person being replaced re-renders them
// onto themselves). From the next render on, each target face is swapped with:
//   |yaw| > 35 deg and a matching source profile: 0.7 x profile + 0.3 x fused
//   otherwise:                                    the fused frontal/quarter vector
// The target's per-frame pose comes from the angle scan above (a frame lookup
// table); frames it did not sample fall back to the nearest scanned frame, then
// to the face's own keypoints.
import React, { useCallback, useEffect, useState } from 'react';
import { getJSON, postJSON } from '../../api.js';

interface PortfolioSummary {
  bins: Record<string, { face_index: number; yaw: number; pitch_up: number; score: number }>;
  profile_left: boolean;
  profile_right: boolean;
  fused_sources: string[];
  dim: number;
  rejected: Record<string, number>;
}

interface StatusResponse {
  source_index: number;
  faces?: number;
  portfolio: PortfolioSummary | null;
  frame_lut: { available: boolean; frames?: number; step?: number; media_path?: string };
}

const CELLS: [string, string][] = [
  ['BIN_5_PROFILE_LEFT', 'PL'], ['BIN_3_HALF_PROFILE_LEFT', 'HL'], ['BIN_1_QUARTER_LEFT', 'QL'],
  ['BIN_0_FRONTAL', 'F'], ['BIN_2_QUARTER_RIGHT', 'QR'], ['BIN_4_HALF_PROFILE_RIGHT', 'HR'],
  ['BIN_6_PROFILE_RIGHT', 'PR'], ['BIN_7_PITCH_UP', 'UP'], ['BIN_8_PITCH_DOWN', 'DN'],
];

const errorOf = (err: unknown): string => (err instanceof Error ? err.message : String(err));

export interface SourceRoutingPanelProps {
  sourceIndex: number | null;
  sourceLabel?: string;
  // Changes whenever a target scan lands (its cache key): that scan is what
  // publishes the frame LUT this panel reports, so the status is re-read then.
  refreshKey?: string | null;
}

export default function SourceRoutingPanel({ sourceIndex, sourceLabel, refreshKey }: SourceRoutingPanelProps) {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async (): Promise<void> => {
    if (sourceIndex == null) { setStatus(null); return; }
    try {
      setStatus(await getJSON(`/api/angle-scan/source-portfolio?index=${sourceIndex}`, { timeout: 8000 }));
      setError(null);
    } catch (err) {
      setStatus(null);
      setError(errorOf(err));
    }
  }, [sourceIndex]);

  useEffect(() => { void refresh(); }, [refresh, refreshKey]);

  const act = async (path: string): Promise<void> => {
    if (sourceIndex == null) return;
    setBusy(true);
    try {
      setStatus(await postJSON(path, { source_index: sourceIndex }));
      setError(null);
    } catch (err) {
      setError(errorOf(err));
    } finally {
      setBusy(false);
    }
  };

  if (sourceIndex == null) {
    return (
      <div className="rounded-lg border border-slate-700/50 p-2 font-mono text-[10px] text-slate-400" data-testid="source-routing">
        Pose routing: map a source face to this person first.
      </div>
    );
  }

  const pf = status?.portfolio || null;
  const lut = status?.frame_lut;
  return (
    <div className="rounded-lg border border-cyan-400/15 bg-slate-900/50 p-2 space-y-1.5" data-testid="source-routing"
         data-active={pf ? 'true' : 'false'}>
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-cyan-300/80">Pose routing</span>
        <span className="font-mono text-[10px] text-slate-400 truncate">
          {sourceLabel || `source #${sourceIndex + 1}`}{status?.faces != null ? ` · ${status.faces} faces` : ''}
        </span>
      </div>
      {pf ? (
        <>
          <div className="grid grid-cols-9 gap-0.5" aria-label="Source angles available">
            {CELLS.map(([name, short]) => (
              <span key={name} data-bin={name} data-has={pf.bins[name] ? 'true' : 'false'}
                    title={pf.bins[name] ? `${name}: source face ${pf.bins[name].face_index + 1}, yaw ${pf.bins[name].yaw}°` : `${name}: none`}
                    className={`text-center rounded font-mono text-[8px] py-0.5 ${pf.bins[name]
                      ? 'bg-cyan-400/20 text-cyan-100' : 'border border-dashed border-slate-600 text-slate-500'}`}>
                {short}
              </span>
            ))}
          </div>
          <div className="font-mono text-[10px] text-slate-400 leading-snug">
            {pf.profile_left && pf.profile_right
              ? 'Profile frames use 0.7 × source profile + 0.3 × fused; others the fused vector.'
              : `No source ${!pf.profile_left && !pf.profile_right ? 'profiles' : (pf.profile_left ? 'right profile' : 'left profile')}: `
                + 'those frames use the fused vector.'}
          </div>
        </>
      ) : (
        <div className="font-mono text-[10px] text-slate-400">
          Off: every frame uses this source's standard reference.
        </div>
      )}
      <div className="font-mono text-[10px] text-slate-500">
        {lut?.available
          ? `target poses: ${lut.frames} scanned frames (every ${lut.step})`
          : 'target poses: none yet (scan above) - render will read pose per face'}
      </div>
      {error && <div role="alert" className="text-[10px] text-rose-200">{error}</div>}
      <div className="flex gap-1.5">
        <button type="button" disabled={busy} onClick={() => act('/api/angle-scan/source-portfolio')}
                className="flex-1 rounded border border-cyan-400/30 hover:bg-cyan-400/10 font-mono text-[10px] uppercase tracking-wider py-1 text-cyan-200 disabled:opacity-40">
          {pf ? 'Rebuild' : 'Enable from source faceset'}
        </button>
        {pf && (
          <button type="button" disabled={busy} onClick={() => act('/api/angle-scan/source-portfolio/clear')}
                  className="rounded border border-slate-600 hover:border-slate-400 font-mono text-[10px] uppercase tracking-wider px-2 py-1 text-slate-300 disabled:opacity-40">
            Off
          </button>
        )}
      </div>
    </div>
  );
}
