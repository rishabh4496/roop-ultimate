import { useEffect, useState } from 'react';

// The waveform of a BOUND external audio track (the file /api/lipsync/audio/add returned), for the timeline's audio row.
// Same envelope as the target's own (`GET /api/audio/waveform` answers from the identical cached decode), asked again while
// it is still building. Coarse only: one request for the whole track; the row is a monitor for the slip, not a scrub surface.

const decodePeaks = (b64) => {
  if (!b64) return new Uint8Array(0);
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
};

export default function useAudioTrackWave(getJSON, trackPath, { retries = 40, delayMs = 1500 } = {}) {
  const [state, setState] = useState({ coarse: null, status: trackPath ? 'loading' : 'idle' });
  useEffect(() => {
    if (!trackPath) { setState({ coarse: null, status: 'idle' }); return undefined; }
    setState({ coarse: null, status: 'loading' });
    let cancelled = false;
    let timer = null;
    let tries = 0;
    const attempt = async () => {
      try {
        const res = await getJSON(`/api/audio/waveform?path=${encodeURIComponent(trackPath)}&bins=2048`);
        if (cancelled) return;
        if (res.state === 'none') { setState({ coarse: null, status: 'none' }); return; }
        if (!res.ready) {
          if (++tries > retries) { setState({ coarse: null, status: 'error' }); return; }
          timer = setTimeout(attempt, delayMs);
          return;
        }
        setState({ coarse: { t0: 0, t1: res.duration, peaks: decodePeaks(res.peaks) }, status: 'ready' });
      } catch {
        if (!cancelled) setState({ coarse: null, status: 'error' });
      }
    };
    attempt();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [getJSON, trackPath, retries, delayMs]);
  return state;
}
