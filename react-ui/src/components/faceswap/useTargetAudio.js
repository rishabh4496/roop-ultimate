import { useEffect, useState } from 'react';

// Does the selected target have an audio stream? true / false / null (not known yet).
//
// Asked of the same waveform envelope the timeline draws (`GET /api/target/waveform`, built once per file by a background
// decode and shared), so asking costs nothing extra. `state: 'none'` is a file with no audio; `building` is asked again.
// An image, a missing target and an error are all "not known" (null) rather than a guess either way.

export function audioPresence(res) {
  if (!res || typeof res !== 'object') return null;
  if (res.state === 'none') return false;
  if (res.state === 'ready' || res.ready === true) return true;
  return null;
}

export default function useTargetAudio(getJSON, targetIndex, enabled = true, { retries = 40, delayMs = 1500 } = {}) {
  const [has, setHas] = useState(null);
  useEffect(() => {
    setHas(null);
    if (!enabled || targetIndex === null || targetIndex === undefined || targetIndex < 0) return undefined;
    let cancelled = false;
    let timer = null;
    let tries = 0;
    const attempt = async () => {
      try {
        const res = await getJSON(`/api/target/waveform?index=${targetIndex}&bins=64`);
        if (cancelled) return;
        const seen = audioPresence(res);
        if (seen !== null) { setHas(seen); return; }
        if (++tries <= retries) timer = setTimeout(attempt, delayMs);
      } catch { /* unknown stays unknown */ }
    };
    attempt();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [getJSON, targetIndex, enabled, retries, delayMs]);
  return has;
}
