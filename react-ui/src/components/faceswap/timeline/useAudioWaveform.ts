/**
 * useAudioWaveform.ts: Hook for fetching downsampled audio peaks from /api/media/waveform.
 * 100% Offline, cached peak envelope for A1 Audio Waveform Lane.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { getJSON } from '../../../api';

export interface WaveformData {
  peaks: number[];
  duration: number;
  buckets: number;
  t0: number;
  t1: number;
  state: 'idle' | 'loading' | 'ready' | 'none' | 'error';
}

export interface UseAudioWaveformOptions {
  targetIndex?: number | null;
  mediaId?: string | null;
  targetKey?: string;
  buckets?: number;
  enabled?: boolean;
}

export function useAudioWaveform({
  targetIndex = null,
  mediaId = null,
  targetKey = '',
  buckets = 2000,
  enabled = true,
}: UseAudioWaveformOptions) {
  const [data, setData] = useState<WaveformData>({
    peaks: [],
    duration: 0,
    buckets: 0,
    t0: 0,
    t1: -1,
    state: 'idle',
  });

  const aliveRef = useRef(true);
  const cacheKeyRef = useRef('');

  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
    };
  }, []);

  const load = useCallback(async () => {
    if (!enabled) return;
    const hasTarget =
      (targetIndex !== null && targetIndex !== undefined && targetIndex >= 0) ||
      Boolean(mediaId);

    if (!hasTarget) {
      setData({
        peaks: [],
        duration: 0,
        buckets: 0,
        t0: 0,
        t1: -1,
        state: 'none',
      });
      return;
    }

    const currentKey = `${targetIndex ?? ''}_${mediaId ?? ''}_${targetKey}_${buckets}`;
    if (cacheKeyRef.current === currentKey && data.state === 'ready') {
      return;
    }

    setData((prev) => ({ ...prev, state: 'loading' }));

    let tries = 0;
    const maxTries = 30;

    const query = mediaId
      ? `media_id=${encodeURIComponent(mediaId)}&buckets=${buckets}&wait=5`
      : `index=${targetIndex}&buckets=${buckets}&wait=5`;

    const fetchStep = async () => {
      if (!aliveRef.current) return;
      try {
        const res = await getJSON(`/api/media/waveform?${query}`);
        if (!aliveRef.current) return;

        if (res.state === 'none') {
          setData({
            peaks: [],
            duration: 0,
            buckets: 0,
            t0: 0,
            t1: -1,
            state: 'none',
          });
          return;
        }

        if (res.state === 'building' || (!res.ready && !res.peaks)) {
          if (++tries > maxTries) {
            setData((prev) => ({ ...prev, state: 'error' }));
            return;
          }
          setTimeout(fetchStep, 1000);
          return;
        }

        let peaksArr: number[] = [];
        if (Array.isArray(res.peaks)) {
          peaksArr = res.peaks;
        } else if (typeof res.peaks === 'string') {
          const bin = atob(res.peaks);
          for (let i = 0; i < bin.length; i++) {
            peaksArr.push(bin.charCodeAt(i));
          }
        }

        cacheKeyRef.current = currentKey;
        setData({
          peaks: peaksArr,
          duration: Number(res.duration) || 0,
          buckets: Number(res.buckets) || peaksArr.length,
          t0: Number(res.t0) || 0,
          t1: Number(res.t1) || -1,
          state: 'ready',
        });
      } catch {
        // Fallback to /api/target/waveform if /api/media/waveform failed
        if (targetIndex !== null && targetIndex >= 0) {
          try {
            const fallbackRes = await getJSON(
              `/api/target/waveform?index=${targetIndex}&bins=${buckets}`
            );
            if (!aliveRef.current) return;

            if (fallbackRes.ready && fallbackRes.peaks) {
              const bin = atob(fallbackRes.peaks);
              const peaksArr: number[] = [];
              for (let i = 0; i < bin.length; i++) {
                peaksArr.push(bin.charCodeAt(i));
              }
              cacheKeyRef.current = currentKey;
              setData({
                peaks: peaksArr,
                duration: Number(fallbackRes.duration) || 0,
                buckets: peaksArr.length,
                t0: 0,
                t1: Number(fallbackRes.duration) || -1,
                state: 'ready',
              });
              return;
            }
          } catch {
            // keep error state below
          }
        }

        if (aliveRef.current) {
          setData((prev) => ({ ...prev, state: 'error' }));
        }
      }
    };

    fetchStep();
  }, [enabled, targetIndex, mediaId, targetKey, buckets, data.state]);

  useEffect(() => {
    load();
  }, [load]);

  return {
    ...data,
    reload: load,
  };
}
