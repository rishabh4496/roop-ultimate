import { useEffect } from 'react';
import { API } from '../api';
import { frameSocket } from '../transport/frameSocket';
import { decodeFrame, releaseFrame } from '../components/faceswap/frameDecoder';
import { useTelemetryStore } from '../store/telemetryStore';
import { usePipStore } from '../store/pipStore';

// Feeds the floating preview with the frames a running render publishes - the same two routes the Processing tab's live view
// uses (components/faceswap/LiveProcessingPeek.jsx), so the window and that view can never disagree about what "live" is:
//
//   /ws/frames   JPEG bytes pushed the moment they are published. The subscriber COPIES them before decoding: the socket hands
//                one buffer to every subscriber and a decode transfers (detaches) its input (see FastCanvasPlayer).
//   HTTP         /api/live_frame?seq=N, fetched when telemetry's live_seq moves - ONLY while the socket is not open, so a
//                connected socket is never doubled by polling.
//
// It runs only while `active` (window on screen AND a render in flight AND not frozen), so a hidden or paused window asks the
// server for nothing. The server counts a live subscriber as a viewer and keeps publishing at the watched cadence; releasing
// the subscription when the window hides lets it stop.

export function usePipLiveFeed(active) {
  useEffect(() => {
    if (!active) return undefined;
    let alive = true;
    let decoding = false;
    let lastSeq = useTelemetryStore.getState().run?.live_seq || 0;

    const offer = (bitmap, frame) => {
      if (!alive) { releaseFrame(bitmap); return; }
      usePipStore.getState().offerLive(bitmap, { frame });
    };

    const offBytes = frameSocket.subscribeLive((f) => {
      if (!f?.bytes || !f.bytes.byteLength || decoding) return;     // one decode in flight: a newer frame replaces a late one
      decoding = true;
      decodeFrame(f.bytes.slice(0), { mime: 'image/jpeg' })
        .then((bmp) => offer(bmp, f.frame))
        .catch(() => { /* a frame that does not decode is skipped: the next one replaces it */ })
        .finally(() => { decoding = false; });
    });

    const offSeq = useTelemetryStore.subscribe((state) => {
      const seq = state.run?.live_seq || 0;
      if (seq === lastSeq) return;
      lastSeq = seq;
      if (!seq || frameSocket.isOpen() || decoding) return;
      decoding = true;
      decodeFrame(`${API}/api/live_frame?seq=${seq}`)
        .then((bmp) => offer(bmp, seq))
        .catch(() => { /* 204 before the first frame, or a reset between poll and fetch */ })
        .finally(() => { decoding = false; });
    });

    return () => { alive = false; offBytes(); offSeq(); };
  }, [active]);
}
