import { useEffect, useRef, useState } from 'react';
import { playChime, playFailTone, notifyDesktop } from './utils';

// ── "Your render finished" ────────────────────────────────────────────────
// A sound always, plus an OS notification when the user has asked for one.
// Fires on the processing -> idle edge, so it cannot repeat while a run is in
// flight or announce a run that was already finished when the tab mounted.
//
// The edge alone does not say whether the run SUCCEEDED. It used to be treated
// as if it did: a four-hour render that died at 90%, or one stopped by hand,
// got the same rising three-note chime and "Render Complete! Your face swap
// processing run has finished." The one moment the alert exists for — you are
// not looking at the screen — is exactly when that reads as good news, so the
// failure went unnoticed until you came back and found no output. The error is
// read at the edge and the outcome drives both the sound and the wording.

const KEY = 'roop_desktop_alerts';

export default function useRunCompleteAlert({ processing, error, notify }) {
  // Persisted, like every other preference of this kind in the app
  // (render-lite, view persistence, the comparison-grid selections). It was
  // plain useState(false), so it reset on every reload — and switching Pinokio
  // tabs IS a reload. You turned alerts on, went to watch something else, came
  // back, and they were quietly off again.
  const [desktopAlerts, setDesktopAlerts] = useState(() => {
    try { return localStorage.getItem(KEY) === '1'; } catch { return false; }
  });

  // notifyDesktop() silently does nothing unless permission is already granted,
  // so ask when the toggle is switched ON — otherwise enabling alerts is a
  // no-op for anyone who has not started a run first (start() also asks).
  const toggleDesktopAlerts = () => {
    const on = !desktopAlerts;
    setDesktopAlerts(on);
    try { localStorage.setItem(KEY, on ? '1' : '0'); } catch { /* private mode */ }
    if (on) {
      try {
        if (!('Notification' in window)) {
          notify('This browser has no desktop notifications', 'error');
        } else if (Notification.permission === 'denied') {
          notify('Desktop notifications are blocked for this site', 'error');
        } else if (Notification.permission === 'default') {
          Notification.requestPermission();
        }
      } catch { /* ignore */ }
    }
  };

  // The error is read through a ref rather than a dependency: it lands in the
  // same poll that clears `processing`, but React may commit them in either
  // order, and an effect keyed on both would fire twice — once per edge.
  const errorRef = useRef(error);
  useEffect(() => { errorRef.current = error; }, [error]);

  const prevProcessingRef = useRef(false);
  useEffect(() => {
    if (prevProcessingRef.current && !processing) {
      const failed = !!errorRef.current;
      if (failed) playFailTone(); else playChime();
      if (desktopAlerts) {
        notifyDesktop(
          failed ? 'Roop Unleashed — run failed' : 'Roop Unleashed Render Complete!',
          failed
            // The message itself, not a generic "check the app": the whole
            // point of the notification is that you are not looking at it.
            ? String(errorRef.current).slice(0, 180)
            : 'Your face swap processing run has finished.',
        );
      }
    }
    prevProcessingRef.current = processing;
  }, [processing, desktopAlerts]);

  return { desktopAlerts, toggleDesktopAlerts };
}
