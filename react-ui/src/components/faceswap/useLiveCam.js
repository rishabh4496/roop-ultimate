import { useEffect, useState } from 'react';
import { getJSON, postJSON } from '../../api';

// ── Live camera (webcam → live swap → optional OBS virtual camera) ────────
// Self-contained: the session lives entirely on the backend, so this owns only
// the form state, a "is it up" flag, and the tick that drives the preview.
// Nothing else in the Face Swap tab reads or writes any of it.
//
// `tick` increments ~5×/s while the session is live and is used as a cache
// buster on the preview image URL. It only runs while active, so an idle tab
// costs nothing.

const PREVIEW_INTERVAL_MS = 200;

// The device is opened asynchronously in the backend's capture thread, so
// /api/livecam/start returning 200 does NOT mean the camera came up — a wrong
// index or a camera held by another app fails afterwards. Wait, then ask.
const OPEN_CONFIRM_MS = 1500;

export default function useLiveCam({ notify }) {
  const [liveActive, setLiveActive] = useState(false);
  const [liveBusy, setLiveBusy] = useState(false);
  const [liveCamNum, setLiveCamNum] = useState(0);
  const [liveRes, setLiveRes] = useState('1280x720');
  const [liveObs, setLiveObs] = useState(false);
  const [liveFps, setLiveFps] = useState(30);
  const [liveInterval, setLiveInterval] = useState(6);
  const [liveAudio, setLiveAudio] = useState(false);
  const [liveAudioInput, setLiveAudioInput] = useState('');
  const [liveAudioOutput, setLiveAudioOutput] = useState('');
  const [liveAudioDevices, setLiveAudioDevices] = useState([]);
  const [liveStatus, setLiveStatus] = useState(null);
  const [liveTick, setLiveTick] = useState(0);

  useEffect(() => {
    if (!liveActive) return undefined;
    const id = setInterval(async () => {
      setLiveTick((t) => t + 1);
      try { setLiveStatus(await getJSON('/api/livecam/status')); } catch { /* backend gone */ }
    }, PREVIEW_INTERVAL_MS);
    return () => clearInterval(id);
  }, [liveActive]);

  // SoundDevice is optional. Empty device lists keep webcam mode usable when
  // no virtual audio cable is installed.
  useEffect(() => {
    getJSON('/api/livecam/audio/devices')
      .then((data) => setLiveAudioDevices(data.devices || []))
      .catch(() => {});
  }, []);

  // If the tab remounts while a cam session is running, pick its state back up.
  useEffect(() => {
    getJSON('/api/livecam/status').then((st) => setLiveActive(!!st.active)).catch(() => {});
  }, []);

  const startLiveCam = async () => {
    setLiveBusy(true);
    try {
      await postJSON('/api/livecam/start', {
        cam_number: liveCamNum,
        resolution: liveRes,
        fps: liveFps,
        detector_interval: liveInterval,
        stream_obs: liveObs,
        audio_input_device: liveAudio ? (liveAudioInput || null) : null,
        audio_output_device: liveAudio ? (liveAudioOutput || null) : null,
      });
      setTimeout(async () => {
        try {
          const st = await getJSON('/api/livecam/status');
          setLiveActive(!!st.active);
          if (!st.active) notify(`Camera ${liveCamNum} could not be opened — check the index / close other apps using it`, 'error');
          else notify('Live camera running' + (liveObs ? ' → streaming to virtual camera' : ''));
        } catch { /* backend gone */ }
        setLiveBusy(false);
      }, OPEN_CONFIRM_MS);
    } catch (e) { notify(e.message, 'error'); setLiveBusy(false); }
  };

  const stopLiveCam = async () => {
    setLiveBusy(true);
    try { await postJSON('/api/livecam/stop', {}); } catch { /* already down */ }
    setLiveActive(false);
    setLiveBusy(false);
    notify('Live camera stopped');
  };

  return {
    liveActive, liveBusy, liveTick,
    liveCamNum, setLiveCamNum,
    liveRes, setLiveRes,
    liveObs, setLiveObs,
    liveFps, setLiveFps,
    liveInterval, setLiveInterval,
    liveAudio, setLiveAudio,
    liveAudioInput, setLiveAudioInput,
    liveAudioOutput, setLiveAudioOutput,
    liveAudioDevices, liveStatus,
    startLiveCam, stopLiveCam,
  };
}
