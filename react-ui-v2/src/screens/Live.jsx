import React from 'react';
import { Video, VideoOff, Loader2, Radio, Mic, Cast, Gauge } from 'lucide-react';
import useLiveCam from '@engine/components/faceswap/useLiveCam';
import { useStudio } from '../engine/studio';
import { useRun } from '../engine/run';
import { Panel, SplitText } from '../fx/effects';
import { Select, Slider, Switch, notify } from '../ui/controls';

// ── Live: webcam -> live swap -> optional OBS virtual camera ──
// The session lives in the engine (roop/virtualcam.py); state, the 5/s preview tick and the
// open-confirm wait are 1.0's useLiveCam. The camera opens asynchronously in the engine's capture
// thread, so /api/livecam/start answering 200 does not mean it came up: the hook asks again after
// 1.5 s and reports a camera that did not open. The swap uses the Studio's cast and settings.

const RESOLUTIONS = ['640x480', '1280x720', '1920x1080'];

export default function Live() {
  const cam = useLiveCam({ notify: (m, t) => notify(m, t === 'error' ? 'error' : t === 'warning' ? 'warn' : 'success') });
  const sources = useStudio((s) => s.sources);
  const processing = useRun((r) => r.progress.processing);
  const inputs = (cam.liveAudioDevices || []).filter((d) => d.max_input_channels > 0);
  const outputs = (cam.liveAudioDevices || []).filter((d) => d.max_output_channels > 0);
  const st = cam.liveStatus || {};
  const stats = st.stats && typeof st.stats === 'object' ? Object.entries(st.stats).filter(([, v]) => typeof v === 'number' || typeof v === 'string').slice(0, 8) : [];
  const off = cam.liveActive || cam.liveBusy;
  return (
    <div className="mx-auto grid max-w-[1400px] gap-3 lg:grid-cols-[minmax(0,1fr)_340px]" data-testid="live">
      <Panel title="Live camera" accent={cam.liveActive ? 'live' : undefined} beam={cam.liveActive}>
        <div className="relative grid aspect-video place-items-center overflow-hidden rounded-xl bg-black" data-testid="live-preview">
          {cam.liveActive
            ? <img src={`/api/livecam/frame?t=${cam.liveTick}`} alt="Live swapped camera" className="h-full w-full object-contain" onError={(e) => { e.currentTarget.style.visibility = 'hidden'; }} onLoad={(e) => { e.currentTarget.style.visibility = 'visible'; }} />
            : <div className="text-center text-[var(--fg-dim)]"><VideoOff size={30} className="mx-auto mb-2 opacity-60" /><div className="text-[13px]">Camera off</div></div>}
          {cam.liveActive && <span className="absolute left-3 top-3 badge badge-danger"><Radio size={11} /> LIVE</span>}
        </div>
        {cam.liveActive && stats.length > 0 && (
          <div className="mt-2 grid grid-cols-2 gap-2 font-mono text-[11px] sm:grid-cols-4" data-testid="live-stats">
            {stats.map(([k, v]) => (
              <div key={k} className="rounded-lg border border-[var(--rule)] bg-black/30 px-2 py-1"><div className="text-[var(--fg-faint)]">{k.replace(/_/g, ' ')}</div><div className="text-white">{typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(1)) : v}</div></div>
            ))}
          </div>
        )}
        {st.error && <div className="mt-2 text-[12px] text-[var(--danger)]" role="alert">{st.error}</div>}
      </Panel>

      <div className="space-y-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight"><SplitText text="Live" className="gradient-text" /></h1>
          <p className="text-[12.5px] text-[var(--fg-dim)]">Swaps your webcam in real time with the Studio's cast and settings, and can feed an OBS virtual camera.</p>
        </div>
        <Panel title="Camera">
          <div className="space-y-2.5">
            <div className="grid grid-cols-2 gap-2">
              <label className="text-[11.5px] text-[var(--fg-dim)]">Camera index
                <input type="number" min={0} max={9} className="input mt-1" value={cam.liveCamNum} disabled={off} onChange={(e) => cam.setLiveCamNum(Math.max(0, Number(e.target.value) || 0))} data-testid="live-cam" />
              </label>
              <Select label="Resolution" value={cam.liveRes} options={RESOLUTIONS} onChange={cam.setLiveRes} disabled={off} />
            </div>
            <Slider label="Target fps" value={cam.liveFps} min={10} max={60} step={1} onChange={cam.setLiveFps} />
            <Slider label="Detect every N frames" hint="Higher is faster; lower follows quick head moves better" value={cam.liveInterval} min={1} max={30} step={1} onChange={cam.setLiveInterval} />
            <Switch label="Stream to a virtual camera" hint="Needs the OBS virtual camera installed" checked={cam.liveObs} onChange={cam.setLiveObs} />
            <Switch label="Pass audio through" hint="Delays audio to match the swapped video" checked={cam.liveAudio} onChange={cam.setLiveAudio} />
            {cam.liveAudio && (
              <div className="grid grid-cols-1 gap-2">
                <Select label={<span className="inline-flex items-center gap-1"><Mic size={11} /> Microphone</span>} value={cam.liveAudioInput} options={[{ value: '', label: 'Default' }, ...inputs.map((d) => ({ value: String(d.index), label: d.name }))]} onChange={cam.setLiveAudioInput} />
                <Select label={<span className="inline-flex items-center gap-1"><Cast size={11} /> Output</span>} value={cam.liveAudioOutput} options={[{ value: '', label: 'Default' }, ...outputs.map((d) => ({ value: String(d.index), label: d.name }))]} onChange={cam.setLiveAudioOutput} />
                {!cam.liveAudioDevices.length && <div className="text-[11.5px] text-[var(--warn)]">No audio devices reported (sounddevice not installed?).</div>}
              </div>
            )}
          </div>
          {!sources.length && <div className="mt-2 text-[12px] text-[var(--warn)]">Add a face to the Studio's cast first: that is who you become.</div>}
          <div className="mt-3 flex gap-2">
            {!cam.liveActive ? (
              <button type="button" className="btn btn-primary flex-1" disabled={cam.liveBusy || processing || !sources.length} onClick={cam.startLiveCam} data-testid="live-start"
                title={processing ? 'A render is running: the live camera needs the GPU' : undefined}>
                {cam.liveBusy ? <Loader2 size={15} className="animate-spin" /> : <Video size={15} />} {cam.liveBusy ? 'Opening camera…' : 'Start'}
              </button>
            ) : (
              <button type="button" className="btn btn-danger flex-1" disabled={cam.liveBusy} onClick={cam.stopLiveCam} data-testid="live-stop"><VideoOff size={15} /> Stop</button>
            )}
          </div>
          {cam.liveActive && st.detector_interval && <div className="mt-2 flex items-center gap-1.5 font-mono text-[11px] text-[var(--fg-faint)]"><Gauge size={11} /> {st.resolution} · detect every {st.detector_interval} · {st.fps_target} fps target{st.virtual_camera ? ` · ${st.virtual_camera}` : ''}</div>}
        </Panel>
      </div>
    </div>
  );
}
