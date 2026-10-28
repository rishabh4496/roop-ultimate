import OutputVideoPlayer from './OutputVideoPlayer';
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { fileUrl } from '../api';
import { useApi } from '../useApi';
import { Section, Select, Slider, Button } from './ui';
import { Icon } from '../icons';
import { logFailure } from '../failureLog';
import { SLOT_IDS } from '../store/slotStore';
import { useHistoryChannel, EXTRAS_OPTS_CHANNEL } from '../history/historyChannels';
import useSnapshotSlots from './faceswap/useSnapshotSlots';
import SnapshotSlots from './faceswap/SnapshotSlots';
import EnhancerPipelineStack from './faceswap/EnhancerPipelineStack';
import VideoFramesetExtractor from './extras/VideoFramesetExtractor';
import MouthCustodyPanel from './faceswap/MouthCustodyPanel';
import { overridePatch } from './faceswap/mouthCustody';
import BroadcastScopesDock from './faceswap/BroadcastScopesDock';
import StereoVRViewport, { DEFAULT_VR_PARAMS } from './faceswap/StereoVRViewport';
import { drawZebraOverlay } from './faceswap/falseColorZebra';
import {
  createReferenceChannel,
  closeReferenceChannel,
  broadcastFrame,
  broadcastTransport,
  broadcastGrading,
  MESSAGE_TYPES,
} from './monitor/referenceChannel';

const RESOLUTIONS = ['Original', '3840x', '2560x', '1920x', '1280x', '1024x', '640x'];
const ROTATIONS = ['None', '90° Clockwise', '90° Counter-Clockwise', '180°'];

// Human labels for the frame post-processor subtypes.
const OP_LABELS = {
  upscale: { esrganx2: 'Real-ESRGAN ×2', esrganx4: 'Real-ESRGAN ×4', esrgan_anime_x4: 'Real-ESRGAN Anime ×4', ultrasharp_x4: 'Ultra-Sharp ×4', lsdirx4: 'LSDIR ×4', clear_reality_x4: 'Clear Reality ×4', span_x4: 'SPAN ×4', compact_x4: 'Compact ×4 (fast AI)', nomos8k_x4: 'Nomos8k ×4' },
  colorize: { deoldify_artistic: 'DeOldify (Artistic)', deoldify_stable: 'DeOldify (Stable)' },
  filter: { stylize: 'Stylize', detailenhance: 'Detail Enhance', pencil: 'Pencil Sketch', cartoon: 'Cartoon', C64: 'C64 Palette' },
};
const OP_TITLES = { upscale: 'AI Upscale', colorize: 'Colorize (B&W → color)', filter: 'Stylize Filter' };

// One definition of "untouched": the initial state, the Reset button and each
// slider's double-click reset all read it, so they cannot disagree.
// `override_audio_path` is '' (not null) for "none": it travels as multipart form text, and FormData.append(k, null) sends "null".
const DEFAULT_OPTS = { resolution: 'Original', rotation: 'None', fps: 30, crop_left: 0, crop_right: 0, crop_top: 0, crop_bottom: 0, override_audio_path: '', audio_offset_ms: 0 };
const CROP_SIDES = [['crop_left', 'Left'], ['crop_right', 'Right'], ['crop_top', 'Top'], ['crop_bottom', 'Bottom']];

export default function Extras({ notify, registerFileListener, meta, settings, setSettings, setTab }) {
  const { getJSON, postFile } = useApi();
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [fileUrlSrc, setFileUrlSrc] = useState('');
  const fileUrlSrcRef = useRef('');
  const [opts, setOpts] = useState(DEFAULT_OPTS);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const set = (k, v) => setOpts((o) => ({ ...o, [k]: v }));
  const patchOpts = (patch) => setOpts((o) => ({ ...o, ...patch }));
  const [uploadingAudio, setUploadingAudio] = useState(false);
  const [scopeActive, setScopeActive] = useState(false);
  const [zebraActive, setZebraActive] = useState(false);
  const [monitorActive, setMonitorActive] = useState(false);
  const [vrActive, setVrActive] = useState(false);
  const [vrParams, setVrParams] = useState(DEFAULT_VR_PARAMS);
  const refChannelRef = useRef(null);

  const openMonitor = useCallback(() => {
    setMonitorActive(true);
    if (typeof window !== 'undefined') {
      window.open(
        '#/monitor',
        'roop_clean_monitor',
        'width=1280,height=720,menubar=no,toolbar=no,location=no,status=no,resizable=yes'
      );
    }
  }, []);

  useEffect(() => {
    if (!monitorActive) return undefined;
    const channel = createReferenceChannel((data) => {
      if (data?.type === MESSAGE_TYPES.PING) {
        broadcastTransport(channel, {
          isPlaying: true,
          fps: opts.fps || 30,
        });
        broadcastGrading(channel, {
          zebraActive,
        });
      }
    });
    refChannelRef.current = channel;

    let animId = 0;
    let lastTs = 0;
    let isBroadcasting = false;

    const streamLoop = (ts) => {
      animId = requestAnimationFrame(streamLoop);
      if (ts - lastTs < 16) return;
      lastTs = ts;

      if (isBroadcasting) return;
      const elem = mediaElemRef.current;
      if (!elem) return;
      const w = elem.videoWidth || elem.naturalWidth || elem.clientWidth || 0;
      const h = elem.videoHeight || elem.naturalHeight || elem.clientHeight || 0;
      if (w === 0 || h === 0) return;

      if (typeof createImageBitmap === 'function') {
        isBroadcasting = true;
        createImageBitmap(elem)
          .then((bmp) => {
            broadcastFrame(channel, bmp, {
              width: w,
              height: h,
              timestamp: Date.now(),
            });
          })
          .catch(() => {})
          .finally(() => {
            isBroadcasting = false;
          });
      }
    };
    animId = requestAnimationFrame(streamLoop);

    return () => {
      if (animId) cancelAnimationFrame(animId);
      closeReferenceChannel(channel);
      refChannelRef.current = null;
    };
  }, [monitorActive, zebraActive, opts.fps]);

  useEffect(() => {
    if (!refChannelRef.current) return;
    broadcastGrading(refChannelRef.current, { zebraActive });
  }, [zebraActive]);
  const mediaElemRef = useRef(null);
  const zebraCanvasRef = useRef(null);

  // Stage 23: the crop / resize / rotate / fps options are part of the undo history, and the four
  // snapshot slots keep them (with the shared enhancement recipe) plus a thumbnail of the source
  // frame. The Editor's viewport shows the UNPROCESSED input (Apply runs on the server), so there is
  // no rendered result frame to flip here and no split-wipe reference; a slot's picture is what the
  // input looked like when it was captured.
  useHistoryChannel('extrasOpts', opts, setOpts, EXTRAS_OPTS_CHANNEL);
  const slotBank = useSnapshotSlots({
    bank: 'extras',
    notify,
    captureFrame: async () => {
      const el = mediaElemRef.current;
      if (!el || typeof createImageBitmap !== 'function') return null;
      try { return { bitmap: await createImageBitmap(el), frameKey: `${fileName}_${el.currentTime ?? 0}` }; } catch { return null; }
    },
  });
  useEffect(() => {
    const onKey = (e) => {
      const digit = /^Digit([1-4])$/.exec(e.code || '');
      if (!digit || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
      if (document.querySelector('[role="dialog"]')) return;
      e.preventDefault();
      slotBank.activate(SLOT_IDS[Number(digit[1]) - 1]);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [slotBank]);

  useEffect(() => {
    if (!zebraActive || !mediaElemRef.current || !zebraCanvasRef.current) return;
    const elem = mediaElemRef.current;
    const canvas = zebraCanvasRef.current;
    const w = elem.videoWidth || elem.naturalWidth || elem.clientWidth || 300;
    const h = elem.videoHeight || elem.naturalHeight || elem.clientHeight || 150;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, w, h);
      try {
        ctx.drawImage(elem, 0, 0, w, h);
        drawZebraOverlay(ctx, w, h, null, 'zebra');
      } catch {
        // ignore cross-origin
      }
    }
  }, [zebraActive, fileUrlSrc]);

  // Frame post-processor (AI upscale / colorize / stylize) state
  const [frameOps, setFrameOps] = useState(null);
  const [operation, setOperation] = useState('upscale');
  const [subtype, setSubtype] = useState('esrganx2');
  const [enhBusy, setEnhBusy] = useState(false);
  const [enhResult, setEnhResult] = useState(null);

  useEffect(() => {
    getJSON('/api/extras/frame_ops').then(setFrameOps).catch(logFailure('Loading the frame post-processors'));
  }, [getJSON]);

  // Keep subtype valid whenever the operation changes.
  /* eslint-disable react-hooks/exhaustive-deps -- intentional: resync only when operation/frameOps change, not on every subtype edit */
  useEffect(() => {
    if (!frameOps) return;
    const list = frameOps[operation] || [];
    if (!list.includes(subtype)) setSubtype(list[0] || '');
  }, [operation, frameOps]);
  /* eslint-enable react-hooks/exhaustive-deps */

  const runEnhance = async () => {
    if (!file) { notify('Pick a file first', 'error'); return; }
    setEnhBusy(true);
    setEnhResult(null);
    try {
      const res = await postFile('/api/extras/enhance', file, { operation, subtype }, { timeout: 0 });
      setEnhResult(res);
      notify('Done');
    } catch (e) { notify(e.message, 'error'); } finally { setEnhBusy(false); }
  };

  const onAddAudio = async (files) => {
    if (!files || !files.length) return;
    setUploadingAudio(true);
    try {
      const res = await postFile('/api/lipsync/audio/add', files[0], undefined, { timeout: 0 });
      patchOpts(overridePatch(res.path, { legacy: false, empty: '' }));
    } catch (e) { notify(e.message, 'error'); } finally { setUploadingAudio(false); }
  };

  const onPick = (e) => {
    const f = e.target?.files?.[0] || e;
    if (f) {
      setFile(f);
      setFileName(f.name);
      setResult(null);
      if (fileUrlSrcRef.current) URL.revokeObjectURL(fileUrlSrcRef.current);
      const url = URL.createObjectURL(f);
      fileUrlSrcRef.current = url;
      setFileUrlSrc(url);
    }
  };

  /* eslint-disable react-hooks/exhaustive-deps -- intentional: subscribe once; notify is stable */
  useEffect(() => {
    if (!registerFileListener) return;
    return registerFileListener((files) => {
      const f = files[0];
      if (f) {
        setFile(f);
        setFileName(f.name);
        setResult(null);
        if (fileUrlSrcRef.current) URL.revokeObjectURL(fileUrlSrcRef.current);
        const url = URL.createObjectURL(f);
        fileUrlSrcRef.current = url;
        setFileUrlSrc(url);
        notify(`Loaded ${f.name} into editor`);
      }
      return true; // consumed
    });
  }, [registerFileListener]);
  /* eslint-enable react-hooks/exhaustive-deps */

  useEffect(() => {
    return () => {
      if (fileUrlSrcRef.current) {
        URL.revokeObjectURL(fileUrlSrcRef.current);
        fileUrlSrcRef.current = '';
      }
    };
  }, []);

  const apply = async () => {
    if (!file) { notify('Pick a file first', 'error'); return; }
    setBusy(true);
    try {
      const res = await postFile('/api/extras/apply', file, opts, { timeout: 0 });
      setResult(res); notify('Done');
    } catch (e) { notify(e.message, 'error'); } finally { setBusy(false); }
  };

  return (
    <div className="space-y-4">
      <Section>
        <h2 className="text-lg font-bold">Media editor</h2>
        <p className="text-sm text-white/50">Resize, rotate, crop and re-time images or videos.</p>
      </Section>

      <div className="grid grid-cols-1 lg:grid-cols-2 3xl:grid-cols-3 gap-4">
        <Section title="Input & transform" icon={Icon.editor}>
          <label className="block cursor-pointer focus-within:outline focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-[var(--accent)]">
            <div className="px-4 py-5 rounded-lg border-2 border-dashed border-white/15 hover:border-[var(--accent)]/50 text-center text-sm text-white/60 mb-4">
              {fileName || 'Pick an image or video'}
            </div>
            <input type="file" accept="image/*,video/*" onChange={onPick} className="sr-only" />
          </label>

          {file && fileUrlSrc && (
            <div className="mb-2">
              <SnapshotSlots api={slotBank} supportsReference={false} />
            </div>
          )}

          {file && fileUrlSrc && (
            <div className="relative overflow-hidden rounded-xl bg-black/45 border border-white/10 aspect-video flex items-center justify-center mb-4 max-h-[300px]">
              {/* Stage 12 & Stage 18 Broadcast Scopes & Zebra overlay controls */}
              <div className="absolute top-2 right-2 flex items-center gap-1 z-30">
                <button
                  type="button"
                  onClick={() => setScopeActive((a) => !a)}
                  aria-pressed={scopeActive}
                  className={`px-2 py-0.5 rounded-md text-micro font-bold transition-colors ${
                    scopeActive
                      ? 'text-amber-300 bg-amber-500/20 border border-amber-400/40 shadow-sm'
                      : 'bg-black/60 text-white/70 hover:text-white'
                  }`}
                  title="Toggle Broadcast Scopes Dock (Waveform, RGB Parade, Vectorscope)"
                >
                  Scopes
                </button>
                <button
                  type="button"
                  onClick={() => setZebraActive((z) => !z)}
                  aria-pressed={zebraActive}
                  className={`px-2 py-0.5 rounded-md text-micro font-bold transition-colors ${
                    zebraActive
                      ? 'text-rose-300 bg-rose-500/20 border border-rose-400/40 shadow-sm'
                      : 'bg-black/60 text-white/70 hover:text-white'
                  }`}
                  title="Toggle Zebra Exposure Overlay (>95 / <5 IRE)"
                >
                  Zebra
                </button>
                <button
                  type="button"
                  onClick={openMonitor}
                  aria-pressed={monitorActive}
                  className={`px-2 py-0.5 rounded-md text-micro font-bold transition-colors ${
                    monitorActive
                      ? 'text-cyan-300 bg-cyan-500/20 border border-cyan-400/40 shadow-sm'
                      : 'bg-black/60 text-white/70 hover:text-white'
                  }`}
                  title="Pop-Out Clean Feed Reference Monitor (#/monitor)"
                >
                  Monitor ↗
                </button>
                <button
                  type="button"
                  onClick={() => setVrActive((v) => !v)}
                  aria-pressed={vrActive}
                  className={`px-2 py-0.5 rounded-md text-micro font-bold transition-colors ${
                    vrActive
                      ? 'text-teal-300 bg-teal-500/20 border border-teal-400/40 shadow-sm'
                      : 'bg-black/60 text-white/70 hover:text-white'
                  }`}
                  title="Toggle WebGL2 3D / Equirectangular VR180/360 Viewport & Stereoscopic Inspector"
                >
                  3D / VR
                </button>
              </div>

              {file.type?.startsWith('video') || /\.(mp4|mkv|mov|avi|webm|gif)$/i.test(fileName) ? (
                <video ref={mediaElemRef} src={fileUrlSrc} muted loop autoPlay className="max-w-full max-h-full object-contain pointer-events-none" />
              ) : (
                <img ref={mediaElemRef} src={fileUrlSrc} alt="Input source" className="max-w-full max-h-full object-contain pointer-events-none" />
              )}

                            {vrActive && (
                <StereoVRViewport
                  active={vrActive}
                  onClose={() => setVrActive(false)}
                  mediaSource={mediaElemRef.current}
                  params={vrParams}
                  onChange={setVrParams}
                />
              )}
              {zebraActive && (
                <canvas ref={zebraCanvasRef} className="absolute inset-0 w-full h-full object-contain pointer-events-none z-10" />
              )}
              {/* Bounding box crop overlay dims the cropped regions */}
              <div 
                className="absolute border-2 border-dashed border-[var(--accent)] shadow-[0_0_0_9999px_rgba(0,0,0,0.6)] pointer-events-none z-20"
                style={{
                  left: `${opts.crop_left}%`,
                  right: `${opts.crop_right}%`,
                  top: `${opts.crop_top}%`,
                  bottom: `${opts.crop_bottom}%`
                }}
              />
            </div>
          )}

          <Select label="Resize width" value={opts.resolution} onChange={(v) => set('resolution', v)} options={RESOLUTIONS} />
          <Select label="Rotation" value={opts.rotation} onChange={(v) => set('rotation', v)} options={ROTATIONS} />
          <Slider label="Output FPS (video)" min={1} max={120} step={1} value={opts.fps} defaultValue={DEFAULT_OPTS.fps} onChange={(v) => set('fps', v)} />
        </Section>

        {file && (file.type?.startsWith('video') || /\.(mp4|mkv|mov|avi|webm|gif)$/i.test(fileName)) && (
          <Section title="Audio" icon={Icon.editor}>
            <p className="text-xs text-muted -mt-2">Replace the clip's audio with an external track and slip it against the video (applied with Apply). The video is not re-encoded.</p>
            <MouthCustodyPanel
              params={opts}
              onPatch={patchOpts}
              onAddAudio={onAddAudio}
              uploading={uploadingAudio}
              showMode={false}
              legacy={false}
              emptyTrack=""
            />
          </Section>
        )}

        <Section title="Crop (%)" icon={Icon.split}>
          {CROP_SIDES.map(([key, label]) => (
            <Slider key={key} label={label} min={0} max={49} step={1} value={opts[key]} defaultValue={DEFAULT_OPTS[key]} onChange={(v) => set(key, v)} />
          ))}
          <div className="flex items-center gap-2">
            <Button variant="primary" onClick={apply} disabled={busy}>{busy ? 'Processing…' : 'Apply'}</Button>
            <Button variant="secondary" onClick={() => setOpts(DEFAULT_OPTS)}>Reset</Button>
          </div>
        </Section>

        <VideoFramesetExtractor notify={notify} setTab={setTab} />

        <Section title="AI frame post-processing" icon={Icon.wand}>
          <p className="text-xs text-muted -mt-2">Runs on the file picked above. Works on images and videos (video is processed frame-by-frame — can be slow).</p>
          <Select label="Operation" value={operation} onChange={setOperation}
            options={frameOps ? Object.keys(frameOps) : ['upscale', 'colorize', 'filter']} />
          {frameOps && (frameOps[operation] || []).length > 0 && (
            <div>
              <div className="text-xs font-medium text-white/70 mb-1.5">{OP_TITLES[operation] || 'Model'}</div>
              <select
                value={subtype}
                onChange={(e) => setSubtype(e.target.value)}
                className="w-full px-3 py-2 rounded-xl glass-input text-white text-sm focus:outline-none cursor-pointer"
              >
                {(frameOps[operation] || []).map((s) => (
                  <option key={s} value={s} className="bg-[#121420]">
                    {(OP_LABELS[operation] && OP_LABELS[operation][s]) || s}
                  </option>
                ))}
              </select>
            </div>
          )}
          <Button variant="primary" onClick={runEnhance} disabled={enhBusy || !file}>
            {enhBusy ? 'Processing…' : 'Run'}
          </Button>
          {operation === 'upscale' && <p className="text-micro text-muted">First run downloads the model (~65 MB). ×4 on video is heavy.</p>}
        </Section>
      </div>

      {settings && setSettings && (
        <Section title="Swap enhancement pipeline" icon={Icon.tune} collapsible defaultOpen={false}>
          <p className="-mt-1 text-mini text-muted">
            The same chain the Face Swap tab runs after every swap, on the same live settings: changing it here changes it there.
          </p>
          <EnhancerPipelineStack
            settings={settings}
            meta={meta}
            onChange={(key, value) => setSettings((s) => ({ ...s, [key]: value }))}
          />
        </Section>
      )}

      {enhResult?.path && (
        <Section title="AI post-processing output">
          {enhResult.kind === 'video'
            ? <OutputVideoPlayer src={fileUrl(enhResult.path)} renderKey={enhResult.path} className="w-full rounded-lg border border-white/10" />
            : <img src={fileUrl(enhResult.path)} alt="output" className="max-w-full rounded-lg border border-white/10" />}
          <a href={fileUrl(enhResult.path)} download className="inline-block mt-2 text-sm text-accent underline">⬇ Download</a>
        </Section>
      )}

      {result?.path && (
        <Section title="Output">
          {result.kind === 'video'
            ? <OutputVideoPlayer src={fileUrl(result.path)} renderKey={result.path} className="w-full rounded-lg border border-white/10" />
            : <img src={fileUrl(result.path)} alt="output" className="max-w-full rounded-lg border border-white/10" />}
          <a href={fileUrl(result.path)} download className="inline-block mt-2 text-sm text-accent underline">⬇ Download</a>
        </Section>
      )}

      <Section>
        <p className="text-xs text-muted">
          AI upscale, colorize and stylize filters are available above. The Gradio "Frame Editor"
          (per-frame canvas painting, tracked re-swap, MP4/GIF compile) is not yet ported to React —
          use the legacy Gradio UI for that workflow.
        </p>
      </Section>

      {/* ── Stage 18 Broadcast Scopes Dock ── */}
      <BroadcastScopesDock
        active={scopeActive}
        onClose={() => setScopeActive(false)}
        sourceCanvas={mediaElemRef.current}
        renderedCanvas={mediaElemRef.current}
        className="fixed bottom-6 right-6 z-50 shadow-2xl"
      />
    </div>
  );
}
