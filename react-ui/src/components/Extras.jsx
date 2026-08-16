import React, { useState, useEffect } from 'react';
import { postFile, fileUrl, getJSON } from '../api';
import { Section, Select, Slider, Button } from './ui';
import { Icon } from '../icons';

const RESOLUTIONS = ['Original', '3840x', '2560x', '1920x', '1280x', '1024x', '640x'];
const ROTATIONS = ['None', '90° Clockwise', '90° Counter-Clockwise', '180°'];

// Human labels for the frame post-processor subtypes.
const OP_LABELS = {
  upscale: { esrganx2: 'Real-ESRGAN ×2', esrganx4: 'Real-ESRGAN ×4', esrgan_anime_x4: 'Real-ESRGAN Anime ×4', ultrasharp_x4: 'Ultra-Sharp ×4', lsdirx4: 'LSDIR ×4', clear_reality_x4: 'Clear Reality ×4', span_x4: 'SPAN ×4', compact_x4: 'Compact ×4 (fast AI)', nomos8k_x4: 'Nomos8k ×4' },
  colorize: { deoldify_artistic: 'DeOldify (Artistic)', deoldify_stable: 'DeOldify (Stable)' },
  filter: { stylize: 'Stylize', detailenhance: 'Detail Enhance', pencil: 'Pencil Sketch', cartoon: 'Cartoon', C64: 'C64 Palette' },
};
const OP_TITLES = { upscale: 'AI Upscale', colorize: 'Colorize (B&W → color)', filter: 'Stylize Filter' };

export default function Extras({ notify, registerFileListener }) {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [fileUrlSrc, setFileUrlSrc] = useState('');
  const [opts, setOpts] = useState({ resolution: 'Original', rotation: 'None', fps: 30, crop_left: 0, crop_right: 0, crop_top: 0, crop_bottom: 0 });
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const set = (k, v) => setOpts((o) => ({ ...o, [k]: v }));

  // Frame post-processor (AI upscale / colorize / stylize) state
  const [frameOps, setFrameOps] = useState(null);
  const [operation, setOperation] = useState('upscale');
  const [subtype, setSubtype] = useState('esrganx2');
  const [enhBusy, setEnhBusy] = useState(false);
  const [enhResult, setEnhResult] = useState(null);

  useEffect(() => {
    getJSON('/api/extras/frame_ops').then(setFrameOps).catch(() => {});
  }, []);

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
      const res = await postFile('/api/extras/enhance', file, { operation, subtype });
      setEnhResult(res);
      notify('Done');
    } catch (e) { notify(e.message, 'error'); } finally { setEnhBusy(false); }
  };

  const onPick = (e) => {
    const f = e.target?.files?.[0] || e;
    if (f) {
      setFile(f);
      setFileName(f.name);
      setResult(null);
      if (fileUrlSrc) URL.revokeObjectURL(fileUrlSrc);
      setFileUrlSrc(URL.createObjectURL(f));
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
        if (fileUrlSrc) URL.revokeObjectURL(fileUrlSrc);
        setFileUrlSrc(URL.createObjectURL(f));
        notify(`Loaded ${f.name} into editor`);
      }
      return true; // consumed
    });
  }, [registerFileListener, fileUrlSrc]);
  /* eslint-enable react-hooks/exhaustive-deps */

  useEffect(() => {
    return () => {
      if (fileUrlSrc) URL.revokeObjectURL(fileUrlSrc);
    };
  }, [fileUrlSrc]);

  const apply = async () => {
    if (!file) { notify('Pick a file first', 'error'); return; }
    setBusy(true);
    try {
      const res = await postFile('/api/extras/apply', file, opts);
      setResult(res); notify('Done');
    } catch (e) { notify(e.message, 'error'); } finally { setBusy(false); }
  };

  return (
    <div className="space-y-6">
      <Section>
        <h2 className="text-lg font-bold">Media editor</h2>
        <p className="text-sm text-white/50">Resize, rotate, crop and re-time images or videos.</p>
      </Section>

      <div className="grid grid-cols-1 lg:grid-cols-2 3xl:grid-cols-3 gap-6">
        <Section title="Input & transform" icon={Icon.editor}>
          <label className="block cursor-pointer focus-within:outline focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-[var(--accent)]">
            <div className="px-4 py-5 rounded-lg border-2 border-dashed border-white/15 hover:border-[var(--accent)]/50 text-center text-sm text-white/60 mb-4">
              {fileName || 'Pick an image or video'}
            </div>
            <input type="file" accept="image/*,video/*" onChange={onPick} className="sr-only" />
          </label>

          {file && fileUrlSrc && (
            <div className="relative overflow-hidden rounded-xl bg-black/45 border border-white/10 aspect-video flex items-center justify-center mb-4 max-h-[300px]">
              {file.type?.startsWith('video') || /\.(mp4|mkv|mov|avi|webm|gif)$/i.test(fileName) ? (
                <video src={fileUrlSrc} muted loop autoPlay className="max-w-full max-h-full object-contain pointer-events-none" />
              ) : (
                <img src={fileUrlSrc} alt="Input source" className="max-w-full max-h-full object-contain pointer-events-none" />
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
          <Slider label="Output FPS (video)" min={1} max={120} step={1} value={opts.fps} onChange={(v) => set('fps', v)} />
        </Section>

        <Section title="Crop (%)" icon={Icon.split}>
          <Slider label="Left" min={0} max={49} step={1} value={opts.crop_left} onChange={(v) => set('crop_left', v)} />
          <Slider label="Right" min={0} max={49} step={1} value={opts.crop_right} onChange={(v) => set('crop_right', v)} />
          <Slider label="Top" min={0} max={49} step={1} value={opts.crop_top} onChange={(v) => set('crop_top', v)} />
          <Slider label="Bottom" min={0} max={49} step={1} value={opts.crop_bottom} onChange={(v) => set('crop_bottom', v)} />
          <div className="flex items-center gap-2">
            <Button variant="primary" onClick={apply} disabled={busy}>{busy ? 'Processing…' : 'Apply'}</Button>
            <Button variant="secondary" onClick={() => setOpts({ resolution: 'Original', rotation: 'None', fps: 30, crop_left: 0, crop_right: 0, crop_top: 0, crop_bottom: 0 })}>Reset</Button>
          </div>
        </Section>

        <Section title="AI frame post-processing" icon={Icon.wand}>
          <p className="text-xs text-white/40 -mt-2">Runs on the file picked above. Works on images and videos (video is processed frame-by-frame — can be slow).</p>
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
          {operation === 'upscale' && <p className="text-micro text-white/45">First run downloads the model (~65 MB). ×4 on video is heavy.</p>}
        </Section>
      </div>

      {enhResult?.path && (
        <Section title="AI post-processing output">
          {enhResult.kind === 'video'
            ? <video src={fileUrl(enhResult.path)} controls className="w-full rounded-lg border border-white/10" />
            : <img src={fileUrl(enhResult.path)} alt="output" className="max-w-full rounded-lg border border-white/10" />}
          <a href={fileUrl(enhResult.path)} download className="inline-block mt-2 text-sm text-[var(--accent)] underline">⬇ Download</a>
        </Section>
      )}

      {result?.path && (
        <Section title="Output">
          {result.kind === 'video'
            ? <video src={fileUrl(result.path)} controls className="w-full rounded-lg border border-white/10" />
            : <img src={fileUrl(result.path)} alt="output" className="max-w-full rounded-lg border border-white/10" />}
          <a href={fileUrl(result.path)} download className="inline-block mt-2 text-sm text-[var(--accent)] underline">⬇ Download</a>
        </Section>
      )}

      <Section>
        <p className="text-xs text-white/40">
          AI upscale, colorize and stylize filters are available above. The Gradio "Frame Editor"
          (per-frame canvas painting, tracked re-swap, MP4/GIF compile) is not yet ported to React —
          use the legacy Gradio UI for that workflow.
        </p>
      </Section>
    </div>
  );
}
