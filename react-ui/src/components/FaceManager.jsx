import React, { useState, useEffect, useRef, useMemo } from 'react';
import { postJSON, postFiles, postFile, fileUrl, API } from '../api';
import { Section, Slider, Toggle, Button } from './ui';

// Detector engines the Face Manager can extract with. SCRFD is buffalo_l's
// built-in detector (the pipeline default); the others help harvest faces from
// footage where a different detector locks on better.
const DETECTORS = [
  { id: 'scrfd', label: 'SCRFD (default)' },
  { id: 'retinaface', label: 'RetinaFace 10G' },
  { id: 'retinaface_r50', label: 'RetinaFace R50' },
  { id: 'yoloface', label: 'YOLOFace' },
  { id: 'yunet', label: 'YuNet' },
];

// Score → colour band. These thresholds are advisory (the numbers are a
// composite FIQA score, not a hard classifier), so keep the bands gentle.
const scoreTone = (s) => (s >= 0.6 ? 'emerald' : s >= 0.4 ? 'amber' : 'red');
const TONE = {
  emerald: { text: 'text-emerald-400', ring: 'ring-emerald-400/40', bg: 'bg-emerald-500/15' },
  amber: { text: 'text-amber-400', ring: 'ring-amber-400/40', bg: 'bg-amber-500/15' },
  red: { text: 'text-red-400', ring: 'ring-red-400/40', bg: 'bg-red-500/15' },
};

export default function FaceManager({ notify, registerFileListener }) {
  const [faces, setFaces] = useState([]);   // data-URL thumbnails
  const [scores, setScores] = useState([]); // parallel FIQA scores 0..1
  const [meta, setMeta] = useState([]);     // parallel breakdown dicts
  const [sel, setSel] = useState(0);
  const [video, setVideo] = useState(null);
  const [frame, setFrame] = useState(1);
  const [maxFrames, setMaxFrames] = useState(1);
  const [built, setBuilt] = useState(null);
  const [busy, setBusy] = useState(false);

  // Extraction options
  const [detector, setDetector] = useState('scrfd');
  const [restore, setRestore] = useState(false);
  const [threshold, setThreshold] = useState(0); // quality gate; 0 = keep all

  // Latest option values for the (once-subscribed) drag-drop listener, so a
  // dropped file uses the detector/restore currently selected in the UI.
  const optsRef = useRef({ detector, restore });
  useEffect(() => { optsRef.current = { detector, restore }; }, [detector, restore]);

  // Merge a faces payload ({faces, scores, meta}) into state.
  const applyPayload = (res) => {
    setFaces(res.faces || []);
    setScores(res.scores || []);
    setMeta(res.meta || []);
  };

  /* eslint-disable react-hooks/exhaustive-deps -- intentional: subscribe once; notify is stable */
  useEffect(() => {
    if (!registerFileListener) return;
    return registerFileListener(async (files) => {
      try {
        const { detector: d, restore: r } = optsRef.current;
        const res = await postFiles('/api/facemgr/add', files, { detector: d, restore: String(r) });
        applyPayload(res);
        if (res.video) { setVideo(res.video); setMaxFrames(res.frames || 1); setFrame(1); }
        notify('Loaded files into faceset');
      } catch (err) { notify(err.message, 'error'); }
      return true; // consumed
    });
  }, [registerFileListener]);
  /* eslint-enable react-hooks/exhaustive-deps */

  const onAddFiles = async (e) => {
    if (!e.target.files.length) return;
    setBusy(true);
    try {
      const res = await postFiles('/api/facemgr/add', e.target.files, { detector, restore: String(restore) });
      applyPayload(res);
      if (res.video) { setVideo(res.video); setMaxFrames(res.frames || 1); setFrame(1); }
      notify(restore ? 'Loaded faces (restored)' : 'Loaded faces');
    } catch (err) { notify(err.message, 'error'); }
    finally { setBusy(false); }
    e.target.value = '';
  };

  const onLoadFaceset = async (e) => {
    if (!e.target.files.length) return;
    setBusy(true);
    try { const res = await postFile('/api/facemgr/faceset', e.target.files[0]); applyPayload(res); notify('Faceset loaded'); }
    catch (err) { notify(err.message, 'error'); }
    finally { setBusy(false); }
    e.target.value = '';
  };

  const cut = async () => {
    setBusy(true);
    try { const res = await postJSON('/api/facemgr/cut', { frame, detector, restore }); applyPayload(res); notify('Faces cut from frame'); }
    catch (e) { notify(e.message, 'error'); }
    finally { setBusy(false); }
  };
  const remove = async () => {
    if (sel < 0 || sel >= faces.length) return;
    const res = await postJSON('/api/facemgr/remove', { index: sel });
    applyPayload(res);
    setSel((s) => Math.max(0, Math.min(s, (res.faces || []).length - 1)));
  };
  const clear = async () => {
    const res = await postJSON('/api/facemgr/clear', {});
    applyPayload(res); setVideo(null); setBuilt(null); setSel(0);
  };
  const prune = async () => {
    setBusy(true);
    try {
      const res = await postJSON('/api/facemgr/prune', { threshold });
      applyPayload(res);
      notify(res.removed ? `Removed ${res.removed} low-quality face${res.removed === 1 ? '' : 's'}` : 'Nothing below the threshold');
      setSel(0);
    } catch (e) { notify(e.message, 'error'); }
    finally { setBusy(false); }
  };
  const build = async () => {
    setBusy(true);
    try { const res = await postJSON('/api/facemgr/build', {}); setBuilt(res); notify('Faceset file created'); }
    catch (e) { notify(e.message, 'error'); }
    finally { setBusy(false); }
  };

  // Roll-up over the current set, and how many the gate would drop.
  const stats = useMemo(() => {
    if (!scores.length) return null;
    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
    const below = scores.filter((s) => s < threshold).length;
    return { avg, below };
  }, [scores, threshold]);

  return (
    <div className="space-y-6">
      <Section>
        <h2 className="text-lg font-bold">Create blending facesets</h2>
        <p className="text-sm text-white/50">
          Add multiple reference images of the same person into a single faceset (.fsz) for better likeness.
          Each face is scored for quality (sharpness, resolution, frontality, detector confidence) so you can
          drop the weak ones before building.
        </p>
      </Section>

      {/* Extraction options */}
      <Section title="Extraction options">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-start">
          <label className="block">
            <span className="text-xs font-semibold uppercase tracking-wider text-white/40">Detector</span>
            <select
              value={detector}
              onChange={(e) => setDetector(e.target.value)}
              className="mt-1.5 w-full px-3 py-2 rounded-lg glass-input text-white text-sm focus:outline-none"
            >
              {DETECTORS.map((d) => <option key={d.id} value={d.id} className="bg-[#121420]">{d.label}</option>)}
            </select>
            <span className="mt-1 block text-mini text-white/45">Engine used to find & align faces on add / cut.</span>
          </label>
          <div>
            <Toggle
              label="Restore faces on add"
              info="Clean soft / compressed references through GFPGAN before storing"
              checked={restore}
              onChange={setRestore}
            />
          </div>
          <div>
            <Slider label="Quality gate" info={`keep ≥ ${Math.round(threshold * 100)}%`}
                    min={0} max={1} step={0.05} value={threshold} onChange={setThreshold} />
            <span className="mt-1 block text-mini text-white/45">Faces below this score are dimmed; “Drop below gate” removes them.</span>
          </div>
        </div>
      </Section>

      <div className="grid grid-cols-1 lg:grid-cols-2 3xl:grid-cols-3 gap-6">
        <Section title="Add faces">
          <label className="block cursor-pointer focus-within:outline focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-[var(--accent)]">
            <div className="px-4 py-5 rounded-lg border-2 border-dashed border-white/15 hover:border-[var(--accent)]/50 text-center text-sm text-white/60">Add images / videos</div>
            <input type="file" accept="image/*,video/*" multiple onChange={onAddFiles} className="sr-only" />
          </label>
          <label className="block cursor-pointer focus-within:outline focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-[var(--accent)]">
            <div className="px-4 py-3 rounded-lg border border-white/10 hover:border-white/30 text-center text-sm text-white/50">Load existing faceset (.fsz)</div>
            <input type="file" accept=".fsz" onChange={onLoadFaceset} className="sr-only" />
          </label>
          {video && (
            <div className="space-y-2">
              <div className="rounded-lg overflow-hidden bg-black/40 border border-white/10">
                <img src={`${API}/api/facemgr/frame?frame=${frame}&t=${frame}`} alt="frame" className="w-full" />
              </div>
              <Slider label="Frame" info={`${frame} / ${maxFrames}`} min={1} max={maxFrames} step={1} value={frame} onChange={setFrame} />
              <Button size="sm" variant="secondary" onClick={cut} disabled={busy}>Use faces from this frame</Button>
            </div>
          )}
        </Section>

        <Section title="Faces in faceset">
          {/* Stats + quality gate summary */}
          {stats && (
            <div className="flex items-center gap-2 text-mini text-white/45 mb-2 flex-wrap">
              <span><span className="font-bold text-white/70">{faces.length}</span> face{faces.length === 1 ? '' : 's'}</span>
              <span className="text-white/20">·</span>
              <span>avg quality <span className={`font-bold ${TONE[scoreTone(stats.avg)].text}`}>{Math.round(stats.avg * 100)}%</span></span>
              {threshold > 0 && stats.below > 0 && (
                <>
                  <span className="text-white/20">·</span>
                  <span className="text-amber-400/80">{stats.below} below gate</span>
                </>
              )}
            </div>
          )}

          <FaceQualityGrid faces={faces} scores={scores} meta={meta} threshold={threshold} selected={sel} onSelect={setSel} />

          <div className="flex flex-wrap gap-2 mt-3">
            <Button size="sm" variant="secondary" onClick={remove} disabled={busy || !faces.length}>Remove selected</Button>
            {threshold > 0 && (
              <Button size="sm" variant="stop" onClick={prune} disabled={busy || !(stats && stats.below)}>
                Drop below gate{stats && stats.below ? ` (${stats.below})` : ''}
              </Button>
            )}
            <Button size="sm" variant="primary" onClick={build} disabled={busy || !faces.length}>Create / update faceset</Button>
            <Button size="sm" variant="stop" onClick={clear} disabled={busy || !faces.length}>Clear all</Button>
          </div>
          {built && (
            <a href={fileUrl(built.path)} download={built.name}
              className="inline-block mt-2 text-sm text-[var(--accent)] underline">⬇ Download {built.name}</a>
          )}
        </Section>
      </div>
    </div>
  );
}

// Faces rendered with a per-face FIQA badge; anything under the gate is dimmed
// with a "will drop" flag so the gate is legible before you commit to pruning.
function FaceQualityGrid({ faces, scores, meta, threshold, selected, onSelect }) {
  if (!faces.length) {
    return <div className="text-sm text-white/35 py-8 text-center border border-dashed border-white/10 rounded-lg">No faces yet</div>;
  }
  return (
    <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 gap-2">
      {faces.map((src, i) => {
        const s = scores[i];
        const has = typeof s === 'number';
        const tone = has ? TONE[scoreTone(s)] : TONE.amber;
        const dropped = threshold > 0 && has && s < threshold;
        const b = meta[i] || {};
        const title = has
          ? `Quality ${Math.round(s * 100)}%  ·  ${b.face_px ?? '?'}px  ·  sharp ${b.sharp_var ?? '?'}  ·  front ${Math.round((b.frontality ?? 0) * 100)}%`
          : 'quality unknown';
        return (
          <button
            key={i}
            type="button"
            onClick={() => onSelect(i)}
            title={title}
            className={`relative aspect-square rounded-lg overflow-hidden border transition-all ${
              selected === i ? 'ring-2 ring-[var(--accent)] border-[var(--accent)]/50' : 'border-white/10 hover:border-white/25'
            } ${dropped ? 'opacity-35 grayscale' : ''}`}
          >
            <img src={src} alt={`face ${i + 1}`} className="w-full h-full object-cover" />
            {has && (
              <span className={`absolute top-1 left-1 px-1.5 py-0.5 rounded-md text-nano font-bold tabular-nums ${tone.bg} ${tone.text} backdrop-blur-sm`}>
                {Math.round(s * 100)}%
              </span>
            )}
            {dropped && (
              <span className="absolute bottom-1 left-1 right-1 text-center text-nano font-bold uppercase tracking-wide text-red-300 bg-red-950/70 rounded py-0.5">
                below gate
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
