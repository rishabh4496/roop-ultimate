import React, { useState, useEffect, useRef, useMemo } from 'react';
import { getJSON, postJSON, postFiles, postFile, fileUrl, API } from '../api';
import { Section, Slider, Toggle, Button } from './ui';
import MorphLab from './facemgr/MorphLab';
import PoseRadar from './facemgr/PoseRadar';
import IdentityWorkstation from './facemgr/IdentityWorkstation';
import { VirtualGrid } from './virtual/VirtualScroll';
import { Icon } from '../icons';

export const POSE_BUCKET_CONFIG = {
  frontal: { label: 'Frontal', short: 'Front', color: '#10b981', bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/40' },
  left_profile: { label: 'Left Profile', short: 'L-Prof', color: '#a855f7', bg: 'bg-purple-500/20', text: 'text-purple-400', border: 'border-purple-500/40' },
  right_profile: { label: 'Right Profile', short: 'R-Prof', color: '#3b82f6', bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/40' },
  high_angle: { label: 'High Angle', short: 'High', color: '#f59e0b', bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/40' },
  low_angle: { label: 'Low Angle', short: 'Low', color: '#06b6d4', bg: 'bg-cyan-500/20', text: 'text-cyan-400', border: 'border-cyan-500/40' },
};

export const getPoseBucket = (m = {}) => {
  if (m.bucket && POSE_BUCKET_CONFIG[m.bucket]) return m.bucket;
  const yaw = Number(m.yaw || 0);
  const pitch = Number(m.pitch || 0);
  if (Math.abs(yaw) >= 40) {
    return yaw < 0 ? 'left_profile' : 'right_profile';
  }
  if (Math.abs(pitch) >= 20) {
    return pitch > 0 ? 'high_angle' : 'low_angle';
  }
  return 'frontal';
};

export const getFaceSharpness = (m = {}) => {
  if (typeof m.sharpness === 'number' && Number.isFinite(m.sharpness)) {
    return Math.max(0, Math.min(1, m.sharpness));
  }
  if (typeof m.sharp_var === 'number' && Number.isFinite(m.sharp_var)) {
    return Math.max(0, Math.min(1, m.sharp_var / 300));
  }
  return null;
};

// The two jobs this screen does: build a faceset from footage, or blend existing identities into a new one.
const VIEWS = [
  { id: 'workstation', label: 'Identity workstation' },
  { id: 'builder', label: 'Faceset builder' },
  { id: 'morph', label: 'Morph Lab' },
];

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

export default function FaceManager({ notify, registerFileListener, setTab }) {
  const [view, setView] = useState('workstation');
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

  // The set lives on the server; this component only mirrors it. Re-read it on mount so leaving the tab and coming
  // back (which unmounts this component) does not show an empty set while the server still holds the faces.
  useEffect(() => {
    let live = true;
    getJSON('/api/facemgr/faces').then((res) => { if (live && res) applyPayload(res); }).catch(() => {});
    return () => { live = false; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps -- mount only; applyPayload only sets state

  /* eslint-disable react-hooks/exhaustive-deps -- intentional: subscribe once; notify is stable */
  useEffect(() => {
    if (!registerFileListener) return;
    return registerFileListener(async (files) => {
      try {
        const { detector: d, restore: r } = optsRef.current;
        const res = await postFiles('/api/facemgr/add', files, { detector: d, restore: String(r) }, { timeout: 0 });
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
      const res = await postFiles('/api/facemgr/add', e.target.files, { detector, restore: String(restore) }, { timeout: 0 });
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
    try { const res = await postFile('/api/facemgr/faceset', e.target.files[0], undefined, { timeout: 0 }); applyPayload(res); notify('Faceset loaded'); }
    catch (err) { notify(err.message, 'error'); }
    finally { setBusy(false); }
    e.target.value = '';
  };

  const cut = async () => {
    setBusy(true);
    try { const res = await postJSON('/api/facemgr/cut', { frame, detector, restore }, { timeout: 0 }); applyPayload(res); notify('Faces cut from frame'); }
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

  const [activeBucketFilter, setActiveBucketFilter] = useState(null);

  // Per-angle pose bucket distribution & sharpness scores
  const poseDistribution = useMemo(() => {
    if (!faces.length) return null;
    const buckets = {
      frontal: { count: 0, sumSharp: 0 },
      left_profile: { count: 0, sumSharp: 0 },
      right_profile: { count: 0, sumSharp: 0 },
      high_angle: { count: 0, sumSharp: 0 },
      low_angle: { count: 0, sumSharp: 0 },
    };
    for (let i = 0; i < faces.length; i++) {
      const m = meta[i] || {};
      const b = getPoseBucket(m);
      const target = buckets[b] || buckets.frontal;
      target.count += 1;
      const s = getFaceSharpness(m);
      if (s !== null) {
        target.sumSharp += s;
      }
    }
    const total = faces.length;
    return Object.entries(buckets).map(([key, data]) => {
      const count = data.count;
      const pct = total > 0 ? Math.round((count / total) * 100) : 0;
      const avgSharp = count > 0 ? Math.round((data.sumSharp / count) * 100) : null;
      return {
        key,
        ...POSE_BUCKET_CONFIG[key],
        count,
        pct,
        avgSharp,
        missing: count === 0,
      };
    });
  }, [faces, meta]);

  // The route is a full-bleed workspace (App gives facemgr `is-workspace`): the tab strip is fixed and the active view
  // owns its own scrolling, so the workstation can fill the viewport while the builder and Morph Lab still scroll.
  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="facemgr-root">
      <div role="tablist" aria-label="Face Manager" className="m-2 flex gap-1 rounded-xl border border-white/10 bg-white/[0.03] p-1 w-fit shrink-0">
        {VIEWS.map((v) => (
          <button
            key={v.id} type="button" role="tab" id={`facemgr-tab-${v.id}`} aria-selected={view === v.id}
            aria-controls={`facemgr-panel-${v.id}`} data-testid={`facemgr-tab-${v.id}`}
            onClick={() => setView(v.id)}
            className={`rounded-lg px-3 py-1.5 text-sm font-semibold transition-colors ${
              view === v.id ? 'bg-[var(--accent)]/15 text-accent' : 'text-white/60 hover:text-white hover:bg-white/[0.06]'}`}
          >
            {v.label}
          </button>
        ))}
      </div>

      {view === 'workstation' ? (
        <div role="tabpanel" id="facemgr-panel-workstation" aria-labelledby="facemgr-tab-workstation" className="min-h-0 flex-1">
          <IdentityWorkstation notify={notify} setTab={setTab} facesetState={{ faces, scores, meta }} onFacesetChange={applyPayload} />
        </div>
      ) : view === 'morph' ? (
        <div role="tabpanel" id="facemgr-panel-morph" aria-labelledby="facemgr-tab-morph" className="min-h-0 flex-1 overflow-auto p-3">
          <MorphLab notify={notify} setTab={setTab} />
        </div>
      ) : (
      <div role="tabpanel" id="facemgr-panel-builder" aria-labelledby="facemgr-tab-builder" className="min-h-0 flex-1 overflow-auto p-3 space-y-4">
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
            <span className="text-xs font-semibold uppercase tracking-wider text-muted">Detector</span>
            <select
              value={detector}
              onChange={(e) => setDetector(e.target.value)}
              className="mt-1.5 w-full px-3 py-2 rounded-lg glass-input text-white text-sm focus:outline-none"
            >
              {DETECTORS.map((d) => <option key={d.id} value={d.id} className="bg-[#121420]">{d.label}</option>)}
            </select>
            <span className="mt-1 block text-mini text-muted">Engine used to find & align faces on add / cut.</span>
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
            <span className="mt-1 block text-mini text-muted">Faces below this score are dimmed; “Drop below gate” removes them.</span>
          </div>
        </div>
      </Section>

      <div className="grid grid-cols-1 lg:grid-cols-2 3xl:grid-cols-3 gap-4">
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
            <div className="flex items-center gap-2 text-mini text-muted mb-2 flex-wrap">
              <span><span className="font-bold text-white/70">{faces.length}</span> face{faces.length === 1 ? '' : 's'}</span>
              <span className="text-white/20" aria-hidden="true">·</span>
              <span>avg quality <span className={`font-bold ${TONE[scoreTone(stats.avg)].text}`}>{Math.round(stats.avg * 100)}%</span></span>
              {threshold > 0 && stats.below > 0 && (
                <>
                  <span className="text-white/20" aria-hidden="true">·</span>
                  <span className="text-amber-400/80">{stats.below} below gate</span>
                </>
              )}
            </div>
          )}

          <PoseRadar
            refreshKey={`${faces.length}:${scores.join(',')}`}
            selected={sel}
            onSelect={setSel}
            onPruned={(res) => { applyPayload(res); setSel(0); }}
            notify={notify}
            busy={busy}
          />

          {/* Pose Bucket Distribution & Sharpness Summary Bar */}
          {poseDistribution && (
            <div className="rounded-xl border border-white/10 bg-black/30 p-3 space-y-2.5" data-testid="pose-distribution-panel">
              <div className="flex items-center justify-between text-mini flex-wrap gap-1">
                <span className="font-bold text-white/90 uppercase tracking-wider text-nano flex items-center gap-1.5">
                  <Icon.meter size={12} className="text-accent" />
                  <span>Pose Bucket Distribution &amp; Sharpness</span>
                </span>
                <span className="text-nano text-white/50 font-mono">
                  {faces.length} total crops · {poseDistribution.filter((b) => !b.missing).length} / 5 angles covered
                </span>
              </div>

              {/* Proportional Segment Distribution Bar */}
              <div className="relative h-2.5 w-full overflow-hidden rounded-full bg-white/5 flex items-stretch border border-white/10">
                {poseDistribution.map((b) => (
                  <div
                    key={b.key}
                    style={{ width: `${b.pct}%`, backgroundColor: b.color }}
                    className="h-full transition-all duration-200"
                    title={`${b.label}: ${b.count} faces (${b.pct}%) · Avg Sharpness: ${b.avgSharp !== null ? `${b.avgSharp}%` : '—'}`}
                  />
                ))}
              </div>

              {/* Angle filter pills with count, percentage and sharpness */}
              <div className="flex flex-wrap gap-1.5 text-nano font-mono">
                {poseDistribution.map((b) => {
                  const isSelected = activeBucketFilter === b.key;
                  return (
                    <button
                      key={b.key}
                      type="button"
                      onClick={() => setActiveBucketFilter(isSelected ? null : b.key)}
                      className={`px-2 py-1 rounded-lg border transition-all flex items-center gap-1.5 cursor-pointer ${
                        isSelected
                          ? 'ring-2 ring-[var(--accent)] border-accent bg-accent/20 text-white font-bold'
                          : b.missing
                          ? 'border-amber-500/30 bg-amber-950/20 text-amber-300'
                          : 'border-white/10 bg-white/[0.03] text-white/80 hover:bg-white/10'
                      }`}
                      title={`Click to ${isSelected ? 'clear filter' : `filter by ${b.label}`}`}
                    >
                      <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: b.color }} />
                      <span className="font-bold font-sans">{b.label}:</span>
                      <span>{b.count} ({b.pct}%)</span>
                      {b.avgSharp !== null ? (
                        <span className="text-emerald-400 font-semibold">· {b.avgSharp}% sharp</span>
                      ) : (
                        <span className="text-amber-400 font-semibold">· missing</span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          <FaceQualityGrid
            faces={faces}
            scores={scores}
            meta={meta}
            threshold={threshold}
            selected={sel}
            onSelect={setSel}
            activeBucketFilter={activeBucketFilter}
          />

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
              className="inline-block mt-2 text-sm text-accent underline">⬇ Download {built.name}</a>
          )}
        </Section>
      </div>
      </div>
      )}
    </div>
  );
}

// Faces rendered with a per-face FIQA badge, pose bucket tag, and sharpness score;
// anything under the gate is dimmed with a "will drop" flag so the gate is legible before you commit to pruning.
function FaceQualityGrid({ faces, scores, meta, threshold, selected, onSelect, activeBucketFilter = null }) {
  if (!faces.length) {
    return <div className="text-sm text-muted py-8 text-center border border-dashed border-white/10 rounded-lg">No faces yet</div>;
  }
  // Windowed: a 500-crop faceset mounts the ~5 rows in view (each crop is an <img> with a base64 payload), so
  // selecting, scoring and scrolling it stay cheap however large the set gets. Tiles are square, so a row is as tall
  // as a cell is wide.
  return (
    <VirtualGrid
      items={faces}
      minCell={84}
      gap={8}
      cellHeight={(w) => Math.round(w)}
      getKey={(_src, i) => i}
      overscan={4}
      maxCols={16}
      // p-1: the selection ring and the keyboard focus ring (2px + 2px offset) sit outside a tile, and the scroller
      // clips its children to its own box.
      className="max-h-[60vh] p-1"
      label="Faces in the faceset"
      testId="face-grid"
      renderCell={(src, i) => {
        const s = scores[i];
        const has = typeof s === 'number';
        const tone = has ? TONE[scoreTone(s)] : TONE.amber;
        const dropped = threshold > 0 && has && s < threshold;
        const b = meta[i] || {};
        const bucketKey = getPoseBucket(b);
        const bucketConf = POSE_BUCKET_CONFIG[bucketKey] || POSE_BUCKET_CONFIG.frontal;
        const sharpness = getFaceSharpness(b);
        const hasSharp = sharpness !== null;
        const isDimmedByFilter = activeBucketFilter && bucketKey !== activeBucketFilter;

        const title = has
          ? `Face ${i + 1} · ${bucketConf.label} (Yaw: ${b.yaw ?? 0}°, Pitch: ${b.pitch ?? 0}°) · Sharpness: ${hasSharp ? `${Math.round(sharpness * 100)}%` : '—'} (var: ${b.sharp_var ?? '—'}) · Quality: ${Math.round(s * 100)}% · ${b.face_px ?? '?'}px`
          : 'quality unknown';
        return (
          <button
            type="button"
            onClick={() => onSelect(i)}
            title={title}
            className={`relative aspect-square rounded-lg overflow-hidden border transition-all ${
              selected === i ? 'ring-2 ring-[var(--accent)] border-[var(--accent)]/50' : 'border-white/10 hover:border-white/25'
            } ${dropped || isDimmedByFilter ? 'opacity-30 grayscale' : ''}`}
          >
            <img src={src} alt={`face ${i + 1}`} className="w-full h-full object-cover" />
            {/* Top-Left: Composite Quality Score */}
            {has && (
              <span className={`absolute top-1 left-1 px-1.5 py-0.5 rounded-md text-nano font-bold tabular-nums ${tone.bg} ${tone.text} backdrop-blur-sm`}>
                {Math.round(s * 100)}%
              </span>
            )}
            {/* Top-Right: Pose Bucket Badge */}
            <span
              className={`absolute top-1 right-1 px-1 py-0.5 rounded text-nano font-bold uppercase tracking-wider ${bucketConf.bg} ${bucketConf.text} border ${bucketConf.border} backdrop-blur-sm`}
            >
              {bucketConf.short}
            </span>
            {/* Bottom-Right: Sharpness Score */}
            {hasSharp && (
              <span
                className="absolute bottom-1 right-1 px-1 py-0.5 rounded text-nano font-mono font-bold bg-black/80 text-emerald-300 border border-emerald-500/30 backdrop-blur-sm"
              >
                S:{Math.round(sharpness * 100)}%
              </span>
            )}
            {dropped && (
              <span className="absolute bottom-1 left-1 right-1 text-center text-nano font-bold uppercase tracking-wide text-red-300 bg-red-950/70 rounded py-0.5 z-10">
                below gate
              </span>
            )}
          </button>
        );
      }}
    />
  );
}
