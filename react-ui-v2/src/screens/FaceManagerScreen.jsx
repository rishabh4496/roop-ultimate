import React, { useState } from 'react';
import { postJSON, postFiles } from '../api';
import { useNotifications } from '../state/appState';
import { Badge, Button, Card, Select, Toggle } from '../components/primitives';

const DETECTORS = [
  { value: 'scrfd', label: 'SCRFD 640 (Default)' },
  { value: 'retinaface', label: 'RetinaFace 10G' },
  { value: 'retinaface_r50', label: 'RetinaFace R50' },
  { value: 'yoloface', label: 'YOLOFace' },
  { value: 'yunet', label: 'YuNet' },
];

export default function FaceManagerScreen() {
  const { notify } = useNotifications();
  const [faces, setFaces] = useState([]);
  const [scores, setScores] = useState([]);
  const [selectedFace, setSelectedFace] = useState(0);
  const [detector, setDetector] = useState('scrfd');
  const [restore, setRestore] = useState(false);
  const [busy, setBusy] = useState(false);

  const onAddFiles = async (e) => {
    if (!e.target.files?.length) return;
    setBusy(true);
    try {
      const res = await postFiles('/api/facemgr/add', e.target.files, {
        detector,
        restore: String(restore),
      });
      setFaces(res.faces || []);
      setScores(res.scores || []);
      notify(`Extracted ${res.faces?.length || 0} faces`, 'success');
    } catch (err) {
      notify(err.message, 'danger');
    } finally {
      setBusy(false);
    }
  };

  const clearFaces = async () => {
    try {
      await postJSON('/api/facemgr/clear', {});
      setFaces([]);
      setScores([]);
      setSelectedFace(0);
      notify('Cleared face manager workbench', 'success');
    } catch (err) {
      notify(err.message, 'danger');
    }
  };

  const saveFaceset = async () => {
    const name = window.prompt('Enter a name for this faceset archive:');
    if (!name) return;
    try {
      await postJSON('/api/faceset/library/save', { name });
      notify(`Saved faceset "${name}.fsz"`, 'success');
    } catch (err) {
      notify(err.message, 'danger');
    }
  };

  return (
    <div className="v2-screen">
      <div className="v2-creation-header">
        <div className="flex items-center gap-3 flex-wrap">
          <span className="v2-eyebrow">Identity Workbench</span>
          <h2>Face & Embedding Manager</h2>
          <Badge tone="accent">{faces.length} Harvested Faces</Badge>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" onClick={clearFaces} disabled={!faces.length}>
            Clear
          </Button>
          <Button variant="primary" size="sm" onClick={saveFaceset} disabled={!faces.length}>
            Save to Faceset (.fsz)
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[340px_1fr] gap-4 items-start">
        {/* Controls Column */}
        <div className="flex flex-col gap-3">
          <Card>
            <div className="v2-card-heading">
              <div>
                <span className="v2-eyebrow">Extraction Parameters</span>
                <h3>Face Harvester</h3>
              </div>
            </div>

            <div className="flex flex-col gap-3">
              <Select
                label="Detector Backbone"
                value={detector}
                onChange={(e) => setDetector(e.target.value)}
                options={DETECTORS}
              />

              <Toggle
                label="Restore Crops with CodeFormer"
                checked={restore}
                onChange={setRestore}
                hint="Enhance extracted face thumbnails before embedding calculation"
              />

              <label className="v2-upload mt-2">
                <input type="file" multiple accept="image/*,video/*" onChange={onAddFiles} />
                <span className="v2-upload-icon">+</span>
                <span>
                  <strong>Harvest Faces from Media</strong>
                  <small>Images or Video Frames</small>
                </span>
              </label>
            </div>
          </Card>
        </div>

        {/* Harvested Face Grid */}
        <Card className="flex flex-col gap-3">
          <div className="v2-card-heading">
            <div>
              <span className="v2-eyebrow">Harvested Faces</span>
              <h3>Quality-Scored Face Crops</h3>
            </div>
          </div>

          {busy ? (
            <div className="min-h-[260px] flex flex-col items-center justify-center gap-2 text-xs text-[var(--muted)]">
              <div className="v2-spinner" />
              <span>Detecting and aligning faces...</span>
            </div>
          ) : faces.length > 0 ? (
            <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2.5 max-h-[calc(100vh-280px)] overflow-y-auto pr-1">
              {faces.map((faceSrc, idx) => {
                const isSelected = selectedFace === idx;
                const score = scores[idx] != null ? Number(scores[idx]) : null;
                const scoreTone = score >= 0.6 ? 'text-emerald-400' : score >= 0.4 ? 'text-amber-400' : 'text-red-400';

                return (
                  <div
                    key={`face-harvest-${idx}`}
                    onClick={() => setSelectedFace(idx)}
                    className={`group relative flex flex-col items-center rounded border p-1 cursor-pointer transition-all ${
                      isSelected
                        ? 'border-[var(--accent)] bg-[var(--accent-soft)] shadow-[0_0_10px_rgba(56,189,248,0.25)]'
                        : 'border-[var(--border)] bg-[var(--raised)] hover:border-white/20'
                    }`}
                  >
                    <img src={faceSrc} alt={`Face #${idx + 1}`} decoding="async" className="w-full aspect-square object-cover rounded pointer-events-none" />
                    <div className="w-full flex items-center justify-between mt-1 px-0.5 text-[9px] font-mono">
                      <span className="text-[var(--muted)]">#{idx + 1}</span>
                      {score !== null && <span className={`font-bold ${scoreTone}`}>{(score * 100).toFixed(0)}%</span>}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="min-h-[220px] flex items-center justify-center text-xs text-[var(--muted)] text-center p-6">
              <p>No faces harvested yet. Upload images or video clips above to extract face crops.</p>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
