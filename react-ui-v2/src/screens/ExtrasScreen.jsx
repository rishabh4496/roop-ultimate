import React, { useEffect, useState } from 'react';
import { getJSON, postFile } from '../api';
import { useNotifications } from '../state/appState';
import { Badge, Button, Card, Field, Select } from '../components/primitives';

const OP_LABELS = {
  upscale: {
    esrganx2: 'Real-ESRGAN ×2',
    esrganx4: 'Real-ESRGAN ×4',
    esrgan_anime_x4: 'Real-ESRGAN Anime ×4',
    ultrasharp_x4: 'Ultra-Sharp ×4',
    lsdirx4: 'LSDIR ×4',
    clear_reality_x4: 'Clear Reality ×4',
    span_x4: 'SPAN ×4',
    compact_x4: 'Compact ×4 (Fast AI)',
    nomos8k_x4: 'Nomos8k ×4',
  },
  colorize: {
    deoldify_artistic: 'DeOldify (Artistic)',
    deoldify_stable: 'DeOldify (Stable)',
  },
  filter: {
    stylize: 'Stylize',
    detailenhance: 'Detail Enhance',
    pencil: 'Pencil Sketch',
    cartoon: 'Cartoon',
    C64: 'C64 Palette',
  },
};

const OP_TITLES = {
  upscale: 'AI Super-Resolution Upscaler',
  colorize: 'DeOldify B&W Colorizer',
  filter: 'Neural & Stylization Filter',
};

export default function ExtrasScreen() {
  const { notify } = useNotifications();
  const [file, setFile] = useState(null);
  const [fileUrlSrc, setFileUrlSrc] = useState('');
  const [frameOps, setFrameOps] = useState(null);
  const [operation, setOperation] = useState('upscale');
  const [subtype, setSubtype] = useState('esrganx2');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);

  useEffect(() => {
    getJSON('/api/extras/frame_ops')
      .then(setFrameOps)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!frameOps) return;
    const list = frameOps[operation] || [];
    if (!list.includes(subtype)) setSubtype(list[0] || '');
  }, [operation, frameOps]);

  const onPickFile = (e) => {
    const f = e.target?.files?.[0];
    if (f) {
      setFile(f);
      setResult(null);
      if (fileUrlSrc) URL.revokeObjectURL(fileUrlSrc);
      setFileUrlSrc(URL.createObjectURL(f));
    }
  };

  const runEnhance = async () => {
    if (!file) {
      notify('Select an image or video to enhance', 'danger');
      return;
    }
    setBusy(true);
    setResult(null);
    try {
      const res = await postFile('/api/extras/enhance', file, { operation, subtype });
      setResult(res);
      notify('Enhancement completed', 'success');
    } catch (e) {
      notify(e.message, 'danger');
    } finally {
      setBusy(false);
    }
  };

  const currentOptions = (frameOps?.[operation] || []).map((key) => ({
    value: key,
    label: OP_LABELS[operation]?.[key] || key,
  }));

  return (
    <div className="v2-screen">
      <div className="v2-creation-header">
        <div className="flex items-center gap-3 flex-wrap">
          <span className="v2-eyebrow">Neural Enhancement Studio</span>
          <h2>AI Upscaler & Post-Processor</h2>
          <Badge tone="accent">{OP_TITLES[operation] || operation}</Badge>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-4 items-start">
        {/* Controls Column */}
        <div className="flex flex-col gap-3">
          {/* File Selector */}
          <Card>
            <div className="v2-card-heading">
              <div>
                <span className="v2-eyebrow">Input Media</span>
                <h3>Select Image to Enhance</h3>
              </div>
            </div>
            <label className="v2-upload">
              <input type="file" accept="image/*" onChange={onPickFile} />
              <span className="v2-upload-icon">+</span>
              <span>
                <strong>{file ? file.name : 'Choose input image'}</strong>
                <small>{file ? `${(file.size / 1024).toFixed(1)} KB` : 'JPEG, PNG, WebP'}</small>
              </span>
            </label>
          </Card>

          {/* Operation Selector */}
          <Card>
            <div className="v2-card-heading">
              <div>
                <span className="v2-eyebrow">Enhancement Model</span>
                <h3>Processing Pipeline</h3>
              </div>
            </div>

            <div className="flex flex-col gap-3">
              <Field label="Operation Mode">
                <div className="grid grid-cols-3 gap-1">
                  {[
                    { id: 'upscale', label: 'AI Upscale' },
                    { id: 'colorize', label: 'Colorize' },
                    { id: 'filter', label: 'Filter' },
                  ].map((op) => (
                    <button
                      key={op.id}
                      type="button"
                      onClick={() => setOperation(op.id)}
                      className={`px-2 py-1.5 text-xs font-semibold rounded border transition-all ${
                        operation === op.id
                          ? 'border-[var(--accent)] bg-[var(--accent)] text-black font-bold'
                          : 'border-[var(--border)] bg-[var(--surface)] text-[var(--muted)] hover:border-white/20'
                      }`}
                    >
                      {op.label}
                    </button>
                  ))}
                </div>
              </Field>

              {currentOptions.length > 0 && (
                <Select
                  label="Neural Architecture / Subtype"
                  value={subtype}
                  onChange={(e) => setSubtype(e.target.value)}
                  options={currentOptions}
                />
              )}

              <Button
                variant="primary"
                size="lg"
                onClick={runEnhance}
                disabled={!file || busy}
                className="mt-2"
              >
                {busy ? 'Processing Enhancement...' : 'Run Enhancement'}
              </Button>
            </div>
          </Card>
        </div>

        {/* Preview / Results Column */}
        <Card className="flex flex-col gap-3">
          <div className="v2-card-heading">
            <div>
              <span className="v2-eyebrow">Visual Stage</span>
              <h3>Before / After Comparison</h3>
            </div>
            {result && <Badge tone="success">Processing Finished</Badge>}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 min-h-[400px]">
            {/* Input Preview */}
            <div className="flex flex-col gap-1.5">
              <span className="text-[10px] font-mono text-[var(--muted)] uppercase">Input Original</span>
              <div className="w-full flex-1 rounded border border-[var(--border)] bg-[#080a0e] flex items-center justify-center overflow-hidden p-2 min-h-[300px]">
                {fileUrlSrc ? (
                  <img src={fileUrlSrc} alt="Original input" className="max-h-[360px] object-contain rounded" />
                ) : (
                  <span className="text-xs text-[var(--muted)]">No input selected</span>
                )}
              </div>
            </div>

            {/* Result Preview */}
            <div className="flex flex-col gap-1.5">
              <span className="text-[10px] font-mono text-[var(--muted)] uppercase">Enhanced Output</span>
              <div className="w-full flex-1 rounded border border-[var(--border)] bg-[#080a0e] flex items-center justify-center overflow-hidden p-2 min-h-[300px]">
                {busy ? (
                  <div className="flex flex-col items-center gap-2 text-xs text-[var(--muted)]">
                    <div className="v2-spinner" />
                    <span>Inference in progress...</span>
                  </div>
                ) : result?.image ? (
                  <img src={result.image} alt="Enhanced result" className="max-h-[360px] object-contain rounded" />
                ) : (
                  <span className="text-xs text-[var(--muted)]">Result will appear here</span>
                )}
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
