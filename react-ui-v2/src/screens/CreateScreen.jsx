import React, { useMemo, useState } from 'react';
import { fileUrl, liveFrameUrl, targetPreviewUrl } from '../api';
import { useNotifications } from '../state/appState';
import { Badge, Button, Card, Field, Notice, Progress, Select, Toggle } from '../components/primitives';
import { LoadingState } from '../components/LoadingState';
import { QueuePanel } from '../components/QueuePanel';
import { useCreationWorkflow } from '../workflow/useCreationWorkflow';
import { useQueue } from '../workflow/useQueue';

function Unavailable({ label, reason = 'Unavailable until the backend exposes this option.' }) {
  return <div className="v2-unavailable"><span>{label}</span><Badge tone="neutral">Unavailable</Badge><small>{reason}</small></div>;
}

function BackendSelect({ label, hint, value, onChange, options }) {
  if (!options.length) return <Unavailable label={label} />;
  return <Select label={label} hint={hint} value={value ?? options[0].value} onChange={(event) => onChange(event.target.value)} options={options} />;
}

function RangeControl({ label, value, fallback, min, max, step, onChange, hint }) {
  const current = Number.isFinite(Number(value)) ? Number(value) : fallback;
  return <Field label={label} hint={hint}><div className="v2-range-line"><input className="v2-range" type="range" value={current} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} /><output>{current}</output></div></Field>;
}

function RuntimeTelemetry({ runtime }) {
  if (!runtime) return null;
  const frame = runtime.frame_progress || {};
  const vram = runtime.vram || {};
  const value = (item) => item == null ? 'UNKNOWN' : String(item);
  const frameLabel = frame.done !== 'UNKNOWN' && frame.total !== 'UNKNOWN' ? `${value(frame.done)} / ${value(frame.total)}` : value(frame.fraction);
  const vramLabel = vram.used_gb !== 'UNKNOWN' && vram.total_gb !== 'UNKNOWN' ? `${value(vram.used_gb)} / ${value(vram.total_gb)} GB` : 'UNKNOWN';
  return <div className="v2-runtime-telemetry" aria-label="Structured runtime telemetry">
    <div><span>Status</span><strong>{value(runtime.status?.message || runtime.status?.code)}</strong></div>
    <div><span>Frames</span><strong>{frameLabel}</strong></div>
    <div><span>FPS / ETA</span><strong>{value(runtime.fps)} / {value(runtime.eta_s)}{runtime.eta_s !== 'UNKNOWN' ? ' s' : ''}</strong></div>
    <div><span>Provider / precision</span><strong>{value(runtime.provider)} / {value(runtime.precision)}</strong></div>
    <div><span>GPU</span><strong title={value(runtime.gpu)}>{value(runtime.gpu)}</strong></div>
    <div><span>VRAM</span><strong>{vramLabel}</strong></div>
  </div>;
}

function MediaUpload({ kind, busy, upload, onFiles }) {
  const isSource = kind === 'source';
  return <label className="v2-upload"><input type="file" accept={isSource ? 'image/*,.fsz' : 'image/*,video/*,.webp'} multiple onChange={(event) => { onFiles(event.target.files); event.target.value = ''; }} /><span className="v2-upload-icon">+</span><span><strong>{isSource ? 'Add source face' : 'Add target media'}</strong><small>{busy === kind ? `${upload?.phase === 'analyzing' ? 'Analyzing' : 'Uploading'} ${upload?.percent || 0}%` : isSource ? 'Image or .fsz faceset' : 'Image, video, GIF, or WebP'}</small></span></label>;
}

function SourcePicker({ workflow }) {
  const faces = workflow.state?.source_faces || [];
  const selected = workflow.selectedSource;
  return <Card className="v2-picker-card"><div className="v2-card-heading"><div><span className="v2-eyebrow">Source</span><h3>Who should appear?</h3></div><Badge tone="neutral">{faces.length} loaded</Badge></div><MediaUpload kind="source" busy={workflow.busy} upload={workflow.upload} onFiles={(files) => workflow.uploadMedia('source', files)} />{faces.length ? <div className="v2-media-grid">{faces.map((face, index) => <button type="button" className={`v2-face-tile ${selected === index ? 'is-selected' : ''}`} key={`${face}-${index}`} onClick={() => workflow.selectSource(index)}><img src={face} alt={`Source face ${index + 1}`} /><span>Face {index + 1}</span></button>)}</div> : <div className="v2-empty-picker">Add a clear face image to begin.</div>}</Card>;
}

function TargetPicker({ workflow }) {
  const targets = workflow.state?.targets || [];
  const selected = workflow.state?.selected_target_index ?? 0;
  return <Card className="v2-picker-card"><div className="v2-card-heading"><div><span className="v2-eyebrow">Target</span><h3>Where should it go?</h3></div><Badge tone="neutral">{targets.length} loaded</Badge></div><MediaUpload kind="target" busy={workflow.busy} upload={workflow.upload} onFiles={(files) => workflow.uploadMedia('target', files)} />{targets.length ? <div className="v2-target-list">{targets.map((target, index) => <button type="button" className={`v2-target-row ${selected === index ? 'is-selected' : ''}`} key={`${target.name}-${index}`} onClick={() => workflow.selectTarget(index)}><img src={targetPreviewUrl(index, target.start_frame || 1)} alt="" /><span><strong>{target.name}</strong><small>{target.frames > 1 ? `${target.frames} frames - ${target.fps || '?'} fps` : 'Still image'}</small></span></button>)}</div> : <div className="v2-empty-picker">Add an image or video as the target.</div>}</Card>;
}

function CreationControls({ workflow }) {
  const { settings: p = {}, options } = workflow;
  const [advanced, setAdvanced] = useState(false);
  return <div className="v2-controls">
    <Card><div className="v2-card-heading"><div><span className="v2-eyebrow">Model and provider</span><h3>Choose the engine</h3></div><Badge tone="accent">Backend-backed</Badge></div><BackendSelect label="Provider" hint="Provider changes are applied by the existing backend settings contract." value={p.provider} onChange={(value) => workflow.setSetting('provider', value)} options={options.providers} /><BackendSelect label="Swap model" value={p.swap_model} onChange={(value) => workflow.setSetting('swap_model', value)} options={options.models} />{p.provider === 'tensorrt' && <BackendSelect label="TensorRT precision" hint="The backend applies this at session startup." value={p.trt_precision} onChange={(value) => workflow.setSetting('trt_precision', value)} options={options.precisions} />}</Card>
    <Card><div className="v2-card-heading"><div><span className="v2-eyebrow">Quality</span><h3>Good defaults, visible control</h3></div><Badge tone="accent">Ready</Badge></div><BackendSelect label="Enhancer" value={p.selected_enhancer} onChange={(value) => workflow.setSetting('selected_enhancer', value)} options={options.enhancers} /><RangeControl label="Original / enhanced blend" value={p.blend_ratio} fallback={0.8} min={0} max={1} step={0.01} onChange={(value) => workflow.setSetting('blend_ratio', value)} /><RangeControl label="Face similarity threshold" hint="Lower values are stricter." value={p.max_face_distance} fallback={0.75} min={0.01} max={1} step={0.01} onChange={(value) => workflow.setSetting('max_face_distance', value)} /><BackendSelect label="Preview / output upscale" value={p.subsample_upscale} onChange={(value) => workflow.setSetting('subsample_upscale', value)} options={options.upscales} /></Card>
    <Card><div className="v2-card-heading"><div><span className="v2-eyebrow">Output</span><h3>What should be delivered?</h3></div></div><BackendSelect label="Output method" value={p.output_method} onChange={(value) => workflow.setSetting('output_method', value)} options={options.outputs} /><BackendSelect label="Video format" value={p.output_video_format} onChange={(value) => workflow.setSetting('output_video_format', value)} options={options.videoFormats} /><BackendSelect label="Video codec" value={p.output_video_codec} onChange={(value) => workflow.setSetting('output_video_codec', value)} options={options.videoCodecs} /><RangeControl label="Video quality" value={p.video_quality} fallback={14} min={0} max={100} step={1} onChange={(value) => workflow.setSetting('video_quality', value)} /></Card>
    <Card className="v2-advanced-card"><button type="button" className="v2-disclosure" onClick={() => setAdvanced((open) => !open)} aria-expanded={advanced}><span><span className="v2-eyebrow">Advanced options</span><strong>Detection, masks, color, and video behavior</strong></span><span>{advanced ? 'Hide -' : 'Show +'}</span></button>{advanced && <div className="v2-advanced-content"><BackendSelect label="Face selection" value={p.face_detection_mode} onChange={(value) => workflow.setSetting('face_detection_mode', value)} options={options.detectionModes} /><BackendSelect label="Detector engine" value={p.detector_engine} onChange={(value) => workflow.setSetting('detector_engine', value)} options={options.detectors} /><RangeControl label="Detection threshold" value={p.face_detector_threshold} fallback={0.6} min={0.1} max={0.9} step={0.05} onChange={(value) => workflow.setSetting('face_detector_threshold', value)} /><RangeControl label="Overlap NMS threshold" value={p.face_detector_nms} fallback={0.4} min={0.1} max={0.9} step={0.05} onChange={(value) => workflow.setSetting('face_detector_nms', value)} /><BackendSelect label="Mask engine" value={p.mask_engine} onChange={(value) => workflow.setSetting('mask_engine', value)} options={options.masks} /><BackendSelect label="Color / lighting match" value={p.color_transfer_mode} onChange={(value) => workflow.setSetting('color_transfer_mode', value)} options={options.colors} /><BackendSelect label="No-face action" value={p.no_face_action} onChange={(value) => workflow.setSetting('no_face_action', value)} options={options.noFace} /><Toggle label="Refine alignment (68-point)" checked={!!p.refine_landmarks} onChange={(value) => workflow.setSetting('refine_landmarks', value)} /><Toggle label="Rescue small faces" checked={!!p.rescue_small_faces} onChange={(value) => workflow.setSetting('rescue_small_faces', value)} /><Toggle label="Lock face identities in video" checked={!!p.track_identities} onChange={(value) => workflow.setSetting('track_identities', value)} /><Toggle label="Auto-rotate faces" checked={p.autorotate_faces !== false} onChange={(value) => workflow.setSetting('autorotate_faces', value)} /><Toggle label="VR mode" checked={!!p.vr_mode} onChange={(value) => workflow.setSetting('vr_mode', value)} /><Unavailable label="Batch matrix recipes" reason="Not exposed by the verified backend workflow." /></div>}</Card>
  </div>;
}

function PreviewPanel({ workflow }) {
  const [failedLiveSeq, setFailedLiveSeq] = useState(0);
  const targetIndex = workflow.state?.selected_target_index ?? 0;
  const hasTarget = Boolean(workflow.selectedTarget);
  const runtime = workflow.runtime;
  const liveSeq = Number(runtime?.frame_progress?.live_seq || workflow.progress?.live_seq || 0);
  const liveSrc = liveSeq > 0 && liveSeq !== failedLiveSeq ? liveFrameUrl(liveSeq) : '';
  const previewSrc = liveSrc || workflow.preview?.image || (hasTarget ? targetPreviewUrl(targetIndex, workflow.frame) : '');
  const frameMax = workflow.selectedTarget?.frames || 1;
  const isProcessing = runtime?.job?.processing ?? workflow.progress?.processing;
  const fraction = typeof runtime?.frame_progress?.fraction === 'number' ? runtime.frame_progress.fraction : Number(workflow.progress?.progress || 0);
  const statusMessage = runtime?.status?.message || workflow.progress?.desc || 'Generating';
  const completedOutput = workflow.output?.files?.[0];
  const completedPath = workflow.progress?.output?.path || (workflow.output?.output_path && completedOutput ? `${workflow.output.output_path}/${completedOutput.name}` : '');
  return <Card className="v2-preview-card"><div className="v2-preview-heading"><div><span className="v2-eyebrow">Preview</span><h2>{hasTarget ? workflow.selectedTarget.name : 'Your media will appear here'}</h2></div><div className="v2-preview-actions"><Badge tone={liveSrc || workflow.preview ? 'success' : 'neutral'}>{liveSrc ? 'Live processed frame' : workflow.preview ? 'Preview ready' : 'Original frame'}</Badge><Button size="sm" onClick={workflow.renderPreview} disabled={!hasTarget || workflow.busy === 'preview' || isProcessing}>{workflow.busy === 'preview' ? 'Rendering...' : 'Preview swap'}</Button></div></div><div className="v2-preview-stage">{previewSrc ? <img src={previewSrc} alt={liveSrc ? 'Latest processed frame' : 'Target preview'} onError={() => { if (liveSrc) setFailedLiveSeq(liveSeq); }} /> : <div className="v2-preview-empty"><span>O</span><strong>Drop target media to start</strong><small>The preview stays the primary workspace.</small></div>}{workflow.busy === 'preview' && <div className="v2-preview-overlay"><LoadingState label="Rendering preview" /></div>}</div>{hasTarget && <div className="v2-frame-control"><Field label={`Frame ${workflow.frame} of ${frameMax}`}><input className="v2-range" type="range" min={1} max={frameMax} step={1} value={Math.min(workflow.frame, frameMax)} onChange={(event) => workflow.setFrame(Number(event.target.value))} /></Field></div>}{liveSrc && <div className="v2-preview-note"><Badge tone="success">Live</Badge><span>Latest processed frame from the main pipeline; updates are sequence-gated.</span></div>}{workflow.preview?.faces?.length > 0 && <div className="v2-preview-note"><Badge tone="success">{workflow.preview.faces.length} face(s) detected</Badge><span>Detection is provided by the backend preview response.</span></div>}{isProcessing && <div className="v2-progress-panel"><Progress value={Math.round(fraction * 100)} label={statusMessage} /><Button variant="danger" size="sm" onClick={workflow.stop} disabled={workflow.busy === 'stop'}>Stop generation</Button></div>}<RuntimeTelemetry runtime={runtime} />{completedPath && !isProcessing && <div className="v2-output-result"><Badge tone="success">Output ready</Badge><a href={fileUrl(completedPath)} target="_blank" rel="noreferrer">Open latest output</a></div>}</Card>;
}

export default function CreateScreen() {
  const { notify } = useNotifications();
  const workflow = useCreationWorkflow(notify);
  const queue = useQueue(notify);
  const [showUnavailable, setShowUnavailable] = useState(false);
  const connectionLabel = workflow.error ? 'Backend unavailable' : workflow.loading ? 'Connecting' : 'Backend connected';
  const connectionTone = workflow.error ? 'danger' : workflow.loading ? 'neutral' : 'success';
  const targetCount = workflow.state?.targets?.length || 0;
  const sourceCount = workflow.state?.source_faces?.length || 0;
  const ready = sourceCount > 0 && targetCount > 0;
  const unavailable = useMemo(() => ['Resume checkpoints', 'Pinokio controls', 'Hardware/GPU selection'], []);

  const addCurrentToQueue = async () => {
    if (!ready) return notify('Add one source face and one target before queueing', 'danger');
    const sourceInfo = workflow.state?.source_faces_info?.[workflow.selectedSource];
    try {
      await queue.add({ target_name: workflow.selectedTarget.name, source_index: workflow.selectedSource, source_name: sourceInfo?.name || '', label: workflow.selectedTarget.name, payload: workflow.buildPayload(false) });
      notify('Generation added to the queue', 'success');
    } catch { /* useQueue reports the backend error */ }
  };

  if (workflow.loading && !workflow.meta) return <LoadingState label="Connecting to the existing backend" />;
  return <div className="v2-create-screen"><div className="v2-creation-header"><div><span className="v2-eyebrow">Creation workflow</span><h2>Make the moment yours.</h2><p>Select a source, choose a target, check the frame, and generate when it looks right.</p></div><Badge tone={connectionTone}>{connectionLabel}</Badge></div>{workflow.error && <Notice tone="danger" title="No backend connection">Controls are disabled until the existing FastAPI backend is available. V2 does not simulate results.</Notice>}<div className="v2-creation-layout"><div className="v2-preview-column"><PreviewPanel workflow={workflow} /></div><aside className="v2-control-column"><SourcePicker workflow={workflow} /><TargetPicker workflow={workflow} /><CreationControls workflow={workflow} /></aside></div><div className="v2-generation-bar"><div><span className="v2-eyebrow">Ready to generate</span><strong>{ready ? 'Your setup is ready.' : 'Add one source face and one target.'}</strong></div><div className="v2-generation-actions"><Button size="lg" onClick={addCurrentToQueue} disabled={!ready || queue.busy === 'add'}>{queue.busy === 'add' ? 'Queueing...' : 'Add to queue'}</Button><Button variant="primary" size="lg" onClick={workflow.start} disabled={!ready || workflow.busy === 'start' || Boolean(workflow.progress?.processing)}>{workflow.busy === 'start' ? 'Starting...' : workflow.progress?.processing ? 'Generating...' : 'Generate'}</Button></div></div><QueuePanel queue={queue} onCancel={queue.cancel} onRetry={queue.retry} onRemove={queue.remove} onReorder={queue.reorder} onStart={queue.start} onPause={queue.pause} onResume={queue.resume} onStop={queue.stop} /><div className="v2-unavailable-summary"><button type="button" className="v2-text-button" onClick={() => setShowUnavailable((open) => !open)}>{showUnavailable ? 'Hide unavailable capabilities' : 'What is not connected yet?'}</button>{showUnavailable && <div className="v2-unavailable-list">{unavailable.map((item) => <Unavailable key={item} label={item} reason="Deferred; no verified backend operation is being invented." />)}</div>}</div></div>;
}
