import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { getJSON, postFiles, postJSON } from '../api';

const numberOr = (value, fallback) => {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
};

function optionList(values) {
  return Array.isArray(values) ? values.map((value) => ({ value, label: String(value) })) : [];
}

export function useCreationWorkflow(notify) {
  const [meta, setMeta] = useState(null);
  const [settings, setSettings] = useState(null);
  const [state, setState] = useState(null);
  const [progress, setProgress] = useState(null);
  const [runtime, setRuntime] = useState(null);
  const [output, setOutput] = useState(null);
  const [selectedSource, setSelectedSource] = useState(0);
  const [frame, setFrame] = useState(1);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState('');
  const [upload, setUpload] = useState(null);
  const [error, setError] = useState('');
  const pollRef = useRef(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [m, s, st, p] = await Promise.all([
        getJSON('/api/meta'),
        getJSON('/api/settings'),
        getJSON('/api/state'),
        getJSON('/api/progress'),
      ]);
      setMeta(m); setSettings(s); setState(st); setProgress(p); setRuntime(p.runtime || null); setError('');
      const target = st.targets?.[st.selected_target_index || 0];
      setFrame(target?.start_frame || 1);
      return st;
    } catch (cause) {
      setError(cause.message || 'Backend unavailable');
      throw cause;
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshState = useCallback(async () => {
    const next = await getJSON('/api/state');
    setState(next);
    const target = next.targets?.[next.selected_target_index || 0];
    if (target && frame < (target.start_frame || 1)) setFrame(target.start_frame || 1);
    return next;
  }, [frame]);

  const refreshProgress = useCallback(async () => {
    const next = await getJSON('/api/progress');
    setProgress(next);
    setRuntime(next.runtime || null);
    if (!next.processing && next.output?.path) setOutput(next.output);
    return next;
  }, []);

  useEffect(() => { refresh().catch(() => {}); }, [refresh]);

  useEffect(() => {
    if (!progress?.processing) return undefined;
    pollRef.current = window.setInterval(() => { refreshProgress().catch(() => {}); }, 1000);
    return () => window.clearInterval(pollRef.current);
  }, [progress?.processing, refreshProgress]);

  useEffect(() => {
    if (!progress || progress.processing || !progress.output?.path) return;
    getJSON('/api/output').then(setOutput).catch(() => {});
  }, [progress]);

  const setSetting = useCallback((key, value) => {
    setSettings((current) => ({ ...current, [key]: value }));
  }, []);

  const uploadMedia = useCallback(async (kind, files) => {
    if (!files?.length) return;
    setBusy(kind);
    setUpload({ kind, phase: 'upload', percent: 0 });
    try {
      const response = await postFiles(`/api/${kind}/add`, files, (percent, phase) => setUpload({ kind, percent, phase }));
      setState((current) => ({ ...current, ...response }));
      await refreshState();
      notify(`${kind === 'source' ? 'Source face' : 'Target media'} added`, 'success');
    } catch (cause) {
      notify(cause.message, 'danger');
    } finally {
      setBusy('');
      setUpload(null);
    }
  }, [notify, refreshState]);

  const selectSource = useCallback(async (index) => {
    setBusy('source-select');
    try { await postJSON('/api/source/select', { index }); setSelectedSource(index); await refreshState(); }
    catch (cause) { notify(cause.message, 'danger'); }
    finally { setBusy(''); }
  }, [notify, refreshState]);

  const selectTarget = useCallback(async (index) => {
    setBusy('target-select');
    try {
      const response = await postJSON('/api/target/select', { index });
      setState((current) => ({ ...current, ...response, selected_target_index: index }));
      setFrame(response.targets?.[index]?.start_frame || 1);
      setPreview(null);
    } catch (cause) { notify(cause.message, 'danger'); }
    finally { setBusy(''); }
  }, [notify]);

  const buildPayload = useCallback((fakePreview = false) => {
    const p = settings || {};
    const targetIndex = state?.selected_target_index || 0;
    return {
      ...p,
      index: targetIndex,
      frame,
      fake_preview: fakePreview,
      enhancer: p.selected_enhancer,
      detection: p.face_detection_mode,
      output_method: p.output_method,
      video_method: p.video_swapping_method,
      upscale: p.subsample_upscale,
      mask_engine: p.mask_engine,
      mask_engine_2: p.mask_engine_2,
      clip_text: p.mask_clip_text,
      sam2_model_size: p.sam2_model_size,
      track_identities: p.track_identities,
      autorotate: p.autorotate_faces,
      face_distance: numberOr(p.max_face_distance, 0.75),
      blend_ratio: numberOr(p.blend_ratio, 0.8),
      num_swap_steps: numberOr(p.num_swap_steps, 1),
      color_transfer_mode: p.color_transfer_mode,
      face_detector_threshold: p.face_detector_threshold,
      face_detector_nms: p.face_detector_nms,
      face_mapping: [],
    };
  }, [frame, settings, state?.selected_target_index]);

  const renderPreview = useCallback(async () => {
    if (!state?.targets?.length) return notify('Add target media before previewing', 'danger');
    setBusy('preview');
    try {
      const response = await postJSON('/api/preview', buildPayload(Boolean(state?.source_faces?.length)));
      setPreview(response);
      if (response.error) notify(response.error, 'danger');
    } catch (cause) { notify(cause.message, 'danger'); }
    finally { setBusy(''); }
  }, [buildPayload, notify, state?.source_faces?.length, state?.targets?.length]);

  const start = useCallback(async () => {
    if (!state?.targets?.length) return notify('Add target media before generating', 'danger');
    if (!state?.source_faces?.length) return notify('Add a source face before generating', 'danger');
    setBusy('start');
    try {
      await postJSON('/api/settings', settings || {});
      await postJSON('/api/swap', buildPayload(false));
      await refreshProgress();
      notify('Generation started', 'success');
    } catch (cause) { notify(cause.message, 'danger'); }
    finally { setBusy(''); }
  }, [buildPayload, notify, refreshProgress, settings, state?.source_faces?.length, state?.targets?.length]);

  const stop = useCallback(async () => {
    setBusy('stop');
    try { await postJSON('/api/stop', {}); await refreshProgress(); notify('Stopping generation', 'info'); }
    catch (cause) { notify(cause.message, 'danger'); }
    finally { setBusy(''); }
  }, [notify, refreshProgress]);

  const pause = useCallback(async () => {
    setBusy('pause');
    try {
      await postJSON('/api/pause', {});
      await refreshProgress();
      notify('Pause requested; waiting for a safe checkpoint', 'info');
    } catch (cause) { notify(cause.message, 'danger'); }
    finally { setBusy(''); }
  }, [notify, refreshProgress]);

  const resume = useCallback(async () => {
    setBusy('resume');
    try { await postJSON('/api/resume', {}); await refreshProgress(); notify('Generation resumed', 'success'); }
    catch (cause) { notify(cause.message, 'danger'); }
    finally { setBusy(''); }
  }, [notify, refreshProgress]);

  const selectedTarget = state?.targets?.[state?.selected_target_index || 0];
  const options = useMemo(() => ({
    providers: optionList(meta?.providers), precisions: optionList(meta?.trt_precisions),
    models: optionList(meta?.swap_models), enhancers: optionList(meta?.enhancers),
    detectors: optionList(meta?.detector_engines), detectionModes: optionList(meta?.face_detection_modes),
    masks: optionList(meta?.mask_engines), colors: optionList(meta?.color_transfer_modes),
    upscales: optionList(meta?.upscale), outputs: optionList(meta?.output_methods),
    videoMethods: optionList(meta?.video_methods), videoFormats: optionList(meta?.video_formats),
    videoCodecs: optionList(meta?.video_codecs), noFace: optionList(meta?.no_face_actions),
  }), [meta]);

  return {
    meta, settings, state, progress, runtime, output, preview, selectedTarget, selectedSource, frame, setFrame,
    setSetting, upload, loading, busy, error, options, refresh, refreshState,
    uploadMedia, selectSource, selectTarget, renderPreview, start, stop, pause, resume, buildPayload,
  };
}
