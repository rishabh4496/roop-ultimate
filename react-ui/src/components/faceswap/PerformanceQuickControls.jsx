import React from 'react';
import { Select, Toggle } from '../ui';
import {
  batchNote, batchOptions, batchPatch, batchValue, decodeOn, decodePatch, encodeNote, encodeOn, encodePatch,
  encodeSwitchable, fp16On, fp16Patch, precisionApplies, zeroCopyOn, zeroCopyPatch,
} from './performanceRules';

// The hardware switches that decide how a render spends the GPU, compact, on #/faceswap:
//
//   Zero-copy restorer buffers   GPU ring for the restorers (perf_zerocopy_enhance)
//   FP16 (TensorRT mixed)        trt_precision mixed <-> fp32
//   Hardware decode (NVDEC)      perf_nvdec
//   Hardware encode (NVENC)      output_video_codec, libx264/libx265 <-> h264_nvenc/hevc_nvenc
//   Micro-batch size             perf_batch_max (Auto / 1 / 2 / 4 / 8)
//
// Detection stride is the pre-pass stride slider in the 3-stage pipeline group above; it is not repeated.
//
// These edit the same settings object as #/settings (the global hardware defaults live there, with the long help),
// so a change in one is the other's value; FaceSwap persists it with the rest of the settings before a run. All of
// them are read per render (settings.LIVE_ENV_SETTINGS) or per job (the codec, the precision), so none needs a restart.
// There is no run button: starting a render stays the one control in the Face Swap run bar.
export default function PerformanceQuickControls({ values, onChangeMany, className = '' }) {
  const p = values || {};
  const precisionNote = precisionApplies(p)
    ? 'Models the precision policy marks unsafe for FP16 (such as XSeg) stay FP32 either way.'
    : 'Only TensorRT sessions use this; the active provider is not TensorRT, so it has no effect.';
  const encNote = encodeNote(p);

  return (
    <div className={`space-y-3 ${className}`} data-testid="performance-quick-controls">
      <Toggle
        label="Zero-copy restorer buffers"
        info="GPEN Realistic and RestoreFormer++ keep their input and output tensors on the GPU in a small ring instead of copying every face through host memory. On by default; turn it off to rule the ring out when comparing outputs."
        checked={zeroCopyOn(p)}
        onChange={(v) => onChangeMany(zeroCopyPatch(v))}
      />

      <div>
        <Toggle
          label="FP16 precision (TensorRT mixed)"
          info="TensorRT runs the models the precision policy allows in FP16 and the rest in FP32. Off = FP32 for every model: slower, and the reference for a fidelity comparison."
          checked={fp16On(p)}
          onChange={(v) => onChangeMany(fp16Patch(v))}
        />
        <p className="mt-1 text-note text-muted" data-testid="fp16-note">{precisionNote}</p>
      </div>

      <Toggle
        label="Hardware decode (NVDEC)"
        info="Decode the video on the GPU. Each file is probed first and falls back to the CPU decoder by itself if NVDEC cannot read it."
        checked={decodeOn(p)}
        onChange={(v) => onChangeMany(decodePatch(v))}
      />

      <div>
        <Toggle
          label="Hardware encode (NVENC)"
          info="Encode the output with the GPU's video encoder: h264_nvenc / hevc_nvenc instead of libx264 / libx265. Same codec family either way."
          checked={encodeOn(p)}
          disabled={!encodeSwitchable(p)}
          onChange={(v) => { const patch = encodePatch(p, v); if (Object.keys(patch).length) onChangeMany(patch); }}
        />
        {encNote && <p className="mt-1 text-note text-muted" data-testid="encode-note">{encNote}</p>}
      </div>

      <div>
        <Select
          label="Micro-batch size"
          info="How many face crops are coalesced into one swap inference. Auto keeps the tiered default and lets the VRAM governor lower it when memory is short."
          value={batchValue(p)}
          onChange={(v) => onChangeMany(batchPatch(v))}
          options={batchOptions(p)}
        />
        <p className="mt-1 text-note text-muted" data-testid="batch-note">{batchNote(p)}</p>
      </div>
    </div>
  );
}
