// The compact "Performance" group on #/faceswap, as DATA: which real setting each switch reads and writes, and what
// the switch means when the setting holds an unusual value. Pure functions of the settings object, so the component
// and its render-check read one set of rules.
//
// Nothing here invents a backend setting. Every key exists in app/settings.py and is consumed:
//
//   perf_zerocopy_enhance  tristate  ROOP_ZEROCOPY_ENHANCE  roop/optimized_processor.zero_copy_enabled()  GPU ring for the restorers
//   perf_pinned_buffers    tristate  ROOP_PINNED_BUFFERS     page-locked host buffers for decoded frames / crops
//   trt_precision          mixed|fp32                        roop/precision_policy (the per-model rows still win)
//   perf_nvdec             tristate  ROOP_NVDEC              NVDEC decode, with a per-file probe and automatic cv2 fallback
//   output_video_codec     codec                             NVENC (h264_nvenc / hevc_nvenc) or software (libx264 / libx265)
//   perf_batch_max         auto|N    ROOP_BATCH_SWAP_MAX     cross-frame swap batch ceiling
//   perf_batch_swap        tristate  ROOP_BATCH_SWAP         the batcher itself
//
// Detection stride is NOT repeated here: it is the "Pre-pass optical-flow stride (K)" slider in the 3-stage pipeline
// group directly above (prepass_detect_interval, 1..8). There is no "Smart Auto" stride: nothing in the backend picks
// K, and a control that said so would be a label over a fixed number.
//
// FP16: the UI says "FP16" because that is what the TensorRT `mixed` mode is (FP16 where the policy allows it, FP32
// for the models it marks unsafe). A global FP16-everywhere request was retired (settings.TRT_PRECISION_CHOICES), so
// the switch is mixed <-> fp32, never a third value.

export const BATCH_CHOICES = ['auto', '1', '2', '4', '8'];

// A tristate setting reads as ON unless it is explicitly 'off' ('auto' = the shipped default, which is on for these).
const triOn = (v) => String(v ?? 'auto').trim().toLowerCase() !== 'off';
// Writing ON as 'auto' (not 'on') keeps "leave the environment alone" meaning what it did before the switch was touched.
const triPatch = (key, on) => ({ [key]: on ? 'auto' : 'off' });

export const zeroCopyOn = (s) => triOn((s || {}).perf_zerocopy_enhance);
export const zeroCopyPatch = (on) => triPatch('perf_zerocopy_enhance', !!on);

export const pinnedOn = (s) => triOn((s || {}).perf_pinned_buffers);
export const pinnedPatch = (on) => triPatch('perf_pinned_buffers', !!on);

export const decodeOn = (s) => triOn((s || {}).perf_nvdec);
export const decodePatch = (on) => triPatch('perf_nvdec', !!on);

/** FP16 (TensorRT mixed) is on unless the global precision is fp32. Anything unrecognised reads as the default, mixed. */
export const fp16On = (s) => String((s || {}).trt_precision ?? 'mixed').trim().toLowerCase() !== 'fp32';
export const fp16Patch = (on) => ({ trt_precision: on ? 'mixed' : 'fp32' });
/** The precision setting only reaches TensorRT sessions; on any other provider the switch would do nothing. */
export const precisionApplies = (s) => String((s || {}).provider ?? '').trim().toLowerCase() === 'tensorrt';

// ── hardware encode ──────────────────────────────────────────────────────────────────────────────────────────────
const HW_ENCODER = /(nvenc|qsv|amf|videotoolbox)/i;
const SW_TO_NVENC = { libx264: 'h264_nvenc', libx265: 'hevc_nvenc' };
const NVENC_TO_SW = { h264_nvenc: 'libx264', hevc_nvenc: 'libx265' };

/**
 * Where the codec sits: 'nvenc' | 'hardware' (QSV/AMF/VideoToolbox) | 'software' | 'other' (a format with no NVENC
 * twin, or an unset value). Only the libx264/libx265 <-> NVENC pairs can be flipped by a switch, because they are the
 * only pairs that keep the codec family (H.264 / HEVC) the same.
 */
export function encoderOf(s) {
  const codec = String((s || {}).output_video_codec ?? '').trim().toLowerCase();
  if (/nvenc/.test(codec)) return 'nvenc';
  if (HW_ENCODER.test(codec)) return 'hardware';
  if (codec in SW_TO_NVENC) return 'software';
  return 'other';
}
export const encodeOn = (s) => ['nvenc', 'hardware'].includes(encoderOf(s));
/** Can the switch change this codec without changing its family? */
export function encodeSwitchable(s) {
  const kind = encoderOf(s);
  return kind === 'nvenc' || kind === 'software';
}
/** The patch for turning hardware encode on/off, `{}` when the codec has no counterpart (never a guess). */
export function encodePatch(s, on) {
  const codec = String((s || {}).output_video_codec ?? '').trim().toLowerCase();
  const next = on ? SW_TO_NVENC[codec] : NVENC_TO_SW[codec];
  return next ? { output_video_codec: next } : {};
}
export function encodeNote(s) {
  const kind = encoderOf(s);
  if (kind === 'nvenc') return null;
  if (kind === 'hardware') return 'A non-NVENC hardware encoder is selected; change it in Settings > Output.';
  if (kind === 'other') return 'This codec has no NVENC twin; change it in Settings > Output.';
  return null;
}

// ── micro-batch ──────────────────────────────────────────────────────────────────────────────────────────────────
/** 'auto' or a positive integer as a string; anything else reads as 'auto'. */
export function batchValue(s) {
  const raw = (s || {}).perf_batch_max;
  if (raw === undefined || raw === null) return 'auto';
  const text = String(raw).trim().toLowerCase();
  if (text === '' || text === 'auto') return 'auto';
  const n = Number(text);
  return Number.isInteger(n) && n >= 1 ? String(n) : 'auto';
}
/** The option list, plus the current value when it is a number outside the short list (never hide what is set). */
export function batchOptions(s) {
  const cur = batchValue(s);
  return BATCH_CHOICES.includes(cur) ? BATCH_CHOICES : [...BATCH_CHOICES, cur];
}
export const batchPatch = (v) => ({ perf_batch_max: String(v ?? 'auto').trim().toLowerCase() === 'auto' ? 'auto' : String(Number(v)) });
/** The batcher can be switched off on its own; a ceiling does nothing then, and the control should say so. */
export const batchingOff = (s) => String((s || {}).perf_batch_swap ?? 'auto').trim().toLowerCase() === 'off';
export function batchNote(s) {
  if (batchingOff(s)) return 'Batched swap is off (Settings > Advanced performance), so this ceiling has no effect.';
  const v = batchValue(s);
  if (v === 'auto') return 'Auto: up to 8 on 12 GB+ cards, 4 below; the VRAM governor lowers it per render if memory is short.';
  if (v === '1') return 'Batching is effectively off: one face crop per swap call.';
  return `Up to ${v} face crops per swap call; the VRAM governor can still lower it per render.`;
}
