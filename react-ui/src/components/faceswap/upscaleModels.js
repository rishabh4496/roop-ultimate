// AI upscale models folded into the swap pass (mirrors the Extras post-processor
// list). value = backend subtype, label = friendly name.
export const AI_UPSCALE_MODELS = [
  { value: 'esrganx2', label: 'Real-ESRGAN ×2' },
  { value: 'esrganx4', label: 'Real-ESRGAN ×4' },
  { value: 'esrgan_anime_x4', label: 'Real-ESRGAN Anime ×4' },
  { value: 'ultrasharp_x4', label: 'Ultra-Sharp ×4' },
  { value: 'lsdirx4', label: 'LSDIR ×4' },
  { value: 'clear_reality_x4', label: 'Clear Reality ×4' },
  { value: 'span_x4', label: 'SPAN ×4' },
  { value: 'compact_x4', label: 'Compact ×4 (fast AI)' },
  { value: 'nomos8k_x4', label: 'Nomos8k ×4' },
  { value: 'lanczos_x2', label: 'Fast Lanczos ×2 (no AI)' },
  { value: 'lanczos_x4', label: 'Fast Lanczos ×4 (no AI)' },
  { value: 'fsr_x2', label: 'FSR-lite ×2 · Lanczos+CAS (no AI)' },
  { value: 'fsr_x4', label: 'FSR-lite ×4 · Lanczos+CAS (no AI)' },
  { value: 'spline_x2', label: 'Spline36 ×2 (no AI)' },
  { value: 'spline_x4', label: 'Spline36 ×4 (no AI)' },
  { value: 'sinc_x2', label: 'Sinc ×2 · sharpest (no AI)' },
  { value: 'sinc_x4', label: 'Sinc ×4 · sharpest (no AI)' },
];
