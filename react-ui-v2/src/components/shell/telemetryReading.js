// What the header's telemetry pill reads out of one /api/system/telemetry sample.
//
// Pure, so the render-check can feed it the shapes the backend really sends
// (with and without nvidia-smi) without mounting anything.
//
// `gpu_util` / `gpu_temp` / `gpu_power` come from nvidia-smi (routes_diagnostics
// _nvidia_smi_stats) and are ABSENT without it. They are never defaulted to 0:
// a GPU that is reported at 0% and a GPU nobody can measure are different things,
// so utilisation is null when the driver did not report it and the trace falls
// back to VRAM fill.

export const pct = (used, total) => (total > 0 ? Math.min(100, Math.max(0, (used / total) * 100)) : null);

export function readingOf(sys) {
  if (!sys || typeof sys !== 'object') return null;
  const vram = pct(Number(sys.vram_used), Number(sys.vram_total));
  const util = typeof sys.gpu_util === 'number' ? sys.gpu_util : null;
  return {
    gpu: util,
    vram,
    cpu: typeof sys.cpu_percent === 'number' ? sys.cpu_percent : null,
    primary: util ?? vram,
  };
}
