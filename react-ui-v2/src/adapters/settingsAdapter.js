import { getJSON, postJSON } from './apiClient';

export const settingsAdapter = {
  getSettings: () => getJSON('/api/settings'),
  saveSettings: (patch, opts) => postJSON('/api/settings', patch, opts),
  getDefaults: () => getJSON('/api/settings/defaults'),
  getBenchmarkStatus: () => getJSON('/api/settings/benchmark_status'),
  runThreadBenchmark: (threads = [4, 8, 12, 16], runs = 3) =>
    postJSON('/api/settings/benchmark_threads', { threads, runs }),
  cancelBenchmark: () => postJSON('/api/settings/benchmark_cancel', {}),
  getHardwareInfo: () => getJSON('/api/system/hardware'),
  getSystemProfile: () => getJSON('/api/system/profile'),
  getTelemetry: () => getJSON('/api/system/telemetry'),
  getRuntimeState: () => getJSON('/api/runtime/state'),
};
