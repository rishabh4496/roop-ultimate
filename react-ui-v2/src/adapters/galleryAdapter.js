import { getJSON, postJSON } from './apiClient';

export const galleryAdapter = {
  getOutputs: () => getJSON('/api/output'),
  deleteOutput: (name) => postJSON('/api/output/delete', { name }),
  revealFolderOrFile: (path) => postJSON('/api/reveal', path ? { path } : {}),
  getHistory: () => getJSON('/api/history'),
  deleteHistoryItem: (id) => postJSON('/api/history/delete', { id }),
  clearHistory: () => postJSON('/api/history/clear', {}),
  getExportPresets: () => getJSON('/api/export/presets'),
  applyExportPreset: (source, preset) => postJSON('/api/export/apply', { source, preset }),
};
