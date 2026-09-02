import { getJSON, postJSON, postFile } from './apiClient';

export const facesetAdapter = {
  getLibrary: () => getJSON('/api/faceset/library'),
  saveToLibrary: (name) => postJSON('/api/faceset/library/save', { name }),
  loadFromLibrary: (filename) => postJSON('/api/faceset/library/load', { filename }),
  deleteFromLibrary: (filename) => postJSON('/api/faceset/library/delete', { filename }),
  renameInLibrary: (filename, name) => postJSON('/api/faceset/library/rename', { filename, name }),
  importFaceset: (file) => postFile('/api/faceset/library/import', file),
  revealLibrary: () => postJSON('/api/faceset/library/reveal', {}),
};
