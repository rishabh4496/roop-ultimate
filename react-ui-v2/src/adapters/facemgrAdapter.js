import { postJSON, postFiles, postFile } from './apiClient';

export const facemgrAdapter = {
  addFiles: (files, detector = 'scrfd', restore = false) =>
    postFiles('/api/facemgr/add', files, { detector, restore: String(restore) }),
  loadFaceset: (file) =>
    postFile('/api/facemgr/faceset', file),
  cutFrame: (frame, detector = 'scrfd', restore = false) =>
    postJSON('/api/facemgr/cut', { frame, detector, restore }),
  removeFace: (index) =>
    postJSON('/api/facemgr/remove', { index }),
  clearFaces: () =>
    postJSON('/api/facemgr/clear', {}),
  pruneFaces: (threshold = 0.5) =>
    postJSON('/api/facemgr/prune', { threshold }),
  buildFaceset: () =>
    postJSON('/api/facemgr/build', {}),
  saveFaceset: (name) =>
    postJSON('/api/facemgr/save', { name }),
};
