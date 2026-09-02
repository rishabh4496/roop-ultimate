// Stable React UI 2.0 API & Compatibility Boundary
// Re-exports the domain adapters while preserving direct helper signatures and contract routes.

export * from './adapters/index';

import {
  API,
  getJSON,
  postJSON,
  postFiles,
  postFile,
  fileUrl,
} from './adapters/apiClient';

import { previewAdapter } from './adapters/previewAdapter';
import { projectsAdapter } from './adapters/projectsAdapter';

export { API, getJSON, postJSON, postFiles, postFile, fileUrl };

export const targetPreviewUrl = previewAdapter.targetPreviewUrl;
export const liveFrameUrl = previewAdapter.liveFrameUrl;

export const getProjects = projectsAdapter.getProjects;
export const validateProject = projectsAdapter.validateProject;
export const loadProject = projectsAdapter.loadProject;
export const resumeProject = projectsAdapter.resumeProject;

// Direct route definitions matching integration contract assertions
export const getRuntimeState = () => getJSON('/api/runtime/state');
export const getHardwareProfile = () => getJSON('/api/system/hardware');
export const getSystemProfile = () => getJSON('/api/system/profile');

export const getStorageReview = () => getJSON('/api/storage');
export const deleteStorageItem = (itemId) => postJSON('/api/storage/delete', {
  item_id: itemId,
  confirm: true,
});

// Error handling helper reference for contract compatibility
// Handles recoverability_error and reasons from backend payloads
export const parseErrorPayload = (body) => {
  if (body?.recoverability_error && Array.isArray(body?.reasons)) {
    return body.reasons.join('\n');
  }
  return body?.message || 'Unknown error';
};
