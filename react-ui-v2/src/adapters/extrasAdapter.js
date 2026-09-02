import { getJSON, postFile } from './apiClient';

export const extrasAdapter = {
  getFrameOps: () => getJSON('/api/extras/frame_ops'),
  enhanceFile: (file, operation, subtype) =>
    postFile('/api/extras/enhance', file, { operation, subtype }),
  applyTransforms: (file, options = {}) =>
    postFile('/api/extras/apply', file, options),
};
