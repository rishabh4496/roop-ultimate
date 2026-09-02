import { API, postJSON } from './apiClient';

export const previewAdapter = {
  renderPreview: (payload, opts) => postJSON('/api/preview', payload, opts),
  renderUpscalePreview: (baseImage, subtype) =>
    postJSON('/api/preview_upscale', { image: baseImage, subtype }),

  targetPreviewUrl: (index, frame = 1, width) =>
    `${API}/api/target/preview?index=${index}&frame=${frame}${width ? `&width=${width}` : ''}`,
  targetPreviewGridUrl: (index, count = 20) =>
    `${API}/api/target/preview_grid?index=${index}&count=${count}`,
  targetPreviewSeqUrl: (index, start = 1, count = 10) =>
    `${API}/api/target/preview_seq?index=${index}&start=${start}&count=${count}`,
  liveFrameUrl: (seq) =>
    `${API}/api/live_frame?seq=${encodeURIComponent(seq)}`,
};
