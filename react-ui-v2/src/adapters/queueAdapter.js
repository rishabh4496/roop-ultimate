import { getJSON, postJSON } from './apiClient';

export const queueAdapter = {
  getQueue: () => getJSON('/api/queue'),
  addJob: (job) => postJSON('/api/queue/add', job),
  startQueue: () => postJSON('/api/queue/start', {}),
  pauseQueue: () => postJSON('/api/queue/pause', {}),
  resumeQueue: () => postJSON('/api/queue/resume', {}),
  stopQueue: () => postJSON('/api/queue/stop', {}),
  cancelJob: (id) => postJSON('/api/queue/cancel', { id }),
  retryJob: (id) => postJSON('/api/queue/retry', { id }),
  reorderQueue: (ids) => postJSON('/api/queue/reorder', { ids }),
  joinJobs: (ids) => postJSON('/api/queue/join', { ids }),
};
