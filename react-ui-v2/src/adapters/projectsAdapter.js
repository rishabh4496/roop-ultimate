import { getJSON, postJSON } from './apiClient';

export const projectsAdapter = {
  getProjects: () => getJSON('/api/projects'),
  validateProject: (id) => postJSON(`/api/projects/${encodeURIComponent(id)}/validate`, {}),
  loadProject: (id) => postJSON(`/api/projects/${encodeURIComponent(id)}/load`, {}),
  resumeProject: (id) => postJSON(`/api/projects/${encodeURIComponent(id)}/resume`, {}),
};
