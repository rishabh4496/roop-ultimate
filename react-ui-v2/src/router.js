import { useCallback, useEffect, useState } from 'react';

export const ROUTES = Object.freeze([
  { id: 'home', label: 'Overview', path: '#/home', description: 'Workstation status' },
  { id: 'create', label: 'Studio', path: '#/create', description: 'Media workstation & live preview' },
  { id: 'batch', label: 'Batch Matrix', path: '#/batch', description: 'Multi-source x multi-target queue' },
  { id: 'facemgr', label: 'Face Manager', path: '#/facemgr', description: 'Face harvester & embedding bench' },
  { id: 'extras', label: 'AI Enhancers', path: '#/extras', description: 'Neural upscaling & colorization' },
  { id: 'gallery', label: 'Outputs', path: '#/gallery', description: 'Rendered media library' },
  { id: 'history', label: 'History', path: '#/history', description: 'Execution telemetry & run logs' },
  { id: 'settings', label: 'Settings', path: '#/settings', description: 'Hardware & inference parameters' },
]);

const routeById = new Map(ROUTES.map((route) => [route.id, route]));

function routeFromHash(hash) {
  const rawId = (hash || '').replace(/^#\//, '').split('/')[0] || 'home';
  const id = rawId === 'workspace' ? 'create' : rawId;
  return routeById.get(id) || ROUTES[0];
}

export function useRouter() {
  const [route, setRoute] = useState(() => routeFromHash(window.location.hash));

  useEffect(() => {
    const onHashChange = () => setRoute(routeFromHash(window.location.hash));
    window.addEventListener('hashchange', onHashChange);
    if (!window.location.hash) window.history.replaceState(null, '', ROUTES[0].path);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const navigate = useCallback((id) => {
    const next = routeById.get(id);
    if (next && window.location.hash !== next.path) window.location.hash = next.path;
  }, []);

  return { route, navigate, routes: ROUTES };
}
