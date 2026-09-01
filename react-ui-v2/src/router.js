import { useCallback, useEffect, useState } from 'react';

export const ROUTES = Object.freeze([
  { id: 'home', label: 'Overview', path: '#/home', description: 'Foundation overview' },
  { id: 'create', label: 'Create', path: '#/create', description: 'Media-first creation workflow' },
  { id: 'settings', label: 'Settings', path: '#/settings', description: 'Appearance and foundation settings' },
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
