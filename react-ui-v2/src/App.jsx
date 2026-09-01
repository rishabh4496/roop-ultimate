import React, { lazy, Suspense } from 'react';
import { AppShell } from './components/AppShell';
import { ErrorBoundary } from './components/ErrorBoundary';
import { LoadingState } from './components/LoadingState';
import { useRouter } from './router';

const HomeScreen = lazy(() => import('./screens/HomeScreen'));
const CreateScreen = lazy(() => import('./screens/CreateScreen'));
const SettingsScreen = lazy(() => import('./screens/SettingsScreen'));

const screens = {
  home: HomeScreen,
  create: CreateScreen,
  settings: SettingsScreen,
};

export default function App() {
  const { route, navigate } = useRouter();
  const Screen = screens[route.id] || HomeScreen;

  return (
    <ErrorBoundary>
      <AppShell route={route} onNavigate={navigate}>
        <Suspense fallback={<LoadingState label="Loading workspace" />}>
          <Screen />
        </Suspense>
      </AppShell>
    </ErrorBoundary>
  );
}
