import { useEffect, useSyncExternalStore } from 'react';
import { getStatus, subscribe, startPipelineStatus } from './pipelineStatus';

/** The backend's current compile / canary state, re-rendering when it changes. */
export function usePipelineStatus() {
  return useSyncExternalStore(subscribe, getStatus, getStatus);
}

/** Mount once, in the app shell: owns the status socket for the page's lifetime. */
export function usePipelineStatusSocket() {
  useEffect(() => startPipelineStatus(), []);
}
