import { useEffect, useState } from 'react';
import { frameSocket } from './frameSocket';

/** Whether /ws/frames is currently open; re-renders only on a transition. */
export function useFrameSocketOpen() {
  const [open, setOpen] = useState(() => frameSocket.isOpen());
  useEffect(() => {
    setOpen(frameSocket.isOpen());
    return frameSocket.onStatus(setOpen);
  }, []);
  return open;
}

/**
 * Keep the frame socket connected while the calling component is mounted
 * (without subscribing to anything), so the first use — pressing Play — does
 * not wait out a handshake. Returns whether it is open.
 */
export function useFrameSocketHold(enabled = true) {
  const open = useFrameSocketOpen();
  useEffect(() => (enabled ? frameSocket.hold() : undefined), [enabled]);
  return open;
}
