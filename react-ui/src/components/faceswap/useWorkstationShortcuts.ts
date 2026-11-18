/**
 * useWorkstationShortcuts.ts: Pro Keyboard Shortcuts & Shuttle Ergonomics for FaceSwap Workstation.
 *
 * Capabilities:
 *   1. Transport & Shuttle Ergonomics (J / K / L):
 *      - 'Space': Play / Pause toggle with browser scroll suppression (WCAG 2.1.4 compliant).
 *      - 'J': Reverse shuttle (-1x, -2x, -4x ladder) via smooth rAF sub-frame clock.
 *      - 'K': Shuttle pause / stop (returns to 1x normal speed).
 *      - 'L': Forward shuttle (1x, 2x, 4x ladder) driving playback rate.
 *   2. Frame Stepping:
 *      - 'Left' / 'Right': Step exactly 1 frame backward / forward.
 *      - 'Shift + Left' / 'Shift + Right': Step 10 frames backward / forward.
 *   3. NLE In / Out Trim & Chapter Markers:
 *      - 'I': Set trim In point at current frame.
 *      - 'O': Set trim Out point at current frame.
 *      - 'M': Add / toggle chapter marker at current playhead frame.
 *   4. Instant Hold-to-Peek Original:
 *      - '\\': Hold to peek unswapped original frame, restore on release (keydown/keyup/blur).
 *   5. Form & Modal Input Guarding:
 *      - Bypasses shortcuts when typing in input, textarea, select, or contenteditable.
 *      - Bypasses shortcuts when modal dialogs ([role="dialog"]) are active.
 *      - Prevents Space from scrolling while preserving button/tab activation.
 *   6. Workstation Wiring & Parity:
 *      - Coordinates track selection, marker jumps, and trim synchronization between
 *        FaceSwap.jsx, NleTimeline.tsx, and PreviewViewport.tsx.
 *   7. 100% Offline Compliance: zero external URLs, fonts, or network CDNs.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { eventMatches, keyupMatches } from '../../store/shortcutStore.js';
import {
  IDLE,
  SHUTTLE_SPEEDS,
  reverseTick,
  shuttleKey,
  shuttleEndsOnPlayChange,
  shuttleLabel,
  shuttleRate,
  shuttleSync,
  stepSize,
} from './timeline/shuttle.js';

// ── Types & Interfaces ─────────────────────────────────────────────────────────

export interface ShuttleState {
  dir: number;   // -1 = reverse, 0 = idle/paused, 1 = forward
  level: number; // 0 = 1x, 1 = 2x, 2 = 4x
}

export interface WorkstationShortcutsOptions {
  // Current playback position & bounds
  frame: number;
  setFrame: (f: number | ((prev: number) => number)) => void;
  maxFrames: number;
  startFrame?: number;
  endFrame?: number;

  // Playback state & video controls
  isPlaying: boolean;
  setIsPlaying: (p: boolean | ((prev: boolean) => boolean)) => void;
  fps?: number;
  setPlaybackRate?: (rate: number) => void;
  videoRef?: React.RefObject<HTMLVideoElement | null>;

  // Trim In / Out points
  trimIn?: number;
  trimOut?: number;
  onUpdateTrim?: (trimIn: number, trimOut: number) => void;
  setFrameMarkerVal?: (which: 'start' | 'end', frame: number) => void | Promise<void>;

  // Markers
  markers?: number[];
  targetKey?: string;
  onToggleMarker?: (frame: number) => void;
  onMarkersChange?: (markers: number[]) => void;

  // Hold-to-peek original unswapped frame ('\')
  isPeekingOriginal?: boolean;
  setIsPeekingOriginal?: (peeking: boolean | ((prev: boolean) => boolean)) => void;

  // Track & Face Selection
  selectedTrackId?: number | string | null;
  selectedFaceIndex?: number | null;
  onSelectTrack?: (trackId: number | string | null) => void;
  onSelectFace?: (faceIndex: number | null) => void;

  // Single-frame image / still compare fallback
  onToggleCompare?: () => void;

  // Master switch & listener attachment
  enabled?: boolean;
  attachListener?: boolean;
}

export interface UseWorkstationShortcutsReturn {
  // Shuttle state
  shuttle: ShuttleState;
  setShuttle: (s: ShuttleState | ((prev: ShuttleState) => ShuttleState)) => void;
  shuttleRateValue: number;
  shuttleLabelText: string;

  // Markers
  markers: number[];
  toggleMarkerAtCurrent: (frame?: number) => void;
  jumpMarker: (dir: -1 | 1) => void;
  clearMarkers: () => void;

  // Trim
  trimIn: number;
  trimOut: number;
  setTrimInAtCurrent: () => void;
  setTrimOutAtCurrent: () => void;
  resetTrim: () => void;

  // Stepping
  stepFrame: (delta: number) => void;

  // Peek
  isPeekingOriginal: boolean;

  // Track selection
  selectTrack: (trackId: number | string | null) => void;
  selectFace: (faceIndex: number | null) => void;

  // Manual event dispatchers
  handleKeyDown: (e: KeyboardEvent | any) => boolean;
  handleKeyUp: (e: KeyboardEvent | any) => boolean;
}

// ── Pure Helpers (Input Guards & Math) ─────────────────────────────────────────

/**
 * Check if the active element is a text input, textarea, select, or contenteditable.
 */
export function isInputFocused(): boolean {
  if (typeof document === 'undefined') return false;
  const el = document.activeElement;
  if (!el) return false;
  const tag = el.tagName ? el.tagName.toUpperCase() : '';
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
    return true;
  }
  if ((el as HTMLElement).isContentEditable) {
    return true;
  }
  return false;
}

/**
 * Check if a modal dialog is currently visible.
 */
export function isModalOpen(): boolean {
  if (typeof document === 'undefined') return false;
  return Boolean(document.querySelector('[role="dialog"]'));
}

/**
 * Check if the active element is an interactive clickable control (button, link, tab).
 */
export function isInteractiveButtonFocused(el: Element | null): boolean {
  if (!el) return false;
  return Boolean(
    el.closest?.('button, a[href], summary, [role="button"], [role="tab"], [role="menuitem"], [role="switch"]')
  );
}

/**
 * Format chapter marker storage key for a target clip.
 */
export function getMarkersStorageKey(targetKey?: string): string {
  return `roop_markers_${targetKey || 'default'}`;
}

// ── Hook Implementation ───────────────────────────────────────────────────────

export function useWorkstationShortcuts({
  frame,
  setFrame,
  maxFrames: rawMaxFrames,
  startFrame: rawStartFrame,
  endFrame: rawEndFrame,
  isPlaying,
  setIsPlaying,
  fps = 30,
  setPlaybackRate,
  videoRef,
  trimIn: controlledTrimIn,
  trimOut: controlledTrimOut,
  onUpdateTrim,
  setFrameMarkerVal,
  markers: controlledMarkers,
  targetKey = '',
  onToggleMarker,
  onMarkersChange,
  isPeekingOriginal: controlledPeeking,
  setIsPeekingOriginal: controlledSetPeeking,
  selectedTrackId = null,
  selectedFaceIndex = null,
  onSelectTrack,
  onSelectFace,
  onToggleCompare,
  enabled = true,
  attachListener = true,
}: WorkstationShortcutsOptions): UseWorkstationShortcutsReturn {
  const maxFrames = Math.max(1, Math.round(rawMaxFrames || 1));
  const startFrame = typeof rawStartFrame === 'number' ? Math.max(1, rawStartFrame) : 1;
  const endFrame = typeof rawEndFrame === 'number' ? Math.min(maxFrames, rawEndFrame) : maxFrames;

  // ── Internal Shuttle State ──────────────────────────────────────────────────
  const [shuttle, setShuttle] = useState<ShuttleState>(IDLE);
  const shuttleRef = useRef<ShuttleState>(shuttle);
  shuttleRef.current = shuttle;

  // ── Internal Hold-to-Peek State ─────────────────────────────────────────────
  const [internalPeeking, setInternalPeeking] = useState<boolean>(false);
  const isPeekingOriginal = controlledPeeking !== undefined ? controlledPeeking : internalPeeking;
  const setPeekingOriginal = controlledSetPeeking || setInternalPeeking;

  // ── Internal Markers State ──────────────────────────────────────────────────
  const [internalMarkers, setInternalMarkers] = useState<number[]>(() => {
    if (typeof localStorage === 'undefined') return [];
    try {
      const raw = localStorage.getItem(getMarkersStorageKey(targetKey));
      const list = raw ? JSON.parse(raw) : [];
      return Array.isArray(list) ? list.filter((n) => Number.isFinite(n) && n >= 1 && n <= maxFrames) : [];
    } catch {
      return [];
    }
  });

  // Reload markers when targetKey changes
  useEffect(() => {
    if (typeof localStorage === 'undefined') return;
    try {
      const raw = localStorage.getItem(getMarkersStorageKey(targetKey));
      const list = raw ? JSON.parse(raw) : [];
      const valid = Array.isArray(list) ? list.filter((n) => Number.isFinite(n) && n >= 1 && n <= maxFrames) : [];
      setInternalMarkers(valid);
    } catch {
      setInternalMarkers([]);
    }
  }, [targetKey, maxFrames]);

  const activeMarkers = controlledMarkers !== undefined ? controlledMarkers : internalMarkers;

  const persistMarkers = useCallback(
    (nextList: number[]) => {
      const sorted = [...new Set(nextList)]
        .filter((n) => Number.isFinite(n) && n >= 1 && n <= maxFrames)
        .sort((a, b) => a - b);
      setInternalMarkers(sorted);
      onMarkersChange?.(sorted);
      if (typeof localStorage !== 'undefined') {
        try {
          if (sorted.length > 0) {
            localStorage.setItem(getMarkersStorageKey(targetKey), JSON.stringify(sorted));
          } else {
            localStorage.removeItem(getMarkersStorageKey(targetKey));
          }
        } catch {
          // Ignore private mode / quota errors
        }
      }
    },
    [maxFrames, targetKey, onMarkersChange]
  );

  const toggleMarkerAtCurrent = useCallback(
    (targetF?: number) => {
      const f = targetF !== undefined ? targetF : frame;
      const exists = activeMarkers.includes(f);
      const next = exists ? activeMarkers.filter((m) => m !== f) : [...activeMarkers, f];
      persistMarkers(next);
      onToggleMarker?.(f);
    },
    [activeMarkers, frame, persistMarkers, onToggleMarker]
  );

  const jumpMarker = useCallback(
    (dir: -1 | 1) => {
      if (activeMarkers.length === 0) return;
      const sorted = [...activeMarkers].sort((a, b) => a - b);
      let target: number | undefined;
      if (dir > 0) {
        target = sorted.find((m) => m > frame);
      } else {
        target = [...sorted].reverse().find((m) => m < frame);
      }
      if (target !== undefined) {
        setFrame(target);
      }
    },
    [activeMarkers, frame, setFrame]
  );

  const clearMarkers = useCallback(() => {
    persistMarkers([]);
  }, [persistMarkers]);

  // ── Trim In / Out Computation ───────────────────────────────────────────────
  const trimIn = controlledTrimIn !== undefined ? controlledTrimIn : startFrame;
  const trimOut = controlledTrimOut !== undefined ? controlledTrimOut : endFrame;

  const setTrimInAtCurrent = useCallback(() => {
    const nextIn = Math.min(frame, trimOut);
    if (onUpdateTrim) {
      onUpdateTrim(nextIn, trimOut);
    }
    if (setFrameMarkerVal) {
      setFrameMarkerVal('start', nextIn);
    }
  }, [frame, trimOut, onUpdateTrim, setFrameMarkerVal]);

  const setTrimOutAtCurrent = useCallback(() => {
    const nextOut = Math.max(frame, trimIn);
    if (onUpdateTrim) {
      onUpdateTrim(trimIn, nextOut);
    }
    if (setFrameMarkerVal) {
      setFrameMarkerVal('end', nextOut);
    }
  }, [frame, trimIn, onUpdateTrim, setFrameMarkerVal]);

  const resetTrim = useCallback(() => {
    if (onUpdateTrim) {
      onUpdateTrim(1, maxFrames);
    }
    if (setFrameMarkerVal) {
      setFrameMarkerVal('start', 1);
      setFrameMarkerVal('end', maxFrames);
    }
  }, [maxFrames, onUpdateTrim, setFrameMarkerVal]);

  // ── Frame Stepping ──────────────────────────────────────────────────────────
  const stepFrame = useCallback(
    (delta: number) => {
      // Stepping frame cancels shuttle and halts playback
      if (shuttleRef.current.dir !== 0) {
        setShuttle(IDLE);
      }
      if (isPlaying) {
        setIsPlaying(false);
      }
      setFrame((curr) => {
        const next = curr + delta;
        return Math.max(1, Math.min(maxFrames, next));
      });
    },
    [isPlaying, setIsPlaying, setFrame, maxFrames]
  );

  // ── Rate & Video Player Sync ────────────────────────────────────────────────
  const shuttleRateValue = shuttleRate(shuttle);
  const shuttleLabelText = shuttleLabel(shuttle);

  // Sync the shuttle to the player. See shuttleSync: the rate is only reset when a shuttle ENDS, not on
  // every render where none is active (that clobbered the transport deck's 0.25x..4x choice on Play).
  const lastShuttleRef = useRef<ShuttleState>(IDLE);
  useEffect(() => {
    const action = shuttleSync(lastShuttleRef.current, shuttle, isPlaying);
    lastShuttleRef.current = shuttle;
    if (!action) return;
    setPlaybackRate?.(action.rate);
    if (videoRef?.current) videoRef.current.playbackRate = action.rate;
    if (action.play) setIsPlaying(true);
  }, [shuttle, isPlaying, setIsPlaying, setPlaybackRate, videoRef]);

  // When playback is changed from outside, end the shuttle. See shuttleEndsOnPlayChange: a play -> stop
  // TRANSITION ends a forward shuttle (a level test cancelled it the instant L was pressed on a stopped player).
  const wasPlayingRef = useRef(isPlaying);
  useEffect(() => {
    const ends = shuttleEndsOnPlayChange(wasPlayingRef.current, isPlaying, shuttle);
    wasPlayingRef.current = isPlaying;
    if (ends) setShuttle(IDLE);
  }, [isPlaying, shuttle]);

  // ── Reverse Shuttle Clock (rAF sub-frame loop) ──────────────────────────────
  const frameRef = useRef<number>(frame);
  frameRef.current = frame;

  useEffect(() => {
    if (shuttle.dir >= 0) return undefined;
    const rate = shuttleRate(shuttle);
    if (!rate) return undefined;

    let carry = 0;
    let last = performance.now();
    let rafId = 0;

    const tick = (now: number) => {
      const dt = now - last;
      last = now;
      const res = reverseTick({ frame: frameRef.current, carry }, dt, rate, fps, 1);
      carry = res.carry;
      if (res.frame !== frameRef.current) {
        setFrame(res.frame);
      }
      if (res.atStart) {
        setShuttle(IDLE);
        return;
      }
      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(rafId);
    };
  }, [shuttle, fps, setFrame]);

  // ── Track & Face Selection Handlers ─────────────────────────────────────────
  const selectTrack = useCallback(
    (trackId: number | string | null) => {
      onSelectTrack?.(trackId);
    },
    [onSelectTrack]
  );

  const selectFace = useCallback(
    (faceIndex: number | null) => {
      onSelectFace?.(faceIndex);
    },
    [onSelectFace]
  );

  // ── Keyboard Event Handlers ─────────────────────────────────────────────────
  const latestPropsRef = useRef({
    frame,
    maxFrames,
    isPlaying,
    setIsPlaying,
    setFrame,
    setTrimInAtCurrent,
    setTrimOutAtCurrent,
    resetTrim,
    toggleMarkerAtCurrent,
    jumpMarker,
    stepFrame,
    onToggleCompare,
    setPeekingOriginal,
    enabled,
  });

  useEffect(() => {
    latestPropsRef.current = {
      frame,
      maxFrames,
      isPlaying,
      setIsPlaying,
      setFrame,
      setTrimInAtCurrent,
      setTrimOutAtCurrent,
      resetTrim,
      toggleMarkerAtCurrent,
      jumpMarker,
      stepFrame,
      onToggleCompare,
      setPeekingOriginal,
      enabled,
    };
  });

  // ── Standalone Key Event Handlers ───────────────────────────────────────────
  const handleKeyDown = useCallback((e: KeyboardEvent | any): boolean => {
    const p = latestPropsRef.current;
    if (!p.enabled) return false;

    // 1. Guard against typing inside input/textarea/select/contenteditable
    if (isInputFocused()) return false;

    // 2. Guard against active modal dialogs
    if (isModalOpen()) return false;

    // 3. Spacebar Handling:
    // Space is the keyboard click for buttons/links/tabs. Do not steal it when an interactive control is focused!
    const isSpace = eventMatches(e, 'transport.playPause') || e.code === 'Space' || e.key === ' ';
    if (isSpace) {
      if (isInteractiveButtonFocused(document.activeElement)) {
        return false; // Let standard button activation occur
      }
      e.preventDefault?.(); // Prevent default page scrolling

      // Toggle playback or compare
      if (p.maxFrames > 1) {
        if (shuttleRef.current.dir !== 0) {
          setShuttle(IDLE);
        }
        p.setIsPlaying((prev) => !prev);
      } else if (p.onToggleCompare) {
        p.onToggleCompare();
      }
      return true;
    }

    // 4. J / K / L Shuttle Ergonomics
    // Shift+J / Shift+L are the slow-shuttle keys (transport.shuttleSlow*), so the physical-key fallbacks stay plain-key only.
    const isJ = eventMatches(e, 'transport.shuttleBack') || (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && e.code === 'KeyJ');
    const isK = eventMatches(e, 'transport.shuttlePause') || (!e.ctrlKey && !e.metaKey && !e.altKey && e.code === 'KeyK');
    const isL = eventMatches(e, 'transport.shuttleForward') || (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && e.code === 'KeyL');

    if (isJ && p.maxFrames > 1) {
      e.preventDefault?.();
      if (!e.repeat) {
        const next = shuttleKey(shuttleRef.current, 'j');
        setShuttle(next);
        p.setIsPlaying(false);
      }
      return true;
    }

    if (isK && p.maxFrames > 1) {
      e.preventDefault?.();
      if (!e.repeat) {
        setShuttle(IDLE);
        p.setIsPlaying(false);
      }
      return true;
    }

    if (isL && p.maxFrames > 1) {
      e.preventDefault?.();
      if (!e.repeat) {
        const next = shuttleKey(shuttleRef.current, 'l');
        setShuttle(next);
      }
      return true;
    }

    // 5. Left / Right Stepping (1 frame) & Shift + Left / Right (10 frames)
    const isStepBack10 = eventMatches(e, 'transport.stepBack10') || eventMatches(e, 'timeline.frameBack10') || (e.shiftKey && (e.key === 'ArrowLeft' || e.key === '<'));
    const isStepForward10 = eventMatches(e, 'transport.stepForward10') || eventMatches(e, 'timeline.frameForward10') || (e.shiftKey && (e.key === 'ArrowRight' || e.key === '>'));

    if (isStepBack10 && p.maxFrames > 1) {
      e.preventDefault?.();
      p.stepFrame(-10);
      return true;
    }
    if (isStepForward10 && p.maxFrames > 1) {
      e.preventDefault?.();
      p.stepFrame(10);
      return true;
    }

    const isStepBack = eventMatches(e, 'transport.stepBack') || eventMatches(e, 'timeline.frameBack') || (!e.shiftKey && (e.key === 'ArrowLeft' || e.key === ','));
    const isStepForward = eventMatches(e, 'transport.stepForward') || eventMatches(e, 'timeline.frameForward') || (!e.shiftKey && (e.key === 'ArrowRight' || e.key === '.'));

    if (isStepBack && p.maxFrames > 1) {
      e.preventDefault?.();
      p.stepFrame(-1);
      return true;
    }
    if (isStepForward && p.maxFrames > 1) {
      e.preventDefault?.();
      p.stepFrame(1);
      return true;
    }

    // 6. I / O Trim In / Out Points
    const isTrimIn = eventMatches(e, 'timeline.setIn') || (!e.ctrlKey && !e.metaKey && !e.altKey && e.code === 'KeyI');
    const isTrimOut = eventMatches(e, 'timeline.setOut') || (!e.ctrlKey && !e.metaKey && !e.altKey && e.code === 'KeyO');

    if (isTrimIn && p.maxFrames > 1) {
      e.preventDefault?.();
      if (!e.repeat) {
        p.setTrimInAtCurrent();
      }
      return true;
    }
    if (isTrimOut && p.maxFrames > 1) {
      e.preventDefault?.();
      if (!e.repeat) {
        p.setTrimOutAtCurrent();
      }
      return true;
    }

    // 7. M Chapter Marker Toggle
    const isMarker = eventMatches(e, 'timeline.marker') || (!e.ctrlKey && !e.metaKey && !e.altKey && e.code === 'KeyM');
    if (isMarker && p.maxFrames > 1) {
      e.preventDefault?.();
      if (!e.repeat) {
        p.toggleMarkerAtCurrent();
      }
      return true;
    }

    // 8. '\\' Hold to Peek Original Unswapped Frame
    const isPeek = eventMatches(e, 'viewport.peekOriginal') || e.code === 'Backslash' || e.key === '\\';
    if (isPeek) {
      e.preventDefault?.();
      if (!e.repeat) {
        p.setPeekingOriginal(true);
      }
      return true;
    }

    return false;
  }, []);

  const handleKeyUp = useCallback((e: KeyboardEvent | any): boolean => {
    const p = latestPropsRef.current;
    // Release peek original on keyup
    const isPeekUp = keyupMatches(e, 'viewport.peekOriginal') || e.code === 'Backslash' || e.key === '\\';
    if (isPeekUp) {
      p.setPeekingOriginal(false);
      return true;
    }
    return false;
  }, []);

  useEffect(() => {
    if (!enabled || !attachListener || typeof window === 'undefined') return undefined;

    const onKeyDown = (e: KeyboardEvent) => {
      handleKeyDown(e);
    };

    const onKeyUp = (e: KeyboardEvent) => {
      handleKeyUp(e);
    };

    const onBlur = () => {
      latestPropsRef.current.setPeekingOriginal(false);
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onBlur);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
    };
  }, [enabled, attachListener, handleKeyDown, handleKeyUp]);

  return {
    shuttle,
    setShuttle,
    shuttleRateValue,
    shuttleLabelText,
    markers: activeMarkers,
    toggleMarkerAtCurrent,
    jumpMarker,
    clearMarkers,
    trimIn,
    trimOut,
    setTrimInAtCurrent,
    setTrimOutAtCurrent,
    resetTrim,
    stepFrame,
    isPeekingOriginal,
    selectTrack,
    selectFace,
    handleKeyDown,
    handleKeyUp,
  };
}

export default useWorkstationShortcuts;
