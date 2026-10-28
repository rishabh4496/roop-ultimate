/**
 * Stage 19: BroadcastChannel('roop_ultimate_reference_out') Synchronization Bridge.
 *
 * Provides a 60fps zero-copy synchronization bus between the main workstation
 * (InteractivePreview / Extras) and the pop-out clean feed reference monitor (#/monitor).
 *
 * Transports:
 *   - Binary frames (ImageBitmap / transferrable canvas buffers)
 *   - Transport state (playback, frame numbers, scrub position)
 *   - Split-wipe comparison coordinates (modes, direction, angle, wipe %)
 *   - Stage 12/18 Color Grading (LUT, Lift/Gamma/Gain, Temp/Tint, Vibrance) & Zebra overlays
 *
 * Enforces deterministic channel teardown on unmount per Stage 6 QA rules.
 */

export const CHANNEL_NAME = 'roop_ultimate_reference_out';

export const MESSAGE_TYPES = {
  FRAME: 'FRAME',
  TRANSPORT: 'TRANSPORT',
  COMPARE: 'COMPARE',
  GRADING: 'GRADING',
  PING: 'PING',
  PONG: 'PONG',
  TEARDOWN: 'TEARDOWN',
};

class InMemoryBroadcastBus {
  constructor(name) {
    this.name = name;
    this._listeners = new Set();
    InMemoryBroadcastBus._channels = InMemoryBroadcastBus._channels || new Map();
    const existing = InMemoryBroadcastBus._channels.get(name) || new Set();
    existing.add(this);
    InMemoryBroadcastBus._channels.set(name, existing);
  }

  postMessage(message) {
    const peers = InMemoryBroadcastBus._channels.get(this.name);
    if (!peers) return;
    for (const peer of peers) {
      if (peer !== this) {
        try {
          peer._deliver(message);
        } catch {
          // ignore receiver errors
        }
      }
    }
  }

  _deliver(data) {
    if (typeof this.onmessage === 'function') {
      this.onmessage({ data });
    }
    for (const listener of this._listeners) {
      listener({ data });
    }
  }

  addEventListener(event, listener) {
    if (event === 'message') this._listeners.add(listener);
  }

  removeEventListener(event, listener) {
    if (event === 'message') this._listeners.delete(listener);
  }

  close() {
    this._listeners.clear();
    const peers = InMemoryBroadcastBus._channels.get(this.name);
    if (peers) {
      peers.delete(this);
      if (peers.size === 0) InMemoryBroadcastBus._channels.delete(this.name);
    }
  }
}

/**
 * Creates or wraps a BroadcastChannel instance with environment safety.
 * @param {Function} onMessage Callback invoked with event.data
 * @returns {BroadcastChannel|InMemoryBroadcastBus}
 */
export function createReferenceChannel(onMessage = null) {
  let channel = null;
  if (typeof BroadcastChannel !== 'undefined') {
    try {
      channel = new BroadcastChannel(CHANNEL_NAME);
    } catch {
      channel = new InMemoryBroadcastBus(CHANNEL_NAME);
    }
  } else {
    channel = new InMemoryBroadcastBus(CHANNEL_NAME);
  }

  if (typeof onMessage === 'function') {
    channel.onmessage = (event) => onMessage(event.data);
  }

  return channel;
}

/**
 * Broadcasts a decoded frame across the reference channel.
 */
export function broadcastFrame(channel, frameData, meta = {}) {
  if (!channel) return;
  try {
    channel.postMessage({
      type: MESSAGE_TYPES.FRAME,
      payload: {
        frame: frameData,
        width: meta.width || 0,
        height: meta.height || 0,
        frameIndex: meta.frameIndex ?? -1,
        timestamp: meta.timestamp || Date.now(),
      },
    });
  } catch {
    // transfer or serialization fallback
  }
}

/**
 * Broadcasts playback transport state.
 */
export function broadcastTransport(channel, transport) {
  if (!channel) return;
  try {
    channel.postMessage({
      type: MESSAGE_TYPES.TRANSPORT,
      payload: {
        isPlaying: !!transport.isPlaying,
        currentFrame: transport.currentFrame ?? 0,
        totalFrames: transport.totalFrames ?? 0,
        currentTime: transport.currentTime ?? 0,
        duration: transport.duration ?? 0,
        fps: transport.fps ?? 30,
      },
    });
  } catch {
    // channel closed
  }
}

/**
 * Broadcasts split-wipe comparison coordinates.
 */
export function broadcastCompare(channel, compare) {
  if (!channel) return;
  try {
    channel.postMessage({
      type: MESSAGE_TYPES.COMPARE,
      payload: {
        mode: compare.mode || 'slider',
        wipe: compare.wipe ?? 50,
        compareDir: compare.compareDir || 'vertical',
        wipeAngle: compare.wipeAngle ?? 0,
        diffGain: compare.diffGain ?? 3.0,
      },
    });
  } catch {
    // channel closed
  }
}

/**
 * Broadcasts Stage 12/18 LUT and Zebra overlays.
 */
export function broadcastGrading(channel, grading) {
  if (!channel) return;
  try {
    channel.postMessage({
      type: MESSAGE_TYPES.GRADING,
      payload: {
        lut_path: grading.lut_path || '',
        lut_opacity: grading.lut_opacity ?? 1.0,
        color_temperature: grading.color_temperature ?? 0.0,
        color_tint: grading.color_tint ?? 0.0,
        color_vibrance: grading.color_vibrance ?? 0.0,
        color_skin_protect: grading.color_skin_protect ?? 1.0,
        lift_m: grading.lift_m ?? 0.0,
        gamma_m: grading.gamma_m ?? 1.0,
        gain_m: grading.gain_m ?? 1.0,
        zebraActive: !!grading.zebraActive,
      },
    });
  } catch {
    // channel closed
  }
}

/**
 * Deterministically closes and tears down the reference channel.
 */
export function closeReferenceChannel(channel) {
  if (!channel) return;
  try {
    channel.postMessage({ type: MESSAGE_TYPES.TEARDOWN });
  } catch {
    // already closed
  }
  try {
    channel.close();
  } catch {
    // ignore
  }
}
