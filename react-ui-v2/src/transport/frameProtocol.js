// ── /ws/frames wire format (client side) ──────────────────────────────────
//
// The server's layout, byte for byte (app/routes_frames.py, pinned by
// app/tests/test_frames_ws.py). Everything is little-endian:
//
//   0 u8 version | 1 u8 kind | 2 u16 flags | 4 u32 stream | 8 u32 frame
//  12 u32 width  | 16 u32 height | 20.. payload (JPEG bytes)
//
// Pure functions only, so the node checks can exercise them without a socket.

export const HEADER_BYTES = 20;
export const VERSION = 1;
export const KIND_LIVE = 1;
export const KIND_PLAY = 2;
export const KIND_END = 3;
export const FLAG_ERROR = 1;

/**
 * Split one binary message into its header fields and payload.
 *
 * The payload is returned as a VIEW-free ArrayBuffer slice, because it is
 * about to be transferred to a worker, and a transfer detaches the WHOLE
 * underlying buffer: transferring a subarray of the message would take the
 * header with it and leave nothing for anyone else holding the message.
 *
 * @param {ArrayBuffer} buf
 * @returns {null | {version:number, kind:number, flags:number, stream:number,
 *           frame:number, width:number, height:number, bytes:ArrayBuffer}}
 */
export function parseFrameMessage(buf) {
  if (!(buf instanceof ArrayBuffer) || buf.byteLength < HEADER_BYTES) return null;
  const v = new DataView(buf);
  const version = v.getUint8(0);
  if (version !== VERSION) return null;
  return {
    version,
    kind: v.getUint8(1),
    flags: v.getUint16(2, true),
    stream: v.getUint32(4, true),
    frame: v.getUint32(8, true),
    width: v.getUint32(12, true),
    height: v.getUint32(16, true),
    bytes: buf.slice(HEADER_BYTES),
  };
}

/** Inverse of parseFrameMessage — for tests and a mock server. */
export function packFrameMessage({ kind, flags = 0, stream = 0, frame = 0, width = 0, height = 0 }, payload) {
  const body = payload ? new Uint8Array(payload) : new Uint8Array(0);
  const out = new Uint8Array(HEADER_BYTES + body.byteLength);
  const v = new DataView(out.buffer);
  v.setUint8(0, VERSION);
  v.setUint8(1, kind);
  v.setUint16(2, flags, true);
  v.setUint32(4, stream >>> 0, true);
  v.setUint32(8, frame >>> 0, true);
  v.setUint32(12, width >>> 0, true);
  v.setUint32(16, height >>> 0, true);
  out.set(body, HEADER_BYTES);
  return out.buffer;
}
