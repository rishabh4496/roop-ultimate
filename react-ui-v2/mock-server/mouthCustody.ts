// Viseme Mouth Custody as the mock serves it: the request rules of app/roop/mouth_custody.py, in TypeScript, so the e2e suite
// meets the same refusals and the same silent-video fallback the real backend gives. The pure model the UI uses is
// react-ui/src/components/faceswap/mouthCustody.js; the shared cases that pin both to the backend live in
// react-ui/.render-check/mouth-custody.cases.json.

import type { MouthCustodyMode, MouthCustodyTelemetry } from '../src/types/api-contract';

export const MOUTH_MODES = ['off', 'source_mouth_restore', 'audio_driven_sync'] as const satisfies readonly MouthCustodyMode[];
export type MouthMode = MouthCustodyMode;

export const OFFSET_LIMIT_MS = 500;
const DRIVING_EXTENSIONS = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.wma', '.mka', '.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v'];

export interface MouthJob {
  mode: MouthMode;
  offsetMs: number;
  override: string | null;
  /** Why AUDIO_DRIVEN_SYNC runs as SOURCE_MOUTH_RESTORE for this clip, or ''. */
  fallback: string;
}

export type MouthResult = { job: MouthJob } | { error: string };

const truthy = (v: unknown): boolean => v === true || v === 'true' || v === 1 || v === '1';

/** The request's mode: the explicit one, else what the legacy switches said (lip-sync outranks restore). */
function deriveMode(payload: Record<string, unknown>): MouthMode | { error: string } {
  const raw = payload.mouth_custody_mode;
  if (raw !== undefined && raw !== null && String(raw).trim() !== '') {
    const text = String(raw).trim().toLowerCase().replace(/[-\s]/g, '_');
    if ((MOUTH_MODES as readonly string[]).includes(text)) return text as MouthMode;
    return { error: `mouth_custody_mode must be one of ${MOUTH_MODES.join(', ')}, got '${String(raw)}'` };
  }
  if (truthy(payload.lipsync_enabled)) return 'audio_driven_sync';
  if (truthy(payload.restore_original_mouth)) return 'source_mouth_restore';
  return 'off';
}

export function normalizeMouthCustody(payload: Record<string, unknown>, opts: { targetHasAudio?: boolean } = {}): MouthResult {
  const mode = deriveMode(payload);
  if (typeof mode !== 'string') return mode;

  let offsetMs = 0;
  const rawOffset = payload.audio_offset_ms;
  if (rawOffset !== undefined && rawOffset !== null && rawOffset !== '') {
    offsetMs = Number(rawOffset);
    if (!Number.isFinite(offsetMs) || Math.abs(offsetMs) > OFFSET_LIMIT_MS) {
      return { error: `audio_offset_ms must be within +-${OFFSET_LIMIT_MS} ms, got ${JSON.stringify(rawOffset)}` };
    }
  }

  let override: string | null = (payload.override_audio_path as string) || null;
  if (!override && payload.lipsync_audio_source === 'upload' && payload.lipsync_audio_path) override = String(payload.lipsync_audio_path);
  if (override) {
    const lower = override.toLowerCase();
    if (!DRIVING_EXTENSIONS.some((ext) => lower.endsWith(ext))) {
      return { error: `override_audio_path must be an audio file (.wav, .mp3, .flac ...) or a video with audio: ${override}` };
    }
  }

  const silent = opts.targetHasAudio === false && !override;
  const fallback = mode === 'audio_driven_sync' && silent ? 'no usable driving audio (target has no audio stream)' : '';
  return { job: { mode, offsetMs, override, fallback } };
}

/** The Viseme Mouth Custody fields of a /ws/telemetry frame. Typed against the contract, so a renamed key does not compile. */
export function mouthTelemetry(job: MouthJob | null, processing: boolean): MouthCustodyTelemetry {
  const running = processing && !!job;
  return {
    // lipsync_ms is the Lip-Sync latency column: present only while audio-driven sync really drives the mouth (not after a fallback).
    lipsync_ms: running && job!.mode === 'audio_driven_sync' && !job!.fallback ? 4.2 : null,
    mouth_mode: running ? job!.mode : '',
    mouth_fallback: running ? job!.fallback : '',
    audio_offset_ms: running ? job!.offsetMs : 0,
  };
}
