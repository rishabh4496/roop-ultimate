import React from 'react';
import { Select, Slider, Toggle, Button } from '../ui';
import FileDrop from './FileDrop';
import {
  MODES, MODE_LABELS, MODE_HINTS, OFFSET_LIMIT_MS, AUDIO_ACCEPT,
  modeOf, modePatch, overridePatch, clampOffset, offsetText, offsetMeaning, advisory, requestFields, SLIP_NUDGES,
} from './mouthCustody';

// "Audio & Mouth Custody": who owns the swapped mouth, and which audio the delivered file carries.
//
//   #/faceswap  the full panel (mode + the lip-sync engine controls + the audio track + the slip).
//   #/extras    `showMode={false}`: the media editor runs no swap, so there is no mouth to own - only the audio half
//               (bind an external track, slip it) applies, and the panel shows only that.
//
// The model is faceswap/mouthCustody.js (pure, pinned to the backend by .render-check/render-mouth-custody.mjs); this file
// is layout. `onPatch(patch)` merges a patch into the caller's params, so a mode change and its legacy keys land together.
//
// `legacy` is whether the caller's params also carry the old lipsync_* pair (the recipe does, a multipart form does not).

export default function MouthCustodyPanel({
  params,
  onPatch,
  onAddAudio,
  uploading = false,
  hasTargetAudio = null,
  isImage = false,
  showMode = true,
  legacy = true,
  emptyTrack = null,
  defaults = {},
  trackName = '',
}) {
  const p = params || {};
  const mode = modeOf(p);
  const { override_audio_path: track } = requestFields(p);
  const offset = clampOffset(p.audio_offset_ms);
  const note = showMode ? advisory(p, { hasTargetAudio, isImage }) : null;
  const audioActive = showMode ? mode === 'audio_driven_sync' : true;
  const setOffset = (v) => onPatch({ audio_offset_ms: clampOffset(v) });
  const fileName = trackName || (track ? String(track).split(/[\\/]/).pop() : '');

  return (
    <div className="space-y-3" data-testid="mouth-custody-panel">
      {showMode && (
        <fieldset className="space-y-2">
          <legend className="text-xs font-medium text-white/70 mb-1.5">Mouth custody</legend>
          <div role="radiogroup" aria-label="Mouth custody mode" className="grid grid-cols-1 gap-1.5">
            {MODES.map((m) => {
              const on = m === mode;
              return (
                <button
                  key={m}
                  type="button"
                  role="radio"
                  aria-checked={on}
                  data-mode={m}
                  onClick={() => onPatch(modePatch(m))}
                  className={`text-left rounded-lg border px-3 py-2 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)] ${
                    on ? 'border-[var(--accent)]/70 bg-[var(--accent)]/10 text-white' : 'border-white/10 bg-white/[0.02] text-white/70 hover:border-white/25'
                  }`}
                >
                  <span className="flex items-center gap-2 text-sm font-medium">
                    <span aria-hidden="true" className={`h-2.5 w-2.5 rounded-full border ${on ? 'border-[var(--accent)] bg-[var(--accent)]' : 'border-white/40'}`} />
                    {MODE_LABELS[m]}
                  </span>
                  {on && <span className="mt-1 block text-xs leading-snug text-white/60">{MODE_HINTS[m]}</span>}
                </button>
              );
            })}
          </div>
        </fieldset>
      )}

      {note && (
        <div role="status" data-testid="mouth-custody-advisory"
             className={`rounded-lg border px-3 py-2 text-xs leading-snug ${note.tone === 'warn' ? 'border-amber-400/40 bg-amber-400/10 text-amber-100' : 'border-white/15 bg-white/5 text-white/70'}`}>
          {note.text}
        </div>
      )}

      {showMode && mode === 'audio_driven_sync' && (
        <div className="space-y-2">
          <Select
            label="Lip-sync Engine / Model"
            value={p.lipsync_model || defaults.lipsync_model || 'wav2lip'}
            onChange={(v) => onPatch({ lipsync_model: v })}
            options={[
              { value: 'wav2lip', label: 'Wav2Lip-HQ (Ultra-low Latency ONNX FP16)' },
              { value: 'musetalk', label: 'MuseTalk (Diffusion UNet)' },
            ]}
          />
          <Select
            label="Audio/Visual Co-articulation Mode"
            value={p.lipsync_coarticulation_mode || defaults.lipsync_coarticulation_mode || 'keep_target_lips'}
            onChange={(v) => onPatch({ lipsync_coarticulation_mode: v })}
            options={[
              { value: 'keep_target_lips', label: 'Keep Target Lips (Matches original actor mouth motion)' },
              { value: 'redub_sync', label: 'Re-dub Sync (Alters mouth & jawline geometry to new voiceover)' },
            ]}
          />
          <Toggle
            label="Mouth Cavity & Teeth Restoration (BiSeNet)"
            info="Reconstructs crisp enamel edges, interdental crevices and natural tongue depth, so a generated mouth does not clamp or blur the teeth."
            checked={p.oral_cavity_restore !== false}
            onChange={(v) => onPatch({ oral_cavity_restore: v })}
          />
          {p.lipsync_coarticulation_mode === 'redub_sync' && (
            <Slider
              label="Jaw Co-articulation Strength"
              info="Deforms mandibular contours and jawline in step with acoustic energy and phoneme aperture."
              min={0}
              max={1}
              step={0.05}
              value={Number.isFinite(Number(p.jaw_coarticulation_strength)) ? Number(p.jaw_coarticulation_strength) : 0.6}
              defaultValue={defaults.jaw_coarticulation_strength}
              onChange={(v) => onPatch({ jaw_coarticulation_strength: v })}
            />
          )}
        </div>
      )}

      <div className="space-y-2" data-testid="mouth-custody-audio">
        <div className="text-xs font-medium text-white/70">
          {showMode ? (audioActive ? 'Driving & delivered audio' : 'Delivered audio') : 'Replace the audio track'}
        </div>
        {track ? (
          <div className="flex items-center justify-between gap-2 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2">
            <span className="min-w-0 truncate text-sm text-white/85" title={String(track)} data-testid="mouth-custody-track">
              External track: {fileName}
            </span>
            <Button variant="ghost" size="sm" onClick={() => onPatch(overridePatch(null, { legacy, empty: emptyTrack }))}>Remove</Button>
          </div>
        ) : (
          <FileDrop
            accept={AUDIO_ACCEPT}
            label="Bind an external audio track"
            onFiles={onAddAudio}
            busy={uploading}
            hint=".wav / .mp3 / .flac - drop here. Without one the video's own audio is used."
          />
        )}
        {!track && showMode && audioActive && (
          <p className="text-xs text-white/50">Using the target video's own audio.</p>
        )}

        <div className="rounded-lg border border-white/10 bg-white/[0.02] p-2.5 space-y-2" role="group" aria-label="Audio offset">
          <Slider
            label="Audio offset"
            info={`Slides the audio against the video, ${-OFFSET_LIMIT_MS} to +${OFFSET_LIMIT_MS} ms. Positive = the audio is heard later. The mouth and the delivered track move together.`}
            min={-OFFSET_LIMIT_MS}
            max={OFFSET_LIMIT_MS}
            step={1}
            value={offset}
            defaultValue={0}
            onChange={setOffset}
          />
          <div className="flex flex-wrap items-center gap-1.5">
            {SLIP_NUDGES.map((d) => (
              <button
                key={d}
                type="button"
                onClick={() => setOffset(offset + d)}
                aria-label={`Nudge audio ${d > 0 ? 'later' : 'earlier'} by ${Math.abs(d)} millisecond${Math.abs(d) === 1 ? '' : 's'}`}
                className="rounded-md border border-white/15 px-2 py-0.5 text-xs tabular-nums text-white/75 hover:border-white/35 focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--accent)]"
              >
                {d > 0 ? `+${d}` : `−${Math.abs(d)}`}
              </button>
            ))}
            <button
              type="button"
              onClick={() => setOffset(0)}
              disabled={offset === 0}
              className="rounded-md border border-white/15 px-2 py-0.5 text-xs text-white/75 hover:border-white/35 disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--accent)]"
            >
              Reset
            </button>
            <span className="ml-auto text-xs tabular-nums text-white/70" data-testid="mouth-custody-offset">{offsetText(offset)}</span>
          </div>
          <p className="text-xs text-white/50" aria-live="polite">{offsetMeaning(offset)}</p>
        </div>
      </div>
    </div>
  );
}
