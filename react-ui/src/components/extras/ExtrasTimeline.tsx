/**
 * ExtrasTimeline: the NLE timeline over the Extras editor's local video.
 *
 * Same component as the Face Swap workstation's, with what Extras does not have switched off:
 * no pipeline steps, no face tracks / coverage heatmap (nothing scans the clip here) and no parameter
 * keyframes (nothing consumes them). What it does have is real:
 *   ruler / playhead / scrub card    frame-accurate over the <video> (see useVideoTransport)
 *   V1 filmstrip, A1 waveform        built in the browser from the local file (see clientMedia.ts)
 *   transport deck, minimap, zoom    the same
 *   In / Out + loop                  a PREVIEW region: /api/extras/apply has no trim, so Apply still
 *                                    exports the whole clip (the toolbar says so)
 *   markers                          colour + note, stored per clip in this browser
 *
 * Keys are scoped to the focused timeline: the media viewport owns a held Space as "pan mode", so a
 * window-level Space handler would play / pause while the user is trying to pan.
 */

import React, { useCallback, type RefObject } from 'react';
import NleTimeline from '../faceswap/NleTimeline';
import { useChapterMarkers } from '../faceswap/nle/useChapterMarkers';
import { useVideoTransport } from '../faceswap/nle/useVideoTransport';

export interface ExtrasTimelineProps {
  file: File;
  url: string;
  videoRef: RefObject<HTMLVideoElement | null>;
}

/** Keys that belong to a button / slider / field when it has focus: the timeline must not also act on them. */
const owns = (target: EventTarget | null) => !!(target as HTMLElement | null)?.closest?.('button, input, select, textarea, [role="slider"], [role="radio"], [role="button"]');

export function ExtrasTimeline({ file, url, videoRef }: ExtrasTimelineProps) {
  const clipKey = `extras_${file.name}_${file.size}_${file.lastModified}`;
  const t = useVideoTransport(videoRef, url);
  const markers = useChapterMarkers(clipKey, t.maxFrames);

  const hostKeys = useCallback((e: React.KeyboardEvent): boolean => {
    if (e.ctrlKey || e.metaKey || e.altKey) return false;
    const k = e.key;
    const big = e.shiftKey ? 10 : 1;
    if (k === ' ') { if (owns(e.target)) return false; t.togglePlay(); return true; }
    if (owns(e.target) && (k === 'ArrowLeft' || k === 'ArrowRight')) return false;
    if (k === 'ArrowLeft' || k === ',') { t.step(-big); return true; }
    if (k === 'ArrowRight' || k === '.') { t.step(big); return true; }
    if (k === 'Home') { t.seekFrame(0); return true; }
    if (k === 'End') { t.seekFrame(t.maxFrames - 1); return true; }
    if (k === 'i' || k === 'I' || k === '[') { t.setTrimIn(t.frame); return true; }
    if (k === 'o' || k === 'O' || k === ']') { t.setTrimOut(t.frame); return true; }
    if (k === 'm' || k === 'M') { markers.add({ frame: t.frame }); return true; }
    return false;
  }, [t, markers]);

  return (
    <div className="mt-3" data-testid="extras-timeline">
      <NleTimeline
        frame={t.frame}
        maxFrames={t.maxFrames}
        fps={t.fps}
        fpsAssumed={t.fpsAssumed}
        isPlaying={t.isPlaying}
        targetKey={clipKey}
        mediaSource={{ url, file }}
        trimIn={t.trimIn}
        trimOut={t.trimOut}
        markers={markers.markers}
        onAddMarker={markers.add}
        onUpdateMarker={markers.update}
        onDeleteMarker={markers.remove}
        playbackRate={t.rate}
        isLooping={t.loop}
        onTogglePlay={t.togglePlay}
        onStep={t.step}
        onPlaybackRateChange={t.setRate}
        onToggleLoop={() => t.setLoop(!t.loop)}
        onSeek={(f) => t.seekFrame(f)}
        onUpdateTrim={t.setTrim}
        showPipelineSteps={false}
        showFaceTrackLanes={false}
        showHealthLane={false}
        showKeyframeLane={false}
        segmentLaneTitle="Markers"
        regionNote="preview loop only: Apply exports the whole clip"
        hostKeys={hostKeys}
      />
    </div>
  );
}

export default ExtrasTimeline;
