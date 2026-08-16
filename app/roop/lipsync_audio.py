"""Lip-sync audio timing — frame/time mapping, kept separate from the model
class so it is unit-testable without onnxruntime/torch installed.

The pipeline decodes video frames independently of audio (see util_ffmpeg's
extract_audio_wav + whatever feature extractor Lipsync_MuseTalk runs once up
front, per-clip). Composing the right audio-feature chunk for frame N needs
two numbers this module doesn't own — fps and frame_start — passed in by the
caller (ProcessMgr), and one it does: how many feature chunks the extractor
produced for the whole clip.
"""

import numpy as np


def frame_time(frame_start: int, frame_idx: int, fps: float) -> float:
    """Seconds into the driving audio for a frame at *frame_idx* (0-based,
    relative to frame_start, matching ProcessMgr's read_frames_thread
    convention) of a clip that begins at frame_start and plays at fps."""
    if fps <= 0:
        return 0.0
    return (frame_start + frame_idx) / fps


def audio_index_for_frame(frame_time_s: float, audio_fps: float, num_chunks: int) -> int:
    """Nearest precomputed audio-feature chunk for a video timestamp, clamped
    to a valid index. audio_fps is the chunk rate of the feature sequence
    (not the video's fps — the two are independent)."""
    if num_chunks <= 0:
        return 0
    if audio_fps <= 0:
        return 0
    idx = int(round(frame_time_s * audio_fps))
    return max(0, min(idx, num_chunks - 1))


class AudioFeatureCache:
    """One clip's worth of precomputed audio features, indexed by video time.

    Built once per run (ProcessMgr.initialize(), mirroring the one-time 3D-recon
    source-crop cache there) rather than per-frame, since feature extraction
    reads the whole audio track at once and frames are processed out of order
    across worker threads.
    """

    def __init__(self, features: np.ndarray, audio_fps: float):
        self.features = features
        self.audio_fps = audio_fps

    @property
    def num_chunks(self) -> int:
        return 0 if self.features is None else len(self.features)

    def features_for_time(self, t_s: float):
        if self.features is None or self.num_chunks == 0:
            return None
        idx = audio_index_for_frame(t_s, self.audio_fps, self.num_chunks)
        return self.features[idx]
