"""Lip-sync audio timing — frame/time mapping, kept separate from the model
class so it is unit-testable without onnxruntime/torch installed.

The pipeline decodes video frames independently of audio (see util_ffmpeg's
extract_audio_wav + whatever feature extractor Lipsync_MuseTalk runs once up
front, per-clip). Composing the right audio-feature chunk for frame N needs
two numbers this module doesn't own — fps and frame_start — passed in by the
caller (ProcessMgr), and one it does: how many feature chunks the extractor
produced for the whole clip.
"""

import math
import os

import numpy as np
from roop.degrade import swallowed as _swallowed


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

    def __init__(self, features: np.ndarray = None, audio_fps: float = 25.0,
                 mel_spectrogram: np.ndarray = None, mel_fps: float = 80.0,
                 phoneme_energy: np.ndarray = None, viseme_openness: np.ndarray = None):
        self.features = features
        self.audio_fps = audio_fps
        self.mel_spectrogram = mel_spectrogram
        self.mel_fps = mel_fps if mel_fps > 0 else 80.0
        self.phoneme_energy = phoneme_energy
        self.viseme_openness = viseme_openness

    @property
    def num_chunks(self) -> int:
        if self.features is not None:
            return len(self.features)
        if self.mel_spectrogram is not None:
            return self.mel_spectrogram.shape[1]
        return 0

    def features_for_time(self, t_s: float):
        if self.features is None or len(self.features) == 0:
            return None
        idx = audio_index_for_frame(t_s, self.audio_fps, len(self.features))
        return self.features[idx]

    def mel_chunk_for_time(self, t_s: float, window_size: int = 16) -> np.ndarray:
        """Extract an 80-channel log-mel spectrogram window centered at timestamp t_s.
        Shape returned: (80, window_size).
        If mel_spectrogram is None, returns zeros of shape (80, window_size).
        """
        if self.mel_spectrogram is None or self.mel_spectrogram.size == 0:
            return np.zeros((80, window_size), dtype=np.float32)

        n_mels, total_steps = self.mel_spectrogram.shape
        center_step = int(round(max(0.0, t_s) * self.mel_fps))
        half = window_size // 2
        start_step = center_step - half
        end_step = start_step + window_size

        if start_step >= 0 and end_step <= total_steps:
            chunk = self.mel_spectrogram[:, start_step:end_step]
        else:
            # Pad boundary with edge reflection or zeros
            chunk = np.zeros((n_mels, window_size), dtype=np.float32)
            src_start = max(0, start_step)
            src_end = min(total_steps, end_step)
            dst_start = max(0, -start_step)
            dst_end = dst_start + (src_end - src_start)
            if src_end > src_start and dst_end > dst_start:
                chunk[:, dst_start:dst_end] = self.mel_spectrogram[:, src_start:src_end]
                # Replicate border frames for any missing edge context
                if dst_start > 0:
                    chunk[:, :dst_start] = chunk[:, dst_start:dst_start + 1]
                if dst_end < window_size:
                    chunk[:, dst_end:] = chunk[:, dst_end - 1:dst_end]

        return chunk.astype(np.float32)

    def phoneme_energy_for_time(self, t_s: float) -> float:
        """Normalized vocal phoneme energy [0.0, 1.0] at time t_s."""
        if self.phoneme_energy is None or len(self.phoneme_energy) == 0:
            return 0.0
        idx = int(round(max(0.0, t_s) * self.mel_fps))
        idx = max(0, min(idx, len(self.phoneme_energy) - 1))
        return float(self.phoneme_energy[idx])

    def viseme_openness_for_time(self, t_s: float) -> float:
        """Estimated viseme openness [0.0, 1.0] conditioned on acoustic formants at time t_s."""
        if self.viseme_openness is None or len(self.viseme_openness) == 0:
            return 0.0
        idx = int(round(max(0.0, t_s) * self.mel_fps))
        idx = max(0, min(idx, len(self.viseme_openness) - 1))
        return float(self.viseme_openness[idx])

    def is_speech_active(self, t_s: float, threshold: float = 0.05) -> bool:
        """Boolean check whether active speech vocalization occurs at timestamp t_s."""
        return self.phoneme_energy_for_time(t_s) >= threshold


# ── 16kHz Mel-Spectrogram & Acoustic Feature Extraction ───────────────────
MEL_SR = 16000
MEL_N_MELS = 80
MEL_N_FFT = 800
MEL_HOP_LENGTH = 200
MEL_WIN_LENGTH = 800
MEL_FMIN = 55.0
MEL_FMAX = 7600.0
MEL_FPS = 80.0  # 16000 / 200


def load_audio_16k(audio_path: str) -> np.ndarray:
    """Ingest audio and return a 16kHz mono float32 array normalized to [-1.0, 1.0]."""
    if not audio_path or not os.path.isfile(audio_path):
        return np.array([], dtype=np.float32)

    try:
        import librosa
        samples, _ = librosa.load(audio_path, sr=MEL_SR, mono=True)
        return samples.astype(np.float32)
    except Exception as _degrade_error:
        _swallowed("roop/lipsync_audio.py:146", _degrade_error,
                   "scipy audio fallback continued")

    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(audio_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.dtype == np.int16:
            samples = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            samples = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            samples = (data.astype(np.float32) - 128.0) / 128.0
        else:
            samples = data.astype(np.float32)
        if sr != MEL_SR and len(samples) > 0:
            target_len = int(round(len(samples) * float(MEL_SR) / float(sr)))
            from scipy import signal
            samples = signal.resample(samples, target_len).astype(np.float32)
        return samples
    except Exception as _degrade_error:
        _swallowed("roop/lipsync_audio.py:167", _degrade_error,
                   "empty audio fallback continued")
        return np.array([], dtype=np.float32)


def extract_mel_spectrogram(samples: np.ndarray, sr: int = MEL_SR,
                            n_mels: int = MEL_N_MELS, n_fft: int = MEL_N_FFT,
                            hop_length: int = MEL_HOP_LENGTH,
                            win_length: int = MEL_WIN_LENGTH,
                            fmin: float = MEL_FMIN, fmax: float = MEL_FMAX) -> np.ndarray:
    """Compute 80-channel log-mel spectrogram from 16kHz audio samples.
    Returns (n_mels, T) float32 array.
    """
    if samples is None or len(samples) == 0:
        return np.zeros((n_mels, 0), dtype=np.float32)

    try:
        import librosa
        mel = librosa.feature.melspectrogram(
            y=samples, sr=sr, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
        log_mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
        return log_mel.astype(np.float32)
    except Exception as _degrade_error:
        _swallowed("roop/lipsync_audio.py:189", _degrade_error,
                   "numpy/scipy mel fallback continued")

    # Pure numpy/scipy fallback STFT
    try:
        from scipy import signal
        window = signal.windows.hann(win_length)
        _, _, Zxx = signal.stft(samples, fs=sr, window=window, nperseg=win_length,
                                noverlap=win_length - hop_length, nfft=n_fft)
        power_spec = np.abs(Zxx) ** 2
        freqs = np.linspace(0, sr / 2, power_spec.shape[0])
        # Simple triangular mel filterbank
        mel_points = np.linspace(2595 * np.log10(1 + fmin / 700.0),
                                 2595 * np.log10(1 + fmax / 700.0), n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        fb = np.zeros((n_mels, power_spec.shape[0]), dtype=np.float32)
        for m in range(n_mels):
            f_m_minus = hz_points[m]
            f_m = hz_points[m + 1]
            f_m_plus = hz_points[m + 2]
            for k, f in enumerate(freqs):
                if f_m_minus <= f < f_m:
                    fb[m, k] = (f - f_m_minus) / (f_m - f_m_minus + 1e-6)
                elif f_m <= f <= f_m_plus:
                    fb[m, k] = (f_m_plus - f) / (f_m_plus - f_m + 1e-6)
        mel = fb @ power_spec
        log_mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
        return log_mel.astype(np.float32)
    except Exception as _degrade_error:
        _swallowed("roop/lipsync_audio.py:217", _degrade_error,
                   "zero mel fallback continued")
        return np.zeros((n_mels, max(1, len(samples) // hop_length)), dtype=np.float32)


def compute_phoneme_features(mel_spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized phoneme energy and acoustic viseme openness curves.
    Returns (phoneme_energy, viseme_openness), each of shape (T,).
    """
    if mel_spec is None or mel_spec.shape[1] == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    power = np.exp(mel_spec)  # linear power (80, T)
    raw_energy = np.mean(power, axis=0)  # (T,)
    e_min, e_max = float(raw_energy.min()), float(raw_energy.max())
    span = max(1e-6, e_max - e_min)
    phoneme_energy = np.clip((raw_energy - e_min) / span, 0.0, 1.0).astype(np.float32)

    # First and second formant energy (mel bins 8..35, approx 300Hz..1800Hz) vs higher frequencies
    formant_power = np.mean(power[8:35, :], axis=0)
    high_power = np.mean(power[40:75, :], axis=0) + 1e-5
    formant_ratio = formant_power / high_power
    r_min, r_max = float(formant_ratio.min()), float(formant_ratio.max())
    r_span = max(1e-6, r_max - r_min)
    normalized_formant = np.clip((formant_ratio - r_min) / r_span, 0.0, 1.0)

    # Viseme openness combines vocalic formant dominance and energy
    viseme_openness = np.clip(0.7 * normalized_formant + 0.3 * phoneme_energy, 0.0, 1.0)
    # Mask out quiet / silent frames so background noise doesn't register as open mouth
    viseme_openness = np.where(phoneme_energy < 0.04, 0.0, viseme_openness).astype(np.float32)

    return phoneme_energy, viseme_openness


def build_mel_audio_cache(audio_wav_path: str, fps: float) -> AudioFeatureCache:
    """Build a synchronized AudioFeatureCache holding 16kHz mel-spectrogram chunks
    and phoneme-to-viseme acoustic parameters.
    """
    samples = load_audio_16k(audio_wav_path)
    if samples is None or len(samples) == 0:
        return AudioFeatureCache(None, audio_fps=fps)

    mel_spec = extract_mel_spectrogram(samples)
    phoneme_energy, viseme_openness = compute_phoneme_features(mel_spec)

    # Precompute per-video-frame 16-frame mel chunks for ultra-low latency inference
    total_mel_steps = mel_spec.shape[1]
    duration_s = total_mel_steps / MEL_FPS
    num_video_frames = max(1, int(math.floor(duration_s * fps)))
    chunks = []
    half_win = 8  # 16 // 2
    for fi in range(num_video_frames):
        ts = fi / fps if fps > 0 else 0.0
        center_step = int(round(ts * MEL_FPS))
        start = max(0, center_step - half_win)
        end = min(total_mel_steps, start + 16)
        c = np.zeros((MEL_N_MELS, 16), dtype=np.float32)
        actual = mel_spec[:, start:end]
        c[:, :actual.shape[1]] = actual
        chunks.append(c)

    chunks_arr = np.stack(chunks, axis=0) if chunks else None

    return AudioFeatureCache(
        features=chunks_arr,
        audio_fps=fps,
        mel_spectrogram=mel_spec,
        mel_fps=MEL_FPS,
        phoneme_energy=phoneme_energy,
        viseme_openness=viseme_openness,
    )

