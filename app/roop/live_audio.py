"""Optional microphone passthrough with a bounded, dynamically adjustable delay.

The application cannot create a system virtual audio cable itself. Users select
an installed cable endpoint (VB-CABLE, BlackHole, PulseAudio monitor, etc.) as
``output_device``. The bridge keeps audio entirely in memory and delays it by
the measured video latency.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Optional

import numpy as np

from roop.degrade import swallowed as _swallowed


class AudioDelayLine:
    """Thread-safe circular float32 audio buffer with a changing delay target."""

    def __init__(self, sample_rate: int = 48000, channels: int = 1,
                 max_delay_ms: float = 250.0):
        self.sample_rate = max(1, int(sample_rate))
        self.channels = max(1, int(channels))
        capacity = max(1024, int(self.sample_rate * max_delay_ms / 1000.0) + self.sample_rate)
        self._ring = np.zeros((capacity, self.channels), dtype=np.float32)
        self._capacity = capacity
        self._write_total = 0
        self._read_total = 0
        self._delay_frames = 0
        self._lock = threading.Lock()

    @property
    def delay_frames(self) -> int:
        with self._lock:
            return self._delay_frames

    def set_delay_ms(self, delay_ms: float) -> None:
        with self._lock:
            self._delay_frames = max(0, min(self._capacity - 1,
                                             int(round(float(delay_ms) * self.sample_rate / 1000.0))))

    def push(self, samples: np.ndarray) -> None:
        data = np.asarray(samples, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        if data.size == 0:
            return
        if data.shape[1] != self.channels:
            if data.shape[1] > self.channels:
                data = data[:, :self.channels]
            else:
                data = np.repeat(data[:, :1], self.channels, axis=1)
        with self._lock:
            for start in range(0, len(data), self._capacity):
                chunk = data[start:start + self._capacity]
                positions = (np.arange(len(chunk)) + self._write_total) % self._capacity
                self._ring[positions] = chunk
                self._write_total += len(chunk)
            self._read_total = max(self._read_total, self._write_total - self._capacity)

    def pop(self, frames: int) -> np.ndarray:
        count = max(0, int(frames))
        out = np.zeros((count, self.channels), dtype=np.float32)
        if count == 0:
            return out
        with self._lock:
            target_end = self._write_total - self._delay_frames
            available = target_end - self._read_total
            # If callbacks were delayed, jump to the newest window rather than
            # allowing an accidental queue to turn into seconds of latency.
            if available > count * 2:
                self._read_total = target_end - count
                available = count
            take = max(0, min(count, available))
            if take:
                start = self._read_total
                positions = (np.arange(take) + start) % self._capacity
                out[count - take:] = self._ring[positions]
                self._read_total += take
        return out


@dataclass
class AudioBridgeStatus:
    active: bool = False
    sample_rate: int = 48000
    channels: int = 1
    input_device: object = None
    output_device: object = None
    delay_ms: float = 0.0
    error: Optional[str] = None


class LiveAudioBridge:
    """SoundDevice input-to-output bridge for a user-selected virtual cable."""

    def __init__(self, sample_rate: int = 48000, channels: int = 1,
                 input_device=None, output_device=None):
        self.status = AudioBridgeStatus(sample_rate=int(sample_rate), channels=int(channels),
                                        input_device=input_device, output_device=output_device)
        self.delay = AudioDelayLine(self.status.sample_rate, self.status.channels)
        self._input_stream = None
        self._output_stream = None
        self._sd = None

    @staticmethod
    def list_devices():
        try:
            import sounddevice as sd
            return [dict(index=i, name=d.get("name", ""),
                         max_input_channels=d.get("max_input_channels", 0),
                         max_output_channels=d.get("max_output_channels", 0),
                         default_samplerate=d.get("default_samplerate", 0))
                    for i, d in enumerate(sd.query_devices())]
        except Exception as exc:
            _swallowed("live_audio.py:list_devices", exc, "audio unavailable")
            return []

    def set_visual_latency_ms(self, latency_ms: float) -> None:
        bounded = max(0.0, min(250.0, float(latency_ms)))
        self.delay.set_delay_ms(bounded)
        self.status.delay_ms = bounded

    def start(self) -> None:
        if self.status.active:
            return
        try:
            import sounddevice as sd
            self._sd = sd
            kwargs = dict(samplerate=self.status.sample_rate,
                          channels=self.status.channels,
                          dtype="float32", blocksize=0)
            self._input_stream = sd.InputStream(device=self.status.input_device, callback=self._on_input,
                                                **kwargs)
            self._output_stream = sd.OutputStream(device=self.status.output_device, callback=self._on_output,
                                                  **kwargs)
            self._input_stream.start()
            self._output_stream.start()
            self.status.active = True
            self.status.error = None
        except Exception as exc:
            self.status.error = str(exc)
            self.stop()
            raise RuntimeError(f"could not start microphone bridge: {exc}") from exc

    def _on_input(self, indata, frames, time_info, status) -> None:
        if status:
            self.status.error = str(status)
        self.delay.push(indata)

    def _on_output(self, outdata, frames, time_info, status) -> None:
        if status:
            self.status.error = str(status)
        outdata[:] = self.delay.pop(frames)

    def stop(self) -> None:
        for stream in (self._input_stream, self._output_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception as _degrade_error:
                    _swallowed("live_audio.py:stop", _degrade_error, "audio stream close continued")
        self._input_stream = None
        self._output_stream = None
        self.status.active = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
