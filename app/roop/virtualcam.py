"""Dedicated low-latency webcam mode.

This replaces the legacy per-frame ``live_swap`` loop while keeping its public
start/stop functions compatible with existing callers. The live path never
writes frames to disk and never shares its processing manager with batch jobs.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

import roop.globals
import ui.globals as ui_globals
from roop.degrade import swallowed as _swallowed

from roop.live_mode import (
    LiveCapture,
    LiveCaptureConfig,
    LiveFrameProcessor,
    LiveStats,
)


cam_active = False
cam_thread = None
vcam = None
_engine = None
_engine_lock = threading.Lock()


class LiveVirtualCamera:
    def __init__(self, camera_index: int, width: int, height: int, fps: float,
                 stream_obs: bool, use_xseg: bool, restore_mouth: bool,
                 detector_interval: int = 6, audio_input_device=None,
                 audio_output_device=None, audio_sample_rate: int = 48000,
                 audio_channels: int = 1):
        self.capture = LiveCapture(LiveCaptureConfig(
            camera_index=int(camera_index), width=int(width), height=int(height),
            fps=float(fps), buffer_size=1))
        self.stream_obs = bool(stream_obs)
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1.0, float(fps))
        self.use_xseg = bool(use_xseg)
        self.restore_mouth = bool(restore_mouth)
        self.detector_interval = max(1, min(30, int(detector_interval)))
        self.audio_input_device = audio_input_device
        self.audio_output_device = audio_output_device
        self.audio_sample_rate = int(audio_sample_rate)
        self.audio_channels = int(audio_channels)
        self.active = False
        self.error = None
        self.thread = None
        self.camera = None
        self.camera_format = None
        self._uses_rgb = False
        self.audio = None
        self.processor = None
        self.stats = LiveStats()

    @staticmethod
    def _options(use_xseg: bool, restore_mouth: bool):
        from roop.ProcessOptions import ProcessOptions
        from roop.core import get_processing_plugins

        mask_engine = "mask_xseg" if use_xseg else None
        swap_model = getattr(getattr(roop.globals, "CFG", None), "swap_model", "realswap")
        previous_enhancer = getattr(roop.globals, "selected_enhancer", None)
        # The live path deliberately has no CodeFormer/GPEN/UltraMax pass. This
        # is a session-local processor graph and does not change the user's batch
        # setting after the graph has been constructed.
        roop.globals.selected_enhancer = "None"
        try:
            plugins = get_processing_plugins(mask_engine, swap_model=swap_model)
        finally:
            roop.globals.selected_enhancer = previous_enhancer
        return ProcessOptions(
            plugins,
            getattr(roop.globals, "distance_threshold", 1),
            getattr(roop.globals, "blend_ratio", 0.8),
            "all", 0, None, None, 1,
            getattr(roop.globals, "subsample_size", 256), False,
            restore_mouth,
            stabilize_face=False,
            stabilize_enhancer=False,
            stabilize_mask=False,
            stabilize_landmarks=False,
            stabilize_hf_texture=False,
            swap_model=swap_model,
            live_mode=True,
        )

    def _build_processor(self):
        from roop.core import live_swap
        from roop.face_util import get_all_faces
        from roop.scene_detector import ContentAwareSceneDetector, flush_pipeline_temporal_buffers

        options = self._options(self.use_xseg, self.restore_mouth)
        scene_detector = ContentAwareSceneDetector(use_gpu=False)

        def swap_frame(frame):
            return live_swap(frame, options)

        def detect_faces(frame):
            # Detection is deliberately called only by LiveFrameProcessor on its
            # keyframe cadence. Intermediate frames use LK, not a hidden second
            # detector or temporal detector interval.
            return get_all_faces(frame) or []

        return LiveFrameProcessor(
            swap_frame=swap_frame,
            detect_faces=detect_faces,
            detector_interval=self.detector_interval,
            scene_detector=scene_detector,
            flush_callback=lambda: flush_pipeline_temporal_buffers(),
        )

    def _open_virtual_camera(self):
        if not self.stream_obs:
            return
        import pyvirtualcam
        pixel_format = getattr(pyvirtualcam.PixelFormat, "RGB", None)
        if pixel_format is None:
            pixel_format = pyvirtualcam.PixelFormat.BGR
        self.camera_format = pixel_format
        self._uses_rgb = pixel_format == getattr(pyvirtualcam.PixelFormat, "RGB", None)
        self.camera = pyvirtualcam.Camera(
            width=self.width, height=self.height, fps=self.fps,
            fmt=pixel_format, print_fps=False)
        print(f"[Live] virtual camera: {self.camera.device} ({self.camera.native_fmt})")

    def _open_audio(self):
        if self.audio_input_device is None and self.audio_output_device is None:
            return
        from roop.live_audio import LiveAudioBridge
        self.audio = LiveAudioBridge(
            sample_rate=self.audio_sample_rate, channels=self.audio_channels,
            input_device=self.audio_input_device, output_device=self.audio_output_device)
        try:
            self.audio.start()
        except Exception as exc:
            _swallowed("virtualcam.py:audio_start", exc, "video continued without audio")
            self.error = f"audio disabled: {exc}"
            self.audio = None

    def start(self):
        if self.active:
            return
        self.error = None
        self.processor = self._build_processor()
        self.capture.start()
        try:
            self._open_virtual_camera()
            self._open_audio()
        except Exception:
            self.capture.stop()
            self._close_outputs()
            raise
        self.stats = LiveStats(started_at=time.perf_counter())
        self.active = True
        self.thread = threading.Thread(target=self._run, name="roop-live-camera", daemon=True)
        self.thread.start()

    def _run(self):
        global cam_active
        try:
            while self.active:
                item = self.capture.mailbox.get(timeout=0.25)
                if item is None:
                    if not self.capture.active:
                        break
                    continue
                frame, captured_at, _sequence = item
                result = self.processor.process(frame, captured_at=captured_at)
                ui_globals.ui_camera_frame = result.frame
                if self.camera is not None:
                    send_frame = result.frame
                    if self._uses_rgb:
                        send_frame = cv2.cvtColor(send_frame, cv2.COLOR_BGR2RGB)
                    self.camera.send(np.ascontiguousarray(send_frame))
                    self.camera.sleep_until_next_frame()
                if self.audio is not None:
                    self.audio.set_visual_latency_ms(result.latency_ms)
                self.stats.update(result, self.capture.mailbox.dropped)
        except Exception as exc:
            self.error = str(exc)
            print(f"[Live] stopped: {exc}")
        finally:
            self.active = False
            cam_active = False
            self._close_outputs()

    def _close_outputs(self):
        if self.audio is not None:
            self.audio.stop()
            self.audio = None
        if self.camera is not None:
            try:
                self.camera.close()
            except Exception as exc:
                _swallowed("virtualcam.py:camera_close", exc, "camera cleanup continued")
                pass
            self.camera = None

    def stop(self):
        global cam_active
        self.active = False
        self.capture.stop()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2.0)
        self.thread = None
        self._close_outputs()
        cam_active = False

    def status(self):
        audio_status = None
        if self.audio is not None:
            audio_status = vars(self.audio.status).copy()
        return {
            "active": bool(self.active),
            "error": self.error or self.capture.error,
            "backend": int(self.capture.backend),
            "camera_index": self.capture.config.camera_index,
            "resolution": f"{self.width}x{self.height}",
            "fps_target": self.fps,
            "detector_interval": self.detector_interval,
            "buffer_size": 1,
            "virtual_camera": self.camera.device if self.camera is not None else None,
            "audio": audio_status,
            "stats": vars(self.stats).copy(),
        }


def start_live_camera(payload: dict):
    global _engine, cam_active, cam_thread
    with _engine_lock:
        if _engine is not None and _engine.active:
            return
        resolution = str(payload.get("resolution", "1280x720"))
        try:
            width, height = (int(x) for x in resolution.lower().split("x", 1))
        except Exception as exc:
            _swallowed("virtualcam.py:resolution", exc, "using default resolution")
            width, height = 1280, 720
        _engine = LiveVirtualCamera(
            camera_index=int(payload.get("cam_number", 0)),
            width=width, height=height,
            fps=float(payload.get("fps", 30.0)),
            stream_obs=bool(payload.get("stream_obs", False)),
            use_xseg=bool(payload.get("use_xseg", False)),
            restore_mouth=bool(payload.get("restore_mouth", False)),
            detector_interval=int(payload.get("detector_interval", 6)),
            audio_input_device=payload.get("audio_input_device"),
            audio_output_device=payload.get("audio_output_device"),
            audio_sample_rate=int(payload.get("audio_sample_rate", 48000)),
            audio_channels=int(payload.get("audio_channels", 1)),
        )
        _engine.start()
        cam_active = True
        cam_thread = _engine.thread


def start_virtual_cam(streamobs, use_xseg, use_mouthrestore, cam_number, resolution):
    """Backward-compatible legacy entry point used by older UI callers."""
    start_live_camera({
        "stream_obs": streamobs,
        "use_xseg": use_xseg,
        "restore_mouth": use_mouthrestore,
        "cam_number": cam_number,
        "resolution": resolution,
    })


def stop_virtual_cam():
    global _engine, cam_active, cam_thread
    with _engine_lock:
        if _engine is not None:
            _engine.stop()
        _engine = None
        cam_thread = None
        cam_active = False


def live_camera_status():
    with _engine_lock:
        if _engine is None:
            return {"active": False}
        return _engine.status()


def audio_devices():
    from roop.live_audio import LiveAudioBridge
    return LiveAudioBridge.list_devices()
