"""Server state: workspace, project (sources/target/people/assignments), render jobs.

Everything heavy (model loading, detection, rendering) runs off the event
loop: request handlers call these methods through ``run_in_threadpool`` and
renders run on their own thread, which drives worker PROCESSES through
:class:`~face_engine.media.ipc_pool.FramePipeline`.
"""
from __future__ import annotations

import logging
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

from face_engine.server.processing import (
    FaceSwapWorker,
    FrameProcessor,
    ProcessorConfig,
    RenderParams,
    model_paths,
    required_models,
)

logger = logging.getLogger(__name__)

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
VIDEO_EXTS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class ServerSettings(BaseModel):
    """Server configuration (env: ``FACE_ENGINE_WORKSPACE`` etc., see ``__main__``)."""

    workspace: Path = Path(".cache/workspace")
    host: str = "127.0.0.1"
    port: int = 8765
    cors_origins: list[str] = Field(default_factory=lambda: [
        "http://127.0.0.1:5173", "http://localhost:5173"])
    max_image_bytes: int = 25 * 1024 * 1024
    max_video_bytes: int = 4 * 1024 ** 3
    max_sources: int = 8
    detect_frames: int = 8
    ui_dist: Path | None = None


class ProjectError(ValueError):
    """Invalid upload or request; the message is shown to the user (HTTP 422)."""


def safe_name(name: str, fallback: str = "file") -> str:
    stem = SAFE_NAME.sub("_", Path(name).name).strip("._")
    return stem or fallback


@dataclass
class Source:
    id: str
    name: str
    path: Path
    embedding: np.ndarray
    thumbnail: str


@dataclass
class Person:
    """A target identity found by detection (a cluster of faces across frames)."""

    id: str
    embedding: np.ndarray
    count: int
    frame: int
    bbox: list[float]
    score: float
    thumbnail: str


@dataclass
class Target:
    path: Path
    kind: Literal["image", "video"]
    width: int
    height: int
    frames: int
    fps: float
    duration: float


@dataclass
class Job:
    id: str
    params: RenderParams
    state: Literal["preparing", "rendering", "completed", "failed", "cancelled"] = "preparing"
    message: str = ""
    frames_done: int = 0
    frames_total: int = 0
    started: float = field(default_factory=time.monotonic)
    finished: float | None = None
    fps: float = 0.0
    latency_ms: float = 0.0
    output: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    @property
    def elapsed(self) -> float:
        return (self.finished or time.monotonic()) - self.started

    @property
    def eta(self) -> float | None:
        if self.state != "rendering" or self.fps <= 0:
            return None
        return max(self.frames_total - self.frames_done, 0) / self.fps

    def snapshot(self) -> dict[str, Any]:
        return {"id": self.id, "state": self.state, "message": self.message,
                "frames_done": self.frames_done, "frames_total": self.frames_total,
                "elapsed_s": round(self.elapsed, 2), "eta_s": None if self.eta is None
                else round(self.eta, 1), "fps": round(self.fps, 2),
                "latency_ms": round(self.latency_ms, 1), "output": self.output,
                "params": self.params.model_dump()}


class AppState:
    """The single project and job manager of one server process."""

    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings
        self.root = settings.workspace.expanduser().resolve()
        for sub in ("uploads", "outputs", "thumbnails"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.sources: dict[str, Source] = {}
        self.target: Target | None = None
        self.people: dict[str, Person] = {}
        self.assignments: dict[str, str] = {}
        self.job: Job | None = None
        self._registry: Any = None
        self._services: dict[str, Any] = {}
        self._preview: tuple[str, FrameProcessor] | None = None

    # ------------------------------------------------------------------ services
    @property
    def registry(self) -> Any:
        if self._registry is None:
            from face_engine.models.zoo import build_default_registry

            self._registry = build_default_registry()
        return self._registry

    def _analysis(self) -> tuple[Any, Any]:
        """Detector + identity encoder for uploads and detection (CUDA, loaded once)."""
        with self.lock:
            if "detector" not in self._services:
                from face_engine.core.config import EngineConfig, Provider
                from face_engine.core.execution import ExecutionEngine
                from face_engine.pipeline.detector import SCRFDDetector
                from face_engine.processors.swapper import IdentityEncoder

                engine = ExecutionEngine(EngineConfig(providers=[Provider.CUDA, Provider.CPU]))
                self._services["engine"] = engine
                self._services["detector"] = SCRFDDetector(
                    engine, self.registry.ensure("scrfd_10g_bnkps", show_progress=False))
                self._services["encoder"] = IdentityEncoder(
                    engine, self.registry.ensure("arcface_w600k_r50", show_progress=False))
            return self._services["detector"], self._services["encoder"]

    # ------------------------------------------------------------------ project
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    def thumb_path(self, name: str) -> Path:
        return self.root / "thumbnails" / name

    def _thumbnail(self, image: np.ndarray, bbox: list[float], name: str) -> str:
        h, w = image.shape[:2]
        x0, y0, x1, y1 = bbox
        pad = 0.25 * max(x1 - x0, y1 - y0)
        crop = image[max(int(y0 - pad), 0):min(int(y1 + pad), h),
                     max(int(x0 - pad), 0):min(int(x1 + pad), w)]
        if crop.size == 0:
            crop = image
        scale = 160 / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(self.thumb_path(name)), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return f"/api/thumbnails/{name}"

    def load_project(self, sources: list[tuple[str, Path]], target: tuple[str, Path]) -> None:
        """Validate uploaded files (already on disk) and make them the project.

        Raises:
            ProjectError: unreadable file, no face in a source, bad target.
        """
        detector, encoder = self._analysis()
        new_sources: dict[str, Source] = {}
        for name, path in sources:
            image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ProjectError(f"source {name!r} is not a readable image")
            faces = detector.detect(image)
            if not faces:
                raise ProjectError(f"no face found in source {name!r}")
            face = max(faces, key=lambda f: f.width * f.height)
            sid = uuid.uuid4().hex[:10]
            new_sources[sid] = Source(sid, name, path, encoder.embed(image, face).embedding,
                                      self._thumbnail(image, face.bbox.tolist(), f"src_{sid}.jpg"))
        tname, tpath = target
        new_target = self._validate_target(tname, tpath)
        with self.lock:
            if self.job is not None and self.job.state in ("preparing", "rendering"):
                raise ProjectError("a render is running; stop it before loading a new project")
            self.sources, self.target = new_sources, new_target
            self.people, self.assignments = {}, {}
            self._preview = None

    def _validate_target(self, name: str, path: Path) -> Target:
        ext = path.suffix.lower()
        if ext in IMAGE_EXTS:
            image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ProjectError(f"target {name!r} is not a readable image")
            h, w = image.shape[:2]
            return Target(path, "image", w, h, 1, 0.0, 0.0)
        from face_engine.media.capturer import probe

        try:
            info = probe(path)
        except (RuntimeError, ValueError) as exc:
            raise ProjectError(f"target {name!r} is not a readable video: {exc}") from None
        if info.frame_count <= 0:
            raise ProjectError(f"target {name!r} has no video frames")
        if info.color_profile.is_hdr:
            raise ProjectError(f"target {name!r} is HDR ({info.color_profile.value}); "
                               "tone-map it to SDR first")
        return Target(path, "video", info.width, info.height, info.frame_count,
                      float(info.fps), info.duration)

    def read_frame(self, index: int) -> np.ndarray:
        """One target frame (BGR)."""
        if self.target is None:
            raise ProjectError("no target loaded")
        if self.target.kind == "image":
            return cv2.imdecode(np.fromfile(str(self.target.path), np.uint8), cv2.IMREAD_COLOR)
        from face_engine.media.capturer import VideoSource

        index = int(np.clip(index, 0, self.target.frames - 1))
        return next(VideoSource(self.target.path).frames(index, index + 1)).image

    def detect_people(self, frames: int | None = None, same_person: float = 0.45) -> list[Person]:
        """Detect faces on evenly spaced frames and group them into people."""
        if self.target is None:
            raise ProjectError("no target loaded")
        detector, encoder = self._analysis()
        count = 1 if self.target.kind == "image" else min(frames or self.settings.detect_frames,
                                                          self.target.frames)
        indices = np.linspace(0, self.target.frames - 1, count).round().astype(int)
        clusters: list[dict[str, Any]] = []
        for idx in sorted(set(indices.tolist())):
            frame = self.read_frame(idx)
            for face in detector.detect(frame):
                emb = encoder.embed(frame, face).embedding
                sims = [float(c["mean"] @ emb) for c in clusters]
                best = int(np.argmax(sims)) if sims else -1
                if best >= 0 and sims[best] >= same_person:
                    c = clusters[best]
                    c["sum"] += emb
                    c["mean"] = c["sum"] / np.linalg.norm(c["sum"])
                    c["count"] += 1
                else:
                    c = {"sum": emb.copy(), "mean": emb, "count": 1, "best": None}
                    clusters.append(c)
                quality = face.score * face.width * face.height
                if c["best"] is None or quality > c["best"][0]:
                    c["best"] = (quality, idx, face, frame)
        people: dict[str, Person] = {}
        for c in sorted(clusters, key=lambda c: -c["count"]):
            pid = uuid.uuid4().hex[:10]
            _, idx, face, frame = c["best"]
            people[pid] = Person(pid, c["mean"].astype(np.float32), c["count"], int(idx),
                                 [round(float(v), 1) for v in face.bbox], round(face.score, 3),
                                 self._thumbnail(frame, face.bbox.tolist(), f"person_{pid}.jpg"))
        with self.lock:
            self.people = people
            self.assignments = {}
        return list(people.values())

    def assign(self, mapping: dict[str, str | None]) -> dict[str, str]:
        with self.lock:
            for person, source in mapping.items():
                if person not in self.people:
                    raise ProjectError(f"unknown person {person!r}")
                if source is None:
                    self.assignments.pop(person, None)
                elif source not in self.sources:
                    raise ProjectError(f"unknown source {source!r}")
                else:
                    self.assignments[person] = source
            self._preview = None
            return dict(self.assignments)

    def project_json(self) -> dict[str, Any]:
        with self.lock:
            t = self.target
            return {
                "sources": [{"id": s.id, "name": s.name, "thumbnail_url": s.thumbnail}
                            for s in self.sources.values()],
                "target": None if t is None else {
                    "kind": t.kind, "width": t.width, "height": t.height, "frames": t.frames,
                    "fps": t.fps, "duration": t.duration, "url": "/api/media/target"},
                "people": [{"id": p.id, "count": p.count, "frame": p.frame, "bbox": p.bbox,
                            "score": p.score, "thumbnail_url": p.thumbnail}
                           for p in self.people.values()],
                "assignments": dict(self.assignments),
                "job": None if self.job is None else self.job.snapshot(),
            }

    # ------------------------------------------------------------------ processing
    def processor_config(self, params: RenderParams) -> ProcessorConfig:
        if not self.sources:
            raise ProjectError("no source face loaded")
        if params.swapper_model == "alphaface_256":
            raise ProjectError("alphaface_256 is not available: no public model release exists")
        with self.lock:
            return ProcessorConfig(
                params=params,
                model_paths=model_paths(required_models(params), self.registry),
                sources={k: s.embedding for k, s in self.sources.items()},
                target_refs={k: p.embedding for k, p in self.people.items()},
                assignments=dict(self.assignments))

    def preview(self, index: int, params: RenderParams) -> tuple[np.ndarray, dict[str, int]]:
        """Process one target frame with ``params`` in this process."""
        config = self.processor_config(params)
        key = params.model_dump_json() + repr(sorted(config.assignments.items()))
        with self.lock:
            if self._preview is None or self._preview[0] != key:
                if self._preview is not None:
                    self._preview[1].close()
                self._preview = (key, FrameProcessor(config))
            processor = self._preview[1]
        frame = self.read_frame(index)
        out, stats = processor.process(frame)
        return out, {"faces": stats.faces, "swapped": stats.swapped}

    # ------------------------------------------------------------------ jobs
    def start_job(self, params: RenderParams) -> Job:
        with self.lock:
            if self.target is None:
                raise ProjectError("no target loaded")
            if self.job is not None and self.job.state in ("preparing", "rendering"):
                raise RuntimeError("a render is already running")
            config = self.processor_config(params)
            job = Job(id=uuid.uuid4().hex[:10], params=params, frames_total=self.target.frames)
            job.thread = threading.Thread(target=self._run_job, args=(job, config, self.target),
                                          name=f"render-{job.id}", daemon=True)
            self.job = job
        job.thread.start()
        return job

    def stop_job(self, timeout: float = 60.0) -> Job | None:
        job = self.job
        if job is None:
            return None
        job.cancel.set()
        if job.thread is not None:
            job.thread.join(timeout)
        return job

    def _run_job(self, job: Job, config: ProcessorConfig, target: Target) -> None:
        stem = safe_name(target.path.stem, "render")
        try:
            if target.kind == "image":
                job.state = "rendering"
                out = self.outputs_dir() / f"{stem}_{job.id}.png"
                processor = FrameProcessor(config)
                try:
                    result, _ = processor.process(self.read_frame(0))
                finally:
                    processor.close()
                if job.cancel.is_set():
                    raise _Cancelled()
                cv2.imencode(".png", result)[1].tofile(str(out))
                job.frames_done = 1
            else:
                out = self._render_video(job, config, target, stem)
            job.output = f"/api/outputs/{out.name}"
            job.state = "completed"
        except _Cancelled:
            job.state = "cancelled"
            job.message = "stopped by the user"
        except Exception as exc:
            from face_engine.media.ipc_pool import PipelineCancelled

            if isinstance(exc, PipelineCancelled) or job.cancel.is_set():
                job.state, job.message = "cancelled", "stopped by the user"
            else:
                logger.exception("render %s failed", job.id)
                job.state, job.message = "failed", str(exc)[-2000:]
        finally:
            job.finished = time.monotonic()

    def _render_video(self, job: Job, config: ProcessorConfig, target: Target, stem: str) -> Path:
        from face_engine.media.capturer import VideoSource
        from face_engine.media.ffmpeg_pipe import FFmpegWriter
        from face_engine.media.ipc_pool import FramePipeline, VideoFrames

        info = VideoSource(target.path).info
        out = self.outputs_dir() / f"{stem}_{job.id}.mp4"
        pipeline = FramePipeline(VideoFrames(str(target.path)), FaceSwapWorker(config),
                                 (info.height, info.width, 3), workers=config.params.workers,
                                 slots=max(4, 2 * config.params.workers + 2),
                                 init=FaceSwapWorker(config).init, stall_timeout=600.0)
        job.state = "rendering"
        last = [time.monotonic()]
        with FFmpegWriter(out, info.width, info.height, info.fps, audio=target.path,
                          color=info.color_profile, expected_frames=info.frame_count) as writer:
            def sink(seq: int, frame: np.ndarray) -> None:
                writer.write(frame)
                now = time.monotonic()
                dt, last[0] = now - last[0], now
                job.frames_done = seq + 1
                if seq > 0:  # the first interval includes model loading
                    job.latency_ms = 0.8 * job.latency_ms + 200.0 * dt if job.latency_ms else dt * 1e3
                    job.fps = 1000.0 / job.latency_ms if job.latency_ms > 0 else 0.0

            try:
                pipeline.run(sink, cancel=job.cancel)
            except BaseException:
                writer.abort()
                raise
            writer.close()
        return out

    def shutdown(self) -> None:
        self.stop_job(timeout=30)
        if self._preview is not None:
            self._preview[1].close()
        if "engine" in self._services:
            self._services["engine"].close()

    def clear_uploads(self) -> None:
        shutil.rmtree(self.root / "uploads", ignore_errors=True)
        (self.root / "uploads").mkdir(parents=True, exist_ok=True)


class _Cancelled(Exception):
    pass
