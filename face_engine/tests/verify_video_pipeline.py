"""Measure total pipeline FPS (decode + inference + encode) on a video, e.g. 1080p.

Not collected by pytest. Usage::

    python -m face_engine.tests.verify_video_pipeline VIDEO [--frames 300]
        [--work none,swap] [--io software,hardware] [--pool 2]

``--io``:
  software  PyAV software decode -> BGR on the host -> upload -> ... ->
            download -> libx264 (``FFmpegWriter``). The typical pipeline.
  hardware  NVDEC (child process) -> NV12 -> GPU colour conversion -> ... ->
            NVENC (``HardwareVideoDecoder(backend="nvdec")`` +
            ``NVENCVideoWriter.write_tensor``).
  mixed     software decode on a thread -> pinned upload -> ... -> NVENC
            (``HardwareVideoDecoder(backend="software")`` + NVENC).
``--work``:
  none      pass-through: the I/O ceiling.
  swap      the Stage 2/3 GPU chain per frame: strided detection
            (``StridedFaceTracker``), batched HyperSwap with the first face of
            the video's first frame as the source, the GPU composite mask
            (box x valid; ``--xseg`` adds XSeg), paste-back.
``--pool N``: additionally render with ``SegmentWorkerPool`` (N segments on GPU
0, ``swap`` work, hardware I/O).

Every arm writes a real MP4 (checked: frame count, codec) and reports wall
clock fps over the whole run: open, decode, work, encode, close. Arms run
twice in reverse order (A B B A) so the one that pays TensorRT/cuDNN warm-up
is not always the same.
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from face_engine.core.config import EngineConfig, Provider
from face_engine.core.execution import ExecutionEngine
from face_engine.media.capturer import VideoSource
from face_engine.media.decoder import HardwareVideoDecoder
from face_engine.media.encoder import NVENCVideoWriter
from face_engine.media.ffmpeg_pipe import FFmpegWriter

_WORK: dict[str, Any] = {}


def swap_processor(device: Any = "cuda", source: str = "", xseg: bool = False,
                   providers: str = "trt") -> Any:
    """Factory used in-process and by ``SegmentWorkerPool`` workers."""
    from face_engine.models.zoo import build_default_registry
    from face_engine.pipeline.aligner import warp_face_inverse_cuda
    from face_engine.pipeline.detector import SCRFDDetector
    from face_engine.pipeline.masker import GPUMasker, MaskerConfig
    from face_engine.pipeline.tracker import StridedFaceTracker, TrackerConfig
    from face_engine.processors import BatchedFaceSwapper, GPUIdentityEncoder

    registry = build_default_registry()
    chain = ([Provider.TENSORRT, Provider.CUDA, Provider.CPU] if providers == "trt"
             else [Provider.CUDA, Provider.CPU])
    engine = ExecutionEngine(EngineConfig(providers=chain))
    detector = SCRFDDetector(ExecutionEngine(EngineConfig(providers=[Provider.CUDA,
                                                                     Provider.CPU])),
                             registry.ensure("scrfd_10g_bnkps", show_progress=False))
    tracker = StridedFaceTracker(detector, TrackerConfig(detection_stride=3))
    swapper = BatchedFaceSwapper(engine, "hyperswap_1a_256",
                                 registry.ensure("hyperswap_1a_256", show_progress=False))
    encoder = GPUIdentityEncoder(engine, registry.ensure("arcface_w600k_r50",
                                                         show_progress=False))
    masker = GPUMasker(engine, registry.ensure("xseg_3", show_progress=False) if xseg else None,
                       None, MaskerConfig(crop_size=256))
    first = next(iter(HardwareVideoDecoder(source, end=1, batch_size=1, device=device,
                                           backend="software")))
    faces = detector.detect_cuda(first.frames.float())
    if len(faces) == 0:
        raise RuntimeError("no face on the first frame to use as the source")
    swapper.set_source(encoder.embed(first.frames.float(), faces.kps[:1])[0])
    stats = {"faces": 0, "frames": 0}

    def process(batch: Any) -> torch.Tensor:
        out = []
        for frame in batch.frames:
            f = frame[None].float()
            found = tracker.update(f)
            stats["frames"] += 1
            if len(found) == 0:
                out.append(f)
                continue
            kps = found.detections.kps
            res = swapper.swap(f, kps)
            mask = masker.generate(f, kps).mask
            if res.model_mask is not None:
                mask = mask * res.model_mask
            out.append(warp_face_inverse_cuda(f, res.crops, res.matrices, mask))
            stats["faces"] += len(found)
        return torch.cat(out)

    process.stats = stats  # type: ignore[attr-defined]
    return process


def run_arm(video: Path, out: Path, io: str, work: str, frames: int, xseg: bool) -> dict[str, Any]:
    src = VideoSource(video)
    info = src.info
    n = min(frames, src.frame_count)
    process = _WORK.get(work)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if io in ("hardware", "mixed"):
        decoder = HardwareVideoDecoder(video, end=n, batch_size=4,
                                       backend="nvdec" if io == "hardware" else "software")
        writer: Any = NVENCVideoWriter(out, info.width, info.height, info.fps,
                                       expected_frames=n, color=info.color_profile)
        for batch in decoder:
            writer.write_tensor(process(batch) if process else batch.frames)
        backend = decoder.backend
    else:
        writer = FFmpegWriter(out, info.width, info.height, info.fps, expected_frames=n,
                              color=info.color_profile)
        batch_frames: list[Any] = []

        class _Batch:
            def __init__(self, frames: torch.Tensor) -> None:
                self.frames = frames

        def flush() -> None:
            t = torch.stack([torch.from_numpy(f).cuda().permute(2, 0, 1) for f in batch_frames])
            result = process(_Batch(t)) if process else t
            for frame in result.round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy():
                writer.write(frame)
            batch_frames.clear()

        for f in src.frames(0, n):
            batch_frames.append(f.image)
            if len(batch_frames) == 4:
                flush()
        if batch_frames:
            flush()
        backend = "software"
    report = writer.close()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    assert report.frames == n, (report.frames, n)
    return {"fps": n / seconds, "frames": n, "decoder": backend,
            "writer": type(writer).__name__}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--work", default="none,swap")
    ap.add_argument("--io", default="software,hardware,mixed")
    ap.add_argument("--pool", type=int, default=0)
    ap.add_argument("--xseg", action="store_true")
    ap.add_argument("--out", default=".temp/verify_video")
    args = ap.parse_args()
    video = Path(args.video)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = VideoSource(video).info
    print(f"{video.name}: {info.width}x{info.height} {info.codec} @ {info.fps} "
          f"({float(info.fps):.3f} fps), {min(args.frames, VideoSource(video).frame_count)} frames"
          f"  GPU {torch.cuda.get_device_name()}", flush=True)
    for work in args.work.split(","):
        if work == "swap":
            _WORK["swap"] = swap_processor(source=str(video), xseg=args.xseg)
        ios = args.io.split(",")
        runs: dict[str, list[dict[str, Any]]] = {io: [] for io in ios}
        for io in [*ios, *reversed(ios)]:
            runs[io].append(run_arm(video, out_dir / f"{work}_{io}.mp4", io, work,
                                    args.frames, args.xseg))
        for io, rs in runs.items():
            fps = [r["fps"] for r in rs]
            print(f"  work={work:5s} io={io:9s} ({rs[0]['decoder']} -> {rs[0]['writer']}): "
                  f"{statistics.mean(fps):6.1f} fps  (runs {', '.join(f'{x:.1f}' for x in fps)})",
                  flush=True)
        if work == "swap":
            stats = _WORK["swap"].stats
            print(f"  swap: {stats['faces']} faces over {stats['frames']} processed frames",
                  flush=True)
    if args.pool:
        from face_engine.media.worker_pool import SegmentWorkerPool

        pool = SegmentWorkerPool([0] * args.pool, encoder="nvenc")
        rep = pool.run(video, out_dir / "pool.mp4",
                       processor="face_engine.tests.verify_video_pipeline:swap_processor",
                       processor_kwargs={"source": str(video), "xseg": args.xseg})
        print(f"  pool x{args.pool} (whole clip, swap, hardware I/O, incl. process start): "
              f"{rep.fps:6.1f} fps over {rep.frames} frames; segments "
              + ", ".join(f"{r['frames']}f {r['frames'] / r['seconds']:.1f} fps" for r in rep.segments),
              flush=True)


if __name__ == "__main__":
    main()
