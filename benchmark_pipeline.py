"""Automated Benchmarking Harness for roop-ultimate on D:\k1.mp4.

Iterates through swapper models:
  ['inswapper_128', 'inswapper_128_fp16', 'hyperswap_256', 'hififace_256']
crossed against masking engines:
  ['none', 'dfl_xseg_v2', 'bisenet_face_parser', 'face_occluder_v3', 'sam2_tracked']

For each permutation:
  - Executes inference across a continuous 300-frame slice of "D:\k1.mp4".
  - Discards first 10 warm-up frames from latency calculations.
  - Measures Mean, 95th-percentile, and 99th-percentile latency per frame (ms).
  - Measures Overall throughput in FPS.
  - Tracks peak NVIDIA GPU VRAM consumption via pynvml / torch.cuda.max_memory_allocated().
  - Computes inter-frame temporal boundary variance (SSIM variance of consecutive mask boundaries).
  - Handles CUDA OOM errors gracefully without interrupting subsequent passes.
  - Exports benchmark_results.json and prints a cleanly formatted Markdown summary table.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Ensure repo root and app on path
ROOT_DIR = Path(__file__).resolve().parent
APP_DIR = ROOT_DIR / "app"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from face_engine.core.zero_copy_engine import ZeroCopyExecutionEngine
from face_engine.models.zoo import build_default_registry
from face_engine.pipeline.hybrid_masker import HybridVideoMasker

# Pynvml for exact physical VRAM query
try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False


def get_peak_vram_mb(device_id: int = 0) -> float:
    """Return peak allocated VRAM in MiB."""
    torch_peak = torch.cuda.max_memory_allocated(device_id) / (1024 * 1024) if torch.cuda.is_available() else 0.0
    if _HAS_NVML:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            nvml_used = info.used / (1024 * 1024)
            return max(torch_peak, nvml_used)
        except Exception:
            pass
    return torch_peak


def ssim_boundary_variance(prev_mask: np.ndarray, curr_mask: np.ndarray) -> float:
    """Calculates structural boundary difference between consecutive masks."""
    if prev_mask is None or curr_mask is None:
        return 0.0
    diff = np.abs(curr_mask.astype(np.float32) - prev_mask.astype(np.float32))
    return float(np.var(diff))


class BenchmarkSuite:
    def __init__(
        self,
        video_path: str = r"D:\k1.mp4",
        slice_frames: int = 300,
        warmup_frames: int = 10,
        device_id: int = 0,
    ) -> None:
        self.video_path = Path(video_path)
        self.slice_frames = slice_frames
        self.warmup_frames = warmup_frames
        self.device_id = device_id
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.registry = build_default_registry()

        # Sanity check video
        self.probe_video()

    def probe_video(self) -> dict[str, Any]:
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Benchmark video file does not exist: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        cap.release()

        self.video_info = {
            "path": str(self.video_path),
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "codec": codec,
        }
        print(f"[Sanity Check] Video OK: {self.video_info}")
        return self.video_info

    def resolve_model_path(self, model_name: str) -> Path:
        """Find model file across registry, app/models, and .cache/models."""
        candidates = [
            self.registry.local_path(model_name) if model_name in self.registry else None,
            Path("app/models") / f"{model_name}.onnx",
            Path(".cache/models") / f"{model_name}.onnx",
        ]
        # Specific filename mappings
        file_map = {
            "hififace_256": "app/models/hififace_unofficial_256.onnx",
            "hyperswap_256": "app/models/hyperswap_1a_256.onnx",
            "face_occluder_v3": "app/models/xseg_3.onnx",
            "dfl_xseg_v2": "app/models/xseg.onnx",
            "face_parser_bisenet34": ".cache/models/bisenet_resnet_34.onnx",
        }
        if model_name in file_map:
            candidates.insert(0, Path(file_map[model_name]))

        for cand in candidates:
            if cand is not None and cand.is_file():
                return cand.resolve()

        raise FileNotFoundError(f"Model {model_name} could not be resolved from {candidates}")

    def get_reference_face(self) -> tuple[np.ndarray, np.ndarray]:
        """Provides a standardized reference face crop and 512-d normalized embedding."""
        t1_path = Path("face_engine/tests/fixtures/t1.jpg")
        if not t1_path.is_file():
            import insightface
            t1_path = Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"

        if t1_path.is_file():
            img = cv2.imread(str(t1_path))
        else:
            img = np.full((512, 512, 3), 128, dtype=np.uint8)

        # Standardized 512-d unit vector
        rng = np.random.RandomState(42)
        emb = rng.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        return img, emb

    def load_frame_slice(self) -> list[np.ndarray]:
        """Load continuous slice of frames from video."""
        cap = cv2.VideoCapture(str(self.video_path))
        frames = []
        for _ in range(self.slice_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames.append(frame)
        cap.release()
        if len(frames) < self.slice_frames:
            print(f"[Warning] Video only yielded {len(frames)} frames of requested {self.slice_frames}")
        return frames

    def run_permutation(
        self,
        swapper_name: str,
        masker_name: str,
        frames: list[np.ndarray],
        ref_latent: np.ndarray,
    ) -> dict[str, Any]:
        """Run single permutation benchmark across continuous frame slice."""
        print(f"\n---> Permutation: Swapper='{swapper_name}' x Masker='{masker_name}' ({len(frames)} frames)")

        # Clear VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        gc.collect()

        latencies_ms = []
        boundary_variances = []
        prev_mask = None
        oom_occurred = False
        oom_error = ""

        # Initialize engines
        try:
            # Swapper resolution & model
            swapper_path = self.resolve_model_path(swapper_name)
            res = 128 if "128" in swapper_name else 256

            swapper_engine = ZeroCopyExecutionEngine(device_id=self.device_id)
            swapper_sess = swapper_engine.load_session(swapper_path, trt_fp16=True)

            # Masker engine
            masker_sess = None
            masker_engine = None
            hybrid_masker = None

            if masker_name == "face_occluder_v3":
                masker_engine = ZeroCopyExecutionEngine(device_id=self.device_id)
                m_path = self.resolve_model_path("face_occluder_v3")
                masker_sess = masker_engine.load_session(m_path, trt_fp16=True)
            elif masker_name == "dfl_xseg_v2":
                masker_engine = ZeroCopyExecutionEngine(device_id=self.device_id)
                m_path = self.resolve_model_path("dfl_xseg_v2")
                masker_sess = masker_engine.load_session(m_path, trt_fp16=True)
            elif masker_name == "bisenet_face_parser":
                masker_engine = ZeroCopyExecutionEngine(device_id=self.device_id)
                m_path = self.resolve_model_path("face_parser_bisenet34")
                masker_sess = masker_engine.load_session(m_path, trt_fp16=True)
            elif masker_name == "sam2_tracked":
                hybrid_masker = HybridVideoMasker(device=self.device)
                hybrid_masker.initialize()

            # Pre-allocate latent tensor
            latent_t = torch.from_numpy(ref_latent).to(device=self.device, dtype=torch.float32).view(1, 512)

            # Fixed landmark proxy for center crop
            kps = np.array([
                [res * 0.35, res * 0.45],
                [res * 0.65, res * 0.45],
                [res * 0.50, res * 0.62],
                [res * 0.38, res * 0.78],
                [res * 0.62, res * 0.78],
            ], dtype=np.float32)

            t0_total = time.perf_counter()

            for i, frame in enumerate(frames):
                t_frame_start = time.perf_counter()

                # Crop to native resolution
                center_crop = cv2.resize(frame, (res, res))

                # 1. Swapper inference
                mean = (0.0, 0.0, 0.0) if "inswapper" in swapper_name else (0.5, 0.5, 0.5)
                std = (1.0, 1.0, 1.0) if "inswapper" in swapper_name else (0.5, 0.5, 0.5)
                rgb = center_crop[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
                norm_crop = ((rgb - np.array(mean).reshape(3, 1, 1)) / np.array(std).reshape(3, 1, 1))[None, ...]
                crop_t = torch.from_numpy(norm_crop).to(device=self.device, dtype=torch.float32)

                feeds = {}
                for inp in swapper_sess.input_names:
                    if inp == "source":
                        feeds[inp] = latent_t
                    else:
                        feeds[inp] = crop_t

                swapper_res = swapper_engine.run_zero_copy(swapper_sess, feeds)
                swapped_crop = swapper_res[swapper_sess.output_names[0]]

                # 2. Masker inference
                mask_result = None
                if masker_name == "none":
                    mask_result = np.ones((res, res), dtype=np.float32)
                elif masker_name in ("face_occluder_v3", "dfl_xseg_v2"):
                    # NHWC [0, 1] input at native 256x256
                    mask_crop_256 = center_crop if res == 256 else cv2.resize(center_crop, (256, 256))
                    m_blob = torch.from_numpy((mask_crop_256.astype(np.float32) / 255.0)[None, ...]).to(device=self.device)
                    inp_name = masker_sess.input_names[0]
                    m_res = masker_engine.run_zero_copy(masker_sess, {inp_name: m_blob})
                    m_raw = m_res[masker_sess.output_names[0]].squeeze().cpu().numpy()
                    mask_256 = 1.0 - np.clip(m_raw, 0.0, 1.0)
                    mask_result = mask_256 if res == 256 else cv2.resize(mask_256, (res, res))
                elif masker_name == "bisenet_face_parser":
                    # Resize to native 512x512
                    p_crop = cv2.resize(center_crop, (512, 512))
                    p_rgb = p_crop[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
                    p_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1) * 255.0
                    p_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1) * 255.0
                    p_blob = torch.from_numpy(((p_rgb - p_mean) / p_std)[None, ...]).to(device=self.device)
                    inp_name = masker_sess.input_names[0]
                    m_res = masker_engine.run_zero_copy(
                        masker_sess,
                        {inp_name: p_blob},
                        output_shapes={masker_sess.output_names[0]: (1, 19, 512, 512)},
                        unreturned_outputs=masker_sess.output_names[1:],
                    )
                    labels = m_res[masker_sess.output_names[0]].argmax(dim=1).squeeze().cpu().numpy()
                    mask_512 = np.isin(labels, [1, 2, 3, 4, 5, 10, 11, 12, 13]).astype(np.float32)
                    mask_result = cv2.resize(mask_512, (res, res))
                elif masker_name == "sam2_tracked":
                    m_t = hybrid_masker.step(i, center_crop, kps)
                    mask_result = m_t.squeeze().cpu().numpy()

                # Sync GPU
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device_id)

                elapsed_ms = (time.perf_counter() - t_frame_start) * 1000.0

                # Warmup filter
                if i >= self.warmup_frames:
                    latencies_ms.append(elapsed_ms)
                    if prev_mask is not None and mask_result is not None:
                        variance = ssim_boundary_variance(prev_mask, mask_result)
                        boundary_variances.append(variance)
                    prev_mask = mask_result

            total_elapsed = time.perf_counter() - t0_total
            peak_vram = get_peak_vram_mb(self.device_id)

            # Cleanup engines
            swapper_engine.cleanup()
            if masker_engine is not None:
                masker_engine.cleanup()
            if hybrid_masker is not None:
                hybrid_masker.release()

        except torch.cuda.OutOfMemoryError as e:
            oom_occurred = True
            oom_error = str(e)
            print(f"[OOM Encountered] Permutation {swapper_name} x {masker_name} failed with CUDA OOM: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            peak_vram = get_peak_vram_mb(self.device_id)
            total_elapsed = 0.0
        except Exception as e:
            oom_error = str(e)
            print(f"[Error] Permutation {swapper_name} x {masker_name} error: {e}")
            peak_vram = get_peak_vram_mb(self.device_id)
            total_elapsed = 0.0

        if not oom_occurred and latencies_ms:
            mean_lat = float(np.mean(latencies_ms))
            p95_lat = float(np.percentile(latencies_ms, 95))
            p99_lat = float(np.percentile(latencies_ms, 99))
            fps = float(len(frames) / (total_elapsed if total_elapsed > 0 else 1.0))
            mean_boundary_var = float(np.mean(boundary_variances)) if boundary_variances else 0.0
        else:
            mean_lat = p95_lat = p99_lat = fps = mean_boundary_var = 0.0

        result = {
            "swapper": swapper_name,
            "masker": masker_name,
            "status": "OOM" if oom_occurred else ("ERROR" if oom_error else "SUCCESS"),
            "error": oom_error,
            "frames_evaluated": len(latencies_ms),
            "warmup_frames_excluded": self.warmup_frames,
            "mean_latency_ms": round(mean_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "p99_latency_ms": round(p99_lat, 2),
            "fps": round(fps, 2),
            "peak_vram_mb": round(peak_vram, 2),
            "inter_frame_boundary_var": round(mean_boundary_var, 6),
        }
        print(f"Result: FPS={result['fps']}, Mean Latency={result['mean_latency_ms']}ms, P95={result['p95_latency_ms']}ms, Peak VRAM={result['peak_vram_mb']}MB, Boundary Var={result['inter_frame_boundary_var']}")
        return result

    def run_all(
        self,
        swappers: Sequence[str] | None = None,
        maskers: Sequence[str] | None = None,
        output_json: str = "benchmark_results.json",
    ) -> list[dict[str, Any]]:
        swappers = swappers or [
            "inswapper_128",
            "inswapper_128_fp16",
            "hyperswap_256",
            "hififace_256",
        ]
        maskers = maskers or [
            "none",
            "dfl_xseg_v2",
            "bisenet_face_parser",
            "face_occluder_v3",
            "sam2_tracked",
        ]

        print("=" * 80)
        print("ROOP-ULTIMATE AUTOMATED BENCHMARK SUITE")
        print(f"Target Video: {self.video_path}")
        print(f"Evaluation Slice: {self.slice_frames} frames (Warm-up: {self.warmup_frames})")
        print(f"Swapper Models: {swappers}")
        print(f"Masking Engines: {maskers}")
        print("=" * 80)

        frames = self.load_frame_slice()
        _, ref_latent = self.get_reference_face()

        results = []
        for swp in swappers:
            for msk in maskers:
                res = self.run_permutation(swp, msk, frames, ref_latent)
                results.append(res)

        # Export JSON
        export_payload = {
            "video_metadata": self.video_info,
            "device": torch.cuda.get_device_name(self.device_id) if torch.cuda.is_available() else "CPU",
            "slice_frames": self.slice_frames,
            "warmup_frames": self.warmup_frames,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2)
        print(f"\n[Export] Results saved to {output_json}")

        # Print Markdown table
        self.print_markdown_table(results)
        return results

    @staticmethod
    def print_markdown_table(results: list[dict[str, Any]]) -> str:
        headers = [
            "Swapper",
            "Mask Engine",
            "Status",
            "FPS",
            "Mean Lat (ms)",
            "P95 Lat (ms)",
            "P99 Lat (ms)",
            "Peak VRAM (MB)",
            "Boundary Var (SSIM)",
        ]
        col_lens = [len(h) for h in headers]
        rows = []
        for r in results:
            row = [
                str(r["swapper"]),
                str(r["masker"]),
                str(r["status"]),
                f"{r['fps']:.1f}",
                f"{r['mean_latency_ms']:.1f}",
                f"{r['p95_latency_ms']:.1f}",
                f"{r['p99_latency_ms']:.1f}",
                f"{r['peak_vram_mb']:.1f}",
                f"{r['inter_frame_boundary_var']:.6f}",
            ]
            for idx, val in enumerate(row):
                col_lens[idx] = max(col_lens[idx], len(val))
            rows.append(row)

        header_line = "| " + " | ".join(h.ljust(col_lens[i]) for i, h in enumerate(headers)) + " |"
        sep_line = "| " + " | ".join("-" * col_lens[i] for i in range(len(headers))) + " |"
        data_lines = ["| " + " | ".join(val.ljust(col_lens[i]) for i, val in enumerate(row)) + " |" for row in rows]

        md = "\n".join([header_line, sep_line] + data_lines)
        print("\n### Benchmark Summary Table\n")
        print(md)
        print("\n")
        return md


def main() -> None:
    parser = argparse.ArgumentParser(description="roop-ultimate Automated Benchmarking Harness")
    parser.add_argument("--video", default=r"D:\k1.mp4", help="Path to target benchmark video")
    parser.add_argument("--frames", type=int, default=300, help="Continuous slice frame count")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup frames to exclude")
    parser.add_argument("--output", default="benchmark_results.json", help="JSON output path")
    args = parser.parse_args()

    suite = BenchmarkSuite(video_path=args.video, slice_frames=args.frames, warmup_frames=args.warmup)
    suite.run_all(output_json=args.output)


if __name__ == "__main__":
    main()
