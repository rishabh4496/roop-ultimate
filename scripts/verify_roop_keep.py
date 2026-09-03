#!/usr/bin/env python3
"""Production verification runner for the local roop-keep corpus.

This is intentionally an orchestrator, not another face-swap implementation.
Each video is rendered by ``app/tests/verify_roop_keep.py``, which invokes the
same RealSwap, tracker, masks, and UltraMax path as the application.  One
worker process is used per video: a CUDA OOM therefore dies with its process
and cannot poison the next video or leave a fragmented CUDA allocator behind.

Examples
--------
    python scripts/verify_roop_keep.py
    python scripts/verify_roop_keep.py --base-dir "G:\\pinokio\\roop-keep" --device cuda
    python scripts/verify_roop_keep.py --resume --save-strips

Outputs are written only beneath ``<base-dir>/output_verified`` unless an
explicit ``--output-dir`` is supplied.  Inputs are never modified.  A damaged
stream is retried through an ffmpeg recovery copy in that output directory;
if it still cannot render, its error is retained in the final report while the
remaining corpus continues.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_HARNESS = REPO_ROOT / "app" / "tests" / "verify_roop_keep.py"
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
DEFAULT_BASE = Path(os.environ.get("ROOP_KEEP_DIR", r"G:\pinokio\roop-keep"))


def video_jobs(base_dir: Path) -> Iterable[Tuple[str, Path]]:
    """Yield all input clips in deterministic order, never looking in output."""
    for kind in ("single", "double"):
        folder = base_dir / kind
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir(), key=lambda candidate: candidate.name.lower()):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                yield kind, path


def job_key(kind: str, path: Path) -> str:
    return "%s/%s" % (kind, path.name)


def locate_python() -> str:
    """Prefer the app venv so direct ``python scripts/...`` is reproducible."""
    candidates = [
        REPO_ROOT / "app" / "env" / "Scripts" / "python.exe",
        REPO_ROOT / "app" / "env" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def find_entry(report: Dict[str, Any], kind: str, name: str) -> Optional[Dict[str, Any]]:
    matches = [item for item in report.get("videos", [])
               if item.get("kind") == kind and item.get("name") == name]
    return matches[-1] if matches else None


def tail(path: Path, lines: int = 30) -> str:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-lines:]).strip()
    except OSError:
        return ""


def looks_like_cuda_oom(text: str) -> bool:
    text = text.lower()
    return any(token in text for token in (
        "cuda out of memory", "cuda error: out of memory", "cudnn_status_alloc_failed",
        "failed to allocate", "out of memory while", "gpu memory exhausted",
    ))


def looks_like_corrupt_stream(text: str) -> bool:
    text = text.lower()
    return any(token in text for token in (
        "invalid data", "moov atom", "error while decoding", "could not open video",
        "cannot open video", "corrupt", "truncated", "error reading frame",
        "error sending packet", "invalid nal", "end of file",
    ))


def run_worker(command: List[str], env: Dict[str, str], log_path: Path) -> Tuple[int, str]:
    """Run one isolated pipeline attempt and retain its full console log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: %s\n\n" % subprocess.list2cmdline(command))
        completed = subprocess.run(command, cwd=str(REPO_ROOT), env=env,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   check=False)
    return completed.returncode, tail(log_path)


def repair_stream(source: Path, recovery_root: Path, kind: str) -> Optional[Path]:
    """Attempt a non-destructive ffmpeg recovery into the verifier output tree."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    destination = recovery_root / kind / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A stream copy preserves the original encoded image data where possible;
    # it is much faster and avoids silently changing model-quality evidence.
    command = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-err_detect", "ignore_err",
               "-fflags", "+discardcorrupt", "-i", str(source), "-map", "0:v:0?",
               "-map", "0:a?", "-c", "copy", str(destination)]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=600, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode == 0 and destination.is_file() and destination.stat().st_size > 0:
        return destination
    return None


def compact(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, int)):
        return ("%%.%df" % digits) % value
    return str(value)


def criterion(entry: Dict[str, Any], key: str) -> Dict[str, Any]:
    return next((item for item in entry.get("criteria", []) if item.get("key") == key), {})


def metrics_for(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize raw evidence into the fields QA reviewers consume per clip."""
    angle = criterion(entry, "angle").get("detail", {})
    occlusion = criterion(entry, "occlusion")
    boundary = criterion(entry, "boundary").get("detail", {})
    identity = criterion(entry, "identity").get("detail", {})
    seconds = entry.get("seconds")
    frames = entry.get("frames_seen")
    fps = float(frames) / float(seconds) if frames and seconds and float(seconds) > 0 else None
    occlusion_samples = occlusion.get("samples", 0) or 0
    occlusion_failures = occlusion.get("failures", 0) or 0
    return {
        "average_processing_fps": round(fps, 3) if fps is not None else None,
        "face_detection_extreme": {
            "yaw_gt_45_faces": angle.get("yaw_gt_45_faces"),
            "inverted_roll_faces": angle.get("inverted_roll_faces"),
            "detected_faces": angle.get("strict_extreme_detected_faces"),
            "success_rate_pct": angle.get("strict_extreme_detection_success_pct"),
        },
        "occlusion_persistence": {
            "candidate_frames": occlusion_samples,
            "survived_frames": max(0, occlusion_samples - occlusion_failures),
            "dropped_frames": occlusion_failures,
            "method": "inferred occluder; advisory without composited ground truth",
        },
        "arcface_cosine_by_faceset": identity.get("cosine_by_faceset", {}),
        "boundary_artifacts": {
            "boundary_ring_ssim": boundary.get("boundary_ring_ssim"),
            "boundary_seam_delta": boundary.get("boundary_seam_delta"),
            "method": boundary.get("method"),
        },
    }


def markdown_report(report: Dict[str, Any]) -> str:
    lines = [
        "# roop-keep verification report",
        "",
        "| Video | FaceSet(s) | Status | Extreme detection | Occlusion survived | FPS | ArcFace cosine | Boundary SSIM | Seam delta |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for entry in report.get("videos", []):
        metrics = entry.get("test_metrics", {})
        extreme = metrics.get("face_detection_extreme", {})
        occ = metrics.get("occlusion_persistence", {})
        boundary = metrics.get("boundary_artifacts", {})
        identity = metrics.get("arcface_cosine_by_faceset", {})
        if entry.get("error"):
            status = "ERROR"
        elif any(item.get("verdict") == "fail" for item in entry.get("criteria", [])):
            status = "FAIL"
        else:
            status = "PASS"
        cosines = ", ".join("%s=%s" % (name, compact((value or {}).get("mean")))
                            for name, value in identity.items()) or "n/a"
        ssim = (boundary.get("boundary_ring_ssim") or {}).get("mean")
        seam = (boundary.get("boundary_seam_delta") or {}).get("mean")
        lines.append("| %s | %s | %s | %s%% | %s/%s | %s | %s | %s | %s |" % (
            entry.get("name", "?"), ", ".join(entry.get("facesets", [])), status,
            compact(extreme.get("success_rate_pct"), 2),
            compact(occ.get("survived_frames"), 0), compact(occ.get("candidate_frames"), 0),
            compact(metrics.get("average_processing_fps")), cosines,
            compact(ssim), compact(seam)))
        if entry.get("error"):
            lines.append("| ↳ error | %s |  |  |  |  |  |  |  |" %
                         entry["error"].replace("|", "\\|"))
    lines.extend([
        "",
        "Notes: detection success counts matched output faces among source faces with yaw >45° or roll ≥150°. "
        "Occlusion and boundary measurements are advisory diagnostics; no ground-truth mask is present in this corpus.",
        "",
    ])
    return "\n".join(lines)


def persist(report: Dict[str, Any], output_dir: Path) -> None:
    report["videos"] = sorted(report.get("videos", []),
                              key=lambda item: (item.get("kind", ""), item.get("name", "")))
    write_json(output_dir / "verification_report.json", report)
    (output_dir / "verification_report.md").write_text(markdown_report(report), encoding="utf-8")


def make_command(python: str, base: Path, output: Path, kind: str, name: str,
                 args: argparse.Namespace) -> List[str]:
    return [python, str(APP_HARNESS), "--base-dir", str(base), "--output-dir", str(output),
            "--kind", kind, "--only", name, "--execution-provider", args.device,
            "--swap-model", args.swap_model, "--enhancer", args.enhancer,
            "--stride", str(args.stride), "--capture-budget", str(args.capture_budget),
            "--resume"] + (["--save-strips"] if args.save_strips else [])


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated RealSwap + UltraMax roop-keep verification.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE,
                        help="corpus containing single/ and double/")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: <base-dir>/output_verified")
    parser.add_argument("--device", choices=["cuda", "tensorrt", "cpu"], default="cuda",
                        help="ONNX execution provider; cuda is the portable GPU default")
    parser.add_argument("--swap-model", default="realswap")
    parser.add_argument("--enhancer", default="UltraMax")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--capture-budget", type=float, default=90.0)
    parser.add_argument("--save-strips", action="store_true")
    parser.add_argument("--resume", action="store_true", help="keep completed video results")
    args = parser.parse_args()

    base = args.base_dir.expanduser().resolve()
    output = (args.output_dir or base / "output_verified").expanduser().resolve()
    if not APP_HARNESS.is_file():
        raise SystemExit("production verifier is missing: %s" % APP_HARNESS)
    if not base.is_dir():
        raise SystemExit("base dir does not exist: %s" % base)
    jobs = list(video_jobs(base))
    if not jobs:
        raise SystemExit("no videos found under %s/single or %s/double" % (base, base))

    report_path = output / "verification_report.json"
    previous = read_json(report_path) if args.resume else {}
    prior_by_key = {job_key(item.get("kind", ""), Path(item.get("name", ""))): item
                    for item in previous.get("videos", [])}
    report: Dict[str, Any] = {
        "started": previous.get("started") if args.resume else datetime.now(timezone.utc).isoformat(),
        "updated": datetime.now(timezone.utc).isoformat(),
        "base_dir": str(base),
        "output_dir": str(output),
        "config": {"device": args.device, "swap_model": args.swap_model,
                   "enhancer": args.enhancer, "stride": args.stride,
                   "capture_budget": args.capture_budget, "isolated_workers": True},
        "recovery_policy": {
            "cuda_oom": "retry once in a fresh worker with singleton GPU pools",
            "corrupt_stream": "retry a non-destructive ffmpeg recovery copy when available",
        },
        "videos": list(prior_by_key.values()),
    }

    python = locate_python()
    baseline_env = os.environ.copy()
    results = dict(prior_by_key)
    total = len(jobs)
    print("[verify] %d video(s), device=%s, output=%s" % (total, args.device, output), flush=True)

    for index, (kind, source) in enumerate(jobs, 1):
        key = job_key(kind, source)
        cached = results.get(key)
        if args.resume and cached and cached.get("criteria") and not cached.get("error"):
            print("[%d/%d] %s -- kept (--resume)" % (index, total, key), flush=True)
            continue

        print("[%d/%d] %s -- RealSwap + %s" % (index, total, key, args.enhancer), flush=True)
        command = make_command(python, base, output, kind, source.name, args)
        code, log_tail = run_worker(command, baseline_env,
                                    output / "logs" / ("%s__%s.attempt1.log" % (kind, source.stem)))
        worker_report = read_json(report_path)
        entry = find_entry(worker_report, kind, source.name)
        combined_error = "\n".join(filter(None, [str((entry or {}).get("error", "")), log_tail]))

        if looks_like_cuda_oom(combined_error):
            print("  CUDA OOM: retrying with singleton GPU pools in a fresh worker", flush=True)
            safe_env = baseline_env.copy()
            safe_env.update({"ROOP_TRT_POOL": "1", "ROOP_DETMASK_POOL": "1",
                             "ROOP_DETECTOR_POOL": "1", "ROOP_EXPR_POOL": "0",
                             "ROOP_BATCH_SWAP": "0", "ROOP_BATCH_SWAP_XFRAME": "0"})
            code, log_tail = run_worker(command, safe_env,
                                        output / "logs" / ("%s__%s.oom-retry.log" % (kind, source.stem)))
            worker_report = read_json(report_path)
            entry = find_entry(worker_report, kind, source.name)
            if entry:
                entry.setdefault("recovery", []).append("cuda_oom_retry_singleton_pools")

        combined_error = "\n".join(filter(None, [str((entry or {}).get("error", "")), log_tail]))
        if entry and entry.get("error") and looks_like_corrupt_stream(combined_error):
            recovered_root = output / "recovered_inputs"
            recovered = repair_stream(source, recovered_root, kind)
            if recovered:
                print("  corrupt stream: retrying repaired copy %s" % recovered.name, flush=True)
                recovered_command = make_command(python, recovered_root, output, kind,
                                                  recovered.name, args)
                code, log_tail = run_worker(
                    recovered_command, baseline_env,
                    output / "logs" / ("%s__%s.repaired-retry.log" % (kind, source.stem)))
                worker_report = read_json(report_path)
                entry = find_entry(worker_report, kind, source.name)
                if entry:
                    entry.setdefault("recovery", []).append("ffmpeg_discardcorrupt_copy_retry")
                    entry["recovered_input"] = str(recovered)
            else:
                entry.setdefault("recovery", []).append("ffmpeg_recovery_unavailable_or_failed")

        if not entry:
            entry = {"name": source.name, "kind": kind,
                     "facesets": ["mehak"] if kind == "single" else ["mehak", "misbah"],
                     "criteria": [],
                     "error": "worker exited %d without a report entry" % code,
                     "worker_log": str(output / "logs" / ("%s__%s.attempt1.log" % (kind, source.stem)))}
        entry["source"] = str(source)
        entry["test_metrics"] = metrics_for(entry)
        results[key] = entry
        report["videos"] = list(results.values())
        report["updated"] = datetime.now(timezone.utc).isoformat()
        persist(report, output)

    report["videos"] = list(results.values())
    report["finished"] = datetime.now(timezone.utc).isoformat()
    persist(report, output)
    errors = sum(bool(item.get("error")) for item in report["videos"])
    failures = sum(any(c.get("verdict") == "fail" for c in item.get("criteria", []))
                   for item in report["videos"])
    print("[verify] complete: %d video(s), %d execution error(s), %d quality failure(s)" %
          (len(report["videos"]), errors, failures), flush=True)
    print("  json: %s" % report_path)
    print("  markdown: %s" % (output / "verification_report.md"))
    return 1 if errors or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
