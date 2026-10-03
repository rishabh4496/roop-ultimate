"""Stage 5 performance / fidelity regression suite (real production renders).

Every end-to-end case drives the SAME path the Start button uses --
``core.batch_process_with_options`` through ``roop.benchmark.regression`` -- never
``BenchmarkRunner`` (``process_frame`` x1 in preview mode; see
docs/development/REGRESSION_BENCHMARK.md). Three input classes: one image, a 1080p
clip and a 4K clip, each rendered TWICE with an identical config:

* arm A is the baseline output. It is also kept as a golden file under
  ``<repo>/.roop/perf_regression/<gpu>_<hash>/`` the first time a key is seen, and every
  later run compares against that file too, so drift across commits is caught and not
  just the render's own non-determinism.
* arm B is the candidate. B vs A is the null control: the pixel noise floor between two
  renders of one unchanged config is 0.7142/255 mean, 22/255 max (AGENTS.md), and the
  contract numbers have to hold ON TOP of it.

Contract (the task's numbers) -- PSNR > 40 dB, SSIM >= 0.995, LPIPS <= 0.005 -- is
gated on the mean over the compared frames, for the whole frame and for the face crop
(a whole-frame number is dominated by untouched background and cannot fail on its own).
The worst frame is reported, not gated. Throughput is LOGGED and compared to the stored
baseline with a warning, never a failure (AGENTS.md: FPS never fails a regression run),
and nothing under 600 frames is an acceptance number.

A render that swaps nothing is a valid picture and would score a perfect fidelity
against itself, so each case also demands that the OUTPUT's identity moved to the source
person (cosine to the source beats cosine to the original target) on a minimum fraction
of face frames: the 4bd577d / Stage 13 failure class. It is an identity test and not a
pixel-difference one on purpose: a large face swapped by a similar-looking source moved
only 3.2/255 mean inside its crop (the bench's 4.0 "changed" threshold read it as
unswapped) while the swap log, which counts intent, cannot see stills at all.

Audio-video timing goes through ``scripts/verify_roop_keep.validate_timestamp_integrity``:
constant frame rate, source audio stream-copied, bounded A/V duration drift. The 4K
fixtures have no audio track, so a synthetic AAC track is muxed into the cut: an
untested mux proves nothing.

Environment knobs (all optional)::

    ROOP_KEEP_DIR            media folder (default <PINOKIO_HOME>/roop-keep)
    ROOP_PERF_FRAMES_1080    frames in the 1080p clip   (default 90)
    ROOP_PERF_FRAMES_4K      frames in the 4K clip      (default 48)
    ROOP_PERF_UPDATE_BASELINE=1   re-record the goldens
    ROOP_PERF_STAB_CHUNK_MB  stabilizer chunk budget pinned for the run (default 1024)
    ROOP_SOAK_RENDERS        renders in the leak test   (default 4)
    ROOP_SOAK_SECONDS        keep rendering for this long instead (hours = 3600*h)
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app"
for _p in (str(APP), str(APP / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- the contract -----------------------------------------------------------
PSNR_MIN = 40.0
SSIM_MIN = 0.995
LPIPS_MAX = 0.005
# Minimum share of face frames whose output identity is closer to the SOURCE person than
# to the original target. The regression bench refuses to record a baseline under 50%
# coverage; a render below this has not run the swapper.
MIN_SWAPPED_FRACTION = 0.5
FPS_WARN_DROP = 0.15

STRIDE_WHOLE_FRAME_LPIPS = 4          # LPIPS on every Nth whole frame (4K is not free)
REPORT: dict = {}
PENDING_GOLDEN: dict = {}


# ---------------------------------------------------------------------------
# fidelity metrics (pure, unit-tested below)
# ---------------------------------------------------------------------------
@dataclass
class Fidelity:
    scope: str
    frames: int
    psnr_mean: float
    psnr_min: float
    ssim_mean: float
    ssim_min: float
    lpips_mean: Optional[float]
    lpips_max: Optional[float]

    def failures(self) -> List[str]:
        out = []
        if not self.psnr_mean > PSNR_MIN:
            out.append(f"PSNR mean {self.psnr_mean:.2f} dB is not > {PSNR_MIN}")
        if not self.ssim_mean >= SSIM_MIN:
            out.append(f"SSIM mean {self.ssim_mean:.5f} is not >= {SSIM_MIN}")
        if self.lpips_mean is None:
            out.append("LPIPS was not measured (lpips is not importable)")
        elif not self.lpips_mean <= LPIPS_MAX:
            out.append(f"LPIPS mean {self.lpips_mean:.5f} is not <= {LPIPS_MAX}")
        return out


class LpipsMeter:
    """AlexNet LPIPS on the GPU (the package's standard configuration)."""

    _model = None

    @classmethod
    def available(cls) -> bool:
        return importlib.util.find_spec("lpips") is not None

    def __call__(self, a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
        import torch
        if LpipsMeter._model is None:
            import lpips
            device = "cuda" if torch.cuda.is_available() else "cpu"
            LpipsMeter._model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        model = LpipsMeter._model
        device = next(model.parameters()).device

        def prep(img):
            t = torch.from_numpy(np.ascontiguousarray(img[..., ::-1])).to(device)
            return t.permute(2, 0, 1).unsqueeze(0).float().div_(127.5).sub_(1.0)

        with torch.no_grad():
            return float(model(prep(a_bgr), prep(b_bgr)).item())


def _agg(scope: str, psnrs: Sequence[float], ssims: Sequence[float],
         lps: Sequence[float]) -> Fidelity:
    if not psnrs:
        raise AssertionError(f"{scope}: no frames were compared")
    return Fidelity(scope, len(psnrs), float(np.mean(psnrs)), float(np.min(psnrs)),
                    float(np.mean(ssims)), float(np.min(ssims)),
                    float(np.mean(lps)) if lps else None,
                    float(np.max(lps)) if lps else None)


def compare_frames(pairs: Iterator, boxes: Optional[list] = None,
                   lpips_stride: int = STRIDE_WHOLE_FRAME_LPIPS) -> dict:
    """Lock-step comparison of (src, ref, cand) frame triples.

    Returns ``{"frame": Fidelity, "face": Fidelity | None, "changed_frames": n,
    "face_frames": n}``. ``changed`` compares the candidate with the INPUT inside the
    face box (mean abs diff > CHANGED_MAD), i.e. it measures outcome, not intent.
    """
    from roop.benchmark.regression import CHANGED_MAD, expand_box, psnr, ssim
    meter = LpipsMeter() if LpipsMeter.available() else None
    fp, fs, fl = [], [], []
    cp, cs, cl = [], [], []
    face_frames = changed = 0
    for index, (src, ref, cand) in enumerate(pairs):
        fp.append(psnr(cand, ref))
        fs.append(ssim(cand, ref))
        if meter is not None and index % max(1, lpips_stride) == 0:
            fl.append(meter(cand, ref))
        frame_boxes = boxes[index] if boxes and index < len(boxes) else []
        if not frame_boxes:
            continue
        height, width = cand.shape[:2]
        frame_changed = counted = False
        for box in frame_boxes:
            x1, y1, x2, y2 = expand_box(box, width, height)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            counted = True
            c, r, s = cand[y1:y2, x1:x2], ref[y1:y2, x1:x2], src[y1:y2, x1:x2]
            cp.append(psnr(c, r))
            cs.append(ssim(c, r))
            if meter is not None:
                cl.append(meter(c, r))
            mad = float(np.mean(np.abs(c.astype(np.int16) - s.astype(np.int16))))
            frame_changed = frame_changed or mad > CHANGED_MAD
        face_frames += int(counted)
        changed += int(counted and frame_changed)
    return {"frame": _agg("frame", fp, fs, fl),
            "face": _agg("face", cp, cs, cl) if cp else None,
            "changed_frames": changed, "face_frames": face_frames}


# ---------------------------------------------------------------------------
# self-tests of the instrument: a gate that cannot fail proves nothing
# ---------------------------------------------------------------------------
def _textured(seed: int = 0, size=(256, 256)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    import cv2
    base = rng.integers(0, 255, (size[0] // 8, size[1] // 8, 3), dtype=np.uint8)
    return cv2.resize(base, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)


def _gate(src, ref, cand) -> Fidelity:
    return compare_frames(iter([(src, ref, cand)]))["frame"]


def test_identical_frames_clear_every_gate():
    img = _textured()
    f = _gate(img, img, img.copy())
    assert f.psnr_mean >= 99 and f.ssim_mean > 0.9999
    if LpipsMeter.available():
        assert f.lpips_mean < 1e-6
    assert f.failures() == [] or not LpipsMeter.available()


def test_render_noise_floor_clears_the_contract():
    """0.71/255 mean abs error (AGENTS.md noise floor) must not trip a gate."""
    img = _textured(1)
    rng = np.random.default_rng(2)
    noise = rng.laplace(0.0, 0.5, img.shape).round()           # mean |e| ~= 0.5-0.7
    cand = np.clip(img.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    mad = float(np.mean(np.abs(cand.astype(np.float32) - img.astype(np.float32))))
    assert 0.3 < mad < 1.2, mad
    f = _gate(img, img, cand)
    assert f.psnr_mean > PSNR_MIN and f.ssim_mean >= SSIM_MIN - 0.004, f


def test_each_gate_can_fail_independently():
    img = _textured(3)
    # a different face-sized change: blur destroys SSIM/LPIPS, mild noise only PSNR
    import cv2
    blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
    f = _gate(img, img, blurred)
    msgs = " | ".join(f.failures())
    assert "SSIM" in msgs and "PSNR" in msgs, msgs
    if LpipsMeter.available():
        assert "LPIPS" in msgs, msgs
    shifted = np.roll(img, 6, axis=1)
    assert _gate(img, img, shifted).failures()


def test_unswapped_candidate_is_reported_unchanged():
    """Identity render: perfect fidelity AND zero changed face frames (so the case
    an identity gate would fail it)."""
    img = _textured(4, (320, 320))
    box = [[40.0, 40.0, 240.0, 240.0]]
    out = compare_frames(iter([(img, img, img.copy())] * 3), boxes=[box] * 3)
    assert out["face_frames"] == 3 and out["changed_frames"] == 0


def test_swapped_candidate_is_reported_changed():
    img = _textured(5, (320, 320))
    cand = img.copy()
    cand[60:220, 60:220] = np.clip(cand[60:220, 60:220].astype(np.int16) + 40, 0, 255)
    out = compare_frames(iter([(img, img, cand)]), boxes=[[[40.0, 40.0, 240.0, 240.0]]])
    assert out["changed_frames"] == 1 and out["face"].psnr_mean < PSNR_MIN


# ---------------------------------------------------------------------------
# A/V timing: scripts/verify_roop_keep.py is the single implementation
# ---------------------------------------------------------------------------
def _verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_roop_keep", REPO / "scripts" / "verify_roop_keep.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_roop_keep"] = module
    spec.loader.exec_module(module)
    return module


def assert_av_timing(source: Path, output: Path, expect_frames: int) -> dict:
    """Constant frame rate + audio mux against the source's own spec."""
    v = _verifier()
    verdict = v.validate_timestamp_integrity(source, output)
    assert verdict["verdict"] == "pass", verdict
    src, out = verdict["source"], verdict["output"]
    assert out["constant_frame_rate"], out
    assert abs(out["fps"] - src["fps"]) < 0.01, (src["fps"], out["fps"])
    assert out["frames"] == expect_frames, (out["frames"], expect_frames)
    assert src["audio_codec"], "fixture has no audio: the mux check would be vacuous"
    assert out["audio_codec"] == src["audio_codec"]
    assert out["audio_sample_rate"] == src["audio_sample_rate"]
    expected = expect_frames / src["fps"]
    assert abs(out["video_duration_seconds"] - expected) <= 2.0 / src["fps"], (
        out["video_duration_seconds"], expected)
    return verdict


def test_verifier_rejects_vfr_and_missing_audio(tmp_path):
    """The A/V checker must be able to fail: a VFR clip and a silent clip."""
    from roop.ffmpeg_path import ffmpeg_binary
    ff = ffmpeg_binary()
    good = tmp_path / "good.mp4"
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=160x120:r=25:d=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:d=2",
                    "-c:v", "libx264", "-c:a", "aac", "-shortest", str(good)], check=True)
    assert_av_timing(good, good, expect_frames=50)
    silent = tmp_path / "silent.mp4"
    subprocess.run([ff, "-y", "-v", "error", "-i", str(good), "-an", "-c:v", "copy", str(silent)],
                   check=True)
    with pytest.raises(AssertionError):
        assert_av_timing(good, silent, expect_frames=50)    # source audio was dropped
    with pytest.raises(AssertionError):
        assert_av_timing(good, good, expect_frames=49)      # wrong frame count


# ---------------------------------------------------------------------------
# end-to-end: real production renders
# ---------------------------------------------------------------------------
def media_dir() -> Path:
    env = os.environ.get("ROOP_KEEP_DIR")
    return Path(env) if env else REPO.parents[1] / "roop-keep"


def _ff() -> str:
    from roop.ffmpeg_path import ffmpeg_binary
    return ffmpeg_binary()


def cut_with_audio(src: Path, dst: Path, frames: int) -> Path:
    """First `frames` frames, lossless video, AUDIO kept (synthesised if absent).

    A synthetic 48 kHz AAC track is added when the source has none so the audio mux
    is genuinely exercised on every case.
    """
    from roop.ffmpeg_path import ffprobe_binary
    has_audio = bool(subprocess.run(
        [ffprobe_binary(), "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=codec_name", "-of", "csv=p=0", str(src)],
        capture_output=True, text=True).stdout.strip())
    fps_text = subprocess.run(
        [ffprobe_binary(), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate", "-of", "csv=p=0", str(src)],
        capture_output=True, text=True).stdout.strip()
    num, den = (int(x) for x in fps_text.split("/"))
    seconds = frames * den / num
    cmd = [_ff(), "-y", "-v", "error", "-i", str(src)]
    if not has_audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    cmd += ["-map", "0:v:0", "-map", "0:a:0" if has_audio else "1:a:0",
            "-frames:v", str(frames), "-t", f"{seconds:.6f}",
            "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "copy" if has_audio else "aac", str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def _first_frame_with_face(clip: str, target_index: int) -> np.ndarray:
    from roop.benchmark.regression import FrameReader
    frame = None
    for i, frame in enumerate(FrameReader(clip, target_index + 1)):
        pass
    return frame.copy()


@dataclass
class Case:
    name: str
    clip: Path                 # the input handed to the renderer
    frames: int
    is_image: bool = False
    av_source: Optional[Path] = None   # the clip whose audio the render must carry


class Rig:
    """One initialised pipeline for the whole module (model load + engine build once)."""

    def __init__(self, tmp: Path):
        from roop.benchmark import regression as rg
        self.rg = rg
        self.tmp = tmp
        self.log = lambda m: print(m, flush=True)
        self.source = rg.default_source()
        self.setup = rg.prepare_pipeline(20, self.source, self.log)   # --threads 20
        # A render is NOT a pure function of its config: the stabilizer's block size is
        # derived from FREE RAM at render start (`_default_stab_chunk_mb`). Measured
        # 2026-10-03, 1080p/90f, same config: 7.0 GB free -> 12-frame blocks, 7.8 GB free
        # -> 16-frame blocks, and the two outputs differ by face SSIM 0.971 / LPIPS 0.030,
        # far outside this suite's own gates. The variable "overrides this exactly", so pin
        # it, or the null control measures the machine's memory pressure instead of the code.
        self._stab_prior = os.environ.get("ROOP_STAB_CHUNK_MB")
        os.environ["ROOP_STAB_CHUNK_MB"] = os.environ.get("ROOP_PERF_STAB_CHUNK_MB", "1024")

    def close(self):
        if self._stab_prior is None:
            os.environ.pop("ROOP_STAB_CHUNK_MB", None)
        else:
            os.environ["ROOP_STAB_CHUNK_MB"] = self._stab_prior

    def render(self, clip: Path, target, out_dir: Path):
        t0 = time.perf_counter()
        out, swaplog = self.rg.render(self.setup, str(clip), target, str(out_dir))
        elapsed = time.perf_counter() - t0
        assert out and os.path.isfile(out), f"render produced no output for {clip.name}"
        return Path(out), swaplog, elapsed


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    pytest.importorskip("torch")
    if not (media_dir() / "single" / "s3.mp4").is_file():
        pytest.skip(f"media folder {media_dir()} has no single/s3.mp4 (set ROOP_KEEP_DIR)")
    rig = Rig(tmp_path_factory.mktemp("perf"))
    yield rig
    rig.close()
    out = REPO / ".roop" / "perf_regression"
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_report.json").write_text(json.dumps(REPORT, indent=2, default=str),
                                          encoding="utf-8")
    print("\n[perf-regression] report ->", out / "last_report.json")


@pytest.fixture(scope="module")
def cases(rig) -> dict:
    n1080 = int(os.environ.get("ROOP_PERF_FRAMES_1080", "90"))
    n4k = int(os.environ.get("ROOP_PERF_FRAMES_4K", "48"))
    media = media_dir() / "single"
    work = rig.tmp / "clips"
    work.mkdir()
    c1080 = cut_with_audio(media / "s3.mp4", work / "hd1080.mp4", n1080)
    c4k = cut_with_audio(media / "s4.mp4", work / "uhd4k.mp4", n4k)
    target1080, idx = rig.rg.capture_target(str(c1080))
    target4k, _ = rig.rg.capture_target(str(c4k), window=n4k, stride=4)
    img = work / "still.png"
    import cv2
    ok, buf = cv2.imencode(".png", _first_frame_with_face(str(c1080), idx))
    assert ok
    buf.tofile(str(img))
    # warm-up pays the engine build and the model load for every resolution
    for clip, target in ((c1080, target1080), (c4k, target4k)):
        small = cut_with_audio(clip, work / f"warm_{clip.stem}.mp4", 12)
        rig.render(small, target, rig.tmp / "warm")
    return {
        "image": (Case("image_1080p", img, 1, is_image=True), target1080),
        "video_1080p": (Case("video_1080p", c1080, n1080, av_source=c1080), target1080),
        "video_4k": (Case("video_4k", c4k, n4k, av_source=c4k), target4k),
    }


def _frames_of(path: Path, limit: int, is_image: bool):
    import cv2
    if is_image:
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        yield img
        return
    from roop.benchmark.regression import FrameReader
    yield from FrameReader(str(path), limit)


def _baseline_dir(rig, case: Case) -> Path:
    key = f"{rig.setup.gpu_name or 'cpu'}_{rig.rg.signature_hash(rig.setup.signature)}"
    root = REPO / ".roop" / "perf_regression" / rig.rg.slug(key) / case.name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _faces(path: Path, case: Case) -> list:
    """Per frame: [(bbox, normed_embedding)] for every face the app's detector finds."""
    from roop.face_util import get_all_faces
    return [[([float(v) for v in f.bbox], np.asarray(f.normed_embedding, dtype=np.float32))
             for f in (get_all_faces(fr.copy()) or [])]
            for fr in _frames_of(path, case.frames, case.is_image)]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def identity_swapped_fraction(source_embedding, input_faces: list, output_faces: list) -> dict:
    """Outcome, not intent: did the output face become the SOURCE person?

    For each input face, find the output face at the same place (IoU >= 0.3) and test
    cos(out, source) > cos(out, original). Pure; unit-tested with synthetic vectors.
    """
    src = np.asarray(source_embedding, dtype=np.float32)
    src = src / (np.linalg.norm(src) or 1.0)
    frames = swapped = 0
    margins = []
    for ins, outs in zip(input_faces, output_faces):
        if not ins:
            continue
        frames += 1
        won = False
        for box, emb in ins:
            match = max(outs, key=lambda o: _iou(box, o[0]), default=None)
            if match is None or _iou(box, match[0]) < 0.3:
                continue
            out = match[1] / (np.linalg.norm(match[1]) or 1.0)
            orig = emb / (np.linalg.norm(emb) or 1.0)
            margin = float(out @ src - out @ orig)
            margins.append(margin)
            won = won or margin > 0.0
        swapped += int(won)
    return {"face_frames": frames, "swapped_frames": swapped,
            "swapped_fraction": swapped / max(1, frames),
            "mean_margin": float(np.mean(margins)) if margins else None}


def test_identity_gate_separates_swapped_from_untouched():
    rng = np.random.default_rng(7)
    unit = lambda v: v / np.linalg.norm(v)                     # noqa: E731
    source, original = unit(rng.normal(size=512)), unit(rng.normal(size=512))
    box = [100.0, 100.0, 300.0, 300.0]
    untouched = [[(box, original)]] * 4
    swapped = [[(box, unit(0.8 * source + 0.2 * original))]] * 4
    assert identity_swapped_fraction(source, untouched, untouched)["swapped_fraction"] == 0.0
    assert identity_swapped_fraction(source, untouched, swapped)["swapped_fraction"] == 1.0
    moved = [[([500.0, 500.0, 600.0, 600.0], source)]] * 4      # face not where the input's was
    assert identity_swapped_fraction(source, untouched, moved)["swapped_fraction"] == 0.0


def _run_case(rig, case: Case, target) -> dict:
    out_a, _, t_a = rig.render(case.clip, target, rig.tmp / f"{case.name}_A")
    out_b, _, t_b = rig.render(case.clip, target, rig.tmp / f"{case.name}_B")
    in_faces = _faces(case.clip, case)
    boxes = [[b for b, _ in frame] for frame in in_faces]

    result = {"case": case.name, "frames": case.frames,
              "fps_A": round(case.frames / t_a, 2), "fps_B": round(case.frames / t_b, 2),
              "seconds_A": round(t_a, 2), "seconds_B": round(t_b, 2),
              "acceptance_number": case.frames >= 600}

    # --- B vs A: the null-control (render determinism) ---------------------------
    def pairs(ref: Path, cand: Path):
        return zip(_frames_of(case.clip, case.frames, case.is_image),
                   _frames_of(ref, case.frames, case.is_image),
                   _frames_of(cand, case.frames, case.is_image))

    comparisons = {"B_vs_A": compare_frames(pairs(out_a, out_b), boxes)}

    # --- candidate vs the stored golden ----------------------------------------
    base = _baseline_dir(rig, case)
    golden = base / ("golden" + out_a.suffix)
    meta = base / "baseline.json"
    update = os.environ.get("ROOP_PERF_UPDATE_BASELINE") == "1"
    if update or not golden.is_file():
        # Recorded by the test only AFTER every assertion passed: a failing run must not
        # become the reference (the first 1080p run recorded an 89-frame golden).
        PENDING_GOLDEN[case.name] = (out_a, golden, meta,
                                     {"fps": result["fps_A"], "frames": case.frames,
                                      "recorded": time.strftime("%Y-%m-%d %H:%M:%S")})
        result["golden"] = "will be recorded if the case passes (nothing to compare against yet)"
    else:
        comparisons["B_vs_golden"] = compare_frames(pairs(golden, out_b), boxes)
        prior = json.loads(meta.read_text(encoding="utf-8")).get("fps")
        if prior and result["fps_B"] < prior * (1 - FPS_WARN_DROP):
            warnings.warn(f"{case.name}: {result['fps_B']} fps is >{FPS_WARN_DROP:.0%} below "
                          f"the recorded {prior} fps (not a failure; see AGENTS.md)")
        result["golden"] = "compared"

    for label, cmp in comparisons.items():
        result[label] = {k: (asdict(v) if isinstance(v, Fidelity) else v)
                         for k, v in cmp.items()}
    result["pixel_changed_fraction"] = (comparisons["B_vs_A"]["changed_frames"]
                                        / max(1, comparisons["B_vs_A"]["face_frames"]))
    identity = identity_swapped_fraction(
        rig.setup.faceset.faces[0].normed_embedding, in_faces, _faces(out_b, case))
    result["identity"] = identity
    result["swapped_fraction"] = identity["swapped_fraction"]
    if case.av_source is not None:
        result["av"] = assert_av_timing(case.av_source, out_b, case.frames)["verdict"]
    REPORT[case.name] = result
    print("[perf-regression]", json.dumps(result, default=str), flush=True)
    return result


def _assert_case(result: dict) -> None:
    problems = []
    for label in ("B_vs_A", "B_vs_golden"):
        if label in result and result[label]["frame"]["frames"] != result["frames"]:
            problems.append(f"{label}: compared {result[label]['frame']['frames']} frames, "
                            f"expected {result['frames']} (zip truncates to the shorter video; set "
                            f"ROOP_PERF_UPDATE_BASELINE=1 if the frame budget changed)")
    if result["swapped_fraction"] < MIN_SWAPPED_FRACTION:
        problems.append(f"only {result['swapped_fraction']:.0%} of face frames carry the source "
                        f"identity (< {MIN_SWAPPED_FRACTION:.0%}): the render did not swap "
                        f"({result['identity']})")
    for label in ("B_vs_A", "B_vs_golden"):
        if label not in result:
            continue
        for scope in ("frame", "face"):
            data = result[label].get(scope)
            if data is None:
                problems.append(f"{label}: no {scope} comparison (no face boxes found)")
                continue
            for msg in Fidelity(**data).failures():
                problems.append(f"{label}/{scope}: {msg} (worst frame PSNR {data['psnr_min']:.1f}, "
                                f"SSIM {data['ssim_min']:.4f})")
    assert not problems, f"{result['case']}:\n  " + "\n  ".join(problems)


@pytest.mark.gpu
@pytest.mark.perf
@pytest.mark.parametrize("name", ["image", "video_1080p", "video_4k"])
def test_end_to_end_fidelity_throughput_and_av_timing(rig, cases, name):
    case, target = cases[name]
    _assert_case(_run_case(rig, case, target))
    if case.name in PENDING_GOLDEN:                      # every assertion held
        out_a, golden, meta, info = PENDING_GOLDEN.pop(case.name)
        shutil.copyfile(out_a, golden)
        meta.write_text(json.dumps(info), encoding="utf-8")


# ---------------------------------------------------------------------------
# soak: no process crash, no per-render growth
# ---------------------------------------------------------------------------
def _quiescent_cuda_mb() -> float:
    import torch
    gc.collect()
    try:
        torch._C._cuda_clearCublasWorkspaces()        # +16 MB/render of cached workspaces
    except Exception:
        pass
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 2 ** 20


@pytest.mark.gpu
@pytest.mark.perf
def test_repeated_renders_do_not_leak_or_crash(rig, cases):
    """K renders of one clip. Quiescent CUDA allocation and RSS must plateau.

    Read AFTER each render returns with cuBLAS workspaces cleared -- a mid-render
    reading moves by whole in-flight plans and the per-render workspace constant
    would otherwise mimic a leak (docs: cublas-workspace-per-stream-leak).
    ROOP_SOAK_SECONDS=10800 turns this into the multi-hour run.
    """
    import psutil
    case, target = cases["video_1080p"]
    proc = psutil.Process()
    budget = float(os.environ.get("ROOP_SOAK_SECONDS", "0"))
    renders = int(os.environ.get("ROOP_SOAK_RENDERS", "4"))
    samples = []
    started = time.perf_counter()
    i = 0
    while True:
        out, _, secs = rig.render(case.clip, target, rig.tmp / f"soak_{i % 2}")
        assert out.stat().st_size > 0
        samples.append({"render": i, "cuda_mb": round(_quiescent_cuda_mb(), 1),
                        "rss_mb": round(proc.memory_info().rss / 2 ** 20, 1),
                        "fps": round(case.frames / secs, 2)})
        print("[soak]", samples[-1], flush=True)
        i += 1
        if (time.perf_counter() - started >= budget) if budget else (i >= renders):
            break
    REPORT["soak"] = samples
    assert len(samples) >= 3
    settled = samples[1:]                         # render 0 pays one-off allocations
    cuda_growth = settled[-1]["cuda_mb"] - settled[0]["cuda_mb"]
    rss_growth = settled[-1]["rss_mb"] - settled[0]["rss_mb"]
    per_render_cuda = cuda_growth / max(1, len(settled) - 1)
    per_render_rss = rss_growth / max(1, len(settled) - 1)
    assert per_render_cuda <= SOAK_CUDA_MB_PER_RENDER, (per_render_cuda, samples)
    assert per_render_rss <= SOAK_RSS_MB_PER_RENDER, (per_render_rss, samples)


# Measured 2026-10-03 (4070, 1080p x4): quiescent CUDA 9.4 MB flat, RSS 5125-5135 MB flat.
# A per-render cuBLAS-workspace strand is +16.25 MB, so 2 MB/render catches it.
SOAK_CUDA_MB_PER_RENDER = 2.0
SOAK_RSS_MB_PER_RENDER = 40.0
