"""Prove Viseme Mouth Custody EXECUTES inside a real render, on real footage, through the real request chain.

AGENTS.md: "before believing no effect, prove the code path executes ... count the stage, not the wrapper". The unit tests
pin the rules (test_mouth_custody*.py); this checks the production frame path actually CALLS them, with a real detector,
a real swap and the real audio of a real clip. Every arm goes through the request chain a client uses
(`api._normalize_mouth_custody` -> `api._apply_lipsync_settings`) and `core.batch_process_with_options`.

Arms, on the first N frames of one real clip (the user's config.yaml models, `config_sync`):

    off          mouth_custody_mode off                                  (the null: nothing lip-sync may run)
    off2         the same again                                          (the pixel noise floor)
    restore      source_mouth_restore
    audio        audio_driven_sync, co-articulation 'redub_sync' (so the generator runs), the clip's own audio
    slip         audio, audio_offset_ms +500
    audio_other  audio_driven_sync on the same video with a DIFFERENT driving track (override noise)
    silent_audio audio_driven_sync on a copy of the clip with NO audio stream      -> must fall back
    silent_rest  source_mouth_restore on the same silent copy                       -> what the fallback must equal
    dub          audio_driven_sync on the silent copy + an override track (noise)   -> must lip-sync and deliver the track

What is counted (wrapped, so a flag that is read but never acted on shows up as 0): the `lipsync` telemetry op (the
column behind `lipsync_ms`), AudioFeatureCache.features_for_time (the audio times the mouth was driven from),
Lipsync_Wav2Lip.infer (the generator), mouth_custody.fallback_for. NOTE: this machine has no Wav2Lip weights on disk, so the
generator runs its built-in viseme path (session None); the proof is that the stage is reached and fed the right audio
time, not a claim about a trained network's output.

Run from app/, one render at a time, nothing else on the GPU:
    env/Scripts/python.exe tests/verify_stage17_render.py [--clip PATH] [--frames 30] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
sys.path.insert(0, APP)
sys.path.insert(0, HERE)
os.chdir(APP)

import numpy as np  # noqa: E402

from verify_stage11_render import NOISE_FLOOR_MEAN, Recorder, decode, region_diff  # noqa: E402


def _noise_wav(path: str, seconds: float, rate: int = 16000) -> str:
    rng = np.random.default_rng(17)
    pcm = (rng.standard_normal(int(seconds * rate)) * 6000).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def _has_audio(path: str) -> bool:
    from roop.ffmpeg_path import ffprobe_binary
    out = subprocess.run([ffprobe_binary(), "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    return bool(out.strip())


def _speech_moves_under_the_slip(clip: str, fps: float = 30.0, slip_s: float = 0.5) -> bool:
    """True when this clip's own audio has a mouth-open stretch that a 0.5 s slip moves onto a different openness.

    A slip can only change pixels where the audio changes: a window that is silent throughout (an intro) would make the
    slip arm identical to the un-slipped one for the right reason, and the proof would fail for the wrong one. Judged
    with the same cache and the same lookup the render uses (`acoustic_for_time`)."""
    from roop import mouth_custody
    from roop.lipsync_audio import build_mel_audio_cache
    from roop.ffmpeg_path import ffmpeg_binary
    wav = os.path.splitext(clip)[0] + ".probe.wav"
    subprocess.run([ffmpeg_binary(), "-y", "-v", "error", "-i", clip, "-ac", "1", "-ar", "16000", wav], check=True)
    cache = build_mel_audio_cache(wav, fps)
    if cache.viseme_openness is None or not len(cache.viseme_openness):
        return False
    moved = 0
    for i in range(int(len(cache.viseme_openness) / cache.mel_fps * fps)):
        t = i / fps
        here = cache.acoustic_for_time(mouth_custody.audio_time_s(t, 0))[1]
        slipped = cache.acoustic_for_time(mouth_custody.audio_time_s(t, slip_s * 1000))[1]
        moved += abs(here - slipped) > 0.2
    return moved >= 3


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=None)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from roop.benchmark import regression as rb
    log = lambda m: print(m, flush=True)  # noqa: E731
    work = args.out or tempfile.mkdtemp(prefix="stage17_")
    os.makedirs(work, exist_ok=True)

    media = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(APP))), "roop-keep")
    candidates = [args.clip] if args.clip else [os.path.join(media, n) for n in ("angle.mp4", "Weeds.mp4", "Love.mp4")]
    candidates = [c for c in candidates if c and os.path.isfile(c)]
    if not candidates:
        raise SystemExit("no clip found; pass --clip")
    setup = rb.prepare_pipeline(args.threads, rb.default_source(), log)
    g = setup.g

    clip = silent = target = None
    for cand in candidates:
        for start_s in ((args.start,) if args.start is not None else (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0)):
            trial = os.path.join(work, "clip_audio.mp4")
            # WITH audio this time: that is the point. Lossless video so the plate is the plate.
            subprocess.run([rb._ffmpeg(), "-y", "-v", "error", "-ss", str(start_s), "-i", cand, "-frames:v", str(args.frames),
                            "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                            "-t", str((args.frames + 2) / 30.0), "-c:a", "aac", trial], check=True)
            if not _has_audio(trial):
                log("[verify] %s @ %.0fs has no audio in the window" % (os.path.basename(cand), start_s))
                continue
            if not _speech_moves_under_the_slip(trial):
                log("[verify] %s @ %.0fs: no speech the slip can move (silent or constant): next window" % (os.path.basename(cand), start_s))
                continue
            try:
                target, _ = rb.capture_target(trial, window=args.frames)
            except RuntimeError:
                log("[verify] no face in %s @ %.0fs" % (os.path.basename(cand), start_s))
                continue
            clip = trial
            break
        if clip:
            break
    if not clip:
        raise SystemExit("no clip with both audio and a face found")
    silent = os.path.join(work, "clip_silent.mp4")
    subprocess.run([rb._ffmpeg(), "-y", "-v", "error", "-i", clip, "-c:v", "copy", "-an", silent], check=True)
    assert _has_audio(clip) and not _has_audio(silent)
    dub_track = _noise_wav(os.path.join(work, "dub.wav"), 4.0)
    plate = decode(clip, args.frames)
    H, W = plate[0].shape[:2]
    bx0, by0, bx1, by1 = [int(v) for v in target.bbox]
    face_box = (max(0, bx0), max(0, by0), min(W, bx1), min(H, by1))
    mouth_box = (face_box[0], int(by0 + 0.55 * (by1 - by0)), face_box[2], face_box[3])
    log("[verify] clip %s: %d frames %dx%d, audio yes; silent copy audio %s; face %s"
        % (os.path.basename(clip), len(plate), W, H, _has_audio(silent), [bx0, by0, bx1, by1]))

    # ── what is counted ────────────────────────────────────────────────────
    import api
    from roop import mouth_custody, stage_telemetry
    from roop.lipsync_audio import AudioFeatureCache
    from roop.processors import Lipsync_Wav2Lip as w2l

    count = {"op_lipsync_calls": 0, "feature_times": [], "infer": 0, "fallback_checks": [], "audio_time_args": []}
    orig_record = stage_telemetry.record_op
    orig_features = AudioFeatureCache.features_for_time
    orig_infer = w2l.Lipsync_Wav2Lip.infer
    orig_fallback = mouth_custody.fallback_for

    def record_wrapper(stage, seconds):
        if stage == "lipsync":
            count["op_lipsync_calls"] += 1
        return orig_record(stage, seconds)

    def features_wrapper(self, t_s, *a, **k):
        count["feature_times"].append(round(float(t_s), 6))
        return orig_features(self, t_s, *a, **k)

    def infer_wrapper(self, *a, **k):
        count["infer"] += 1
        return orig_infer(self, *a, **k)

    def fallback_wrapper(mode, audio_available):
        out = orig_fallback(mode, audio_available)
        count["fallback_checks"].append((mode, bool(audio_available), out))
        return out

    stage_telemetry.record_op = record_wrapper
    AudioFeatureCache.features_for_time = features_wrapper
    w2l.Lipsync_Wav2Lip.infer = infer_wrapper
    mouth_custody.fallback_for = fallback_wrapper

    arms = {}

    def run(arm, source_clip, **fields):
        """One render through the request chain a client uses."""
        for key in ("op_lipsync_calls", "infer"):
            count[key] = 0
        count["feature_times"], count["fallback_checks"] = [], []
        payload = {"lipsync_model": "wav2lip", "lipsync_coarticulation_mode": fields.pop("coart", "redub_sync"), **fields}
        err = api._normalize_mouth_custody(payload)
        if err is not None:
            raise SystemExit("request refused: %s" % (err.body,))
        api._apply_lipsync_settings(payload)
        g.restore_original_mouth = bool(payload["restore_original_mouth"])
        setup.options.restore_original_mouth = bool(payload["restore_original_mouth"])
        out, swaplog = rb.render(setup, source_clip, target, os.path.join(work, arm))
        swapped = sum(1 for v in swaplog.values() if v)
        arms[arm] = {
            "video": out, "swapped_frames": swapped, "mode": payload["mouth_custody_mode"],
            "op_lipsync_calls": count["op_lipsync_calls"], "infer_calls": count["infer"],
            "feature_times": sorted(count["feature_times"]), "fallback_checks": list(count["fallback_checks"]),
            "fallback_reason": getattr(g, "mouth_custody_fallback", ""),
            "delivered_has_audio": bool(out and _has_audio(out)),
        }
        log("[verify] %-12s mode=%-20s swapped %2d/%d lipsync_op=%-3d infer=%-3d features=%-3d fallback=%r audio_out=%s"
            % (arm, payload["mouth_custody_mode"], swapped, len(plate), count["op_lipsync_calls"], count["infer"],
               len(count["feature_times"]), arms[arm]["fallback_reason"][:40], arms[arm]["delivered_has_audio"]))
        return out

    try:
        run("off", clip, mouth_custody_mode="off")
        run("off2", clip, mouth_custody_mode="off")
        run("restore", clip, mouth_custody_mode="source_mouth_restore")
        run("audio", clip, mouth_custody_mode="audio_driven_sync")
        run("slip", clip, mouth_custody_mode="audio_driven_sync", audio_offset_ms=500)
        run("audio_other", clip, mouth_custody_mode="audio_driven_sync", override_audio_path=dub_track)
        run("silent_audio", silent, mouth_custody_mode="audio_driven_sync")
        run("silent_rest", silent, mouth_custody_mode="source_mouth_restore")
        run("dub", silent, mouth_custody_mode="audio_driven_sync", override_audio_path=dub_track)
    finally:
        stage_telemetry.record_op = orig_record
        AudioFeatureCache.features_for_time = orig_features
        w2l.Lipsync_Wav2Lip.infer = orig_infer
        mouth_custody.fallback_for = orig_fallback

    vids = {k: decode(v["video"], len(plate)) for k, v in arms.items() if v["video"]}

    def mean_diff(a, b, box):
        d = region_diff(vids[a], vids[b], box)
        return float(np.mean(d)) if d else float("nan")

    report = {
        "frames": len(plate), "noise_floor_mean": NOISE_FLOOR_MEAN,
        "arms": {k: {kk: vv for kk, vv in v.items() if kk not in ("video", "feature_times")} | {"feature_times_n": len(v["feature_times"])}
                 for k, v in arms.items()},
        "null_mouth_mean": mean_diff("off", "off2", mouth_box),
        "restore_vs_off_mouth_mean": mean_diff("off", "restore", mouth_box),
        "audio_vs_off_mouth_mean": mean_diff("off", "audio", mouth_box),
        "slip_vs_audio_mouth_mean": mean_diff("audio", "slip", mouth_box),
        "other_audio_vs_audio_mouth_mean": mean_diff("audio", "audio_other", mouth_box),
        "silent_fallback_vs_restore_mouth_mean": mean_diff("silent_rest", "silent_audio", mouth_box),
        "dub_vs_silent_restore_mouth_mean": mean_diff("silent_rest", "dub", mouth_box),
    }
    a, s = arms["audio"]["feature_times"], arms["slip"]["feature_times"]
    report["slip_time_shift"] = {
        "audio_times": [a[0], a[-1]] if a else None, "slip_times": [s[0], s[-1]] if s else None,
        "expected_slip": [round(max(0.0, t - 0.5), 6) for t in a] == s if a and s else None,
    }
    log("[verify] " + json.dumps(report, indent=2, default=float))
    json.dump(report, open(os.path.join(work, "report.json"), "w"), indent=2, default=float)

    # ── verdicts ──────────────────────────────────────────────────────────
    A = arms
    null = report["null_mouth_mean"]
    floor = max(2.0, 3 * null) if null == null else 2.0
    floor_dep = max(NOISE_FLOOR_MEAN * 2, null * 2)            # "changed", not "close": a dependence must clear the noise
    verdicts = [
        ("V1 off / restore: no lip-sync stage ran (op, generator, audio lookups all 0)",
         all(A[k]["op_lipsync_calls"] == 0 and A[k]["infer_calls"] == 0 and not A[k]["feature_times"] for k in ("off", "off2", "restore"))),
        ("V2 audio: the swap swapped, and the lipsync op, the generator and the audio lookups all ran",
         A["audio"]["swapped_frames"] > 0 and A["audio"]["op_lipsync_calls"] > 0 and A["audio"]["infer_calls"] > 0 and len(A["audio"]["feature_times"]) > 0),
        ("V3 audio changed the mouth region beyond the pixel noise floor",
         report["audio_vs_off_mouth_mean"] > max(NOISE_FLOOR_MEAN * 2, null * 2)),
        ("V4 slip +500 ms: every audio time is exactly max(0, t - 0.5) of the un-slipped arm's",
         bool(report["slip_time_shift"]["expected_slip"])),
        ("V4b the OUTPUT depends on the audio: a different driving track changes the mouth beyond noise",
         report["other_audio_vs_audio_mouth_mean"] > floor_dep),
        ("V4c the OUTPUT depends on the slip: +500 ms changes the mouth beyond noise",
         report["slip_vs_audio_mouth_mean"] > floor_dep),
        ("V5 silent video: the mode fell back (reason stated), no lipsync stage ran",
         bool(A["silent_audio"]["fallback_reason"]) and A["silent_audio"]["op_lipsync_calls"] == 0 and A["silent_audio"]["infer_calls"] == 0),
        ("V6 silent video: the fallback's pixels are source_mouth_restore's (within noise)",
         report["silent_fallback_vs_restore_mouth_mean"] <= floor),
        ("V7 dub: an override track on a silent video lip-synced and was delivered",
         A["dub"]["op_lipsync_calls"] > 0 and A["dub"]["infer_calls"] > 0 and A["dub"]["delivered_has_audio"] and not A["dub"]["fallback_reason"]),
        ("V8 the clip's own audio is delivered in the audio arm; the silent copy stays silent",
         A["audio"]["delivered_has_audio"] and not A["silent_rest"]["delivered_has_audio"]),
    ]
    bad = 0
    for name, ok in verdicts:
        log("[verify] %s  %s" % ("PASS" if ok else "FAIL", name))
        bad += 0 if ok else 1
    log("[verify] %s" % ("ALL PASS" if not bad else "%d FAILED" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
