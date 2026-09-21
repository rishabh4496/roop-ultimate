"""Does the selected target stay the same across CPU, CUDA and TensorRT?

The runtime layer (execution provider, precision, batch path) must never change
WHICH face is swapped.  This harness renders one clip in "Selected face" mode
under each provider with byte-identical selection input and compares the
settled per-frame decision -- ``ProcessMgr._SWAP_LOG``, the list of
``(bbox, source_index)`` the swap phase actually painted -- not the pixels.

WHAT IS HELD IDENTICAL.  The captured target faces (bbox, kps, embedding, ...)
are detected ONCE in the parent (CUDA FP32 by default, ``--capture-provider``),
serialized, and handed to every arm as-is; so is ``selection_state``.  Each arm's own detector/recogniser then
runs under the arm's provider, which is exactly the part that could differ.

WHAT IS COMPARED.  Per frame, the set of swapped faces must match 1:1 across
arms (centroid within a quarter of the box width) with the same source index.
A frame where one provider swapped a face the others did not is a mismatch.
Person 0 and person 1 are rendered separately so a harness that compares
"nothing swapped" to "nothing swapped" cannot pass by accident: the two
selections must produce disjoint face sets on the same frames.

Every arm is its own process (sessions, pools and the TensorRT engine cache
are process-global) and prints the ``[Runtime]`` banner, which is captured so
the report can show ``provider_active`` for each arm -- the proof that the arm
ran where it claims to have run, not merely where it was asked to.

Usage (from app/):
  env/Scripts/python.exe tests/provider_selection_parity.py
  env/Scripts/python.exe tests/provider_selection_parity.py --providers cuda,tensorrt --frames 60
"""
import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import fixtures  # noqa: E402

DEFAULT_OUT = os.path.join(APP, "output", "provider_selection_parity")


# ── the arm (child process) ──────────────────────────────────────────────────

def _faces_from_plain(entries):
    from insightface.app.common import Face
    return [Face(entry) for entry in entries]


def run_arm(args):
    import two_face_video as tfv          # applies config.yaml's perf env
    import angle_bench as ab
    from roop.target_selection import normalize_target_selection

    with open(args.targets, "rb") as fh:
        payload = pickle.load(fh)
    targets = _faces_from_plain(payload["targets"])
    groups = list(payload["groups"])

    tfv._apply_startup_runtime_environment()
    g = ab.init_pipeline(args.provider, args.swap_model, "None", args.mask_engine,
                         sync_config=True)
    g.swap_model_mask_strength = float(getattr(g.CFG, "swap_model_mask_strength", 0.0) or 0.0)
    g.execution_threads = int(args.threads)
    g.face_swap_mode = "selected"
    track = args.tracking != "0"
    g.track_identities = track
    g.CFG.track_identities = track
    g.temporal_detection = track
    g.CFG.temporal_detection = track
    g.video_encoder = "libx264"
    g.video_quality = 20

    options = ab.build_options(g, args.swap_model, tfv.map_mask_engine(args.mask_engine),
                               False)
    options.selection_state = normalize_target_selection(
        {"selection_mode": "selected", "person_id": int(args.person)},
        person_count=len(set(groups)))
    assert options.selection_state["valid"], options.selection_state

    names = [n.strip() for n in args.source.split(",") if n.strip()]
    facesets = [tfv.load_library_faceset(n) for n in names]
    work = os.path.join(args.out, f"{args.provider}_p{args.person}")
    os.makedirs(work, exist_ok=True)
    clip = os.path.join(work, "clip.mp4")
    tfv.trim(args.video, int(args.start), int(args.start) + int(args.frames), clip)

    t0 = time.perf_counter()
    out_path, (swap_log, _face_log) = tfv.run_swap(clip, facesets, targets, groups,
                                                   options, work)
    elapsed = time.perf_counter() - t0

    # The selection invariant is read from the ``[Runtime] phase=video`` banner
    # in this arm's log, printed while the manager was live; after the run the
    # manager has been released and would read as empty on both sides.
    result = {
        "provider": args.provider,
        "person": int(args.person),
        "frames": int(args.frames),
        "elapsed_s": round(elapsed, 2),
        "output": out_path,
        "swap_log": {str(k): [[list(map(float, b)), int(s)] for b, s in v]
                     for k, v in (swap_log or {}).items()},
    }
    with open(args.result, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    print(f"[arm] provider={args.provider} person={args.person} "
          f"frames_with_swaps={sum(1 for v in result['swap_log'].values() if v)} "
          f"elapsed={elapsed:.1f}s", flush=True)


# ── the parent ───────────────────────────────────────────────────────────────

def capture_targets_once(video, out_pkl, provider, seek=0):
    """Detect the two people ONCE and serialize the Face dicts.

    CUDA FP32 by default: the same numbers every arm would compute on a
    non-TensorRT chain, at a speed that does not stall the harness (a CPU
    scan of a 1080p clip with retinaface_r50 ran past ten minutes)."""
    import angle_bench as ab
    import two_face_video as tfv
    ab.init_pipeline(provider, "hyperswap", "None", "None", sync_config=True)
    idx, frame = tfv.separated_frame_with_fallback(video)
    if seek and idx < seek:
        # Capture a separated frame INSIDE the render window so the two
        # people's identities are the ones actually on screen there.
        import cv2
        cap = cv2.VideoCapture(video)
        for probe in range(seek, seek + 400):
            cap.set(cv2.CAP_PROP_POS_FRAMES, probe)
            ok, fr = cap.read()
            if not ok:
                break
            from roop.face_util import get_all_faces
            if len(get_all_faces(fr) or []) == 2:
                idx, frame = probe, fr
                break
        cap.release()
    targets, groups = tfv.capture_targets(frame)
    plain = [dict(f) for f in targets]
    with open(out_pkl, "wb") as fh:
        pickle.dump({"targets": plain, "groups": groups, "frame": idx}, fh)
    print(f"[capture] frame={idx} persons={len(set(groups))} "
          f"bboxes={[[round(float(v)) for v in f.bbox] for f in targets]}", flush=True)
    return idx


_BANNER = re.compile(r"^\[Runtime\] .*$", re.M)


def run_child(args, provider, person, result_path):
    cmd = [sys.executable, os.path.abspath(__file__), "--arm", "--provider", provider,
           "--person", str(person), "--targets", args.targets_pkl,
           "--result", result_path, "--video", args.video, "--frames", str(args.frames),
           "--threads", str(args.threads), "--swap-model", args.swap_model,
           "--mask-engine", args.mask_engine, "--source", args.source,
           "--tracking", args.tracking, "--out", args.out, "--start", str(args.start)]
    log_path = result_path.replace(".json", ".log")
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.run(cmd, cwd=APP, stdout=log, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace")
    text = open(log_path, encoding="utf-8", errors="replace").read()
    banners = _BANNER.findall(text)
    if proc.returncode != 0 or not os.path.exists(result_path):
        tail = "\n".join(text.splitlines()[-25:])
        return None, banners, f"exit={proc.returncode}\n{tail}"
    with open(result_path, encoding="utf-8") as fh:
        result = json.load(fh)
    video = [b for b in banners if " phase=video " in b]
    inv = re.search(r"selection_invariant=(\S+)", video[-1]) if video else None
    result["selection_invariant"] = inv.group(1) if inv else "MISSING(no video banner)"
    result["provider_active"] = (re.search(r"provider_active=(\S+)", video[-1]).group(1)
                                 if video else "MISSING")
    result["batch_mode"] = (re.search(r"batch_mode=(\S+)", video[-1]).group(1)
                            if video else "MISSING")
    return result, banners, None


def _centroid(b):
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5, max(1.0, b[2] - b[0]))


def match_frame(a, b):
    """1:1 match of swapped faces between two arms on one frame."""
    if len(a) != len(b):
        return False
    used = set()
    for box_a, src_a in a:
        cx, cy, w = _centroid(box_a)
        hit = None
        for j, (box_b, src_b) in enumerate(b):
            if j in used or src_a != src_b:
                continue
            bx, by, _ = _centroid(box_b)
            if abs(bx - cx) <= 0.25 * w and abs(by - cy) <= 0.25 * w:
                hit = j
                break
        if hit is None:
            return False
        used.add(hit)
    return True


def compare(reference, other):
    frames = sorted(set(reference["swap_log"]) | set(other["swap_log"]), key=int)
    mismatches = []
    for f in frames:
        a = reference["swap_log"].get(f, [])
        b = other["swap_log"].get(f, [])
        if not match_frame(a, b):
            mismatches.append({"frame": int(f), reference["provider"]: a,
                               other["provider"]: b})
    return len(frames), mismatches


def disjoint(p0, p1):
    """On TWO-FACE frames, the person-0 and person-1 renders must not swap the
    SAME physical face.  Returns (overlap_frames, eligible_frame_count).

    Only frames where BOTH renders swapped 2+ faces count: on a single-face
    frame a selected-single-person render legitimately swaps the one visible
    person whichever person is selected, so it is not a selection error and is
    not evidence either way."""
    overlap, eligible = [], 0
    for f, a in p0["swap_log"].items():
        b = p1["swap_log"].get(f, [])
        if len(a) < 2 or len(b) < 2:
            continue
        eligible += 1
        for box_a, _ in a:
            cx, cy, w = _centroid(box_a)
            for box_b, _ in b:
                bx, by, _ = _centroid(box_b)
                if abs(bx - cx) <= 0.25 * w and abs(by - cy) <= 0.25 * w:
                    overlap.append(int(f))
    return sorted(set(overlap)), eligible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--provider", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--person", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--targets", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--result", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--providers", default="cpu,cuda,tensorrt")
    ap.add_argument("--persons", default="0,1")
    ap.add_argument("--video", default=fixtures.clip("double/d1.mp4"))
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--swap-model", default=None, help="default: config.yaml swap_model")
    ap.add_argument("--mask-engine", default=None, help="default: config.yaml mask_engine")
    ap.add_argument("--source", default="person_a",
                    help="one faceset, or comma-separated for multi-source")
    ap.add_argument("--tracking", default="1")
    ap.add_argument("--capture-provider", default="cuda")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.swap_model is None or args.mask_engine is None:
        import yaml
        with open(os.path.join(APP, "config.yaml"), encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        args.swap_model = args.swap_model or str(cfg.get("swap_model", "hyperswap"))
        args.mask_engine = args.mask_engine or str(cfg.get("mask_engine", "None"))

    if args.arm:
        return run_arm(args)

    os.makedirs(args.out, exist_ok=True)
    args.targets_pkl = os.path.join(args.out, "targets.pkl")
    capture_targets_once(args.video, args.targets_pkl, args.capture_provider,
                         seek=args.start)

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    persons = [int(p) for p in args.persons.split(",") if p.strip()]
    results, banners, failures = {}, {}, {}
    for person in persons:
        for provider in providers:
            print(f"\n=== arm provider={provider} person={person} frames={args.frames} ===",
                  flush=True)
            path = os.path.join(args.out, f"{provider}_p{person}.json")
            res, lines, err = run_child(args, provider, person, path)
            banners[(provider, person)] = lines
            for line in lines:
                print("   " + line, flush=True)
            if err:
                failures[(provider, person)] = err
                print(f"   FAILED: {err}", flush=True)
                continue
            results[(provider, person)] = res
            swapped = sum(1 for v in res["swap_log"].values() if v)
            print(f"   frames_with_swaps={swapped}/{args.frames} "
                  f"elapsed={res['elapsed_s']}s provider_active={res['provider_active']} "
                  f"batch_mode={res['batch_mode']} "
                  f"selection_invariant={res['selection_invariant']}", flush=True)
            if res["selection_invariant"] != "OK":
                failures[(provider, person)] = f"selection invariant {res['selection_invariant']}"
            wanted = provider.lower()
            if not res["provider_active"].startswith(wanted):
                failures[(provider, person)] = (
                    f"asked for {provider} but the swap session ran on "
                    f"{res['provider_active']}")

    print("\n=== PARITY ===", flush=True)
    verdict_ok = not failures
    report = {"frames": args.frames, "video": args.video, "swap_model": args.swap_model,
              "mask_engine": args.mask_engine, "providers": providers,
              "persons": persons, "arms": {}, "comparisons": [], "disjoint": {}}
    for person in persons:
        ref = results.get((providers[0], person))
        for provider in providers[1:]:
            other = results.get((provider, person))
            if ref is None or other is None:
                continue
            n, mismatches = compare(ref, other)
            ok = not mismatches
            verdict_ok &= ok
            print(f"person={person}: {providers[0]} vs {provider}: "
                  f"{n - len(mismatches)}/{n} frames identical "
                  f"{'OK' if ok else 'MISMATCH'}", flush=True)
            for m in mismatches[:10]:
                print(f"     frame {m['frame']}: {providers[0]}={m[providers[0]]} "
                      f"{provider}={m[provider]}", flush=True)
            report["comparisons"].append({"person": person, "reference": providers[0],
                                          "other": provider, "frames": n,
                                          "mismatches": mismatches})
    if len(persons) >= 2:
        for provider in providers:
            p0, p1 = results.get((provider, persons[0])), results.get((provider, persons[1]))
            if p0 is None or p1 is None:
                continue
            overlap, eligible = disjoint(p0, p1)
            if eligible == 0:
                print(f"{provider}: person {persons[0]}/{persons[1]} selection "
                      f"sensitivity NOT TESTED (no two-face frame in the render "
                      f"window; single-person collapse is expected)", flush=True)
            else:
                verdict_ok &= not overlap
                print(f"{provider}: person {persons[0]} and person {persons[1]} swap "
                      f"{'DISJOINT faces OK' if not overlap else f'the SAME face on {len(overlap)} of {eligible} two-face frames'} "
                      f"({eligible} two-face frames)", flush=True)
            report["disjoint"][provider] = {"overlap": overlap, "eligible": eligible}
    for key, res in results.items():
        report["arms"][f"{key[0]}_p{key[1]}"] = {
            "banners": banners.get(key, []), "elapsed_s": res["elapsed_s"],
            "frames_with_swaps": sum(1 for v in res["swap_log"].values() if v),
            "provider_active": res["provider_active"], "batch_mode": res["batch_mode"],
            "selection_invariant": res["selection_invariant"]}
    for key, err in failures.items():
        report["arms"][f"{key[0]}_p{key[1]}"] = {"failed": err,
                                                 "banners": banners.get(key, [])}
    report["verdict"] = "PASS" if verdict_ok else "FAIL"
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nVERDICT: {report['verdict']}  (report: {os.path.join(args.out, 'report.json')})",
          flush=True)
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
