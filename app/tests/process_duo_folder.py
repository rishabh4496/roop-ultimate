"""Grade the duo/ clips with TWO facesets bound to two people.

Replaces a version that measured nothing about the thing it was named for. That
one called `sb.run_swap(...)` as `out_file, elapsed, face_log = ...` and then
never read `face_log` at all: it recorded fps, frame counts and six PNGs, and
the session notes it produced still claimed "100% swapped, zero identity
flipping". Neither half of that claim was computed anywhere. The bench was blind
to exactly the failure it was cited as clearing.

This drives `two_face_video.py` per clip rather than re-implementing grading,
because that tool already solved the two traps that make a two-faceset bench lie:

  * It grades from the DECISION (`ProcessMgr._SWAP_LOG`), which records what the
    pipeline actually pasted, at the composite. Re-detecting the OUTPUT and
    comparing embeddings suffers the shared-recognition-crop problem in exactly
    the way the pipeline does, so on contact frames it reports each person as
    the other whatever the swap did -- a correct fix scores as a failure and a
    broken run scores as fine.
  * It blanks the cosine columns where the output crop is itself contaminated,
    and carries `contam` so the blanks are counted rather than assumed. Note the
    consequence, measured in gradeability_survey.py: d1 is 0% gradeable for
    identity. Its WRONG-FACESET number is still exact, because that one comes
    from the decision and never from re-detection.

Settings come from config.yaml, not from this file's own opinions -- the version
this replaces hardcoded `detail_transfer_strength = 0.40` while production ran 0.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import fixtures
import yaml


def live_settings():
    """The stack the user actually renders with."""
    with open(os.path.join(APP, 'config.yaml'), 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    return {
        'swap_model': cfg.get('swap_model', 'realswap'),
        'enhancer': cfg.get('selected_enhancer', 'UltraMax'),
        'mask_engine': cfg.get('mask_engine', 'RealityUX'),
        'swap_model_mask': float(cfg.get('swap_model_mask_strength', 0.0) or 0.0),
        'stabilize_mask': '1' if cfg.get('stabilize_mask') else '0',
        'stabilize_mask_strength': float(cfg.get('stabilize_mask_strength', 0.5) or 0.5),
        'provider': cfg.get('provider', 'cuda'),
    }


_COVER = re.compile(r"box (\d+) from the left \(([^)]+)\): (\d+) frames, "
                    r"swapped (\d+) \(([\d.]+)%\), on/off transitions (\d+)")
_WRONG = re.compile(r"WRONG FACESET APPLIED on (\d+) of (\d+) swaps")
_OTHER = re.compile(r"re-measured as the other person on (\d+) of (\d+)")


def parse(out):
    people, cur = [], None
    for line in out.splitlines():
        m = _COVER.search(line)
        if m:
            cur = {'box': int(m.group(1)), 'name': m.group(2),
                   'frames': int(m.group(3)), 'swapped': int(m.group(4)),
                   'pct': float(m.group(5)), 'flips': int(m.group(6)),
                   'wrong': None, 'wrong_of': None, 'other': None, 'other_of': None}
            people.append(cur)
            continue
        if cur is None:
            continue
        m = _WRONG.search(line)
        if m:
            cur['wrong'], cur['wrong_of'] = int(m.group(1)), int(m.group(2))
        m = _OTHER.search(line)
        if m:
            cur['other'], cur['other_of'] = int(m.group(1)), int(m.group(2))
    return people


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="harjot,gargee")
    ap.add_argument("--in-dir", default=fixtures.clip_dir('duo'))
    ap.add_argument("--clips", default="", help="comma-separated stems, default all")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--end", type=int, default=0, help="0 = whole clip")
    ap.add_argument("--tag-prefix", default="duo")
    args = ap.parse_args()

    s = live_settings()
    print("=" * 88)
    print("DUO FOLDER - TWO FACESETS, GRADED FROM THE PIPELINE'S OWN DECISION")
    print("  sources     : " + args.sources)
    print("  swap_model  : {}    enhancer: {}    mask: {}".format(
        s['swap_model'], s['enhancer'], s['mask_engine']))
    print("  swap mask   : {}    threads: {}    provider: {}".format(
        s['swap_model_mask'], args.threads, s['provider']))
    print("=" * 88, flush=True)

    videos = sorted(glob.glob(os.path.join(args.in_dir, "*.mp4")))
    if args.clips.strip():
        want = {c.strip() for c in args.clips.split(",")}
        videos = [v for v in videos
                  if os.path.splitext(os.path.basename(v))[0] in want]

    logdir = os.path.join(APP, "output", "bench_two_face", "_logs")
    os.makedirs(logdir, exist_ok=True)

    results = []
    for i, v in enumerate(videos, 1):
        stem = os.path.splitext(os.path.basename(v))[0]
        print("\n" + "=" * 88)
        print("[{}/{}] {}".format(i, len(videos), stem))
        print("=" * 88, flush=True)
        cmd = [sys.executable, os.path.join(HERE, "two_face_video.py"),
               "--tag", "{}_{}".format(args.tag_prefix, stem),
               "--video", v,
               "--sources", args.sources,
               "--provider", s['provider'],
               "--swap-model", s['swap_model'],
               "--enhancer", s['enhancer'],
               "--mask-engine", s['mask_engine'],
               "--swap-model-mask-strength", str(s['swap_model_mask']),
               "--stabilize-mask", s['stabilize_mask'],
               "--stabilize-mask-strength", str(s['stabilize_mask_strength']),
               "--tracking", "1",
               "--threads", str(args.threads)]
        if args.end:
            cmd += ["--end", str(args.end)]

        # Streamed to a log rather than captured in memory, so a long roster can
        # be watched while it runs (this takes tens of minutes; a silent
        # capture_output makes the fps reporting the project asks for
        # impossible). Parsed back from the same file afterwards.
        log_path = os.path.join(logdir, "{}.log".format(stem))
        print("[duo] log: {}".format(log_path), flush=True)
        t0 = time.perf_counter()
        # Separate process per clip: render and grader both hold real memory,
        # and one clip failing must not take the rest of the roster with it.
        with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
            p = subprocess.run(cmd, cwd=APP, stdout=lf,
                               stderr=subprocess.STDOUT, text=True)
        dt = time.perf_counter() - t0
        with open(log_path, "r", encoding="utf-8", errors="replace") as lf:
            out = lf.read()
        # The console here is cp1252 and the logs carry progress-bar box-drawing
        # characters, so echoing the tail raw kills the whole roster with a
        # UnicodeEncodeError AFTER the clip has already been rendered and
        # graded -- which is exactly the worst moment to lose it. Coerce to
        # whatever stdout can actually represent.
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        tail = "\n".join(out.splitlines()[-30:])
        print(tail.encode(enc, "replace").decode(enc, "replace"), flush=True)
        if p.returncode != 0:
            print("[duo] {} FAILED rc={}".format(stem, p.returncode), flush=True)
            results.append((stem, dt, None))
            continue
        results.append((stem, dt, parse(out)))

    print("\n\n" + "=" * 104)
    print("DUO SUMMARY - the number that matters is WRONG FACESET, from the decision")
    print("=" * 104)
    print("{:<7}{:<10}{:>8}{:>13}{:>7}{:>17}{:>18}{:>11}".format(
        "clip", "person", "frames", "swapped", "flips",
        "WRONG FACESET", "looks-like-other", "gradeable"))
    print("-" * 104)
    for stem, dt, people in results:
        if not people:
            print("{:<7}{:<10}".format(stem, "FAILED"))
            continue
        for pp in people:
            wrong = ("n/a" if pp['wrong'] is None else "{}/{} ({:.1f}%)".format(
                pp['wrong'], pp['wrong_of'],
                100.0 * pp['wrong'] / max(1, pp['wrong_of'])))
            other = ("n/a" if pp['other'] is None
                     else "{}/{}".format(pp['other'], pp['other_of']))
            grad = ("0%" if not pp['other_of'] else "{:.0f}%".format(
                100.0 * pp['other_of'] / max(1, pp['swapped'])))
            print("{:<7}{:<10}{:>8}{:>13}{:>7}{:>17}{:>18}{:>11}".format(
                stem, pp['name'], pp['frames'],
                "{} ({:.0f}%)".format(pp['swapped'], pp['pct']),
                pp['flips'], wrong, other, grad))
    print("-" * 104)
    print("WRONG FACESET    = the pipeline pasted the OTHER person's faceset, from its")
    print("                   own decision at the composite. Exact on contact frames.")
    print("                   This is the two-faceset bug, if there is one.")
    print("looks-like-other = output re-measured closer to the other faceset, over the")
    print("                   frames where that measurement is meaningful at all.")
    print("gradeable        = share of swapped frames with usable cosine columns. A low")
    print("                   number does NOT invalidate WRONG FACESET.")


if __name__ == "__main__":
    main()
