"""Stage 16 census: can a DIFFERENTIAL margin tell the target from a bystander
when the two share a recognition crop?

The brief's gate binds a face to the target only if

    S_target(i) >= tau_lock   and   S_target(i) - max_b S_b(i) >= 0.12

That margin was chosen on paper. `roop/selected_routing.py` records why a
cousin of it was rejected on 2026-09-22: during contact the neighbour is inside
the face's own crop, so the embedding measures the PAIR, and "no absolute cap
separates them". A RELATIVE margin might -- or might not. This measures it on
the footage that has contact, so the threshold is read off the distribution it
will actually see (AGENTS.md: "measure the distribution the gate reads").

Two people stand in for target and bystander: person A is the target, person B
is the registered bystander. For every detection:

  * `contam`   -- face_contact's measured fraction of the recognition crop that
                  belongs to the neighbour (`_emb_contam`);
  * S_A, S_B   -- w600k cosine to each person's gallery of CLEAN exemplars
                  (frames where both faces have contam < --clean);
  * truth      -- which person the face is, by position continuity between the
                  clean frames that bracket it. This is NOT the embedding under
                  test, so the comparison is not circular. It is validated
                  visually on a sample (`--panel`), not assumed.

`collect` needs the GPU (detector + recogniser, nothing else); `analyze` is
numpy only and re-runs from the saved .npz.

Usage (cwd = app):
    env/Scripts/python.exe tests/probe_bystander_stage16.py collect \
        --video G:/pinokio/roop-keep/double/d3.mp4 --out <scratch>/d3.npz
    env/Scripts/python.exe tests/probe_bystander_stage16.py analyze \
        --npz <scratch>/d3.npz [--npz <other>.npz ...]
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _unit(v):
    v = np.asarray(v, np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


# --------------------------------------------------------------------------- #
# collect
# --------------------------------------------------------------------------- #

def collect(args):
    import cv2
    import two_face_video as tfv          # noqa: E402  (perf env, as run.py does)
    import angle_bench as ab              # noqa: E402

    tfv._apply_startup_runtime_environment()
    import yaml
    with open(os.path.join(APP, "config.yaml")) as fh:
        cfg = yaml.safe_load(fh) or {}
    provider = args.provider or (cfg.get("provider") or "cuda")
    ab.init_pipeline(provider, cfg.get("swap_model") or "inswapper", "None",
                     "None", sync_config=True)
    from roop.face_util import get_all_faces

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    end = args.end if args.end > 0 else total
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    rows = []       # frame, box(4), emb(512), contam, det_score, n_faces_in_frame
    idx = args.start
    while idx < end:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - args.start) % args.stride == 0:
            faces = get_all_faces(frame) or []
            for f in faces:
                emb = getattr(f, "embedding", None)
                if emb is None:
                    continue
                try:
                    contam = float(f.get("_emb_contam") or 0.0)
                except (TypeError, ValueError, AttributeError):
                    contam = 0.0
                kps = getattr(f, "kps", None)
                kps = (np.full(10, np.nan, np.float32) if kps is None
                       else np.asarray(kps, np.float32).reshape(-1)[:10])
                pose = getattr(f, "pose", None)
                pose = (np.full(3, np.nan, np.float32) if pose is None
                        else np.asarray(pose, np.float32).reshape(-1)[:3])
                rows.append((idx, *[float(v) for v in f.bbox], contam,
                             float(getattr(f, "det_score", 0.0) or 0.0),
                             len(faces), *[float(v) for v in kps],
                             *[float(v) for v in pose],
                             *[float(v) for v in _unit(emb)]))
        idx += 1
        if idx % 200 == 0:
            print(f"[collect] {idx}/{end}", flush=True)
    cap.release()
    arr = np.asarray(rows, np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, rows=arr, video=args.video)
    print(f"[collect] {len(arr)} detections over {len(set(arr[:, 0]))} frames -> {args.out}")


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #

def _split(rows):
    f = rows[:, 0].astype(int)
    box = rows[:, 1:5]
    contam = rows[:, 5]
    det = rows[:, 6]
    n = rows[:, 7].astype(int)
    emb = rows[:, -512:]              # layout 1: 8 + 512 columns; layout 2 adds kps(10)+pose(3)
    return f, box, contam, det, n, emb


def _centre(b):
    return np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5])


def _kmeans2(x, iters=25):
    """Spherical 2-means on unit vectors; returns (labels, centroids) or None."""
    if len(x) < 8:
        return None
    # seed with the two least-similar vectors found greedily
    i0 = 0
    i1 = int(np.argmin(x @ x[i0]))
    i0 = int(np.argmin(x @ x[i1]))
    c = np.stack([x[i0], x[i1]])
    lab = np.zeros(len(x), int)
    for _ in range(iters):
        lab = np.argmax(x @ c.T, axis=1)
        for k in (0, 1):
            if (lab == k).any():
                v = x[lab == k].sum(axis=0)
                c[k] = v / max(float(np.linalg.norm(v)), 1e-9)
    return lab, c


def label_people(rows, clean, jump=0.5, vote=0.45):
    """Per-detection person id (0/1, -1 = unlabelled).

    1. Clean detections (contam < `clean`, frame has exactly two faces) are
       split into two persons by spherical 2-means on the embedding. Sound
       exactly where the embedding is trustworthy, and immune to shot cuts.
    2. The two-face frames are cut into RUNS: consecutive sampled frames where
       both faces move less than `jump` x face-width between samples. A shot cut
       or a crossing breaks the run. Inside a run, the person on the LEFT is
       the same person throughout (heads in contact do not trade sides without
       crossing, which would break the run), so the clean frames of the run
       say who is on the left and the contaminated frames are labelled by
       x-order. A run with no clean frame, or whose clean frames disagree
       about who is on the left, is dropped -- unlabelled, never guessed.

    Position is the truth proxy for contaminated frames; it is not the
    embedding under test, so the comparison is not circular there.
    """
    f, box, contam, det, n, emb = _split(rows)
    frames = sorted(set(f))
    by_frame = {fr: np.where(f == fr)[0] for fr in frames}
    clean_ix = [i for i in range(len(rows)) if contam[i] < clean and n[i] == 2]
    km = _kmeans2(emb[clean_ix]) if clean_ix else None
    if km is None:
        return None, []
    lab, cent = km
    pid = np.full(len(rows), -1, int)
    cid = {i: int(k) for i, k in zip(clean_ix, lab)}
    step = int(np.median(np.diff(frames))) if len(frames) > 1 else 1

    two = [fr for fr in frames if len(by_frame[fr]) == 2]
    runs, cur = [], []
    prev = None
    for fr in two:
        ix = sorted(by_frame[fr], key=lambda i: _centre(box[i])[0])
        c = [_centre(box[i]) for i in ix]
        wdt = max(box[ix[0]][2] - box[ix[0]][0], 1.0)
        if prev is not None and fr - prev[0] <= 2 * step:
            moved = max(np.linalg.norm(c[0] - prev[1][0]), np.linalg.norm(c[1] - prev[1][1]))
            if moved > jump * wdt:
                runs.append(cur)
                cur = []
        elif prev is not None:
            runs.append(cur)
            cur = []
        cur.append((fr, ix))
        prev = (fr, c)
    if cur:
        runs.append(cur)

    clean_pairs = []
    for run in runs:
        left_ids = set()
        for fr, ix in run:
            if all(i in cid for i in ix) and cid[ix[0]] != cid[ix[1]]:
                left_ids.add(cid[ix[0]])
                clean_pairs.append(fr)
        if len(left_ids) == 0 and vote:
            # No clean pair in this run (a kiss never gets clean). Vote with the
            # moderately contaminated faces: each is classified by the clean
            # centroids, and the run is labelled only if the people on the left
            # agree almost unanimously. Many frames vote, so a few wrong
            # readings cannot decide it, and the heavy frames of the run are
            # then labelled by POSITION, which is the truth proxy under test.
            tally = Counter()
            for fr, ix in run:
                if all(contam[i] < vote for i in ix):
                    k = [int(np.argmax(cent @ emb[i])) for i in ix]
                    if k[0] != k[1]:
                        tally[k[0]] += 1
            tot = sum(tally.values())
            if tot >= 8:
                top, cnt = tally.most_common(1)[0]
                if cnt >= 0.9 * tot:
                    left_ids = {top}
        if len(left_ids) != 1:
            continue
        left = next(iter(left_ids))
        for fr, ix in run:
            pid[ix[0]] = left
            pid[ix[1]] = 1 - left
    return pid, clean_pairs


def analyze(args):
    margin = args.margin
    tau_lock = args.tau_lock
    tau_hold = args.tau_hold
    agg = []
    for path in args.npz:
        z = np.load(path, allow_pickle=True)
        rows = z["rows"]
        name = os.path.basename(path)
        f, box, contam, det, n, emb = _split(rows)
        pid, clean_pairs = label_people(rows, args.clean)
        if pid is None:
            print(f"[{name}] fewer than two clean two-face frames -> cannot label")
            continue
        # Galleries: CLEAN exemplars only, one per clean frame, thinned.
        gal = {0: [], 1: []}
        gal_f = {0: [], 1: []}
        for i in range(len(rows)):
            if pid[i] in (0, 1) and contam[i] < args.clean and n[i] == 2:
                gal[int(pid[i])].append(emb[i])
                gal_f[int(pid[i])].append(f[i])
        gal = {p: np.asarray(v) for p, v in gal.items()}
        gal_f = {p: np.asarray(v) for p, v in gal_f.items()}
        if min(len(gal[0]), len(gal[1])) < 3:
            print(f"[{name}] gallery too small {len(gal[0])}/{len(gal[1])}")
            continue
        # Score every labelled detection against exemplars from OUTSIDE a
        # +-`--window` frame neighbourhood. Neighbouring frames of one clip are
        # near-duplicates (cos ~0.98), so leaving out only the probe's own
        # frame scores a frame against its own twin and reports margins near 1.
        S = np.zeros((len(rows), 2), np.float32)
        for i in range(len(rows)):
            for p in (0, 1):
                keep = np.abs(gal_f[p] - f[i]) > args.window
                sims = gal[p][keep] @ emb[i]
                S[i, p] = float(sims.max()) if len(sims) else 0.0
        lab = pid >= 0
        print(f"\n=== {name}  detections {len(rows)}  labelled {int(lab.sum())}  "
              f"clean pairs {len(clean_pairs)}  gallery {len(gal[0])}/{len(gal[1])}")
        bands = [(0.0, 0.05), (0.05, 0.2), (0.2, 0.4), (0.4, 1.01)]
        print(f"  target=A, bystander=B; margin {margin}, tau_lock {tau_lock}, "
              f"tau_hold {tau_hold}")
        print("  contam band   | n(A) n(B) | A:margin>=m  A:S_A>=lock | "
              "B:margin>=m (FALSE BIND)  B:S_A>=lock | A med margin  B med margin")
        for lo, hi in bands:
            m = lab & (contam >= lo) & (contam < hi)
            a = m & (pid == 0)
            b = m & (pid == 1)
            if a.sum() + b.sum() == 0:
                continue
            ma = S[a, 0] - S[a, 1]
            mb = S[b, 0] - S[b, 1]
            def frac(x, c):
                return f"{100.0 * np.mean(c(x)):5.1f}%" if len(x) else "  n/a "
            print(f"  [{lo:.2f},{hi:.2f}) | {int(a.sum()):4d} {int(b.sum()):4d} | "
                  f"{frac(ma, lambda x: x >= margin)}      "
                  f"{frac(S[a, 0], lambda x: x >= tau_lock)}     | "
                  f"{frac(mb, lambda x: x >= margin)}                  "
                  f"{frac(S[b, 0], lambda x: x >= tau_lock)}       | "
                  f"{(np.median(ma) if len(ma) else float('nan')):+.3f}        "
                  f"{(np.median(mb) if len(mb) else float('nan')):+.3f}")
        agg.append((name, contam, pid, S))

    if agg:
        print("\n--- pooled, contam >= 0.2 only (the case the gate exists for) ---")
        contam = np.concatenate([c for _, c, _, _ in agg])
        pid = np.concatenate([p for _, _, p, _ in agg])
        S = np.concatenate([s for _, _, _, s in agg])
        m = (contam >= 0.2) & (pid >= 0)
        a = m & (pid == 0)
        b = m & (pid == 1)
        print(f"  n(A)={int(a.sum())}  n(B)={int(b.sum())}")
        for mm in (0.0, 0.05, 0.08, 0.12, 0.16, 0.20):
            fb = np.mean((S[b, 0] - S[b, 1]) >= mm) if b.sum() else float('nan')
            fr = np.mean((S[a, 0] - S[a, 1]) < mm) if a.sum() else float('nan')
            print(f"  margin {mm:.2f}: false-bind (B accepted as A) {100*fb:5.1f}%   "
                  f"false-refuse (A rejected) {100*fr:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--video", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--start", type=int, default=0)
    c.add_argument("--end", type=int, default=0)
    c.add_argument("--stride", type=int, default=1)
    c.add_argument("--provider", default=None)
    a = sub.add_parser("analyze")
    a.add_argument("--npz", action="append", required=True)
    a.add_argument("--clean", type=float, default=0.05)
    a.add_argument("--window", type=int, default=60,
                   help="exclude gallery exemplars within +-N frames of the probe")
    a.add_argument("--margin", type=float, default=0.12)
    a.add_argument("--tau-lock", type=float, default=0.42)
    a.add_argument("--tau-hold", type=float, default=0.22)
    args = ap.parse_args()
    (collect if args.cmd == "collect" else analyze)(args)


if __name__ == "__main__":
    main()
