"""Compare two two_face_video arms on the identity columns, frame by frame.

The per-arm summary two_face_video prints answers "did the pipeline swap this
face, and did it pick the right faceset". Both are decided by track/identity
ASSIGNMENT, which sits UPSTREAM of whichever model paints the pixels -- so those
lines are structurally incapable of telling one swap model from another, and
have twice been read as if they could. The columns that do discriminate are
`own` (cosine to the faceset this person should be wearing) and `other`, and
they need pairing across arms to be read at all.

Pairs on (frame, person). Reports only rows where BOTH arms filled the identity
columns in, since a row blank in either arm is not a comparison -- and reports
how many rows that discarded, because a change that alters WHICH frames are
gradable is itself a finding.

Usage:
    env/Scripts/python.exe tests/compare_two_face.py d5_hyperswap_mm25 d5_realswap_mm25
"""

import argparse
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ARMS = os.path.join(APP, "output", "bench_two_face")


def load(tag):
    path = os.path.join(ARMS, tag, "rows.csv")
    if not os.path.exists(path):
        raise SystemExit(f"no rows.csv for arm {tag!r} (looked in {path})")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(v):
    if v is None or str(v).strip() == "":
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def tstat(d):
    n = len(d)
    if n < 2:
        return float("nan")
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    if var <= 0:
        return float("inf") if m else 0.0
    return m / math.sqrt(var / n)


def swapped(r):
    """The pipeline's OWN verdict, with the pixel test only as a fallback.

    two_face_video makes this same choice and explains why at length: on contact
    footage the neighbour's swap overlaps this face's box, so a face the
    pipeline explicitly REFUSED still reads as touched.
    """
    why = (r.get("why") or "")
    if why.strip():
        return "swapped (" in why
    t = fnum(r.get("touched"))
    return t is not None and t > 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_a", help="baseline arm tag")
    ap.add_argument("arm_b", help="arm under test")
    args = ap.parse_args()

    a, b = load(args.arm_a), load(args.arm_b)
    key = lambda r: (r["frame"], r["person"])
    bi = {key(r): r for r in b}

    print(f"{args.arm_a}: {len(a)} rows   {args.arm_b}: {len(b)} rows")

    # `who` is which captured person the PLATE face actually is -- the right key
    # for a per-person split, unlike `person`, which is the left-to-right box
    # index and swaps between the two people whenever they cross.
    people = sorted({r.get("who", "") for r in a if str(r.get("who", "")).strip()})
    groups = [(w, f"person {w}") for w in people] + [("", "ALL")]

    for wsel, label in groups:
        pairs, dropped, n_sw = [], 0, 0
        for ra in a:
            rb = bi.get(key(ra))
            if rb is None:
                continue
            if wsel != "" and str(ra.get("who", "")) != str(wsel):
                continue
            if not (swapped(ra) and swapped(rb)):
                continue
            n_sw += 1
            oa, ob = fnum(ra.get("own")), fnum(rb.get("own"))
            xa, xb = fnum(ra.get("other")), fnum(rb.get("other"))
            if None in (oa, ob, xa, xb):
                dropped += 1
                continue
            pairs.append((oa, ob, xa, xb))
        if not pairs:
            print(f"\n  {label}: nothing gradable "
                  f"({n_sw} swapped rows, all blank in one arm or the other)")
            continue

        # `own` is a DISTANCE to the faceset this face should wear, so lower is
        # better; `margin` = other - own, how much more like the right person
        # than the wrong one, where higher is better.
        d_own = [p[1] - p[0] for p in pairs]
        d_mar = [(p[3] - p[1]) - (p[2] - p[0]) for p in pairs]
        ma = sum(p[0] for p in pairs) / len(pairs)
        mb = sum(p[1] for p in pairs) / len(pairs)
        ga = sum(p[2] - p[0] for p in pairs) / len(pairs)
        gb = sum(p[3] - p[1] for p in pairs) / len(pairs)
        wins = sum(1 for x in d_own if x < 0)
        conf_a = sum(1 for p in pairs if p[2] < p[0])
        conf_b = sum(1 for p in pairs if p[3] < p[1])

        print(f"\n  {label}: {len(pairs)} gradable of {n_sw} swapped rows "
              f"({dropped} blank in one arm)")
        print(f"    {'metric':<22}{args.arm_a[:18]:>19}{args.arm_b[:18]:>19}"
              f"{'delta':>11}{'t':>9}")
        print(f"    {'own (lower better)':<22}{ma:>19.4f}{mb:>19.4f}"
              f"{sum(d_own)/len(d_own):>+11.4f}{tstat(d_own):>9.1f}")
        print(f"    {'margin (higher bett)':<22}{ga:>19.4f}{gb:>19.4f}"
              f"{sum(d_mar)/len(d_mar):>+11.4f}{tstat(d_mar):>9.1f}")
        print(f"    {'B closer to own':<22}{100.0*wins/len(pairs):>18.1f}%")
        print(f"    {'looks like the OTHER':<22}{conf_a:>13} rows{conf_b:>14} rows")


if __name__ == "__main__":
    main()
