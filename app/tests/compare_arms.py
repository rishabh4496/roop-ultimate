"""Compare two angle_video arms frame-by-frame on (pair, yaw, roll).

The sweep's summary table averages within an arm, which hides the thing an A/B
needs: whether a change helped the SAME frame. Pairing on the frame key gives a
paired t-statistic and a win rate, and those are what separate a real shift from
arm-to-arm noise -- except that this pipeline is bit-deterministic (verified: a
re-run reproduces every column to 6 decimal places), so any non-zero difference
here IS the change and the t-statistic only says how consistent it is.

Split at |yaw| >= 90 because the two regimes are not comparable: both swap models
score identity 0.24-0.36 at profile against 0.65-0.79 off it, so pooling them
lets the profile tail drag a frontal result around.

Usage:
    env/Scripts/python.exe tests/compare_arms.py yaw_realswap_lidband realswap_mm25_after
"""

import argparse
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ARMS = os.path.join(APP, "output", "bench_angle_video")

# Higher is better for these; every other numeric column is a DRIFT from the
# plate, where lower is better. Getting this backwards silently reverses every
# verdict, so it is stated once here rather than inferred per column.
HIGHER_IS_BETTER = {"id_source", "id_plate", "ghost", "detected"}
SKIP = {"pair", "yaw", "roll", "note", "floor"}


def load(tag):
    path = os.path.join(ARMS, tag, "rows.csv")
    if not os.path.exists(path):
        raise SystemExit(f"no rows.csv for arm {tag!r} (looked in {path})")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(row, col):
    v = row.get(col, "")
    if v is None or str(v).strip() == "":
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def paired(a_rows, b_rows, col):
    """(deltas, b_wins) over frames where BOTH arms produced a number."""
    key = lambda r: (r["pair"], r["yaw"], r["roll"])
    bi = {key(r): r for r in b_rows}
    out, wins = [], 0
    better_high = col in HIGHER_IS_BETTER
    for ra in a_rows:
        rb = bi.get(key(ra))
        if rb is None:
            continue
        xa, xb = num(ra, col), num(rb, col)
        if xa is None or xb is None:
            continue
        d = xb - xa
        out.append(d)
        if (d > 0) if better_high else (d < 0):
            wins += 1
    return out, wins


def tstat(d):
    n = len(d)
    if n < 2:
        return float("nan")
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    if var <= 0:
        return float("inf") if m else 0.0
    return m / math.sqrt(var / n)


def band(rows, profile):
    out = []
    for r in rows:
        try:
            y = abs(float(r["yaw"]))
        except (TypeError, ValueError):
            continue
        if (y >= 90) == profile:
            out.append(r)
    return out


def report(a_rows, b_rows, a_tag, b_tag, label):
    if not a_rows or not b_rows:
        print(f"\n  {label}: no rows")
        return
    cols = [c for c in a_rows[0].keys() if c not in SKIP]
    print(f"\n  {label} ({len(a_rows)} frames in {a_tag})")
    print(f"    {'column':<10}{a_tag[:16]:>17}{b_tag[:16]:>17}"
          f"{'delta':>11}{'t':>9}{'B better':>10}")
    for c in cols:
        d, wins = paired(a_rows, b_rows, c)
        if not d:
            continue
        av = [num(r, c) for r in a_rows]
        av = [x for x in av if x is not None]
        ma = sum(av) / len(av)
        mb = ma + sum(d) / len(d)
        arrow = "" if abs(sum(d) / len(d)) > 1e-9 else "  (identical)"
        print(f"    {c:<10}{ma:>17.4f}{mb:>17.4f}{sum(d)/len(d):>+11.4f}"
              f"{tstat(d):>9.1f}{100.0*wins/len(d):>9.1f}%{arrow}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_a", help="baseline arm tag")
    ap.add_argument("arm_b", help="arm under test")
    args = ap.parse_args()

    a, b = load(args.arm_a), load(args.arm_b)
    key = lambda r: (r["pair"], r["yaw"], r["roll"])
    shared = set(map(key, a)) & set(map(key, b))
    print(f"{args.arm_a}: {len(a)} rows   {args.arm_b}: {len(b)} rows   "
          f"paired on {len(shared)}")
    if not shared:
        raise SystemExit("no frames in common -- were these rendered from the "
                         "same facesets and yaws?")

    for profile, label in ((False, "NON-PROFILE |yaw| < 90"),
                           (True, "PROFILE |yaw| >= 90")):
        report(band(a, profile), band(b, profile), args.arm_a, args.arm_b, label)

    d_all = []
    for c in ("id_source", "eyes", "ghost"):
        d, _ = paired(a, b, c)
        d_all.append((c, sum(d) / len(d) if d else 0.0))
    if all(abs(v) < 1e-9 for _, v in d_all):
        print("\n>>> The two arms are IDENTICAL on identity, eyes and ghost. "
              "For a bit-deterministic pipeline that means the change did not "
              "reach the pixels these columns read.")


if __name__ == "__main__":
    main()
