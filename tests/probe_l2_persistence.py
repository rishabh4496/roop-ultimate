"""Does a persisting L2 window keep the ROI working set resident when a full-frame decode ring floods the cache?

    app\\env\\Scripts\\python.exe tests\\probe_l2_persistence.py [--mib 4 8 16] [--reps 300] [--out report.json]

An API call that returns success proves nothing (see roop/nvenc_quality.py for the lesson), so this measures the effect.

The first version of this probe ran the flood on a second stream and timed a read loop beside it. That could not resolve
anything: 40 reductions took 15 ms (~16 GB/s on a card that moves 500), i.e. the time was SMs taken by the other stream's
copy kernels, not where the bytes came from. This version is sequential on ONE stream, so nothing competes:

    repeat:  flood (copy 2 x 512 MiB through L2)  ->  read the working set once (timed, CUDA events)

After a 1 GiB flood a working set that is not protected has been evicted, so the read comes from VRAM; one that persists is
served from L2. The window on the stream covers the working set; the flood kernels' addresses lie outside it, so they get
the window's miss property (streaming) - exactly the infer-stream configuration, with the decode ring as the streaming traffic.

Arms are interleaved round-robin (clocks drift), the first repetitions are dropped, and medians are compared against the
spread between rounds of the SAME arm: a difference inside that spread is not an effect.
"""
import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from app.core import l2_persistence as l2  # noqa: E402


@triton.jit
def _read_kernel(ptr, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(ptr + i, mask=i < n, other=0.0).to(tl.float32)
    tl.store(out + tl.program_id(0), tl.sum(v, axis=0))


def one_arm(arm, ws, arena, flood, stream, reps, out, do_flood=True):
    l2.clear(stream)
    applied = None
    if arm == "persist":
        applied = l2.pin_arena(stream, arena, 16 * l2.MiB, 1.0)
    n = ws.numel()
    grid = (triton.cdiv(n, 4096),)
    times = []
    with torch.cuda.stream(stream):
        for r in range(reps + 5):
            if do_flood:
                flood[1].copy_(flood[0])
                flood[0].copy_(flood[1])
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(stream)
            _read_kernel[grid](ws, out, n, BLOCK=4096)
            b.record(stream)
            b.synchronize()
            if r >= 5:
                times.append(a.elapsed_time(b) * 1000.0)                      # microseconds
    l2.clear(stream)
    return statistics.median(times), applied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=float, nargs="+", default=[4, 8, 16])
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ok, why = l2.available()
    if not ok:
        print(f"[PENDING] {why}")
        return 2
    info = l2.query_l2()
    print(f"L2 {info.l2_bytes / l2.MiB:.1f} MiB, max persisting {info.max_persisting_bytes / l2.MiB:.2f} MiB, "
          f"max window {info.max_window_bytes / l2.MiB:.0f} MiB  ({torch.cuda.get_device_name(0)})")
    stream = torch.cuda.Stream()
    flood = [torch.ones(512 * l2.MiB, dtype=torch.uint8, device="cuda") for _ in range(2)]
    out = torch.empty(1 << 16, dtype=torch.float32, device="cuda")
    report = []
    for mib in args.mib:
        arena = l2.PersistentArena(int(mib * l2.MiB) + l2.ALIGN)
        ws = arena.alloc("roi", (int(mib * l2.MiB // 2),), torch.float16)
        ws.normal_()
        torch.cuda.synchronize()
        rounds = {"off": [], "persist": []}
        warm = statistics.median(one_arm("off", ws, arena, flood, stream, args.reps, out, do_flood=False)[0] for _ in range(3))
        applied = None
        for _ in range(args.rounds):
            for arm in rounds:
                t, ap_ = one_arm(arm, ws, arena, flood, stream, args.reps, out)
                rounds[arm].append(t)
                applied = ap_ or applied
        off, per = statistics.median(rounds["off"]), statistics.median(rounds["persist"])
        spread = lambda v: 100.0 * (max(v) - min(v)) / statistics.median(v)
        row = dict(working_set_mib=mib, warm_no_flood_us=warm, off_us=off, persist_us=per, delta_pct=100.0 * (per / off - 1.0),
                   off_spread_pct=spread(rounds["off"]), persist_spread_pct=spread(rounds["persist"]), rounds=rounds, applied=applied)
        report.append(row)
        print(f"working set {mib:5.1f} MiB: CONTROL warm read, no flood {warm:6.1f} us | after a 1 GiB flood  off {off:7.1f} us (spread {row['off_spread_pct']:.0f}%)   "
              f"persist {per:7.1f} us (spread {row['persist_spread_pct']:.0f}%)   {row['delta_pct']:+.1f}%", flush=True)
        print("   window:", applied, flush=True)
        del arena, ws
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(dict(l2=info.__dict__, rows=report, device=torch.cuda.get_device_name(0)), fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
