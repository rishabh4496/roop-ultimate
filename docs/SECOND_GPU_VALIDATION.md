# Validating the second GPU

Everything here is capability-detected at runtime. Nothing needs editing to
move between cards: models, thread count, pool sizes and the safety policy all
come from the machine's own `config.yaml` and its own detected VRAM tier.

The project's second target is the **RTX 3060 Laptop (6 GB)**. If the card in
front of you is a different one — a 3070, say — the same commands apply
unchanged, and the only thing that must change is the `--target` label and the
row you file the numbers under. **Never file a result under another GPU's row,
and never average two GPUs into one number.**

## 0. Before anything

```bash
cd app
env/Scripts/python.exe tests/diag_device.py
```

This prints the detected profile and proves TensorRT is really executing
rather than silently falling back to CPU. Read it before trusting any number
that follows.

Then confirm the locked fixture is present. `double/d4.mp4` must fingerprint
as **1280x720, 13305 frames**. A clip named `d4.mp4` that is 854x480 is a
DIFFERENT VIDEO — that substitution has already put a wrong result into this
record once. The harness checks this itself and flags
`comparable_to_locked_baseline: false`, so read that field.

## 1. The controlled baseline

```bash
env/Scripts/python.exe tests/baseline_controlled.py --tag phase2_<gpu> --target "RTX 3060"
```

Compare against that GPU's own previous row, never against the 4070's.

## 2. Gate E — the unified scheduler

This is the one item where the 4070 answer is **NEUTRAL at best (-3.7%,
p~0.16)** and the second target has never been measured. The scheduler's
RAM-aware admission was written for memory pressure, which is a small card's
problem, so it may well behave differently there. It currently defaults ON.

Run it **order-balanced**, not as a single pair. A single A-then-B pair is
what produced the withdrawn +1.0%:

```bash
# three OFF-first pairs
for p in 1 2 3; do
  for s in 0 1; do
    env/Scripts/python.exe tests/baseline_controlled.py --tag il_p${p}_s${s} \
      --target "RTX 3060" --start 0 --end 600 --env ROOP_UNIFIED_SCHEDULER=$s
  done
done
# three ON-first pairs, to cancel the within-pair order effect
for p in 1 2 3; do
  for s in 1 0; do
    env/Scripts/python.exe tests/baseline_controlled.py --tag mir_p${p}_s${s} \
      --target "RTX 3060" --start 0 --end 600 --env ROOP_UNIFIED_SCHEDULER=$s
  done
done
```

Average the six within-pair deltas. On the 4070 the second arm of a pair was
faster in 5 of 6 pairs **regardless of which treatment it carried**, which is
why the mirrored set is not optional.

## 3. Establish that target's measurement resolution FIRST

Before believing any A/B, run the same configuration three times and look at
the spread:

```bash
for i in 1 2 3; do
  env/Scripts/python.exe tests/baseline_controlled.py --tag null_$i \
    --target "RTX 3060" --start 0 --end 600
done
```

On the 4070 this gave 10.46 / 10.06 / 10.46 — 4% — while a 20-minute set
spread ~8% and one disturbed arm fell 34%. The 3060's historical drift is
~15%. **An effect smaller than that spread cannot be reported as an effect**,
in either direction.

## 4. The rest of the matrix

```bash
env/Scripts/python.exe tests/phase12_benchmark.py --target "RTX 3060" --start 0 --end 600
env/Scripts/python.exe tests/phase13_benchmark.py --target "RTX 3060" --start 0 --end 600 \
    --codecs libx264,h264_nvenc,hevc_nvenc --segment-sizes auto,600 --out output/phase13_fwd
env/Scripts/python.exe tests/phase13_benchmark.py --target "RTX 3060" --start 0 --end 600 \
    --codecs hevc_nvenc,h264_nvenc,libx264 --segment-sizes 600,auto --out output/phase13_rev
env/Scripts/python.exe tests/phase14_autotune.py --target "RTX 3060" --force
```

Phase 13 is run twice on purpose: the 4070's original single-order table
ranked codecs in exactly its own run order and had to be withdrawn.

## 5. Cross-target validation owed for the 2026-08-31 fixes

Four fixes were made on the 4070 and need confirming on the second card:

1. **`_prof('decode')` / `_prof('encode')` in the scheduler's frame path.**
   Check `encode_write_seconds` is non-zero in any Phase 13 arm. A zero there
   means the encoder stage is unmeasured again.
2. **The autotuner's measured noise floor.** Phase 14's report now carries
   `measured_noise_spread`, `min_improvement_used` and `confirmation`. On a
   card with ~15% drift the threshold should come out far above the 1% floor.
   If it promotes something, check `confirmation.confirmed` is true.
3. **The pinned capture frame.** `WORKLOAD["capture_frame"]` is 4930 for
   `double/d4.mp4`. It was previously chosen by a 30-second wall-clock scan,
   which on a half-speed machine buys half the scan and can select a different
   seed — so the two GPUs could have been comparing different source captures
   while both reported the locked fixture.
4. **The hardware-signature migration.** The first launch on the second card
   SHOULD print the `[Hardware] this config was tuned on ...` line, because
   that is a real GPU change. If it prints on *every subsequent* launch too,
   the migration is not working.

## 6. Still failing, and not closeable from the 4070

**Phase 3's strict `<2.5 GB` RSS gate FAILS on the 3060 at 3.73 GB peak**, on
the 720p locked fixture. That is the project's one hard acceptance failure. It
is a 3060 measurement and must be closed there.

Also open on that target: `DMDNet` errors with
`TypeError: 'NoneType' object is not subscriptable`, and 22.2% of frames on
d4 detect no face at all — the long-standing "15% no-face rate" reproducing.
