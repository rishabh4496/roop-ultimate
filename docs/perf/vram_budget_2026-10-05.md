# Keeping peak VRAM under ~90% of the card (2026-10-05)

Brief: using the Stage 0 VRAM logs, keep peak VRAM below ~90% of total with **no model or precision change**.
(1) After `_precompute_temporal` + `_release_replayed_analysis`, also shrink the detector and FaceAnalysis pools to width 1.
(2) Do not leave the analyser, detector and processor pools all at width 2 when the budget is exceeded: decide through
`session_pool.TensorRTResourceManager` from measured free memory, not GPU name. (3) On Windows print a warning when VRAM use
crosses 95%, naming NVIDIA Control Panel > CUDA - Sysmem Fallback Policy > Prefer No Sysmem Fallback for python.exe.
Acceptance: bit-identical output, peak VRAM reported, render fps versus baseline.

**Result: all three done. Output is bit-identical in every arm (d4 600 frames, sha256 `490d9baa…` = the 2026-10-04 baseline's;
r50 pair `ef3a0cb9…` in both arms). Under another application's 5 GiB the peak went from 96.7% to 76.9% of the card with no fps
loss. On an idle card with the live scrfd config item 1 is worth ~50 MiB (the card was already at 54%); it is worth 1.24 GB when
the detector is `retinaface_r50`.** RTX 4070, `config.yaml` live (scrfd, TensorRT mixed, 2/2/2 pools, Restore Ultra, DFL XSeg, 10
threads), d4.mp4 frames 0..600, `ROOP_STAB_CHUNK_MB=990.8` pinned, code `e2b69f5` + this change.

## What the Stage 0 logs say

`STAGE0_BENCHMARK_REPORT.md` / `benchmark_stage0_results.json` (2026-09-30, provider cuda, 20 threads, 13 scenarios in ONE process):
peak VRAM per scenario 5,027 → 6,259 → 7,742 → 9,093 → 10,558 → 11,937 → 12,110 … **12,136 MiB (98.8%)**, and from scenario 7 on
the same stage ran at 0.3-0.9 fps (scenario 1: 11.6 fps) with GPU utilisation 88-95% - the paging signature (a card pinned near
100% VRAM reads as a hang; see `session_pool._advisory_pool_size`). That is **growth of ~1.3 GB per scenario inside one process**;
this session did not diagnose what accumulates there, and did not test consecutive renders in one app session (open, below).
What it does establish is the failure mode the three changes are for: past ~95% the render crawls instead of failing.

On the production path at the shipped config the card is nowhere near that: `baseline_2026-10-04.md` and this session's baseline
peak at 6.7-7.3 GB (54-60%). The pressure arms below simulate what pushes it over: another application holding VRAM.

## 1. Pools shrink to width 1 after the pre-pass

`face_util.shrink_analysis_pools(1)`, called from `ProcessMgr._release_replayed_analysis` right after the aux sessions go:
the FaceAnalysis pool keeps instance 0, every hybrid detector pool (`yoloface`, `retinaface`, `retinaface_r50_gpu`, `yunet`) is
trimmed through the lease queue (`session_pool.shrink_lease_pool`: take every instance out - which waits for any lease still out;
a lease that does not return in 30 s means no shrink - keep the first, drop the rest). `_ANALYSER_POOL_CEILING` stops
`_ensure_face_analyser` rebuilding the pool at its configured width on the next call; it ends with the pool
(`release_face_analyser`, run in the render's cleanup) and is also cleared at every run start, so a ceiling that outlived its
pool cannot hold the next render's pre-pass at width 1. Every consumer of the analysis pool already reads `analysis_pooled()`
(width, not config), so with one instance the swap phase's verify/rescue/autorotate calls take the analysis stage lock, as a
width-1 pool always did. (The swap phase of d4 makes 170 + 37 detector calls; its pre-pass 2,598.)

| config, idle card, no governor | `[VRAM]` after the aux release | after the shrink | `frame 500` | peak (sampler) | frame-loop fps | pre-pass fps | sha256 |
|---|---:|---:|---:|---:|---:|---:|---|
| scrfd, HEAD (2 runs) | 3,382 | - | 6,681 | 6,667 / 6,681 | 14.92 / 14.87 | 34.4 / 34.6 | `490d9baa…` |
| scrfd, patched | 3,392 | **3,336 (-56)** | 6,633 | 6,634 | 14.92 | 34.0 | `490d9baa…` |
| scrfd, patched + governor | - | - | - | 6,601 | 15.30 | 34.4 | `490d9baa…` |
| r50, HEAD | 5,744 | - | 9,000 | **9,001 (73.3%)** | 13.09 | 25.9 | `ef3a0cb9…` |
| r50, patched | 5,743 | **4,503 (-1,240)** | 7,762 | **7,763 (63.2%)** | 13.57 | 25.6 | `ef3a0cb9…` |

(MiB, device-wide `nvidia-smi` through the harness sampler; `[VRAM]` lines are the stage log.) With scrfd the instance that is left
once the aux sessions are gone is `det_10g`, which is small; with `retinaface_r50` each instance carries a 104 MB network plus its
TensorRT contexts. The patched governor arm prints `[Runtime] analysis pools shrunk to width 1 for the swap phase: 1 analyser and
0 (scrfd) / 1 (r50) detector instance(s) released`.

## 2. The render's plan lowers pool widths, enforced by the resource manager

`vram_governor.plan_job` already stepped the swap batch down and then GPEN's resolution (which changes the look). It now steps
**pool widths between the two**: one context of one pool at a time, whichever frees the most estimated memory first, until the
projection fits or every pool is at 1; a pool whose models are not in the job frees nothing and is left alone. The plan must leave
`max(vram_safety_margin_gb, 10% of the card)` free - the 90% ceiling - so a small slider cannot plan a peak past it. The widths go
to `TensorRTResourceManager.set_budget_caps` at `admit()` and are cleared in `finish()`; `select_pool_size` applies them to every
pool (explicit settings too: the cap is the physical kind), outside the memoised decision so a cap is never frozen into the next
render. Decisions use `vram_governor.query_vram_mb` (NVML, device-wide, every process); nothing reads a GPU name. A signature no
render has run yet (narrower pools) borrows the largest learned ratio of the same configuration, since 1.0 would call the narrower
plan cheaper than the wider one measured.

Under pressure: another process holds 5.00 GiB (`ballast.py`; device already at 6,320 MiB before the render), same calibration
prior, same pinned render, **through the governor** (`--governor`, the production admission):

| arm | pools | governor | peak (sampler) | frame-loop fps | pre-pass fps | sha256 |
|---|---|---|---:|---:|---:|---|
| A′ HEAD | 2/2/2 | batch 8→1, then "WARNING: still short of the margin after every step-down" | **11,875 MiB (96.7%)** | 15.69 | 34.1 | `490d9baa…` |
| B′ patched | **1/1/1** | batch 8→1, detmask 2→1, enhancer 2→1, swap 2→1; `[SessionPool] … pool 2 -> 1` on all five builds | **9,443 MiB (76.9%)** | 16.65 | 37.1 | `490d9baa…` |

One render per arm, not counterbalanced: the +6% fps in B′ is **not** claimed as a speedup (the only claim is no loss), though it
agrees with the standing finding that more contexts do not pay on this card. The plan stayed "short" in B′ because this config's
`vram_safety_margin_gb` is 3.0 GB, which binds well before the 10% floor (1,228 MB); the real peak was 77%. The governor's measured
job peak was 2,186 MB against an estimate of 2,446.

## 3. The Windows warning

`vram_governor.warn_if_vram_critical`, once per process, `sys.platform == 'win32'`, used/total >= 95% (inclusive). It is checked at
admission, on every 0.5 s sample of the render's peak sampler, and at every `[VRAM]` stage log, and goes through `bar_write` (a bare
`print` from the sampler thread was observed to land on the progress bar's line). With a 7.5 GiB ballast (device at 8,880 MiB
before the render) the minimal plan still reached 96.9% and the run printed, once:

    [VramGovernor] WARNING: GPU memory is 96% used (11759 of 12282 MiB) at the render. Past ~95% the Windows driver spills to shared
    system memory over PCIe and throughput collapses (measured on a 12 GB card: 11.6 fps down to 0.3-0.6 fps while GPU utilisation
    read 93-95%), which looks like a hang. To make an over-commit fail fast instead: NVIDIA Control Panel > Manage 3D Settings >
    Program Settings > add G:\pinokio\api\roop-ultimate\app\env\Scripts\python.exe > 'CUDA - Sysmem Fallback Policy' > 'Prefer No
    Sysmem Fallback'. Or free VRAM: close other GPU applications, or lower ROOP_TRT_POOL / ROOP_DETMASK_POOL.

That render (16.32 fps, bit-identical) did not collapse at 96.9% - this card paged nothing in a 170 s render - so the warning is
about the cliff, not evidence this particular run hit it.

## Not done / not measured

- **The Stage 0 growth itself** (1.3 GB per scenario in one process) was not diagnosed, and **consecutive renders in one app
  session were not tested**. The plan is per render and `release_resources()` runs before admission, but a render that starts with
  the previous one's arenas still resident will see less free memory and step down sooner.
- The governor runs on the production path (`core.batch_process_regular`) only. Headless `--project --render`, previews and the
  `two_face_video.py` harness bypass it; the harness now has `--governor` (and `baseline_snapshot.py --governor`) to measure it.
- A pool cap only shapes the NEXT build; pools already resident keep their width until released (every render releases its
  processors first).
- The RTX 3060: not measured. Its pools are already 0/0 (single context) and 10% of 6 GB (614 MB) is below the default margin, so
  the planner changes nothing there at its live config (`test_3060_live_config_is_left_alone`).
- The enhancer / swap / mask pool step order is "whichever frees the most estimated MB", an assumption; this session measured only
  the all-1 outcome under pressure, not the fps cost of each single step.

Reproduce: `tests/baseline_snapshot.py --only d4 --no-null --no-warmup --pin-chunk-mb 990.8 --window 0 600 [--governor]
[--detector-engine retinaface_r50]`; ballast: `docs/perf/vram_budget_ballast.py <GiB> <stopfile>` (a second process holding N GiB of CUDA memory); raw results in `vram_budget_2026-10-05.json`. Tests: `app/tests/test_vram_pool_budget.py` (41).
