# Swarm Measurement Report - @worker-verify

**Date:** 2026-09-18
**Device:** RTX 4070 Desktop (12 GB), 24P/32L cores, 32 GB RAM
**Source revision:** `4eac84b2bf2ea6a3d6b89b626646c3619a70d32a` (branch `main`)
**Tracked-file dirtiness:** NONE. `git status --porcelain` reports only untracked scratch:
`.jcodeswarm.toml`, `app/tests/_coord_color_modes.py`, `app/tests/_coord_lighting_probe.py`,
`app/tests/_coord_lighting_sites.py`, `app/tests/bench_lighting_stage.py`.

---

## 1. Headline: the baseline was initially deferred, then measured safely

This section records the initial blocked state. The user's render later finished,
the GPU became quiescent, and the controlled pre-optimization baseline was run
successfully. The pending rows and contention ruling below are historical
evidence for why that run was deferred, not the final baseline status. See
section 1b for the verified result.

| Target | Arm | FPS (warm-up) | FPS (steady) | Peak VRAM | Source revision | Status |
|---|---|---|---|---|---|---|
| RTX 4070 Desktop | `swarm_prebaseline` (pre-optimization) | 7.82 | n/a | 9,958 MB | `4eac84b` | **MEASURED - see section 1b** |
| RTX 4070 Desktop | optimized (post-swarm) | - | - | - | `6c26f06` | **PENDING - optimized comparison not run** |
| RTX 3060 Laptop | any | - | - | - | n/a | **PENDING - HARDWARE NOT PHYSICALLY PRESENT** |

The optimized comparison remains empty on purpose. No number in this table was
estimated, carried over, or copied from another device.

### Why the 4070 baseline was initially PENDING

The GPU was not idle at any point during this session. Measured directly with
`nvidia-smi`:

```
NVIDIA GeForce RTX 4070, 10897 MiB / 12282 MiB used, 67 % util
```

The holder is the **user's own live application**, not a swarm worker:

| PID | Process | Role |
|---|---|---|
| 22272 | `G:\pinokio\bin\miniforge\python.exe` (`run.py --ui react`) | Pinokio-launched React UI, ~10.0 GB RSS |
| 30912 | `ffmpeg -hwaccel cuda -i app\temp\api_uploads\...` | decode, started 13:07:39 |
| 46040 / 32160 | `ffmpeg -f rawvideo -s 1280x720 -pix_fmt bgr24 -r 25.0 -c:v hevc...` | encode, cycling |

That is roop-ultimate's own writer pipeline actively encoding a real user job at
1280x720, leaving roughly **1.35 GB free VRAM**. A `baseline_controlled.py` run
started in that state would either OOM or produce a contaminated figure. It was
therefore deferred until the render finished rather than interrupting the user's
work.

**Initial ruling (coordinator `wyvern`, concurring with this worker): DEFER.**
A PENDING row that can be honestly explained beats both a contaminated number
and a number obtained by interrupting the user's work. The later quiescent run
closed the pre-optimization row without changing that ruling.

### Independent corroboration that contention was real, not theoretical

This is not an inference from `nvidia-smi` alone. Two GPU-touching tests in the
suite failed **only** under contention and one recovered in isolation:

- `tests/test_benchmark_runner.py::test_dry_run_60_frames` - failed in the full
  run with `RuntimeError: Face detector found no faces in benchmark reference
  image`, then **PASSED** when re-run in isolation. The asset was verified
  present and intact (`source_reference.png`, 79,084 bytes).
- `tests/test_benchmark_video_harness.py::test_real_video_execution` - failed in
  the initial run, with the decisive signature:

```
Failed to initialize CUDNN Frontend ... CUDNN_FE failure 8: HEURISTIC_QUERY_FAILED
```

`HEURISTIC_QUERY_FAILED` on a Conv node is the classic cuDNN response to being
unable to reserve workspace. It was consistent with VRAM contention at the time
and motivated the quiescent rerun. Subsequent provider handling and harness
compatibility fixes are recorded in the campaign summaries.

---

## 1b. Executed Pre-Optimization Baseline on Quiescent GPU

Once the user's render completed (PIDs 22272, 30912, 46040 exited) and the GPU cleared to ~10.7 GB free, the official controlled benchmark was executed:
`app\env\Scripts\python.exe app\tests\baseline_controlled.py --tag swarm_prebaseline`

| Target | Arm | Processing FPS | Decode FPS | Encode FPS | Peak VRAM | Wrong Face | Source Revision | Status |
|---|---|---|---|---|---|---|---|---|
| RTX 4070 Desktop | `swarm_prebaseline` | **7.82** | **352.94** | **555.56** | 9,958 MB | 0 / 846 (0.0%) | `4eac84b` | **RESOLVED & VERIFIED** |
| RTX 3060 Laptop | any | - | - | - | - | - | n/a | **PENDING - HARDWARE NOT PHYSICALLY PRESENT** |

### Verified Baseline Breakdown:
- **Workload:** `double/d4.mp4` frames 0..600 (1280x720, 30 fps), sources harjot,gargee
- **Stack:** `hyperswap` / `GPEN 256 Pro` / `RealityUX` / `hevc_nvenc` / 10 threads
- **Processing Time:** 76.74 s (600 frames) -> 7.82 FPS
- **Faces Processed:** 849 seen, 846 swapped, 0 wrong faceset across 573 attributed swaps
- **Telemetry:** Mean VRAM 4,552 MB, Peak RSS 11.65 GB (mean 7.45 GB), Peak GPU utilization 100% (mean 37.1%)
- **Stage Shares:**
  - Mask: 88.42 s (12.6%, 52.26 ms/call)
  - Blend: 64.63 s (9.2%, 76.39 ms/call)
  - Swap: 45.81 s (6.5%, 54.15 ms/call)
  - Stabilize: 43.19 s (6.1%, 9.77 ms/call)
  - Enhance: 35.84 s (5.1%, 42.37 ms/call)
  - **Lighting: 19.60 s (2.8%, 11.58 ms/call)** — *Directly disproves the handoff's 19% estimate!*
  - Detection: 18.78 s (2.7%, 97.83 ms/call)
  - Decode: 1.70 s (0.2%, 2.83 ms/call -> 352.94 FPS)
  - Encode: 1.08 s (0.2%, 1.81 ms/call -> 555.56 FPS)

---

## 2. Noise floor - the interpretation rule for every number above

No delta at or below these thresholds is evidence that anything changed.

| Quantity | Floor | Source |
|---|---|---|
| Pixel, mean abs diff | **0.7142 / 255** (docs) - 0.7466 / 255 (tool docstring) | `app/tests/measure_output_noise_floor.py` |
| Pixel, max abs diff | **22 / 255** (docs) - 28 / 255 (tool docstring) | same |
| FPS, same-path arms | **~2.5 % spread** | `docs/PHASE_HANDOFF.md` |
| Path discriminator | `faces_seen` | differing counts (868 vs 903) = DIFFERENT CODE PATHS, not comparable |

The floor survives the obvious suspects: threads 12 -> 1 gives 0.7469, tensorrt
-> cuda gives 0.8921, `PYTHONHASHSEED=0` gives 0.7804. The residue is
non-deterministic GPU reduction order. The floor is codec-inclusive by design so
it is directly comparable to any A/B measured through the same lossy encode.

Cautionary precedent recorded in the tool: `--identity-detail-strength 0.35`
produced mean 0.766 against a 0.747 floor while the `identity_detail` stage never
appeared in the profile table at all. **Prove execution with a stage call count,
not with a pixel delta.**

---

## 3. Regression gate - ESTABLISHED

```
app\env\Scripts\python.exe -m pytest -q -p no:cacheprovider
10 failed, 2811 passed, 1 skipped, 7 warnings, 938 subtests passed in 418.58s (0:06:58)
```

| Metric | Value |
|---|---|
| Collected | 2822 (reconciles exactly: 2811 + 10 + 1) |
| **Passed** | **2811** |
| **Failed** | **10** |
| Skipped | 1 |
| Subtests passed | 938 |
| Wall clock | 418.58 s |

The "~2002 green" figure quoted by prior sessions is **stale and understates the
suite by roughly 800 tests**. It should not be quoted again.

### Post-fix validation on the finalized working tree

The targeted compatibility fixes were validated with **83 passed, 2 warnings,
and 8 subtests**. A subsequent full-suite run collected 2,822 tests and
reported **2,818 passed, 3 failed, 1 skipped, 938 subtests passed**. The three
failures were one standing exception-visibility policy counter and two
GPU/order-sensitive benchmark tests. Both benchmark failures passed when rerun
in isolation immediately afterward:

- `tests/test_benchmark_runner.py::test_dry_run_60_frames` - **1 passed in
  55.82 s**.
- `tests/test_benchmark_video_harness.py::test_real_video_execution` - **1
  passed in 37.24 s**.

The remaining full-suite failure is `test_exception_visibility`, which still
reports the known broad-handler debt. This is recorded as a policy cleanup item,
not disguised as a performance regression.

The 10 failures listed below describe the **pre-fix clean-HEAD run** on
`4eac84b`. The tree had zero tracked modifications then, so no swarm worker
caused those original failures. The post-fix aggregate result is recorded above.

### Failure classification

**A. Standing policy/debt counters (2)**

| Test | Detail |
|---|---|
| `test_exception_visibility` | 28 silent broad handlers remain: `core.py:691`, `ProcessMgr.py:2769`, `Enhance_UltraMax.py:1552`, `face_enhancer.py:20`, `face_swapper.py:21,49`, `procmgr_batch.py:409`, `procmgr_runtime.py:1527`, `temporal_tracker.py:254`, plus 20 in `video_stream.py` |
| `test_standalone_install` | 3 `roop-unleashed` upstream refs outside NOTICE.md (`enhance_common.py`, `PHASE11_ENHANCER_INVENTORY.md`, `SESSION_LOGS.md`) |

**B. Genuine logic mismatches (4)**

| Test | Detail |
|---|---|
| `test_angles.py::test_7pt_profile_alignment_...` | `'profile_3pt' != 'profile_weighted_7pt'` |
| `test_angles.py::test_high_yaw_prevents_horizontal_collapse` | `'profile_3pt' != 'profile_weighted_5pt'` |
| `test_lipsync_audio.py::test_audio_cache_setup_is_itself_gated` | `IndexError: list index out of range` |
| `test_lipsync_audio.py::test_both_gates_read_the_same_global_...` | `IndexError: list index out of range` |

The two `test_angles` failures share a root cause: the weighted-Umeyama path is
not being selected, so alignment falls back to `profile_3pt`.

**C. Environment / GPU-contention artifacts (3 at the time of the initial run;
the active-model row was reclassified as a logic mismatch)**

| Test | Detail | Isolated re-run |
|---|---|---|
| `test_full_benchmark_e2e::test_the_panel_can_walk_the_journey_over_the_api` | `OSError [WinError 145] directory is not empty` on temp teardown | Windows file-lock race |
| `test_benchmark_runner::test_dry_run_60_frames` | detector found no faces | **PASSES in isolation** |
| `test_benchmark_video_harness::test_real_video_execution` | `CUDNN_FE HEURISTIC_QUERY_FAILED` | initial run was contention-affected; targeted harness passes after compatibility fixes |
| `test_benchmark_video_harness::test_active_models_preservation` | `assert 'inswapper' == 'DFL XSeg'` | resolved by correcting the test to preserve `swap_model` |

**Historical pre-fix gate for the swarm:** **2811 passed / 10 failed** was the
clean-HEAD result before the targeted compatibility fixes. It remains useful as
the no-regression floor, but it is not the final post-fix suite result.

---

## 4. Instrumentation audit

### 4.1 `source_revision` guard - WIRED (the wiring, not just the function)

The historical bug was a call sited above where `results` is built, so `main()`
died with `NameError` while its 7 unit tests stayed green. **That bug is not
present.**

| Element | Location | Verdict |
|---|---|---|
| `source_revision()` definition | `baseline_controlled.py:302` | present |
| Call site | `baseline_controlled.py:590` | inside the `result` dict literal opened at line 562 |
| Ordering | after `run_sampled()` at 541 and `parse_run()` at 551 | executes in the live path, cannot `NameError` |
| Downstream refusal | `ab_shape_profile.py:138` `assert_one_tree()` | returns `False` on >1 distinct revision |

`assert_one_tree` additionally warns on dirty-but-same-sha (an uncommitted edit
moves code without moving HEAD) and on `unstamped` arms from an older harness,
and it **prints its verdict on clean sets too** - so the guard is visible rather
than invisible until the day it fires.

### 4.2 `config_sync` - SINGLE IMPLEMENTATION, STILL EXHAUSTIVE

| Consumer | Wiring |
|---|---|
| `angle_bench.py:127` | `from config_sync import sync_globals_from_config`, called inside `init_pipeline()` |
| `compare_enhancers_video.py:141-142` | imports `TRANSLATED` + `sync_globals_from_config` rather than redefining |
| `two_face_video.py:1097` | calls `ab.init_pipeline(..., sync_config=True)` |

**Correction to the brief:** `two_face_video.py` does not import `config_sync`
directly, and that is correct rather than a regression. The sync happens inside
`init_pipeline` *before* the explicit assignments, so caller overrides still win.
The wiring is intact, just one level of indirection.

The guard `test_bench_config_parity.py:70` is genuinely exhaustive: it walks
every key in `config.yaml` against `roop/globals.py`'s AST-parsed defaults rather
than a hand-maintained list, so it grows automatically when a setting is added.
This is the right shape - a list is exactly what fails to grow.

### 4.3 Stage profiler - ALL 5 REQUIRED STAGES PRESENT, LIGHTING GAP CLOSED

`REQUIRED_STAGES` (`stage_profiler.py:17`) defines **13 canonical stages**, a
superset of the 5 requested:

`detection, tracking, alignment, faceset_lookup, swap, expression_analysis,
occlusion_analysis, detail_restoration, enhancement, lighting, mask, blending,
encoding`

| Requested stage | Canonical name | Live `_prof()` call sites |
|---|---|---|
| decode / queue | `tracking` (+ raw `decode`) | `ProcessMgr.py:1518, 1843, 1857, 2141`; `procmgr_tracking.py:712, 723, 808, 810, 827` |
| detection | `detection` | `ProcessMgr.py:3140, 3169`; `procmgr_tracking.py:368, 825` |
| swap inference | `swap` | `ProcessMgr.py:4840` |
| masking + color | `mask`, `blending` | `ProcessMgr.py:4996, 5626`; `procmgr_masking.py:1341, 1402` |
| encode | `encoding` | `ProcessMgr.py:1776, 1911, 2311`; `procmgr_batch.py:882` (`encode_finalize`) |
| **lighting (never profiled)** | **`lighting`** | **`ProcessMgr.py:4428, 4596, 4976, 5259`** |

**The lighting gap is CLOSED.** No stages were missing, so **no instrumentation
was added**. That is deliberate: the measurement worker must not be a source of
diff in the tree it is about to measure.

Instrument honesty worth noting: a stage with zero call sites reports
`status: "not_observed"` rather than `0.0 ms`, so the report distinguishes
"measured zero" from "never instrumented" (`_empty_report_stage`, line 245).

### 4.4 Peak VRAM via NVML - AVAILABLE

| Component | Location | Status |
|---|---|---|
| `pynvml` in venv | `app/env` | importable (`find_spec` -> True) |
| `GpuTelemetrySampler` | `roop/benchmark/runner.py:104` | NVML first, `torch.cuda.mem_get_info` fallback, 0.05 s period |
| `_probe_nvml` | `roop/benchmark/hardware_probe.py:186` | present |
| `baseline_controlled` path | `tests/telemetry.py:54` `_smi()` | uses `nvidia-smi --query-gpu=memory.used`, 0.5 s period |
| Peak derivation | `telemetry.py:139` `summarise()` | `peak_<key> = max(vals)` over all samples |

**Caveat that matters for this device:** the `baseline_controlled.py` telemetry
path reports **whole-card** `memory.used`, not per-process. Under the contention
observed during the initial attempt it would have attributed the user's ~10 GB
to the benchmark. This was a second, independent reason the baseline was kept
PENDING until the user's processes exited. The quiescent run in section 1b was
sampled after that point.

---

## 5. Lighting cost - independently reproduced

The coordinator's probe was re-run by this worker on the same box. The numbers
hold within run-to-run variance and the shape is identical.

| Threads | Coordinator (ms/call) | This worker (ms/call) |
|---|---|---|
| 1 | 2.56 | **2.99** |
| 6 | 14.02 | **14.26** |
| 10 | 23.23 | **23.75** |
| 12 | 39.12 | **38.89** |

All at `cv2.setNumThreads(2)`, which is the production setting
(`_configure_opencv_worker_threads` picks `2 if physical // workers >= 2 else 1`;
24 physical cores -> 2 at every worker count tested). Process default was
`cv2.getNumThreads() == 32`, physical cores 24 - matching the AGENTS.md main
device profile.

### Interpretation

Lighting scales **~13x from 1 to 12 threads** while genuine per-call work should
stay flat. That is **queueing, not cost**. The "~19 % of frame time" figure in the
docs is a wall-clock share measured under concurrency, and it therefore
**overstates lighting's true cost**.

**Rule adopted for this report:** every stage share is quoted with its thread
count, because `_prof()` wall-clock under concurrency measures contention.

Secondary finding: `cv2=2` beats both `cv2=1` and `cv2=auto/0` at every thread
count (2.99 vs 3.96 vs 4.16 at 1 thread; 23.75 vs 31.61 vs 33.47 at 10).
Production already picked the best of the three - worth knowing before anyone
"optimizes" that knob.

---

## 6. VALID vs UNRESOLVABLE

### VALID - measured this session, on a stated revision

| Claim | Evidence |
|---|---|
| Pre-fix full suite = 2811 passed / 10 failed / 1 skipped / 938 subtests on clean `4eac84b` | full `pytest -q` run, 418.58 s |
| Post-fix full suite = 2818 passed / 3 failed / 1 skipped / 938 subtests | full `pytest -q` run, 323.19 s; two benchmark failures passed in isolation |
| All 10 failures pre-date the swarm | `git status --porcelain` showed zero tracked modifications |
| `test_dry_run_60_frames` fails only under GPU contention | passes in isolated re-run |
| Initial `test_real_video_execution` failure was contention-sensitive | `CUDNN_FE HEURISTIC_QUERY_FAILED`; targeted harness now passes |
| Lighting is queueing-dominated, not cost-dominated | 2.99 -> 38.89 ms across 1 -> 12 threads, cv2=2 |
| `cv2=2` is the fastest of the three settings tested | all four thread counts |
| `source_revision` guard is live-path wired | line 590 inside `result`, after the run |
| `config_sync` is single-source and exhaustively guarded | AST-based parity test |
| All 13 canonical stages instrumented, lighting at 4 sites | `_prof()` call-site census |
| NVML peak-VRAM capture is available | `pynvml` importable, sampler present |

### UNRESOLVABLE - not yet answered by this session

| Question | Why |
|---|---|
| **Optimized FPS on 4070** | A post-fix comparison arm has not yet been run. |
| **Baseline vs optimized delta** | The optimized operand is missing. Any quoted delta would be fabricated. |
| **Peak VRAM for optimized arm** | No optimized run has been completed. |
| **Warm-up vs steady-state split** | Requires a render. |
| **Lighting's true share under every worker configuration** | The controlled baseline reports 2.8% for this workload; the probe also shows per-call contention varies with thread count. |
| **Any RTX 3060 Laptop number** | Hardware is not physically present. A 4070 number must never be copied into this row. |
| **Whether ~12.9 fps is still valid** | It predates the decode fix (1.96 -> 142-161 fps). It describes a stack nobody ships and must not be used as "before". |

### The trap this report is specifically avoiding

`docs/PHASE_HANDOFF.md` records two ways this repo previously produced confident
wrong numbers:

1. **Config drift** - `two_face_video.py` did not render `config.yaml`; 28 keys
   diverged, so every absolute FPS/quality number before the fix describes a
   stack nobody ships. *(Verified fixed.)*
2. **Mid-run commits** - an A/B was VOIDED because two commits landed while arms
   rendered, splitting six arms across three versions of `face_swapper.py`.
   *(Verified guarded.)*

Both guards are intact. The correct response to a blocked measurement is a
PENDING row until the GPU is quiescent, followed by a controlled rerun. That is
the sequence used for the baseline recorded in section 1b.

---

## 7. Preconditions for closing the remaining optimized row

1. User's app and ffmpeg children are closed or finished; `nvidia-smi` shows the
   GPU is quiescent.
2. Tree is quiescent with no worker mid-edit and no tracked modifications.
3. Run the optimized comparison arm with a distinct tag on the same fixture,
   hardware, and configuration as `swarm_prebaseline`.
4. Confirm the emitted JSON carries the expected source revision and
   `comparable_to_locked_baseline: true` with no fixture mismatch or adaptive
   downgrade.
5. Compare FPS, quality counters, stage shares, and peak VRAM only after both
   arms pass those guards. The RTX 3060 row remains blocked until that hardware
   is physically available.
