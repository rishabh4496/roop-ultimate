# Handoff - shape-profile A/B: CLOSED, no measurable effect (2026-09-03, final)

Device: RTX 4070, driver 616.56, TRT 10.9.0.34, ORT 1.23.2, i9-14900K, 31.7 GB.
Commits `d542608`, `c264e86`, `4ec1f0f`, `66d754e`, `91c9d48`. Suite **2002
green**, 1 skipped. Supersedes the VOID run described below; the guard that
distinguishes the two is now in the harness.

## The result

Re-run on a quiescent tree, every arm verified to share one revision:

    [ab] source revision per arm
      profile0_rep0 / rep1 / profile1_rep0 / rep1     91c9d481f  (all four)

    profile=OFF   n=2  mean 8.32   min 8.30  max 8.34   spread 0.5%
    profile=ON    n=2  mean 8.23   min 8.13  max 8.34   spread 2.6%
    profile ON vs OFF: -1.0%

**No measurable effect. `ROOP_TRT_SHAPE_PROFILE` stays at its default of 1.**

The bound is much tighter than the earlier attempt allowed. All four measured
arms report `faces_seen` 903 -- the same code path -- and spread **2.5%**
between them. The effect sits inside that, so if the profile does anything on
this pipeline it is smaller than this rig resolves at 600 frames. That is a
real closure, not the "33% noise, cannot tell" non-answer the contaminated run
produced.

**The 7.6% null-control spread is NOT the noise floor, and `faces_seen` says
why.** `null_0` reports **868** faces seen against 903-904 everywhere else, so
the two "identical config" null arms took different PATHS -- the documented
discriminator (a parallel block re-processes its warm-up frames; a sequential
pass does not). Its 7.87 fps is a different execution path, not a slow repeat
of the same one. Read the four same-path arms for resolution, not the null
pair.

Mechanism agrees with the null result: detection is ~4.5% of frame time here,
the pipeline feeds one `det_size` for a whole render so TensorRT already sees a
stable shape after frame one, and this project has measured several
stage-level wins that were neutral end to end. Nothing argued the effect should
be nonzero.

**If the profile is ever removed, remove it for cache-namespace cost** -- each
profiled model gets its own engine directory, which orphaned 886 MB once
already -- **not for -1.0%.**

## Decode, confirmed end to end

`decode_fps` **142.5-161.7** across all six arms, against **1.96** before
`d542608`. `decode` and `track_decode`, previously the two largest stages at
55% of wall clock combined, no longer appear among the top stages. Throughput
on the locked fixture went 0.96 -> 8.13-8.49 fps.

## The guard that made this run trustworthy

`baseline_controlled.py` now stamps `source_revision` (HEAD, dirty, dirty .py
paths) into every arm JSON, and `ab_shape_profile.py` refuses to quote a delta
when arms disagree. It reported honestly here: all four arms on `91c9d481f`,
flagged **dirty** -- another session's uncommitted `core.py` /
`face_swapper.py` edits were present, shared by every arm. Internally valid,
not reproducible from a commit; that limitation is recorded rather than hidden.

Note the guard shipped broken in `66d754e`: the call sat above where `results`
is built, so `main()` died with `NameError` before rendering a frame while its
seven unit tests stayed green. Testing a function is not testing its wiring.
`TestGuardIsActuallyWiredIntoMain` now runs `main()` with `run_arm` stubbed and
is verified to fail with the original `NameError` when the misplacement is
reintroduced.

## NEXT

1. **`lighting` is the largest stage after the swap** (~19% of frame time,
   32-37 ms/call, ~4.5 calls/frame). Hidden under decode until now; never
   profiled. This is where per-face work reduction should look.
2. **Re-baseline the locked fixture** -- every end-to-end number on this device
   predates the decode fix, including the ~12.9 fps figure in
   `RECODE_STATUS.md`.
3. **Nothing here is measured on the 3060**, which had the same stall and is
   host-RAM constrained; the fix removes a per-frame pinned allocation, so the
   effect there may differ in kind rather than size.
4. **Commit or stash before benchmarking** -- this run was internally valid but
   dirty. The guard will keep saying so.

---

# Handoff - shape-profile A/B VOID (contaminated), and the decode stall it found (2026-09-03, later)

## CORRECTION, written minutes after the section below

**The A/B arms did not all run the same code, so the run is VOID -- not merely
unresolvable.** Two feature commits landed in this working tree WHILE the arms
were rendering, both touching `app/roop/processors/frame/face_swapper.py`, the
hot path of every arm:

    13:48:54  null_0         done   <- original face_swapper.py
    13:49:36  7da4d08 feat(nvof)      +238 lines   *** MID-RUN ***
    13:52:30  null_1         done   <- import raced the commit (ambiguous)
    13:55:58  profile1_rep0  done   <- NVOF present
    13:59:00  b084915 feat(lighting)  +245 lines   *** MID-RUN ***
    13:59:19  profile0_rep0  done   <- started pre-relighting
    14:02:53  profile0_rep1  done   <- NVOF + relighting
    14:06:28  profile1_rep1  done   <- NVOF + relighting

Each arm is its own process and loads the module at start, so the six arms are
split across THREE versions of the swapper. Counterbalancing assumes only the
treatment varies; here the largest variable was the tree.

**What this retracts.** The section below reads the 33.4% null-control spread as
the rig's resolution. It is not: `null_0` and `null_1` were "identical config"
arms that ran DIFFERENT CODE, and the 5.48 -> 7.68 step sits exactly across the
NVOF commit. The post-fix resolution of this machine is UNMEASURED. The -2.4%
was already being reported as no result and still is, now for a stronger
reason.

**What survives, and why.** The decode finding is unaffected. It was measured
at the reader in isolation (2.0 -> 468 fps) by direct instrumentation, before
either commit, and its mechanism is a deterministic `1 / 0.5s` stall rather
than a throughput delta needing statistics. The pre-fix arms all ran on
`b353917` with no commits landing during them, so the 0.96/0.96 null and the
55%-of-wall-clock decode share are clean.

**The rule this actually adds** (the one below about tight nulls stands on the
PRE-fix data and is unchanged): *a benchmark must record the tree it ran on.*
`baseline_controlled.py` should stamp `git rev-parse HEAD` plus dirty state
into every arm JSON and the harness should refuse to summarise arms whose
stamps disagree. Nothing in this repo does that today, which is why six arms
across three code versions summarised without a murmur. This project has been
here before -- the 2026-08-31 4070 session reconciled a concurrent session's
ten commits landing mid-benchmark -- and it was caught that time by noticing,
not by a guard.

**Re-run required** on a quiescent tree before anything is claimed about the
shape profile or about this machine's post-fix resolution.

---

## Original section, retained as written (its null-spread reasoning is retracted above)


Device: physical RTX 4070, driver 616.56, TRT 10.9.0.34, ORT 1.23.2,
i9-14900K, 31.7 GB RAM. Commit `d542608`. Suite 1989 -> **1993 green**, 1
skipped (4 new tests). The A/B below is finished; the section under this one
describes the paused state it resumed from.

## The A/B: NO RESULT. Do not quote the -2.4%.

    profile=OFF   n=2  mean 7.84  min 7.29  max 8.39   spread 14.0%
    profile=ON    n=2  mean 7.65  min 7.52  max 7.79   spread  3.5%
    profile ON vs OFF: -2.4%

    null control  n=2       5.48 / 7.68              spread 33.4%

**The null control spread is 33.4%. The measured effect is 2.4%.** The rig
cannot resolve it, so the sign is not evidence and the feature is NOT being
changed on it. `ROOP_TRT_SHAPE_PROFILE` stays at its default of 1.

The face-count guard passed -- (900,903), (901,904), (902,904) across all six
arms -- so no arm gained speed by finding fewer faces. That is the one thing
this run establishes cleanly.

Position accounts for much of the spread: `null_0` is the first 600-frame arm
of the process at 5.48 while every later arm sits at 7.29-8.39. Excluding it,
five arms spread 14%. Either figure is many times the effect. Resolving 2.4%
here would need roughly 25 arms per side at ~3.5 min each -- about three hours
of rendering for a number with no mechanism arguing it should be nonzero.

**Recommended: close it as unresolvable rather than fund that.** The profile's
real cost is not throughput, it is that each profiled model gets its own engine
cache namespace (see the 886 MB orphaning in the section below); if it is ever
removed, remove it for that reason, not for -2.4%.

## What the A/B actually found: decode ran at 2.0 fps

The first attempt was stopped after four arms because the null control came
back at **0.96 fps** against this fixture's ~12.9 fps baseline. The stage table
said why, and it was not the treatment:

    decode         306.5s  28.3%  510.87 ms/call
    track_decode   297.0s  27.4%  494.98 ms/call
    read_wait  87466 ms on one chunk, against 14.98s of processing at 13.0 FPS

The swap pipeline was running at its normal 13.0 FPS and starving.

**Root cause.** `PinnedBufferPool.acquire()` defaulted to `timeout=0.5`. A
buffer only returns through `release()`, and nothing in the codebase calls it:
`roop/nvdec_reader.py` is the pool's only consumer and its decoded frames
escape to ProcessMgr. So the pool is permanently empty after its first
`capacity` acquires, and every acquire past that point waited the full half
second for a refill that could not arrive -- then allocated a fresh buffer
anyway.

`1 / 0.5s = 2.0 fps`, and that is what it measured. Eliminated in order:

    raw disk read                        1815 MB/s
    plain cv2                             760 fps
    ffmpeg -hwaccel cuda from a shell     ~900 fps  (correct byte count)
    the app's own NVDEC reader            2.0 fps   <- the defect
    the same reader, after the fix        468 fps

End to end, `decode_fps` went **1.96 -> 134.5-158.7**, and `decode` and
`track_decode` -- previously the two largest stages at 55% combined -- dropped
out of the top seven entirely.

**Why nothing caught it.** The frames were CORRECT, only late. rc stayed 0, the
swap audit stayed at 100%, output integrity would have passed, and every stage
was individually healthy. The stall is in the producer, so it surfaced only as
`read_wait` -- a field no gate reads. The suite was green through all of it
because no test asserted that an exhausted pool does not stall.
`tests/test_buffer_pool_acquire.py` now does, and was verified to FAIL on the
pre-fix code (4.9s vs 0.29s).

## THE RULE THIS ADDS: a tight null control can mean a blind one

The pre-fix null control read **0.96 / 0.96, spread 0.0%** -- by a wide margin
the quietest null this project has ever recorded, on a machine documented at
4-8%. It was not quiet. Both arms were pinned behind a constant 0.5 s-per-frame
stall that dominated everything downstream of it, so the rig looked
exquisitely repeatable *and could not have detected any change to the stage
under test*. The same fixture, once decode was fixed, spreads 33.4%.

A null control establishes that a rig is repeatable. It does NOT establish that
the rig is SENSITIVE to the thing being varied. When a null comes back far
tighter than the machine's known noise, treat it as a signal that something
constant is dominating the measurement -- not as licence to believe a small
delta.

(RETRACTED: the "33.4%" quoted just above as the post-fix spread is not the
rig's noise -- those two null arms ran different code, per the correction at
the top of this file. The PRE-fix half of this rule -- 0.96/0.96 at 0.0% spread
while blind -- is measured and stands.)

## Second harness defect, fixed in the same commit

`tests/ab_shape_profile.py` read `fps` / `faces_swapped` / `faces_seen` at the
top level of the arm JSON, while `baseline_controlled.py` nests them under
`run`. Every arm printed `0.00 fps  swapped ?/?`; the zero mean tripped the
delta guard (`if off and on and st.mean(off)`) so **no result would have
printed at all**; and the face-count guard degraded to `[(None, None)]` -- the
check meant to catch an arm that got faster by finding fewer faces. Exit code 0
throughout. Fixed via a `metric()` helper that reads `run` first.

The module docstring's claim that `retinaface_r50` is the only profiled model
is also corrected: the engine cache written by a real render shows three, as
the previous handoff recorded.

## NEXT

1. **`lighting` is now the second-largest stage** -- 87.9-100.3 s, ~19% of
   frame time, 32.6-37.1 ms/call at ~4.5 calls per frame. It was invisible
   underneath decode. Nothing is claimed about it yet; it is simply the largest
   thing left after the swap itself, and it has never been profiled.
2. **Re-baseline the locked fixture.** Every end-to-end number on this device
   predates the decode fix. The ~12.9 fps baseline in `RECODE_STATUS.md` was
   measured when the stall was present, so it is not the ceiling either.
3. **Nothing here is measured on the 3060**, where the same stall was present
   and host RAM is the constrained resource -- the fix removes a per-frame
   pinned allocation, so the effect there may differ in kind, not just size.
4. Inherited and untouched: Phase 3's RSS gate still fails on the 3060 at
   3.73 GB; interacting faces remains characterized but unsolved.

---

# Handoff - execution providers (2026-09-03, PAUSED mid-benchmark)

Device: physical RTX 4070, driver 616.56, TRT 10.9.0.34, ORT 1.23.2,
i9-14900K, 31.7 GB RAM. Commits `97d562a`, `32b2ef7`, `fe91dee`.
Suite 1986 -> 1989 green (29 provider tests, no skips).

## State: SHIPPED code, UNFINISHED measurement

Three things landed and are covered by tests. One measurement was started,
found a bug in the code under test, and was stopped before producing numbers.

**The end-to-end A/B has NO result yet.** Nothing about throughput is claimed.

## Resume here

    env/Scripts/python.exe tests/ab_shape_profile.py --reps 2 --end 600 --null

Process per arm, both engine caches warmed first, counterbalanced, face counts
printed as the guard. Expect ~35 min: the warm-up ON arm must build the three
profiled engines (they exist only under the orphaned namespace below), the rest
are cached.

Read `faces_swapped/faces_seen` beside fps in every arm. Identical counts are
the guard -- an arm that goes faster by finding fewer faces has not got faster.

### What the aborted run already established

* The profile IS applied in a real render, not just in unit tests: TensorRT
  wrote a 57 MB engine plus a `.profile` sidecar into the production namespace
  `..._sp12x512input8x3x1280x1280`.
* **THREE models get a profile in the live stack, not one.** The render created
  `_sp12x512input8x3x1280x1280` (retinaface_r50), `_spet8x3x256x256source8x512`
  (hififace, the second net inside realswap -- batch-dynamic) and
  `_spx512x512input8x3x512x512`. The commit message for `97d562a` says the
  detector is the only profiled model in the render path; that was true of the
  models sampled by hand and is WRONG for the running stack. Correct it when
  the A/B is written up.
* The 120-frame warm-up arm rendered correctly: 120 frames, 159 swaps,
  `frame_total` 21.71 s. Everything else in its 1813 s wall clock was the
  engine rebuild described below.

## The bug the benchmark found (fixed in `fe91dee`)

`97d562a` added `fp16_enable` and `fp8_eligible` to `builder_config`, which is
hashed into the TensorRT engine cache directory name. Both are constant on
every supported card, so the namespace moved
`_c3b1a9752fee69034` -> `_cc3a4d61f058c77bc` and orphaned every engine.
Cost, measured from the arm's own log: **27 minutes of rebuilding before the
first frame**, swapper and enhancer included.

The capability gate is now recorded only when it actually changed the outcome.
Namespace verified back to the pre-existing `_c40b4a7d8494df527`.

**A cache-identity field is not free.** One that cannot vary is
indistinguishable from a correct invalidation: nothing errors, nothing warns,
the app spends half an hour rebuilding and then runs normally.

## Housekeeping decision owed

`models/trt_cache/*cc3a4d61f058c77bc*` -- **886 MB** of engines built under the
bad digest across four directories. Regenerable and now unreachable. Left in
place rather than deleted without asking. Safe to remove.

## OpenVINO offload: validated, and it fails OPEN

Validated against a real `OpenVINOExecutionProvider` in an isolated venv
(onnxruntime-openvino 1.23.0 + openvino 2025.3.0). That wheel conflicts with
`onnxruntime-gpu`, so it must NEVER be installed into `app/env` -- it would
take CUDA/TensorRT down. Throwaway venv only.

Asking for a device the machine lacks does not raise:

    device_type=CPU    build 0.4s  EP active   -> ran on OpenVINO
    device_type=GPU    build 0.3s  EP ABSENT   -> ran on CPUExecutionProvider
    device_type=NPU    build 0.3s  EP ABSENT   -> ran on CPUExecutionProvider
    device_type=GPU.1  build 0.4s  EP ABSENT   -> ran on CPUExecutionProvider

Working session, no exception, ORT logs to stderr and drops the provider. A
mismatched OpenVINO runtime does the same (errors 126 then 127, both silent).
So `build_session_with_fallback` cannot see it and `provider_available()` --
which only asks whether the EP is LISTED -- would report detection as offloaded
while it ran on CPU. `openvino_device_usable()` now probes a one-node graph and
asks the CONSTRUCTED session what it got.

This machine: i9-14900K, **no NPU, no Intel iGPU exposed**, so `auto` correctly
declines. OpenVINO 2026.3 enumerates the RTX 4070 itself as `GPU`
(`FULL_DEVICE_NAME` "NVIDIA GeForce RTX 4070 (dGPU)"), which is why `auto`
skips GPU.0 -- written as a guess in `97d562a`, now a measured decision.

Still owed: a machine with a real NPU or Intel iGPU. Detection feeds every
identity gate, so the quality question is untouched by anything here.

---

# Phase Handoff - final validation campaign, RTX 4070 (2026-09-01, later)

Device: physical RTX 4070, driver 616.56, TensorRT 10.9.0.34, ORT 1.23.2.
Full evidence: `docs/FINAL_VALIDATION_MATRIX.md`, section
`# RTX 4070 campaign (2026-09-01)`.

## Read this first - it changes how to read every earlier benchmark here

**`two_face_video.py` did not render the config the user runs.** It is the
end-to-end harness `baseline_controlled.py` and the whole Phase/Gate campaign
go through, and it inherited `angle_bench.init_pipeline`'s "state every setting
explicitly" semantics -- right for an angle A/B, wrong for end-to-end. 28 keys
diverged from `config.yaml`. The ones no harness set anywhere:

    target_conditioned_appearance   False  vs  True     <- LIVE feature, off
    detail_transfer_strength          0.0  vs  0.4      <- whole path dead
    color_match_after_enhance       False  vs  True
    codeformer_fidelity               0.5  vs  0.55
    parser_regions                   None  vs  the five configured regions

**A/B arms remain valid** -- both sides were equally off -- but no absolute
value from this harness was production, and any quality grading against real
footage measured a stack nobody ships. Fixed by
`angle_bench.init_pipeline(sync_config=True)` over the shared
`tests/config_sync.py`; guarded by `tests/test_bench_config_parity.py`, which
was verified to fail on the pre-fix state on all 28 keys.

Because silence now means "the user's config", an A/B can no longer express
"off" by omitting a flag. Each config-backed toggle gained an explicit negative
(`--no-target-conditioned-appearance` and friends).

## The instrument to use before believing any pixel comparison

`tests/measure_output_noise_floor.py`. Two renders of one unchanged
configuration on this pipeline differ on EVERY frame:

    mean 0.7142/255   max 22/255   (three pairs, agreeing to 0.4%)

and the floor survives threads 12 -> 1 (0.7469), tensorrt -> cuda (0.8921) and
`PYTHONHASHSEED=0` (0.7804). Frame 0 differs at one worker; detected boxes are
identical while identity cosines are not. It is non-deterministic GPU reduction
order, not scheduling, tactics or hash order.

**A pixel delta at or below ~0.71/255 mean is not evidence a feature ran.**
Prove execution from a `ROOP_PROFILE` stage call count instead. That is not
hypothetical: `--identity-detail-strength 0.35` measured 0.766 against this
floor while the `identity_detail` stage never appeared in the profile at all.

The floor was also used as a regression check, which is the intended second
use: after editing three except handlers in `procmgr_masking.paste_upscale`, a
production render sat at 0.7158 / 0.7175 / 0.7209 against the three pre-change
renders -- inside the floor, so the default path is unchanged.

## What was fixed

| # | defect | status |
|---|---|---|
| 1 | the end-to-end harness rendered module defaults, not `config.yaml` | FIXED |
| 2 | identity detail restored nothing on V1 facesets, silently | FIXED (reported once per cause) |
| 3 | the adaptive enhancer restored nothing on 60 of 60 faces, silently, as the fastest arm of the sweep | FIXED (visibility only) |
| 4 | `faceset_mean` was not format-neutral -- the 3060 campaign's D.9, carried as FOUND NOT FIXED | FIXED |
| 5 | adaptive fallback printed per face (120,000 lines on a long render) | FIXED |
| 6 | an absent `quality` entered the adaptive band as 0.0 | FIXED |
| 7 | three compositing/occlusion quality layers fell back to the legacy path in silence | FIXED (reported once per cause) |

## What was verified, with evidence

* **all 14 selectable enhancers execute** end to end, one `enhance` call per
  swapped face, zero wrong-faceset. DMDNet works on this card, so the inherited
  "DMDNet is broken" is 3060-specific.
* **single-image swap** (`tests/image_swap_smoke.py`): identity to source
  0.05 -> 0.67 on every graded frame, with a `--control` arm that must fail and
  does.
* **the application boots**: `/api/meta`, `/api/settings`, `/api/progress`,
  `/api/system/telemetry` all 200, no private underscore keys leaked, clean
  shutdown with VRAM released.
* **no host memory leak**: 5,979-frame `double/d3.mp4` render, 289 samples, peak
  15.26 GB, quarter means 14.68 / 14.78 / 14.79 / 13.07 -- flat then falling.
* **interacting faces on `double/d3.mp4`**: 17 wrong-faceset applications of
  2,952 attributable swaps (0.58%). Recorded as a NEW baseline -- the
  2026-08-23 audit's 10 was `duo/d3.mp4`, a different clip sharing the filename,
  and `duo/` does not exist on this machine.

## Exact starting point for the next session

1. **Re-baseline everything through `two_face_video.py`.** Every absolute number
   taken through it predates the config sync and was measured on a different
   stack from production. A/B ratios survive; absolute FPS, identity and quality
   values do not.
2. **Decide the adaptive enhancer's thresholds, with the missing half.** The
   distribution is now measured (d4: 0.7665 / 0.7994 / 0.8188 against a 0.68
   cut; d6: 0.42-0.47, where it engages). What is NOT measured is whether
   restoring a 0.80-quality face improves it. Do that comparison before moving
   any cut -- four gate changes here were implemented and reverted for exactly
   this gap.
3. **`track matched but has no source` is 10.0% of faces on d3**, the largest
   single refusal class on that clip. The project already characterises this as
   intake rather than gating: a track whose best frames never entered the source
   bank binds to nothing. Untouched by this campaign.
4. **33.1% of d3's swaps had INTERPOLATED landmarks** and therefore bypassed the
   identity gates. Unquantified as a quality risk.
5. **Twelve settings have no UI**: `identity_detail_strength`,
   `temporal_compositing_*` (7), `temporal_quality_*` (4). They are in
   `settings.py` and `api.py` but in no React control. Deliberately NOT added
   here: their own handoffs record them as OPEN/INCOMPLETE, and exposing an
   unvalidated feature in the main panel presents it as ready.
6. **Nothing in this campaign is measured on the RTX 3060.** All seven fixes are
   device-independent code, but the enhancer sweep, the noise floor, the image
   smoke and the d3 baseline are 4070 numbers only.

## Do not break

Everything in the previous section's "Do not break" list, plus:
`tests/config_sync.py` is now the single implementation of the config sync --
`compare_enhancers_video.sync_globals_from_config` re-exports it, and a second
copy is how this defect happened in the first place.

---

# Phase Handoff - Phase 11 Temporal Identity Per-Face Cost

Date: 2026-09-01
Device: physical RTX 4070, driver 616.56, TensorRT 10.9.0.34, ORT 1.23.2

## Read this first

Three things from this session change how you should read any benchmark here.

1. **A regression in `1c0efd7` had disabled the swap on the shipped default
   path** (Phase 9). Fixed in `da30500`. Any render made between those two
   commits swapped almost nothing while returning 0, reporting `100.0%` in the
   swap audit, and running *faster* than the correct pipeline. Discard them.
2. **This machine drifted 2.9x during Phase 10, on the unchanged default
   configuration.** The null control read 12.91, then 4.5, then 8.58 fps with
   nothing changed. Host RAM available moved 14.1 -> 4.5 -> 15.0 GB, and
   `_default_stab_chunk_mb` is `available * 0.40 / 6`, so the block geometry --
   and therefore which code path a run takes -- depends on free memory at start.
   **Run a null control per measurement window, not per session, and record
   `faces_seen` beside fps.**

3. **The locked 1280x720 `double/d4.mp4` fixture is outside this repository.**
   It is available at `G:/pinokio/roop-keep/double/d4.mp4` on the 4070 host and
   was used for three Phase 11 feature-level renders. The 3060 still has no
   Phase 11 run.

## Current state

| item | state |
|---|---|
| Phases 0-5 | VERIFIED, untouched |
| Phase 6 (pose/source harness) | implemented, committed |
| Phase 6B `ROOP_TEMPORAL_IDENTITY` | opt-in; **now runs at width**, quality still unvalidated |
| Phase 7 `ROOP_TEMPORAL_OCCLUSION` | opt-in; falls back to sequential by design, quality unvalidated |
| Phase 8 `ROOP_TEMPORAL_EXPRESSION` | opt-in; measured, recommended for promotion, still default off |
| Phase 9 | the dedent fix + first real temporal measurements |
| Phase 10 | parallel-block execution for identity/occlusion; measured |
| Phase 11 | identity per-face cost reduction; **OPEN / INCOMPLETE** |

All three temporal flags remain **disabled by default**. No saved user
configuration, `.fsz` format, model, provider policy, pool setting or look value
was changed.

## What changed in Phase 10

`ProcessMgr.py:1671` used to set `threads = 1` whenever identity or occlusion was
on. Phase 9 measured that at -62.9%, and proved with a `threads=1` no-flag
control that the pinning was the entire cost.

Ordered is not the same as serial. `_run_stab_parallel` already gives each worker
a contiguous block, in frame order, with its own filter instances and a warm-up
it discards. These engines now ride that path:

- `warmup_frames(eps)` on both engines, derived from their own recurrences
  (identity 15 frames, occlusion 44), so `_stab_warmup_frames` picks them up.
- `clone_for_block()` on both. Identity carries the pre-pass-derived identity /
  pose / source state (read-only during the swap) and clears only the three
  fields the swap phase mutates. Occlusion carries nothing.
- `ProcessMgr._temporal_engine(name)` -- every mutating site reads through it.
- `set_ordered` now asks "is this worker seeing frames in order", true on the
  sequential loop and inside a block. It used to be `not _parallel_stab`, which
  made the occlusion engine return `disabled` for every frame of a parallel run.
- `_stab_min_block_multiple` (1 by default = no-op; 3 for these engines) floors
  the block at 3x the warm-up, because at a 1:1 ratio the priming costs more than
  the extra workers return.

## Evidence

Locked fixture, counterbalanced ABBA, every arm's path verified from
`faces_seen` (679 = one sequential pass, >750 = parallel re-processing warm-up):

| arm | position | fps | path | wrong faceset |
|---|---:|---:|---|---:|
| NEW | 1 | 8.55 | parallel | 0 |
| OLD | 2 | 4.41 | sequential | 0 |
| OLD | 3 | 4.43 | sequential | 0 |
| NEW | 4 | 7.07 | parallel | 0 |

**NEW 7.81, OLD 4.42, +76.7%.** OLD arms agree to 0.5%; the worst NEW arm beats
the best OLD arm by 60%.

Output equivalence, sequential vs parallel, 600 frames: mean absolute difference
**0.35% of full scale**, max 1.40/255, and at the 45-frame block boundaries
**0.857 vs 0.883 elsewhere (ratio 0.97)** -- boundary frames are not worse than
the rest, so the warm-up is doing its job.

The occlusion decision, measured in one early window: parallel at a 1:1
block/warm-up ratio was **3.75 fps against 5.18 pinned to one worker**, so the
floor now routes it to sequential instead. That is the old behaviour, not a
regression, and not a win.

Suite: **1596 tests, 1 skipped, 0 new failures**; 16 new contracts in
`tests/test_temporal_parallel_blocks.py`.

Not claimed: any 3060 number, any quality validation of either flag, any
cross-window fps comparison, and any optimality for the 3x floor.

## Phase 11 implementation and evidence

`blend_output` now uses a 128px working crop for its low-frequency correction,
controlled by `ROOP_TEMPORAL_IDENTITY_LOWPASS_SIZE`. `0` is the old
full-resolution reference path. `stabilize_mask` avoids redundant validation
copies while retaining state ownership and an independent return buffer.

Files changed:

- `app/roop/temporal_identity.py`
- `app/tests/test_temporal_identity.py`
- `app/tests/bench_temporal_identity_cost.py`
- `docs/ENV_FLAGS.md`
- `docs/OPTIMIZATION_PROGRESS.md`
- `docs/PHASE_HANDOFF.md`

Three counterbalanced 1200-call pairs at 256x256 measured:

| path | blend calls/s | mask calls/s |
|---|---:|---:|
| full-resolution reference (`0`) | 747.9 | 1283.6 |
| reduced working crop (`128`) | 1277.4 | 1290.6 |
| change | +70.8% | +0.5% |

The mask result is neutral within host noise and is recorded as an allocation
reduction, not a speed promotion. The reduced identity path remained finite,
uint8, dimensionally valid, detail-preserving, and under the synthetic MAE
bound against the reference. This is not real-footage quality validation.

Validation: targeted temporal set **38 passed, 1 warning**; full suite **1605
passed, 1 skipped, 595 subtests passed, 2 existing warnings**; Python
compilation and `git diff --check` passed. Three physical RTX 4070
feature-level renders also completed 600/600 with zero wrong FaceSets; the RTX
3060 feature run and retained-output visual review remain pending.

The 128px path remains **opt-in experimental**. The approximation is not
byte-identical to the reference; `lowpass_size=0` remains available for
diagnostics. No temporal flag is promoted to default.

## Complete-phase checklist audit — NOT COMPLETE

| Requirement | Evidence | Status | Missing |
|---|---|---|---|
| IMPLEMENT | Bounded low-pass identity path and mask allocation reduction | PASS | None in scoped implementation |
| TEST | 38 targeted passes; full suite passes | PASS | None in unit coverage |
| BENCHMARK | Reproducible component A/B plus three locked-fixture 4070 renders | PARTIAL | Physical RTX 3060 run and comparable cross-arm attribution |
| REGRESSION TEST | 1605 passed, 1 skipped, 595 subtests; all 4070 arms had 0 wrong FaceSets | PARTIAL | Retained-output manual visual regression review |
| DOCUMENT | `ENV_FLAGS.md` and `OPTIMIZATION_PROGRESS.md` updated | PASS | None |
| HANDOFF | This file records commands, constraints, and next starting point | PASS | It intentionally hands off the missing validation |

Phase 11 must not be marked complete until the partial benchmark and regression
items are closed with a physical RTX 3060 feature run, comparable attribution,
and retained-output visual review. The current synthetic benchmark and the
automated wrong-FaceSet check are not substitutes.

## Requested Phase 9 handoff — identity-specific detail preservation

Status: **OPEN / INCOMPLETE**.

Implemented starting point:

- `app/roop/faceset_v2.py` creates and aggregates signed canonical
  high-frequency residuals with persistence confidence.
- `app/roop/identity_detail.py` decodes, template-warps, exposure-scales,
  masks, smooths, and composites the representation.
- `app/roop/ProcessMgr.py` invokes it after enhancer, post-enhance colour,
  merger, manual mask, and temporal low-band stages.
- `app/roop/temporal_identity.py` owns bounded per-track detail history and
  clears it on source changes.
- `identity_detail_strength` is 0 by default and is available through config,
  preview, and swap API payloads. Existing target-texture
  `detail_transfer_strength` is intentionally separate.

Validation already completed:

- Focused command → **28 passed, 1 warning**:
  `app/env/Scripts/python.exe -m pytest app/tests/test_identity_detail.py app/tests/test_faceset_v2.py app/tests/test_temporal_identity.py -q`.
- Component benchmark → 0.839016 synthetic retention correlation; 43.3257%
  temporal-delta reduction; 290.71 restorations/s.
- 4070 controlled V1-backed smoke, GPEN 256 Pro / RealityUX / TensorRT /
  RealSwap, locked `double/d4.mp4`, strength 0.35 → **120/120**, return code
  0, no identity-detail runtime errors. It did not exercise V2 metadata because
  the locked `harjot/gargee` archives are legacy V1.

Exact next-phase starting point:

1. Create or obtain a V2 copy of the locked source archives without changing
   source identities or the target fixture; confirm `FaceSet.format_version ==
   2` and `identity_detail_for()` returns a valid residual for every selected
   source.
2. Extend the retained-output harness so the V2-backed run keeps its output
   video and records per-frame detail metrics. Run off / strength 0.35 / a
   confidence-reduced arm on the 4070 with identical capture, enhancer, mask,
   provider, codec, stabilizers, and frame range. Compare retention, temporal
   delta, wrong FaceSets, FPS, RSS, and VRAM only when paths are comparable.
3. Manually review mole, freckles, scar, wrinkle, and microtexture regions over
   frontal, turned, low-resolution, motion-blurred, and dark frames, including
   occluders and expression changes. Confirm omission beats flicker when
   confidence drops.
4. Run the same V2-backed component and real-footage matrix on the physical
   RTX 3060 while preserving `blend_ratio=0.85`, `face_mask_blend=25`,
   `merger_sharpen=0.55`, and `stabilize_enhancer_strength=0.6`; keep its
   single-context/global GPU guard and 1536 MB hard cap.
5. Test GPEN, UltraMax, and at least one additional restorer with identity
   detail enabled; verify post-restorer ordering and visual retention. Do not
   promote the feature or change its default until these checks pass.

Do not restart Phase 11 temporal optimization from scratch. Its 4070 evidence,
open 3060 validation, and retained-output visual-review requirements remain
unchanged below.

## Next session: exact starting point

1. On the 4070, rerun an order-balanced off/reference (`0`)/128px set with
   retained output videos or a quality-review harness. Record FPS,
   `faces_seen`, wrong FaceSets, output finiteness/order, peak RSS, and peak
   VRAM; do not attribute raw FPS when face counts diverge.
2. Manually review annotated occluder, eyes, mouth, hair, and difficult-pose
   frames from those retained outputs; synthetic MAE is not enough for
   promotion.
3. Run the component and real-video checks on the physical RTX 3060 while
   preserving `blend_ratio=0.85`, `face_mask_blend=25`,
   `merger_sharpen=0.55`, and `stabilize_enhancer_strength=0.6`. Do not copy
   4070 results or caches.
4. Resume the Phase 10 follow-up: measure the quality cost of lowering
   `ROOP_OCCLUSION_ENTER_ALPHA` from 0.90 toward 0.75 before considering its
   3x warm-up block, then sweep the 2x/3x/4x minimum block floor in one stable
   measurement window.
5. Decide whether the identity experiment can be promoted or must remain
   opt-in only after those results. Interacting-face behavior and the inherited
   3060 RSS gate remain open.

## Requested Phase 10 handoff - target-conditioned lighting and color realism

Status: **OPEN / INCOMPLETE**.

Implemented starting point:

- `app/roop/appearance_conditioning.py` owns robust target illumination/chroma
  analysis, NORMAL/DARK/VERY_DARK classification, low-light restoration and
  sharpening factors, restorer protection, and the bounded per-track EMA.
- `app/roop/procmgr_color.py` extends the existing color-transfer path with
  target-conditioned low-frequency spatial illumination, exposure/highlight
  quantile anchors, local contrast, and bounded skin-region chroma. It does not
  create a wholesale texture paste.
- `app/roop/ProcessMgr.py` analyzes the aligned target crop once per face,
  reuses its stabilized result for both color passes, protects dark output from
  GPEN/UltraMax/other restorers, and passes the tier into merger sharpening and
  clarity. The appearance engine is cloned/reset with the existing ordered
  contiguous-block stabilizer lifecycle.
- `app/settings.py`, `app/api.py`, and the React Face Swap controls expose the
  feature. It is opt-in and leaves the existing custom 4070/3060 look defaults
  unchanged when off.

Validation already completed:

- Focused tests: **49 passed, 1 warning**.
- Full suite: **1618 passed, 1 skipped, 598 subtests passed, 2 warnings**.
- Component benchmark: **23.3602 ms/call** analysis; stable-light temporal
  colour delta reduction **82.2298%** (0.02481532 to 0.00440974) over the
  daylight/indoor/tungsten/fluorescent/sunset/blue/mixed/night/street/low-
  exposure/backlighting fixture set.
- 4070 real integration smoke: **120/120 frames**, **294/294 faces**,
  **0 wrong FaceSets**, **3.67 fps**, approximately **10.01 GB peak RSS**,
  target appearance enabled with RealSwap/RealityUX/GPEN 256 Pro/TensorRT.
  Legacy V1 sources mean this does not assess V2 identity detail.

Exact next-phase starting point:

1. Keep the current code and run the component benchmark on the physical RTX
   3060 laptop with its single-context/global GPU guard, 1536 MB stabilization
   cap, adaptive block sizing, and preserved look values:
   `blend_ratio=0.85`, `face_mask_blend=25`, `merger_sharpen=0.55`,
   `stabilize_enhancer_strength=0.6`. Record analysis cost, RSS, VRAM, and any
   runtime fallback; do not copy 4070 results or caches.
2. Prepare or obtain a locked V2 source archive and verify
   `FaceSet.format_version == 2` and `identity_detail_for()` before combining
   this phase with requested Phase 9 detail preservation.
3. Run retained-output real-footage arms for daylight, indoor/tungsten,
   fluorescent, sunset, blue/mixed, night/street-light, low exposure, and
   backlighting. Compare feature off/on with identical source, target, face
   detections, enhancer, mask, codec, and frame range; measure luminance/chroma
   error, spatial shadow retention, frame-to-frame color delta, wrong FaceSets,
   output finiteness/order, FPS, RSS, and VRAM.
4. Manually review retained frames for partial shadows, colored night casts,
   highlight rolloff, low-resolution/motion-blurred faces, occlusions, and
   expression changes. Include GPEN 256 Pro, GPEN Realistic, UltraMax, and at
   least one additional restorer; confirm none lifts VERY_DARK faces or destroys
   target-conditioned lighting.
5. If any real scene changes tier within a shot, verify the tier-change
   hysteresis/EMA admits the transition without warm-neutral-blue oscillation.
   Only after this matrix passes should the default be reconsidered; until then
   keep `target_conditioned_appearance` opt-in.

## Standing rules that earned their keep this session

- **Null control per measurement window.** A 2.9x drift on the unchanged default
  made two arms look like a regression and a null result respectively.
- **Record the path, not just the number.** `faces_seen` distinguishes a
  sequential fallback from a slowdown for free on this fixture; without it they
  are identical.
- **Read the face count beside the fps.** Phase 9's regression presented as +47%.
- **A regression test must be shown to fail on the broken code.**
- **Prove a code path executes before believing "no effect".**

## Do not break

RealityUX, RealSwap, GPEN 256 Pro, GPEN Realistic, UltraMax and every other
enhancer; TensorRT and the FP16/FP32/mixed precision policy; source-bank and 3D
paths; detector alternatives and `det_size` handling; V1/V2 `.fsz`
compatibility; provider fallbacks; face-overlap ownership; the default
(no-flag) stabilization geometry, which `_stab_min_block_multiple = 1` keeps
bit-identical; the RTX 4070 pool settings; the RTX 3060 single-context guard and
its laptop look values.

## Requested Phase 11 handoff - adaptive enhancer orchestration

Status: **OPEN / INCOMPLETE**.

The exact implementation starting point is the new opt-in `Adaptive` selection
in `app/roop/core.py`, backed by `app/roop/adaptive_enhancer.py` and the bridge
`app/roop/processors/AdaptiveEnhancer.py`. `ProcessMgr` publishes pose, mask
occlusion, target appearance, and identity-detail-required signals before the
wrapper runs. The wrapper calls at most one existing candidate per face and
keeps the existing manual branch untouched. Candidate model code is lazy and
bounded by hardware/profile policy; `VERY_DARK`, extreme pose, low confidence,
heavy occlusion, and unstable tracks prefer omission.

Validation commands:

```text
app/env/Scripts/python.exe -m pytest app/tests/test_adaptive_enhancer.py app/tests/test_phase11_inventory.py app/tests/test_runtime_optimizer.py app/tests/test_settings_wiring.py -q
app/env/Scripts/python.exe app/tests/bench_adaptive_enhancer_video.py --help
app/env/Scripts/python.exe app/tests/bench_adaptive_enhancer_video.py --clip <locked-clip> --source <faceset> --enhancers Adaptive,GPEN 256 Pro,GPEN Realistic,UltraMax --adaptive-profile BALANCED
```

The video harness records runtime/FPS, process RSS, peak VRAM when available,
plate-relative output quality, temporal consistency, detected identity
similarity, and high-frequency/detail retention. Compare identical source,
target, detector, mask, swapper, codec, and frame range for each arm. The
existing `bench_phase11_enhancers.py` remains the isolated model benchmark.

Observed on 2026-09-01: the locked 4070 two-face smoke using
`double/d4.mp4`, RealSwap, RealityUX, TensorRT, Adaptive/BALANCED, and 12
workers produced 120/120 output frames, 120/120 swaps for each of two tracked
people, 240 face rows, and 0 wrong-FaceSet applications. The full video matrix
attempt entered the renderer but stalled after CUDA stream-906 and the
existing optional RealSwap secondary-network fallback warnings; it was stopped
and must be repeated with the stream/fallback condition resolved. No runtime,
quality, or memory value from that stalled attempt is accepted as a benchmark.

Required gates still open:

1. The final full regression suite is complete: **1641 passed, 1 skipped, 599
   subtests passed, 2 existing warnings**.
2. Review the retained-output 4070 Adaptive smoke for high-quality, moderate,
   dark, extreme-angle, occluded, blurred, and identity-detail frames.
3. Repeat the video matrix on the physical RTX 3060 with one context/global GPU
   guard, 1536 MB stabilization cap, RSS under 2.5 GB, and preserved look values
   (`blend_ratio=0.85`, `face_mask_blend=25`, `merger_sharpen=0.55`,
   `stabilize_enhancer_strength=0.6`).
4. Use locked V2 FaceSets and verify identity-detail retention is measured before
   and after each restoration family; confirm restorers do not erase it.
5. Manually inspect for flicker, artificial sharp points, lighting mismatch,
   hallucinated dark-scene detail, wrong FaceSets, and expression/occlusion
   failures before changing any default.

Exact next-phase starting point: run the full suite, then the 4070 retained
Adaptive video smoke and the `bench_adaptive_enhancer_video.py` matrix with a
locked V2 source archive; record its output JSON in the Phase 11 matrix. Keep
`selected_enhancer` on its current manual default until the 3060 and visual
gates above pass.

## Requested Phase 12 handoff - temporal compositing and natural blending

Status: **OPEN / INCOMPLETE**.

The exact implementation starting point is the existing final paste authority
`app/roop/procmgr_masking.py::MaskingMixin.paste_upscale`, now wired to
`app/roop/temporal_compositing.py::TemporalCompositeController`. The controller
is opt-in via `temporal_compositing`; when disabled, the previous linear ROI
blend remains the active path. When enabled, it receives the already-trimmed
model/ownership/occlusion matte, target-face pose/confidence, target appearance
tier, and frame index. It stabilizes a compact canonical matte per track,
attenuates semantic outer boundaries, preserves target spatial lighting through
low-frequency local adaptation, and protects generated high-frequency identity
detail. Existing enhancers and identity-detail ordering are unchanged.

Validation already completed:

- Full suite: **1648 passed, 1 skipped, 599 subtests passed, 2 warnings**.
- Component matrix: `app/tests/bench_temporal_compositing.py` completed
  frontal, lateral, profile, hair, glasses, hand occlusion, dark, and bright
  conditions; report at `app/output/phase12_temporal_compositing.json`.
- Matching 4070 real smoke: `double/d4.mp4`, 4 frames, OFF **0.47 FPS / 326.27
  ms / 4704 MB VRAM / 8.659 GB RSS**, ON **0.49 FPS / 318.28 ms / 4675 MB VRAM /
  8.544 GB RSS**, both return code 0, 2 face swaps, 0 wrong FaceSets. This is
  safety/cost evidence only because detector coverage was 1/4 frames.
- Poisson/gradient-domain candidate: available through OpenCV but not selected
  for production because it is CPU/full-patch work with no measured benefit
  over the bounded two-band ROI method.

Required gates still open:

1. Run the real Phase 12 matrix with locked annotated footage covering frontal,
   lateral, profile, hair, glasses, hand occlusion, dark, and bright scenes;
   record seam/edge quality, lighting transition, identity-detail retention,
   temporal shimmer, FPS, RSS, and VRAM for OFF/ON with identical inputs.
2. Repeat on the physical RTX 3060 laptop using the single-context/global GPU
   guard and 1536 MB stabilization cap. Preserve
   `blend_ratio=0.85`, `face_mask_blend=25`, `merger_sharpen=0.55`, and
   `stabilize_enhancer_strength=0.6`; do not transfer 4070 results or caches.
3. Review retained frames for jaw/cheek/forehead boundaries, hair, shadows,
   skin transitions, glasses and hands, bright/dark scenes, and hard/soft/hard
   edge oscillation. Include GPEN 256 Pro, GPEN Realistic, UltraMax, and
   identity-detail-enabled output to verify downstream restorers do not erase
   preserved details.
4. Compare the eight-condition output against the existing mask/ownership and
   target-conditioned appearance gates; if a real scene reveals over-feathering,
   target texture leakage, or an artificial sharp point, fix the same paste path
   and rerun the focused/full tests before changing defaults.

Exact next-phase starting point: run
`app/env/Scripts/python.exe app/tests/phase12_benchmark.py --target "RTX 3060"`
on the physical laptop with the locked annotated clip, then run the retained
4070 OFF/ON scene matrix and visual review. Keep `temporal_compositing=false`
as the default until both hardware and visual gates pass.

## Requested Phase 13 handoff - temporal artifact detection and selective correction

Status: **OPEN / INCOMPLETE**.

The exact implementation starting point is `app/roop/temporal_quality.py`,
called from the existing `ProcessMgr.process_face` path after alignment/source
selection and after enhancement/detail processing. It is opt-in through
`temporal_quality_control` and can be enabled for a controlled run with
`ROOP_TEMPORAL_QC=1`; `temporal_quality_logging` or `ROOP_TEMPORAL_QC_LOG=1`
adds records containing anomaly type, track, frame index, confidence, and
correction applied. Keep it disabled for legacy/default comparisons until the
gates below pass.

Validation already completed:

- `app/env/Scripts/python.exe -m pytest app/tests/test_temporal_quality.py -q`:
  **10 passed**.
- Combined Phase 13/11/12 focused integration set: **61 passed, 1 warning**.
- `app/tests/bench_temporal_quality.py --iterations 2000`: normal event rate
  **0.0**, one edge-triggered event per persistent anomaly case, approximately
  **0.654–0.678 ms** per inspect/record cycle.
- Full repository suite: **1658 passed, 1 skipped, 599 subtests passed, 2
  warnings**. The warnings are the existing Albumentations update notice and
  the existing NaN-to-uint8 enhancer guard fixture.
- Disabled controller remains a no-op; high-motion eye/jaw discontinuities are
  detected but do not reuse a prior transform and therefore preserve motion.

Required gates still open:

1. Run the full repository suite after Phase 13 and record the exact result;
   inspect for regressions in adaptive enhancer, V2 identity detail, target
   appearance, temporal compositing, source-bank selection, and mask ownership.
2. Run a locked real-video A/B with identical source, detections, swapper,
   enhancer, mask, codec, frame range, and workers. Record anomaly counts by
   type, correction counts, corrected-frame latency, FPS, RSS, VRAM, output
   finiteness/order, wrong-FaceSet count, identity similarity, detail retention,
   and temporal consistency. Review retained frames for drift, popping,
   brightness/color jumps, geometry jumps, hallucinated detail, detail loss,
   eye/jaw discontinuity, and flicker.
3. Repeat the relevant run on the physical RTX 3060 laptop with one context and
   the global GPU guard, 1536 MB stabilization cap, RSS under 2.5 GB, and the
   preserved look values (`blend_ratio=0.85`, `face_mask_blend=25`,
   `merger_sharpen=0.55`, `stabilize_enhancer_strength=0.6`). Do not transfer
   4070 caches or results.
4. Review low-resolution, motion-blurred, dark, occluded, lateral/profile, and
   high-motion expression frames. Confirm that correction is event-driven,
   real motion is retained, no generic blur is introduced, and GPEN 256 Pro,
   GPEN Realistic, UltraMax, RealityUX, and RealSwap do not erase preserved V2
   identity detail or reintroduce the detected artifact.
5. If real footage shows false positives or missed artifacts, adjust the same
   controller thresholds/correction gate, add a failing regression fixture,
   rerun the full suite and benchmark, then update this handoff. Do not make
   QC the default until these gates pass.

Exact next-phase starting point: run
`app/env/Scripts/python.exe -m pytest -q` from the repository root, then run
`app/env/Scripts/python.exe app/tests/bench_temporal_quality.py --iterations 2000`
and the existing locked video harness with
`--temporal-quality-control --temporal-quality-logging`. Start with
`temporal_quality_control=false` for the control arm, use the same Phase 11/12
scene matrix, and retain the existing RTX 4070/RTX 3060 hardware gates.

## Requested Phase 14 handoff — end-to-end GPU performance optimization

Phase 14 added an opt-in detailed profiler to the existing _prof hook and a
machine-readable comparator:

- app/roop/stage_profiler.py
- app/tests/phase14_bottleneck_report.py
- app/tests/test_stage_profiler.py
- app/tests/test_phase14_profile.py
- updates to app/roop/procmgr_runtime.py, ProcessMgr.py,
  procmgr_tracking.py, procmgr_masking.py, baseline_controlled.py, and
  bench_phase8_transfer.py

Run the active-path profile with:

app/env/Scripts/python.exe app/tests/baseline_controlled.py --tag phase14_detail --start 0 --end 60 --out app/output/phase14_profile --env ROOP_PROFILE_DETAIL=1

Use ROOP_PROFILE_DETAIL_SYNC=1 only for a short diagnostic arm. Compare
identical current-code arms with:

app/env/Scripts/python.exe app/tests/phase14_bottleneck_report.py --before <roi-off-json> --after <roi-on-json> --out <report-json>

The measured change is ROI-only blending (ROOP_BLEND_ROI_WARP=1, rollback
0). The 4070 paired detailed arms were 86.35 s / 2.75 FPS with ROI off and
82.07 s / 2.87 FPS with ROI on; blend wall time was 11.10 s versus 9.12 s
across the same 144 face calls, with zero wrong-FaceSet applications in both.
The result is promising but not yet a universal speed claim because the
standalone warp microbenchmark showed only ~0.9% median improvement and the
paired video result is one short arm. Synchronized 20-frame diagnostics
confirmed non-zero fence costs and must not be used as a throughput result.

### Phase 14 completion audit

| Requirement | Status | Evidence / remaining gap |
|---|---|---|
| IMPLEMENT | PASS | Existing stage hook, canonical matrix, opt-in profiler, measured ROI blend path, rollback flag |
| TEST | PASS | Phase 14 profiler/profile tests pass in the application environment; ROI output-equivalence test passes |
| BENCHMARK | PARTIAL | 4070 event-only/synchronized arms and transfer benchmark recorded; repeated arms, detail/expression/occlusion, and 3060 remain |
| REGRESSION TEST | PARTIAL | Focused tests pass; run and record the complete post-Phase-14 suite and longer preview/batch soak |
| DOCUMENT | PASS | ENV flags, progress, benchmark evidence, limitations, and this handoff |
| HANDOFF | PASS | Exact commands and unresolved gates below |

### Exact next-phase starting point

This is a resume point for **Phase 14**, not permission to begin a new phase:

1. Run app/env/Scripts/python.exe -m pytest -q after the final Phase 14
   edits and record the exact result.
2. Run at least three paired ROI-off/ROI-on arms on the locked 4070 workload
   with identical clips, models, workers, encoder, and environment; compare
   FPS, CPU/GPU/sync, sampled full-card peak/steady VRAM, RSS, output
   finiteness/order, wrong-FaceSet count, and retained visual quality.
3. Run the same detailed profile with the real V2 detail, expression, and
   occlusion options enabled so those rows are measured rather than left
   not_observed; use the phase-specific harnesses where each option has a
   locked fixture.
4. Repeat the accepted arm on the physical RTX 3060 with one GPU context,
   global GPU guard, 1536 MB stabilization cap, RSS below 2.5 GB, and
   blend_ratio=0.85, face_mask_blend=25, merger_sharpen=0.55,
   stabilize_enhancer_strength=0.6.
5. Review daylight/indoor/dark/occluded/lateral/profile and motion-blurred
   retained frames, including enhancer paths RealityUX, RealSwap, GPEN 256,
   GPEN Realistic, and UltraMax, before marking Phase 14 complete.

Do not change concurrency, TensorRT session ownership, provider transfer
policy, or enhancer defaults from the current evidence alone. Do not start
Phase 15 until these Phase 14 gaps are closed and the line-by-line checklist
in OPTIMIZATION_PROGRESS.md is updated from PARTIAL to PASS.

## Requested Phase 15 handoff — cross-hardware and regression validation

Status: **OPEN / INCOMPLETE**.

The audit implementation is intentionally observational:

- `app/roop/regression_audit.py` owns the backend/precision/workflow/lifecycle manifest and provider/cache classification.
- `app/tests/phase15_regression_audit.py` generates the report without building, deleting, or reusing TensorRT engines.
- `app/tests/test_phase15_regression_audit.py` protects unavailable-provider, enhancer-selector, coverage, and stale-cache behavior.

The current report is at `app/output/phase15_validation/audit.json`. It records the RTX 4070 host's TensorRT/CUDA/CPU providers as `available_not_validated`, ROCm/DirectML/CoreML as `unavailable`, 190 `not_run` workload/lifecycle rows, 19 stale-driver candidates, and 9 unscoped cache candidates. Do not delete those cache directories as part of this phase; review/rebuild them under the active driver-qualified namespace instead.

Validation already completed for this implementation:

- `app/env/Scripts/python.exe -m pytest -q app/tests/test_phase15_regression_audit.py` — **6 passed**.
- Combined focused provider/precision/hardware/inventory/Phase 14 set — **64 passed, 1 warning**.
- The CLI audit completed successfully and wrote the JSON report above.
- The retained-output integrity sweep over ten existing Phase 12/13 videos
  passed **10/10** (600 frames each; zero black, uniform, NaN, or duplicate
  frames). This catches encoder/output corruption only and does not replace
  the missing cross-hardware workload evidence.

Required gates before marking Phase 15 complete:

1. Run the full repository suite from the repository root and record its exact count after the final Phase 15 edits: `app/env/Scripts/python.exe -m pytest -q`.
2. On the RTX 4070, run fresh-process precision arms using the same locked clip/source/models/codec and record PASS/FAIL plus identity, detail, finiteness, FPS, VRAM, RSS, provider, and precision: `app/env/Scripts/python.exe app/tests/precision_matrix.py --clip G:/pinokio/roop-keep/double/d4.mp4 --source harjot --out app/output/phase15_validation/rtx4070_precision`. Use fresh processes for every provider/precision; never switch precision in a live ORT/TensorRT process.
3. Run the complete enhancer/profile video matrix on the same locked inputs with `bench_adaptive_enhancer_video.py`, including every label accepted by `compare_enhancers_video.VALID_ENHANCERS` and FAST, BALANCED, REALISTIC, and MAX QUALITY. Grade outputs independently with `app/tests/phase16_integrity.py`; a successful process is not sufficient.
4. Exercise image swap, multi-face swap, new and legacy faceset loading, faceset creation, preview, batch, and a long-video run. Record startup, shutdown, model release, memory release, repeated jobs, provider changes, precision changes, output ordering/finiteness, and anomaly logs. Re-run the audit report after each provider/device switch so cache namespaces are compared to the active runtime.
5. Repeat the supported provider rows on the physical RTX 3060 laptop with the single-context/global GPU guard, 1536 MB stabilization cap, RSS under 2.5 GB, and preserved look values: `blend_ratio=0.85`, `face_mask_blend=25`, `merger_sharpen=0.55`, `stabilize_enhancer_strength=0.6`. Do not transfer RTX 4070 engines or results. A missing physical device remains `pending`, not pass.
6. Run the AMD ROCm/DirectML and Apple CoreML subsets on their real supported hosts. If a provider is not applicable or cannot be installed, record `unavailable` with the host/runtime reason; do not mark it passed by CPU fallback. Verify CPU fallback separately.
7. Inspect regressions for stale engine reuse, provider/device mismatch, corrupted state, ONNX/TensorRT context races, memory leaks, preview/batch interference, identity/detail loss, brightness/color jumps, and temporal artifacts. Add a failing fixture for each defect found, then rerun the full suite and relevant benchmark.

Exact next-phase starting point: remain in Phase 15 and run the full suite (step 1), then the fresh-process RTX 4070 precision matrix (step 2). After all Phase 15 rows are evidenced and the checklist in `docs/OPTIMIZATION_PROGRESS.md` is PASS, the next phase may start with an output-integrity review using `app/tests/phase16_integrity.py` on retained artifacts. No Phase 16 completion or default change should be inferred from this audit-only implementation.
## Requested Phase 16 handoff — final production quality gate

Status: **OPEN / INCOMPLETE**. This is the final gate, so there is no later optimization phase to infer or start from an incomplete report.

Added without changing runtime processing behavior:

- `app/roop/final_quality_gate.py` — standardized 17-category clip manifest, four quality modes, five named component arms, every registered enhancer, metric schema, `.fsz` checks, and evidence-only winner selection.
- `app/tests/phase16_final_quality_gate.py` — report generator.
- `app/tests/test_phase16_final_quality_gate.py` — manifest, evidence, enhancer-preservation, winner, and legacy/V2 compatibility contracts.
- `docs/FINAL_ARCHITECTURE.md` — complete pipeline/resource-ownership architecture.

Generate the current report with:

`app/env/Scripts/python.exe app/tests/phase16_final_quality_gate.py --out app/output/phase16_validation/final_report.json`

Current report state: 17 categories, 0 ready clips, 425 benchmark rows, 0 complete runs, no selected winners, and `OPEN_INCOMPLETE`. The report also lists the Phase 1–15 audit status. The 425 rows are 17 clips × (FAST/BALANCED/REALISTIC/MAX QUALITY + RealityUX + RealSwap + GPEN 256 Pro + GPEN Realistic + UltraMax + 16 registered enhancer labels).

Final post-edit validation: **1,676 passed, 1 skipped, 599 subtests passed, 2 warnings**. The warnings are the existing Albumentations update notice and the existing NaN-to-uint8 enhancer-guard fixture warning. `git diff --check` passed.

Required final-production gates:

1. Supply real annotated clips for all 17 categories. Do not use a filename or process return code as quality evidence.
2. Run every row with fresh processes where provider or precision changes are involved. Record total time, FPS, frame latency, peak VRAM, CPU/GPU utilization, detection, swap, enhancer, blending, dropped, and fallback frames.
3. Record identity, temporal flicker, expression, eye state, pose, occlusion, boundaries, color, low-light realism, and identity-detail retention from retained output and annotations.
4. Independently inspect steep/inverted, lateral, fast motion, blinking/speaking, dark/night/mixed lighting, foreign-object, hands, glasses/hair, crossing faces, low resolution, and motion blur.
5. Run old and new `.fsz` archives through the compatibility check and existing FaceSet tests. Confirm V2 metadata/checksums and legacy PNG loading on representative production files.
6. Re-run all previous-phase tests and physical gates: RTX 4070/3060 constraints, CPU fallback, AMD ROCm/DirectML where supported, Apple CoreML where supported, every enhancer/mode, provider fallback, preview, batch, long video, startup/shutdown, release, repeated jobs, switching providers/GPUs/precision, and cache namespace validation.
7. Use `app/tests/phase16_integrity.py` on every retained output and investigate black, uniform, NaN, duplicate, unreadable, frame-count, duration, or audio regressions.
8. Populate evidence JSON and rerun the report. Only when every required row is complete and independently reviewed may the winner fields be used: fastest, best balanced, best quality, best night, best difficult angle, and best multi-face.

Exact continuation point: collect and register the 17 annotated clips, then execute the 425-row matrix through the existing render/benchmark harnesses and feed complete evidence into `phase16_final_quality_gate.py`. Until the report shows all rows complete and the progress checklist is PASS, the optimization program remains open and no defaults should be promoted.
