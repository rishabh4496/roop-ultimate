# Parallel-stabilization warm-up: what is state-independent (2026-10-09)

Target: `ProcessMgr._run_stab_parallel` (`_process_block`, ProcessMgr.py ~2652) and
`procmgr_stabilization._stab_parallel_geometry`. Config analysed is the live `app/config.yaml`: hyperswap, DFL XSeg,
Restore Ultra, `temporal_detection: true`, `track_identities: true`, `stabilize_face/mask/enhancer/landmarks/hf_texture: true`,
`stabilize_enhancer_strength 0.5`, `stabilize_mask_strength 0.5`, threads 10, `swap_mode selected`.

Method tags: **[M]** measured in a real render today, **[R]** read from the code (file:line), **[A]** assumption that
the Step 2 acceptance test (bit-identical output at a pinned geometry) must confirm.

## 0. The redundancy, measured

* **[M]** d4 and s7, 600 frames, `ROOP_STAB_CHUNK_MB` pinned: `wu=6 block=24 workers=10 blocks_per_chunk=20`,
  `stab.blocks 25`, `stab.warmup_frames 144`, `stab.output_frames 600`. Warm-up is **144 of 744 = 19.4%** of all frames
  processed (24 blocks pay 6; block 0 pays none). Redundant share of a block is `wu / (block + wu)` = 20% at 4x.
* **[M]** `WU = 6` is set by the enhancer filter (strength 0.5 -> `base_cutoff 0.22` -> 6 frames) and the mask filter
  (same class, same 6). **No kps filter takes part** (section 1). `warmup_frames` per filter: one_euro.py:317 (kps, 10 at
  `min_cutoff 0.1`), :554 (enhancer/mask `ema_warmup_frames(_alpha(1, base_cutoff))`).
* The warm-up frames of block k are the **last 6 output frames of block k-1** (`range(max(0, ca-WU), ca)`,
  ProcessMgr.py:2701, `_combined` carries the previous chunk's tail across chunk boundaries). So every warm-up frame
  is a frame some other block also processes as real output: each frame is inferred twice, nothing else is shared.
* The warm-up runs the **entire** `process_frame` (swap, enhance, mask, lighting, paste, composite) and throws the
  picture away; only the filters' internal state survives it.

## 1. The question that decides everything: does any filter change what the networks see?

`temporal_detection: true` in this config. **[R]** procmgr_batch.py:214-226: when `_temporal_mode`, the run sets
`self.kps_stabilizer = None`, `self._kps_stab_factory = None` and `self._landmark_smoother.enabled = False`. The
tracking pre-pass (`procmgr_tracking._build_temporal_faces`) already smooths kps **and** landmark_2d_106 sequentially
per track, once, before any block runs, and caches the result in `self._temporal_faces[frame]`.

* **[M]** Probe (a wrapper on `ProcessMgr._apply_stab` installed in the render process, 240 frames of d4, parallel
  path, 20 blocks x 24f): **0 calls**. The render log reports `[Temporal] track 0 [LandmarkSmooth] ... applied 61/62`
  from the pre-pass, and the geometry's `warm-up 6f` (a live kps filter would force >= 10).
* Consequence: with `temporal_detection` on, `face.kps` that reaches `canonicalize_face_alignment`
  (face_analyser.py:674; `M = estimate_norm(kps, size, mode)`, `aligned = warpAffine(plate, M)`) is the cached,
  already-smoothed value. **The aligned crop and M are a pure function of (plate, cached face) - no block, no
  filter, no thread enters.** `_cur_kps_stab()` is `None`, `do_kps_stab` is False (ProcessMgr.py:3478).
* **If `temporal_detection` is off** the live kps filter is built, `_apply_stab` rewrites `face.kps` (and 106
  landmarks) from per-block filter state before alignment (ProcessMgr.py:3389-3396, 3515, 4554), and M then differs
  between a block's warm-up and the neighbour that owns the frame (the first warm-up frame passes the raw kps
  through; the rest carry a seed residual up to 1%). In that mode M-keyed entries essentially never match. **The
  cache must be off there, or keyed so it cannot hit.**

## 2. Per-face pipeline (swap_mode selected, one face), in execution order

`process_frame` -> `swap_faces` -> `process_face` (ProcessMgr.py:3095 / 3426 / 4971). Processor order is
**swap -> enhance -> mask** (`get_processing_plugins`, core.py:954-1020 appends mask engines after the enhancer;
`self.processors` keeps dict order, ProcessMgr.py:1347).

| # | step | inputs | filter state read/written | state-independent? |
|---|---|---|---|---|
| 1 | faces for the frame | `_temporal_faces[frame]` (pre-pass), `_track_assignments` | none (read-only dicts) | yes |
| 2 | source pick | track assignment, `source_bank/portfolio` (off/deterministic), `pose_embedding_for_target(kps)` | none | yes **[A]** |
| 3 | alignment: `aligned_img`, `M` | plate, cached kps | none (section 1) | **yes** |
| 4 | swap net + colour transfer -> `fake_frame` (256) | aligned_img, source embedding | none; but the net is reached through the cross-frame **swap batcher** (batch mates vary with timing, `BatchSwap ... max_batch=8 wait=4ms`) | yes **[A: batch-composition invariant]** |
| 5 | restorer (RF++ net, `_restore_ultra_recombine`) -> raw `enhanced_frame` (512) | fake_frame, plate, kps, M | none (`EnhanceGate` threshold 0 = off, `enhance_min_face_px: 0`) | yes |
| 6 | mask processors (`process_mask`, one per masker: **`mask_xseg` then `mask_occluder`** in the live app, because occlusion masking appends the occluder) | aligned_img, M, plate, `region` | **`process_mask` applies `MaskStabilizer` INSIDE itself** for the dense maskers (procmgr_masking.py ~1480: `_ms.apply(img_mask, kps, t)`) before it returns, so even `defer_composite=True` returns an already-FILTERED mask; ProcessMgr then applies the same filter a second time (t == last_t) | **NO** - the net's output is pure, what `process_mask` returns is not |
| 7 | mask filter, second application | the mask `process_mask` returned | same per-track `prev`/`last_t` (`t_e = max(0,1) = 1`, so it blends the mask with itself) | **NO** |
| 8 | `_composite_mask` of the swap crop and the enhanced crop under the FILTERED mask | filtered mask, aligned_img, raw crops | none | depends on 6-7 |
| 9 | **enhancer filter** `EnhancerStabilizer.apply(enhanced_frame, kps, t)` | step-8 enhanced crop | per-track `prev` crop | **NO** (input already carries step 7) |
| 10 | **HF carry** `_hf_stabilizer.stabilize(...)` (flow-warped high band) | step-9 output | per-track previous high band + DIS flow handles | **NO** |
| 11 | expression restore (strength 0), detail transfer, identity detail, paste back, face compositing | filtered crop + plate | none cross-frame | stateless, depends on 7-10 |

The filter chain is **7 -> 8 -> 9 -> 10, each stage feeding the next**, so the three filter states are all functions
of the raw mask (6) and the raw enhanced crop (5) only through that chain. Nothing in 7-10 feeds back into 1-6.

## 3. The four intermediates asked about

| intermediate | produced at | depends on any filter's state? | notes |
|---|---|---|---|
| aligned crop + `M` | step 3 | **No** in this config (section 1). **Yes** if `temporal_detection` is off. | cheap (alignment 4.9 ms/face), recomputed on every call anyway |
| raw swap output (`fake_frame`) | step 4 | No | 70 ms/face (`swap` stage incl. colour transfer, thread time) |
| raw enhanced crop | step 5 | No | 116 ms/face |
| raw mask net output | inside step 6, before the filter | No - but **not separable from outside `process_mask`**: the function returns it already filtered | 26 ms/call (mostly hull / mouth / blur CPU; the net is ~2.4 ms) - left running on every visit |

Per-face intermediates **downstream** of a filter (not cacheable, must be recomputed per block): filtered mask,
composited crops, enhancer-filtered crop, HF-filtered crop, final paste.

**Correction made while wiring this up (first attempt cached the mask and was wrong).** The cache stores only
`(fake_frame, enhanced_frame, scale_factor, swap_model_mask)` - the output of swap and enhance. Every mask processor
runs normally on every visit, hit or not, so each block's `MaskStabilizer` is fed by its own `process_mask` calls
exactly as today (including the live app's second masker and the double application). That makes the cache
indifferent to how many maskers there are or what they do, at the price of re-running the mask stage on a warm-up
visit: the mask net is ~2.4 ms of GPU, the rest is CPU. The GPU work that dedups is the swap net and the restorer
(the two expensive networks).

## 4. Hidden state audited (anything that would make steps 1-6 not a pure function of the frame)

* **`target_face` is shared and mutated.** `faces = list(_tfaces)` is a shallow copy (ProcessMgr.py:3530); the Face objects
  are the cached ones. Writes in steps 1-6 are attribute stores of values that are themselves pure functions of the
  frame (`target_face.matrix = M`, `_adaptive_yaw/pitch/roll`, `_pose_v5_*`, `plate_ctx` set then cleared). With
  `_apply_stab` unreachable (section 1) there is **no write that depends on a block's state**; two blocks writing the
  same value to the same frame's Face is benign. **[M]** probe confirmed `_apply_stab` never ran, so the
  double-smoothing hazard (a second block reading kps a first block already smoothed in place) does not exist here. It
  WOULD exist with a live kps filter: another reason the cache is gated on section 1.
* **Cross-thread state read in steps 1-6:** `EnhanceGate._skipping` (hysteresis per track): off (threshold 0).
  `_nonfrontal_router` (events shared across workers): off (`ROOP_NONFRONTAL_MASK` `0`). `_angle_route_stats`: counters
  only. `self._tls.swap_model_mask`: written by the swap branch, read by the mask engine in the same call (hits skip
  both). `_stab_history`: `None` on the parallel path.
* **Shared-Face stamps by the occluder.** `_stamp_occlusion_state` writes `occlusion_state`, `_occluded_landmark_frac`,
  `_landmarks_symmetric` onto the cached Face from the block's *stabilized* mask (procmgr_masking.py:1509). Two blocks
  therefore stamp the same Face object with block-dependent values today, and the hull / mouth cut-out may read them.
  That race pre-dates this work and is untouched (mask processors still run on every visit); the A/A hash equality
  below is the evidence it is benign in practice.
* **Counters a hit skips:** `_prof` stage timings, `total_swaps`, audit buckets that are bumped inside steps 4-6. The
  swap audit's `faces seen / swapped` are bumped in `swap_faces` before `process_face`, so they are unaffected;
  per-stage ms/call will read lower by construction (that is the point) and `warmup` counters must say so.
* **Order-dependent engines that make the cache ineligible if ON:** `temporal_identity`, `temporal_occlusion`,
  `target_appearance`, `temporal_compositing`, `temporal_quality` (all `false` in this config; they keep per-track output
  history and are cloned per block, ProcessMgr.py:2670), `autorotate_faces` with an applied rotation (the crop space is
  per-frame; handled by the same `rotation_action is None` guards the filters use).
* **Determinism of the networks:** **[M]** the four ABBA renders of d4 (and of s7) today have identical output sha256
  and decoded-video md5 at a pinned `ROOP_STAB_CHUNK_MB`, including the two arms with the XSeg CUDA graph. So a render
  at a pinned geometry is bit-reproducible on this machine, which is what makes "bit-identical" a usable acceptance.
  The "non-deterministic GPU reduction order" noise floor in AGENTS.md does not show up for this config.

## 5. Filter inventory

| filter | where | state | WU (config) | input depends on |
|---|---|---|---|---|
| kps (`KpsStabilizer`) | `_apply_stab` | tracks[{OneEuro, centroid}] | 10 | **not built** with `temporal_detection` |
| landmark smoother | `_apply_landmark_stab` | per-track EMA | 11 | **disabled** with `temporal_detection` (pre-pass owns it) |
| mask (`MaskStabilizer`) | mask branch | per-track prev mask | 6 | raw mask (6) |
| enhancer (`EnhancerStabilizer`) | after the loop | per-track prev crop | 6 | composited enhanced crop (8) |
| HF (`HighFrequencyFlowStabilizer`) | after enhancer filter | prev high band + DIS | 3 | enhancer filter output (9) |
| temporal_* engines | `clone_for_block` | ordered history | 15-44 | **off** |

## 6. Conclusions that shape Step 2

1. The seam is **before the first mask processor** (after swap and enhance). Everything before it is a pure function
   of `(global frame, face, M)`; everything from the first `process_mask` on touches a block's mask filter and runs
   per visit.
2. Key `(global frame index, track id, M.tobytes())`. M is computed on every call before the cache is consulted, so the
   M component is a free correctness check; it only matters when M can vary (it cannot in this config - section 1 -
   but it makes the cache safe, not merely correct by assumption, if a future config reintroduces a live kps filter).
3. Producer/consumer ordering: block k's warm-up frames are block k-1's *last* frames, so the first block to ask is
   normally block k (at its start), not k-1 (at its end). A per-key future makes k-1 reuse what k computed, or the
   reverse when k-1 got there first. Total work becomes one inference per distinct frame instead of per visit.
4. Upper bound on skipped GPU work = the warm-up share: **19.4%** at the shipped 4x geometry (600 frames), 12.5% of a
   block at 8x, lower as chunks grow. This removes GPU inference (not just redistributes thread time), which is the
   only lever AGENTS.md says moves a GPU-bound render.
5. Eligibility (cache on only when all hold): parallel path, `_temporal_mode`, `kps_stabilizer is None`, no ordered
   temporal engine, processors `swap -> [enhance] -> mask*` with the enhancer on a verified stateless list
   (`restore_ultra`, `restoreformer++`), `warmup <= block`; per face: no applied rotation, no frontalization, no
   temporal-quality engine. Otherwise the run takes today's code path untouched.

## 7. Open assumptions the acceptance test will check ([A] above)

* Swap output is invariant to the batch it was coalesced into (the cross-frame batcher). RF++ was measured bit-identical
  at B=1/2/4/8 on its dynamic-batch engine (restorer_batch_2026-10-09.json); the swapper has not been measured.
* The source-selection and pose-embedding calls in step 2 are pure functions of the cached face.
* No other `process_face` local survives from steps 4-6 into the code after the seam. To be verified by diffing the
  locals read after the seam against the cached set before wiring.

---

# Results (2026-10-09)

## Step 2 - the dedup cache (`roop/stab_dedup.py`, `ProcessMgr.process_face`, `_run_stab_parallel`)

Cached: `(fake_frame, enhanced_frame, scale_factor, swap_model_mask)` - the swap and restore output, published at the first
mask processor. Every mask processor still runs on every visit. Keyed `(global frame, face_index, source index, track id,
M bytes)`; only the last `warmup` frames of a block that a later block warms up from are cached; entries are removed on
their second visit; a hard byte cap (`ROOP_STAB_DEDUP_MB`, default 384) can only reduce the saving. `ROOP_STAB_DEDUP=0`
restores today's behaviour exactly. Default ON when eligible (section 6).

ABBA, live config, 600 frames, `ROOP_STAB_CHUNK_MB` pinned from a discarded warm-up, `tests/ab_stab_dedup.py`
(`stab_dedup_2026-10-09.json`). A = `ROOP_STAB_DEDUP=0` (the pre-change path), B = default.

| clip | A fps | B fps | B/A | A-vs-A / B-vs-B | swap calls A -> B | peak RSS A / B |
|---|---|---|---|---|---|---|
| d4 | 15.16, 14.92 | 16.22, 15.91 | **1.068** | 1.6% / 1.9% | 828 -> 692 (-136) | 11.96, 11.95 / 12.08, 11.89 GB |
| s7 | 17.39, 17.15 | 19.54, 18.33 | **1.096** | 1.4% / 6.4% | 652 -> 526 (-126) | 12.03, 12.08 / 12.10, 12.07 GB |

* **Output bit-identical:** file sha256 and decoded-video md5 are the same in all four arms of both clips (d4
  `6650e371b02b`, s7 `891f687daa2c`), and the same hashes appeared in an earlier, independent session.
* **Warm-up inference "skipped":** every overlap frame is now computed ONCE instead of twice. d4: 136 overlap face-visits
  computed, 136 served, 0 bypassed, 0 producer failures, 0 waits; s7: 126 / 126 / 0. That removes **16.4% (d4) and 19.3%
  (s7) of all swap + restore runs**, the whole of the duplicated work (50% of the overlap's inference). Which visitor
  does the work flipped: the WARM-UP visit (block start) arrives first and computes; the later OUTPUT visit (end of the
  previous block) is served. So "warm-up inference skipped" is true of the total, not of the warm-up pass itself.
* **Peak RAM:** no change within the run-to-run spread (cache peak held: 90 MB on 720p). The RAM column above is the
  whole process tree.
* fps: the effect is larger than the noise on both clips (d4 6.8% vs <= 1.9%; s7 9.6% vs 6.4% B-vs-B, and both B arms beat both A arms).

## Step 3 - larger blocks / RAM-derived chunk size

**The premise was already in the code, and the brief's numbers would make it worse.** `_default_stab_chunk_mb` already
derives the per-chunk budget from `psutil` available RAM (`avail * share / live_copies`, share 0.75 on a >= 28 GB machine,
0.40 on the laptop, cap 4096 MB) and `_stab_parallel_geometry` already auto-selects the 8x block when it fits two whole
rounds. A 40% cap would take this desktop from 1674 to 893 MB at 10.9 GB free and make 8x engage LESS. Nothing was
changed there. What the auto-8x rule needs (20 blocks of 48 frames, 2.5 GB at 2.6 MB/frame, i.e. ~16.5 GB free) is not
available on this machine at the moment (free RAM was 10.9 GB at the lowest and about 13.3 GB at the highest in today's
runs, by the budgets they derived).

**Fidelity against a sequential render** (`tests/ab_stab_blocks_vs_sequential.py`, d4 frames 0-240, `ROOP_STAB_PARALLEL=0`
as ground truth). First attempt compared the hevc_nvenc outputs and was misleading: mean |diff| 0.64-0.66/255 at every
block size, with 74% of the difference in pixels off by <= 2 and spread over the whole frame. A rate-controlled encoder
turns one sub-pixel difference into a whole-frame, persistent quantization difference. Re-run with the encoder out of
the loop (`ROOP_BENCH_CODEC=libx264 ROOP_BENCH_CRF=0`, lossless in YUV; new env override in `two_face_video.py`):

| arm | mean abs diff vs sequential (of 255) | frames differing (of 240) | max pixel diff |
|---|---|---|---|
| sequential, repeated | 0.00000 | 0 | 0 |
| parallel 24f blocks (shipped) | 0.01877 | 125 | 13 |
| parallel 48f blocks | 0.01342 | 71 | 14 |
| parallel 96f blocks | 0.01250 | 56 | 14 |

(`stab_blocks_vs_sequential_2026-10-09_lossless.json`; the encoded run is `..._2026-10-09.json`.) Larger blocks do move the
output monotonically closer to sequential and never further (-33% mean, -55% differing frames from 24f to 96f), and the
sequential render is exactly reproducible (two runs, identical sha, in both codecs). The shipped blocks are already
0.019/255 from it on average. Block size is clamped to 16x by `_mult = max(2, min(16, ...))`; 32x is the same as 16x.

**Speed at equal chunk RAM** (cache on in both arms, 4x = 24f blocks x 2 rounds vs 8x = 48f blocks x 1 round, ~480
frames per chunk; `stab_dedup_2026-10-09_mult.json`):

| clip | 4x fps | 8x fps | 8x/4x | spread 4x / 8x | swap calls 4x / 8x |
|---|---|---|---|---|---|
| d4 | 15.32, 16.19 | 14.31, 14.18 | **0.904** | 5.5% / 0.9% | 692 / 685 |
| s7 | 19.21, 19.14 | 18.31, 18.62 | **0.963** | 0.4% / 1.7% | 526 / 526 |

Both 8x arms are below both 4x arms on both clips. With the duplicated swap/restore gone, the work 8x was meant to
shave is already gone (swap calls identical to within 7), and the one-round chunk gives up the work-stealing slack.

**Verdict: no default changes for Step 3.** Bigger blocks are closer to sequential by a margin that is already tiny, and
cost 4-10% of fps at the RAM this machine has. If a future machine has >= 16.5 GB free the auto-8x rule will still take
it; the numbers above are the evidence for revisiting it there.

## Findings outside the brief

* **The "pixel noise floor 0.7142/255 between two renders of one config" in AGENTS.md did not reproduce.** At a pinned
  geometry, 16 repeated parallel renders (hevc_nvenc, d4 and s7 x 8 each across two sessions, cache and XSeg graph on and off) were bit-identical per clip, and the
  sequential render was bit-identical in both codecs. The encoded 0.65/255 above is encoder amplification of a real,
  tiny upstream difference, not GPU non-determinism. The number may come from renders whose geometry was not pinned
  (free RAM changed the block grid). Not edited in AGENTS.md - for the owner to decide.
* `process_mask` applies `MaskStabilizer` inside itself (dense maskers) and ProcessMgr applies it again right after
  (same `t`); with two maskers (`mask_xseg` + the occluder the app appends) the single filter instance is fed
  alternately. Pre-existing, untouched, and the reason the cache stops before the first mask processor.
* The occluder stamps `occlusion_state`/`_landmarks_symmetric` onto the shared cached Face from the block's stabilized
  mask; two blocks do this to the same object today. Pre-existing; the identical hashes show it is benign here.

## Not measured

* The 3060 laptop (no TensorRT; small-card enhancer policy may strip the enhancer, in which case the cache is a no-op
  or smaller): the cache is bounded by `ROOP_STAB_DEDUP_MB` and can only reduce its own saving.
* Clips with `temporal_detection` off, or any ordered temporal engine on: the cache is not built (tests), so no claim.
* 1080p: cache size scales with frame area; the cap is the guarantee, the 90 MB figure is 720p only.
