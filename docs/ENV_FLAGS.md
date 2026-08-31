# Environment Flags (`ROOP_*`)

All runtime tunables read from environment variables. Every flag is **opt-in or
safe-by-default** — with none set, the app runs its validated default behavior.
Values are read at process start (some at module import), so set them **before**
launching and restart the app to change them.

Defaults below are what the code falls back to when the variable is unset.

> The Pinokio launcher already wires a few of these from the perf settings UI
> (`ROOP_TRT_POOL`, `ROOP_DETMASK_POOL`, `ROOP_ENCODER_PRESET`, `ROOP_PROFILE`,
> `ROOP_BATCH_SWAP`, `ROOP_NVDEC`) — see `app/run.py`.

---

## Measuring instead of guessing: the hardware benchmark

Most of the throughput flags below ship with a **VRAM-tiered default** — a table
in `roop/session_pool.py` that maps card size to pool size. That table was
measured on a handful of cards, and a tier is a guess about every card that is
not one of them. **Settings → Performance → Hardware benchmark** replaces the
guess with a measurement of the machine in front of you.

It times the models your *current settings* select — the selected swapper at its
real tile count, the selected enhancer, the selected mask engine, the selected
detector plus the per-face aux models, the expression restorer — built through
the app's own provider rules (including the two places TensorRT is forced to
FP32, and the models the TensorRT EP is removed from). Then it:

* sweeps each stage's pool width and finds the knee, with per-instance VRAM cost
  read from `torch.cuda.mem_get_info` (whole-device, so it sees onnxruntime and
  TensorRT allocations);
* runs a **composite frame** — detect + swap×tiles + enhance + mask, plus the
  real `align_crop`/paste CPU work — over worker-thread counts, with pooled
  stages leasing and un-pooled stages taking the same global lock `_gpu_guard`
  imposes, and picks the thread count per workload mode;
* re-runs that frame with the **costliest** stage's pool one step wider, because
  a stage that plateaus *alone* can still pay off in a frame by staying off the
  global lock — measured on an RTX 4070, the swapper flattened at 4 contexts in
  isolation but a 5th took the heavy frame from 9.58 to 10.12 fps. One knob, not
  all four: widening everything at once took the same card to 11961 MiB of 12282
  and it started paging;
* compares TensorRT against CUDA per stage, checks batched swap, and times
  libx265/libx264/NVENC encode and cv2/NVDEC decode.

### What it writes, and when it takes effect

| What | Where it lands | When it applies |
|------|----------------|-----------------|
| thread counts | `benchmark_results` | **immediately** — `Settings.resolve_threads` reads them per run |
| pool sizes | `perf_trt_pool`, `perf_detmask_pool`, `perf_detector_pool`, `perf_expr_pool` | **restart** — exported to env by `run.py` at startup |
| `perf_batch_swap`, `perf_nvdec` | same | **restart** |
| video codec | `output_video_codec` | **immediately** — `_run_swap` re-reads it from config every run |
| encoder preset | `perf_encoder_preset` | **restart** |

The provider recommendation is reported but **not** applied: switching it
rebuilds TensorRT engines, which is a multi-minute cost the user should choose.

**The encoder is not chosen by speed**, and this is deliberate. Encoding is not a
separate pass — `ProcessMgr.write_frames_thread` streams frames into a live
`FFMPEG_VideoWriter` while the swap runs — so an encoder already faster than the
pipeline produces frames buys nothing in wall clock. On the reference machine
`hevc_nvenc p5` measured 170 fps against libx265 medium's 42.5, for a file **15x
larger** on the same frames. So the benchmark keeps your configured codec and
moves only the PRESET, picking the cheapest one that clears the fastest mode's
frame rate with 2x headroom (for the cores it will not get during a real render,
and for a target that may be larger than the 1080p test clip). At a fixed CRF a
faster x264/x265 preset is perceptually equivalent, so that move costs nothing
but a slightly larger file. The codec itself changes only when NO preset of it
can keep up — a 4K target, or a CPU busy enough that the software encoders fall
behind — and then it goes to the fastest measured, usually NVENC, which also
moves the work onto the GPU's dedicated encode engine and off the worker cores.
Size never decides across codecs: CRF 18 does not mean the same quality to x264
and x265, so comparing their file sizes compares two different qualities.

`env/Scripts/python.exe -m roop.bench --profile full --no-apply` runs the same
thing from a terminal and prints the tables without touching the config.

> Note for anyone reading old screenshots: the previous "GPU & Thread Benchmark"
> reported figures like *18,698 FPS* and *12 threads*. It ran a batched
> `torch.matmul` on one thread, used the batch size as the "thread count", and
> touched no model, no pool and no execution provider — so it always picked
> roughly the largest batch that still fit. Those numbers meant nothing and the
> thread counts derived from them were arbitrary. See `roop/bench.py`'s docstring.

## Throughput / GPU concurrency

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_TRT_POOL` | unset (1) | Pool of N independent TensorRT **swapper** contexts (N≥2) to break single-context serialization. Validated ~+46% video throughput at 2. |
| `ROOP_DETMASK_POOL` | unset (auto) | Pool of N independent detect/mask sessions (FaceAnalysis + mask engines). Set explicitly (e.g. 2–8) to parallelize detection/masking. |
| `ROOP_DETECTOR_POOL` | = detmask pool | Independent instances of the standalone detector. Each hybrid engine brings its own detector and only borrows buffalo_l's aux models, so without this the detector stays single-file however wide `ROOP_DETMASK_POOL` is set, and widening that pool only adds queue time. Now honoured by **all three** hybrid engines (retinaface, yoloface, yunet) — before, only retinaface was pooled and the other two still held a detect-time mutex. Defaults to the detmask pool size so a worker never waits. `retinaface_r50.onnx` is ~104 MB per instance (yoloface_8n ~9 MB, yunet ~350 KB) — turn this down before the detmask pool when VRAM is tight. Now also has a UI setting (*Advanced performance → Detector pool*, `perf_detector_pool`), because the benchmark measures it separately from the detect/mask pool and the two disagree — on an RTX 4070 the detector was still scaling at 4 instances while recognition plateaued at 2. |
| `ROOP_TRT_WORKSPACE_FRACTION` | ~auto | Fraction of VRAM TensorRT may use as build workspace. Lower if a build OOMs. |
| `ROOP_TRT_PARTITION_ITERATIONS` | 2000 | TensorRT partition search iterations during engine build. |
| `ROOP_BATCH_SWAP` | 0 | Batch multiple face crops through one swap inference call (bit-identical). Phase-1 pixel-boost batching. |
| `ROOP_BATCH_SWAP_XFRAME` | 0 | Cross-frame swap batching collector. Requires `ROOP_BATCH_SWAP=1`. |
| `ROOP_BATCH_SWAP_MAX` | = threads | Max batch size for batched swap. |

## Stabilization (enhancer flicker / kps smoothing)

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_STAB_PARALLEL` | **1 (on)** | Run the flicker/kps stabilizers multi-threaded (contiguous blocks + derived warm-up). `0` restores the sequential path, which processes the whole clip on **one** thread — measured 2.5-3x slower on the swap pass. |
| `ROOP_STAB_2PASS` | 1 (on) | Two-pass kps stabilization (precompute pass 1 + parallel lookup pass 2) when enhancer-stab is off. `0` disables. |
| `ROOP_STAB_WARMUP` | auto | Warm-up frames per parallel block. Auto solves `(1-a)^W <= 1%` from the filter's own smoothing factor: 4 frames at strength 0, 6 at 0.5, 39 at 1.0. It was a fixed 4, which left 62% of the seed at strength 1.0 — a step at every block boundary. Override only to A/B a suspected seam. |
| `ROOP_STAB_CHUNK` | auto | Frames per parallel stabilization chunk (auto = `stab_width * max(4*warmup, 24)`, so each block is long enough to amortise its own warm-up). |
| `ROOP_STAB_CHUNK_MB` | 1536 | Memory budget for the decoded-frame chunk. If `4*warmup` blocks won't fit for every thread, the run stabilises narrower rather than shortening blocks below their warm-up. |
| `ROOP_STAB_BLOCKS_PER_THREAD` | 1 | >1 enables work-stealing dispatch to fix idle-thread imbalance in parallel stabilization. |

## Encoding / decoding

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_NVDEC` | auto (on) | Hardware NVDEC video reader. `0` disables; `1` forces (skips the auto probe). |
| `ROOP_NVENC_PRESET` | `p5` | NVENC encoder preset (`p1`–`p7`). |
| `ROOP_ENCODER_PRESET` | model default | CPU x264/x265 preset (e.g. `faster`) — lossless encode speedup at fixed CRF. |
| `ROOP_FFMPEG_COLORSPACE` | `bt709` | Keep the established BGR→BT.709 conversion, or explicitly use `off` when the producer already converted the frames. |
| `ROOP_OUTPUT_QUEUE_DEPTH` | auto | Explicit per-worker processed-frame handoff depth, bounded to 1–4; automatic mode remains RAM/profile controlled. |
| `ROOP_RESUME` | 1 (on) | Crash-resume: write segments every chunk + manifest so an interrupted run continues. `0` disables. |
| `ROOP_RESUME_SEGMENT_SECONDS` | 120 | Automatic resume-segment duration; converted to frames from the detected source FPS. Explicit `ROOP_RESUME_CHUNK` overrides it. |
| `ROOP_RESUME_CHUNK` | auto | Explicit frames per resume segment (min 50). Automatic mode uses `ROOP_RESUME_SEGMENT_SECONDS`; also controls the console part tabs and completion lines. |
| `ROOP_LIVE_PREVIEW` | 1 (on) | Publish the most recent processed frame for the processing box's live view (`/api/live_frame`). Throttled and downscaled, so unlike the per-frame full-frame copy this replaced, the cost does not scale with frame rate — measured 3.7 ms per publish on 1080p and 5.3 ms on 4K, i.e. ~1% of one thread at the default interval. `0` disables it and the box falls back to the last rendered preview still. |
| `ROOP_LIVE_PREVIEW_MS` | 500 | Minimum gap between published live frames. Lower is smoother and proportionally more expensive; a publish inside the gap costs one clock read. |
| `ROOP_LIVE_PREVIEW_WIDTH` | 960 | Width the live frame is downscaled to before encoding (min 160). 960 rather than the box's ~480 CSS px because that box is 2x on a HiDPI display, where a 480px frame is visibly pixelated. |
| `ROOP_LIVE_PREVIEW_QUALITY` | 88 | JPEG quality of the live frame (40-100). Lower shows blocking on flat skin, which makes the live view look worse than the render it reports on. |
| `ROOP_RESUME_KEEP` | 0 (off) | Keep the segment parts + manifest after a deliberate **Stop** so that run can be resumed later. Off = Stop merges the parts into one finished output and deletes them. Crash-resume is unaffected either way. |

## Runtime autotuning

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_RUNTIME_PROFILE_DIR` | `models/runtime_profiles` | Atomic cache directory for hardware/software/model/workload-specific profiles. |
| `ROOP_RUNTIME_PROFILE_INVALIDATE` | off | Invalidate the matching cached profile before deriving a fresh bounded policy. |
| `ROOP_RUNTIME_MONITOR` | off | Enable lightweight rolling runtime telemetry for end-to-end/stage FPS, latency, queues, workers, CPU, GPU, VRAM, and RAM. |
| `ROOP_RUNTIME_DIAGNOSTICS` | off | Print the final telemetry summary and every adaptive action; has no effect unless monitoring is enabled. |
| `ROOP_RUNTIME_ADAPTIVE` | off | Enable hysteretic safe-boundary adaptation of future batch/in-flight/buffer work. Active contexts and in-flight inference are never changed. |
| `ROOP_RUNTIME_SAMPLE_INTERVAL` | `1.0` | Minimum seconds between resource samples; stage timing remains aggregated without per-frame probes. |
| `ROOP_RUNTIME_WINDOW` | `16` | Number of resource samples retained for rolling decisions (bounded to 4-64). |
| `ROOP_CPU_P_INDICES` / `ROOP_CPU_E_INDICES` | unset | Optional comma-separated logical CPU indices for P-core/E-core utilization reporting when the platform exposes the topology. |

The normal application path loads a matching profile and does not repeat a
long candidate search on every launch. A manual retune uses short real
pipeline warmups, tests at most 12 candidates in staged order, and reports
end-to-end FPS plus VRAM/RAM/CPU/GPU, stability, quality, and startup costs.

When enabled, Phase 15 samples the live pipeline at a coarse interval and
classifies the rolling bottleneck. Adaptive changes require three consecutive
windows and then observe a cooldown; queue geometry, encoder changes, and
stabilization buffers are deferred to a safe next boundary/run. Normal runs
leave this telemetry and diagnostic output disabled.

## Terminal progress output

How a run reports itself in the **console** (Pinokio's log / a shell). The web
UI's progress bar and ETA are fed separately and are unaffected by all of these.

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_PROGRESS_STYLE` | `auto` | `auto` draws a live tqdm bar when stderr is a terminal and prints one compact line per chunk otherwise (a captured log cannot rewrite a line in place, so a redrawn bar becomes one 451-character line per frame). `bar` forces the live bar, `chunk` forces the per-chunk lines. |
| `ROOP_PROGRESS_EVERY` | 25 (or explicit `ROOP_RESUME_CHUNK`) | Frames per reported chunk. When an explicit resume chunk is configured, a line in the terminal covers the same stretch as a tab in the console's part strip. |
| `ROOP_PROGRESS_SECS` | 15 | Longest silence allowed between lines. A chunk can take minutes on a slow model, and a console that has said nothing for that long reads as a hang — whichever comes first, frames or seconds, triggers the line. |

## Timeline / preview frame decoding

These govern how the **preview and timeline** fetch single frames. None of them
touch the render pipeline, which uses its own readers.

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_SEEK_WALK` | 90 | Largest FORWARD frame gap the capturer will reach by decoding through the frames in between (`grab()`, ~0.7 ms each) instead of seeking. A cv2 seek costs a flat ~125-180 ms whatever the distance, so walking wins out to roughly 190 frames; scrub drags and frame-stepping are nothing but short hops. `0` restores seek-always. |
| `ROOP_PIPE_WALK` | 24 | Same idea for the ffmpeg-pipe reader (used for the HEVC formats cv2 mis-seeks). Lower because each skipped frame is a full decode and the alternative — respawning ffmpeg — is cheaper than cv2's seek. `0` disables. |
| `ROOP_FRAME_CACHE_MB` | 192 | Byte budget for decoded preview frames, kept so the raw preview, the swap preview and the hover thumbnail don't each seek for the same frame, and so scrubbing back over ground just covered is free. `0` disables the cache. |
| `ROOP_PREVIEW_PNG` | 0 | `1` sends the live preview frame as lossless PNG instead of quality-95 JPEG. Bigger and slower (≈1.4 MB vs 0.28 MB per 1080p frame); for pixel-peeping only. |

## Quality / precision

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_SWAP_FP16` | 0 | `1` lets the swapper run FP16 under TRT (**not recommended** — causes rainbow/smudge from FP16 overflow; default forces FP32). |
| `ROOP_GPEN_FP16` | 0 | `1` lets GPEN run FP16 (not recommended; default FP32 for correct faces). |
| `ROOP_UPSCALE_TRT` | 0 | `1` runs ESRGAN x4 upscalers under TensorRT (**not recommended** — goes all-black under TRT FP16; default forces CUDA/CPU FP32). |
| `ROOP_UPSCALE_TILE` | 256 | Tile size (px) for AI upscalers; lower if VRAM is tight on heavy ×4 models. |
| `ROOP_CAS_STRENGTH` | 0.5 | Contrast-Adaptive Sharpening strength for the `fsr` classical upscaler (0 = plain Lanczos). |
| ~~`ROOP_YAW_ALIGN`~~ | — | **REMOVED — this flag no longer exists.** The three angle-handling layers it drove were deleted outright in `31aa622`, not merely defaulted off: `estimate_norm` now does the bare similarity fit unconditionally, and there is no `yaw_align` key left anywhere in the app, the API or the UI (verified 2026-08-18). Setting it does nothing. The measurements that justified removing it are kept in *Angle handling* below, because they are the reason not to rebuild it the same way — read that section before proposing any new pose-based alignment. |

### Angled-face alignment modes

Fitting the 5 keypoints to a **frontal** template goes wrong two ways once the
head leaves frontal. At high yaw the eyes project to nearly the same point, so
the fit is ill-conditioned in rotation; and at any pitch the eye→mouth distance
foreshortens, so the template answers by stretching the crop. Measured against
the project's own 3-D reference head, over yaw 0–90° × pitch ±40°:

| | fit error @ 0° | @ 90° | crop-scale swing over the grid | rotation swing as the head nods ±25° @ 90° |
|---|---|---|---|---|
| `off` | 8 px | 60 px | **1.39×** | ~30° |
| `stabilize` | 8 px | 60 px | 1.39× | **<0.5°** |
| `pose` | 8 px | **~0 px** | **1.07×** | **<0.5°** |

- **`stabilize`** keeps the frontal template but takes the rotation from the
  eye→mouth axis, so pitch stops leaking into in-plane roll. This is a
  **temporal** fix — it reduces wobble *between* frames and will look identical
  on a single still, so judge it on video. It fades in from 40° off-axis and is
  fully engaged by 70°; below 40° it is bit-identical to `off`.

  It used to engage on a hard `yaw_ratio < 0.40` gate instead, which was the
  largest single source of per-frame wobble in the pipeline — the two fits are
  up to 30° apart in crop rotation, so crossing that gate was a step change of
  that size (18.4° in one frame along a turn-plus-nod sweep), and a *motionless*
  head parked on the gate had its crop rotating ±11° on detector noise alone.
  Because `yaw_ratio` is pitch-contaminated the gate also wandered — yaw 56° on a
  level head, yaw 66° with 20° of nod, and never at all below 90° once the nod
  reached 30°, so the mode was silently dead on the tilted profiles it was
  written for. The band fixes both: worst frame-to-frame rotation jump 0.09°
  (vs 0.10° with the mode `off`), and the nod-coupled swing at yaw 60° drops from
  20.1° to 2.8°.
- **`pose`** replaces the template with the reference head projected at the
  **solved yaw and pitch**. It solves the head pose from the 5 keypoints by
  weak perspective (accurate to <0.05° anywhere in the range, including a true
  90° profile) rather than inverting a scalar proxy.

  That 1.39× → 1.07× crop-scale figure is the one to care about: with a fixed
  template the pasted face changes size as the head turns and nods — it
  *breathes* — which reads as both misalignment and per-frame wobble.

  It fades in over 15°→40° off-axis rather than switching at a threshold,
  because a threshold **flickers**: the crop geometry differs by a finite jump
  either side of it, so a head parked near the boundary pops between two
  transforms frame to frame. Worst single-frame crop step along a turn-with-nod
  sweep: 0.145° rotation with the fixed template, 0.029° with `pose`.

  Note this supersedes a narrower earlier version of `pose` that was keyed on a
  yaw-only proxy. Pitch inflates that proxy, so a profile head that was *also*
  tilted read as a mid-angle face and got **no** correction at all — the most
  extreme poses in the range were the ones it silently skipped.

⚠️ Lower fit error is **not** automatically a better swap, and the "~0 px @ 90°"
above is measured against the **pose** template, which is congruent to the input
by construction. Against the template the models were actually **trained** on,
the residual grows 44 px → 85 px (512 crop) from frontal to profile *in every
mode* — no similarity transform can map a profile onto a frontal template. So
alignment cannot put an angled crop back inside the training distribution; it can
only stop the crop breathing and stop pitch leaking into roll. That is worth
having, and it is why `pose` is now the default, but it is not a fix for
extreme-angle distortion. The layer that addresses that is the off-axis fade
below.

### Angle handling — the three shared layers

Lateral and down-lateral faces distorting is the most-reported angle problem, and
it looks model-specific: hyperswap, hififace and inswapper each fail differently
at different angles, so it is natural to go looking in whichever swapper is
selected. It is not there. **All 13 swap models reach the same `align_crop`, the
same `paste_upscale`, and the same crop-space fade**, so all three corrections are
shared code and apply identically to every model:

| Layer | What it fixes | Control | Default |
|---|---|---|---|
| 1. Pose-matched alignment | Crop breathes 1.354× in scale over the pose sphere, so the model is handed a face at a size it was not trained on and the crop wobbles frame to frame. Holds it to 1.072×. | ~~`ROOP_YAW_ALIGN`~~ — **deleted in `31aa622`, no control remains** | removed |
| 2. Hidden-surface trim | Meant to stop a profile pasting swap pixels over hair and neck. Measured, **everything it actually removed was forehead** — see below. Its top edge is now opened up, which takes it to 0% at every pose on test geometry. Leave it off. | *Angle: trim hidden surface* | off |
| 3. Off-axis fade | Past ~55° off-axis the crop is outside every model's training distribution and the model stops reconstructing and starts inventing. Fades the swapped crop back toward the original footage instead. **This is the layer that bounds how wrong an extreme angle can look.** | *Angle: extreme-angle fade* (0–100) | 0 |

All three key on the **same** pose solve and share the same off-axis fade band, so
they are one continuous function of pose rather than three gates that can flicker
independently. Frontal faces are untouched by all three.

### All three ship OFF, and why

They shipped **on** on 2026-08-07 and were reported worse on real footage the next
day — flicker, a swapped face that does not match the original's size, and doubled
features, all on lateral poses. Sharing one pose solve is what makes them coherent;
it is also what makes them fail together, because **that solve is only as accurate
as one reference head is a match for the person in frame**, and in five keypoints
most of the yaw signal is carried by how far the nose stands out from the face:

| true yaw | 15° | 30° | 45° | 60° | 75° |
|---|---|---|---|---|---|
| reference head | 15.0 | 30.0 | 45.0 | 60.0 | 75.0 |
| nose +40 % | **24.6** | **44.6** | 59.6 | 71.3 | 81.1 |
| nose −40 % | **4.5** | **9.7** | **16.5** | **27.1** | 47.8 |

(`app/tests/test_pose_shape.py`, which pins these so they cannot drift.)

A prominent-nosed person turning 30° is read as 45°, which saturates the 15–40°
engagement band: they get the **full** pose-matched crop, the hidden-surface trim
and the fade all drawn for a pose they are not in. It is a per-person systematic
error, not frame noise, so no amount of temporal smoothing touches it — and a
flat-nosed person gets the opposite, no correction where it is needed. Fixing it
properly needs a pose estimate that is not a single-frame fit to a fixed head:
per-identity head-shape calibration over a clip, or the 106-point contour instead
of 5 keypoints.

Until then all three are opt-in, and the two whose failure modes are visible have
been bounded so that switching them on cannot produce the artefacts above:

* the **trim** can no longer remove more than 25 % of the matte, so a wrong pose
  costs a soft over-trim rather than half a face showing the original texture;
* the **fade** withdraws the swap from the outside in rather than cross-dissolving
  the whole crop, so it can no longer superimpose two differently-shaped faces and
  double the eyes, nose and mouth.

### Layer 2 only ever cut the forehead

Reported as *"with the trim on, only the middle of the face is swapped and the
rest keeps the original texture"*, with a seam across the brow. Measured against
the real paste mask and split by region, over yaw 0–75° × pitch −40…+20°:

| region | removed |
|---|---|
| below the brow line | **0.0 %** at every pose tested |
| forehead | 0.0 – 8.1 % |

**All of it was forehead.** Nothing else was ever removed at any pose — so two
earlier versions of this section, which described it as an under-chin correction
that fires on a tilted head, were describing the pitch *dependence* of a forehead
cut and calling it a chin cut.

The forehead is the one part of the face where neither shape has evidence. The
106-point landmarks stop at the eyebrows, so the paste mask's hull extrapolates
upward by 0.6 of the brow-chin distance and the visibility polygon extrapolates by
carrying the reference ellipsoid past the brows. Two guesses at the same unknown,
and the difference between them was being cut out of the face. On a head tilted
down they diverge most, which is where it was reported.

The trim's **top edge is now opened up** — a suffix-maximum down each column keeps
the left/right terminator, which is the real visible-surface information, and
discards its opinion about how high the face goes. On synthetic geometry that
takes the layer to **0.0 % at every pose**: everything it was measurably doing was
the artefact. Whatever value is left rests on a *real* detector's 106 points
over-claiming the far side of a real profile, which the synthetic head does not
reproduce and nothing here can measure. Off by default; do not reach for it.

**Why the safety test never caught this.** The property it asserts is "no
truly-visible landmark is trimmed", and there are no landmarks above the eyebrows.
A safety property expressed in terms of the evidence is blind exactly where the
evidence stops — which is where extrapolation happens, and where the bug was.

Cost: **+8.3 ms per off-axis face at 1080p (+6% of `paste_upscale`)**, and nothing
at all on frontal faces. It started at +29 ms; the difference was doing the weight
arithmetic at frame size (15 ms of float temporaries) and using an ellipse rather
than a rect structuring element for a 41×41 dilation (5.7 ms → 0.36 ms). If you
touch this, keep the arithmetic in crop space.

Layer 3 ramps from 0 at 60° off-axis to its full value at 90°, smoothstepped. The
setting is the **ceiling** reached at 90°, so 0 disables it and every face is
bit-identical to leaving it off. Raise it if extreme angles still distort; lower it
if profiles lose too much likeness. Note it fades toward the *plate*, so the faded
region also loses any enhancement — that is the intended "leave the original face
mostly alone" behaviour.

## Track stitching (the swap switching off for whole stretches)

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_TRACK_STITCH` | 1 (on) | Chain tracklets that are one person interrupted, on geometry, before any identity gate runs. `0` restores the previous behaviour. |
| `ROOP_TRACK_STITCH_GAP` | `45` | How many frames a face may be missing and still be the same person. |
| `ROOP_TRACK_STITCH_DIST` | `1.5` | How far it may have moved over that gap, as a multiple of its own width, from the position predicted by its last velocity. |
| `ROOP_TRACK_STITCH_SIZE` | `1.8` | How much its apparent size may change, as a ratio either way. |
| `ROOP_TRACK_STITCH_EMB` | `1.05` | Appearance **veto** — above this the two are clearly different people. Deliberately looser than every other identity gate; see below. |
| `ROOP_TRACK_STITCH_AMBIG` | `0.6` | How much better the best candidate must be than the runner-up before a link is taken at all. |
| `ROOP_TRACK_INHERIT_MAX` | `0.6` | Second pass: a fragment the first pass refused may inherit a source from a track that matched, if it is this close **to that track**. `0` disables. |
| `ROOP_TRACK_INHERIT_GAIN` | `0.15` | ...and only if the track explains it this much better than the captured stills do. Justifies a fragment that INTERLEAVES with the owner's track. |
| `ROOP_TRACK_INHERIT_MARGIN` | `0.25` | The other justification, for the shape the gain cannot reach: a later SHOT of the same person, which is disjoint from the owner by construction and whose gain is small because a close-up profile is unlike the frontal still *and* somewhat unlike the wide-shot track. Accepts when the track is at least this much nearer this person's tracks than anyone else's. Measured over the 15 tracks of a two-person clip with one frontal capture each: same person in a different shot **0.11-0.68** track-to-track, different people **0.85-1.08**, against photo distances of **0.17-1.05** that straddle the 0.60 gate with no gap at all. The smallest correct margin was 0.32. **Requires two or more selected people** - with one there is nobody to be further from, the margin is vacuous, and what would be left is a bare absolute bar on a disjoint track, which is exactly the bystander the containment rule exists to refuse; with one person this path does not exist. The pass runs to a fixed point, so identity propagates along the clip (shot 1 vouches for shot 2, which vouches for shot 3). `0` disables it. |

Reported as *"the swap flickers enormously when the face touches an object, at
lateral poses, whatever the mask engine"*. The audit from that clip:

```
15 tracks over 287 frames, 2 matched to a source (gate 0.60)
faces seen 1784, swapped 780, NOT swapped 1004 (56.3%)
   of the refusals: "track matched but has no source"  951
```

So the swap was off for whole stretches, not single frames, and the cause sits
upstream of every identity gate: **the track broke.** For the same person a
profile sits 0.7–1.0 in cosine distance from a frontal capture, which is past the
scan's association bar (`EMB_MAX` 0.7), past the source-assignment gate (0.60)
and past the per-frame fallback (0.75). A turned or occluded stretch therefore
becomes a track of its own — and that fragment is then judged on a mean built
entirely from the frames that broke it. Nothing downstream can recover from that,
which is why no threshold and no mask engine changed anything.

The link that survives what appearance cannot is **spatio-temporal**: a track that
ends here and one that begins a moment later, in the same place, at the same size,
is one person interrupted. Fragments are chained on that, before any identity gate
sees them and before the mean is finalised — so a chain's identity is averaged
over all of its segments, which is also a better estimate than any fragment had
alone. Appearance is demoted to a veto for the clearly impossible.

The two failure modes are **asymmetric**, and the gates are set accordingly: a
missed link costs what the pipeline already did, while a wrong link hands one
person's face to another for a stretch. So it is one-to-one (each fragment takes
at most one predecessor and gives at most one successor) and it refuses ambiguity
outright — a fragment with two comparable candidates is left alone rather than
guessed at.

Look for `(stitched down from N)` in the `[Track]` line to see it working.

### When the fragment overlaps the track instead of following it

Stitching only joins tracks separated by a **gap**. A person whose track keeps
breaking mid-shot produces fragments that *interleave* with the main track
instead — same stretch of clip, alternating frames — and those cannot be chained.
They still have to be judged, and judging them is where the second pass comes in.
From a real clip (one target, one bystander, 717 frames):

```
track  0   715 frames   0.27  -> source 0
track  2   133 frames   0.72  -> NO SOURCE (over the gate)
track  1   715 frames   1.05  -> NO SOURCE      (the other person)
tracks 3,6,8,9,10       0.93-1.07 -> NO SOURCE
```

Track 2 is 19 % of the clip, sitting in the band where a person's own turned or
badly-lit stretch lives, and it was thrown away while the bystander sat 0.33
further out. **No threshold fixes that** — 0.72 against a *photograph* is
genuinely ambiguous, and loosening the gate to admit it admits the 0.93 too.

It is not ambiguous against the **track**. A track that ran through the same clip
shares the camera, the lighting and the grade, and its mean averages many poses
rather than a handful of captured angles. So a refused fragment is compared to
the track that matched, and may inherit its source — under three conditions, each
of which exists because removing it reintroduces a bug that has already been
reported:

* **close to that track** (`ROOP_TRACK_INHERIT_MAX`);
* **the track explains it better than the photo does** (`ROOP_TRACK_INHERIT_GAIN`)
  — a stranger is equally far from both, because the matched track *is* the person
  in the photo, so proximity alone cannot separate it from the target on a bad
  stretch;
* **contained inside the track's span while never sharing a frame with it** — a
  broken-off stretch interleaves, a bystander appears in a run where the target is
  off screen. This is the distinction `ROOP_TRACK_ASSIGN_MARGIN` was built on, and
  without it the second pass hands the bystander the swap all over again.

Shown as `-> source 0 (via track 0, d=0.35)` in the per-track block.

## Interacting faces (two or more swaps that touch)

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_FACE_DEMARCATE` | 1 (on) | Draw a boundary between faces that overlap, instead of letting each one paste its whole matte. `0` restores the old behaviour — use it to check whether an artefact on interacting faces comes from this or from the swap itself. |
| `ROOP_FACE_DEMARCATE_FEATHER` | `0.02` | Width of the hand-over band where one face blends into the other, as a fraction of face radius (~4 px on a 200 px face). Larger mixes the two faces over a wider band; smaller risks aliasing along the join. |
| `ROOP_FACE_DEMARCATE_DEPTH` | `0.10` | How much on-screen size counts as "in front", per doubling of face radius. `0` treats every face as the same distance away, so the boundary sits on the geometric midline regardless of who is nearer the camera. |
| `ROOP_FACE_DEMARCATE_MIN_BAND_PX` | `16.0` | Floor on the hand-over band, in PIXELS. Detection noise moves a boundary by a fixed number of pixels rather than a fixed fraction, so without a pixel floor the join between two large faces shimmers frame to frame. |
| `ROOP_FACE_DEMARCATE_MAX_BAND` | `0.16` | Cap on that floor as a fraction of face radius, so a small face is not turned into all ramp and no face. |
| `ROOP_FACE_DEMARCATE_DUP` | `0.6` | How much of one claim must sit inside the other before the two are treated as **one face detected twice** rather than two faces. A duplicate is dropped instead of competed with — left in, it wins half of the swapped face's own pixels and carves it apart, on only the frames where the duplicate was detected, so it flickers. Measured both ways round: which of the two boxes ends up holding the source is arbitrary. |
| `ROOP_FACE_DEMARCATE_DUP_SEP` | `0.35` | ...and how concentric they must be, as centre separation over the smaller claim's radius. Containment alone is not enough: it is also 1.0 for a genuinely separate smaller head nested in a bigger one's padded box, and dropping *that* claim is the smear this whole feature exists to remove. Duplicates measure 0.07–0.24 here, separate faces 0.52–1.08. |

### Before the pixels: the detector and the recogniser (`roop/face_contact.py`)

The flags above decide who owns a pixel once both faces have been swapped. Two
things go wrong earlier than that, and they are what produce "the two people
swap into each other / flicker / go un-swapped, but only when they are close".

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_FACE_MERGE` | 1 (on) | Drop the phantom detection the detector fires **at the junction between two touching faces** — the left person's mouth and chin plus the right person's nose and mouth, which really does look like one frontal face. On the measured clip it appears on 346 of the 1074 two-face frames, at det_score up to 0.99, so no confidence floor removes it, and NMS deliberately protects it (it is offset from both parents, which is exactly the signature `ROOP_NMS_CENTER_FRAC` exists to spare). Left in, it gets a track of its own, its embedding is a chimera of two people, it competes for a source (one source per frame, so it can take the real face's), it competes for pixels, and it comes and goes frame to frame — a flicker on **both** real faces. `0` keeps them. The run reports how many it dropped in the SWAP AUDIT. |
| `ROOP_FACE_MERGE_COVER` | `0.80` | How much of the candidate must be covered by its two neighbours together. A junction box is *made of* them; a real face has at least the far side of its own head showing. |
| `ROOP_FACE_MERGE_EACH` | `0.15` | Minimum overlap with **each** neighbour, as a fraction of the candidate's area. |
| `ROOP_FACE_MERGE_SIZE` | `0.60` | Size floor against the smaller neighbour. A junction spans two faces and is never much smaller than either; a distant head framed between two nearer ones is, and this is what keeps it. |
| `ROOP_FACE_MERGE_PAIR_SEP` | `0.35` | The two neighbours must be two different faces, not one face detected twice — same measurement and same value as `ROOP_FACE_DEMARCATE_DUP_SEP`. |
| `ROOP_FACE_MERGE_BETWEEN` | `0.15` | How far along the a→b axis the candidate's centre must sit to count as *between* them (0.15–0.85). This is what spares three real faces in a row: their neighbours sit on the **same** side of the middle one, a junction's parents sit one on each side. |
| `ROOP_EMB_CONTAM` | `0.35` | Fraction of a face's **recognition crop** that another face may cover before its embedding stops being used to decide identity. ArcFace fits five keypoints to a frontal template, and for a near-profile face that resolves to a crop about 1.7 box widths across, reaching well past the nose. While the faces are apart that extra area is background; once they touch, the two crops converge onto very nearly the same picture and the identities converge with them. Measured, distance from each face to its own reference against how much of its crop the other one covers: `<0.2` → own 0.05 / other 1.06, 0 % wrong; `0.2–0.3` → 0.43 / 1.02, 0 %; `0.3–0.4` → 0.40 / 0.99, 2 %; `0.4–0.6` → 0.63 / 0.92, **14 % read as the wrong person**. Past ~0.4 the mean distance to the *right* person is already over the 0.60 track-assignment gate. A flagged face keeps its place in its track on position, is left out of the track's mean embedding, is denied appearance-only Re-ID, and is not offered to the per-frame identity matcher. `0` disables the whole mechanism. |

Masking the neighbour out before recognition does **not** work and was measured:
a flat fill inside the crop destroys the embedding outright (own-distance 0.18 →
0.74 on frames where the plain crop was still correct). The pixels cannot be
removed because they are inside the face's own aligned crop — so the embedding
is not repaired, it is disbelieved, and position carries those frames instead.

Reproduce with `env/Scripts/python.exe tests/two_face_video.py --tag before` (add
`ROOP_FACE_MERGE=0 ROOP_EMB_CONTAM=0` for the baseline).

### What goes wrong when two swapped faces meet

Each face is pasted under a matte built from its own geometry alone — a hull
that has been dilated, given a forehead extension reaching 60 % of the face
height past the brow, then feathered outward. It knows nothing about anyone
standing next to it, so when two faces touch, three things go wrong at once:

* **Smudging.** Every face after the first used to read from the *running
  composite*: its alignment crop, its colour reference, the mask engine's view
  of it and its mouth/eye plates all came from a buffer that already held the
  neighbour's swapped pixels. The swapper was being fed a chimera. Every read is
  now taken from the untouched plate (`process_face(..., plate=)`); only the
  write goes to the composite.
* **Bleeding.** Face A's matte spills a band of A's swap over B's cheek, and
  A's invented forehead across B's face. Ownership (`roop/face_overlap.py`)
  trims each matte where the other face owns the pixels.
* **Flipping.** Whoever is pasted *last* wins the contested pixels, and the
  paste order used to be match order — which is not stable frame to frame (in
  identity-lock mode the faces are re-sorted by identity distance every frame).
  Faces are now painted far-to-near on a quantised size key, so the nearer one
  stays on top for as long as it is nearer.

Ownership is decided from geometry, once per frame, before anything is pasted:
each face gets a signed distance field to its own hull, normalised by its own
size so a near face and a far one are compared on equal terms, and a pixel goes
to whichever face is deepest inside relative to its own scale. Faces that do not
interact are detected by bounding-box test and nothing is computed for them, so
the ordinary single-face frame pays nothing and cannot be changed by this.

The boundary is enforced by paint order rather than by splitting contested
pixels 50/50 between the two faces. Splitting looks right and is wrong: the
mattes are applied in sequence, so two at 0.5 give `0.5·B + 0.25·A + 0.25·plate`
— a hairline of untouched footage down the join. Instead each face defends its
territory only against faces painted *before* it, and carries on one hand-over
band into the territory of faces painted *after* it, so the face underneath
backs the ramp of the face on top and the plate never shows through.

## Masking convention overrides

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_NONFRONTAL_MASK` | `auto` | Which masking path a dense masker (XSeg/XSeg3/occluder/faceparser/clip2seg) takes. `auto` routes by the non-frontal test; `0` always masks in canonical crop space; `1` always masks on the unwarped bounding-box crop. Use `0` to isolate whether a bad profile frame is caused by the mask *routing* or by the swap itself. |
| `ROOP_NONFRONTAL_HYST` | 1 (on) | The routing latch. `0` reverts to the bare per-frame threshold. See below. |

### Why the routing decision is latched

The non-frontal test picks between two *different* mask derivations, so when its
verdict changes the mask boundary moves. Driven straight off a per-frame score,
that verdict chatters: on a **still** head under 1 px of keypoint noise it
flipped up to 215 times in 600 frames (an ordinary head tilted up ~30°, sitting
right on the threshold). The mask edge visibly moves on a head that is not
moving at all.

So the verdict is latched with hysteresis — enter above 1.15× the threshold,
leave below 0.85×, hold in between. The band is sized from the score's measured
noise (~0.16–0.19 spread near the threshold); 0.15 is the first half-width that
zeroes every hot spot. Measured verdict flips per 600 frames:

| pose (still head) | no latch | 1 thread | 4 | 8 | 16 |
|---|---|---|---|---|---|
| yaw 0, pitch +30 | 215 | **0** | **0** | **0** | **0** |
| yaw 5, pitch +30 | 194 | **0** | **0** | **0** | **0** |
| yaw 10, pitch +30 | 156 | **0** | **0** | **0** | **0** |
| yaw 0, pitch −30 | 30 | **0** | **0** | **0** | **0** |

A genuinely turning head still re-routes every time it crosses — that is the
point of a latch rather than a longer smoothing window.

**How it survives multithreading.** Workers get frames round-robin
(`frame % threads`), so each walks the whole clip at stride N and adjacent
frames belong to different workers. A latch is an evolving state, and neither
obvious way of sharing one works: one shared state machine gets driven through
every crossing N times over (measured *worse* than no latch), while a state
machine per worker makes each worker cross at a slightly different frame, so a
moving face ripples at every transition.

The way out is that a hysteresis latch is equivalent to a **query** —
*"the value of the most recent decisive frame at or before t"* — which is a pure
function of the frame index, not an evolving state. Workers publish decisive
frames into one shared index-keyed log and all read the same answer back,
whatever order they arrive in.

Flips on a head nodding across the threshold (400 frames, 4 genuine crossings,
so **8** is the correct answer and **16** is what no latch gives):

| arrival order | flips |
|---|---|
| sequential (1 thread) | **8** — exact |
| reordered within 4–48 frames (worst of 30 runs) | **11** |
| reordered without limit | 28 — *worse than no latch* |

Only the first two can happen. Reordering is bounded by the pipeline's own
plumbing: one reader deals frames round-robin into per-thread `Queue(3)` and
blocks when any of them fills, so drift stays around three rounds — 24 frames at
8 threads. The unbounded row is what a microbenchmark with no per-frame work
produces, and is listed only because it is the reason the test pins a window
instead of shuffling freely.

The residual is a worker querying frame *t* before a sibling has published a
decisive frame in *(t−N, t]*, and it sits at genuine transitions, where the two
mask paths agree most closely anyway. `ProcessMgr.process_face` calls
`observe()` as soon as the pose is known — a whole swap and enhance before the
verdict is needed — which shrinks that window; once a frame's neighbours are in
the log, the result is completely order-independent. Closing it entirely would
mean workers waiting on each other, which is not worth a stall in the render
loop.
| `ROOP_OCCLUDER_RAW` | 0 | `1` skips the Face Occluder mask inversion (flip polarity if the mask is inverted). |
| `ROOP_XSEG3_RAW` | 0 | `1` skips the XSeg3 mask inversion. |

## AI-upscale second pass

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_UPSCALE_NVENC` | 1 (on) | NVENC encode for the post-swap AI upscale pass. `0` opts out. |
| `ROOP_UPSCALE_THREADS` | 4 | Worker threads for the upscale pass (heavy models auto-cap to 1). |
| `ROOP_UPSCALE_MAX_DIM` | 8192 | Max output dimension for GPU-encoded upscale output; raise for very large frames. |

## Detection / tracking

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_NMS_CENTER_FRAC` | 0.25 | Centre separation, in face-widths, above which two overlapping detections are treated as **two touching faces** rather than one face detected twice — suppression now requires the boxes to overlap *and* be concentric. Plain NMS answers both questions with the single `face_detector_nms` (0.40) IoU number, and the second one loses: two same-size boxes reach IoU 0.40 only by sharing 57% of their union, which ordinary side-by-side contact never does, but one head partly **behind** another at similar scale does — and the face further back is deleted before anything else in the pipeline sees it, so it is never tracked, matched or swapped. Duplicate boxes of one face are concentric (the anchors are centred on the same thing) while two faces have centres a real distance apart however much their boxes overlap. Bounded: for equal boxes 0.25 face-widths ≈ IoU 0.60, so the effective threshold widens from 0.40 to ~0.60 and only for offset pairs. Shared by **all five engines** — the two that suppress inside insightface (the default `scrfd` engine and `retinaface` 10g; both arrive as `insightface.model_zoo.retinaface.RetinaFace` for these model files, whatever the engine is named) get it bound onto the detector instance, site-packages untouched, and YuNet, whose NMS is inside OpenCV, runs OpenCV permissively and suppresses here instead. `0` restores plain NMS exactly (asserted against the previous implementation in `test_nms.py`). Raise it if a single face starts being swapped twice; lower it toward 0 if faces in contact were never the problem. |
| `ROOP_TEMPORAL_GAP` | 10 | Max consecutive detection misses gap-filled by the tracked detection pre-pass. |
| `ROOP_TEMPORAL_STEP` | 1 (**launcher sets `2`**) | Scan stride for the "Analyzing faces" pre-pass. The pre-pass is detection-bound — measured `track_wait` 10.81 ms/frame against `track_detect` 43.52 ms over a 4-instance pool, i.e. the loop is blocked on the detector ~97% of the time — so `2` roughly halves it, and the pre-pass is ~30% of a long run. Skipped frames are filled by *linear* interpolation, which is exact for steady motion and lags a fast head turn; set back to `1` in `start_react.js` for whip pans / quick cuts. Capped at `ROOP_TEMPORAL_GAP`. |
| `ROOP_INTERP_MAX_TRAVEL` | 0.5 | Continuity guard on gap-fill: bridge a detection gap only if the face could have travelled between the two anchors, at most this many face-widths **per skipped frame**. The scan's Re-ID fallback matches on embedding alone with no spatial constraint, so a track can jump across the frame between consecutive observations; filling that in manufactures a face for every frame in between, sliding over the background. Those invented faces defeat every identity check by construction (their embedding *is* the track mean) and, since a source is used at most once per frame, the real face in those frames is then refused. `0` disables the guard. |
| `ROOP_INTERP_MAX_SCALE` | 2.0 | Companion size guard: refuse to bridge when the two anchors differ in width by more than this factor (a close-up and a distant face are not the same observation). `0` disables. |
| `ROOP_TRACK_READAHEAD` | 1 | Decode frames for the pre-pass on their own thread so decoding overlaps detection (~15% off the pre-pass; bit-identical). `0` decodes inline as before. With it on, the `track_decode` stage times the wait *for* a frame, so it only grows if the decoder is genuinely the slower half. |
| `ROOP_TRACK_ROI_CROP` | 0 | `1` enables ROI-crop pre-pass during identity tracking. |
| `ROOP_CAPTURE_PIPE` | auto | How preview/timeline decode video. `auto` uses OpenCV except on HEVC pixel formats where it silently returns the wrong frame (10-bit, 4:2:2, 4:4:4 — measured up to 16 frames off), which go through an ffmpeg pipe instead. `1` forces the pipe for everything (slower seeks, always exact), `0` forces OpenCV (fast, wrong on those formats). |

## Upside-down / heavily rolled faces

## Swap outcome guard

Past ~90° of yaw a head shows cheek and ear and no face at all, but the detector
still returns one at 0.985 confidence, every gate passes it, and the swapper
paints a complete frontal face onto the side of a head pointing away. No test
built on POSE separates that from a legitimate profile — two were measured and
discarded — but the OUTCOME does: a real swap leaves the eyes, nose and mouth
where they were. So the result is re-detected and the swap is undone if the
keypoints moved (`face_util.swap_moved_the_face`; the discards show up in SWAP
AUDIT as *"the swap put the face somewhere it was not"*).

That is a detection per swapped face, and as first shipped it was an unconditional
**full** face analysis — 11.7 ms against a per-face budget of roughly 210 ms for
swap + mask + enhance, of which 6.6 ms computed a 512-d embedding and 174
landmark points that were discarded on the next line. It now runs the detector
alone (**+49.6%** on the guard, bbox and keypoints bit-identical) and only where
the failure is reachable at all. It also reports into STAGE TIMING as `verify`,
so it can never again be a cost with nothing in the table to blame.

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_VERIFY_SWAP` | **1 (on)** | The guard itself. `0` disables it — every swap stands, including the ones painted onto the back of a head. |
| `ROOP_VERIFY_SWAP` | 1 (on) | Master switch for the outcome guard. `0` turns it off - previously only reachable by pushing `ROOP_VERIFY_MIN_OFFAXIS` past any angle a head can reach. Worth having because the guard's cost is not symmetric: on two people in tight profile it re-detects a 120px face whose eyes are six pixels apart, and even after the shape term below it discards 79 of 2282 swapped faces (3.5%) on the measured clip. Turning it off there takes the second person from 95.1% of frames swapped to 98.1% and her on/off transitions from 39 to 15. Leave it on for ordinary footage - it is the only thing between a head turned past 90 degrees and a complete frontal face painted on its cheek. |
| `ROOP_SWAP_SHAPE_TOL` | `0.75` | Second condition the guard must ALSO meet before discarding: how much the five-point constellation changed after removing translation, normalised by its own extent. Scale-free, and unlike the displacement test it does not divide by an interocular distance that has collapsed to a few pixels. Displacement alone discards **28.4%** of correct swaps on a two-person contact clip, because a legitimate swap of a 120px profile moves the points ~10px and ten pixels over six is 1.7. Measured: correct swaps (n=229) **max 0.730, nothing above 0.75**; the yaw +-90 studio sweep (n=148) is bimodal - 76 under 0.75, which look correct in the contact sheet, and a second mode from **1.0 to 4.3** with doubled mouths and ghosted features. Requiring both conditions can only ever discard a subset of what displacement alone discarded, so the per-model tolerances keep their meaning. **Known limit:** a frontal face pasted on a profile at the same orientation *and* extent measures 0.706, below the correct-swap maximum, so that particular hypothetical is not separable by this metric at any threshold (nor by displacement without the 28% collateral). `0` restores the displacement-only rule. |
| `ROOP_VERIFY_MIN_OFFAXIS` | 30 | Degrees of yaw *or* pitch below which the re-detect is skipped. A rolled face (one autorotate turned) is always checked regardless. `0` checks every face, which is the behaviour the guard shipped with. Placed off the readings, not the nominal angle: over the four yaw ±90 plates of `app/tests/angle_video.py` — where the guard refuses 119–134 of 131 faces per clip — the lowest value read here is **43.6°**, and the frontal plates read 9.6° at worst. 30 leaves 13.6° of margin on the side where being wrong costs a wrecked swap, which is what `solve_pose_5pt`'s 15–20° per-person head-shape error needs. Verified equal: same discard counts (133/123/134/119–120) with the filter on and off. |

## Upside-down / heavily rolled faces

Between roll ~140° and ~220° the detector does **not** lose the face: it reports
~0.98 confidence and returns a self-consistent set of 5 keypoints describing an
*upright* face, with the two "mouth" points on the forehead. Worst error of each
candidate orientation axis over a full turn, measured by
`app/tests/frontal_roll_video.py`:

| axis | worst error over 0-360° |
|------|------------------------|
| 5 detector keypoints | 172° |
| 2D-106 chin→forehead midline | 179° |
| **3D-68 eye-mid→mouth-mid** | **5.4°** |

The first two fail on the same frames *in the same direction*, so their
agreement is not evidence, and nothing derived from the 5 keypoints separates
the cases (the arcface fit residual, eye-to-mouth ratio, nose offset and bbox
placement were each measured across the turn and all overlap the healthy range
at roll 180). So the 68-point model is required, and `autorotate_faces` pulls it
in.

**It is not, however, run per face.** Keeping it in the analysis loop cost
**2.23 ms per face — 19% of detection throughput** (230 → 187 detections/s over
4 threads; RTX 4070, TensorRT, retinaface_r50 @640), paid on every frame of
every clip to re-answer a question whose answer on ordinary footage never
changes. Since the pre-pass *is* detection, that came straight off the whole
run. So when `autorotate_faces` is the only requester the model is loaded but
kept out of the per-face loop, and run on demand for the frames whose
orientation is actually in doubt (the pose features — 3D recon, source bank,
frontalization, landmark refine — read the landmarks of every face they touch,
so when any of those is on it stays in the loop as before).

"In doubt" is decided by the cheap keypoint axis, which is allowed to say *when
to ask* but never *what the answer is*:

* **the ramp** — the keypoint axis already reads ≥ `ROOP_LM68_ARM_DEG`. A head
  reaching the blind band has to rotate through 20–140° first, where the
  keypoints are right, so a roll is caught on the way in;
* **the probe** — every `ROOP_LM68_PROBE`-th detection regardless, which is what
  covers a cut *straight to* an already-rolled head: there is no ramp to see, so
  nothing else can catch it. At the default the worst case is ~0.4 s at 30 fps;
* either arms the model for `ROOP_LM68_HOLD` detections, so a rolled sequence is
  measured continuously rather than rediscovered frame by frame — and a probe
  that lands on a head the keypoints are reading wrong arms it too.

Verified end to end on the full-turn clip with the shipped defaults
(`app/tests/frontal_roll_video.py`): **90/90 frames swapped, 0 inverted**, and
identity 0.97–0.99 against the target through the entire 140–220° band — the
same result as running the model on every face.

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_LM68_ARM_DEG` | 20 | Keypoint-axis tilt (degrees) that arms the 68-point model. Well inside the band where the keypoints are still accurate to 9.1°. |
| `ROOP_LM68_PROBE` | 12 | Measure the 68-point axis every Nth detection whatever the keypoints say. `0` disables the probe, leaving only the ramp — which cannot see a cut to an already-inverted face. |
| `ROOP_LM68_HOLD` | 90 | How many detections the model stays armed for once something asks for it (~3 s at 30 fps). |
| `ROOP_UPRIGHT_REMEASURE` | **1 (on)** | Re-detect a heavily rolled face on an uprighted frame and adopt that reading (keypoints, landmarks **and embedding**), gated on the turn actually having stood the face up. Fixes the geometry the detector crushes when the head is inverted — interocular 177→136 px and eye→mouth 202→134 px at roll 180, which puts the recognition embedding at ~0.0 cosine against the *same person* upright, below the 0.128 two-different-people floor, so the face stops matching the selected target and is never swapped. Costs one extra detection pass **only for frames containing a rolled face** — it fired on 0/381 frames of ordinary footage. `0` restores the old behaviour for an A/B. |
| `ROOP_DEBUG_ROLL` | 0 | Dump the track latch's per-observation estimate, resolved roll and whether it was believed. The summary coast count alone cannot tell a latch that usefully crossed an ambiguous pocket from one that rejected good readings and walked the track away from the truth. |

## Process priority

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_PRIORITY` | `high` | Windows process priority to dodge EcoQoS background throttling (`above_normal` \| `normal` to relax). |

## Ports

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_API_PORT` | 8001 | FastAPI/React backend port. |
| `ROOP_GRADIO_PORT` | CFG `server_port` | Legacy Gradio UI port. |

## Debug / diagnostics

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_PROFILE` | 0 | `1` prints per-stage wall-clock timing (analyze/detect/mask/swap/enhance/…) summed across worker threads. Unrelated to `ROOP_YAW_ALIGN`. |
| `ROOP_DEBUG_MATCH` | unset | Prints identity-matching diagnostics during the swap pass. |
| `ROOP_DEBUG_ANGLE` | 0 | `1` prints, per face per masker: the yaw/pitch keypoint proxies, the non-frontal verdict, which masking path ran, and what fraction of the canonical crop the unwarped box covers. Noisy on video — use on a single preview frame. |

### SWAP AUDIT (always on, no flag)

Every run in identity-tracking mode prints a breakdown at the end of why each
detected face was or was not swapped:

```
==== SWAP AUDIT — why each detected face was or was not swapped ====
  faces seen                          48210  100.0%
  swapped (identity lock)             44907   93.1%
  veto: single-person absolute         2611    5.4%
  fallback missed (over match thr…)    2611    5.4%
  ...
  -> 3303 of 48210 detected faces (6.9%) were NOT swapped.
```

This exists because *"the swap flickers on and off"* is indistinguishable in the
output between the gates that can cause it — a frame where a face was found but
nothing was painted looks the same whichever gate refused it. The largest
refusal line names the gate to loosen:

| Line | Gate to look at |
|------|-----------------|
| `veto: single-person absolute` | `ROOP_TRACK_VETO_SINGLE` — the per-frame absolute veto misfiring on occluded/hard frames |
| `veto: far from assigned person` | `ROOP_TRACK_VETO` (multi-person) |
| `veto: another person fits better` | `ROOP_TRACK_VETO_MARGIN` |
| `veto: source used twice in frame` | Two faces claimed one source — overlapping faces, or a false detection claiming first |
| `no track entry matched` | `ROOP_TRACK_EMB_MAX` refused to associate the face with any track |
| `fallback missed (over match threshold)` | Per-frame matching also failed — raise **max face distance** |
| `  of those, gap-filled` | How many swapped faces were *invented* by gap-fill rather than detected (see `ROOP_INTERP_MAX_SCALE` / `ROOP_TEMPORAL_GAP`) |

Counters are unsynchronised increments across worker threads — a lost count is
harmless for a breakdown meant to show relative magnitude.

## Identity tracking

Cosine distances are scipy convention (0..2). A same-person **profile** frame sits
0.7–1.0 from a frontal one, so these gates are deliberately loose; wrong-person
matches are rejected by *relative* cross-person comparisons at the call sites,
not by tightening these absolute cutoffs.

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_TRACK_VETO` | 0.85 | Distance beyond which a tracked source is refused for a face (guards ID switches, crossings, bystanders). `0` disables the veto — a tracked source then applies wherever spatial association points. |
| `ROOP_TRACK_VETO_MARGIN` | 0.15 | Also veto when a *different* selected person explains the face this much better. |
| `ROOP_TRACK_EMB_MAX` | 0.7 | Appearance gate for track association, shared by the tracking scan and the swap-time re-association. A detection this far from the track's identity is refused outright rather than merely penalised — standard tracking-by-detection practice (BoT-SORT `appearance_thresh`). The scan always did this; the swap-time side did not, which is how a track could hand its source to whoever stood closest. `0` disables the swap-time gate. |
| `ROOP_ADAFACE` | off | Use **AdaFace** (matching-only) instead of w600k to decide who is who. w600k still feeds the swapper, so swap output is unchanged. All-or-nothing per run: if any captured target face has no aligned crop, the run stays on w600k. Requires a calibrated `ROOP_ADAFACE_DIST` — measure it first with `tools/calibrate_identity.py`. |
| `ROOP_ADAFACE_DIST` | 0.5 | Match threshold on AdaFace's **own** distance scale. `max_face_distance` does not apply to it. The tuned veto constants are rescaled automatically by the ratio to this value, preserving "the veto is looser than the match gate". The 0.5 default is a placeholder — calibrate it. |
| `ROOP_TRACK_VETO_SINGLE` | 0 (off) | Absolute veto for the **single selected person** case, which `ROOP_TRACK_VETO` deliberately skips. Written to catch a tracker identity switch back when a single-person tracked swap had *no* swap-time identity check at all. `ROOP_TRACK_EMB_MAX` is now that check and runs first, so a face reaching this veto has already been matched within 0.7 of a track mean that itself passed `ROOP_TRACK_ASSIGN_MAX` against the captured person — identity is established twice over, from the track mean rather than one frame. What remains is the failure its own comment warns about: the distance is computed from the **current** frame, which is exactly what occlusion corrupts, so a hand or a passing face pushes it over, the source is vetoed, the face falls to per-frame matching at the tighter threshold, that fails too, and the frame goes unswapped — visible as flicker. It also never refused gap-filled faces (they carry the track mean by construction), so it was hardest on frames where detection had succeeded. The launcher previously pinned this to `1.0`; it is now left off. Re-enable only if strangers get swapped **and** the run's SWAP AUDIT shows no `veto: single-person absolute`. |
| `ROOP_TRACK_ASSIGN_MAX` | 0.6 | Gate for binding a **track** to a source, capped by `max_face_distance` (the tighter wins). Deliberately stricter than per-frame matching: the decision is durable (every face on that track is swapped for as long as it runs) and is made from the track's *mean* embedding, which is much cleaner evidence than one frame. Measured: a real person's track mean sat at 0.36 while background/blur false detections clustered at 0.85–1.0 — i.e. right where the per-frame threshold sits, which is how a 33k-frame clip bound 16 of its 81 tracks to one selected person. A refused track is not dropped; its frames fall through to per-frame matching. `0` restores the old behaviour (gate == `max_face_distance`). |
| `ROOP_TRACK_OVERLAP_FRAC` | 0.15 | Fraction of a track's frames that must overlap an already-assigned track of the same person before it counts as a genuinely concurrent second body rather than an occlusion handoff. |
| `ROOP_TRACK_REID_MAX` | 0.5 | Appearance bar for the scan's **Re-ID fallback** when the candidate is a **retired** track (unseen for STALE=15 frames). A track seen more recently than that keeps `ROOP_TRACK_EMB_MAX`: recency is itself evidence, and the faces reaching Re-ID against a recently-seen track are the occluded / motion-blurred / partially-detected ones the tracker exists to carry — a face crossed by an object is often detected on a shrunken box that misses the predicted one, so it lands here with a degraded embedding. Applying the tight bar there breaks those frames into a track of their own and blinks the swap off exactly when something passes in front of the subject (verified: it fails `test_occluded_frames_stay_on_the_same_track`), which is the regression `ROOP_TRACK_VETO_SINGLE` was reverted for. Retired tracks have neither spatial nor temporal evidence and are generally re-acquired *un*occluded, so there is no hard-frame allowance to make. It used to share `ROOP_TRACK_EMB_MAX` (0.7) with the primary IoU+appearance path, i.e. the association with the least evidence behind it was held to the standard of the one with the most; standard tracking-by-detection does the opposite (BoT-SORT/ByteTrack hold the fallback stage to a stricter threshold). The symmetric version is how an unselected face joined the target's track: someone entering the shot for the first time has no track of their own to win the nearest-match comparison, so that single 0.7 is all that stands between them and the target's retired track — and different people, normally 0.93–1.07, drop well under it on a profile or motion blur. Once absorbed they inherit the target's source, inside a single track where the assignment margin cannot see them and (with one selected person) no swap-time veto runs. 0.5 is the bound the track already uses to decide a detection is too far off to update its own `emb_mean`. A refused Re-ID does not drop the face — it starts its own track, judged on its own mean — so the cost is more fragments, not lost swaps. Raise toward 0.7 if a target stops locking after every turn; `0` restores the old shared gate. Reported by `[Track] … N detections refused by the appearance-only fallback`. |
| `ROOP_TRACK_ASSIGN_FLOOR` | 0.45 | Floor under `ROOP_TRACK_ASSIGN_MARGIN`: below this distance the margin never refuses a track. The margin is *relative*, so an unusually good anchor makes it unusually strict — a clean frontal capture matching a clean frontal track anchors near 0.10, which would then refuse that same person's profile-heavy fragment at 0.40, a distance nothing else in the pipeline treats as a stranger. Typical anchors (0.30–0.40) put the margin at 0.45–0.55 anyway, so this only bites at the good end, and it stays clear of the 0.5–0.6 band the margin exists to cut. Matters most alongside `ROOP_TRACK_REID_MAX`, which trades a tighter Re-ID for more fragments — each of which then has to pass this gate. |
| `ROOP_TRACK_ASSIGN_MARGIN` | 0.15 | How much further than a person's **closest** track a later track of theirs may sit and still be bound to the same source. A person legitimately owns several tracklets (tracking fragments constantly — 60–130 tracks on a 23k-frame clip), and `ROOP_TRACK_OVERLAP_FRAC` refuses only the ones running *concurrently* with a track they already own. The converse does not hold: a bystander's fragment lying entirely inside a stretch where the target is **off screen** is concurrent with nothing, so that guard never examines it, and it inherited the target's source for exactly those frames — "when the target isn't in the frame, the other face gets swapped". An absolute gate cannot separate that from a bad stretch of the real target; the distance to the person's own best track can (their fragments cluster near it at ~0.36, a stranger scraping under `ROOP_TRACK_ASSIGN_MAX` sits at 0.5–0.6). A refused track is not dropped — it loses identity *locking* and falls through to per-frame matching at the full threshold, so a genuine target fragment still swaps. Raise it if a target stops locking after re-entering a shot; `0` disables. Reported by `[Track] … N refused as too far from their person's closest track`. |
| `ROOP_TRACK_TRUEMEAN` | 1 (on) | Identity-lock matches on the true mean embedding. `0` restores the old recency-biased EMA (the "only the first faceset swaps" behaviour). |

## Temporal identity output cost

The temporal identity layer remains opt-in through `ROOP_TEMPORAL_IDENTITY=1`.
Its output correction is low-frequency, so the default experimental path works
on a bounded crop before restoring the correction to the current frame. The
current frame remains the base image; expression and fine texture are not
replaced by history.

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_TEMPORAL_IDENTITY_LOWPASS_SIZE` | 128 | Working-crop maximum for the two Gaussian low-pass passes. Smaller values reduce CPU work but increase the approximation; `0` retains the full-resolution reference path for diagnostics. |

## Multi-angle target bank

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_ANGLE_MANUAL_MAX` | 0.90 | Max distance accepted by `/api/target/add_angle` (manual capture). |
| `ROOP_ANGLE_ACCEPT` | 0.60 | Distance under which `/api/target/auto_angles` accepts a harvested angle. |
| `ROOP_ANGLE_SEED_MAX` | 0.85 | Max distance for a seed frame in `/api/target/auto_angles`. |
| `ROOP_ANGLE_LONE_ACCEPT` | 0.45 | Tighter `ACCEPT` for frames where auto-capture's **relative** guards cannot run. Its cross-person and runner-up checks are relative — which is why they are strict without punishing a hard pose — but both need a second face: the runner-up check is literally `len(scored) > 1`, and the cross-person one needs *another captured person*. On a frame holding one face, and by definition on every frame where the target has walked off, neither runs and only absolute distances remain. Tightens the distance to the **bank** rather than to the seed on purpose: a genuine profile arrives by chaining, so it sits near an intermediate angle already banked and its bank distance is small, while a marginal admission sits out near the limit — tightening the seed distance instead would have hit the profiles hardest. Raise toward `ROOP_ANGLE_ACCEPT` if auto-capture stops finding angles in single-person footage. |
| `ROOP_ANGLE_BLUR_FRAC` | 0.5 | Refuse to bank a crop whose sharpness is below this fraction of the **median sharpness of the faces scanned so far**. A blurred crop's embedding collapses toward the middle of the space and sits near everybody, which is the realistic way a face that is not this person clears the absolute gates. Relative, not absolute: measured on face-like crops the sharpness axis reads 0.96–1.00 clean and 0.03 mildly blurred, but the whole range slides with grain, compression and focus, so a fixed line either passes everything on a crisp clip or rejects everything on a soft one. The composite FIQA score cannot do this job either — a heavily blurred *200px* face still totals 0.505, because size and detector confidence prop up exactly the frame whose embedding is worthless. `0` disables. |
| `ROOP_ANGLE_BLUR_WARMUP` | 8 | Candidates needed before the blur median is trusted; below it nothing is refused for blur. |
| `ROOP_ANGLE_MIN_PX` | 64 | Hard floor on banked face size — below this there is not enough detail for a reliable embedding (matches the FIQA resolution axis's own zero point). |
| `ROOP_ANGLE_MIN_QUALITY` | 0.35 | Weak backstop on overall intake quality, judged by `face_quality.image_quality` — the FIQA composition with **frontality removed**. Using the normal composite here would be exactly wrong: it weights frontality, so it would reject the profiles auto-capture exists to collect while admitting a crisp frontal frame of the wrong person. |
| `ROOP_ANGLE_REVIEW` | 0.70 | Banked angles this far from the original capture are listed in the response (and the toast) as worth checking. **Not** a rejection — a true extreme profile lands here too. It exists because pollution is otherwise invisible: a wrong angle looks like any other thumbnail, and its cost arrives much later as the wrong person being swapped, since every swap-time gate takes the *minimum* over a person's angles and one bad entry speaks for all of them. |

## Face quality (FIQA)

Composite heuristic score used by the Face Manager's quality gate — a weighted
blend, not a trained network. Weights need not sum to 1; missing components are
renormalised away.

| Flag | Default | Effect |
|------|---------|--------|
| `ROOP_FIQA_W_DET` | 0.25 | Weight of the detector confidence term. |
| `ROOP_FIQA_W_SHARP` | 0.30 | Weight of the sharpness term. |
| `ROOP_FIQA_W_RES` | 0.20 | Weight of the resolution term. |
| `ROOP_FIQA_W_POSE` | 0.15 | Weight of the frontality term. |
| `ROOP_FIQA_W_NORM` | 0.10 | Weight of the embedding-norm term. |

## Expression restore (LivePortrait)

The **😀 Expression restore** slider and **Expression region** selector live in
the Face Swap tab. One env flag exists, for the GridSample rewrite below.

| variable | default | meaning |
|---|---|---|
| `ROOP_EXPR_PATCH_GRIDSAMPLE` | `1` | Rewrite the warping module's 5-D `GridSample` nodes so it runs entirely on the GPU. `0` keeps the stock model and its CPU partition. |
| `ROOP_EXPR_POOL` | `0` (off) | Independent restorer instances so the stage runs N-wide instead of serialised. **Costs ~Nx its VRAM** — see below. |

### What the feature costs, and the one stage still running single-file

Measured across three runs of the same pipeline, 192-frame chunks on an RTX 4070:

| run | expression restore | mean chunk | max |
|---|---|---|---|
| off | — | **9.00 s** | 11.34 s |
| on | stock (CPU partition) | 17.49 s | 30.09 s |
| on | rewritten (below) | **13.53 s** | 16.79 s |

So the rewrite cut the feature's cost by ~23%, and what remains is ~4.5 s per
chunk over having it off. Almost all of that adds **in series**: expression
restore is the only GPU stage without a session pool, so while swap, mask and
detect run N-wide behind `_gpu_guard(pooled=True)`, this one holds the global
lock.

`ROOP_EXPR_POOL=N` gives it independent TensorRT contexts and removes that.
Measured with 8 threads sharing one restorer, as ProcessMgr uses it:

| `ROOP_EXPR_POOL` | throughput | per call | VRAM held |
|---|---|---|---|
| 0 *(serialised)* | 28.7 faces/s | 34.8 ms | 1162 MB |
| **2** | **39.2 faces/s** | 25.5 ms | 1822 MB |
| 3 | 38.6 faces/s | 25.9 ms | 2518 MB |
| 4 | 37.4 faces/s | 26.7 ms | 3180 MB |

**2 is the sweet spot: +37% for +660 MB.** Past that the GPU is saturated and
extra contexts cost more than they return. Note that pool 0 with 8 threads lands
at exactly the single-threaded 34.8 ms, which is what "fully serialised" looks
like. Output is **bit-exact** against the unpooled path, and identical across
slots — the pool changes scheduling only.

Still **off by default**, and deliberately not VRAM-auto-tuned like
`ROOP_TRT_POOL`/`ROOP_DETMASK_POOL`, whose defaults were chosen before these
models existed. A 12 GB card running 4 swapper + 4 detmask instances measured
1.0 GB free mid-render, so `=2` fits with roughly 380 MB to spare — enough, but
not comfortable. If it is unstable, drop `ROOP_TRT_POOL` to 3.

The +37% is the isolated gain. In the real pipeline the pooled stage also stops
holding the global lock, so it can overlap with swap/mask/detect as well; that
part is unmeasured and can only be larger, not smaller.

### The constraint

`warping_spade.onnx` warps a 5-D feature volume `(1,32,16,64,64)` with two
`GridSample` nodes, and **nothing in the GPU stack executes them as shipped**:

* TensorRT's `IGridSampleLayer` is documented 4-D only, so its parser rejects
  both nodes (`addGridSample ... nbDims == 4` / `INVALID_NODE` for
  `/dense_motion_network/GridSample` and `/GridSample`).
* onnxruntime's CUDA GridSample kernel is likewise 4-D only and fails at run
  time with *"Only 4-D tensor is supported"*.

onnxruntime's answer is to partition just those two nodes to CPU. Profiled per
call: 134 ms for the `(22,4,16,64,64)` node plus 12 ms for the
`(1,32,16,64,64)` one — 146 ms, against 1.5 ms for the same maths on the GPU.

### The rewrite (default on)

`roop/gridsample5d.py` replaces each 5-D `GridSample` with its definition —
eight corner gathers and a trilinear weighted sum — using only ops TensorRT
builds natively. The result is cached beside the source as
`warping_spade-trt.onnx`, with a `.meta` stamp recording `PATCH_VERSION` and the
source mtime so an edited rewrite cannot serve a stale graph.

Measured on an RTX 4070, warping module only, FP16 TensorRT, error against an
FP32 CPU run of the stock model:

| path | median | max abs err |
|---|---|---|
| stock — TensorRT + 2 nodes on CPU | 165.2 ms | 4.99e-02 |
| **rewritten — full TensorRT** | **26.8 ms** | 4.71e-02 |
| rewritten — CUDA only | 87.8 ms | 4.40e-03 |
| stock — CPU only *(ground truth)* | 2784 ms | — |
| stock — CUDA | ❌ fails outright | — |

**6.16x on TensorRT**, and it gives CUDA-only machines a working GPU path for
the first time. Accuracy is unchanged: the rewritten graph's FP16 error is no
larger than the stock TensorRT path's.

End-to-end, a whole `Expression_LivePortrait.Run()` (appearance + motion x2 +
stitching + warping + the numpy pre/post) goes **237.8 ms -> 34.0 ms per face,
7.0x** — lifting the stage's throughput ceiling from ~4 to ~29 faces/sec.

The gain is larger than the 146 ms of CPU kernel time because the partition cost
more than the kernels. Profiling the provider assignment shows the stock model
was **9 separate TensorRT subgraphs** wrapped around the 2 CPU nodes, with a
device round-trip at every boundary; the rewritten model is **1 engine**.

Two constraints that are not obvious and cost real debugging:

* **The flattened index is computed in INT32, never in float.** TensorRT runs
  this graph with FP16 enabled; FP16's largest value is 65504 with a spacing of
  32 near the top, and the big node addresses 16·64·64 = 65536 voxels. With the
  index in float the module was still 6.7x faster and *agreed to 3e-06 on CPU* —
  but produced 0.71 max abs error on a [0,1] output under TensorRT.
* **Applied only when a GPU provider is present.** On CPU the expansion is ~9%
  *slower* than onnxruntime's own kernel (2.66 s vs 2.44 s): eight gathers beat
  one fused kernel only when there are cores to spread them over.

### Fallback path

With `ROOP_EXPR_PATCH_GRIDSAMPLE=0`, or if the rewrite fails, the processor
picks providers for that one session itself: TensorRT if available, otherwise
CPU, never CUDA. The parser errors are then silenced around session
construction — `SessionOptions.log_severity_level` does **not** work for this,
since the messages come from the TensorRT logger bridged through onnxruntime's
*global* logger, so the global severity is raised for that one session build and
restored afterwards. On the rewritten path they are not silenced, because the
rewrite removes their cause and any remaining error is real.

FasterLivePortrait also publishes `warping_spade-fix.onnx`, which swaps the node
for a custom `GridSample3D` op. That needs a plugin library this project does not
ship — the model fails to load at all — so it is deliberately not used.

Models (~537 MB) download to `app/models/liveportrait/` on first use.

## Face enhancer cost (measured)

The enhancer is ~36% of a render's wall clock and the swap phase runs the GPU at
~98%, so its share of GPU work is the main lever on speed once threading is
fixed. Isolated cost of one 512² face, RTX 4070, TensorRT FP16, engines warm
(±15% run to run):

| enhancer | ms/call | vs RestoreFormer++ | est. effect on total render |
|---|---|---|---|
| GFPGAN v1.4 | ~12 | 0.50x | ~18% faster |
| GPEN-BFR-512 | ~18 | 0.79x | ~8% faster |
| CodeFormer | ~22 | 0.96x | ~1% faster |
| RestoreFormer++ | ~23 | 1.00x | baseline |

Speed only — quality is a separate judgement and belongs to whoever is watching
the output. Reproduce with `app/tools/bench_stages.py` (covers enhancer, mask and swap models).

### Enhancer concurrency, and what it costs in VRAM

Cost per call is only half the story. `_gpu_guard` waives the global TensorRT
lock **only for a processor that owns a `SessionPool`** — so an unpooled
enhancer serialises this stage to one thread while every other worker blocks on
the lock, and since enhance is ~36% of wall clock that is a hard ceiling **no
`max_threads` value can lift**. For a serialised stage of service time `S`
against `P` of other per-face work, throughput is `min(T/(S+P), 1/S)` and
saturates at `T* = 1 + P/S` — about 4 threads with CodeFormer on a 4070.

Pooled today: **RestoreFormer++**, **CodeFormer**. Still unpooled (and so still
serialised): GFPGAN, GPEN (all sizes), DMDNet, KEEP.

Measured pool cost, RTX 4070, `trt_precision: mixed`, 4 contexts, **each one
actually run**:

| enhancer | 4 contexts | per extra context |
|---|---|---|
| RestoreFormer++ | 3345 MB | 683 MB |
| CodeFormer fp16 | 2763 MB | 530 MB |
| CodeFormer fp32 | 2700 MB | 560 MB |

Reproduced twice; the per-extra-context figures are stable to 1 MB
(529/530, 560/560, 682/683), while the first context varies with whatever CUDA
context init is already on the card.

What the pool actually buys, same machine, 8 calls over 4 threads, all contexts
warmed, serial and concurrent interleaved over 5 rounds (median):

| | serial | concurrent | speedup |
|---|---|---|---|
| CodeFormer fp16 | 41.4 ms/call | 27.4 ms/call | **1.51x** |
| CodeFormer fp32 | 37.3 ms/call | 28.9 ms/call | **1.29x** |

Not 4x, because a single CodeFormer inference already keeps the GPU busy. This
understates the change: it isolates the enhancer, whereas the pool's real job in
a render is letting other workers run mask/swap/paste **while** one enhances,
instead of every thread blocking on the global lock.

**`Codeformer (fp16)` buys nothing under `trt_precision: mixed`** — same VRAM,
and its serial ms/call is if anything slightly worse than the fp32 tier's (the
two overlap run to run). `mixed` builds FP16 kernels from either set of weights,
so the fp16 ONNX has nothing left to give. Its measured 1.60x (162.9 -> 102.0
ms) is a **CUDA** result and only applies with `provider: cuda`.

Three traps before re-deriving any of this, each of which returns a confident
wrong number:

- **ONNX file size does not predict VRAM.** fp32 is 359 MB on disk against
  fp16's 180 MB, yet costs slightly *less* on the card. What lives in VRAM is
  the engine, not the file.
- **Creating a session is not what allocates the execution context.** Measured
  without running an inference, every extra context looks like ~10 MB and the
  whole question looks free.
- **Warm up every context before timing.** The first concurrent batch pays each
  extra context's first-inference allocation (~530 MB). Timing that against a
  serial batch that runs afterwards on warm contexts reported **0.10x** — i.e.
  concurrency ten times slower — which is pure ordering artefact.

`trt_precision: fp32` builds genuinely FP32 engines and is **not** covered by
that table. Each pooled enhancer falls back to a single session behind the lock
if building the extras runs out of memory, which is what carries that case.

**Benchmarking ONNX outside the app — three traps**, each of which silently
yields CPU numbers that look like a plausible result (measured 483 ms/call for
GFPGAN on CPU vs 12 ms on TensorRT, a 43x error):

1. `import torch` BEFORE `onnxruntime` makes ORT reject the CUDA EP.
2. `tensorrt_libs` must be on the DLL path.
3. That alone is not enough — `onnxruntime_providers_tensorrt.dll` also needs
   `cublas64_12.dll`, which in this environment lives in
   `env/Lib/site-packages/torch/lib`. Add the directory (that does not import
   torch); without it ORT falls back to CPU for the WHOLE session, not to CUDA.

Always assert `sess.get_providers()[0]` rather than assuming.

## Mask engine cost (measured)

RTX 4070, TensorRT FP16, isolated, one crop:

| engine | ms/call |
|---|---|
| Face Parser (BiSeNet) | 2.45 |
| DFL XSeg | 2.94 |
| Face Occluder v3 (XSeg-3) | 4.69 |
| Face Occluder | 5.02 |
| MobileSAM (encoder) | 5.82 + decoder |
| FastSAM | 5.84 |

**The engine choice is not a speed lever.** The whole spread is ~3 ms against a
masking stage measured at ~42 ms/face — nearly all of that stage is the CPU work
around the model (landmark hull, mouth mask, blurs, non-frontal unwarp), not the
model. Pick on how well each handles your occlusions. SAM2 is excluded: it runs a
whole-clip pre-pass rather than per-crop inference, and MobileSAM's decoder needs
real encoder output, so the harness cannot feed it synthetically.

## Swap model cost (measured)

**Per FACE, not per inference.** Pixel boost tiles any model smaller than the
subsample size and runs it `(subsample/output_size)**2` times
(`procmgr_tiling.implode_pixel_boost`), so ms/call alone ranks these wrongly.
At the 256px default:

| model | ms/call | tiles | per face |
|---|---|---|---|
| ghost_1 | 3.74 | 1 | **3.74** |
| hyperswap_1a | 5.28 | 1 | 5.28 |
| hyperswap_1c | 5.29 | 1 | 5.29 |
| hyperswap_1b | 5.58 | 1 | 5.58 |
| hififace | 6.14 | 1 | 6.14 |
| simswap | 6.72 | 1 | 6.72 |
| ghost_3 | 11.41 | 1 | 11.41 |
| simswap_512 | 11.58 | 1 | 11.58 |
| blendswap | 13.20 | 1 | 13.20 |
| reswapper | 16.37 | 1 | 16.37 |
| uniface | 19.38 | 1 | 19.38 |
| **inswapper** | 5.27 | **4** | **21.08** |

`inswapper` is 128px: second-cheapest per call, **most expensive per face**, and
it is the original project default. Everything else is 1 tile at 256.

## Identity-specific detail restoration

FaceSet V2 archives cache a signed canonical high-frequency residual with
per-pixel persistence confidence. The video path uses the cross-reference
consensus map, adapts amplitude to target exposure/local contrast, excludes
structural eye/nose/mouth regions, and gates the result with the generated-face
visibility mask. It does not paste a source crop or read target texture for
this feature. Restoration runs after enhancers, post-enhance colour matching,
merger operations, and temporal low-band blending.

| Setting | Default | Meaning |
|---|---:|---|
| `identity_detail_strength` | 0.0 | API/config setting, 0..1. Opt-in source identity-detail restoration. V1 archives and strength 0 are no-ops. |

Benchmark: `app/env/Scripts/python.exe app/tests/bench_identity_detail.py`.
Current synthetic result: 0.839016 detail-retention correlation and 43.3257%
reduction in alternating-detail temporal delta; this is not a real-footage
acceptance score.

The spread is ~17 ms against a swap stage reporting ~46 ms/face, so — as with
masking — the model is the minority of the cost and this is a quality choice
rather than a speed one. The enhancer is the only stage where the model itself
dominates (see the enhancer table above).

## Target-conditioned lighting and colour realism

The requested Phase 10 appearance path extends the existing target-referenced
colour transfer; it does not create a second texture-transfer implementation.
The source still determines identity. The aligned target crop supplies robust
exposure, white balance/scene cast, spatial low-frequency illumination,
shadow/highlight rolloff, local contrast, and skin-region chroma. No target
high-frequency texture is copied by this feature.

| Setting | Default | Meaning |
|---|---:|---|
| `target_conditioned_appearance` | `false` | Opt-in target-conditioned appearance matching. |
| `target_conditioned_appearance_strength` | `0.75` | Bounded low-frequency lighting/chroma blend, 0..1. |
| `target_conditioned_appearance_temporal_alpha` | `0.30` | Per-track EMA admission for stable lighting; higher follows changes faster. |
| `target_conditioned_appearance_cache_size` | `256` | Bounded per-track appearance-state cache. |

The detector classifies each aligned target as `NORMAL`, `DARK`, or
`VERY_DARK`. DARK blends restorers back toward the target-conditioned input and
halves merger clarity/sharpening. VERY_DARK uses a stronger input guard and
lower sharpening, so GPEN/UltraMax/other restorers cannot turn a dark face into
a bright clean face or hallucinate exposure. The stabilizer is cloned for the
existing contiguous block path and primed with its derived warm-up; it does
not disable the 4070 pools or the 3060 single-context/RSS safety policy.

The React control sends these settings through both preview and swap payloads.
The reproducible component benchmark is:

```text
app/env/Scripts/python.exe app/tests/bench_target_appearance.py
```

The 2026-09-01 host result covered daylight, indoor, tungsten, fluorescent,
sunset, blue, mixed, night, street-light, low-exposure, and backlighting
fixtures. Appearance analysis averaged **23.3602 ms/call**; the stable-light
colour temporal delta fell from **0.02481532** to **0.00440974** (**82.2298%**
reduction). This is component evidence, not a visual acceptance score.

## Adaptive enhancer orchestration (Phase 11)

`selected_enhancer: Adaptive` enables per-face/frame selection over the existing
GPEN 256 Pro, GPEN Realistic, and UltraMax processors. All manual enhancer
choices remain independent and unchanged. The selector can choose no enhancer
for a high-quality face, a light path for dark/low-quality input, a stronger
path for moderate quality, or hold/omit during unstable, occluded, very-dark,
or extreme-angle frames. It never sequentially runs the candidate set.

| Setting | Default | Meaning |
|---|---:|---|
| `adaptive_enhancer_profile` | `BALANCED` | `FAST`, `BALANCED`, `REALISTIC`, or `MAX QUALITY` preference policy. |
| `ROOP_ADAPTIVE_MAX_LOADED` | auto | Optional cache bound, clamped to 1..3; sub-7 GB defaults to 1. |

Adaptive is opt-in and is not a replacement for RealityUX, RealSwap, GPEN
256 Pro, GPEN Realistic, UltraMax, or other existing paths. The wrapper's
telemetry reports selected paths, reasons, recent metrics, and observed output
quality. Video measurements are produced by
`app/tests/bench_adaptive_enhancer_video.py`.
