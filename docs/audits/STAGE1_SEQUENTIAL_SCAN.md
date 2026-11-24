# Stage 1 sparse pre-scan: sequential NVDEC decode (2026-11-24)

Scope: `app/roop/sparse_range_scan.py` (Stage 1 of the 3-stage render). RTX 4070, the live `config.yaml` models
(`retinaface_r50`, `max_face_distance 0.75`), real clips from `roop-keep`. **Not covered:** Stage 2 (pre-pass) and
Stage 4 (finalize) were read, not changed or timed; the RTX 3060, DirectML and CPU rows were not run; no full render
was run with this change (the card was shared with a running app, see "Not verified").

## What was wrong

The hardware path (`_scan_active_ranges_hw`, in since `cb28f0a`/`500dfca`) sampled **keyframes** (plus a midpoint
where the GOP exceeded 60 frames) and read each sample with `decoder[i]`.

1. **It missed faces.** Real clips have 2-6 s GOPs (b1 mean 2.2 s / max 10.5 s, Monica 5.7 s, Love 4.1 s), so
   samples were 1-4 s apart. The dilation pad came from the *nominal* stride (`S = 4` -> `M = 7` frames), not from
   the real sample gap. Every missed face frame goes out unswapped.
2. **It was slow for the wrong reason.** `decoder[i]` re-decodes from the previous IDR. With a stub detector the scan of
   3000 frames of b1 took 9.2 s: 5.1 s in 164 cut-localisation probes (31 ms each) and 3.4 s in 159 sample reads
   (21 ms each). Detection ran on one thread; the cv2 path's detector pool was not used.
3. **A stop or a detector error returned a partial answer.** The cv2 path fails safe to the full range; the hardware
   path `break`-ed out of its loop on `should_stop` and returned the ranges found so far. Detector errors were
   counted and ignored.
4. **Its result was never checkpointed.** The mode label `nv_hardware_keyframes` is not in
   `stage_cache.CACHEABLE_STAGE1_MODES`, so a resumed job re-ran Stage 1.
5. A stream that ended before the window did left every later frame "inactive" (unswapped) on both paths.

## What changed

* `run_demux_hw`: decode **every** frame of the trimmed window sequentially (`NvZeroCopyReader`, `read_batch(16)`),
  compare each with the previous on a 64x36 GPU luma thumbnail, one sync per chunk. Cuts are exact frames, so nothing
  is bisected. Samples: the first frame of every shot, every `S` frames inside it, the window's last frame.
* `ShotLedger`: a shot with a positive sample is already fully active (`build_active_ranges` expands an island to its
  shot), so the demuxer stops sampling it and the workers drop queued samples of it.
* One orchestration for both sources: the worker pool (`prescan_pool_size()` instances), the barrier, the warm-up, the
  fail-safe reasons. The hardware path now fails safe on stop, demux error, detector errors with no positive, and a
  decode shorter than the window (`decoded N of M frames`). The cv2 path is unchanged.
* `build_active_ranges(expand_to_edges=True)` for the sequential source: the clip's first and last shot expand to the
  clip edge like every other shot (default `False`: the cv2 path and its tests are untouched).
* `mode` is `sparse_auto_range` again (cacheable); the source is in `stats['decode_backend']`
  (`nvdec_sequential` / `cv2`), plus `shots`, `skipped_samples`.
* The decoder tag stays `stage1-keyframes` (report, HUD badge, `face_bank` and the fallback relabel read it).

## Measurements

Ground truth = production `get_all_faces` (full rescue ladder) on every frame of the window. "persistent" ignores
truth runs shorter than 3 frames (isolated single-frame detections 100+ frames from any cut; the tracker does not
swap them). Probe: `app/tests/probe_stage1_sequential.py` (`truth`, then `scan`). Seconds are one scan, warm.

| window | frames | mode | old (keyframes) s / active / missed | new (sequential) s / active / missed raw (persistent) |
|---|---|---|---|---|
| b1 3000-6000 (192 cuts) | 3000 | universal | 15.87 / 84.5% / **78** of 1235 | 8.06 / 97.3% / 3 (**0**) |
| b1 3000-6000 | 3000 | targeted | 15.56 / 64.3% / **67** of 313 | 11.07 / 87.2% / 12 (**0** of 85) |
| Monica 500-3500 (56 cuts) | 3000 | universal | 13.29 / 100% / 0 | 2.58 / 98.4% / 0 |
| Monica | 3000 | targeted | 12.44 / 96.0% / 0 | 3.80 / 92.6% / 0 |
| Love 600-2100 (11 cuts) | 1500 | universal | 2.85 / 100% / 0 | 1.17 / 100% / 0 |
| Love | 1500 | targeted | 3.95 / 90.0% / 0 | 2.97 / 92.5% / 0 |
| Weeds 0-1770 (15 cuts) | 1770 | universal | not run | 2.79 / 86.4% / 0 |
| Weeds | 1770 | targeted | not run | 5.55 / 75.6% / 0 |
| b1 12000-15000 (284 cuts) | 3000 | universal | not run | 10.82 / 97.2% / 2 (**0**) |
| b1 12000-15000 | 3000 | targeted | not run | 20.98 / 31.5% / 0 |

Love universal/targeted with the cv2 source for reference: 3.49 s and 11.80 s (92.9% active), 0 missed.

* **Cost of the recall, said plainly:** on b1 the new mask is wider (84.5% -> 97.3% universal, 64.3% -> 87.2%
  targeted). The old scan's lower figure was partly the 78 / 67 face frames it never looked at. Stage 2 and 3 now do
  that work. On Monica it is narrower, on Love slightly wider (targeted).
* **Frame alignment:** NVDEC cut indices against dense cv2 cut indices: Love 11/11, Monica 56/56, Weeds 15/15, b1
  190/191 (one cv2-only, two NVDEC-only; threshold-edge flicker). Decode-only scan of 3000 frames: 1-2 s.
* **Detector-bound:** worker utilisation 89-99% on b1 (2 workers). A negative sample costs 4 detector passes (upright
  plus three rescue turns); b1 is 60% negative.

## Measured and NOT shipped (each with its number)

* **Rescue turns off** (`rescue=False`): b1 universal 8.06 -> 3.84 s but 10 persistent face frames lost on b1 12000
  and 13 on b1 3000-6000 targeted. Kept on.
* **Stricter universal det gate** (per-sample scores replayed offline against truth, 5 windows): upright >= 0.50:
  15 persistent frames lost, active 95.5% -> 92.1%; up >= 0.50 | turn >= 0.70: 9 lost, 93.2%. Current
  `any >= 0.35` loses 6 offline (3 on b1 12000, 3 on Monica) and 0 in the real scans of the same windows. No variant is both safe and cheaper.
* **No whole-shot expansion (pad only):** active 95.5% -> 92.9%, persistent frames lost 6 -> 14. Kept.
* **Coarser grid (offline, truth-detector):** step 8/12/16 on b1 targeted loses 27 / 39 / 50 of 85 persistent
  frames. Step `S` kept.
* **Targeted similarity floor 0.15 -> 0.25** (`SIM_RELAX` 0.12 -> 0): zero persistent frames lost on the five windows,
  mean active 76.3% -> 71.1% (b1 12000: 32% -> 20%, Love 92.8% -> 79.1%); 0.28 starts losing frames. **Not shipped:**
  the 0.12 margin is the guard for faces whose embedding dips on the sampled frame (pose, blur) and five windows of
  one operator's footage are not the population. A margin of 0.05 gets 73.0%. Owner decision.

## Tests

`app/tests/test_sparse_range_scan_sequential.py` (20, no GPU: grid, exact cuts, flash shorter than the stride, chunk
boundary, ledger early exit, stop, refused sample, short stream, failed download, edge expansion) and
`NvdecSequentialScanTest` in `test_sparse_range_scan.py` (a real MJPG file through NVDEC: cut at exactly frame 30,
trim offsets, a stream shorter than the window -> `fail_safe_full`). The cv2 alignment test is pinned to
`ROOP_HW_IO=0`. Four tests that patched the removed `_scan_active_ranges_hw` now patch `NvZeroCopyReader`.
Run: 53 + 20 + the wider Stage 1 set (hardware fallback, filmstrip/cuts, stage cache, video_io_hw) green; light CI
profile: 2746 passed, 7 failed, none in files this change touches (3 `test_lipsync_acoustics`, 1
`test_phase_harness_contracts` naming `test_nvenc_writer_lut_kwargs.py`, 3 `test_update_manifest_head`, which names
`react-ui/package.json`).

## Not verified

* No full 3-stage render with this change: the card was at 5 GB / 23% utilisation from a running app, and a render
  holds 12-15 GB (`AGENTS.md`: one render at a time). The regression benchmark (`run.py --benchmark --benchmark-mode
  regression`) and `test_pipeline_3stage_sync_and_fps.py` must be run on an idle card. By reading, that test's Stage 1
  assertions (cuts exactly `[60, 120]`, one range `(0, 154)`, decoder `NVDEC-DLPack`) hold.
* RTX 3060 (6 GB, single-context tier: how many detector workers `prescan_pool_size()` gives it was not checked), DirectML, CPU: not run. VFR sources: not run through the new source.
* Stage 2 and Stage 4: no change and no timing. Stage 2's detector cost on frames with no face (the full rescue ladder
  runs only on an EMPTY result, about twice a face frame on b1: 15.5 fps dense against 22-24 on face-dense clips) is
  the cost a tighter Stage 1 mask would save; the measurements above say the mask cannot be tightened without losing
  frames, so that lever is closed for now.
