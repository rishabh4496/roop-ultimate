# Regression benchmark

```
python run.py --benchmark --benchmark-mode regression        # from app/
portable\run.bat --benchmark --benchmark-mode regression     # portable runtime
```

| flag | default | |
|---|---|---|
| `--benchmark-frames N` | 300 | frames in the timed render |
| `--benchmark-clip PATH` | generated | the default is a deterministic synthetic 1080p/30 clip (`roop/assets/benchmark/regression_solo_300f_1080p.mp4`, generated on first use) |
| `--benchmark-source IMG` | 2nd faceset plate | the source face |
| `--benchmark-threads N` | `max_threads` | pass `20` on the bench rigs (AGENTS.md) |
| `--benchmark-update-baseline` | | record this run as the new baseline |

Exit codes: **0** pass (or baseline recorded), **1** regression, **2** the suite could
not run. Implementation: `app/roop/benchmark/regression.py`. Unit tests:
`app/tests/test_regression_benchmark.py`.

## What it runs

It runs a **real render** through `core.batch_process_with_options`: the worker pool, swap
batcher, stabilizers and ffmpeg writer that the Start button uses. It is configured from
`config.yaml` through `tests/config_sync` (`init_pipeline(sync_config=True)`). It does
**not** run `BenchmarkRunner`, which is `process_frame` on one thread in preview mode and
never reaches those stages.

Order of work:

1. Target capture: the most confident face in the first 150 frames, sampling every 5th
   frame. It deliberately does not take the first face found (see below).
2. Raw decode (the app's ffmpeg to a pipe) and raw encode (`FFMPEG_VideoWriter`, the
   configured codec), each run alone.
3. A 60-frame warm-up render, which pays the TensorRT engine build and the model load.
4. The timed render. It is observed through `procmgr_runtime.set_stage_sink`, which hooks
   every `_prof` stage. NVML is sampled every 50 ms.
5. The quality pass against the golden render.

## What it reports

- **Throughput**: raw decode/encode fps, end-to-end fps, p50/p99 per-frame latency and
  frame-time variance (`frame_total`, one sample per frame per worker), and the p99
  interval between successive encodes.
- **Per stage**: calls, busy seconds, ms/call and per-thread fps for decode, detect, swap,
  mask, enhance and encode. Detect counts per-frame `detect` plus the tracking pre-pass's
  `detection`, because with `temporal_detection` on, detection happens only in the
  pre-pass. Per-thread fps is summed across workers. It is not a wall-clock share and not
  a speedup budget (AGENTS.md).
- **VRAM peak and GPU utilisation** (NVML, device-wide).
- **Swap coverage, counted two ways**:
  - *decided*: frames the pipeline pasted a swap onto (the swap log);
  - *changed*: frames whose face region differs from the input by more than 4.0/255 mean,
    about 5.6x the documented render noise floor.

  Counting both matters: the first measures intent, the second measures outcome.
- **Quality against the golden render**: face-crop SSIM (mean and worst frame), face-crop
  PSNR, and whole-frame PSNR.

## What fails it

- Output frame count differs from the input.
- A stage never executed: decode, detect, swap or encode with zero calls.
- Swap coverage (decided or changed) falls more than 2 points below the baseline.
- `face_ssim_mean < 0.97`, `face_ssim_min < 0.90`, `face_psnr_mean < 35 dB`, or
  `frame_psnr_mean < 40 dB`.

**FPS never fails the run**; a drop over 15% is printed as a warning. A 300-frame window
is warm-up-contaminated: AGENTS.md sets 600 frames as the minimum for an acceptance
claim. Use `tests/ab_temporal_detection.py` (counterbalanced, 600 frames) to decide a
performance question.

A first run on a new key refuses to record a baseline if coverage is below 50% or a stage
never ran. A broken pipeline therefore cannot become the reference.

## Baselines

Baselines are stored per machine and per look: `<repo>/.roop/regression/<gpu>_<hash>/`,
holding `baseline.json` and `golden.mp4`. The hash covers the settings that decide what
the output looks like (`SIGNATURE_KEYS`: swap model, enhancer, masks, blend, detector,
stabilizers, codec and so on), plus the clip and source SHA-256 and the frame count.
Performance-only knobs (threads, pools, batch sizes) are deliberately left out. Changing
them must not change the picture, and catching it when it does is part of the job.
Changing a look setting starts a new baseline instead of failing. Each run writes
`runs/<timestamp>/report.json`.

## Measured on the RTX 4070 (2026-09-25)

Live configuration: hyperswap, Restore Ultra, XSeg plus occluder, TensorRT mixed, 20
threads, hevc_nvenc. Synthetic clip, 300 frames.

| | baseline | null control (same code) | positive control (a76091a's parent) |
|---|---:|---:|---:|
| end-to-end fps | 6.46 | 6.41 | **11.81** |
| p99 frame latency | 594 ms | 720 ms | |
| VRAM peak | 9498 MB | 9803 MB | |
| decided / changed | 300 / 300 | 300 / 300 | **0 / 0** |
| face SSIM mean / min | | 1.0 / 1.0 | 0.867 / 0.856 |
| face PSNR | | 100 dB | 25.8 dB |
| frame PSNR | | 100 dB | **43.8 dB** |
| verdict | recorded | PASS | REGRESSION (6 failures) |

Baseline stages: decode 15.6 ms, detect 31.5 ms (297 calls), swap 58.4 ms (408), mask
22.9 ms (816), enhance 104.5 ms (408), encode 3.7 ms per call.

Notes on the table:

- **This configuration is deterministic.** The null control's output was byte-identical
  to the golden file (same SHA-256; ffmpeg's own PSNR reports infinity). AGENTS.md's
  noise floor of 0.71/255 was measured on other configurations and still applies to
  them. The thresholds are set loose enough for a non-deterministic configuration, not
  tuned to this one.
- **The positive control is the real bug this suite found.** Commit 4bd577d reused the
  name `faces` in `ProcessMgr.swap_faces`, and every selected-mode render swapped nothing.
  That render was 83% "faster" and **passed the whole-frame PSNR floor** (43.8 dB > 40 dB),
  because the face is about 2% of a 1080p frame. It was caught by the face-crop metrics,
  both coverage counts, and the zero `swap` call count.
- **p99 latency and VRAM moved between two identical runs** (594 vs 720 ms; 9.5 vs
  9.8 GB). Neither is gated.

## Traps met while building it

- `api.map_mask_engines` returns ONE engine as a plain string. Iterating it produced
  `['m', 'a', 's', 'k', ...]` (`engine_list()` handles it).
- The first version captured the first box found in the first 30 frames. On `b1.mp4`,
  where nobody is on screen until about frame 30, that was a 0.80-score false box at the
  frame edge. Every real track was then refused as somebody else, and the run swapped
  nothing.
- With `temporal_detection` on, `_prof('detect')` never fires, so a check on `detect`
  alone reports a stage that ran as missing.
