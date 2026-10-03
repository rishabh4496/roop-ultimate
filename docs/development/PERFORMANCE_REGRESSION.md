# Performance and fidelity regression suite

`tests/test_performance_regression.py` — real production renders (the Start button's path,
`core.batch_process_with_options` through `roop.benchmark.regression`), never
`BenchmarkRunner`.

```
app\env\Scripts\python.exe -m pytest tests/test_performance_regression.py   # ~25 min on the 4070
```

A bulk `pytest` run **deselects** the four render tests (`perf` marker; `conftest.py`) and
keeps the seven cheap instrument tests. Naming the file or passing `-m perf` runs them.
Needs `lpips` (installed with `pip install --no-deps lpips`; its AlexNet backbone downloads
to `TORCH_HOME` on first use, ~233 MB) and the media folder (`ROOP_KEEP_DIR`, default
`<PINOKIO_HOME>/roop-keep`).

## What it checks

Three inputs — one 1080p still, a 90-frame 1080p clip (`single/s3.mp4`), a 48-frame 4K clip
(`single/s4.mp4`) — each rendered **twice** with one config (arm A, arm B):

| check | gate |
|---|---|
| fidelity, whole frame **and** face crop, B vs A and B vs the stored golden | PSNR > 40 dB, SSIM >= 0.995, LPIPS (AlexNet) <= 0.005 — on the mean; worst frame is reported |
| the render swapped | output face closer (cosine) to the SOURCE person than to the original target on >= 50% of face frames |
| compared frame count | equals the input's (a `zip` over two videos truncates silently) |
| A/V timing | `scripts/verify_roop_keep.validate_timestamp_integrity`: constant frame rate, source audio stream-copied, A/V drift bounded; plus output fps == source fps, frame count == input, video duration == frames/fps |
| soak | 4 renders (`ROOP_SOAK_RENDERS`, or `ROOP_SOAK_SECONDS=10800` for hours): quiescent CUDA allocation <= +2 MB/render, RSS <= +40 MB/render |
| throughput | logged, compared with the stored golden's fps and **warned** past -15%; never fails (AGENTS.md) |

Goldens live in `<repo>/.roop/perf_regression/<gpu>_<config hash>/<case>/` and are recorded
by the test only after the case passes. `ROOP_PERF_UPDATE_BASELINE=1` re-records. The last
report is `.roop/perf_regression/last_report.json`.

The 4K fixtures carry no audio, so every case muxes a real AAC track (a synthetic 48 kHz sine
where the source has none): an untested mux proves nothing.

Self-tests prove each gate can fail (blur trips SSIM + PSNR + LPIPS, a shifted frame trips,
an untouched candidate reads as unswapped, a silent or wrong-count output fails the A/V check).

## Measured 2026-10-03 (RTX 4070, hyperswap + Restore Ultra, hevc_nvenc, `--threads 20`)

| case | A vs B | B vs golden | swapped | A/V |
|---|---|---|---|---|
| image 1080p | bit-identical | bit-identical | 1/1, margin +0.447 | n/a |
| video 1080p, 90 f | 100 dB / SSIM 1.0 / LPIPS 0 | same | 97.8% | pass |
| video 4K, 48 f | 100 dB / SSIM 1.0 / LPIPS 0 | (first run: recorded) | 100%, margin +0.580 | pass |

Soak, 4 x 1080p: quiescent CUDA 9.4 MB flat, RSS 5125 -> 5129 MB.
Throughput (whole job, **includes ~35 s model init per render**, 90/48 frames, so not an
acceptance number): 1080p 1.5-1.7 fps, 4K 0.20 fps (48 f in ~240 s), still image 0.03 fps.

The multi-hour contract was **not** run; the soak is the same harness at 4 renders.

## What the suite found

1. **`restore_audio` dropped the last video frame of every untrimmed render whose source audio
   ended before its video** (90 -> 89 at 29.97 fps). `-shortest` again; the trimmed branch had
   been fixed for it (b1.mp4, 120 -> 117) and the single-command branch had not. Fixed:
   bounded by the render's own video length. `app/tests/test_restore_audio_frame_count.py`.
2. **A render is not a pure function of its config.** The stabilizer's block size comes from
   FREE RAM at render start: 7.0 GB free -> 12-frame blocks, 7.8 GB free -> 16-frame blocks,
   and the two outputs of one config differ by face SSIM 0.971 / LPIPS 0.030 (frame PSNR 47.3,
   worst face frame 36.4 dB, SSIM 0.945) — outside the contract's own gates. The rig pins
   `ROOP_STAB_CHUNK_MB=1024` (`ROOP_PERF_STAB_CHUNK_MB`); pinned, A, B and a golden recorded in
   another process are bit-identical. The code documents the block-boundary seed residual as
   <= 1%; this is the measured size of it on a real face. Not changed.
3. **The bench's "changed" pixel threshold (4.0/255 mean in the face crop) reads a real swap
   as no swap** when the face is large and the source resembles the target (3.24 here), and
   the swap log cannot see stills (`frame_idx is None`). The "did it swap" gate is identity.
4. `verify_roop_keep.timestamp_integrity` looked `ffprobe` up on PATH only. Pinokio puts none
   there, so on this machine every check returned "unreadable". It now falls back to the app's
   own resolver.

## Multi-stream CUDA scheduling and multi-GPU: what exists, what was not built

**Not built into `app/roop/core.py`**, deliberately.

* The three-stream pipeline the Stage 5 brief describes **already exists**, in
  `face_engine/core/cuda_streams.py` (`CUDAStreamPipeline`): decode / inference / encode on
  three `torch.cuda.Stream`s over a `CUDARingBuffer`, hand-offs by `torch.cuda.Event`, never a
  device-wide sync, with `face_engine/tests/test_cuda_streams.py`. It cannot run in this
  environment: it needs PyAV (`import av`), which is not installed (and PyPI wheels have no
  NVDEC).
* The app's production render is a different engine: decode and encode are **ffmpeg
  subprocess pipes** (NVDEC / NVENC), not torch work, so "stream 1 = decode + H2D" and
  "stream 3 = encode" have nothing to attach to. The ORT/TensorRT sessions already run on a
  dedicated stream ordered by `wait_stream` (`inference_engine.py`). The device-resident
  alignment/warp/composite layer (`CudaAffineBatch`, Stage 4) has no production caller;
  wiring it in is a rewrite of the frame loop, and the render is GPU-bound, where moving
  work between streams moves the clock only if it removes GPU work (AGENTS.md).
* A per-thread stream triple would also multiply cuBLAS workspaces (8.1 MiB per
  handle x stream; see the cuBLAS-per-stream leak note), on a 6 GB / 16 GB laptop.

No stream gain for the app was measured, so none is claimed.

**Distributed rendering** (`app/roop/distributed_render.py`) needs nothing from streams: each
worker is a separate OS process (`run.py --render`), pinned to exactly one GPU by
`CUDA_VISIBLE_DEVICES=<uuid>` and addressing it as `--cuda_device_id 0`, so streams, the
caching allocator and cuBLAS workspaces are process-private and cannot race. Verified with
stubbed workers in `app/tests/test_distributed_render.py` (every chunk once, one GPU per
worker, never two chunks on one GPU, workers never outnumber chunks; the overlap detector is
shown to fire on a duplicated GPU). Only one GPU is present, so no real multi-GPU run was made.
