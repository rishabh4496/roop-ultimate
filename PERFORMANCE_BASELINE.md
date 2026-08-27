# Performance Baseline

Updated: 2026-08-28. These measurements are repository benchmark evidence, not a substitute for a real-video end-to-end run.

## Test configuration

- Device: NVIDIA GeForce RTX 4070, 11.99 GB reported VRAM.
- Provider: TensorRT mixed precision.
- Workload catalogue: RetinaFace R50 detector, recognition, landmarks, Realswap, UltraMax, XSeg, and BiSeNet.
- Benchmark mode: full, `--no-apply`; current user settings were not changed.
- Current relevant settings: `perf_trt_pool=2`, `perf_detmask_pool=2`, `perf_detector_pool=2`, `perf_expr_pool=auto`, `max_threads=12`, `output_video_codec=hevc_nvenc`, `perf_nvdec=on`.

## Before Phase 3

The pre-change benchmark used context levels 1/2/4/8 and stopped during a cold XSeg build. Partial calls/sec results were:

| Stage | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| Detector | 318.2 | 407.6 | 450.1 | 438.9 |
| Recognition | 843.2 | 923.4 | 757.5 | not measured |
| Landmarks | 1452.5 | 1491.9 | not measured | not measured |
| Realswap | 196.9 | 255.9 | 250.2 | not measured |
| UltraMax | 36.5 | 43.2 | 40.0 | not measured |

The process was safely stopped after no output during the cold XSeg build. No complete before end-to-end FPS, latency, CPU, VRAM, RAM, or queue-depth record exists; do not compare this partial run as a complete baseline.

## After Phase 3 working-tree changes

The complete sweep used 1/2/3/4 contexts:

| Stage | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| Detector | 336.6 | 405.4 | 423.2 | 414.1 |
| Recognition | 807.9 | 942.5 | 836.4 | 720.6 |
| Landmarks | 1347.2 | 1560.6 | 1420.9 | 1250.9 |
| Realswap | 188.1 | 244.7 | 252.9 | 250.5 |
| UltraMax | 36.3 | 41.3 | 38.6 | 39.8 |
| XSeg | 162.4 | 422.1 | 432.3 | 402.0 |
| BiSeNet | 76.8 | 92.4 | 90.7 | 94.1 |

Knee recommendations: detector 3; recognition 2; landmarks 2; realswap 2 by the 4% knee rule (raw maximum 3); UltraMax 2; XSeg 2; BiSeNet 2. Heavy composite throughput was `21.67 FPS`; widening the TRT pool from 2 to 4 reduced it to `20.15 FPS`.

Thread curves were standard `1:9.70, 2:17.22, 4:33.72, 6:41.12, 8:49.97, 12:59.30, 16:66.73, 32:65.41`, enhanced `1:7.92, 2:14.64, 4:24.62, 6:27.50, 8:27.49, 12:29.12, 16:28.22, 32:30.19`, and heavy `1:7.17, 2:11.95, 4:19.21, 6:21.67, 8:21.55, 12:20.23`.

Observed stage-sweep free VRAM fell from about `10.85 GB` initially to about `8.2 GB` at the UltraMax four-context point. A separate probe during a cold mask build reported `3.13 GB` used and `8.88 GB` free; it is not a final peak measurement. Peak RAM and queue depth were not emitted by this synthetic benchmark and must be captured in the real-video validation.

## Interpretation

The evidence supports bounded context-knee selection and rejects blindly increasing context count. It does not establish a complete end-to-end improvement because the pre-change run was incomplete and the synthetic benchmark does not measure full decode/detect/swap/enhance/stabilize/composite/encode latency as one real video pipeline.
