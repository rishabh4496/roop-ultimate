| Stage | CPU | CUDA | TENSORRT |
|---|---|---|---|
| detector (per frame, ms) | 23.13 (p95 23.81) | 5.54 (p95 30.46) | 2.88 (p95 3.35) |
| stabilizer (per face, ms) | 0.06 (p95 0.06) | 0.06 (p95 0.06) | 0.06 (p95 0.06) |
| face_parser (per face, ms) | 23.79 (p95 24.34) | 7.08 (p95 7.53) | 6.54 (p95 7.52) |
| swapper (per face, ms) | 258.50 (p95 262.58) | 19.62 (p95 20.38) | 16.49 (p95 17.35) |
| occlusion_masker (per face, ms) | 3.66 (p95 7.61) | 3.66 (p95 7.61) | 3.66 (p95 7.61) |
| laplacian_blender (per face, ms) | 3.06 (p95 3.15) | 3.06 (p95 3.15) | 3.06 (p95 3.15) |
| sum of stage medians (ms) | 312.20 | 39.01 | 32.68 |
| peak VRAM added (MiB) | 20 MiB | 1466 MiB | 1819 MiB |

GPU: NVIDIA GeForce RTX 4070. Median and p95 over the timed iterations after warm-up. 'fell back' means the session did not run on the requested provider.
