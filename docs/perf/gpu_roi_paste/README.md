# `CudaAffineBatch.paste_faces_roi` (2026-10-22) — built and priced, NOT wired into the render

There is no `GpuFaceAligner`; `CudaAffineBatch` (`optimized_processor.py`) is the existing GPU aligner: bicubic `grid_sample`
warp, inverse-warp paste, feathered box matte, occlusion multiply. Its `paste_faces` builds the sampling grid and runs
`grid_sample` over the WHOLE frame per face and clones the batch. `paste_faces_roi` samples only each face's destination box
(crop corners through the inverse matrix, +2 px) by folding the box origin into the matrix translation, so the sampling positions
are the same ones. Tests `app/tests/test_cuda_affine_roi.py` (CPU and CUDA; equal to `paste_faces` within 1e-4 of full scale =
0.026 of an 8-bit level; untouched outside the box; the grid is under 1% of the frame at 1080p). Lanczos4 is not available on this
path (`grid_sample` has bilinear / bicubic / nearest only).

Median ms for all of a frame's pastes (`probe_gpu_roi_paste.py`, `report.json`, RTX 4070). "host" = upload + paste + download BGR;
"vram" = frame already on the device. Cells are 1 face / 4 faces:

| frame | CPU canvas paste | GPU full-frame, host | GPU full-frame, vram | GPU ROI, host | GPU ROI, vram |
|---|---|---|---|---|---|
| 1280x720 | 7.8 / 29.4 | 1.45 / 3.72 | 0.81 / 2.98 | 1.68 / 2.97 | 1.08 / 2.39 |
| 1920x1080 | 8.6 / 30.0 | 3.43 / 9.53 | 2.10 / 8.26 | 2.26 / 3.79 | 0.99 / 2.43 |
| 3840x2160 | 11.5 / 33.0 | 15.8 / 40.8 | 9.77 / 34.2 | 7.40 / 8.53 | 1.11 / 2.44 |

ROI mode is 5x (host) to 14x (resident) cheaper than the full-frame GPU paste at 4K with 4 faces, and its cost no longer grows
with the frame. Why it is not wired: the GPU arm does LESS than production (no landmark hull, adaptive feather, temporal matte EMA,
overlap trim, rim illumination), so it is a ceiling; the mouth / eye / expression / lip-sync restores after the paste are host-frame
code; and moving CPU work onto the GPU of a GPU-bound pipeline adds GPU work. After the canvas matte the CPU paste is flat at
~8-11 ms per face, about 1.5 s of thread time in a 42 s render. Owner call, with a pixel-parity-then-counterbalanced-A/B plan.
