# Face-neighbourhood matte in `paste_upscale` (2026-10-22)

Code: `procmgr_masking.MaskingMixin._matte_roi_eligible` / `_matte_roi`, `create_landmark_mask(origin=)`. Off switch
`ROOP_BLEND_ROI_MATTE=0`. Tests `app/tests/test_matte_roi.py` (42). Probes `app/tests/probe_matte_roi_real_faces.py`,
`ab_env_e2e.py`.

## What was wrong
`paste_upscale` built its matte on a full-frame canvas (zeroed frame, full-frame warp of the ellipse, a second zeroed frame for
the landmark hull, `np.minimum`, `np.where`, a full-frame float32 divide) to produce a matte non-zero over ~230 px. Real
`paste_upscale`, one 230 px face, cv2 on one thread, per face:

| frame | before | canvas | saved |
|---|---|---|---|
| 1280x720 | 15.4 ms | 10.4 ms | 4.9 ms |
| 1920x1080 | 25.6 ms | 10.5 ms | 15.1 ms |
| 3840x2160 | 74.6 ms | 10.5 ms | 64.1 ms |

The canvas path costs the same at every resolution. It is taken only when nothing needs the matte in frame space (no temporal
compositor, face-overlap region, model mask, other faces to protect, overlay or distance edge mode); the live config's
`temporal_compositing: false` and `swap_model_mask_strength: 0` match that for single-face frames.

## Parity
NOT bit-identical (cv2's fixed-point source-coordinate rounding shifts when the destination origin is translated), and nothing
non-zero is ever left outside the canvas.
* 36 randomised synthetic faces: 10 differ, by 1 level, at 4-29 matte-edge pixels.
* **135 real detected faces** (Love, Weeds, Monica Bellucci, d1, d4; `real_faces_parity.txt`): canvas path taken 129/135, max
  difference **1 level**, median 0 differing pixels per face, p95 4, max 6.

## What is NOT measured
**The effect on the render clock.** Both end-to-end attempts failed: the 8-render 4K schedule was killed by the system for low
memory (nothing recorded), and the 4-render 1080p schedule (`e2e_d1_1080p.log`) ran at 0.18-1.0 fps against ~14 normally
under memory pressure, null control 107% apart: void. Its output also taught something: the encoded files differed by a mean of
1.0/255 over 87% of the picture (up to 12 levels) while the pastes differ by almost nothing; a VBR encoder's quantiser follows
cumulative bits, so an end-to-end diff of ENCODED files cannot test a +-1-level change. Compare the paste, as the probe does.
The repo's frame: a CPU saving moves a GPU-bound render only if CPU was the limit; at 4K it may be, at 720p it is not.
