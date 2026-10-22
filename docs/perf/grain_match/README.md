# Sensor-noise match after the paste (2026-10-22)

Code `app/roop/grain_match.py`, wired in `paste_upscale` (after the rim match, before the composite); counters printed in the swap
audit as `grain match: ...`. `ROOP_GRAIN_MATCH=0` disables it; default 1 (self-gating). Tests `test_grain_match.py` (21; the wiring
tests have the feature off as the control, and a removed wiring fails two of them).

Relation to what already existed: `merger_grain_match` (live 0.2) measures the plate floor on the aligned crop BEFORE the paste and
adds that sigma. This stage measures AFTER the warp, in frame space, adds only the missing variance sqrt(max(s_plate^2 - s_paste^2, 0))
(variances add), seeds from (frame, track, position) so a render repeats, and says what it did.

Estimator: Immerkaer's 3x3 Laplacian + median absolute deviation (reads white noise of sigma s as s, within 1%, edges included).

Measured (`probe_swap_noise_floor.py`): finest-scale luma noise of the face region of 16 clips is 0.25-0.74 levels (the 8-bit floor
is 0.29), except one 640x360 stock clip at 2.1. Weeds, live config, 150 frames: plate face 0.49, swapped face **2.22 (4.5x the
plate)**, ring control 0.25 -> 0.25. d1 1080p, 127 faces: plate 0.48, swap 1.31, `0/127 faces got grain`. On this footage the stage
does nothing and the output is unchanged; the swap is the noisier one. Unverified on grainy footage (none here; the mechanism is
tested on synthetic noise only). The swapped face also has a temporal excess of 1.91 (mean |d out - d plate| in the face), the
measured side of the "sticker" look; it is the opposite of a grain deficit.
