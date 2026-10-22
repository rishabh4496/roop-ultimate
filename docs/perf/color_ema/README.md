# Temporal colour-transfer EMA — measured, NOT wired (2026-10-22)

Probe: `app/tests/probe_color_ema.py` (same rig and clips as `probe_skin_tone_transfer.py`: a real faceset swapped by the
production pipeline with the colour transfer OFF; every arm is then applied offline to the SAME swapped crop).
Code under test: `app/roop/temporal_color.py` (`lct_affine` is bit-identical to `_color_transfer_lct`, see
`app/tests/test_temporal_color.py`). Live config uses `color_transfer_mode: lct`; the code default is `rct`.

Raw output: `probe_frames260.full.txt` (402 usable frames, 4 clips), `probe_step.full.txt` (synthetic 1.12x brightness step).

| arm | pumping (dE/frame) | \|error\| vs plate (dE) |
|---|---|---|
| none | 0.163 | 1.247 |
| lct (shipped) | **0.141** | 0.920 |
| lct + EMA of statistics (alpha 0.15) | 0.154 | 0.930 |
| lct + EMA of the transform (alpha 0.15) | 0.157 | 0.923 |
| rct (code default) | **0.138** | 0.992 |
| rct + EMA of statistics | 0.165 | 1.002 |
| rct + EMA of the transform | 0.165 | 0.996 |

Pumping = mean |frame-to-frame change of the skin-tone error|; lower is steadier. The EMA raises it 9-20% and does not
improve accuracy. The shipped per-frame transform already pumps less than no transfer at all (0.141 vs 0.163).

Why: the per-frame transform tracks the plate, which cancels the swap's own fluctuation. A recurrence freezes that
compensation. The earlier mask-isolated version (2026-10-19) reached the same conclusion.

Not measured: the lag cost of the EMA on a real lighting change. The step test did not separate any arm (error stayed
below the 2.3 dE JND even with the transfer off), so it says nothing about lag. Only 4 clips, one source bank.
