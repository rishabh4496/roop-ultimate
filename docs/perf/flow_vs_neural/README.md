# Flow-placed landmarks vs the landmark net on the flow box (2026-10-22)

`app/tests/probe_flow_vs_neural_landmarks.py`, K=4 simulated on real frames. Truth = production detector + `landmark_2d_106`
on every frame. Flow = `FlowCoaster` (a SIMILARITY fit of the tracked cloud applied to the KEY FRAME's landmarks: it carries the
key frame's expression and cannot follow a blink or a jaw opening). Neural = `landmark_2d_106` run on the box the flow produced.
Error in % of eye distance. `neural at the true box` = 0.00 on both clips: the harness reproduces the truth.

| Weeds.mp4 (219 scored) | median | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| flow | 1.21 | 4.39 | 5.58 | 6.91 | 7.46 |
| neural on the flow box | **0.58** | **1.99** | **2.87** | **4.64** | 5.54 |

Neural is better on 92% of frames (90% on Love). **Love is not a clean result**: flow p95 52%, max 70% (neural 35% / 82%) because
that clip has hard cuts and this probe has none of the pre-pass's shot-boundary handling; both arms fail on a cut. Medians there
(1.32 -> 0.78) agree with Weeds.

The pre-pass hook (built after these probes; see the last section) needed the colour frame at `_flow_miss` and the detector's GPU
guard, and applies to every flow-bridged face (K=1 gap-fills too). It does NOT address why K>1 loses swaps (identity binding,
Phase 14).
NVOFA itself is unavailable here (the installed OpenCV wheel has no CUDA), so the `nvof` backend remains unverified; the CPU
Lucas-Kanade backend produced these numbers.
Landmarks stay FP32: `PRECISION_POLICY.md` records mixed precision drifting `2d106det` / `1k3d68` by 1.2-8.8 px.

## BUILT (opt-in) and checked through the REAL pre-pass
`app/roop/landmark_refresh.py`, hooked in `procmgr_tracking._flow_miss` (a ring of the last few decoded frames is kept beside the
grey ring) and used by `_flow_bridge_geometry`: a re-measured step keeps what the net measured instead of receiving the linear
drift correction toward the closing detection, while `bridge_gap`'s tolerance check (flow must arrive at the closing detection)
still decides whether the gap is bridged at all. `ROOP_PREPASS_LANDMARK_REFRESH=1` to enable; OFF by default. A net answer more
than 0.25 x eye distance from the flow geometry is refused. Tests `app/tests/test_landmark_refresh.py` (13; mutation-checked).

`app/tests/probe_landmark_refresh_prepass.py` (Weeds.mp4, 300 frames, three real pre-passes, `prepass_k4_weeds.json`), landmarks
of the 219 flow-placed faces against the K=1 run's detection of the same frame, % of eye distance:

| | n | median | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| K=4, refresh off | 217 | 1.24 | 4.11 | 6.22 | 19.98 | 47.56 | 2.34 |
| K=4, refresh on | 217 | **0.93** | **2.68** | **4.07** | 21.96 | 45.56 | **1.87** |

The pre-pass printed `244/271 flow-placed face(s) had their landmarks re-measured (27 answers refused ..., 0 errors)`. The
gain is smaller than the isolated probe's because the bridge already corrects flow toward the closing detection; the extreme tail
(p99, max: faces the refresh does not touch) is unchanged. NOT measured: any effect on the swap (the swap-rate-guarded
end-to-end A/B needs full renders, which were killed for low memory), other clips, and the 3060. It does not address the K>1
swap-rate loss (identity binding).
