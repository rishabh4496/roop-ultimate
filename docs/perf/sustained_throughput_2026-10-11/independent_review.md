# Independent Validation and Adversarial Review

**Date:** 2026-10-11  
**Part 1 investigator:** Agent E, Gemini 3.8 Flash Tiered via Antigravity, high effort.  
**Part 2 final reviewer:** Attempted with Gemini 3.8 Flash Tiered via Antigravity, high effort, but the worker failed before producing a report.

## Part 1: independent static review

The independent reviewer inspected production source without reading other swarm reports. It identified potential progressive mechanisms in VRAM governor step-down, per-lease admission pressure, permanent swap fallbacks, temporal track retention, disabled GC, segment-finalization work, host-memory-derived geometry, whole-clip temporal pre-pass retention, and high-frequency VRAM queries.

The review also required these acceptance gates:

- at least 600 frames, with null control and counterbalanced A/B arms
- pinned stabilization geometry
- faces/s and swap rate beside FPS
- constant-workload slope testing rather than a constant speedup claim
- proof that a faster arm did not skip work or silently fall back to original frames

Those gates are adopted. The current 600-frame results satisfy the first three for separate baseline/final arms, but not the same-process job sequence or segment-boundary gate.

## Coordinator response to high-risk findings

- H5 digest reuse was implemented with a fallback and a targeted test.
- H2 cleanup was implemented, but same-process VRAM recovery remains unverified.
- H11 query caching was implemented and tested for TTL behavior.
- H10 stale-track pruning was implemented and tested for recent-state preservation.
- H4 permanent fallback latching was not changed because existing tests require its current behavior and transient versus permanent failures were not classified.
- H9 GC behavior was not changed because no controlled pause/memory experiment was run.
- The recorded telemetry process-RSS fields were rejected because the sidecar targeted only the runner process. The sidecar now aggregates the target process tree.

## Review outcome

No final independent approval was produced because the final Gemini worker failed. Therefore this repository state must not be described as adversarially validated. The final engineering decision is **Partially validated**, pending the same-process, segment-boundary, corrected-memory telemetry, and final independent review experiments.
