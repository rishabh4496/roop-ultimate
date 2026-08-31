# Phase Handoff - Phase 4 FaceSet V2 Multi-Angle Identity Bank

Date: 2026-08-31

## Current state

Phase 4 is implemented and regression validated. The changes are uncommitted;
base commit: `f8d2e2f`. FaceSet V2 adds deterministic `metadata.json` to the
existing root-PNG `.fsz` archive, while preserving the current FaceSet fields,
legacy loading, and source-bank behavior. Phase 3 temporal tracking remains in
the working tree and is not to be reverted.

Do not revert the unrelated `.geminiignore` working-tree edit or the earlier
Phase 2 occlusion-response files. No launcher file was changed.

## Implementation

- `app/roop/faceset_v2.py` defines the versioned deterministic archive metadata,
  quality scoring/filtering, pose bins, global/per-pose identity vectors,
  appearance/detail caches, validation, migration, and lookup.
- `app/roop/FaceSet.py` keeps the old fields and adds optional V2 metadata,
  selection, and lighting helpers. V2 `AverageEmbeddings()` is a no-op so
  pose-specific vectors are not destroyed.
- `app/source_gallery.py` validates V2 archives and matches metadata entries to
  individual detections, including repeated full-frame references containing
  multiple faces.
- `app/routes_faceset.py` writes V2 on library save and rejects corrupt imports.
- `app/roop/ProcessMgr.py` uses the V2 cached selector only under the existing
  `use_source_bank` opt-in; legacy pose selection remains intact.
- `docs/FACESET_V2.md` is the schema and migration reference.

## Configuration

V2 creation controls do not change existing configuration files or old `.fsz`
files:

```text
ROOP_FACESET_V2_MIN_QUALITY=0.35
ROOP_FACESET_V2_MAX_ENTRIES=32
ROOP_FACESET_V2_MAX_PER_BIN=6
```

Phase 3 temporal controls remain documented in the Phase 3 section of the
progress log.

Keep TensorRT/CUDA, FP16/FP32/mixed precision, detector alternatives/pooling,
FaceSet/source-bank, 3D reconstruction, RealityUX, RealSwap, GPEN 256 Pro,
GPEN Realistic, UltraMax, all enhancer paths, CPU/DirectML/Apple/AMD fallback
contracts, and the RTX 3060 custom look values unchanged.

## Evidence

- V2 suite: **9 passed**.
- V2 plus existing compatibility set: **117 passed**.
- Full suite: **1523 passed, 1 skipped, 0 failures** in **43.039 s**.
- 24-reference deterministic benchmark: **4,449,161 bytes**, prepare **48.306
  ms**, write **248.907 ms**, read **24.918 ms**, **43,431.75 lookups/s**.
- Python compilation, static undefined-name checks, tracked JavaScript syntax
  checks, and `git diff --check` passed.
- Regressions discovered: one missing `numpy` import in the new loader path was
  caught by the full suite and fixed; no remaining regressions.
- No physical GPU FaceSet build or visual identity-quality measurement was run.

## Next session instructions

1. Start with `git status`, recent commits, `logs` (latest first), this file,
   `docs/OPTIMIZATION_PROGRESS.md`, and `docs/PERFORMANCE_OPTIMIZATION_HANDOFF.md`.
2. Keep `G:\pinokio\prototype\system\examples\mochi\start.js` as the
   launcher reference. No launcher changes are needed for Phase 3; if one is
   proposed, reread `PINOKIO.md` and reapply the captured `input.event[1]`
   `local.set` URL pattern.
3. Build a real-photo multi-angle bank with frontal, mild, medium, strong, and
   profile examples where available, without requiring exact user angles.
4. Measure source-selection accuracy and identity/detail/expression/lighting
   quality with V2 source-bank and 3D paths separately on RTX 4070 and RTX 3060.
   Record archive bytes, creation/load time, FPS, VRAM, and RSS.
5. Verify that low-quality and near-duplicate references are filtered without
   losing complementary pose coverage. Do not promote new defaults from the
   synthetic benchmark alone.
6. Before finalizing the next phase, rerun targeted/full tests, available real
   benchmarks, Python/launcher syntax checks, `git diff --check`, and rewrite
   both state documents with exact evidence and next action.

## Do not break

Preserve output order, audio/segment behavior, `.fsz` compatibility,
wrong-FaceSet/finite/output-integrity guards, the existing detector pool and
ROI helper, and all provider/hardware fallback paths. Do not enable unrelated
expensive models by default.
