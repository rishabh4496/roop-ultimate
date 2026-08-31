# FaceSet V2 format

FaceSet V2 is an additive extension of the existing `.fsz` archive. A legacy
archive is a ZIP containing root-level PNG reference members (`0.png`,
`1.png`, ...). V2 keeps those members byte-for-byte as the source images and
adds a deterministic `metadata.json` member. Existing FaceSet fields and the
legacy loader remain supported.

## Archive layout

```text
person.fsz
├── metadata.json       # schema, version, identity index and cached analysis
├── 0.png               # reference image, retained for V1 compatibility
├── 1.png
└── ...
```

`metadata.json` has `schema: "roop.fsz"` and `version: 2`. JSON keys are sorted,
compact, UTF-8/ASCII-safe, and ZIP timestamps are fixed so the same FaceSet
inputs produce the same archive bytes. Each reference member has a SHA-256
entry under `integrity.sha256`; invalid ZIP CRCs, missing members, unsafe names,
invalid geometry, unsupported versions, invalid JSON, and checksum mismatches
are rejected before loading.

## Metadata sections

Each object in `sources` is a pose-specific reference and retains its own
embedding. It contains:

- `identity`: raw ArcFace embedding, normalized embedding, and quality
  confidence;
- `geometry`: bbox, 68-point landmarks, 3D 68-point landmarks when available,
  2D landmarks, yaw/pitch/roll, face scale, and facial proportions;
- `quality`: face pixels, sharpness, blur, detector confidence, landmark
  confidence, exposure, saturation, occlusion/visibility estimate, pose
  suitability, and composite score;
- `appearance`: luminance percentiles, skin-region BGR statistics, local
  contrast, color-temperature estimate, shadow fraction, and highlight fraction;
- `expression`: eye-open, mouth-open, smile-width, and compact expression
  descriptors when 68 landmarks are available;
- `identity_details`: a compact high-frequency descriptor, detail mask, and
  candidate local detail masks. Candidates are intentionally unlabeled
  high-frequency details; they are not asserted to be a mole, freckle, scar, or
  wrinkle without a later consumer deciding that from cross-reference support;
- `pose_bin`: automatically classified as frontal, mild/medium/strong left or
  right, or profile left/right.

The top-level `identity` is a quality-weighted normalized mean of the selected
pose-specific embeddings. It is a compact global identity representation only;
it never replaces the per-source vectors. `pose_bank` maps each angle class to
source indices, while `index.normalized_embeddings` provides a compact lookup
array for cosine matching.

## Selection and runtime behavior

Creation scores face size, sharpness, detector and landmark confidence,
exposure, saturation, occlusion/visibility, and pose availability. Entries
below `ROOP_FACESET_V2_MIN_QUALITY` (default `0.35`) or below 32 face pixels are
excluded. The selector first covers represented pose bins, then fills remaining
slots by quality while suppressing near-duplicate embeddings within a bin.
Defaults cap the bank at 32 entries and 6 entries per bin; these can be changed
with `ROOP_FACESET_V2_MAX_ENTRIES` and `ROOP_FACESET_V2_MAX_PER_BIN`.

`FaceSet.select_reference_index()` uses the cached pose, quality, optional
lighting vector, and optional embedding. `ProcessMgr` invokes this only when the
existing `use_source_bank` option is enabled. Lighting comparison is classical
image-statistics work on the target crop; it adds no detector or neural
inference. Existing V1 source-bank selection remains the fallback.

## Migration and loading

Loading a V1 archive remains supported. Its PNGs are extracted and analyzed by
the existing detector path, and the old `AverageEmbeddings()` behavior is
preserved. Saving that loaded FaceSet through the library save action produces
a V2 archive and recovers the original first pose embedding from
`embeddings_backup` where available. This is the normal loss-minimizing
migration path.

For tooling, `roop.faceset_v2.migrate_legacy_fsz()` can migrate a PNG-only
archive. Passing an already loaded `FaceSet` and its `ref_images` creates full
cached identity/geometry metadata. Migrating raw PNGs without a loaded FaceSet
is intentionally lossless but metadata-only; the next normal load can enrich
the in-memory representation after detection.

No existing `.fsz` is overwritten automatically, no user configuration is
rewritten, and V2 remains readable by the current application because its root
PNG members are unchanged. A corrupt V2 archive is rejected rather than
silently downgraded to V1.
