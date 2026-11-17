# Offline Studio integration

Studio's Detection Inspector exposes the requested detection, masking,
restoration and compositing controls. The parameter menu in the NLE timeline
animates those same controls. Enums, booleans and Pixel Boost use held values;
numeric curves support linear, smoothstep and hold interpolation independently.
Frames in the API and saved timeline are absolute, zero-based source frames.
The ordinary frame counter remains one-based.

Use **Pre-Scan**, assign a faceset to each track, set active ranges and In/Out,
then render. Pre-Scan now calls the ordered production tracking pass and refuses
success without a trajectory. Project state is checkpointed per media item in
localStorage (`roop_studio_projects_v1`, with a previous-checkpoint backup).
Bindings, strokes, masks, roto, pins, keys, ranges and trim points recover on
reload. Ctrl+Z / Ctrl+Y use the application's shared history. Checkpoints are
local to this browser profile; storage failures are reported in the UI.

## API

`POST /api/target/prescan` accepts the selected `target_media_id` and the same
Studio settings/timeline as preview/render. It returns the trajectory summary
and source color description. `GET /api/target/color` returns the four ffprobe
color tags, including the `color_trc` alias for `color_transfer`.
`GET /api/models/offline` reports local presence and verified SHA256 per asset.

`POST /api/preview/frame` and `POST /api/swap` accept these wire controls:

| Group | Parameters |
|---|---|
| Detection | `detector_model`, `multi_angle_detect`, `stabilization_strength`, `pose_adaptive_faceset` |
| Masking | `mask_engine`, `yaw_erosion_bias`, `expression_restorer_weight`, `preserve_inner_mouth` |
| Restoration | `pixel_boost` (256/512/768), `ultramax_ramp` (256–384), `temporal_restore_alpha`, `restore_identity_weight` |
| Compositing | `color_transfer_mode` (including lab/cdf/adain), `grain_match_strength`, `motion_blur_match` |

Put values on `timeline.keyframes[]` with `frame`, and optionally an
`interpolations` map for per-parameter curves. Bindings use `track_id`,
`faceset_path`, `enabled`, `locked` and inclusive `active_ranges`.
Existing saved setting names remain canonical; wire aliases are translated
before preview/render configuration. Settings owns defaults and persistence.

```javascript
const result = await fetch(`${baseUrl}/api/target/prescan`, {
  method: 'POST', headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({target_media_id, detector_model: 'gnkps'})
}).then(r => r.json())
```

```python
import requests
result = requests.post(f'{base_url}/api/target/prescan', timeout=3600,
                       json={'target_media_id': media_id, 'detector_model': 'gnkps'}).json()
```

```bash
curl -X POST "$BASE_URL/api/target/prescan" -H "Content-Type: application/json" \
  -d '{"target_media_id":"MEDIA_ID","detector_model":"gnkps"}'
```

## Runtime contracts and evidence

* `faceset_v2._entry` records pose, detector score and native-crop Laplacian.
  The existing V2 selection filters blurred/duplicate samples. Both creation
  paths use that writer. `source_gallery._ingest_faceset` lazily caches legacy
  analysis via `faceset_builder.cache_missing_metadata`; it does not rewrite
  old archives, filter legacy banks or change their centroid semantics.
* `model_integrity.require_local_model` verifies pinned size/SHA and exact-byte
  aliases without fetching. Optional `offline_only` assets do not auto-download
  at boot. Missing/corrupt requested GNKPS assets fail explicitly. Parser
  aliases retain BiSeNet's existing FP32 lock. GNKPS weights retain the
  InsightFace non-commercial research terms; a registry entry is not a new
  licence grant. Other licence/provenance fields accompany the manifest entries.
* `engine_cache.qualify_providers` attaches ORT's supported engine filename
  prefix: model content, effective precision, graph input shapes, profile/build
  constraints, requested batch and Pixel Boost. The parent cache still isolates
  GPU/driver/runtime. The new filenames require an initial cold build. No
  precision, canary or fidelity gate was relaxed.
* `Enhance_UltraMax.Run` and `RunBatch` lease `VramSessionManager` reservations.
  Live free VRAM caps batch 1–8, with a margin for 4K frames and concurrent
  pending allocations. Cards below 7 GiB stay at one. Recognized batch OOMs
  retry smaller slices in order. Existing pools, locks and precision remain.
  Explicit batches above one use the established cross-worker batch coordinator;
  differing keyed settings cannot share a scalar fidelity input.
* `procmgr_batch` consumes `video_muxer.output_plan`; the established source-color
  writer converts and tags SDR using the source matrix/range/primaries/transfer.
  HDR pixels remain owned by `hdr_pipeline`; SDR output must carry SDR tags.
  `capturer` and Pre-Scan use the same extracted metadata.
* `studio_parameters` scopes timeline values to each frame worker; tracking,
  mask selection, swap tiling, restoration and merger read those values.
  Analysis-key edits invalidate the cached tracking stage. Disabled tracks
  reach the actual pending-swap gate.

Automated contracts: `app/tests/test_studio_integration.py`,
`test_preview_and_timeline_api.py`, `test_faceset_v2.py`,
`test_ffmpeg_writer_colour.py`, `test_hdr_color.py`, and the existing
`tests/test_release_signoff_vfr_color_cancel.py` real FFmpeg roundtrips.
Frontend checks are `npm run test:studio` and
`npx --no-install playwright test -c e2e/playwright.config.js studio-integration.spec.js`.
The browser test observes requests and rejects external network hosts.

Real offline probes use `scripts/verification/verify_studio_integration.py`
(`--part prescan|batch|compile-cpu --out PATH`). The batch probe checks the
restorer independently of Stage 7's plate replacement mask; it does not claim
a physical 4K render or speed improvement. A physical 3060, 4K multi-face peak
memory campaign, active-enhancer full render and fidelity/A-V/leak acceptance
remain separate gates. A free-memory estimate cannot guarantee admission
against another process allocating GPU memory after the query.

Current run results and known failures are recorded in `docs/CHANGELOG.md` and
the media directory's `RECODE_STATUS.md`; do not substitute old suite totals.
