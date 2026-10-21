"""The render pipeline's fixed choices (Phase 12).

These used to be user settings and are now decisions the pipeline makes for every video:

  * **In-memory streaming.** Frames are decoded, swapped and piped straight into the encoder
    (`ProcessMgr.run_batch_inmem`). The disk route (extract every frame to PNG/JPG, swap the files, re-encode:
    `core.batch_process`' `not use_new_method` branch) is no longer reachable from the React app or
    `POST /api/swap`; `Keep frames` and `Wait before creating video` only meant something on that route.
  * **The 3-stage pipeline.** Stage 1 (sparse pre-scan) runs for every video. A still image never reaches it
    (images go through `run_batch`, not `run_batch_inmem`) and an animated WebP is excluded in
    `procmgr_batch` itself. A scan that finds nothing passes the frames through untouched; one that cannot look
    fails safe to the full range (`sparse_range_scan`), so the automatic default cannot drop a swap.
  * **Audio is kept.** Stage 4 stream-copies the source audio onto the render (`util_ffmpeg.restore_audio`,
    `-c:a copy`) and verifies the result (`finalize_output`). There is no user switch for dropping it.

`skip_audio` and `sparse_auto_range` stay readable from the request PAYLOAD, and only there: they are the
internal channel `distributed_render` uses to render audio-less chunks, and the one the test harnesses use to
build the dense 1-stage baseline (`tests/verify_three_stage_pipeline.py`). Neither is in config.yaml, in
`settings.schema.json` or on screen.
"""
from typing import Any, Dict, Mapping, Optional


def streaming_render_flags(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    payload = payload or {}
    return {
        'use_new_method': True,
        'keep_frames': False,
        'wait_after_extraction': False,
        'skip_audio': bool(payload.get('skip_audio', False)),
        'sparse_auto_range': bool(payload.get('sparse_auto_range', True)),
    }
