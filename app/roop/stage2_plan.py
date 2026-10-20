"""A read-only index of what the Stage 2 pre-pass decided: which frames hold a face that was
matched to a source, and to which.

This is NOT the Stage 2 -> Stage 3 hand-off. That is ``ProcessMgr._temporal_faces``
({frame: [Face]}, consumed and evicted frame by frame by ``swap_faces``) together with
``_track_source_map`` ({track_id: (source, embedding)}); both already exist and the swap pass
reads them. The plan is a thin view over those two, built once the pre-pass is done, so the
question "which frames will the swap pass actually do work on, and for whom" has an answer
that is not a re-derivation.

It deliberately holds NO Face objects and no landmark arrays. The swap pass evicts
``_temporal_faces[frame]`` as it goes so a long clip's faces are not all resident (a Face is
~5.8 KB: 106x2 + 68x3 float32 landmarks, two 512-float embeddings, kps, box, pose); a plan
that referenced them would pin every one for the whole render - ~580 MB per 100k face-frames
on a machine the project holds under 2.5 GB of RSS. A record here is a box, a pose, a score
and two integers.

Consumers: ``summary`` (the ``[Stage2]`` line printed after the pre-pass), the invariant
``check`` run beside it, and ``ProcessMgr.stage2_frame_plan`` for anything that wants the
index. Making the swap pass SKIP frames absent from it is not done: the swap pass has
per-frame fallbacks (nearest-centroid re-association, the 0.75 distance fallback) that can
still swap a face whose track was left without a source, so a skip would change swap rate
and has to be measured with the regression benchmark first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class VerifiedTargetFace:
    frame_idx: int
    track_id: int
    source_index: int
    det_score: float
    bbox: np.ndarray                # float32 (4,)
    pose: Optional[np.ndarray]      # float32 (3,) or None
    synthetic: bool                 # gap-filled / coasted: landmarks were invented, not detected


def _field(face: Any, name: str) -> Any:
    if isinstance(face, Mapping):
        return face.get(name)
    return getattr(face, name, None)


def _source_of(binding: Any) -> Optional[int]:
    """The source index of a ``_track_source_map`` value ((source, embedding) or a bare int)."""
    if isinstance(binding, (tuple, list)):
        binding = binding[0] if binding else None
    return None if binding is None else int(binding)


def build_stage2_plan(faces_by_frame: Mapping[int, List[Any]],
                      track_source_map: Mapping[Any, Any]
                      ) -> Tuple[Dict[int, List[VerifiedTargetFace]], Dict[str, int]]:
    """{frame: [VerifiedTargetFace]} for frames with at least one matched face, plus counts.

    A face is matched when it carries the id of a track that was assigned a source."""
    plan: Dict[int, List[VerifiedTargetFace]] = {}
    stats = dict(frames_with_faces=0, frames_matched=0, faces=0, matched=0, synthetic=0,
                 unmatched_faces=0, untracked_faces=0)
    for f_idx in sorted(faces_by_frame):
        faces = faces_by_frame[f_idx] or []
        if faces:
            stats['frames_with_faces'] += 1
        records = []
        for face in faces:
            stats['faces'] += 1
            tid = _field(face, '_track_id')
            if tid is None:
                stats['untracked_faces'] += 1
                continue
            src = _source_of(track_source_map.get(tid))
            if src is None:
                stats['unmatched_faces'] += 1
                continue
            try:
                score = float(_field(face, 'det_score'))
            except (TypeError, ValueError):
                score = 0.0
            pose = _field(face, 'pose')
            synthetic = bool(_field(face, '_interpolated') or _field(face, '_coasted'))
            records.append(VerifiedTargetFace(
                frame_idx=int(f_idx), track_id=int(tid), source_index=src,
                det_score=score if np.isfinite(score) else 0.0,
                bbox=np.asarray(_field(face, 'bbox'), dtype=np.float32).copy(),
                pose=None if pose is None else np.asarray(pose, dtype=np.float32).copy(),
                synthetic=synthetic))
            stats['matched'] += 1
            stats['synthetic'] += int(synthetic)
        if records:
            plan[int(f_idx)] = records
            stats['frames_matched'] += 1
    return plan, stats


def check(plan: Mapping[int, List[VerifiedTargetFace]], active_mask: Optional[Any],
          covered: int) -> Dict[str, int]:
    """Invariants that must hold for the index to mean anything.

    outside_active  plan frames the Stage 1 mask had switched OFF - the swap pass passes those
                    through untouched, so a matched face there is work that will not happen
    past_scan       plan frames at or beyond what the pre-pass scanned (stale data)
    """
    outside = past = 0
    mask_len = 0 if active_mask is None else len(active_mask)
    for f_idx in plan:
        if active_mask is not None and 0 <= f_idx < mask_len and not bool(active_mask[f_idx]):
            outside += 1
        if covered and f_idx >= covered:
            past += 1
    return dict(outside_active=outside, past_scan=past)


def summary(stats: Mapping[str, int], problems: Mapping[str, int],
            active_frames: Optional[int]) -> str:
    line = (f'[Stage2] plan: {stats["frames_matched"]} frame(s) hold a face matched to a source '
            f'({stats["matched"]} face(s), {stats["synthetic"]} gap-filled); '
            f'{stats["frames_with_faces"] - stats["frames_matched"]} more hold only unmatched '
            f'faces ({stats["unmatched_faces"] + stats["untracked_faces"]})')
    if active_frames is not None:
        line += (f'; {max(0, active_frames - stats["frames_with_faces"])} of {active_frames} '
                 f'active-range frame(s) hold no face')
    line += '.'
    if problems.get('outside_active'):
        line += (f' WARNING: {problems["outside_active"]} matched frame(s) lie OUTSIDE the active '
                 f'ranges and will pass through unswapped.')
    if problems.get('past_scan'):
        line += (f' WARNING: {problems["past_scan"]} matched frame(s) lie past what the pre-pass '
                 f'scanned.')
    return line
