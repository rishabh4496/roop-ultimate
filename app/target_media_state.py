"""Stable target-media identity and isolated target-face contexts.

The processing pipeline still exposes the historical ``roop.globals`` target
arrays.  This module owns the durable boundary around those arrays: every
media entry gets a random, process-stable id and every id has an independent
copy of the target-face context.  The API can therefore keep the old globals
as the active-context compatibility surface without treating them as the
application's source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import uuid


def new_target_media_id() -> str:
    """Return an opaque id that is independent of path, name, and list order."""
    return uuid.uuid4().hex


@dataclass
class TargetMediaContext:
    """Identity-bearing state for one target media item."""

    target_faces: list = field(default_factory=list)
    target_face_group: list = field(default_factory=list)
    target_face_names: dict = field(default_factory=dict)
    target_thumbs: list = field(default_factory=list)
    selected_target_face_index: int = 0
    source_mapping: dict | list = field(default_factory=dict)

    def clone(self) -> "TargetMediaContext":
        return TargetMediaContext(
            target_faces=list(self.target_faces),
            target_face_group=list(self.target_face_group),
            target_face_names=dict(self.target_face_names),
            target_thumbs=list(self.target_thumbs),
            selected_target_face_index=int(self.selected_target_face_index or 0),
            source_mapping=(dict(self.source_mapping) if isinstance(self.source_mapping, dict)
                            else list(self.source_mapping or [])),
        )


class TargetMediaContextStore:
    """In-memory store keyed only by stable target-media ids."""

    def __init__(self):
        self._contexts: dict[str, TargetMediaContext] = {}

    def ensure_media_id(self, entry) -> str:
        value = getattr(entry, "media_id", None)
        if not value:
            value = new_target_media_id()
            setattr(entry, "media_id", value)
        return str(value)

    def save(self, media_id: str, *, target_faces=None, target_face_group=None,
             target_face_names=None, target_thumbs=None,
             selected_target_face_index=0, source_mapping=None) -> TargetMediaContext:
        context = TargetMediaContext(
            target_faces=list(target_faces or []),
            target_face_group=list(target_face_group or []),
            target_face_names=dict(target_face_names or {}),
            target_thumbs=list(target_thumbs or []),
            selected_target_face_index=max(0, int(selected_target_face_index or 0)),
            source_mapping=(dict(source_mapping) if isinstance(source_mapping, dict)
                            else list(source_mapping or [])),
        )
        self._contexts[str(media_id)] = context
        return context.clone()

    def load(self, media_id: str) -> TargetMediaContext:
        return self._contexts.get(str(media_id), TargetMediaContext()).clone()

    def has(self, media_id: str) -> bool:
        return str(media_id) in self._contexts

    def remove(self, media_id: str) -> None:
        self._contexts.pop(str(media_id), None)

    def clear(self) -> None:
        self._contexts.clear()

    def ids(self) -> set[str]:
        return set(self._contexts)
