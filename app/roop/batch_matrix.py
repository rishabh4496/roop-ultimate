"""Track -> faceset resolution for the Batch Matrix strategies, and the record of what the pre-pass actually decided.

The four Batch Matrix strategies (``one_to_many``, ``grouped``, per-file ``matrix`` and explicit ``tracks``; the
UI's ``recipes`` stages the same single-target jobs as ``one_to_many`` / ``matrix``) all converge on ONE job shape
(``docs/development/BATCH_MATRIX_DATA_FLOW.md``): one target x one person -> faceset mapping, carried as
``face_mapping`` / ``source_mapping_ids`` / ``source_mapping_names`` on the payload. The strategy is therefore not what
decides which faceset a track gets; the payload is, and these functions make that decision a pure, deterministic
function of the payload and the current gallery:

    plan_track_facesets      BEFORE the render: payload + {track: person rank} + gallery -> {track: faceset}
    resolve_track_facesets   AFTER Stage 2: ``ProcessMgr._track_source_map`` + the run's facesets -> {track: faceset}

Both return the same ``TrackBinding`` rows, in ascending track order, and neither reads a dict's insertion order,
a gallery position that an id could replace, or the clock: the same inputs give the same answer on every call and in
every process. A gallery INDEX is only the last-resort fallback (the same priority ``processing_request`` uses:
stable id, then name, then raw index), and a binding that rests on it says so in ``via``.

``set_active`` / ``active`` hold the bindings of the job that is running, for ``GET /api/trajectory/spans`` and the
job record, the way ``trajectory_cache.set_active`` holds the trajectory.
"""
from __future__ import annotations

import os
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from roop.processing_request import (normalize_source_index_mapping, resolve_source_mapping_ids,
                                     resolve_source_mapping_names)

STRATEGIES = ("one_to_many", "grouped", "matrix", "tracks")
# The UI strategy whose jobs are one_to_many / matrix jobs once staged.
STRATEGY_ALIASES = {"recipes": "matrix", "per_file_matrix": "matrix", "per-file matrix": "matrix",
                    "one-to-many": "one_to_many", "multi_track": "tracks", "multitrack": "tracks",
                    "multi-track": "tracks", "fsz": "tracks"}


@dataclass(frozen=True)
class TrackBinding:
    track_id: int
    source_index: Optional[int]       # position in the list the binding was resolved against (None = no faceset)
    faceset_id: Optional[str]         # the stable gallery id (absolute path, or ``memory:<uuid>``)
    faceset_path: Optional[str]       # the file on disk, or None for an in-memory faceset
    via: str                          # prepass | id | name | index | track_binding | none
    reason: Optional[str] = None      # why there is no faceset

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── gallery entries ────────────────────────────────────────────────────────────────────────────────────
def _abs(path: Any) -> Optional[str]:
    text = str(path or "").strip()
    return os.path.abspath(text) if text else None


def gallery_entries(facesets: Iterable[Any]) -> List[Dict[str, Any]]:
    """``[{index, id, path, name}]`` for the facesets of a run, in list order. Works on ``FaceSet`` objects (their
    ``_source_id`` / ``_source_path``) and on ``source_faces_info`` rows (``id`` / ``name``)."""
    out: List[Dict[str, Any]] = []
    for index, fs in enumerate(facesets or ()):
        if isinstance(fs, Mapping):
            sid, path, name = fs.get("id"), fs.get("path"), fs.get("name")
            faces = fs.get("count", fs.get("faces"))
            faces = 1 if faces is None else int(faces)
        else:
            sid, path = getattr(fs, "_source_id", None), getattr(fs, "_source_path", None)
            name = os.path.basename(str(path or "")) or None
            faces = len(getattr(fs, "faces", None) or ())
        path = _abs(path)
        sid = str(sid or "").strip() or path or None
        if sid and sid.startswith("memory:"):
            path = None
        elif path is None and sid:
            path = _abs(sid)
        out.append({"index": index, "id": sid, "path": path, "faces": faces,
                    "name": str(name or "").strip() or (os.path.basename(path) if path else None)})
    return out


def _binding(track_id: int, entry: Optional[Mapping[str, Any]], via: str, reason: Optional[str] = None
             ) -> TrackBinding:
    if entry is None:
        return TrackBinding(int(track_id), None, None, None, "none", reason or "no faceset")
    if not entry.get("id") and not entry.get("path"):
        # Assigned, but the faceset carries no stable identity (an in-memory set nobody has listed in the gallery,
        # which is how the regression benchmark builds its source). The assignment is real; only the NAME is missing.
        return TrackBinding(int(track_id), int(entry["index"]), None, None, via,
                            reason or "source %d has no id or path (an in-memory faceset)" % int(entry["index"]))
    return TrackBinding(int(track_id), int(entry["index"]), entry.get("id"), entry.get("path"), via, None)


def _source_of(value: Any) -> Optional[int]:
    """The source index of a ``_track_source_map`` value: ``(source, embedding)``, a bare int, or None."""
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ── after Stage 2: what the pre-pass decided ───────────────────────────────────────────────────────────
def resolve_track_facesets(track_source_map: Optional[Mapping[Any, Any]], entries: Sequence[Mapping[str, Any]]
                           ) -> Dict[int, TrackBinding]:
    """``{track: TrackBinding}`` from the pre-pass's own ``track -> (source, embedding)`` map. ``entries`` is the run's
    faceset list (``gallery_entries``), the list the source index points into. A track with no source, a negative
    source or one past the list resolves to a binding with ``source_index None`` and the reason, never to faceset 0. A
    track assigned to a faceset that has no id or path keeps its ``source_index`` and gets ``faceset_id None`` plus a
    reason: assigned is not the same as nameable."""
    out: Dict[int, TrackBinding] = {}
    by_index = {int(e["index"]): e for e in entries}
    for raw_tid in sorted(track_source_map or {}, key=lambda t: (str(type(t)), t)):
        try:
            tid = int(raw_tid)
        except (TypeError, ValueError):
            continue
        src = _source_of(track_source_map[raw_tid])
        if src is None or src < 0:
            out[tid] = _binding(tid, None, "none", "the pre-pass assigned this track no source")
        elif src not in by_index:
            out[tid] = _binding(tid, None, "none", "source %d is not in this run's faceset list" % src)
        elif not by_index[src].get("faces", 1):
            out[tid] = _binding(tid, None, "none", "source %d is an empty faceset (that person is skipped)" % src)
        else:
            out[tid] = _binding(tid, by_index[src], "prepass")
    return out


# ── before the render: payload -> track bindings ───────────────────────────────────────────────────────
def strategy_of(payload: Mapping[str, Any]) -> str:
    """The strategy a job was staged by, from ``batch_strategy`` when present, else from its shape: explicit
    ``track_bindings`` -> ``tracks``, anything else is a person-rank mapping (``one_to_many`` / ``grouped`` /
    ``matrix`` all stage the same rank mapping, so they resolve identically)."""
    raw = str(payload.get("batch_strategy") or payload.get("strategy") or "").strip().lower()
    raw = STRATEGY_ALIASES.get(raw, raw)
    if raw in STRATEGIES:
        return raw
    return "tracks" if payload.get("track_bindings") else "one_to_many"


def _match_ref(ref: Any, entries: Sequence[Mapping[str, Any]]):
    """``(entry, via, reason)`` for one explicit reference: an id or path (case-folded, normalised), else a unique
    file name. An integer is refused: a gallery position is not stable (reorder, removal) and an explicit binding is
    exactly where that must not be guessed."""
    if isinstance(ref, bool) or isinstance(ref, (int, float)):
        return None, "none", "a gallery position is not a stable reference; pass the faceset id or path"
    text = str(ref or "").strip()
    if not text:
        return None, "none", "empty faceset reference"
    key = os.path.normcase(os.path.abspath(text)) if not text.startswith("memory:") else text.casefold()
    for e in entries:
        for cand in (e.get("id"), e.get("path")):
            if cand and (os.path.normcase(os.path.abspath(cand)) if not str(cand).startswith("memory:")
                         else str(cand).casefold()) == key:
                return e, "id", None
    named = [e for e in entries if e.get("name") and str(e["name"]).casefold() == os.path.basename(text).casefold()]
    if len(named) == 1:
        return named[0], "name", None
    if len(named) > 1:
        return None, "none", "%d facesets are named %s; pass the id" % (len(named), os.path.basename(text))
    return None, "none", "no faceset matches %s" % text


def plan_track_facesets(payload: Mapping[str, Any], entries: Sequence[Mapping[str, Any]],
                        track_person: Optional[Mapping[Any, Any]] = None, tracks: Iterable[Any] = ()
                        ) -> Dict[int, TrackBinding]:
    """``{track: TrackBinding}`` for a job, before it renders.

    ``tracks``         every track id to answer for (the plan's tracks); explicit bindings add their own.
    ``track_person``   ``{track: person rank}`` for the rank strategies (the caller knows which person a track is).

    ``tracks`` strategy: ``payload['track_bindings'] = {track: faceset id | path | name}``.
    Rank strategies: ``face_mapping[rank]`` resolved the way ``processing_request`` does it, by
    ``source_mapping_ids``, then ``source_mapping_names``, then the raw index.
    """
    strategy = strategy_of(payload)
    count = len(entries)
    out: Dict[int, TrackBinding] = {}
    if strategy == "tracks":
        raw = payload.get("track_bindings") or {}
        wanted = {int(t) for t in tracks} | {int(t) for t in raw if str(t).lstrip("-").isdigit()}
        by_tid = {int(k): v for k, v in raw.items() if str(k).lstrip("-").isdigit()}
        for tid in sorted(wanted):
            if tid not in by_tid:
                out[tid] = _binding(tid, None, "none", "no binding for this track")
                continue
            entry, via, reason = _match_ref(by_tid[tid], entries)
            out[tid] = _binding(tid, entry, "track_binding" if entry is not None else "none", reason)
        return out

    mapping = payload.get("face_mapping")
    mapping = list(mapping) if isinstance(mapping, (list, tuple)) else []
    mode = str(payload.get("swap_mode") or "selected")
    ids = [e.get("id") for e in entries]
    names = [e.get("name") for e in entries]
    resolved = None
    via = "index"
    if payload.get("source_mapping_ids") is not None:
        resolved = resolve_source_mapping_ids(mapping, list(payload["source_mapping_ids"]), ids, count, mode)
        via = "id"
    if resolved is None and payload.get("source_mapping_names") is not None:
        resolved = resolve_source_mapping_names(mapping, list(payload["source_mapping_names"]), names, count, mode)
        via = "name"
    if resolved is None:
        resolved = normalize_source_index_mapping(mapping, count, mode)
        via = "index"
    resolved = resolved or []
    by_index = {int(e["index"]): e for e in entries}
    for raw_tid in sorted({int(t) for t in tracks} | {int(t) for t in (track_person or {})}):
        rank = (track_person or {}).get(raw_tid)
        try:
            rank = None if rank is None else int(rank)
        except (TypeError, ValueError):
            rank = None
        if rank is None:
            out[raw_tid] = _binding(raw_tid, None, "none", "the track is not associated with a target person")
        elif not 0 <= rank < len(resolved):
            out[raw_tid] = _binding(raw_tid, None, "none", "person %d has no mapping" % rank)
        elif resolved[rank] is None or int(resolved[rank]) < 0 or int(resolved[rank]) not in by_index:
            out[raw_tid] = _binding(raw_tid, None, "none", "person %d is mapped to skip or to a missing faceset" % rank)
        else:
            out[raw_tid] = _binding(raw_tid, by_index[int(resolved[rank])], via)
    return out


# ── the active job's bindings ──────────────────────────────────────────────────────────────────────────
_lock = threading.Lock()
_active: Dict[int, TrackBinding] = {}
_active_project: Optional[str] = None


def set_active(bindings: Optional[Mapping[int, TrackBinding]], project_id: Optional[str] = None) -> None:
    global _active, _active_project
    with _lock:
        _active = dict(bindings or {})
        _active_project = project_id


def clear_active() -> None:
    set_active(None, None)


def active() -> Dict[int, TrackBinding]:
    with _lock:
        return dict(_active)


def active_json() -> List[Dict[str, Any]]:
    with _lock:
        return [b.as_dict() for _, b in sorted(_active.items())]
