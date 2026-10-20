"""Persistence of the Stage 1 / Stage 2 results, so an interrupted job resumes at Stage 3.

WHAT IS PERSISTED, AND WHY NOT "THE FRAME PLAN".
The Stage 2 -> Stage 3 hand-off is not `stage2_frame_plan` (that index deliberately holds no Face
objects, see roop/stage2_plan.py). The swap pass consumes

    ProcessMgr._temporal_faces      {frame: [Face]}     gap-filled, smoothed, track-tagged faces
    ProcessMgr._track_assignments   {frame: [(centre, source, track-mean-embedding)]}
    ProcessMgr._track_source_map    {track: (source, track-mean-embedding)}
    ProcessMgr._shot_boundaries     cut frames
    ProcessMgr._temporal_covered    how far the scan reached

so those are what a resume has to get back. The plan is stored beside them and used as an integrity
check: it is rebuilt from the RESTORED faces and must equal the plan the pre-pass sealed.

FRAME NUMBERS. Everything is stored as an ABSOLUTE decoder frame (`frame_start + frame`). A resumed
run starts later (the committed segments are skipped), so the same cache must be re-based: a frame
before the new start is dropped, the rest shift down. A cache that does not cover the new window from
its first frame is refused (`covered` is a prefix count, it cannot describe a hole at the front).

FORMAT. One uncompressed zip (`.npz`) written member by member (so a 100k-face clip never holds a
second copy of its faces in memory) and replaced atomically. `faces.*` is COLUMNAR: one array per
face field when every face has it in one dtype and shape (the normal case: bbox, kps, both landmark
sets, the embeddings), per-face tagged JSON otherwise. Nothing is pickled (`allow_pickle=False` on the
way in), so a tampered cache file cannot execute code. The file's SHA-256 and size live in the
project record (`project_checkpoint.verify_sidecar`).

NOTHING LOSSY. A value this codec cannot restore EXACTLY (type, dtype, shape, value) raises
`Unserializable` at write time, the cache is not written, and the caller logs it and carries on with a
normal resume. A cache that restored "nearly" would swap a different face than the interrupted run.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import zipfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: Rows per JSON block of a face column that has no dtype-uniform array form (the nested pose block, mixed
#: scalar types). Written and parsed one block at a time, so a 100k-face clip never holds one document of
#: all its faces' nested dicts, and a late resume parses only the blocks its window touches.
JSON_BLOCK_ROWS = 2048

FORMAT_NAME = "roop-stage-cache"
FORMAT_VERSION = 1
META_MEMBER = "meta.json"

# Stage-1 modes worth keeping. `fail_safe_full` is a scan that FAILED (stopped, demux error, nothing
# decoded) and fell back to "process everything": correct, but a retry might do better, so it is not
# frozen into the project.
CACHEABLE_STAGE1_MODES = ("sparse_auto_range", "no_target_found", "full_short_clip")


class Unserializable(Exception):
    """A value the cache cannot restore exactly. The cache is not written."""


# ── tagged JSON values ───────────────────────────────────────────────────────────────────────────────
# JSON natively round-trips None, bool, int, str, list and finite python floats EXACTLY (floats are
# written with repr, which is the shortest string that parses back to the same double). Everything else
# is tagged so the type survives: tuples (a tuple and a list compare differently, and `isinstance` checks
# in the pipeline care), numpy scalars (float32 != float64 for hash/format purposes), non-finite floats,
# dicts (any key type, insertion order), and arrays that are not worth a column.

class _Arrays:
    """Registry for arrays met inside tagged JSON. De-duplicated by object identity, so the track-mean
    embedding that every assignment entry of a track shares is stored once; grouped by (dtype, shape) so a
    registry of 40 embeddings is one (40, 512) member, not 40."""

    def __init__(self) -> None:
        self._index: Dict[int, Tuple[str, int]] = {}
        self._groups: Dict[str, List[np.ndarray]] = {}
        self._keep: List[np.ndarray] = []        # id() is only unique while the object is alive

    @staticmethod
    def _group(arr: np.ndarray) -> str:
        return "%s:%s" % (arr.dtype.str, "x".join(str(d) for d in arr.shape) or "scalar")

    def ref(self, arr: np.ndarray) -> Dict[str, Any]:
        hit = self._index.get(id(arr))
        if hit is None:
            group = self._group(arr)
            bucket = self._groups.setdefault(group, [])
            hit = (group, len(bucket))
            bucket.append(np.ascontiguousarray(arr))
            self._index[id(arr)] = hit
            self._keep.append(arr)
        return {"$nd": [hit[0], hit[1]]}

    def members(self) -> Iterable[Tuple[str, np.ndarray]]:
        for n, (group, items) in enumerate(sorted(self._groups.items())):
            yield "arrays/%d" % n, np.stack(items)

    def group_names(self) -> List[str]:
        return [g for g, _ in sorted(self._groups.items())]


def _enc(value: Any, arrays: _Arrays) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    # numpy scalars BEFORE the python numbers: np.float64 IS a subclass of float, and would otherwise be
    # written (and come back) as a plain float.
    if isinstance(value, np.bool_):
        return {"$np": "bool", "v": bool(value)}
    if isinstance(value, np.generic):
        if value.dtype.kind not in "iuf":
            raise Unserializable("numpy scalar of dtype %s" % value.dtype)
        scalar = value.item()
        if isinstance(scalar, float) and not math.isfinite(scalar):
            scalar = repr(scalar)
        return {"$np": value.dtype.str, "v": scalar}
    if type(value) is int:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"$f": repr(value)}
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufc":
            raise Unserializable("array of dtype %s" % value.dtype)
        return arrays.ref(value)
    if isinstance(value, tuple):
        return {"$t": [_enc(v, arrays) for v in value]}
    if isinstance(value, list):
        return [_enc(v, arrays) for v in value]
    if isinstance(value, (set, frozenset)):
        return {"$s": sorted(_enc(v, arrays) for v in value), "frozen": isinstance(value, frozenset)}
    if isinstance(value, dict):
        if type(value) is dict:
            return {"$d": [[_enc(k, arrays), _enc(v, arrays)] for k, v in value.items()]}
        if _is_face_class(type(value)):
            # insightface's Face turns every dict assigned into one into another Face (`__setattr__`), so a
            # real face carries nested Faces (the temporal pose block). Same class back, same order.
            _check_face_consistent(value)
            return {"$face": [[_enc(k, arrays), _enc(v, arrays)] for k, v in value.items()]}
        raise Unserializable("dict subclass %s inside a face" % type(value).__name__)
    raise Unserializable("%s value" % type(value).__name__)


def _is_face_class(cls: type) -> bool:
    return cls.__name__ == "Face" and cls.__module__.startswith("insightface")


def _check_face_consistent(face: Any) -> None:
    """A Face keeps every field twice: as a dict item AND as an instance attribute (`__setattr__` does
    both, `__getattr__` answers None for anything missing). `del face[k]` drops only the item. A face in
    which the two disagree reads differently through `face.k` and `face['k']`; restoring it as a clean
    Face would silently change which of the two readings the pipeline sees, so it is refused."""
    items, attrs = set(face.keys()), set(vars(face))
    if items != attrs:
        odd = sorted(map(str, items ^ attrs))[:4]
        raise Unserializable("a face whose attributes and items disagree (%s)" % ", ".join(odd))


def _dec(value: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    if isinstance(value, list):
        return [_dec(v, arrays) for v in value]
    if not isinstance(value, dict):
        return value
    if "$t" in value:
        return tuple(_dec(v, arrays) for v in value["$t"])
    if "$d" in value:
        return {_hashable(_dec(k, arrays)): _dec(v, arrays) for k, v in value["$d"]}
    if "$face" in value:
        return _face_class("Face")({_dec(k, arrays): _dec(v, arrays) for k, v in value["$face"]})
    if "$s" in value:
        items = [_hashable(_dec(v, arrays)) for v in value["$s"]]
        return frozenset(items) if value.get("frozen") else set(items)
    if "$f" in value:
        return float(value["$f"])
    if "$np" in value:
        v = value["v"]
        if value["$np"] == "bool":
            return np.bool_(v)
        return np.dtype(value["$np"]).type(float(v) if isinstance(v, str) else v)
    if "$nd" in value:
        group, i = value["$nd"]
        return arrays[group][i]
    raise Unserializable("unknown tag %r in a cache file" % (sorted(value),))


def _rows(stacked: np.ndarray) -> List[np.ndarray]:
    """One array object per stored row, created ONCE. `stacked[i]` makes a new view on every access, and the
    pre-pass shares one track-mean embedding between every assignment entry of a track and the source map:
    that sharing has to survive the round trip."""
    return [stacked[i] for i in range(len(stacked))]


def _hashable(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    return v


# ── the sidecar file ─────────────────────────────────────────────────────────────────────────────────

class SidecarWriter:
    """Atomic, member-by-member writer. `with SidecarWriter(path) as w: w.array(..); w.json(..)`."""

    def __init__(self, path: str):
        self.path = path
        self._tmp = None
        self._zip: Optional[zipfile.ZipFile] = None
        self.members: Dict[str, str] = {}

    def __enter__(self) -> "SidecarWriter":
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, self._tmp = tempfile.mkstemp(prefix=".stagecache-", suffix=".tmp", dir=directory)
        self._fh = os.fdopen(fd, "wb")
        self._zip = zipfile.ZipFile(self._fh, "w", compression=zipfile.ZIP_STORED, allowZip64=True)
        return self

    def array(self, name: str, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr)
        if arr.dtype.kind not in "biufc":
            raise Unserializable("member %s has dtype %s" % (name, arr.dtype))
        with self._zip.open(name + ".npy", "w", force_zip64=True) as member:
            np.lib.format.write_array(member, arr, allow_pickle=False)

    def json(self, name: str, obj: Any) -> None:
        self._zip.writestr(name, json.dumps(obj, separators=(",", ":"), allow_nan=False))

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if exc_type is None:
                self._zip.close()
                self._fh.flush()
                os.fsync(self._fh.fileno())
                self._fh.close()
                delay = 0.02
                for attempt in range(8):                 # same transient-PermissionError dance as the JSON record
                    try:
                        os.replace(self._tmp, self.path)
                        self._tmp = None
                        break
                    except PermissionError:
                        if attempt == 7:
                            raise
                        import time
                        time.sleep(delay)
                        delay = min(delay * 2, 0.4)
            else:
                try:
                    self._zip.close()
                finally:
                    self._fh.close()
        finally:
            if self._tmp:
                try:
                    os.remove(self._tmp)
                except OSError:
                    pass
        return False


def file_digest(path: str) -> Tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


class SidecarReader:
    def __init__(self, path: str):
        self._npz = np.load(path, allow_pickle=False)
        self.meta = json.loads(self._npz_text(META_MEMBER))
        if self.meta.get("format") != FORMAT_NAME or self.meta.get("version") != FORMAT_VERSION:
            raise Unserializable("not a %s v%d file" % (FORMAT_NAME, FORMAT_VERSION))

    def _npz_text(self, name: str) -> str:
        with self._npz.zip.open(name) as fh:        # a non-.npy member; NpzFile only decodes .npy itself
            return fh.read().decode("utf-8")

    def array(self, name: str) -> np.ndarray:
        return self._npz[name]

    def json(self, name: str) -> Any:
        return json.loads(self._npz_text(name))

    def close(self) -> None:
        self._npz.close()


# ── Face table (columnar) ────────────────────────────────────────────────────────────────────────────

def _face_class(name: str):
    if name == "Face":
        from insightface.app.common import Face
        return Face
    if name == "dict":
        return dict
    raise Unserializable("face type %s" % name)


def _face_type_name(face: Any) -> str:
    cls = type(face)
    if cls is dict:
        return "dict"
    if _is_face_class(cls):
        _check_face_consistent(face)
        return "Face"
    raise Unserializable("face of type %s.%s" % (cls.__module__, cls.__name__))


def _column_kind(values: List[Any]) -> Optional[str]:
    """The columnar kind that stores `values` (all present) exactly, or None for per-value JSON."""
    first = values[0]
    if isinstance(first, np.ndarray):
        if first.dtype.kind not in "biufc":
            return None
        for v in values:
            if (not isinstance(v, np.ndarray) or v.dtype != first.dtype or v.shape != first.shape):
                return None
        return "nd"
    if isinstance(first, np.generic) and first.dtype.kind in "biufc":
        for v in values:
            if not isinstance(v, np.generic) or v.dtype != first.dtype:
                return None
        return "np"
    if type(first) is float:
        return "float" if all(type(v) is float and math.isfinite(v) for v in values) else None
    if type(first) is bool:
        return "bool" if all(type(v) is bool for v in values) else None
    if type(first) is int:
        if all(type(v) is int and -2 ** 63 <= v < 2 ** 63 for v in values):
            return "int"
        return None
    return None


def write_faces(w: SidecarWriter, faces_by_frame: Mapping[int, Sequence[Any]], frame_base: int) -> Dict[str, Any]:
    """Columnar-encode every face. Returns the meta block; arrays go straight to `w`."""
    frames = sorted(faces_by_frame)
    flat: List[Any] = []
    frame_of: List[int] = []
    for f in frames:
        for face in faces_by_frame[f] or ():
            flat.append(face)
            frame_of.append(int(f) + int(frame_base))
    face_names = [_face_type_name(f) for f in flat]          # also refuses a face whose attrs and items disagree
    type_names = sorted(set(face_names))
    type_code = np.array([type_names.index(n) for n in face_names], dtype=np.uint8)
    w.array("faces/frame", np.asarray(frame_of, dtype=np.int64))
    w.array("faces/type", type_code)

    key_order: List[str] = []
    seen = set()
    for face in flat:
        for key in face.keys():
            if key not in seen:
                seen.add(key)
                key_order.append(key)
    for key in key_order:
        if not isinstance(key, str):
            raise Unserializable("non-string face key %r" % (key,))

    columns: Dict[str, Any] = {}
    for n, key in enumerate(key_order):
        idx = [i for i, face in enumerate(flat) if key in face]
        values = [flat[i][key] for i in idx]
        kind = _column_kind(values) if values else None
        entry: Dict[str, Any] = {"n": n, "present": len(idx), "all": len(idx) == len(flat)}
        if len(idx) != len(flat):
            w.array("faces/%d.idx" % n, np.asarray(idx, dtype=np.int64))
        if kind == "nd" or kind == "np":
            entry["kind"] = kind
            w.array("faces/%d.data" % n, np.stack(values) if kind == "nd" else np.array(values))
        elif kind in ("float", "int", "bool"):
            entry["kind"] = kind
            w.array("faces/%d.data" % n, np.array(values, dtype={"float": np.float64, "int": np.int64,
                                                                   "bool": np.bool_}[kind]))
        else:
            entry["kind"] = "json"
            entry["block"] = JSON_BLOCK_ROWS
            blocks = 0
            for b0 in range(0, len(values), JSON_BLOCK_ROWS):
                reg = _Arrays()
                encoded = [_enc(v, reg) for v in values[b0:b0 + JSON_BLOCK_ROWS]]
                w.json("faces/%d.j%d" % (n, blocks), {"values": encoded, "groups": reg.group_names()})
                for gi, (_, arr) in enumerate(reg.members()):
                    w.array("faces/%d.j%d.a%d" % (n, blocks, gi), arr)
                blocks += 1
            entry["blocks"] = blocks
        columns[key] = entry
    return {"count": len(flat), "types": type_names, "keys": key_order, "columns": columns,
            "frames": len(frames)}


def read_faces(r: SidecarReader, block: Mapping[str, Any], first_abs: int, last_abs: int,
               frame_shift: int) -> Dict[int, List[Any]]:
    """Faces with absolute frame in [first_abs, last_abs), re-keyed to `abs - frame_shift`."""
    frame = r.array("faces/frame")
    lo = int(np.searchsorted(frame, first_abs, side="left"))     # rows are written in frame order
    hi = int(np.searchsorted(frame, last_abs, side="left"))
    n_total = int(block["count"])
    if len(frame) != n_total:
        raise Unserializable("face count differs from its table")
    types = [_face_class(t) for t in block["types"]]
    type_code = r.array("faces/type")
    rows: List[Dict[str, Any]] = [dict() for _ in range(hi - lo)]
    for key in block["keys"]:
        entry = block["columns"][key]
        n = entry["n"]
        if entry["all"]:
            row_of = np.arange(lo, hi)
            sel = slice(lo, hi)
            positions = row_of - lo
        else:
            idx = r.array("faces/%d.idx" % n)
            sel = (idx >= lo) & (idx < hi)
            positions = idx[sel] - lo
        kind = entry["kind"]
        if kind in ("nd", "np", "float", "int", "bool"):
            data = r.array("faces/%d.data" % n)
            data = data[sel] if not entry["all"] else data[lo:hi]
            for j, p in enumerate(positions):
                v = data[j]
                if kind == "nd":
                    rows[p][key] = v.copy()          # own the memory: the npz is closed after the restore
                elif kind == "np":
                    rows[p][key] = v                  # a numpy scalar of the stored dtype
                elif kind == "float":
                    rows[p][key] = float(v)
                elif kind == "int":
                    rows[p][key] = int(v)
                else:
                    rows[p][key] = bool(v)
        else:
            # value index k -> destination row. `all`: the column has one value per face, so k is the row.
            if entry["all"]:
                wanted = [(i, i - lo) for i in range(lo, hi)]
            else:
                idx = r.array("faces/%d.idx" % n)
                wanted = [(int(k), int(idx[k]) - lo) for k in np.nonzero((idx >= lo) & (idx < hi))[0]]
            size = int(entry["block"])
            current, block_values, arrays = None, None, None
            for k, p in wanted:                              # ascending k: each block is parsed once
                b = k // size
                if b != current:
                    doc = r.json("faces/%d.j%d" % (n, b))
                    arrays = {g: _rows(r.array("faces/%d.j%d.a%d" % (n, b, gi))) for gi, g in enumerate(doc["groups"])}
                    block_values, current = doc["values"], b
                rows[p][key] = _dec(block_values[k - b * size], arrays)
    out: Dict[int, List[Any]] = {}
    for j in range(hi - lo):
        cls = types[int(type_code[lo + j])]
        face = cls(rows[j])
        out.setdefault(int(frame[lo + j]) - frame_shift, []).append(face)
    return out


# ── whole-stage capture / restore ────────────────────────────────────────────────────────────────────

def serialize_plan(plan: Mapping[int, Sequence[Any]], frame_base: int = 0) -> List[Any]:
    """The Stage 2 frame plan as plain JSON, absolute frames. Floats are float32 -> double: exact."""
    out = []
    for f in sorted(plan):
        recs = []
        for r in plan[f]:
            recs.append([int(r.track_id), int(r.source_index), float(r.det_score),
                         [float(x) for x in np.asarray(r.bbox, dtype=np.float32).reshape(-1)],
                         None if r.pose is None else [float(x) for x in np.asarray(r.pose, dtype=np.float32).reshape(-1)],
                         bool(r.synthetic)])
        out.append([int(f) + int(frame_base), recs])
    return out


def plan_window(serialized: Sequence[Any], first_abs: int, last_abs: int, shift: int) -> List[Any]:
    """The part of a serialized plan inside [first_abs, last_abs), re-keyed by `- shift`."""
    return [[f - shift, recs] for f, recs in serialized if first_abs <= f < last_abs]


def ranges_digest(ranges_abs: Optional[Sequence[Sequence[int]]]) -> str:
    if ranges_abs is None:
        return "none"
    text = json.dumps([[int(a), int(b)] for a, b in ranges_abs], separators=(",", ":"))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def ranges_to_abs(ranges: Optional[Iterable[Sequence[int]]], frame_start: int) -> Optional[List[List[int]]]:
    if ranges is None:
        return None
    return [[int(a) + int(frame_start), int(b) + int(frame_start)] for a, b in ranges]


def rebase_ranges(ranges_abs: Sequence[Sequence[int]], new_start: int, new_count: int
                  ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Absolute inclusive ranges -> ranges relative to `new_start`, clipped to the window, plus the mask."""
    mask = np.zeros(max(0, int(new_count)), dtype=bool)
    out: List[Tuple[int, int]] = []
    for a, b in ranges_abs:
        lo, hi = max(int(a) - new_start, 0), min(int(b) - new_start, new_count - 1)
        if hi < lo:
            continue
        out.append((lo, hi))
        mask[lo:hi + 1] = True
    return out, mask


def covers_window(bundle_start: int, bundle_end: int, new_start: int, new_count: int) -> bool:
    """True when [bundle_start, bundle_end) contains every frame of the new window."""
    return bundle_start <= new_start and new_start + new_count <= bundle_end


def window_ranges(ranges_abs: Optional[Sequence[Sequence[int]]], first: int, last: int
                  ) -> Optional[List[List[int]]]:
    """`ranges_abs` clipped to the absolute frames [first, last), as plain lists; None stays None.

    A Stage 2 result was sealed against the Stage 1 ranges of the WHOLE original window. A resumed run's
    Stage 1 is clipped to the shorter window (a range wholly before it is gone, one that straddles it starts
    at its edge), so the two are only comparable once BOTH are cut to the window the resume uses."""
    if ranges_abs is None:
        return None
    out = []
    for a, b in ranges_abs:
        lo, hi = max(int(a), int(first)), min(int(b), int(last) - 1)
        if hi >= lo:
            out.append([lo, hi])
    return out


def write_stage2(path: str, pm: Any, *, frame_start: int, frame_count: int,
                 ranges_abs: Optional[Sequence[Sequence[int]]], plan: Mapping[int, Sequence[Any]]) -> Dict[str, Any]:
    """Write the hand-off of a COMPLETED pre-pass. Returns the record entry (JSON-safe).

    Raises `Unserializable` (nothing is left on disk) for a value that would not restore exactly."""
    faces = getattr(pm, "_temporal_faces", None)
    if not faces:
        raise Unserializable("the pre-pass produced no faces")
    covered = int(getattr(pm, "_temporal_covered", 0) or 0)
    if covered < int(frame_count):
        raise Unserializable("the pre-pass covered %d of %d frames (stopped or short)" % (covered, frame_count))
    assignments = getattr(pm, "_track_assignments", None) or {}
    source_map = getattr(pm, "_track_source_map", None) or {}
    shots = getattr(pm, "_shot_boundaries", None)
    reg = _Arrays()
    with SidecarWriter(path) as w:
        block = write_faces(w, faces, frame_start)
        assign_enc = [[int(f) + int(frame_start), _enc(list(entries), reg)] for f, entries in sorted(assignments.items())]
        smap_enc = [[_enc(k, reg), _enc(v, reg)] for k, v in source_map.items()]
        shots_enc = None if shots is None else {"type": type(shots).__name__,
                                                "frames": sorted(int(s) + int(frame_start) for s in shots)}
        if shots_enc and shots_enc["type"] not in ("set", "list", "tuple", "frozenset"):
            raise Unserializable("shot boundaries of type %s" % shots_enc["type"])
        for name, arr in reg.members():
            w.array(name, arr)
        meta = {
            "format": FORMAT_NAME, "version": FORMAT_VERSION,
            "frame_start": int(frame_start), "frame_count": int(frame_count),
            "covered_abs": int(frame_start) + covered,
            "ranges_abs": None if ranges_abs is None else [[int(a), int(b)] for a, b in ranges_abs],
            "faces": block, "assignments": assign_enc, "source_map": smap_enc,
            "arrays": reg.group_names(), "shots": shots_enc,
            "plan": serialize_plan(plan, frame_start),
        }
        w.json(META_MEMBER, meta)
    digest, size = file_digest(path)
    return {"frame_start": int(frame_start), "frame_count": int(frame_count),
            "covered_abs": meta["covered_abs"], "ranges_digest": ranges_digest(ranges_abs),
            "faces": block["count"], "frames": block["frames"], "plan_frames": len(meta["plan"]),
            "sidecar": {"sha256": digest, "size": size}}


def restore_stage2(path: str, pm: Any, *, new_start: int, new_count: int,
                   current_ranges_abs: Optional[Sequence[Sequence[int]]], build_plan) -> Dict[str, Any]:
    """Apply a stored hand-off to `pm` for a run whose window is [new_start, new_start + new_count).

    `build_plan(faces_by_frame, track_source_map)` -> (plan, stats) is `stage2_plan.build_stage2_plan`
    (injected so this module stays importable without the pipeline). Raises `Unserializable` for ANY
    reason the cache must not be used; on a raise `pm` is left untouched.
    """
    r = SidecarReader(path)
    try:
        meta = r.meta
        b_start, b_end = int(meta["frame_start"]), int(meta["covered_abs"])
        if not covers_window(b_start, b_end, new_start, new_count):
            raise Unserializable("cache covers frames %d-%d, the run needs %d-%d"
                                 % (b_start, b_end - 1, new_start, new_start + new_count - 1))
        first, last = int(new_start), int(new_start) + int(new_count)
        saved_ranges = meta.get("ranges_abs")
        if ((saved_ranges is None) != (current_ranges_abs is None)
                or window_ranges(saved_ranges, first, last) != window_ranges(current_ranges_abs, first, last)):
            raise Unserializable("Stage 1 ranges differ from the ones Stage 2 was computed against")
        faces = read_faces(r, meta["faces"], first, last, new_start)
        if not faces:
            # A pre-pass that finds no face turns itself OFF and hands the clip to per-frame detection
            # (see _precompute_temporal): that is not the same as "cache says nobody is here".
            raise Unserializable("no cached face falls inside the resumed window")
        arrays = {g: _rows(r.array("arrays/%d" % i)) for i, g in enumerate(meta["arrays"])}
        assignments = {}
        for f_abs, entries in meta["assignments"]:
            if first <= f_abs < last:
                assignments[int(f_abs) - new_start] = _dec(entries, arrays)
        source_map = {}
        for k, v in meta["source_map"]:
            source_map[_dec(k, arrays)] = _dec(v, arrays)
        shots = None
        if meta["shots"] is not None:
            frames = [s - new_start for s in meta["shots"]["frames"] if first <= s < last]
            shots = {"set": set, "frozenset": frozenset, "list": list, "tuple": tuple}[meta["shots"]["type"]](frames)
        plan, _ = build_plan(faces, source_map)
        rebuilt = serialize_plan(plan, new_start)
        saved = plan_window(meta["plan"], first, last, 0)
        if rebuilt != saved:
            raise Unserializable("the plan rebuilt from the restored faces differs from the sealed plan "
                                 "(%d vs %d frames)" % (len(rebuilt), len(saved)))
        pm._temporal_faces = faces
        pm._temporal_covered = min(int(new_count), b_end - int(new_start))
        pm._track_scanned = pm._temporal_covered
        pm._track_assignments = assignments
        pm._track_source_map = source_map
        pm._track_pose_source_map = {}
        pm._shot_boundaries = shots
        pm.stage2_frame_plan = plan
        return {"frames": len(faces), "faces": sum(len(v) for v in faces.values()),
                "plan_frames": len(plan), "assignments": len(assignments)}
    finally:
        r.close()
