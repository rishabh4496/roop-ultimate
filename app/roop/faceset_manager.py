"""Read-only access to a faceset's identity embeddings in a chosen recognition model's space.

WHAT THIS IS NOT. It never rewrites an archive and never touches the swapper's identity. Every
embedding stored in a ``.fsz`` is in the buffalo_l / w600k_r50 space (registry key ``default``) BY
DESIGN: the swapper consumes exactly that vector (``roop/swap_identity.py``), and the temporal
identity code pairs a source's stored w600k vector with the target's w600k vector. "Migrating" an
archive to another backbone would overwrite the swap identity, so this module only COMPUTES a
second set of embeddings, in memory, alongside the archive.

WHY RE-EMBEDDING IS CHEAP AND SAFE. A legacy ``.fsz`` is a ZIP of reference PNGs with no embeddings
at all, and ``source_gallery._ingest_faceset`` re-detects and re-embeds them on every load; a V2
archive's ``metadata.json`` is an index and cache over that. So there is no stale vector to detect in
the existing formats -- only a question of which model's space a caller wants:

  * the stored/swap space (``default``): V2 archives hand back their cached normalised embeddings
    without detection or an engine; legacy archives (and V2 rows with no cached vector) take the
    detector's own w600k embedding;
  * any other registered model: the reference images are detected, aligned with the repo's
    ``align_crop`` and passed through the ACTIVE recognition engine (``face_analyser``).

Archives written today carry no embedding-model tag; ``identity.embedding_model`` is honoured if a
future writer adds one, and absence means ``default``.

Nothing here has a render consumer yet (like the rest of the recognition API): a gate that wants
these vectors must first have a calibrated threshold for that model.
"""

import hashlib
import threading
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from roop.faceset_v2 import read_faceset_archive
from roop.recognition_registry import get_model_spec

STORED_MODEL_ID = "default"                  # the space every embedding written into an .fsz lives in
MAX_MEMBER_BYTES = 64 * 1024 * 1024          # a reference image larger than this is skipped, not decoded
_CACHE_SIZE = 16

Detector = Callable[[np.ndarray], Sequence[Any]]
Embedder = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, float]]


@dataclass(frozen=True)
class FacesetEntry:
    member: str                  # the reference image inside the archive
    face_index: int              # which detected face of that image (0 for a V2 row's matched face)
    embedding: np.ndarray        # unit length, read-only


@dataclass(frozen=True)
class FacesetEmbeddings:
    path: str
    sha256: str
    format: str                              # 'legacy' | 'v2'
    stored_model_id: str                     # the space the archive's own embeddings are in
    tagged: bool                             # True only if the archive itself declares it
    model_id: str                            # the space `entries` are in
    status: str                              # 'stored' | 'detected' | 'recomputed'
    entries: Tuple[FacesetEntry, ...]
    skipped: Tuple[Tuple[str, str], ...]     # (member, reason): never silently dropped

    @property
    def embeddings(self) -> List[np.ndarray]:
        return [e.embedding for e in self.entries]

    @property
    def matches_stored(self) -> bool:
        return self.model_id == self.stored_model_id


_cache: "OrderedDict[Tuple[str, str], FacesetEmbeddings]" = OrderedDict()
_cache_lock = threading.Lock()


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _field(face: Any, name: str) -> Any:
    value = getattr(face, name, None)
    if value is None and isinstance(face, dict):
        value = face.get(name)
    return value


def _unit(vector: Any) -> Optional[np.ndarray]:
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if arr.size == 0 or not np.isfinite(norm) or norm <= 1e-8:
        return None
    out = (arr / norm).astype(np.float32)
    out.setflags(write=False)
    return out


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _legacy_members(names: Sequence[str]) -> List[str]:
    return sorted(n for n in names if "/" not in n and "\\" not in n and n.lower().endswith(".png"))


def _stored_model(metadata: Optional[Dict[str, Any]]) -> Tuple[str, bool]:
    tag = ((metadata or {}).get("identity") or {}).get("embedding_model")
    if not tag:
        return STORED_MODEL_ID, False
    get_model_spec(str(tag))                               # a tag this build cannot interpret is an error
    return str(tag), True


def describe_faceset(fsz_path: str) -> Dict[str, Any]:
    """What an archive is and which embedding space it declares; reads, never writes."""
    metadata = read_faceset_archive(fsz_path)
    stored, tagged = _stored_model(metadata)
    with zipfile.ZipFile(fsz_path, "r") as zf:
        names = zf.namelist()
    if metadata is None:
        members, cached = _legacy_members(names), 0
    else:
        sources = metadata.get("sources") or []
        members = [s.get("reference_member") for s in sources]
        cached = sum(1 for v in ((metadata.get("index") or {}).get("normalized_embeddings") or []) if v)
    return {"format": "legacy" if metadata is None else "v2",
            "version": 1 if metadata is None else int(metadata.get("version", 2)),
            "stored_model_id": stored, "tagged": tagged, "reference_members": members,
            "cached_embeddings": cached}


def _decode(zf: zipfile.ZipFile, member: str) -> Tuple[Optional[np.ndarray], str]:
    try:
        info = zf.getinfo(member)
    except KeyError:
        return None, "member missing from the archive"
    if info.file_size > MAX_MEMBER_BYTES:
        return None, "reference image larger than %d MB" % (MAX_MEMBER_BYTES // (1024 * 1024))
    frame = cv2.imdecode(np.frombuffer(zf.read(member), dtype=np.uint8), cv2.IMREAD_COLOR)
    return (frame, "") if frame is not None else (None, "not a decodable image")


def _iou_in_target(target: np.ndarray, box: np.ndarray) -> float:
    """Overlap of `box` with the metadata's bbox, as a fraction of the metadata's area (the loader's rule)."""
    ix0, iy0 = max(target[0], box[0]), max(target[1], box[1])
    ix1, iy1 = min(target[2], box[2]), min(target[3], box[3])
    inter = max(0.0, float(ix1 - ix0)) * max(0.0, float(iy1 - iy0))
    return inter / max(1.0, float((target[2] - target[0]) * (target[3] - target[1])))


def _default_detector() -> Detector:
    from roop.face_util import get_all_faces            # needs the app's initialised detector
    return lambda frame: get_all_faces(frame) or []


def _default_embedder(model_id: str) -> Embedder:
    """The active engine, and ONLY if it is already the requested model: reading must not download or hot-swap."""
    from roop import face_analyser
    loaded = face_analyser.recognition_model_name()
    if loaded != model_id:
        raise ValueError(
            "recognition model %r requested but %s is loaded; call "
            "face_analyser.set_recognition_model(%r) first (loading a faceset does not swap engines)"
            % (model_id, "none" if loaded is None else repr(loaded), model_id))
    return face_analyser.extract_face_embedding


def load_and_validate_faceset(fsz_path: str, active_model_id: str, *,
                              detector: Optional[Detector] = None,
                              embedder: Optional[Embedder] = None) -> FacesetEmbeddings:
    """Embeddings for every reference face of `fsz_path` in `active_model_id`'s space. READ ONLY.

    Raises ValueError for an unknown model, a corrupt archive, or an engine that is not the requested
    model. Faces that cannot be embedded are listed in ``skipped`` with the reason, never dropped.
    Results are cached per (archive content hash, model) and are immutable.
    """
    spec = get_model_spec(active_model_id)                 # ValueError names the valid models
    metadata = read_faceset_archive(fsz_path)              # V2: schema + per-member SHA-256 verified
    digest = _sha256_file(fsz_path)
    key = (digest, active_model_id)
    # Only the production detector/embedder are cached: a result computed by an injected stand-in must not
    # be served to a later caller (or poison one).
    cacheable = detector is None and embedder is None
    if cacheable:
        with _cache_lock:
            if key in _cache:
                _cache.move_to_end(key)
                return replace(_cache[key], path=fsz_path)

    stored_id, tagged = _stored_model(metadata)
    same_space = active_model_id == stored_id
    entries: List[FacesetEntry] = []
    skipped: List[Tuple[str, str]] = []
    status = "recomputed" if not same_space else "detected"

    with zipfile.ZipFile(fsz_path, "r") as zf:
        names = zf.namelist()
        if metadata is None:
            plan = [(m, None) for m in _legacy_members(names)]
            if not plan:
                raise ValueError("FaceSet archive contains no PNG reference members")
        else:
            sources = metadata.get("sources") or []
            cached = (metadata.get("index") or {}).get("normalized_embeddings") or []
            plan = [(s.get("reference_member"), (s, cached[i] if i < len(cached) else None))
                    for i, s in enumerate(sources)]

        # V2 + same space + a cached vector for every row: hand them back, no detection, no engine.
        if metadata is not None and same_space and plan and all(_unit(p[1][1]) is not None for p in plan):
            entries = [FacesetEntry(member, 0, _unit(row[1])) for member, row in plan]
            status = "stored"
        else:
            detect = detector or _default_detector()
            embed = None
            if not same_space:
                embed = embedder or _default_embedder(active_model_id)
            frames: Dict[str, Tuple[Optional[np.ndarray], List[Any]]] = {}
            used: Dict[str, set] = {}
            for member, row in plan:
                if member not in frames:
                    frame, reason = _decode(zf, member)
                    if frame is None:
                        skipped.append((str(member), reason))
                        frames[member] = (None, [])
                    else:
                        frames[member] = (frame, list(detect(frame)))
                    used[member] = set()
                frame, faces = frames[member]
                if frame is None:
                    continue
                if row is None:                                    # legacy: every detected face, in order
                    chosen = [(i, f) for i, f in enumerate(faces)]
                    if not chosen:
                        skipped.append((member, "no face detected"))
                else:                                              # V2: the face matching this row's bbox
                    target = (row[0].get("geometry") or {}).get("bbox")
                    best, best_value = None, -1.0
                    for i, face in enumerate(faces):
                        if i in used[member]:
                            continue
                        box = _field(face, "bbox")
                        if target is None or box is None:
                            value = 0.0 if best is not None else 1.0
                        else:
                            value = _iou_in_target(np.asarray(target, np.float32).reshape(4),
                                                   np.asarray(box, np.float32).reshape(4))
                        if value > best_value:
                            best, best_value = i, value
                    if best is None:
                        skipped.append((member, "no detected face matches this metadata row"))
                        chosen = []
                    else:
                        used[member].add(best)
                        chosen = [(best, faces[best])]
                for index, face in chosen:
                    if same_space:
                        vector = _unit(_field(face, "embedding"))
                    else:
                        kps = _field(face, "kps")
                        vector, quality = embed(frame, kps) if kps is not None else (None, 0.0)
                        if vector is not None and (quality <= 0.0 or np.asarray(vector).size != spec.output_dim):
                            vector = None
                        vector = _unit(vector)
                    if vector is None:
                        skipped.append((member, "face %d: unusable landmarks or embedding" % index))
                    else:
                        entries.append(FacesetEntry(member, index, vector))

    result = FacesetEmbeddings(
        path=fsz_path, sha256=digest, format="legacy" if metadata is None else "v2",
        stored_model_id=stored_id, tagged=tagged, model_id=active_model_id, status=status,
        entries=tuple(entries), skipped=tuple(skipped))
    if cacheable:
        with _cache_lock:
            _cache[key] = result
            while len(_cache) > _CACHE_SIZE:
                _cache.popitem(last=False)
    return result
