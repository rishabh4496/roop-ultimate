"""Pose-indexed archive creation and additive, lazy legacy analysis.

The V2 writer owns selection and serialization. Legacy loads never rewrite an
archive or change its embedding averaging semantics; their analysis is cached
separately and invalidated by the archive and per-reference embedding digests.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from roop.faceset_v2 import _entry, write_faceset_v2


def build_faceset(path, faceset, images=None, **options):
    return write_faceset_v2(path, faceset, images or faceset.ref_images, **options)


def cache_missing_metadata(path, faceset, cache_dir=None):
    """Return one analysis row per *loaded* face, without filtering old banks.

    Detection has already happened in the source loader. Computing pose and
    Laplacian variance here adds no model session or network dependency.
    A read-only cache still permits the archive to load.
    """
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    for i, face in enumerate(faceset.faces):
        vector = (faceset.embeddings_backup if i == 0 and faceset.embeddings_backup is not None
                  else face.get('embedding') if isinstance(face, dict) else face.embedding)
        digest.update(np.asarray(vector, dtype=np.float32).tobytes())
    key = digest.hexdigest()
    root = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[1] / 'models' / 'runtime_profiles' / 'facesets'
    target = root / f'pose_v1_{key}.json'
    rows = None
    try:
        cached = json.loads(target.read_text(encoding='utf-8'))
        if (cached.get('key') == key and isinstance(cached.get('sources'), list)
                and len(cached['sources']) == len(faceset.faces)
                and all(isinstance(row, dict) and isinstance(row.get('geometry'), dict)
                        and 'yaw' in row['geometry'] and 'pitch' in row['geometry']
                        for row in cached['sources'])):
            rows = cached['sources']
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if rows is None:
        rows = [_entry(face, image, i) for i, (face, image) in
                enumerate(zip(faceset.faces, faceset.ref_images))]
        temporary = None
        try:
            root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=root,
                                             delete=False, suffix='.tmp') as handle:
                temporary = handle.name
                json.dump({'key': key, 'sources': rows}, handle, allow_nan=False)
            os.replace(temporary, target)
        except OSError:
            pass
        finally:
            if temporary and os.path.exists(temporary):
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
    faceset.legacy_pose_metadata = rows
    # Consumers that need canonical poses can use the cache while V1 retains
    # its own format version and original identity centroid behavior.
    faceset._canonical_poses = [
        (row['geometry']['yaw'], row['geometry']['pitch'])
        if row['geometry']['yaw'] is not None else None for row in rows]
    return rows
