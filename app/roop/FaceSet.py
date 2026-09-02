import numpy as np

from roop.faceset_v2 import (FORMAT_NAME, FORMAT_VERSION, measure_lighting,
                              parse_pose_matrix_key, pose_matrix_cell,
                              select_reference_index)

class FaceSet:
    faces = []
    ref_images = []
    embedding_average = 'None'
    embeddings_backup = None

    def __init__(self):
        self.faces = []
        self.ref_images = []
        self.embeddings_backup = None
        self.face_3d = None   # populated by face_3d_recon when use_3d_recon is enabled (first valid face's crop)
        # 3D recon per-face crop bank: list parallel to self.faces, each entry a
        # {'src_crop','src_M','src_lm68'} dict or None. Lets 3D recon warp the
        # source-bank-SELECTED face (not just face[0]) so the two features compose.
        self.face_3d_bank = None  # type: list[dict | None] | None
        # Multi-angle source bank: list of (yaw_deg, pitch_deg) or None per face in self.faces
        # Populated by ProcessMgr.initialize() when use_source_bank is enabled.
        self.face_poses = None  # type: list[tuple[float, float] | None] | None
        # V2 is an additive metadata/index layer.  These fields are optional so
        # old in-memory FaceSets and old .fsz archives retain their exact shape.
        self.format_name = FORMAT_NAME
        self.format_version = 1
        self.faceset_metadata = None
        self.face_metadata = []
        self.pose_bank = None
        # Spec'd V2 surface, populated by `attach_v2_metadata`. `pose_bins` is
        # keyed by the `(yaw_bin, pitch_bin)` TUPLE in memory even though it is
        # stored under a flat string key on disk, because JSON has no tuple key.
        self.pose_bins = {}
        self.dermal_patch = None
        self.identity_embedding = None
        self.normalized_embedding = None
        self.faceset_valid = True
        self.faceset_migration = None

    def AverageEmbeddings(self):
        # V2 deliberately keeps each pose-specific face embedding intact.  The
        # global identity vector is stored separately in metadata and exposed as
        # `identity_embedding`; callers that need legacy behaviour still get the
        # original averaging path for V1 FaceSets.
        if self.format_version >= 2:
            return
        if len(self.faces) > 1 and self.embeddings_backup is None:
            first_face = self.faces[0]
            if hasattr(first_face, 'embedding'):
                self.embeddings_backup = first_face.embedding
                embeddings = [face.embedding for face in self.faces]
                first_face.embedding = np.mean(embeddings, axis=0)
            else:
                self.embeddings_backup = first_face['embedding']
                embeddings = [face['embedding'] for face in self.faces]
                first_face['embedding'] = np.mean(embeddings, axis=0)

    def attach_v2_metadata(self, metadata):
        """Attach validated V2 metadata without replacing detector Face objects."""
        self.faceset_metadata = metadata
        self.format_name = metadata.get('schema', FORMAT_NAME)
        self.format_version = int(metadata.get('version', FORMAT_VERSION))
        self.face_metadata = list(metadata.get('sources') or [])
        self.pose_bank = metadata.get('pose_bank') or {}
        self.dermal_patch = metadata.get('dermal_patch') or None
        self.pose_bins = {}
        for key, cell in (metadata.get('pose_bins') or {}).items():
            parsed = parse_pose_matrix_key(key)
            if parsed is None or not isinstance(cell, dict):
                continue
            vector = self._unit_vector(cell.get('embedding'))
            if vector is not None:
                self.pose_bins[parsed] = vector
        identity = metadata.get('identity') or {}
        # `default_embedding` is the spec'd top-level name; the nested identity
        # block carries the same vector and remains the fallback for archives
        # written before that key existed.
        value = (metadata.get('default_embedding')
                 or identity.get('embedding')
                 or identity.get('normalized_embedding'))
        if value is not None:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(arr))
            if norm > 1e-8 and np.isfinite(arr).all():
                self.identity_embedding = (arr / norm).astype(np.float32)
                self.normalized_embedding = self.identity_embedding.copy()
        poses = []
        for entry in self.face_metadata:
            geo = entry.get('geometry') or {}
            poses.append((geo.get('yaw'), geo.get('pitch')))
        self.face_poses = poses if poses else None
        for index, face in enumerate(self.faces):
            try:
                face['faceset_v2_index'] = index
            except Exception:
                try:
                    setattr(face, 'faceset_v2_index', index)
                except Exception:
                    pass

    @staticmethod
    def _unit_vector(value):
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        if arr.size == 0 or not np.isfinite(arr).all():
            return None
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-8:
            return None
        return (arr / norm).astype(np.float32)

    @property
    def default_embedding(self):
        """The global normalized centroid, for both V2 and legacy FaceSets.

        V2 reads the stored centroid. V1 has none, so it is derived from the
        faces in memory -- and deliberately from `embeddings_backup` when
        `AverageEmbeddings` has already overwritten `faces[0].embedding` with
        the legacy mean, so this returns the same vector whether or not that
        in-place mutation has happened yet.
        """
        if self.identity_embedding is not None:
            return self.identity_embedding
        vectors = []
        for index, face in enumerate(self.faces or []):
            if index == 0 and self.embeddings_backup is not None:
                value = self.embeddings_backup
            elif isinstance(face, dict):
                value = face.get('embedding')
            else:
                value = getattr(face, 'embedding', None)
            vector = self._unit_vector(value)
            if vector is not None:
                vectors.append(vector)
        if not vectors or len({v.shape for v in vectors}) != 1:
            return None
        return self._unit_vector(np.mean(np.asarray(vectors), axis=0))

    def pose_bin_embedding(self, pose=None, fallback=True):
        """Return the 3x3 pose-cell centroid for `pose`.

        Falls back along a widening path -- exact cell, then same yaw column,
        then `default_embedding` -- so a V1 FaceSet and a V2 FaceSet with an
        empty cell both answer with a usable vector instead of ``None``.
        """
        cell = pose_matrix_cell(pose) if pose is not None else ("center", "center")
        vector = self.pose_bins.get(cell)
        if vector is not None:
            return vector
        if not fallback:
            return None
        for pitch_bin in ("center", "up", "down"):
            vector = self.pose_bins.get((cell[0], pitch_bin))
            if vector is not None:
                return vector
        return self.default_embedding

    def select_reference_index(self, pose=None, appearance=None, embedding=None):
        """Fast V2 lookup with legacy pose-bank fallback."""
        if self.format_version >= 2 and self.faceset_metadata:
            index = select_reference_index(self.faceset_metadata, pose=pose,
                                           appearance=appearance, embedding=embedding)
            return max(0, min(int(index), max(0, len(self.faces) - 1)))
        if pose is not None and self.face_poses:
            yaw, pitch = float(pose[0]), float(pose[1])
            valid = [(i, (yaw - float(y or 0.0)) ** 2 + (pitch - float(p or 0.0)) ** 2)
                     for i, (y, p) in enumerate(self.face_poses) if y is not None]
            if valid:
                return min(valid, key=lambda item: item[1])[0]
        return 0

    def select_pose_aware_reference(self, target_pose, appearance=None,
                                    expression=None, previous_index=None):
        """Return a Phase 5 V2 source selection, or ``None`` for legacy sets.

        The old ``select_reference_index`` remains untouched for V1 archives
        and callers that depend on its exact yaw/pitch behaviour.  Phase 5
        callers opt into the richer result only when V2 metadata is present.
        """
        if self.format_version < 2 or not self.faceset_metadata:
            return None
        from roop.pose_source_selector import select_pose_aware_source
        return select_pose_aware_source(
            self.faceset_metadata, target_pose, appearance=appearance,
            expression=expression, previous_index=previous_index)

    def identity_detail_for(self, source_index=0):
        """Return the persistent V2 detail map, or a safe per-source fallback."""
        if self.format_version < 2 or not self.faceset_metadata:
            return None
        details = self.faceset_metadata.get('identity_details') or {}
        persistent = details.get('high_frequency')
        if isinstance(persistent, dict) and persistent.get('residual_q'):
            return persistent
        try:
            index = int(source_index)
        except (TypeError, ValueError):
            index = 0
        if 0 <= index < len(self.face_metadata):
            return ((self.face_metadata[index].get('identity_details') or {})
                    .get('high_frequency'))
        return None

    @staticmethod
    def lighting_for_frame(image, bbox=None):
        return measure_lighting(image, bbox)
