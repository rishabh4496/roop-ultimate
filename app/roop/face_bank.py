"""UI-facing Face Bank: video scanning, identity clustering, and character management.

Scans target videos across frames/scenes, extracts 512-dimensional normalized embeddings
(ArcFace/AdaFace), clusters all detected faces into distinct physical people using
DBSCAN or Agglomerative Clustering, and prepares thumbnail galleries for multi-target mapping.
"""

import base64
import os
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

import roop.globals
from roop.degrade import swallowed as _swallowed
from roop.face_clustering import (
    cluster_face_embeddings,
    compute_cosine_distance,
    extract_face_embedding,
    normalize_embedding,
)
from roop.face_util import (
    _attach_source_crops,
    clamp_cut_values,
    get_all_faces,
    offaxis_deg,
    solve_pose_5pt,
)
from roop.scene_detector import ContentAwareSceneDetector


def crop_to_dataurl(crop: np.ndarray, quality: int = 85) -> str:
    """Encode an image crop as a base64 JPEG data URL."""
    if crop is None or crop.size == 0:
        return ""
    try:
        success, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            return ""
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except (cv2.error, ValueError, TypeError):
        return ""


def extract_face_crop(img: np.ndarray, face: Any, pad_pct: float = 0.15) -> np.ndarray:
    """Extract padded square face crop from frame for thumbnail display."""
    if img is None or face is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    try:
        bbox = getattr(face, 'bbox', None)
        if bbox is None and isinstance(face, dict):
            bbox = face.get('bbox')
        if bbox is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)

        x0, y0, x1, y1 = [int(v) for v in bbox]
        h, w = img.shape[:2]
        bw, bh = x1 - x0, y1 - y0
        pad_x = int(bw * pad_pct)
        pad_y = int(bh * pad_pct)

        sx = max(0, x0 - pad_x)
        sy = max(0, y0 - pad_y)
        ex = min(w, x1 + pad_x)
        ey = min(h, y1 + pad_y)

        crop = img[sy:ey, sx:ex]
        if crop.size == 0:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return crop
    except (ValueError, TypeError, KeyError, AttributeError):
        return np.zeros((64, 64, 3), dtype=np.uint8)


class FaceBank:
    """Persistent Face Bank manager for unique identities in target clips."""

    def __init__(self):
        self.characters: List[Dict[str, Any]] = []
        self.last_video_path: Optional[str] = None
        self.scene_cuts: Set[int] = set()

    def clear(self):
        self.characters.clear()
        self.last_video_path = None
        self.scene_cuts.clear()

    def scan_video(
        self,
        video_path: str,
        max_samples: int = 150,
        time_budget: float = 60.0,
        clustering_method: str = "dbscan",
        eps: float = 0.45,
        distance_threshold: float = 0.45,
        min_det_score: float = 0.55,
        min_face_size: int = 50,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Scan video, detect faces, cluster into unique characters, and build Face Bank.

        Returns list of character dictionaries with distinct thumbnails and metadata.
        """
        if not video_path or not os.path.isfile(video_path):
            return []

        t0 = time.time()
        self.clear()
        self.last_video_path = video_path

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

        # 1. Detect scene cuts and select sample frames
        scene_detector = ContentAwareSceneDetector()
        cuts: Set[int] = set()

        # Build candidate frame indices: scene cuts + regular intervals
        sample_indices: List[int] = []
        if total_frames <= max_samples:
            sample_indices = list(range(total_frames))
        else:
            step = max(1, int(total_frames / max_samples))
            sample_indices = list(range(0, total_frames, step))

        collected_faces: List[Dict[str, Any]] = []

        # Read and detect on selected frames
        for s_idx, f_num in enumerate(sample_indices):
            if (time.time() - t0) > time_budget:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Scene cut tracking
            if scene_detector.observe_frame(frame, f_num):
                cuts.add(f_num)

            detected = get_all_faces(frame) or []
            for face in detected:
                det_score = float(getattr(face, 'det_score', 0.0) or 0.0)
                if det_score < min_det_score:
                    continue

                bbox = getattr(face, 'bbox', None)
                if bbox is None:
                    continue
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                if min(bw, bh) < min_face_size:
                    continue

                emb = extract_face_embedding(face, frame)
                if emb is None:
                    continue

                # Compute pose off-axis angle
                kps = getattr(face, 'kps', None)
                pitch, yaw, roll = solve_pose_5pt(kps) if kps is not None else (0.0, 0.0, 0.0)
                off_axis = offaxis_deg(pitch, yaw)

                crop = extract_face_crop(frame, face)
                _attach_source_crops(face, frame)

                collected_faces.append({
                    'face': face,
                    'embedding': emb,
                    'frame_idx': f_num,
                    'det_score': det_score,
                    'off_axis': off_axis,
                    'crop': crop,
                    'yaw': yaw,
                    'pitch': pitch,
                })

            if on_progress and (s_idx % 10 == 0 or s_idx == len(sample_indices) - 1):
                pct = int((s_idx + 1) * 100 / max(1, len(sample_indices)))
                on_progress(s_idx + 1, len(sample_indices), f"Scanning faces: {pct}%")

        cap.release()
        self.scene_cuts = cuts

        if not collected_faces:
            return []

        # 2. Cluster faces by identity
        embeddings = [cf['embedding'] for cf in collected_faces]
        cluster_labels = cluster_face_embeddings(
            embeddings,
            method=clustering_method,
            eps=eps,
            distance_threshold=distance_threshold,
        )

        # 3. Aggregate clusters into characters
        unique_labels = sorted(set(cluster_labels))
        characters = []

        for c_id in unique_labels:
            indices = [i for i, lbl in enumerate(cluster_labels) if lbl == c_id]
            members = [collected_faces[i] for i in indices]

            # Compute cluster mean embedding
            embs = np.asarray([m['embedding'] for m in members], dtype=np.float32)
            mean_raw = np.mean(embs, axis=0)
            mean_norm = normalize_embedding(mean_raw)

            # Sort by quality: low off-axis + high detection score
            members.sort(key=lambda m: (m['off_axis'], -m['det_score']))
            best_member = members[0]

            # Collect pose angles: front, left profile, right profile
            angles = [best_member]
            # Find left profile (yaw > 25)
            left_profiles = [m for m in members if m['yaw'] > 25]
            if left_profiles:
                left_profiles.sort(key=lambda m: -m['det_score'])
                angles.append(left_profiles[0])
            # Find right profile (yaw < -25)
            right_profiles = [m for m in members if m['yaw'] < -25]
            if right_profiles:
                right_profiles.sort(key=lambda m: -m['det_score'])
                angles.append(right_profiles[0])

            thumb_url = crop_to_dataurl(best_member['crop'])

            char_record = {
                'person_id': f"tp_{c_id}",
                'display_rank': int(c_id),
                'name': f"Actor {c_id + 1}",
                'thumbnail': thumb_url,
                'cluster_size': len(members),
                'best_frame': int(best_member['frame_idx']),
                'off_axis': float(best_member['off_axis']),
                'det_score': float(best_member['det_score']),
                'mean_embedding': mean_norm.tolist() if mean_norm is not None else [],
                'target_faces': [a['face'] for a in angles],
                'angles_count': len(angles),
            }
            characters.append(char_record)

        # Sort characters by cluster size descending (main characters first)
        characters.sort(key=lambda c: -c['cluster_size'])
        # Re-assign display ranks
        for rank, ch in enumerate(characters):
            ch['display_rank'] = rank
            ch['name'] = f"Actor {rank + 1}"

        self.characters = characters
        return characters

    def apply_to_globals(self, characters: Optional[List[Dict[str, Any]]] = None) -> int:
        """Populate roop.globals with target faces and person IDs from Face Bank."""
        chars = characters if characters is not None else self.characters
        if not chars:
            return 0

        roop.globals.TARGET_FACES.clear()
        roop.globals.TARGET_FACE_GROUP.clear()
        roop.globals.TARGET_FACE_PERSON_IDS.clear()
        roop.globals.TARGET_REFERENCE_FACE_IDS.clear()
        if getattr(roop.globals, 'TARGET_FACE_NAMES', None) is None:
            roop.globals.TARGET_FACE_NAMES = {}
        else:
            roop.globals.TARGET_FACE_NAMES.clear()

        # Import UI globals safely
        try:
            from roop import ui_globals
            ui_globals.ui_target_thumbs.clear()
        except (ImportError, AttributeError):
            ui_globals = None

        count = 0
        from roop.api_state import new_target_reference_face_id

        for rank, ch in enumerate(chars):
            person_id = ch['person_id']
            roop.globals.TARGET_FACE_NAMES[rank] = ch['name']
            for face in ch.get('target_faces', []):
                roop.globals.TARGET_FACES.append(face)
                roop.globals.TARGET_FACE_GROUP.append(rank)
                roop.globals.TARGET_FACE_PERSON_IDS.append(person_id)
                roop.globals.TARGET_REFERENCE_FACE_IDS.append(new_target_reference_face_id())
                crop = extract_face_crop(None, face)
                if ui_globals is not None:
                    try:
                        from roop import util
                        ui_globals.ui_target_thumbs.append(util.convert_to_gradio(crop))
                    except Exception as _exc:
                        _swallowed("face_bank.py:thumb_convert", _exc, "fallback continued")
                count += 1

        return count


# Singleton instance
_GLOBAL_FACE_BANK = FaceBank()


def get_face_bank() -> FaceBank:
    """Return singleton FaceBank instance."""
    return _GLOBAL_FACE_BANK
