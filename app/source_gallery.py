"""Source face gallery: the in-memory list of loaded facesets, its
reordering/removal operations, and the payloads the UI reads it through.

A layer of its own because api.py's /api/source routes and the faceset
library routes both build on it; leaving it in api.py would have forced
routes_faceset to import back from api.py, which is a cycle.

Moved verbatim from api.py.
"""

import os
import shutil

import cv2
import numpy as np

import roop.globals as roop_globals
import ui.globals as ui_globals
from roop import utilities as util
from roop.FaceSet import FaceSet
from roop.face_util import extract_face_images
from roop.capturer import get_image_frame
from api_media import _rgb_to_dataurl
from roop.faceset_v2 import read_faceset_archive
from roop.face_reference import clustered_faceset


# ── Injected by api.py at import time ────────────────────────────────────
# Never rebound in api.py, only read or mutated in place, so binding the same
# object here keeps one shared value instead of two that drift apart.
API_TEMP = None


def _mask_offsets_from_cfg():
    c = roop_globals.CFG
    return [c.mask_top, c.mask_bottom, c.mask_left, c.mask_right,
            c.face_mask_blend, c.mouth_mask_blend,
            c.mouth_top_scale, c.mouth_bottom_scale,
            c.mouth_left_scale, c.mouth_right_scale]

def estimate_face_pose_from_kps(kps):
    try:
        left_eye_x, left_eye_y = kps[0]
        right_eye_x, right_eye_y = kps[1]
        nose_x, nose_y = kps[2]
        left_mouth_x, left_mouth_y = kps[3]
        right_mouth_x, right_mouth_y = kps[4]
        
        dx_left = abs(nose_x - left_eye_x)
        dx_right = abs(right_eye_x - nose_x)
        ratio = dx_left / (dx_right + 1e-6)
        
        yaw_label = "Front"
        if ratio < 0.65:
            yaw_label = "Left Profile"
        elif ratio > 1.55:
            yaw_label = "Right Profile"
            
        eye_y = (left_eye_y + right_eye_y) * 0.5
        mouth_y = (left_mouth_y + right_mouth_y) * 0.5
        dy_eyes = abs(nose_y - eye_y)
        dy_mouth = abs(mouth_y - nose_y)
        v_ratio = dy_eyes / (dy_mouth + 1e-6)
        
        pitch_label = ""
        if v_ratio < 0.65:
            pitch_label = "Up Tilt"
        elif v_ratio > 1.45:
            pitch_label = "Down Tilt"
            
        if pitch_label:
            if yaw_label == "Front":
                return pitch_label
            return f"{yaw_label} + {pitch_label}"
        return yaw_label
    except Exception:
        return "Front"

def _get_source_faces_info():
    source_faces_info = []
    for fs in roop_globals.INPUT_FACESETS:
        faces_poses = []
        for face in fs.faces:
            kps = getattr(face, 'kps', None)
            if kps is None and isinstance(face, dict) and 'kps' in face:
                kps = face['kps']
            poses_str = estimate_face_pose_from_kps(kps) if kps is not None else "Front"
            faces_poses.append(poses_str)
        source_faces_info.append({
            "count": len(fs.faces),
            "poses": faces_poses
        })
    return source_faces_info

def _sources_append(faceset, thumb_rgb):
    roop_globals.INPUT_FACESETS.append(faceset)
    ui_globals.ui_input_thumbs.append(thumb_rgb)

def _sources_pop(idx):
    """Remove one entry. Returns True when something was actually removed."""
    if not (0 <= idx < len(roop_globals.INPUT_FACESETS)):
        return False
    roop_globals.INPUT_FACESETS.pop(idx)
    if 0 <= idx < len(ui_globals.ui_input_thumbs):
        ui_globals.ui_input_thumbs.pop(idx)
    return True

def _sources_move(idx, new_idx):
    n = len(roop_globals.INPUT_FACESETS)
    if not (0 <= idx < n and 0 <= new_idx < n):
        return False
    for arr in (roop_globals.INPUT_FACESETS, ui_globals.ui_input_thumbs):
        if idx < len(arr) and new_idx < len(arr):
            arr.insert(new_idx, arr.pop(idx))
    return True

def _sources_clear():
    roop_globals.INPUT_FACESETS.clear()
    ui_globals.ui_input_thumbs.clear()

def _sources_desync():
    """Non-empty message when the two lists have fallen out of step.

    Checked on every gallery payload, so a divergence surfaces on the very next
    UI refresh instead of being discovered later as "that face didn't swap".
    """
    nf = len(roop_globals.INPUT_FACESETS)
    nt = len(ui_globals.ui_input_thumbs)
    if nf == nt:
        return ""
    msg = (f"source gallery out of step: {nf} faceset(s) but {nt} thumbnail(s) — "
           f"clear the input faces and re-add them")
    print(f"[SOURCES] BUG: {msg}", flush=True)
    return msg

def _source_faces_payload():
    payload = {
        "source_faces": [_rgb_to_dataurl(t) for t in ui_globals.ui_input_thumbs],
        "source_faces_info": _get_source_faces_info(),
        "faceset_count": len(roop_globals.INPUT_FACESETS),
    }
    desync = _sources_desync()
    if desync:
        payload["desync"] = desync
    return payload


def _largest_detected_face(detections):
    """Pick the intended subject from a reference image without extra inference."""
    def area(item):
        try:
            face = item[0]
            bbox = face.get("bbox") if isinstance(face, dict) else face.bbox
            box = np.asarray(bbox, dtype=np.float32).reshape(4)
            return max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
        except (AttributeError, TypeError, ValueError):
            return 0.0
    return max(detections or [], key=area, default=None)


def add_reference_folder(paths, min_cosine=0.65):
    """Add many images as one identity-bearing, pose-aware FaceSet.

    The existing analyser detects each reference once.  ArcFace validation and
    clustering do not create a second CUDA/ORT session, which preserves the
    small-GPU single-context safety policy.
    """
    candidates = []
    for path in paths or []:
        if not util.has_image_extension(path):
            continue
        frame = get_image_frame(path)
        if frame is None:
            continue
        item = _largest_detected_face(extract_face_images(path, (False, 0)))
        if item is not None:
            candidates.append((item[0], frame, path, item[1]))
    cluster = clustered_faceset([(face, frame, path)
                                 for face, frame, path, _ in candidates],
                                min_cosine=min_cosine)
    accepted_paths = {sample.path for sample in cluster.samples}
    accepted = [item for item in candidates if str(item[2]) in accepted_paths]
    if not accepted:
        raise ValueError("no reference image survived identity clustering")

    face_set = FaceSet()
    face_set.faces = [face for face, _, _, _ in accepted]
    face_set.ref_images = [frame for _, frame, _, _ in accepted]
    for face in face_set.faces:
        face.mask_offsets = _mask_offsets_from_cfg()
    face_set.face_poses = cluster.poses
    face_set.reference_embeddings = cluster.embeddings
    face_set.reference_weights = [sample.weight for sample in cluster.samples]
    face_set.reference_paths = [sample.path for sample in cluster.samples]
    face_set.reference_rejected = cluster.rejected
    face_set.identity_embedding = cluster.embedding
    face_set.normalized_embedding = cluster.embedding.copy()
    face_set._source_path = os.path.abspath(accepted[0][2])
    thumb_index = min(range(len(accepted)), key=lambda i: sum(
        value * value for value in cluster.poses[i]))
    _sources_append(face_set, util.convert_to_gradio(accepted[thumb_index][3]))
    return face_set

def _ingest_faceset(path):
    # Validate V2 before touching the shared extraction directory. A corrupt
    # metadata member or checksum must fail loudly instead of silently loading
    # a partially valid source bank. Legacy PNG-only archives return None and
    # follow the unchanged V1 path below.
    faceset_metadata = read_faceset_archive(path)
    unzipfolder = os.path.join(os.environ.get("TEMP", API_TEMP), "faceset")
    if os.path.isdir(unzipfolder):
        shutil.rmtree(unzipfolder, ignore_errors=True)
    os.makedirs(unzipfolder, exist_ok=True)
    util.unzip(path, unzipfolder)
    face_set = FaceSet()
    best_crop = None
    best_score = None
    def append_face(fd, frame):
        nonlocal best_score, best_crop
        face = fd[0]
        face.mask_offsets = _mask_offsets_from_cfg()
        face_set.faces.append(face)
        face_set.ref_images.append(frame)
        # Use the most frontal face as the gallery thumbnail, not just the first
        # one (which is often a profile in a multi-angle set).
        kps = getattr(face, "kps", None)
        if kps is None and isinstance(face, dict):
            kps = face.get("kps")
        score = _frontality(kps) if kps is not None else 999.0
        if best_score is None or score < best_score:
            best_score, best_crop = score, fd[1]

    if faceset_metadata is not None:
        # A V2 archive can contain the same full reference frame more than once
        # when a source image held multiple faces. Match each cached metadata
        # entry to one detected face by bbox so loading never duplicates all
        # detections for every metadata row or changes source-bank ordering.
        detected = {}
        used = {}
        members = ((faceset_metadata.get("index") or {}).get("reference_members")
                   or [entry.get("reference_member")
                       for entry in faceset_metadata.get("sources", [])])
        for member in members:
            filename = os.path.join(unzipfolder, member)
            detected[member] = (get_image_frame(filename),
                                extract_face_images(filename, (False, 0)))
            used[member] = set()
        for entry in faceset_metadata.get("sources", []):
            member = entry.get("reference_member")
            frame, candidates = detected.get(member, (None, []))
            if not candidates:
                continue
            target = ((entry.get("geometry") or {}).get("bbox"))
            target = np.asarray(target, dtype=np.float32).reshape(4) if target is not None else None
            best_i, best_value = None, -1.0
            for i, fd in enumerate(candidates):
                if i in used[member]:
                    continue
                box = getattr(fd[0], "bbox", None)
                if box is None and isinstance(fd[0], dict):
                    box = fd[0].get("bbox")
                if target is None or box is None:
                    value = 0.0 if best_i is not None else 1.0
                else:
                    box = np.asarray(box, dtype=np.float32).reshape(4)
                    ix0, iy0 = max(target[0], box[0]), max(target[1], box[1])
                    ix1, iy1 = min(target[2], box[2]), min(target[3], box[3])
                    inter = max(0.0, float(ix1 - ix0)) * max(0.0, float(iy1 - iy0))
                    area = max(1.0, float((target[2] - target[0]) * (target[3] - target[1])))
                    value = inter / area
                if value > best_value:
                    best_i, best_value = i, value
            if best_i is not None:
                used[member].add(best_i)
                append_face(candidates[best_i], frame)
    else:
        for file in sorted(os.listdir(unzipfolder)):
            if file.endswith(".png"):
                filename = os.path.join(unzipfolder, file)
                frame = get_image_frame(filename)
                for fd in extract_face_images(filename, (False, 0)):
                    append_face(fd, frame)
    if len(face_set.faces) > 0:
        # Keep the originating archive with the in-memory FaceSet so a durable
        # processing project can reload the exact source rather than only a
        # thumbnail or an embedding snapshot.
        face_set._source_path = os.path.abspath(path)
        if faceset_metadata is not None:
            face_set.attach_v2_metadata(faceset_metadata)
        elif len(face_set.faces) > 1:
            face_set.AverageEmbeddings()
        # A faceset with no usable crop still gets a (blank) gallery slot: the
        # two lists are positional, so skipping the thumbnail would shift every
        # later face's picture onto the wrong faceset.
        _sources_append(face_set,
                        util.convert_to_gradio(best_crop) if best_crop is not None else None)

def _frontality(kps):
    """0 = perfectly frontal, larger = more turned to a profile. Uses the same
    eye/nose geometry as estimate_face_pose_from_kps (ratio ~1 == front)."""
    try:
        import math
        le_x = kps[0][0]
        re_x = kps[1][0]
        nose_x = kps[2][0]
        dx_left = abs(nose_x - le_x)
        dx_right = abs(re_x - nose_x)
        ratio = dx_left / (dx_right + 1e-6)
        return abs(math.log(ratio + 1e-6))
    except Exception:
        return 999.0

def _shrink_for_thumb(img, max_side=256):
    """Downscale a thumbnail candidate so sidecars (and the base64 data URLs the
    library list embeds for every entry) stay small even when the source is a
    full-resolution frame."""
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    scale = max_side / float(s)
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)
