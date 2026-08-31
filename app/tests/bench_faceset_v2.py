"""Measure FaceSet V2 creation and lookup overhead without model claims."""

import os
import sys
import tempfile
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.faceset_v2 import (prepare_faceset_v2, read_faceset_archive,
                             select_reference_index, write_faceset_v2)
from roop.FaceSet import FaceSet


def face(i, x):
    emb = np.zeros(512, dtype=np.float32)
    emb[i % 512] = 1.0
    return {
        "bbox": np.array([x, 80, x + 180, 260], dtype=np.float32),
        "landmark_2d_68": np.tile(np.array([[160.0, 160.0]], dtype=np.float32), (68, 1)),
        "pose": np.array([0.0, float((-1 if i % 2 else 1) * min(75, i * 5)), 0.0], dtype=np.float32),
        "embedding": emb,
        "det_score": 0.96,
        "landmark_confidence": 0.95,
    }


def image(seed):
    rng = np.random.default_rng(seed)
    img = rng.integers(20, 235, (320, 320, 3), dtype=np.uint8)
    cv2.rectangle(img, (70, 50), (250, 280), (145, 120, 105), -1)
    return img


def main():
    count = 24
    faces = [face(i, 70) for i in range(count)]
    images = [image(i) for i in range(count)]
    fs = FaceSet()
    fs.faces = faces
    t0 = time.perf_counter()
    metadata, selected = prepare_faceset_v2(faces, images, min_quality=0.0)
    prepare_s = time.perf_counter() - t0
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "bench.fsz")
        t1 = time.perf_counter()
        write_faceset_v2(path, fs, images, min_quality=0.0)
        write_s = time.perf_counter() - t1
        t2 = time.perf_counter()
        loaded = read_faceset_archive(path)
        read_s = time.perf_counter() - t2
        t3 = time.perf_counter()
        for i in range(10000):
            select_reference_index(loaded, pose=(float(i % 80), 0.0))
        lookup_s = time.perf_counter() - t3
        print(f"sources={len(selected)} archive_bytes={os.path.getsize(path)}")
        print(f"prepare_ms={prepare_s * 1000:.3f} write_ms={write_s * 1000:.3f} read_ms={read_s * 1000:.3f}")
        print(f"lookup_10k_per_sec={10000 / max(1e-9, lookup_s):.2f}")


if __name__ == "__main__":
    main()
