"""Canvas matte vs full-frame matte on REAL detected faces (production detector + landmarks), compared before any encode.

    app\\env\\Scripts\\python.exe app\\tests\\probe_matte_roi_real_faces.py [--per-clip 40]

Why this exists: an end-to-end A/B compares ENCODED files, and an encoder in VBR carries cumulative-bit state, so a few
+-1 pixels early in a clip move the quantiser for every later frame: two renders can differ by up to ~12 levels over 87%
of the picture (measured, d1.mp4, 2026-10-22) when the pastes differ by almost nothing. The paste is the thing under
test, so it is compared here, on real faces: frame -> production detector -> 512 aligned crop used as the "swap" ->
``paste_upscale`` with ``ROOP_BLEND_ROI_MATTE`` 0 and 1 -> max / count of differing pixels.
"""
import argparse
import os
import sys
import threading
import types

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures  # noqa: E402

CLIPS = ["Love.mp4", "Weeds.mp4", "Monica Bellucci .mp4", "double/d1.mp4", "double/d4.mp4", "double/d6.mp4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-clip", type=int, default=40)
    args = ap.parse_args()
    import angle_bench as ab
    from roop import face_util
    from roop.face_util import align_crop
    from roop.procmgr_masking import MaskingMixin
    ab.init_pipeline("cuda", "realswap", "None", "None")
    os.environ["ROOP_GRAIN_MATCH"] = "0"

    class Stub(MaskingMixin):
        def __init__(self):
            self.options = types.SimpleNamespace(show_face_area_overlay=False, blend_ratio=1.0)
            self._tls = threading.local()
            self.taken = 0

        def _matte_roi(self, *a, **k):
            out = super()._matte_roi(*a, **k)
            self.taken += out is not None
            return out

    rows = []
    for clip in CLIPS:
        if not fixtures.available(clip):
            print(f"[probe] {clip} not available, skipped")
            continue
        cap = cv2.VideoCapture(fixtures.clip(clip))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        got = 0
        for k in range(args.per_clip * 4):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (k + 0.5) / (args.per_clip * 4)))
            ok, fr = cap.read()
            if not ok or got >= args.per_clip:
                continue
            faces = [f for f in (face_util.get_all_faces(fr) or []) if getattr(f, "det_score", 0) >= 0.6]
            if len(faces) != 1 or getattr(faces[0], "landmark_2d_106", None) is None:
                continue
            f = faces[0]
            crop, M = align_crop(fr, f.kps, 512)
            outs = {}
            taken = 0
            for env in ("0", "1"):
                os.environ["ROOP_BLEND_ROI_MATTE"] = env
                stub = Stub()
                outs[env] = stub.paste_upscale(crop, crop, M, fr.copy(), 1, [0, 0, 0, 0, 25.0, 10.0],
                                               face_landmarks=f.landmark_2d_106, face_kps=f.kps, inplace=True,
                                               track_id=1, frame_index=k)
                taken = stub.taken if env == "1" else taken
            d = np.abs(outs["0"].astype(np.int16) - outs["1"].astype(np.int16))
            rows.append((clip, fr.shape[1], int(d.max()), int((d > 0).any(axis=2).sum()), taken))
            got += 1
        print(f"[probe] {clip}: {got} faces", flush=True)
    a = np.array([(r[2], r[3], r[4]) for r in rows])
    print(f"\nfaces {len(a)}  canvas path taken {int(a[:, 2].sum())}/{len(a)}")
    print(f"max level difference: {int(a[:, 0].max())}   differing pixels per face: median {int(np.median(a[:, 1]))}  "
          f"p95 {int(np.percentile(a[:, 1], 95))}  max {int(a[:, 1].max())}   faces with any difference: {int((a[:, 1] > 0).sum())}")
    for clip in dict.fromkeys(r[0] for r in rows):
        sub = [r for r in rows if r[0] == clip]
        print(f"  {clip:24s} {sub[0][1]:5d}px wide  faces {len(sub):3d}  max diff {max(r[2] for r in sub)}  "
              f"max pixels {max(r[3] for r in sub)}")


if __name__ == "__main__":
    main()
