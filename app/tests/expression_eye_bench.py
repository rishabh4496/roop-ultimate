"""Expression / gaze / blink bench — does each control move the OUTPUT toward
the target, and what does it cost?

Not a unit test: needs the GPU, the models and a real swap. Every arm renders
through `roop.core.live_swap`, the same per-frame path the preview uses, with
the user's config (sync_config=True) and only the three controls varied.

Graded per frame on the ORIGINAL target and the OUTPUT, both cropped with the
target frame's own alignment so the two share one geometry, with LivePortrait's
203-point landmarker (the one blink sync itself uses — so the lid column is
the quantity the control targets; the mouth, pupil and identity columns are
independent checks):

  lid_err    |eye opening(out) - eye opening(target)|, both eyes, as
             LivePortrait's calc_eye_close_ratio. Blink sync should cut it.
  blink_hit  of the frames where the TARGET's eyes are closed (ratio < 0.15),
             the share where the output's are too.
  mouth_err  |lip opening(out) - lip opening(target)| (calc_lip_close_ratio,
             points 90-102 / 48-66). Expression strength should cut it.
  pupil_err  dark-iris centroid position along the corner-to-corner axis
             (0..1), |out - target|, open eyes only. A PROXY for gaze: no gaze
             model exists in the repo; this sees horizontal eye direction only.
  id_src     cosine of the output face to the source identity. Must not fall
             materially: a control that "tracks" by un-swapping would win the
             other columns and lose here.
  swapped    frames the swap changed at all; an arm that swapped nothing would
             read perfect on every column above.

    app/env/Scripts/python.exe tests/expression_eye_bench.py --faceset akansha
"""
import argparse
import csv
import glob
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
sys.path.insert(0, APP)
sys.path.insert(0, HERE)

import fixtures  # noqa: E402

ARMS = {
    # name: (strength, gaze, blink)
    "off":          (0.0, None, False),
    "legacy_all_1": (1.0, None, False),
    "expr_1":       (1.0, 0.0, False),
    "gaze_1":       (0.0, 1.0, False),
    "blink":        (0.0, 0.0, True),
    "gaze_1_blink": (0.0, 1.0, True),
    "all":          (1.0, 1.0, True),
    "gaze_05_blink": (0.0, 0.5, True),
    "legacy_blink": (1.0, None, True),
}
CLOSED = 0.15


def _ratio(p, a, b, c, d):
    return float(np.linalg.norm(p[a] - p[b]) / (np.linalg.norm(p[c] - p[d]) + 1e-6))


class Grader:
    def __init__(self):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        self.lm = ort.InferenceSession(
            os.path.join(APP, "models", "liveportrait", "landmark.onnx"),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        # root=APP: FaceAnalysis appends /models itself.
        self.fa = FaceAnalysis(name="buffalo_l", root=APP,
                               allowed_modules=["detection", "recognition"],
                               providers=["CUDAExecutionProvider"])
        self.fa.prepare(ctx_id=0, det_size=(640, 640))

    def target_face(self, frame):
        faces = self.fa.get(frame)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def crop(self, frame, kps):
        from roop.face_util import estimate_norm
        M = estimate_norm(kps, 224, "arcface")
        return cv2.warpAffine(frame, M, (224, 224), borderMode=cv2.BORDER_REPLICATE)

    def points(self, crop):
        x = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return self.lm.run(None, {"input": x})[2].reshape(-1, 2) * 224

    @staticmethod
    def eyes(p):
        return _ratio(p, 6, 18, 0, 12), _ratio(p, 30, 42, 24, 36)

    @staticmethod
    def mouth(p):
        return _ratio(p, 90, 102, 48, 66)

    @staticmethod
    def pupil(crop, p, lo, corner_a, corner_b):
        """Dark-iris centroid inside the eye contour lo..lo+23, projected on the
        corner axis -> 0..1. None when the eye is closed or the patch is empty."""
        poly = p[lo:lo + 24].astype(np.int32)
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        vals = gray[mask > 0]
        if vals.size < 12:
            return None
        thr = np.percentile(vals, 30)
        w = np.maximum(0.0, thr - gray) * (mask > 0)
        tot = float(w.sum())
        if tot < 1e-3:
            return None
        ys, xs = np.indices(w.shape)
        c = np.array([(xs * w).sum() / tot, (ys * w).sum() / tot], np.float32)
        a, b = p[corner_a], p[corner_b]
        ax = b - a
        return float(np.dot(c - a, ax) / (np.dot(ax, ax) + 1e-6))

    def grade(self, crop):
        p = self.points(crop)
        el, er = self.eyes(p)
        pl = self.pupil(crop, p, 0, 0, 12) if el >= CLOSED else None
        pr = self.pupil(crop, p, 24, 24, 36) if er >= CLOSED else None
        return dict(el=el, er=er, mouth=self.mouth(p), pl=pl, pr=pr)

    def identity(self, frame, src_embed):
        f = self.target_face(frame)
        if f is None or src_embed is None:
            return None
        a = np.asarray(f.normed_embedding, np.float32)
        b = np.asarray(src_embed, np.float32)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def read_frames(path, limit):
    cap = cv2.VideoCapture(path)
    out = []
    while len(out) < limit:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faceset", default="akansha")
    ap.add_argument("--clips", default=None,
                    help="glob; default <clip root>/expression/*.mp4 via fixtures")
    ap.add_argument("--frames", type=int, default=600, help="per clip")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default=os.path.join(APP, "output", "bench_expression_eye"))
    args = ap.parse_args()

    from angle_bench import init_pipeline, build_options, load_faceset
    import yaml
    with open(os.path.join(APP, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    g = init_pipeline(cfg.get("provider", "cuda"), cfg.get("swap_model", "hyperswap"),
                      cfg.get("selected_enhancer", "None"), cfg.get("mask_engine", "None"),
                      sync_config=True)
    # The UI name ("DFL XSeg") is not the plugin key ("mask_xseg"); api.py
    # translates it for the real run, and so does two_face_video.
    from two_face_video import map_mask_engine
    mask_key = map_mask_engine(cfg.get("mask_engine", "None")) or "None"
    options = build_options(g, cfg.get("swap_model", "hyperswap"), mask_key)
    from roop.core import live_swap

    src_fs = load_faceset(os.path.join(APP, "facesets", args.faceset + ".fsz"))
    src_embed = getattr(src_fs, "embedding", None)
    if src_embed is None:
        src_embed = np.mean([f.normed_embedding for f in src_fs.faces], axis=0)
    grader = Grader()

    pattern = args.clips or os.path.join(
        fixtures.clip_dir("expression", required=True), "*.mp4")
    clips = sorted(glob.glob(pattern))
    if not clips:
        raise SystemExit(f"no clips match {pattern}")
    material = []
    for c in clips:
        frames = read_frames(c, args.frames)
        for i, fr in enumerate(frames):
            tf = grader.target_face(fr)
            if tf is not None:
                tcrop = grader.crop(fr, tf.kps)
                material.append((os.path.basename(c)[-6:-4], i, fr, tf.kps, grader.grade(tcrop)))
    print(f"[expr_eye] {len(material)} graded target frames from {len(clips)} clips; "
          f"model={cfg.get('swap_model')} enh={cfg.get('selected_enhancer')} "
          f"mask={cfg.get('mask_engine')} provider={cfg.get('provider')}", flush=True)
    closed = sum(1 for m in material if min(m[4]["el"], m[4]["er"]) < CLOSED)
    print(f"[expr_eye] target frames with closed eyes: {closed}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    summary = []
    last_report = time.time()
    for arm in args.arms.split(","):
        strength, gaze, blink = ARMS[arm]
        g.expression_restore_strength = strength
        g.expression_restore_region = "all"
        g.expression_gaze_follow = gaze
        g.expression_blink_sync = blink
        rows = []
        t0 = time.time()
        for n, (clip, i, fr, kps, tg) in enumerate(material):
            out = live_swap(fr.copy(), options, input_facesets=[src_fs])
            if out is None:
                out = fr
            ocrop = grader.crop(out, kps)
            og = grader.grade(ocrop)
            changed = float(np.abs(out.astype(np.int16) - fr.astype(np.int16)).mean()) > 0.5
            pe = [abs(og[k] - tg[k]) for k in ("pl", "pr")
                  if og[k] is not None and tg[k] is not None]
            rows.append(dict(
                clip=clip, frame=i, swapped=int(changed),
                t_el=tg["el"], t_er=tg["er"], o_el=og["el"], o_er=og["er"],
                lid_err=(abs(og["el"] - tg["el"]) + abs(og["er"] - tg["er"])) / 2,
                t_closed=int(min(tg["el"], tg["er"]) < CLOSED),
                o_closed=int(min(og["el"], og["er"]) < CLOSED),
                mouth_err=abs(og["mouth"] - tg["mouth"]),
                pupil_err=(float(np.mean(pe)) if pe else ""),
                id_src=grader.identity(out, src_embed)))
            if time.time() - last_report > 180:
                fps = (n + 1) / (time.time() - t0)
                print(f"  [{arm}] {n + 1}/{len(material)} frames, {fps:.2f} fps", flush=True)
                last_report = time.time()
        dt = time.time() - t0
        with open(os.path.join(args.out, f"{arm}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        closed_rows = [r for r in rows if r["t_closed"]]
        pup = [r["pupil_err"] for r in rows if r["pupil_err"] != ""]
        ids = [r["id_src"] for r in rows if r["id_src"] is not None]
        s = dict(arm=arm, strength=strength, gaze=gaze, blink=blink,
                 frames=len(rows), swapped=sum(r["swapped"] for r in rows),
                 lid_err=float(np.mean([r["lid_err"] for r in rows])),
                 blink_hit=(sum(r["o_closed"] for r in closed_rows) / len(closed_rows)
                            if closed_rows else float("nan")),
                 false_blink=(sum(r["o_closed"] for r in rows if not r["t_closed"])
                              / max(1, len(rows) - len(closed_rows))),
                 mouth_err=float(np.mean([r["mouth_err"] for r in rows])),
                 pupil_err=float(np.mean(pup)) if pup else float("nan"),
                 id_src=float(np.mean(ids)) if ids else float("nan"),
                 fps=len(rows) / dt)
        summary.append(s)
        print(f"[expr_eye] {arm:<13} swapped {s['swapped']}/{s['frames']}  "
              f"lid {s['lid_err']:.4f}  blink_hit {s['blink_hit']:.2f}  "
              f"false_blink {s['false_blink']:.3f}  mouth {s['mouth_err']:.4f}  "
              f"pupil {s['pupil_err']:.4f}  id {s['id_src']:.3f}  {s['fps']:.2f} fps",
              flush=True)
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    main()
