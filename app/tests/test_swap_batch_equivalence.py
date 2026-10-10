"""Is the batched hyperswap the same picture as the B=1 one?  (GPU-gated; `-m gpu`)

    env/Scripts/python.exe -m pytest tests/test_swap_batch_equivalence.py -s
    env/Scripts/python.exe tests/test_swap_batch_equivalence.py --out docs-or-any.json     # the table, no pytest

WHY. `RunBatchMulti` coalesces crops of DIFFERENT faces/sources into one inference. ORT's CUDA and TensorRT
InstanceNormalization were measured wrong at batch > 1 on HyperSwap's generator (face_engine/utils/onnx_batch.py,
2026-09-28: row 0 differed from its B=1 result by 1.73 at B=2; rows bled into each other by up to 136 levels), and the
app's own `_relax_batch_dim` only rewrites Reshape constants - it does not decompose the norm. Nothing in the suite ran
a batched row against its B=1 row, so the claim "RunBatchMulti is numerically identical to Run" was untested.

WHAT. 16 REAL aligned crops (production alignment: `canonicalize_face_alignment` at the swapper's template, `to_blob`
with the swapper's mean/std) from several clips, each paired with its OWN source identity (a different library faceset
per crop). Each (crop, source) is run
  * at B=1 through the production swapper session (`Run`),
  * batched through `RunBatchMulti` at B=2/4/8 (groups of consecutive crops, so every batch mixes identities),
  * at B=1 on two CUDA sessions of the same (batch-relaxed) graph - a GENUINE FP32 one (`fp32_graph`) and the shipped graph's
    own dtypes (the file is FP16 end to end, so that is FP16 compute) - and at B=1 again (the repeatability control).
Per row against its B=1 result: max abs diff (model units, [-1, 1]), SSIM (8-bit picture, data_range 255, the canary's
own implementation) and identity: the AdaFace cosine of the output to its source (AdaFace is not what the pipeline
matches with; an independent judge), reported as the batched-minus-B=1 delta.

PASS (per row, every row): SSIM >= 0.998 and |identity delta| <= 0.005.

The warm-up uses real crops, not zeros: a constant tensor has zero variance, which is exactly the case a norm layer's
epsilon decides, and it builds the batched TensorRT shapes on data that exercises them. The primary (hyperswap) net is
isolated by clearing `secondary` for the test, so the compared quantity is the hyperswap session and not the
hififace composite on top of it.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PASS_SSIM = 0.998
PASS_ID_DELTA = 0.005
BATCHES = (2, 4, 8)
N_CROPS = 16
# clip, frames to sample (fractions of the clip). Several people, poses and resolutions; every crop is a real detection.
#
# More candidates than rows (21 for 16): a sampled frame can hold no detection, and the 16 are then picked evenly across
# whatever was found, so the set still spans every clip.
CLIP_PLAN = (("double/d4.mp4", 6), ("double/d1.mp4", 4), ("double/d6.mp4", 3), ("Love.mp4", 4), ("single/s7.mp4", 4))


class Unavailable(Exception):
    """The machine cannot run this measurement (no CUDA, model, clip or faceset): a skip, never a pass."""


def _need_stack():
    try:
        import torch
        if not torch.cuda.is_available():
            raise Unavailable("no CUDA device")
        import onnxruntime
        if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
            raise Unavailable("onnxruntime has no CUDA provider")
    except ImportError as exc:
        raise Unavailable("ML stack missing: %s" % exc)
    if not os.path.exists(os.path.join(APP, "config.yaml")):
        raise Unavailable("no config.yaml")


def _build_swapper():
    """The production swapper, built the way the app builds it (live config, sync_config)."""
    from settings import Settings
    import angle_bench as ab
    cfg = Settings(os.path.join(APP, "config.yaml"))
    ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), "None", "None", sync_config=True)
    from roop.processors import FaceSwapInsightFace as fsi
    if not fsi._BATCH_SWAP:
        raise Unavailable("ROOP_BATCH_SWAP is off on this machine: RunBatchMulti would fall back to B=1 and "
                          "the comparison would be B=1 against itself")
    p = fsi.FaceSwapInsightFace()
    p.Initialize({"devicename": "cuda", "swap_model": str(cfg.swap_model)})
    if p.model_swap_insightface is None:
        raise Unavailable("swapper did not load")
    p.secondary = None            # isolate the hyperswap session from the hififace composite (see module docstring)
    return p, cfg


def _library_sources(n):
    import two_face_video as tfv
    names = sorted(f[:-4] for f in os.listdir(tfv.LIB) if f.endswith(".fsz"))
    out = []
    for name in names:
        try:
            fs = tfv.load_library_faceset(name)
        except (Exception, SystemExit):     # SystemExit from a missing/odd faceset: skip it, take another
            continue
        if fs.faces:
            out.append((name, fs))
        if len(out) >= n:
            break
    if len(out) < n // 2:
        raise Unavailable("only %d usable library facesets" % len(out))
    return out


def _real_crops(p, n):
    """n production-aligned crops from real detections; returns [(label, aligned_uint8_bgr)]."""
    import cv2
    import fixtures
    from roop.face_analyser import canonicalize_face_alignment
    from roop.face_util import get_all_faces
    crops = []
    for rel, k in CLIP_PLAN:
        path = fixtures.clip(rel)
        if not os.path.exists(path):
            continue
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frac in np.linspace(0.08, 0.92, k):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
            ok, frame = cap.read()
            if not ok:
                continue
            faces = list(get_all_faces(frame) or [])
            if not faces:
                continue
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            aligned, _m, _a = canonicalize_face_alignment(frame, face, 256, p.model_template)
            crops.append(("%s@%d" % (os.path.basename(rel), int(total * frac)), np.ascontiguousarray(aligned)))
        cap.release()
    if len(crops) < n:
        raise Unavailable("only %d real crops found (need %d); clips missing?" % (len(crops), n))
    step = len(crops) / float(n)               # spread the pick over all clips
    return [crops[int(i * step)] for i in range(n)]


class Judge:
    """AdaFace identity of a swapped 256 crop to its source; independent of the pipeline's w600k matcher."""

    def __init__(self, p):
        from roop import recognizer_adaface as ada
        if ada.enabled():
            raise Unavailable("ROOP_ADAFACE is on: it would not be an independent recogniser")
        from roop.face_util import align_crop, swap_template_points
        self.ada, self.align_crop = ada, align_crop
        self.kps256 = np.asarray(swap_template_points(256, p.model_template), np.float32)
        self._src = {}

    def source(self, name, fs):
        if name not in self._src:
            embs = []
            for face, img in list(zip(fs.faces, fs.ref_images))[:6]:
                crop, _ = self.align_crop(img, face.kps, 112, mode=self.ada.ALIGN_MODE)
                e = self.ada.embed_crop(crop)
                if e is not None:
                    e = np.asarray(e, np.float64)
                    embs.append(e / max(1e-9, np.linalg.norm(e)))
            if not embs:
                raise Unavailable("no AdaFace embedding for faceset %s" % name)
            m = np.mean(embs, axis=0)
            self._src[name] = m / np.linalg.norm(m)
        return self._src[name]

    def score(self, bgr256, name, fs):
        crop, _ = self.align_crop(bgr256, self.kps256, 112, mode=self.ada.ALIGN_MODE)
        e = self.ada.embed_crop(crop)
        if e is None:
            return float("nan")
        e = np.asarray(e, np.float64)
        return float(np.dot(e / max(1e-9, np.linalg.norm(e)), self.source(name, fs)))


def to_picture(chw, denormalize):
    """Raw model output [3,H,W] -> the 8-bit BGR picture the pipeline would paste (procmgr_tiling.normalize_swap_frame)."""
    x = np.asarray(chw, np.float32).transpose(1, 2, 0)
    if denormalize:
        x = (x + 1.0) / 2.0
    return np.clip(np.round(x * 255.0), 0, 255)[:, :, ::-1].astype(np.float32)


def fp32_graph(model_arg):
    """The swap graph with EVERY float16 tensor made float32 (initializers, Constant/attribute tensors, Cast targets).

    hyperswap_1a_256.onnx is FP16 end to end: FP32 I/O, four boundary Casts, 295 FP16 initializers, and all 16
    InstanceNormalization nodes fed FP16 (checked 2026-10-10). A CUDA/CPU session of the shipped file therefore computes in
    FP16 - it is NOT an FP32 reference. This builds one."""
    import onnx
    from onnx import TensorProto as T, numpy_helper
    m = onnx.load_from_string(bytes(model_arg)) if isinstance(model_arg, (bytes, bytearray)) else onnx.load(model_arg)

    def up(t):
        if t.data_type == T.FLOAT16:
            t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name))
    for init in m.graph.initializer:
        up(init)
    for node in m.graph.node:
        for a in node.attribute:
            if a.type == onnx.AttributeProto.TENSOR:
                up(a.t)
            if node.op_type == "Cast" and a.name == "to" and a.i == T.FLOAT16:
                a.i = T.FLOAT
    del m.graph.value_info[:]
    return m.SerializeToString()


def cuda_session(model_arg):
    import onnxruntime
    from roop.utilities import get_onnx_session_options
    return onnxruntime.InferenceSession(
        model_arg, get_onnx_session_options(),
        providers=[("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"])


def measure(batches=BATCHES, n=N_CROPS):
    """Everything the test asserts on; also the report. Raises Unavailable when the machine cannot run it."""
    _need_stack()
    from roop import swap_canary
    from roop.procmgr_tiling import to_blob
    p, cfg = _build_swapper()
    sources = _library_sources(n)
    crops = _real_crops(p, n)
    judge = Judge(p)
    rows = [{"label": crops[i][0], "src": sources[i % len(sources)][0], "fs": sources[i % len(sources)][1],
             "blob": to_blob(crops[i][1], p.model_mean, p.model_standard_deviation)} for i in range(n)]
    for r in rows:
        r["src_face"] = r["fs"].faces[0]
    denorm = bool(getattr(p, "model_denormalize", False))

    # warm-up on REAL crops: every batch shape, twice, discarded
    for b in (1,) + tuple(batches):
        for _ in range(2):
            p.RunBatchMulti([(r["src_face"], None, r["blob"]) for r in rows[:b]]) if b > 1 else \
                p.Run(rows[0]["src_face"], None, rows[0]["blob"])

    def one(r):
        return np.asarray(p.Run(r["src_face"], None, r["blob"]), np.float32)

    b1 = [one(r) for r in rows]
    b1_again = [one(r) for r in rows]
    refs = {"fp32": cuda_session(fp32_graph(p._model_arg)),          # a genuine FP32 graph
            "graph_dtype": cuda_session(p._model_arg)}               # the shipped FP16 graph on CUDA (what the canary uses)

    def ref_run(sess, group):
        feed = {p.image_input_name: np.concatenate([r["blob"] for r in group], axis=0),
                p.embed_input_name: np.concatenate([p._compute_source_input(r["src_face"]) for r in group], axis=0)}
        return np.asarray(sess.run(None, feed)[0], np.float32)

    ref1 = {k: [ref_run(sess, [r])[0] for r in rows] for k, sess in refs.items()}

    # INFORMATION, not asserted: the same batched-vs-B=1 comparison on the CUDA sessions. The production swapper is
    # TensorRT; ORT's CUDA InstanceNormalization was measured wrong at batch > 1 on this generator (face_engine,
    # 2026-09-28), which is what a TensorRT-less card with >= 10 GB (batching on) would run.
    cuda_batched = {k: {} for k in refs}
    for k, sess in refs.items():
        for b in batches:
            out = [None] * n
            for s0 in range(0, n, b):
                res = ref_run(sess, rows[s0:s0 + b])
                for j in range(len(res)):
                    out[s0 + j] = res[j]
            cuda_batched[k][b] = out
    del refs

    def metrics(out, base, r):
        a, b = to_picture(out, denorm), to_picture(base, denorm)
        return {"max_abs": float(np.abs(out - base).max()),
                "ssim": float(swap_canary.ssim(a, b, 255.0)),
                # `to_picture` already returns BGR (the model emits RGB), which is what AdaFace's crop expects
                "id_delta": judge.score(a.astype(np.uint8), r["src"], r["fs"])
                            - judge.score(b.astype(np.uint8), r["src"], r["fs"])}

    report = {"n_crops": n, "denormalize": denorm,
              "providers": [str(x) for x in p.model_swap_insightface.get_providers()],
              "crops": [{"label": r["label"], "src": r["src"]} for r in rows],
              "controls": {"b1_repeat": [metrics(b1_again[i], b1[i], rows[i]) for i in range(n)],
                           "b1_vs_cuda_fp32": [metrics(b1[i], ref1["fp32"][i], rows[i]) for i in range(n)],
                           "b1_vs_cuda_graph_dtype": [metrics(b1[i], ref1["graph_dtype"][i], rows[i]) for i in range(n)],
                           **{"cuda_%s_B=%d_vs_B=1" % (k, b): [metrics(cuda_batched[k][b][i], ref1[k][i], rows[i])
                                                               for i in range(n)]
                              for k in cuda_batched for b in batches}},
              "batched": {}}
    # PROOF THE BATCH RAN. "Identical" is also what a silent sequential fallback looks like, so every inference the swapper
    # really issues is spied on: the batch dimension of the image feed must be exactly B, n/B times.
    seen = []
    real_infer = p._infer

    def spy(feed):
        seen.append(int(feed[p.image_input_name].shape[0]))
        return real_infer(feed)
    p._infer = spy
    report["infer_calls"] = {}
    for b in batches:
        per_row = [None] * n
        del seen[:]
        for s in range(0, n, b):
            group = rows[s:s + b]
            outs = p.RunBatchMulti([(r["src_face"], None, r["blob"]) for r in group])
            if p._batch_unsupported:
                raise AssertionError("RunBatchMulti fell back to sequential at B=%d: the batch path did not run" % b)
            for j, o in enumerate(outs):
                per_row[s + j] = metrics(np.asarray(o, np.float32), b1[s + j], group[j])
        report["infer_calls"][str(b)] = {"batch_sizes_seen": sorted(set(seen)), "calls": len(seen),
                                         "expected_calls": n // b}
        report["batched"][str(b)] = per_row
    p._infer = real_infer

    def agg(ms):
        return {"min_ssim": min(m["ssim"] for m in ms), "max_abs": max(m["max_abs"] for m in ms),
                "max_abs_id_delta": max(abs(m["id_delta"]) for m in ms if m["id_delta"] == m["id_delta"])}
    report["summary"] = {"b1_repeat": agg(report["controls"]["b1_repeat"]),
                         "b1_vs_cuda_fp32": agg(report["controls"]["b1_vs_cuda_fp32"]),
                         "b1_vs_cuda_graph_dtype": agg(report["controls"]["b1_vs_cuda_graph_dtype"]),
                         **{"cuda_%s_B=%d" % (k, b): agg(report["controls"]["cuda_%s_B=%d_vs_B=1" % (k, b)])
                            for k in ("fp32", "graph_dtype") for b in batches},
                         **{"B=%s" % b: agg(v) for b, v in report["batched"].items()}}
    for b in batches:
        s = report["summary"]["B=%d" % b]
        s["pass"] = bool(s["min_ssim"] >= PASS_SSIM and s["max_abs_id_delta"] <= PASS_ID_DELTA)
    p.Release()
    return report


def format_table(rep):
    lines = ["%-18s %10s %12s %12s %s" % ("comparison", "min SSIM", "max |diff|", "max |id d|", "verdict")]
    for k, s in rep["summary"].items():
        verdict = ("PASS" if s.get("pass") else "FAIL") if "pass" in s else "(control)"
        lines.append("%-18s %10.5f %12.5f %12.5f %s" % (k, s["min_ssim"], s["max_abs"], s["max_abs_id_delta"], verdict))
    for b, ran in rep.get("infer_calls", {}).items():
        lines.append("  B=%s ran as: batch sizes %s, %d inference calls (expected %d)" % (
            b, ran["batch_sizes_seen"], ran["calls"], ran["expected_calls"]))
    for b, rows in rep["batched"].items():
        worst = sorted(range(len(rows)), key=lambda i: rows[i]["ssim"])[:3]
        lines.append("  B=%s worst rows: %s" % (b, ", ".join(
            "%s/%s ssim %.4f diff %.3f id %+.4f" % (rep["crops"][i]["label"], rep["crops"][i]["src"],
                                                   rows[i]["ssim"], rows[i]["max_abs"], rows[i]["id_delta"])
            for i in worst)))
    return "\n".join(lines)


@pytest.fixture(scope="module")
def report():
    try:
        rep = measure()
    except Unavailable as exc:
        pytest.skip(str(exc))
    print("\n" + format_table(rep))
    return rep


@pytest.mark.gpu
@pytest.mark.parametrize("batch", BATCHES)
def test_batched_rows_match_batch_one(report, batch):
    ran = report["infer_calls"][str(batch)]
    assert ran["batch_sizes_seen"] == [batch] and ran["calls"] == ran["expected_calls"], (
        "B=%d did not run as %d-row inferences (saw batch sizes %s, %d calls, expected %d): the comparison below would "
        "be B=1 against itself" % (batch, batch, ran["batch_sizes_seen"], ran["calls"], ran["expected_calls"]))
    s = report["summary"]["B=%d" % batch]
    assert s["min_ssim"] >= PASS_SSIM, "B=%d: worst row SSIM %.5f < %.3f\n%s" % (
        batch, s["min_ssim"], PASS_SSIM, format_table(report))
    assert s["max_abs_id_delta"] <= PASS_ID_DELTA, "B=%d: worst identity delta %.5f > %.3f\n%s" % (
        batch, s["max_abs_id_delta"], PASS_ID_DELTA, format_table(report))


@pytest.mark.gpu
def test_the_instrument_can_tell_rows_apart(report):
    """The crops are real and different, so B=1 repeats must agree with themselves and the FP32 reference must not be
    a copy of the production output (otherwise 'SSIM >= 0.998' could pass on a measurement that sees nothing)."""
    assert report["summary"]["b1_repeat"]["min_ssim"] >= 0.9995, "B=1 is not repeatable: the thresholds mean nothing"
    sims = [m["ssim"] for m in report["controls"]["b1_vs_cuda_fp32"]]
    assert min(sims) < 1.0, "production B=1 is bit-identical to a genuine FP32 graph: is the production session really FP16?"
    assert len({c["src"] for c in report["crops"]}) >= 8, "fewer than 8 distinct sources in the 16 rows"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        rep = measure()
    except Unavailable as exc:
        print("UNAVAILABLE:", exc)
        return 3
    print(format_table(rep))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
        print("wrote", args.out)
    return 0 if all(v.get("pass", True) for v in rep["summary"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
