"""tools/quality_harness.py: the metrics are real, the guard fires, and nothing in the file can fabricate a number.

The harness exists because tools/benchmark_hyperswap_audit.py printed identity cosines it had typed in (0.88, 0.84,
0.81 ... plus Gaussian noise), "measured" eye error as a keypoint set against itself plus noise, and ran ONE ONNX file
under six model names. Each failure mode is pinned here: known-answer tests for every metric function, a guard test
for each way two "different" models can be indistinguishable, parsers checked against real render-log text, and a scan
of the harness source for random number generators.
"""
import importlib.util
import math
import os
import re
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_PATH = os.path.join(ROOT, "tools", "quality_harness.py")
_spec = importlib.util.spec_from_file_location("quality_harness", _PATH)
qh = importlib.util.module_from_spec(_spec)
sys.modules["quality_harness"] = qh                 # dataclasses resolves annotations through sys.modules
_spec.loader.exec_module(qh)

REAL_AUDIT_LOG = """\
[Track] 2 tracks over 240 frames, 2 matched to a source (gate 0.60)
==== SWAP AUDIT — why each detected face was or was not swapped ====
  faces seen                                                       508  100.0%
  swapped (identity lock)                                          508  100.0%
    of those, partly behind an object (masked, still swapped)      315   62.0%
     Separately, 40 frames of 294 (13.6% of frames) had NO face detected at all — that is the detector losing the face
===================================================================
[Session] swapper:hyperswap file=hyperswap_1a_256.onnx provider=TensorrtExecutionProvider trt_fp16=on input=source:Nx512,target:Nx3x256x256 requested=Tensorrt,CUDA,CPU
[Session] enhancer:restoreformer++ file=restoreformer_plus_plus.onnx provider=TensorrtExecutionProvider trt_fp16=off input=input:1x3x512x512 requested=Tensorrt,CUDA,CPU
"""


def textured(seed, size=96):
    """Deterministic test imagery for the metric tests ONLY (the harness itself never generates data)."""
    rng = np.random.RandomState(seed)
    base = rng.randint(0, 256, (size // 8, size // 8, 3)).astype(np.uint8)
    return cv2_resize(base, size)


def cv2_resize(a, size):
    import cv2
    return cv2.resize(a, (size, size), interpolation=cv2.INTER_CUBIC)


class BoxAndRecallTests(unittest.TestCase):
    def test_iou_known_values(self):
        self.assertEqual(qh.box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(qh.box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(qh.box_iou((0, 0, 10, 10), (5, 0, 15, 10)), 50.0 / 150.0, places=9)

    def test_recall_counts_matched_reference_faces(self):
        ref = [(0, 0, 10, 10), (100, 100, 120, 120)]
        self.assertEqual(qh.detection_recall(ref, [(1, 1, 11, 11)]), (1, 2))
        self.assertEqual(qh.detection_recall(ref, ref), (2, 2))
        self.assertEqual(qh.detection_recall(ref, []), (0, 2))

    def test_one_candidate_box_cannot_recall_two_people(self):
        ref = [(0, 0, 10, 10), (0, 0, 10, 11)]
        self.assertEqual(qh.detection_recall(ref, [(0, 0, 10, 10)]), (1, 2))

    def test_threshold_is_respected(self):
        self.assertEqual(qh.detection_recall([(0, 0, 10, 10)], [(5, 0, 15, 10)], 0.5), (0, 1))     # IoU 0.333
        self.assertEqual(qh.detection_recall([(0, 0, 10, 10)], [(5, 0, 15, 10)], 0.3), (1, 1))


class ImageMetricTests(unittest.TestCase):
    def test_cosine(self):
        self.assertAlmostEqual(qh.cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(qh.cosine([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(qh.cosine([1, 0], [-2, 0]), -1.0)
        self.assertTrue(math.isnan(qh.cosine(None, [1, 0])))
        self.assertTrue(math.isnan(qh.cosine([0, 0], [1, 0])))

    def test_ssim_is_one_for_identical_and_falls_with_damage(self):
        a = textured(1)
        mask = np.full(a.shape[:2], 255, np.uint8)
        self.assertAlmostEqual(qh.masked_ssim(a, a, mask), 1.0, places=9)
        noisy = np.clip(a.astype(np.int16) + np.random.RandomState(3).randint(-30, 31, a.shape), 0, 255).astype(np.uint8)
        other = textured(2)
        s_noisy, s_other = qh.masked_ssim(a, noisy, mask), qh.masked_ssim(a, other, mask)
        self.assertLess(s_noisy, 0.99)
        self.assertLess(s_other, s_noisy)

    def test_ssim_only_reads_inside_the_mask(self):
        a = textured(4, 128)
        b = a.copy()
        b[:, :40] = 255 - b[:, :40]                       # damage the left strip only
        mask = np.zeros(a.shape[:2], np.uint8)
        mask[:, 80:] = 255                                # far from the damage (window radius is 5 px)
        self.assertAlmostEqual(qh.masked_ssim(a, b, mask), 1.0, places=9)
        mask[:, :40] = 255
        self.assertLess(qh.masked_ssim(a, b, mask), 0.9)

    def test_psnr_known_answer_and_cap(self):
        a = np.full((32, 32, 3), 100, np.uint8)
        b = np.full((32, 32, 3), 110, np.uint8)
        mask = np.full((32, 32), 255, np.uint8)
        self.assertAlmostEqual(qh.masked_psnr(a, b, mask), 10.0 * math.log10(255.0 ** 2 / 100.0), places=9)
        self.assertEqual(qh.masked_psnr(a, a, mask), qh.PSNR_CAP_DB)

    def test_psnr_only_reads_inside_the_mask(self):
        a = np.full((32, 32, 3), 100, np.uint8)
        b = a.copy()
        b[:, :16] = 200
        mask = np.zeros((32, 32), np.uint8)
        mask[:, 16:] = 255
        self.assertEqual(qh.masked_psnr(a, b, mask), qh.PSNR_CAP_DB)

    def test_mask_iou(self):
        a = np.zeros((10, 10), np.float32)
        b = np.zeros((10, 10), np.float32)
        a[:5, :] = 1.0
        b[:, :5] = 1.0
        self.assertAlmostEqual(qh.mask_iou(a, b), 25.0 / 75.0, places=9)
        self.assertEqual(qh.mask_iou(a, a), 1.0)
        # two empty masks = no face found in either = not measurable, never a free perfect score
        self.assertTrue(math.isnan(qh.mask_iou(np.zeros((4, 4)), np.zeros((4, 4)))))

    def test_summarize_ignores_nan_and_none(self):
        s = qh.summarize([1.0, 3.0, float("nan"), None])
        self.assertEqual((s["n"], s["mean"], s["min"]), (2, 2.0, 1.0))
        self.assertEqual(qh.summarize([])["n"], 0)


class PoseDetailJitterTests(unittest.TestCase):
    def test_yaw_bins(self):
        self.assertEqual([qh.yaw_bin(v) for v in (0, 19.9, -20, 44.9, 45, 74.9, -75, 90, 179)],
                         ["0-20", "0-20", "20-45", "20-45", "45-75", "45-75", "75+", "75+", "75+"])
        self.assertIsNone(qh.yaw_bin(None))
        self.assertIsNone(qh.yaw_bin(float("nan")))

    def test_laplacian_variance_sees_detail_and_respects_the_mask(self):
        flat = np.full((64, 64, 3), 128, np.uint8)
        mask = np.full((64, 64), 255, np.uint8)
        self.assertEqual(qh.masked_laplacian_variance(flat, mask), 0.0)
        busy = flat.copy()
        busy[::2, ::2] = 255
        self.assertGreater(qh.masked_laplacian_variance(busy, mask), 1000.0)
        half = np.zeros((64, 64), np.uint8)
        half[:, 32:] = 255
        detail_left = busy.copy()
        detail_left[:, 32:] = 128                       # detail only on the left; mask covers only the right
        self.assertEqual(qh.masked_laplacian_variance(detail_left, half), 0.0)
        self.assertTrue(math.isnan(qh.masked_laplacian_variance(flat, np.zeros((64, 64), np.uint8))))

    def test_identity_jitter_counts_only_adjacent_frames(self):
        self.assertAlmostEqual(qh.identity_jitter([0, 1, 2], [0.5, 0.7, 0.6]), (0.2 + 0.1) / 2, places=12)
        self.assertAlmostEqual(qh.identity_jitter([2, 0, 1], [0.6, 0.5, 0.7]), (0.2 + 0.1) / 2, places=12)   # order-free
        self.assertTrue(math.isnan(qh.identity_jitter([0, 5], [0.1, 0.9])))             # a gap is not a flicker
        self.assertEqual(qh.identity_jitter([0, 1, 2], [0.5, 0.5, 0.5]), 0.0)           # steady identity = no jitter

    def test_new_metrics_are_in_the_guarded_set(self):
        for m in ("eye_drift_px", "mouth_drift_px", "skin_detail", "identity_jitter"):
            self.assertIn(m, qh.MODEL_DEPENDENT)
            self.assertIn(m, qh.FLAT_KEYS)


class ParserTests(unittest.TestCase):
    def test_swap_audit_counts_come_from_the_log(self):
        a = qh.parse_swap_audit(REAL_AUDIT_LOG)
        self.assertEqual(a["faces seen"], 508)
        self.assertEqual(a["swapped (identity lock)"], 508)
        self.assertEqual(a["of those, partly behind an object (masked, still swapped)"], 315)
        self.assertEqual(a["frames with no face detected at all"], 40)
        self.assertEqual(a["frames considered"], 294)

    def test_track_count_and_sessions(self):
        self.assertEqual(qh.parse_track_count(REAL_AUDIT_LOG), 2)
        self.assertIsNone(qh.parse_track_count("nothing"))
        s = qh.parse_sessions(REAL_AUDIT_LOG)
        self.assertEqual([(r["tag"], r["file"], r["trt_fp16"]) for r in s],
                         [("swapper:hyperswap", "hyperswap_1a_256.onnx", "on"),
                          ("enhancer:restoreformer++", "restoreformer_plus_plus.onnx", "off")])

    def test_candidate_spec(self):
        v = qh.parse_candidate("mixed|swap_model=hyperswap|trt_precision=MIXED|env.ROOP_X=1")
        self.assertEqual((v.name, v.swap_model, v.trt_precision, v.env), ("mixed", "hyperswap", "mixed", {"ROOP_X": "1"}))
        for bad in ("swap_model=x", "n|trt_precision=fp32", "n|swap_model=x|bogus=1", qh.REFERENCE_NAME + "|swap_model=x"):
            with self.assertRaises(ValueError):
                qh.parse_candidate(bad)

    def test_capture_fixture_compares_the_decision_not_the_diagnostics(self):
        # the two real d1 renders: same frames chosen, different scan time and detector-precision floats
        fp32 = ["[bench] auto-capture: 2 people, seed frame 26, separation 0.970, 197 frames scanned in 54.9s",
                "[bench]   person 0: frame 360, off-axis 69.0 deg (best frame)",
                "[bench]   person 1: frame 152, off-axis 72.9 deg (best frame)"]
        fp16 = ["[bench] auto-capture: 2 people, seed frame 26, separation 0.974, 197 frames scanned in 47.7s",
                "[bench]   person 0: frame 360, off-axis 68.9 deg (best frame)",
                "[bench]   person 1: frame 152, off-axis 73.1 deg (best frame)"]
        self.assertEqual(qh.capture_fixture(fp32), qh.capture_fixture(fp16))
        self.assertEqual(qh.capture_fixture(fp32), (2, ((0, 360), (1, 152)), None))
        moved = [fp16[0], fp16[1].replace("frame 360", "frame 361"), fp16[2]]
        self.assertNotEqual(qh.capture_fixture(fp32), qh.capture_fixture(moved))
        fewer = [fp16[0].replace("2 people", "1 people"), fp16[1]]
        self.assertNotEqual(qh.capture_fixture(fp32), qh.capture_fixture(fewer))
        manual = ["[bench] target faces captured from frame 4930"]
        self.assertEqual(qh.capture_fixture(manual), (None, (), 4930))
        self.assertNotEqual(qh.capture_fixture(manual), qh.capture_fixture(["[bench] target faces captured from frame 4931"]))

    def test_the_seed_frame_alone_is_not_a_different_fixture(self):
        # d6: an FP32 and an FP16 detector seeded the search at 452 / 456 and captured the SAME two people from the SAME frames
        a = ["[bench] auto-capture: 2 people, seed frame 452, separation 0.871, 244 frames scanned in 99.0s",
             "[bench]   person 0: frame 400, off-axis 50.0 deg (best frame)", "[bench]   person 1: frame 94, off-axis 41.0 deg (best frame)"]
        b = [a[0].replace("seed frame 452", "seed frame 456"), a[1], a[2]]
        self.assertEqual(qh.capture_fixture(a), qh.capture_fixture(b))
        self.assertEqual((qh.capture_seed(a), qh.capture_seed(b)), (452, 456))
        self.assertIsNone(qh.capture_seed(["[bench] target faces captured from frame 4930"]))

    def test_render_errors_catch_a_render_that_returned_zero_but_did_not_finish(self):
        broken = ("[Stabilize] write thread failed: OSError: Roop Ultimate error: the ffmpeg encoder process exited "
                  "unexpectedly (code None) before the video was finished.")
        self.assertEqual(sorted(qh.render_errors(broken)), ["encoder process exited", "write thread failed"])
        self.assertEqual(qh.render_errors("[Fallback] scene_detector.py: ModuleNotFoundError: No module named 'scenedetect'"), [])
        self.assertEqual(qh.render_errors(REAL_AUDIT_LOG), [])
        self.assertEqual(qh.render_errors(""), [])
        self.assertEqual(qh.render_errors(None), [])

    def test_render_problems_check_the_frame_count_of_the_video_itself(self):
        import cv2
        import tempfile
        d = tempfile.mkdtemp()
        path = os.path.join(d, "v.avi")
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
        for k in range(7):
            w.write(np.full((32, 32, 3), 10 * k, np.uint8))
        w.release()
        self.assertEqual(qh.count_video_frames(path), 7)
        self.assertEqual(qh.render_problems("clean log", path, 7), [])
        self.assertTrue(any("7 frames, expected 8" in p for p in qh.render_problems("clean log", path, 8)))     # the 299-of-300 case
        self.assertTrue(any("no output video" in p for p in qh.render_problems("clean log", None, 7)))
        self.assertTrue(any("write thread failed" in p for p in qh.render_problems("write thread failed", path, 7)))

    def test_input_signatures_match_across_the_apps_shape_spellings(self):
        a = qh.normalize_input_sig("source:Nx512,target:Nx3x256x256")
        b = qh.normalize_input_sig("target:Nonex3x256x256,source:unk__7x512")
        self.assertEqual(a, b)
        self.assertNotEqual(a, qh.normalize_input_sig("target:Nx3x128x128,source:Nx512"))
        self.assertEqual(qh.normalize_input_sig("xseg_input:0:unk__1491x256x256x3"), "xseg_input:0:Nx256x256x3")
        self.assertEqual(qh.normalize_input_sig(""), "")

    def test_session_lines_carry_their_input_signature(self):
        s = qh.parse_sessions(REAL_AUDIT_LOG)
        self.assertEqual(s[0]["input"], "source:Nx512,target:Nx3x256x256")
        self.assertEqual(s[1]["input"], "input:1x3x512x512")

    def test_per_model_log_names_a_pathless_session_from_the_render_log_and_keeps_sessions_apart(self):
        import json
        import tempfile
        d = tempfile.mkdtemp()
        rec = lambda file, prov, sig, before, after: {                                     # noqa: E731
            "file": file, "provider": prov, "trt_fp16": "on" if "Tensorrt" in prov else "n/a",
            "input_signature": sig, "vram_before_mib": before, "vram_after_mib": after,
            "vram_delta_mib": after - before, "first_call_ms": 12.0}
        path = os.path.join(d, "fi.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in (rec("?", "TensorrtExecutionProvider", "source:Nx512,target:Nx3x256x256", 1000, 2400),
                      rec("?", "CUDAExecutionProvider", "image:Nx3x64x64", 2400, 2410),
                      rec("det_10g.onnx", "TensorrtExecutionProvider", "input.1:1x3xNxN", 500, 560)):
                fh.write(json.dumps(r) + "\n")
        meta = {"first_inference_log": path, "sessions": qh.parse_sessions(REAL_AUDIT_LOG)}
        rows = qh.per_model_log({"v": [meta]})["v"]
        by = {(r["file"], r["provider"]): r for r in rows}
        swap = by[("hyperswap_1a_256.onnx", "TensorrtExecutionProvider")]
        self.assertEqual(swap["first_inference_delta_mib"]["mean"], 1400.0)
        self.assertIn(("?", "CUDAExecutionProvider"), by)                  # no matching Session line: left unnamed, not guessed
        self.assertEqual(len(rows), 3)                                     # three sessions, three rows

    def test_lossless_encode_detection_uses_the_real_ffmpeg_banner(self):
        lossless = ("Stream #0:0[0x1](und): Video: h264 (High 4:4:4 Predictive) (avc1 / 0x31637661), "
                    "yuv420p(tv, bt709, progressive), 1280x720, 56767 kb/s, 30 fps")
        lossy = "Stream #0:0: Video: h264 (High) (avc1 / 0x31637661), yuv420p, 1280x720, 4000 kb/s, 30 fps"
        hevc = "Stream #0:0: Video: hevc (Main) (hvc1 / 0x31637668), yuv420p, 1280x720, 9000 kb/s, 30 fps"
        self.assertTrue(qh.is_lossless_encode(lossless))
        self.assertFalse(qh.is_lossless_encode(lossy))
        self.assertFalse(qh.is_lossless_encode(hevc))
        self.assertFalse(qh.is_lossless_encode(""))
        self.assertFalse(qh.is_lossless_encode(None))

    def test_reference_must_really_be_fp32(self):
        fp16 = [{"file": "hyperswap_1a_256.onnx", "provider": "TensorrtExecutionProvider", "trt_fp16": "on"}]
        self.assertTrue(qh.verify_reference_is_fp32(fp16, []))
        ok = [{"file": "x.onnx", "provider": "TensorrtExecutionProvider", "trt_fp16": "off"},
              {"file": "y.onnx", "provider": "CUDAExecutionProvider", "trt_fp16": "n/a"}]
        self.assertEqual(qh.verify_reference_is_fp32(ok, []), [])
        self.assertTrue(qh.verify_reference_is_fp32([], [{"tag": "t", "file": "z.onnx",
                                                          "provider": "TensorrtExecutionProvider", "trt_fp16": "on"}]))


def variant(swap_model, files, **metrics):
    return {"swap_model": swap_model, "model_key": qh.model_key(files), "metrics": dict(metrics)}


class GuardTests(unittest.TestCase):
    A = [("hyperswap_1a_256.onnx", "aa" * 32)]
    B = [("inswapper_128.onnx", "bb" * 32)]

    def test_different_networks_with_different_numbers_pass(self):
        v = {"hyper": variant("hyperswap", self.A, **{"d4.identity_cos": 0.61, "d4.ssim": 0.93}),
             "insw": variant("inswapper", self.B, **{"d4.identity_cos": 0.48, "d4.ssim": 0.90})}
        self.assertEqual(qh.distinctness_violations(v), [])

    def test_a_model_dependent_metric_identical_across_two_models_fails(self):
        v = {"hyper": variant("hyperswap", self.A, **{"d4.identity_cos": 0.61, "d4.ssim": 0.93}),
             "insw": variant("inswapper", self.B, **{"d4.identity_cos": 0.61, "d4.ssim": 0.90})}
        out = qh.distinctness_violations(v)
        self.assertTrue(any("d4.identity_cos" in p and "identical" in p for p in out), out)

    def test_a_hard_coded_constant_is_caught_even_when_every_pair_differs_elsewhere(self):
        v = {"a": variant("m1", [("a.onnx", "01" * 32)], **{"ALL.psnr_db": 94.5, "ALL.ssim": 0.91}),
             "b": variant("m2", [("b.onnx", "02" * 32)], **{"ALL.psnr_db": 94.5, "ALL.ssim": 0.88}),
             "c": variant("m3", [("c.onnx", "03" * 32)], **{"ALL.psnr_db": 94.5, "ALL.ssim": 0.86})}
        out = qh.distinctness_violations(v)
        self.assertTrue(any("same value" in p and "ALL.psnr_db" in p for p in out), out)

    def test_one_network_behind_two_names_fails(self):
        # the old audit tool: hyperswap_1b and hyperswap_1c have no ONNX on disk and ran hyperswap_1a's file
        v = {"hyper_1a": variant("hyperswap_1a", self.A, **{"d4.ssim": 0.93}),
             "hyper_1b": variant("hyperswap_1b", self.A, **{"d4.ssim": 0.93})}
        out = qh.distinctness_violations(v)
        self.assertTrue(any("one model behind two names" in p for p in out), out)

    def test_two_precisions_of_one_model_are_not_compared(self):
        v = {"mixed": variant("hyperswap", self.A, **{"d4.ssim": 0.93}),
             "fp16": variant("hyperswap", self.A, **{"d4.ssim": 0.93})}
        self.assertEqual(qh.distinctness_violations(v), [])

    def test_metrics_that_cannot_depend_on_the_swapper_are_exempt_unless_strict(self):
        v = {"hyper": variant("hyperswap", self.A, **{"d4.ssim": 0.93, "d4.detection_recall": 1.0}),
             "insw": variant("inswapper", self.B, **{"d4.ssim": 0.90, "d4.detection_recall": 1.0})}
        self.assertEqual(qh.distinctness_violations(v), [])
        out = qh.distinctness_violations(v, strict_all=True)
        self.assertTrue(any("detection_recall" in p for p in out), out)

    def test_an_unlocatable_swapper_file_means_nothing_can_be_trusted(self):
        v = {"x": variant("hyperswap_1b", [("hyperswap_1b_256.onnx", None)], **{"d4.ssim": 0.9}),
             "y": variant("inswapper", self.B, **{"d4.ssim": 0.8})}
        self.assertTrue(any("could not be located" in p for p in qh.distinctness_violations(v)))

    def test_model_key_ignores_the_name_and_sees_the_content(self):
        self.assertEqual(qh.model_key(self.A), qh.model_key(list(self.A)))
        self.assertNotEqual(qh.model_key(self.A), qh.model_key([("hyperswap_1a_256.onnx", "cc" * 32)]))
        self.assertIn("?", qh.model_key([("x.onnx", None)]))


class NothingIsFabricatedTests(unittest.TestCase):
    """The tool this replaces typed its numbers in. The harness source must not be able to."""

    @classmethod
    def setUpClass(cls):
        with open(_PATH, encoding="utf-8") as fh:
            cls.src = fh.read()
        cls.code = "\n".join(l for l in cls.src.splitlines() if not l.lstrip().startswith("#"))

    def test_no_random_number_generator_anywhere(self):
        for needle in ("np.random", "numpy.random", "import random", "random.seed", "default_rng", "RandomState"):
            self.assertNotIn(needle, self.code, needle)

    def test_no_metric_is_assigned_a_literal(self):
        for m in re.finditer(r"^\s*['\"]?(identity|ssim|psnr|iou|recall|cosine)[A-Za-z_]*['\"]?\s*[:=]\s*[0-9.]+\s*,?\s*$",
                             self.code, re.M):
            self.fail("metric assigned a literal: %r" % m.group(0))

    def test_no_fallback_constant_for_an_empty_measurement(self):
        # `x if x else 0.86` was how the old tool reported a number when it had measured nothing. A non-zero decimal
        # after else/or is that pattern; a plain 0.0 is a definition (the IoU of an empty union) and is allowed.
        self.assertNotRegex(self.code, r"\belse\s+0\.[1-9]")
        self.assertNotRegex(self.code, r"\bor\s+0\.[1-9]")
        self.assertNotRegex(self.code, r"\belse\s+[1-9]\d*\.\d+")

    def test_the_identity_judge_is_not_the_pipelines_recogniser(self):
        self.assertIn("recognizer_adaface", self.src)
        self.assertIn("ada.enabled()", self.src)
        # a detected Face's own `.embedding` / `.normed_embedding` IS the pipeline's w600k vector
        for needle in (".embedding", "normed_embedding", "compute_cosine_distance", "recognition_engine"):
            self.assertNotIn(needle, self.code, needle)


class FixtureHookTests(unittest.TestCase):
    """Both ends of the capture-fixture pin, and the reason it exists, stay wired."""

    def test_the_render_harness_saves_and_loads_the_captured_targets_as_plain_dicts(self):
        with open(os.path.join(HERE, "two_face_video.py"), encoding="utf-8") as fh:
            src = fh.read()
        for needle in ("ROOP_BENCH_FIXTURE_IN", "ROOP_BENCH_FIXTURE_OUT", "fixture pinned from", "fixture saved to"):
            self.assertIn(needle, src, needle)
        self.assertIn("[dict(t) for t in targets]", src)        # never pickle a Face: its __getattr__ answers None to dunders
        self.assertNotIn("pickle.dump(targets", src)

    def test_the_insightface_face_really_cannot_be_pickled_directly(self):
        import pickle
        from insightface.app.common import Face
        f = Face(kps=np.zeros((5, 2), np.float32), embedding=np.ones(4, np.float32))
        with self.assertRaises(TypeError):
            pickle.dumps(f)
        g = Face(pickle.loads(pickle.dumps(dict(f))))
        self.assertTrue(np.array_equal(g.kps, f.kps) and np.array_equal(g.embedding, f.embedding))

    def test_the_harness_pins_a_candidate_whose_own_capture_differs(self):
        with open(_PATH, encoding="utf-8") as fh:
            src = fh.read()
        for needle in ("fixture_in=fx", "fixture_out=fx", "not bit-reproducible", "the render did not load the pinned capture fixture"):
            self.assertIn(needle, src, needle)


class BenchmarkLabellingTests(unittest.TestCase):
    """The other benchmark tools generate their inputs. They must say so, and the audit must not invent numbers."""

    TOOLS = os.path.join(ROOT, "tools")
    GENERATORS = re.compile(r"np\.random|default_rng|RandomState|np\.full\(")

    @staticmethod
    def code_of(src):
        return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))

    def tool_sources(self):
        for name in sorted(os.listdir(self.TOOLS)):
            if name.startswith("benchmark_") and name.endswith(".py"):
                with open(os.path.join(self.TOOLS, name), encoding="utf-8") as fh:
                    yield name, fh.read()

    def test_a_benchmark_that_generates_data_declares_it(self):
        seen = 0
        for name, src in self.tool_sources():
            if name == "benchmark_hyperswap_audit.py":
                continue                                     # checked on its own: it must NOT generate anything
            if self.GENERATORS.search(self.code_of(src)):
                seen += 1
                self.assertIn("declare(", src, "%s generates its inputs but never calls declare()" % name)
                self.assertIn('"synthetic_inputs"', src, "%s does not store the declaration in its JSON" % name)
                self.assertIn("SYNTHETIC INPUTS", src.split('"""')[1], "%s docstring does not say so" % name)
        self.assertGreaterEqual(seen, 6)

    def test_the_rewritten_audit_has_no_generator_and_no_typed_in_score(self):
        src = dict(self.tool_sources())["benchmark_hyperswap_audit.py"]
        # the module docstring names the old fabrications; only the executable code after it is checked
        body = self.code_of(src).split('"""', 2)[2]
        for needle in ("np.random", "default_rng", "RandomState", "94.5", "92.0", "96.2", "0.865", "0.88 +", "sim ="):
            self.assertNotIn(needle, body, needle)
        self.assertNotRegex(body, r"\bor\s+0\.[1-9]")
        self.assertIn("quality_harness", body)

    def test_audit_skips_a_missing_model_rather_than_running_another_file_under_its_name(self):
        spec = importlib.util.spec_from_file_location("hs_audit", os.path.join(self.TOOLS, "benchmark_hyperswap_audit.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["hs_audit"] = mod
        spec.loader.exec_module(mod)
        runnable, skipped = mod.availability(download_missing=False)
        from roop.processors.FaceSwapInsightFace import SWAP_MODELS
        models = os.path.join(APP, "models")
        for name, key in runnable:
            for f in [SWAP_MODELS[key]["file"]] + mod.EXTRA_FILES.get(name, []):
                self.assertTrue(os.path.exists(os.path.join(models, f)), "%s is runnable but %s is not on disk" % (name, f))
        for s in skipped:
            self.assertTrue("not on disk" in s["reason"] or "not registered" in s["reason"], s)
        self.assertEqual(len(runnable) + len(skipped), len(mod.VARIANTS))
        self.assertEqual({n for n, _ in runnable} | {s["variant"] for s in skipped}, {n for n, _ in mod.VARIANTS})

    def test_the_stage_reports_that_quote_these_tools_carry_the_banner(self):
        for name in ("STAGE3_DETECTOR_OPTIMIZATION_REPORT.md", "STAGE4_ALIGNMENT_GEOMETRIC_STABILITY_REPORT.md",
                     "STAGE5_HYPERSWAP_AUDIT_REPORT.md", "STAGE6_RESTORE_ULTRA_AUDIT_REPORT.md",
                     "STAGE7_XSEG3_AUDIT_REPORT.md", "STAGE8_COMPOSITING_AUDIT_REPORT.md",
                     "STAGE9_TEMPORAL_TRACKING_AUDIT_REPORT.md"):
            with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
                self.assertIn("<!-- synthetic-input-banner -->", fh.read(), name)

    def test_stage5_json_is_marked_invalid_unless_a_real_run_replaced_it(self):
        import json
        with open(os.path.join(ROOT, "benchmark_stage5_hyperswap.json"), encoding="utf-8") as fh:
            j = json.load(fh)
        self.assertTrue("INVALID" in j or j.get("schema_version") == 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
