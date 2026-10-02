"""Recognition settings: hardware tiers, catalogue, apply signals, and the routes.

The tier tests feed classify_hardware() hand-made facts, so every tier is covered without
that hardware; the route tests mount only the router (no api.py, no GPU) with the engine
faked at face_analyser.set_recognition_model.
"""
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import routes_recognition as rr
    from roop import ui_recognition as ui
    from roop import face_analyser
    _IMPORT_ERROR = None
except ImportError as exc:                       # light profile
    _IMPORT_ERROR = exc


def setUpModule():
    if _IMPORT_ERROR is not None:
        raise unittest.SkipTest(f"recognition UI stack not importable here: {_IMPORT_ERROR}")


ALL_NV = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
CUDA_ONLY = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def nv(cap, vram, name="Some GPU"):
    return {"cuda": True, "name": name, "capability": cap, "vram_gb": vram}


NO_GPU = {"cuda": False, "name": "", "capability": None, "vram_gb": 0.0}


class TestTiers(unittest.TestCase):
    def tier(self, probe, providers, system="Windows", machine="AMD64"):
        return ui.classify_hardware(probe, providers, system, machine)

    def test_ada_with_tensorrt_recommends_tensorrt(self):
        a = self.tier(nv((8, 9), 11.99, "RTX 4070"), ALL_NV)
        self.assertEqual((a.tier, a.provider, a.model_hint), (ui.TIER_ENTHUSIAST_ADA, "tensorrt", None))
        self.assertEqual((a.compute_capability, a.architecture), ("8.9", "Ada Lovelace"))

    def test_ada_without_tensorrt_or_below_7gb_recommends_cuda_and_says_why(self):
        for providers, vram, why in ((CUDA_ONLY, 12.0, "not available"), (ALL_NV, 6.0, "below 7 GB")):
            a = self.tier(nv((8, 9), vram), providers)
            self.assertEqual((a.tier, a.provider), (ui.TIER_ENTHUSIAST_ADA, "cuda"))
            self.assertIn(why, a.reason)

    def test_ampere_is_cuda_heuristic_and_never_tensorrt(self):
        a = self.tier(nv((8, 6), 6.0, "RTX 3060 Laptop"), ALL_NV)
        self.assertEqual((a.tier, a.provider), (ui.TIER_MID_AMPERE, "cuda"))
        self.assertIn("HEURISTIC", a.strategy)

    def test_tier_follows_capability_not_the_marketing_name(self):
        """An unknown future card is classified by what it can do, not treated as a 4070."""
        self.assertEqual(self.tier(nv((8, 6), 24, "RTX 4090 (mislabelled)"), ALL_NV).tier, ui.TIER_MID_AMPERE)
        self.assertEqual(self.tier(nv((12, 0), 32, "Future"), ALL_NV).tier, ui.TIER_ENTHUSIAST_ADA)
        self.assertEqual(self.tier(nv((7, 5), 8, "RTX 2070"), ALL_NV).tier, ui.TIER_CUDA_GENERIC)

    def test_cuda_card_with_a_cpu_only_ort_does_not_claim_cuda(self):
        a = self.tier(nv((8, 9), 12), ["CPUExecutionProvider"])
        self.assertEqual((a.tier, a.provider), (ui.TIER_CPU, "cpu"))

    def test_directml_on_windows_without_cuda(self):
        a = self.tier(NO_GPU, ["DmlExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual((a.tier, a.provider), (ui.TIER_DIRECTML, "directml"))
        self.assertEqual(self.tier(NO_GPU, ["DmlExecutionProvider", "CPUExecutionProvider"], "Linux").tier,
                         ui.TIER_CPU)

    def test_apple_silicon(self):
        a = self.tier(NO_GPU, ["CoreMLExecutionProvider", "CPUExecutionProvider"], "Darwin", "arm64")
        self.assertEqual((a.tier, a.provider), (ui.TIER_APPLE_SILICON, "coreml"))
        self.assertEqual(self.tier(NO_GPU, ["CoreMLExecutionProvider", "CPUExecutionProvider"],
                                   "Darwin", "x86_64").tier, ui.TIER_CPU)
        self.assertEqual(self.tier(NO_GPU, ["CPUExecutionProvider"], "Darwin", "arm64").tier, ui.TIER_CPU)

    def test_cpu_tier_suggests_a_registered_lighter_model_and_nothing_else_does(self):
        cpu = self.tier(NO_GPU, ["CPUExecutionProvider"])
        self.assertEqual((cpu.tier, cpu.model_hint), (ui.TIER_CPU, "mobilefacenet"))
        self.assertIn(cpu.model_hint, ui.RECOGNITION_REGISTRY)
        for probe, providers in ((nv((8, 9), 12), ALL_NV), (nv((8, 6), 6), CUDA_ONLY), (nv((7, 5), 8), CUDA_ONLY)):
            self.assertIsNone(self.tier(probe, providers).model_hint)

    def test_every_recommended_provider_is_a_selector_option(self):
        for probe, providers, system, machine in (
                (nv((8, 9), 12), ALL_NV, "Windows", "AMD64"), (nv((8, 6), 6), CUDA_ONLY, "Windows", "AMD64"),
                (NO_GPU, ["DmlExecutionProvider"], "Windows", "AMD64"),
                (NO_GPU, ["CoreMLExecutionProvider"], "Darwin", "arm64"), (NO_GPU, [], "Linux", "x86_64")):
            self.assertIn(self.tier(probe, providers, system, machine).provider, ui._PROVIDERS)


class TestCatalogAndOptions(unittest.TestCase):
    def test_catalog_lists_every_registered_model_with_its_specs(self):
        with tempfile.TemporaryDirectory() as d:
            rows = {r["name"]: r for r in ui.model_catalog(d)}
        self.assertEqual(set(rows), set(ui.RECOGNITION_REGISTRY))
        self.assertEqual((rows["facerecognizersf"]["output_dim"], rows["adaface"]["color_space"],
                          rows["default"]["input"]), (128, "BGR", "112x112"))
        self.assertFalse(any(r["downloaded"] for r in rows.values()))

    def test_downloaded_flag_and_shared_file(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "glintr100.onnx"), "wb").close()
            rows = {r["name"]: r for r in ui.model_catalog(d)}
        self.assertTrue(rows["glintr100"]["downloaded"] and rows["antelopev2"]["downloaded"])
        self.assertEqual(rows["glintr100"]["same_file_as"], ["antelopev2"])
        self.assertEqual(rows["default"]["same_file_as"], [])

    def test_provider_options_cover_the_five_strategies_plus_follow_app(self):
        with mock.patch("onnxruntime.get_available_providers", return_value=CUDA_ONLY):
            opts = {o["value"]: o for o in ui.provider_options()}
        self.assertEqual(set(opts), {"app", "cuda", "tensorrt", "directml", "coreml", "cpu"})
        self.assertTrue(opts["cuda"]["available"] and opts["app"]["available"])
        self.assertFalse(opts["tensorrt"]["available"])
        self.assertIn("TensorrtExecutionProvider", opts["tensorrt"]["reason"])


class TestSelection(unittest.TestCase):
    def test_normalise(self):
        self.assertEqual(ui.normalise_selection("adaface", "TensorRT"), ("adaface", "tensorrt"))
        self.assertEqual(ui.normalise_selection(None, None), ("default", "app"))
        with self.assertRaisesRegex(ValueError, "Unsupported recognition model"):
            ui.normalise_selection("nope", "cpu")
        with self.assertRaisesRegex(ValueError, "Unsupported provider"):
            ui.normalise_selection("default", "tpu")

    def test_invalid_saved_values_reset_instead_of_crashing_the_panel(self):
        bad = SimpleNamespace(recognition_model="deleted_model", recognition_provider="tpu")
        self.assertEqual(ui.saved_selection(bad), ("default", "app"))
        self.assertEqual(ui.saved_selection(SimpleNamespace()), ("default", "app"))
        self.assertEqual(ui.saved_selection(SimpleNamespace(recognition_model="glintr100",
                                                            recognition_provider="cpu")), ("glintr100", "cpu"))


def _fake_engine(model="adaface", device="cuda", degraded=False, log=()):
    info = {"model": model, "path": "x", "requested_device": device, "degraded": degraded,
            "active_providers": ["CPUExecutionProvider"] if degraded else ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "gpu": None, "fallback_log": list(log)}
    return SimpleNamespace(describe=lambda: info, model_name=model)


class TestApply(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        for name in ("set_recognition_model", "recognition_model_name", "get_recognition_engine"):
            p = mock.patch.object(face_analyser, name)
            setattr(self, name, p.start())
            self.addCleanup(p.stop)
        self.recognition_model_name.return_value = None

    def test_follow_app_passes_none_and_signals_the_change(self):
        self.set_recognition_model.return_value = _fake_engine("adaface", "cuda")
        out = ui.apply_selection("adaface", "app", self.dir)
        self.set_recognition_model.assert_called_once_with("adaface", None, None, self.dir)
        self.assertEqual((out["changed"], out["reinitialized"], out["downloaded"], out["degraded"]),
                         (True, True, True, False))
        self.assertIn("embedding API", out["scope"])
        self.assertIn("AdaFace", out["message"])

    def test_explicit_provider_is_forwarded_and_no_download_when_present(self):
        open(os.path.join(self.dir, "adaface_ir101.onnx"), "wb").close()
        self.set_recognition_model.return_value = _fake_engine("adaface", "tensorrt")
        out = ui.apply_selection("adaface", "tensorrt", self.dir)
        self.assertEqual(self.set_recognition_model.call_args[0][:2], ("adaface", "tensorrt"))
        self.assertFalse(out["downloaded"])

    def test_degraded_session_is_a_warning_not_a_success_message(self):
        self.set_recognition_model.return_value = _fake_engine("default", "cuda", True, ["CUDA failed"])
        out = ui.apply_selection("default", "cuda", self.dir)
        self.assertTrue(out["degraded"])
        self.assertIn("WARNING", out["message"])
        self.assertIn("CUDA failed", out["message"])

    def test_unchanged_selection_reports_changed_false(self):
        self.recognition_model_name.return_value = "adaface"
        self.get_recognition_engine.return_value = _fake_engine("adaface", "cuda")
        self.set_recognition_model.return_value = _fake_engine("adaface", "cuda")
        self.assertFalse(ui.apply_selection("adaface", "cuda", self.dir)["changed"])

    def test_invalid_selection_never_reaches_the_engine(self):
        with self.assertRaises(ValueError):
            ui.apply_selection("nope", "cpu", self.dir)
        self.set_recognition_model.assert_not_called()


class _Cfg:
    def __init__(self):
        self.recognition_model, self.recognition_provider, self.saves = "default", "app", 0

    def save(self):
        self.saves += 1


class TestRoutes(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.cfg = _Cfg()
        app = FastAPI()
        app.include_router(rr.router)
        self.client = TestClient(app)
        for p in (mock.patch.object(rr.roop_globals, "CFG", self.cfg),
                  mock.patch.object(rr, "models_dir", return_value=self.dir),
                  mock.patch.object(face_analyser, "recognition_model_name", return_value=None),
                  mock.patch.object(ui, "probe_hardware", return_value=NO_GPU),
                  mock.patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"])):
            p.start()
            self.addCleanup(p.stop)
        rr.bind_progress({"processing": False})

    def test_status_has_everything_the_panel_draws(self):
        body = self.client.get("/api/recognition").json()
        self.assertEqual(body["selection"], {"model": "default", "provider": "app"})
        self.assertEqual(body["advice"]["tier"], ui.TIER_CPU)
        self.assertEqual(len(body["models"]), len(ui.RECOGNITION_REGISTRY))
        self.assertIsNone(body["active"])
        self.assertIn("Live swap matching", body["scope"])

    def test_apply_persists_only_after_a_successful_build(self):
        with mock.patch.object(face_analyser, "set_recognition_model", return_value=_fake_engine("glintr100", "cpu")):
            r = self.client.post("/api/recognition/apply", json={"model": "glintr100", "provider": "cpu"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual((self.cfg.recognition_model, self.cfg.recognition_provider, self.cfg.saves),
                         ("glintr100", "cpu", 1))
        self.assertEqual(r.json()["after"]["selection"], {"model": "glintr100", "provider": "cpu"})

    def test_failed_build_saves_nothing_and_says_so(self):
        with mock.patch.object(face_analyser, "set_recognition_model", side_effect=RuntimeError("hash mismatch")):
            r = self.client.post("/api/recognition/apply", json={"model": "adaface", "provider": "cpu"})
        self.assertEqual(r.status_code, 502)
        self.assertIn("hash mismatch", r.json()["message"])
        self.assertEqual((self.cfg.recognition_model, self.cfg.saves), ("default", 0))

    def test_bad_input_is_a_400_and_a_running_render_is_a_409(self):
        self.assertEqual(self.client.post("/api/recognition/apply", json={"model": "nope"}).status_code, 400)
        self.assertEqual(self.client.post("/api/recognition/apply", json={"provider": "tpu"}).status_code, 400)
        rr.bind_progress({"processing": True})
        with mock.patch.object(face_analyser, "set_recognition_model") as build:
            r = self.client.post("/api/recognition/apply", json={"model": "adaface"})
        self.assertEqual(r.status_code, 409)
        build.assert_not_called()
        self.assertEqual(self.cfg.saves, 0)


if __name__ == "__main__":
    unittest.main()
