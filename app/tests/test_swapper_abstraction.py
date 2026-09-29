"""BaseFaceSwapper contract, ModelRegistry, and swap_model flag routing.

Three layers:
  * light (numpy only): registry resolution, tensor validation, provider names,
    the abstract contract;
  * the real ONNX graphs: topology read with onnx.load, static-shape refusals;
  * equivalence: the registry's HiFiFace / HyperSwap swappers against the
    production FaceSwapInsightFace.Run on the same crop and identity, on CPU.
    Bit-identical is the bar -- this layer is a refactor of that path, not a
    second implementation of it.
Model-file tests skip when the file is absent (the 3060 and CI).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from roop.processors.frame import model_registry as mr
from roop.processors.frame.swapper_base import (BaseFaceSwapper, TensorSpec,
                                                resolve_providers, validate_tensor)

MODELS = APP_DIR / "models"
HIFIFACE = MODELS / "hififace_unofficial_256.onnx"
HIFI_CONVERTER = MODELS / "crossface_hififace.onnx"
HYPERSWAP_1A = MODELS / "hyperswap_1a_256.onnx"

needs_hififace = pytest.mark.skipif(
    not (HIFIFACE.is_file() and HIFI_CONVERTER.is_file()), reason="hififace model files absent")
needs_hyperswap = pytest.mark.skipif(not HYPERSWAP_1A.is_file(), reason="hyperswap_1a absent")


# ── Registry ─────────────────────────────────────────────────────────────────

class TestRegistry:
    @pytest.mark.parametrize("name, cls, spec", [
        ("hififace_256", "HiFiFaceSwapper", "hififace"),
        ("hyperswap_1a_256", "HyperSwapSwapper", "hyperswap"),
        ("hyperswap_1b_256", "HyperSwapSwapper", "hyperswap_1b"),
        ("hyperswap_1c_256", "HyperSwapSwapper", "hyperswap_1c"),
    ])
    def test_required_names(self, name, cls, spec):
        entry = mr.DEFAULT_REGISTRY.entry(name)
        assert entry is not None
        assert entry.class_path.endswith(":" + cls)
        assert entry.spec_key == spec

    @pytest.mark.parametrize("alias, name", [
        ("hififace", "hififace_256"), ("HiFiFace_256", "hififace_256"),
        ("hyperswap", "hyperswap_1a_256"), ("hyperswap_1b", "hyperswap_1b_256"),
        (" hyperswap_1c ", "hyperswap_1c_256"),
    ])
    def test_aliases_and_case(self, alias, name):
        assert mr.DEFAULT_REGISTRY.entry(alias).name == name

    def test_unknown_name(self):
        assert mr.DEFAULT_REGISTRY.entry("nope") is None
        assert "nope" not in mr.DEFAULT_REGISTRY
        with pytest.raises(KeyError):
            mr.DEFAULT_REGISTRY.resolve("nope")

    def test_register_is_dynamic_and_lazy(self):
        reg = mr.ModelRegistry()

        class Dummy(BaseFaceSwapper):
            def __init__(self, spec_key=None):
                super().__init__()
                self.spec_key = spec_key
            def initialize_session(self, model_path, execution_provider, **kw): self.session = object()
            def pre_process(self, target_crop, source_embedding): return {}
            def infer(self, inputs): return np.zeros((1, 3, 4, 4), np.float32)
            def post_process(self, swap_crop, affine_matrix, target_frame, mask): return target_frame

        reg.register("dummy_4", Dummy, spec_key="dummy", aliases=("dmy",))
        made = reg.create("DMY")
        assert isinstance(made, Dummy) and made.spec_key == "dummy"
        # A dotted path is not imported until something resolves it.
        reg.register("ghost_later", "module_that_does_not_exist:Nothing")
        assert "ghost_later" in reg
        with pytest.raises(ModuleNotFoundError):
            reg.resolve("ghost_later")

    def test_alias_collision_refused(self):
        reg = mr.ModelRegistry()
        reg.register("a", "m:A", aliases=("shared",))
        with pytest.raises(ValueError):
            reg.register("b", "m:B", aliases=("shared",))
        reg.register("b", "m:B", aliases=("shared",), replace=True)
        assert reg.entry("shared").name == "b"

    def test_non_swapper_class_refused(self):
        reg = mr.ModelRegistry()
        reg.register("notaswapper", "collections:OrderedDict")
        with pytest.raises(TypeError):
            reg.resolve("notaswapper")


# ── swap_model flag routing: zero breaking changes ────────────────────────────

KNOWN = ("inswapper", "reswapper", "hyperswap", "hyperswap_1b", "hyperswap_1c",
         "hififace", "realswap", "simswap", "ghost_1")


class TestFlagRouting:
    @pytest.mark.parametrize("key", KNOWN)
    def test_existing_keys_pass_through(self, key):
        assert mr.canonical_swap_model(key) == key
        assert mr.resolve_swap_model_key(key, KNOWN) == key

    @pytest.mark.parametrize("name, key", [
        ("hififace_256", "hififace"), ("hyperswap_1a_256", "hyperswap"),
        ("hyperswap_1b_256", "hyperswap_1b"), ("hyperswap_1c_256", "hyperswap_1c"),
    ])
    def test_new_names_route_to_spec_keys(self, name, key):
        assert mr.resolve_swap_model_key(name, KNOWN) == key

    @pytest.mark.parametrize("junk", [None, "", "not_a_model", 7])
    def test_unknown_falls_back_like_before(self, junk):
        assert mr.canonical_swap_model(junk) == junk      # untouched
        assert mr.resolve_swap_model_key(junk, KNOWN) == "inswapper"

    def test_against_the_real_table(self):
        pytest.importorskip("onnx")
        from roop.processors.FaceSwapInsightFace import SWAP_MODELS
        for key in SWAP_MODELS:
            assert mr.resolve_swap_model_key(key) == key
        for name in mr.DEFAULT_REGISTRY.names():
            assert mr.resolve_swap_model_key(name) in SWAP_MODELS

    def test_initialize_routes_through_registry(self):
        # The processor must resolve its flag with the registry, not its own
        # `in SWAP_MODELS` check (which would send hififace_256 to inswapper).
        src = (APP_DIR / "roop" / "processors" / "FaceSwapInsightFace.py").read_text(encoding="utf-8")
        body = src[src.index("    def Initialize(self, plugin_options: dict):"):]
        body = body[:body.index("spec = SWAP_MODELS[swap_model]")]
        assert "resolve_swap_model_key(" in body
        assert "if swap_model not in SWAP_MODELS" not in body

    def test_get_processing_plugins(self):
        pytest.importorskip("torch")
        import roop.globals
        from roop.core import get_processing_plugins
        old = roop.globals.selected_enhancer
        roop.globals.selected_enhancer = "None"
        try:
            for flag, want in (("hififace_256", "hififace"), ("hyperswap_1b_256", "hyperswap_1b"),
                               ("realswap", "realswap"), ("inswapper", "inswapper")):
                plugins = get_processing_plugins(None, swap_model=flag)
                assert plugins["faceswap"]["swap_model"] == want
        finally:
            roop.globals.selected_enhancer = old


# ── Tensor validation ─────────────────────────────────────────────────────────

class TestValidateTensor:
    SPEC = TensorSpec("target", ("batch_size", 3, 256, 256))

    def test_accepts_and_is_contiguous(self):
        out = validate_tensor(np.zeros((1, 3, 256, 256), np.float32), self.SPEC, batch=1)
        assert out.flags["C_CONTIGUOUS"]

    def test_non_contiguous_is_repaired_not_refused(self):
        base = np.random.rand(1, 256, 256, 3).astype(np.float32)
        view = base.transpose(0, 3, 1, 2)
        assert not view.flags["C_CONTIGUOUS"]
        out = validate_tensor(view, self.SPEC)
        assert out.flags["C_CONTIGUOUS"] and np.array_equal(out, view)

    @pytest.mark.parametrize("arr", [
        np.zeros((1, 3, 255, 256), np.float32),      # wrong static spatial
        np.zeros((1, 4, 256, 256), np.float32),      # wrong channels
        np.zeros((3, 256, 256), np.float32),         # wrong rank
        np.zeros((1, 3, 256, 256), np.float64),      # wrong dtype, never cast
        np.zeros((0, 3, 256, 256), np.float32),      # empty symbolic batch
    ])
    def test_refusals(self, arr):
        with pytest.raises(ValueError):
            validate_tensor(arr, self.SPEC)

    def test_batch_pin(self):
        with pytest.raises(ValueError):
            validate_tensor(np.zeros((2, 3, 256, 256), np.float32), self.SPEC, batch=1)

    def test_static_batch_of_one(self):
        spec = TensorSpec("target", (1, 3, 256, 256))
        with pytest.raises(ValueError):
            validate_tensor(np.zeros((2, 3, 256, 256), np.float32), spec)


class TestProviders:
    def test_aliases(self):
        assert resolve_providers("cuda") == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert resolve_providers("TensorRT")[0] == "TensorrtExecutionProvider"
        assert resolve_providers("CPUExecutionProvider") == ["CPUExecutionProvider"]
        assert resolve_providers("CUDAExecutionProvider")[-1] == "CPUExecutionProvider"
        lst = [("TensorrtExecutionProvider", {"a": 1}), "CPUExecutionProvider"]
        assert resolve_providers(lst) == lst

    def test_unknown(self):
        with pytest.raises(ValueError):
            resolve_providers("quantum")


class TestContract:
    def test_abstract(self):
        with pytest.raises(TypeError):
            BaseFaceSwapper()

        class Partial(BaseFaceSwapper):
            def initialize_session(self, model_path, execution_provider, **kw): pass
        with pytest.raises(TypeError):
            Partial()

    def test_to_crop_matches_normalize_swap_frame(self):
        pytest.importorskip("torch")
        from roop.procmgr_tiling import PixelBoostMixin

        class P:
            model_denormalize = True
        out = np.random.uniform(-1.1, 1.1, (3, 64, 64)).astype(np.float32)

        class S(BaseFaceSwapper):
            model_denormalize = True
            initialize_session = pre_process = infer = post_process = None
        S.__abstractmethods__ = frozenset()
        want = PixelBoostMixin().normalize_swap_frame(out, P()).astype(np.uint8)
        assert np.array_equal(S().to_crop(out), want)
        assert np.array_equal(S().to_crop(out[None]), want)


# ── Paste ───────────────────────────────────────────────────────────────────────

class TestPasteBack:
    def setup_method(self):
        pytest.importorskip("cv2")

    def test_zero_mask_leaves_frame(self):
        frame = np.random.randint(0, 255, (120, 160, 3), np.uint8)
        crop = np.full((32, 32, 3), 200, np.uint8)
        M = np.array([[1, 0, -40], [0, 1, -30]], np.float32)   # crop origin at (40,30)
        out = BaseFaceSwapper.paste_back(crop, M, frame, np.zeros((32, 32), np.float32))
        assert np.array_equal(out, frame)
        assert out is not frame

    def test_full_mask_places_crop(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        crop = np.full((32, 32, 3), 200, np.uint8)
        M = np.array([[1, 0, -40], [0, 1, -30]], np.float32)
        out = BaseFaceSwapper.paste_back(crop, M, frame, np.ones((1, 1, 32, 32), np.float32))
        assert (out[31:61, 41:71] == 200).all()
        assert (out[:25] == 0).all() and (out[:, :35] == 0).all()

    @pytest.mark.parametrize("bad", ["frame", "affine", "mask"])
    def test_refusals(self, bad):
        frame = np.zeros((50, 50, 3), np.uint8)
        crop = np.zeros((16, 16, 3), np.uint8)
        M = np.eye(2, 3, dtype=np.float32)
        mask = np.ones((16, 16), np.float32)
        if bad == "frame":
            frame = frame.astype(np.float32)
        elif bad == "affine":
            M = np.eye(3, dtype=np.float32)
        else:
            mask = np.ones((8, 8), np.float32)
        with pytest.raises(ValueError):
            BaseFaceSwapper.paste_back(crop, M, frame, mask)


# ── The real graphs ───────────────────────────────────────────────────────────

@pytest.mark.gpu
class TestOnnxTopology:
    @needs_hififace
    def test_hififace(self):
        from roop.processors.frame.swapper_base import inspect_onnx_topology
        ins, outs = inspect_onnx_topology(str(HIFIFACE))
        got = {s.name: s.shape for s in ins}
        assert got == {"target": ("batch_size", 3, 256, 256), "source": ("batch_size", 512)}
        assert all(np.dtype(s.dtype) == np.float32 for s in ins + outs)
        assert [(s.name, s.shape[1:]) for s in outs] == [("output", (3, 256, 256)),
                                                         ("mask", (1, 256, 256))]

    @needs_hyperswap
    def test_hyperswap(self):
        from roop.processors.frame.swapper_base import inspect_onnx_topology
        ins, outs = inspect_onnx_topology(str(HYPERSWAP_1A))
        assert [(s.name, s.shape) for s in ins] == [("source", (1, 512)),
                                                    ("target", (1, 3, 256, 256))]
        assert outs[0].shape == (1, 3, 256, 256)

    @needs_hififace
    def test_class_refuses_wrong_graph(self):
        from roop.processors.frame.onnx_swappers import HiFiFaceSwapper
        # The converter is a [1,512] -> [1,512] graph: no image input at all.
        with pytest.raises(ValueError, match="expected one"):
            HiFiFaceSwapper().initialize_session(str(HIFI_CONVERTER), "cpu")

    def test_class_refuses_foreign_spec(self):
        pytest.importorskip("onnx")
        from roop.processors.frame.onnx_swappers import HiFiFaceSwapper, OnnxSpecSwapper
        with pytest.raises(ValueError):
            HiFiFaceSwapper(spec_key="hyperswap")
        with pytest.raises(ValueError, match="not an embedding"):
            OnnxSpecSwapper(spec_key="blendswap")
        with pytest.raises(ValueError, match="composite"):
            OnnxSpecSwapper(spec_key="realswap")


# ── Equivalence with the production processor ─────────────────────────────────

def _production(swapper):
    """A FaceSwapInsightFace wired to the SAME session, skipping Initialize's
    TensorRT/pool/warm-up machinery, so Run's own latent + infer path is what
    runs (the batch-fallback tests drive it the same way)."""
    import threading
    from roop.processors.FaceSwapInsightFace import FaceSwapInsightFace
    p = FaceSwapInsightFace()
    p.model_swap_insightface = swapper.session
    p.loaded_model_key = swapper.spec_key
    p.embedding_mode = swapper.embedding_mode
    p.converter = swapper.converter
    p.converter_input = swapper.converter_input
    p.emap = swapper.emap
    p.image_input_name = swapper.image_input_name
    p.embed_input_name = swapper.embed_input_name
    p.model_has_mask = True
    p._mask_tls = threading.local()
    return p


def _source_face(embedding):
    from insightface.app.common import Face
    return Face(embedding=embedding)


@pytest.mark.gpu
class TestEquivalence:
    @pytest.fixture(autouse=True)
    def _cpu_only(self, monkeypatch):
        monkeypatch.setenv("ROOP_ORT_IO_BINDING", "0")

    def _check(self, name, rng):
        sw = mr.create_swapper(name)
        sw.initialize_session(None, "cpu")
        crop = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        emb = rng.normal(0, 1, 512).astype(np.float32) * 20.0     # raw ArcFace-scale
        feed = sw.pre_process(crop, emb)
        for name_, arr in feed.items():
            assert arr.flags["C_CONTIGUOUS"] and arr.dtype == np.float32
            assert arr.shape[0] == 1 and name_ in sw.input_specs
        out = sw.infer(feed)
        assert out.shape == (1, 3, 256, 256)
        assert sw.last_mask is not None and sw.last_mask.shape == (1, 1, 256, 256)

        prod = _production(sw)
        face = _source_face(emb)
        np.testing.assert_array_equal(sw.prepare_latent(emb), prod._compute_latent(face))
        from roop.procmgr_tiling import to_blob
        want = prod.Run(face, None, to_blob(crop, sw.model_mean, sw.model_standard_deviation))
        np.testing.assert_array_equal(out[0], want)
        # production stashes the mask squeezed to (S,S)
        np.testing.assert_array_equal(prod.take_masks()[0], sw.last_mask[0, 0])

        # post_process: the paste of that same crop, whole frame out.
        frame = rng.integers(0, 256, (400, 300, 3), dtype=np.uint8)
        M = np.array([[0.8, 0.0, -20.0], [0.0, 0.8, -60.0]], np.float32)
        pasted = sw.post_process(out, M, frame, sw.last_mask)
        assert pasted.shape == frame.shape and pasted.dtype == np.uint8
        assert not np.array_equal(pasted, frame)
        return sw

    @needs_hififace
    def test_hififace_matches_production(self):
        sw = self._check("hififace_256", np.random.default_rng(1))
        assert sw.model_template == "mtcnn_512"
        # batch-dynamic graph; the contract still pins pre_process to one crop
        crop = np.zeros((256, 256, 3), np.uint8)
        with pytest.raises(ValueError):
            sw.pre_process(crop[:255], np.ones(512, np.float32))
        with pytest.raises(ValueError):
            sw.pre_process(crop.astype(np.float32), np.ones(512, np.float32))
        with pytest.raises(ValueError):
            sw.pre_process(crop, np.ones(511, np.float32))
        with pytest.raises(ValueError):
            sw.pre_process(crop, np.zeros(512, np.int64))

    @needs_hyperswap
    def test_hyperswap_matches_production(self):
        sw = self._check("hyperswap_1a_256", np.random.default_rng(2))
        assert sw.model_template == "arcface" and sw.model_denormalize
        feed = sw.pre_process(np.zeros((256, 256, 3), np.uint8), np.ones(512, np.float32))
        feed[sw.image_input_name] = np.concatenate([feed[sw.image_input_name]] * 2)
        feed[sw.embed_input_name] = np.concatenate([feed[sw.embed_input_name]] * 2)
        with pytest.raises(ValueError, match="static 1"):
            sw.infer(feed)          # the export is fixed at batch 1

    @needs_hyperswap
    def test_infer_repairs_non_contiguous_feed(self):
        sw = mr.create_swapper("hyperswap_1a_256")
        sw.initialize_session(str(HYPERSWAP_1A), "CPUExecutionProvider")
        feed = sw.pre_process(np.full((256, 256, 3), 90, np.uint8), np.ones(512, np.float32))
        want = sw.infer(dict(feed))
        feed[sw.image_input_name] = np.asfortranarray(feed[sw.image_input_name])
        assert not feed[sw.image_input_name].flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(sw.infer(feed), want)

    def test_uninitialized(self):
        pytest.importorskip("onnx")
        sw = mr.create_swapper("hififace_256")
        assert not sw.is_initialized
        with pytest.raises(RuntimeError):
            sw.pre_process(np.zeros((256, 256, 3), np.uint8), np.ones(512, np.float32))
