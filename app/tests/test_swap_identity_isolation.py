"""The swapper's identity input is isolated from the tracking recogniser.

Tracking may run any registered model (face_analyser.set_recognition_model); the swappers must
only ever see the 512-d buffalo_l/w600k_r50 vector on face.embedding. Three things enforce it:

  1. roop/swap_identity.validate_identity_embedding at the swapper's identity entry points
     (size, finiteness, zero vector) -- tested here on the real cache and the real converter path.
  2. An import ratchet: nothing in app/ outside the recognition modules may import the tracking
     API. This is what closes the case a shape check cannot -- a 512-d vector from the WRONG
     model (AdaFace, Glint-R100) -- so adding such an import must be a deliberate edit of
     ALLOWED_RECOGNITION_IMPORTERS, not a side effect.
  3. The latent is computed once per source face (cache hit on every later frame).
"""
import ast
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.swap_identity import SWAP_EMBEDDING_DIM, validate_identity_embedding  # noqa: E402

try:
    from roop.hyperswap_optimizer import HyperSwapSourceCache
    from roop.processors.FaceSwapInsightFace import FaceSwapInsightFace
    _IMPORT_ERROR = None
except ImportError as exc:                       # light profile
    _IMPORT_ERROR = exc

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Face(dict):
    """insightface.Face-like: item and attribute access, `embedding` as an attribute."""

    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding


def _vec(seed=0, n=SWAP_EMBEDDING_DIM):
    return np.random.RandomState(seed).randn(n).astype(np.float32)


class TestValidate(unittest.TestCase):
    def test_accepts_512_in_any_container_and_returns_a_float32_row(self):
        for value in (_vec(), _vec().reshape(1, 512), _vec().tolist(), _vec().astype(np.float64)):
            out = validate_identity_embedding(value)
            self.assertEqual((out.shape, out.dtype, out.flags["C_CONTIGUOUS"]), ((1, 512), np.float32, True))

    def test_rejects_other_widths_and_names_the_hazard(self):
        for n in (128, 256, 511, 513, 1024):
            with self.assertRaisesRegex(ValueError, r"%d values.*never reach the swapper" % n):
                validate_identity_embedding(_vec(n=n), "src")

    def test_rejects_stacked_vectors(self):
        with self.assertRaises(ValueError):
            validate_identity_embedding(np.stack([_vec(1), _vec(2)]))

    def test_rejects_non_finite_zero_and_non_numeric(self):
        bad = np.full(512, np.nan, np.float32)
        inf = _vec()
        inf[3] = np.inf
        for value, pattern in ((bad, "NaN/Inf"), (inf, "NaN/Inf"), (np.zeros(512), "zero vector"),
                               (np.full(512, 1e-9), "zero vector"), (None, "no identity embedding"), ("abc", "not numeric")):
            with self.assertRaisesRegex(ValueError, pattern):
                validate_identity_embedding(value, "src")

    def test_message_names_the_caller(self):
        with self.assertRaisesRegex(ValueError, "^source face for hyperswap_1a"):
            validate_identity_embedding(_vec(n=128), "source face for hyperswap_1a")


@unittest.skipIf(_IMPORT_ERROR is not None, "swapper stack not importable here")
class TestSourceCache(unittest.TestCase):
    def test_latent_is_computed_once_per_source_face(self):
        cache, face = HyperSwapSourceCache(), _Face(_vec(1))
        first = cache.get_latent(face, "hyperswap_1a", "normed")
        for _ in range(50):                                         # 50 later frames
            again = cache.get_latent(face, "hyperswap_1a", "normed")
        self.assertIs(again, first)
        self.assertEqual((cache.misses, cache.hits), (1, 50))
        self.assertEqual(first.shape, (1, 512))
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1.0, places=5)

    def test_same_embedding_on_a_new_face_object_still_hits_the_lru(self):
        cache = HyperSwapSourceCache()
        cache.get_latent(_Face(_vec(2)), "m", "normed")
        cache.get_latent(_Face(_vec(2)), "m", "normed")
        self.assertEqual((cache.misses, cache.hits), (1, 1))

    def test_emap_mode_projects_and_renormalises(self):
        cache, emap = HyperSwapSourceCache(), np.random.RandomState(3).randn(512, 512).astype(np.float32)
        v = _vec(4)
        out = cache.get_latent(_Face(v), "inswapper_128", "normed_emap", emap=emap)
        expect = (v / np.linalg.norm(v)) @ emap
        np.testing.assert_allclose(out[0], expect / np.linalg.norm(expect), atol=1e-5)

    def test_wrong_width_is_refused_and_nothing_is_cached(self):
        cache = HyperSwapSourceCache()
        face = _Face(_vec(n=128))
        with self.assertRaisesRegex(ValueError, "128 values"):
            cache.get_latent(face, "hyperswap_1a", "normed")
        self.assertEqual((len(cache._cache), cache.misses), (0, 0))
        self.assertNotIn("_latent_hyperswap_1a", face)

    def test_zero_embedding_no_longer_becomes_a_cached_zero_latent(self):
        """It used to: a swap toward nobody, kept in the cache, reported as a success."""
        cache, face = HyperSwapSourceCache(), _Face(np.zeros(512, np.float32))
        with self.assertRaisesRegex(ValueError, "zero vector"):
            cache.get_latent(face, "hyperswap_1a", "normed")
        self.assertEqual(len(cache._cache), 0)
        self.assertNotIn("_latent_hyperswap_1a", face)

    def test_missing_embedding_still_raises_as_before(self):
        with self.assertRaisesRegex(ValueError, "no embedding"):
            HyperSwapSourceCache().get_latent(_Face(None), "m", "normed")

    def test_switching_the_tracking_recogniser_cannot_change_the_latent(self):
        """Behavioural half of the isolation: drive the tracking API to a 128-d model and back."""
        from roop import face_analyser as fa
        cache, face = HyperSwapSourceCache(), _Face(_vec(5))
        before = cache.get_latent(face, "hyperswap_1a", "normed").copy()

        class _Engine:                                       # a 128-d tracking model (SFace-like)
            spec = type("S", (), {"output_dim": 128, "input_size": (112, 112)})()
            model_name = "facerecognizersf"

            def compute_embedding(self, crop):
                return np.ones(128, np.float32) / np.sqrt(128), 1.0

        with mock.patch.object(fa, "get_recognition_engine", return_value=_Engine()):
            vec, _ = fa.extract_face_embedding(np.zeros((112, 112, 3), np.uint8),
                                               np.array([[38, 51], [73, 51], [56, 71], [41, 92], [70, 92]], np.float32))
        self.assertEqual(vec.shape, (128,))
        self.assertTrue(np.array_equal(face.embedding, _vec(5)))                  # the Face is untouched
        fresh = HyperSwapSourceCache().get_latent(_Face(_vec(5)), "hyperswap_1a", "normed")
        np.testing.assert_array_equal(fresh, before)                              # bit-identical
        with self.assertRaises(ValueError):                                       # and the 128-d result cannot be fed in
            HyperSwapSourceCache().get_latent(_Face(vec), "hyperswap_1a", "normed")


@unittest.skipIf(_IMPORT_ERROR is not None, "swapper stack not importable here")
class TestSwapperPaths(unittest.TestCase):
    def _swapper(self, mode, key="ghost_1_256"):
        sw = FaceSwapInsightFace()
        sw.embedding_mode, sw.loaded_model_key = mode, key
        sw.emap = np.eye(512, dtype=np.float32)
        sw.converter = mock.Mock()
        sw.converter.run.side_effect = lambda _o, feed: [np.asarray(feed["input"], np.float32) * 2.0]
        return sw

    def test_converter_path_validates_before_the_converter_runs(self):
        sw = self._swapper("converted_norm")
        for bad in (_vec(n=128), np.zeros(512, np.float32)):
            with self.assertRaises(ValueError):
                sw._compute_latent(_Face(bad))
        sw.converter.run.assert_not_called()

    def test_converter_path_runs_once_per_source_face_and_caches_on_the_face(self):
        sw, face = self._swapper("converted_norm"), _Face(_vec(6))
        a = sw._compute_latent(face)
        b = sw._compute_latent(face)
        self.assertIs(a, b)
        self.assertEqual(sw.converter.run.call_count, 1)
        self.assertEqual(a.shape, (1, 512))
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=5)

    def test_default_inswapper_path_uses_the_shared_cache_and_rejects_bad_vectors(self):
        sw = self._swapper("normed_emap", "inswapper_128")
        from roop.hyperswap_optimizer import get_hyperswap_source_cache
        get_hyperswap_source_cache().clear()
        self.addCleanup(get_hyperswap_source_cache().clear)
        latent = sw._compute_latent(_Face(_vec(7)))
        self.assertEqual(latent.shape, (1, 512))
        with self.assertRaisesRegex(ValueError, "128 values"):
            sw._compute_latent(_Face(_vec(n=128)))


# ---------------------------------------------------------------- the import ratchet

# Names of the tracking-side recognition API (face_analyser) and the modules behind it.
RECOGNITION_NAMES = frozenset({
    "set_recognition_model", "get_recognition_engine", "release_recognition_engine",
    "recognition_model_name", "extract_face_embedding", "IdentityBank", "fuse_quality_weighted",
})
RECOGNITION_MODULES = frozenset({"roop.recognition_engine", "roop.recognition_registry", "roop.ui_recognition"})

# The only production files allowed to touch it (app-relative, '/' separators). Adding a file here
# is the explicit decision to route a recogniser into the pipeline: read roop/swap_identity.py
# first, and calibrate the model's match threshold before any gate uses it.
ALLOWED_RECOGNITION_IMPORTERS = frozenset({
    "roop/face_analyser.py",          # defines the API; imports engine/registry lazily
    "roop/recognition_engine.py",
    "roop/recognition_registry.py",
    "roop/ui_recognition.py",
    "roop/worker_pool.py",            # tracking-side recipe for process pools; no render uses it
    "routes_recognition.py",
})


def recognition_references(source: str):
    """Names/modules of the tracking API that `source` imports or touches (AST, not grep)."""
    tree = ast.parse(source)
    hits, aliases = [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in RECOGNITION_MODULES:
                hits.append(module)
            if module == "roop":
                for a in node.names:
                    if a.name in ("recognition_engine", "recognition_registry", "ui_recognition"):
                        hits.append("roop." + a.name)
                    if a.name == "face_analyser":
                        aliases.add(a.asname or a.name)
            if module == "roop.face_analyser":
                hits += [a.name for a in node.names if a.name in RECOGNITION_NAMES]
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name in RECOGNITION_MODULES:
                    hits.append(a.name)
                if a.name == "roop.face_analyser":
                    aliases.add(a.asname or "roop.face_analyser")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in RECOGNITION_NAMES:
            base = node.value
            dotted = base.id if isinstance(base, ast.Name) else (
                "roop.face_analyser" if isinstance(base, ast.Attribute) and base.attr == "face_analyser" else None)
            if dotted in aliases:
                hits.append(node.attr)
    return sorted(set(hits))


def _production_files():
    for root, dirs, files in os.walk(APP):
        dirs[:] = [d for d in dirs if d not in ("tests", "env", "tools", "node_modules", "__pycache__",
                                                 "models", "docs", "assets", "outputs")]
        if os.path.abspath(root) == APP or os.path.relpath(root, APP).split(os.sep)[0] == "roop":
            for name in files:
                if name.endswith(".py"):
                    yield os.path.relpath(os.path.join(root, name), APP).replace(os.sep, "/")


class TestImportRatchet(unittest.TestCase):
    def test_scanner_sees_each_way_of_reaching_the_api(self):
        """A scanner that finds nothing would pass for the wrong reason."""
        cases = {
            "from roop.face_analyser import extract_face_embedding, get_all_faces": ["extract_face_embedding"],
            "from roop.recognition_engine import RecognitionInferenceEngine": ["roop.recognition_engine"],
            "import roop.recognition_registry": ["roop.recognition_registry"],
            "from roop import ui_recognition": ["roop.ui_recognition"],
            "from roop import face_analyser as fa\nfa.set_recognition_model('x')": ["set_recognition_model"],
            "import roop.face_analyser\nroop.face_analyser.IdentityBank()": ["IdentityBank"],
            "from roop.face_analyser import get_all_faces, FaceTracker": [],
            "from roop.face_clustering import extract_face_embedding": [],       # a different function
        }
        for source, expected in cases.items():
            self.assertEqual(recognition_references(source), expected, source)

    def test_scanner_walks_the_production_tree(self):
        files = set(_production_files())
        self.assertIn("roop/ProcessMgr.py", files)
        self.assertIn("roop/processors/FaceSwapInsightFace.py", files)
        self.assertIn("api.py", files)
        self.assertGreater(len(files), 150)

    def test_only_the_recognition_modules_touch_the_tracking_api(self):
        offenders = {}
        found_allowed = set()
        for rel in _production_files():
            with open(os.path.join(APP, rel), encoding="utf-8") as fh:
                try:
                    refs = recognition_references(fh.read())
                except SyntaxError:
                    continue
            if refs and rel in ALLOWED_RECOGNITION_IMPORTERS:
                found_allowed.add(rel)
            elif refs:
                offenders[rel] = refs
        self.assertEqual(offenders, {},
                         "production code now reaches the tracking recognition API. The swapper must keep "
                         "using the 512-d w600k vector on face.embedding; read roop/swap_identity.py, and if "
                         "this wiring is intended add the file to ALLOWED_RECOGNITION_IMPORTERS.")
        self.assertIn("roop/ui_recognition.py", found_allowed)
        self.assertIn("routes_recognition.py", found_allowed)

    def test_the_swapper_modules_in_particular_never_import_it(self):
        for rel in ("roop/processors/FaceSwapInsightFace.py", "roop/hyperswap_optimizer.py",
                    "roop/processors/face_swapper.py", "roop/swap_identity.py"):
            path = os.path.join(APP, rel)
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as fh:
                    self.assertEqual(recognition_references(fh.read()), [], rel)


if __name__ == "__main__":
    unittest.main()
