"""Recognition model registry: integrity of the table and of the resolver.

Offline -- the network is replaced by a fake `requests.get`. The one thing this file
cannot prove is that the live URLs still serve the pinned bytes; that was checked by
downloading every entry when the registry was written (see the module docstring).
"""

import hashlib
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests                                          # noqa: E402
from roop import recognition_registry as reg             # noqa: E402


class _FakeResponse:
    def __init__(self, payload, content_length=None, status=200):
        self._payload = payload
        self.headers = {"content-length": str(len(payload) if content_length is None else content_length)}
        self._status = status

    def raise_for_status(self):
        if self._status >= 400:
            raise requests.HTTPError(f"{self._status}")

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._payload), 7):          # odd chunking on purpose
            yield self._payload[i:i + 7]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestRegistryTable(unittest.TestCase):
    def test_every_spec_is_well_formed(self):
        for key in reg.get_registered_models():
            s = reg.get_model_spec(key)
            self.assertEqual(s.name, key)
            self.assertRegex(s.sha256, r"^[0-9a-f]{64}$", key)
            self.assertTrue(s.url.startswith("https://") and s.filename, key)
            self.assertFalse(os.path.isabs(s.filename), key)
            self.assertIn(s.color_space, ("RGB", "BGR"))
            self.assertEqual((len(s.input_size), len(s.mean), len(s.std)), (2, 3, 3))
            self.assertTrue(all(v > 0 for v in s.std), key)

    def test_filenames_are_safe_relative_paths(self):
        for s in reg.RECOGNITION_REGISTRY.values():
            self.assertNotIn("..", s.filename.replace("\\", "/").split("/"))

    def test_same_filename_means_same_bytes(self):
        """antelopev2/glintr100 share one file; two specs may never disagree about it."""
        seen = {}
        for s in reg.RECOGNITION_REGISTRY.values():
            self.assertEqual(seen.setdefault(s.filename, (s.sha256, s.url)), (s.sha256, s.url), s.filename)

    def test_unknown_model_names_the_valid_ones(self):
        with self.assertRaises(ValueError) as e:
            reg.get_model_spec("nope")
        self.assertIn("default", str(e.exception))

    def test_default_and_adaface_match_the_paths_the_app_reads(self):
        """Registry must reuse what is already on disk, not fork a second copy."""
        self.assertEqual(reg.get_model_spec("default").filename, "buffalo_l/w600k_r50.onnx")
        try:
            from roop import recognizer_adaface as ada
        except ImportError as e:                         # light profile: no cv2/onnxruntime
            self.skipTest(f"recognizer_adaface not importable here: {e}")
        spec = reg.get_model_spec("adaface")
        self.assertEqual(spec.filename, ada.MODEL_FILE)
        self.assertEqual(spec.url, ada.MODEL_URL)


class TestResolver(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        p = mock.patch.object(reg, "_verified", {})
        p.start()
        self.addCleanup(p.stop)

    def _use_spec(self, payload, sha=None, filename="m.onnx"):
        spec = reg.RecognitionModelSpec(
            name="t", display_name="t", url="http://example.invalid/m.onnx", filename=filename,
            sha256=sha or hashlib.sha256(payload).hexdigest())
        p = mock.patch.dict(reg.RECOGNITION_REGISTRY, {"t": spec})
        p.start()
        self.addCleanup(p.stop)

    def _serve(self, response):
        p = mock.patch.object(reg.requests, "get", lambda *a, **k: response)
        p.start()
        self.addCleanup(p.stop)

    def _path(self, *parts):
        return os.path.join(self.dir, *parts)

    def test_existing_matching_file_is_reused_without_network(self):
        payload = b"model-bytes"
        self._use_spec(payload)
        with open(self._path("m.onnx"), "wb") as f:
            f.write(payload)
        with mock.patch.object(reg.requests, "get",
                               side_effect=AssertionError("network used for a verified file")):
            self.assertEqual(reg.resolve_model_path("t", self.dir), os.path.abspath(self._path("m.onnx")))

    def test_missing_file_is_downloaded_and_verified(self):
        payload = os.urandom(1000)
        self._use_spec(payload, filename="sub/m.onnx")
        self._serve(_FakeResponse(payload))
        out = reg.resolve_model_path("t", self.dir)
        self.assertEqual(out, os.path.abspath(self._path("sub", "m.onnx")))
        with open(out, "rb") as f:
            self.assertEqual(f.read(), payload)
        self.assertEqual(os.listdir(self._path("sub")), ["m.onnx"])      # no .part left

    def test_hash_mismatch_raises_and_leaves_nothing(self):
        self._use_spec(b"x", sha="0" * 64)
        self._serve(_FakeResponse(b"served-bytes"))
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            reg.resolve_model_path("t", self.dir)
        self.assertEqual(os.listdir(self.dir), [])

    def test_bad_download_does_not_destroy_the_existing_file(self):
        payload = b"good"
        self._use_spec(payload)
        with open(self._path("m.onnx"), "wb") as f:
            f.write(b"user-supplied")
        self._serve(_FakeResponse(b"also-bad"))
        with self.assertRaises(RuntimeError):
            reg.resolve_model_path("t", self.dir)
        with open(self._path("m.onnx"), "rb") as f:
            self.assertEqual(f.read(), b"user-supplied")
        self.assertEqual(os.listdir(self.dir), ["m.onnx"])

        self._serve(_FakeResponse(payload))                  # a good download replaces it
        reg.resolve_model_path("t", self.dir)
        with open(self._path("m.onnx"), "rb") as f:
            self.assertEqual(f.read(), payload)

    def test_truncated_download_is_rejected(self):
        payload = b"0123456789" * 10
        self._use_spec(payload)
        self._serve(_FakeResponse(payload, content_length=len(payload) + 50))
        with self.assertRaisesRegex(RuntimeError, "Truncated"):
            reg.resolve_model_path("t", self.dir)
        self.assertEqual(os.listdir(self.dir), [])

    def test_http_error_is_a_runtimeerror(self):
        self._use_spec(b"x")
        self._serve(_FakeResponse(b"", status=404))
        with self.assertRaisesRegex(RuntimeError, "Could not download"):
            reg.resolve_model_path("t", self.dir)


if __name__ == "__main__":
    unittest.main()
