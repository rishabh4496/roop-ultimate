"""Offline/connected behavior for network-backed optional resources."""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import utilities  # noqa: E402


class OfflineOperationTests(unittest.TestCase):
    def setUp(self):
        utilities.reset_online_state()

    def tearDown(self):
        utilities.reset_online_state()

    def test_connected_probe_is_cached_briefly(self):
        socket = mock.patch("socket.create_connection")
        with socket as create_connection:
            create_connection.return_value.__enter__.return_value = object()
            self.assertTrue(utilities.is_online(timeout=0.01, hosts=("model.test",)))
            self.assertTrue(utilities.is_online(timeout=0.01, hosts=("model.test",)))
            create_connection.assert_called_once_with(("model.test", 443), timeout=0.01)

    def test_disconnected_probe_fails_closed(self):
        with mock.patch("socket.create_connection", side_effect=OSError("offline")):
            self.assertFalse(utilities.is_online(timeout=0.01, hosts=("model.test",)))

    def test_existing_model_does_not_probe_or_need_network(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "already-present.onnx"
            model.write_bytes(b"local model")
            with mock.patch.object(utilities, "is_online",
                                   side_effect=AssertionError("local model probed network")):
                utilities.conditional_download(directory, [
                    "https://model.test/already-present.onnx"
                ])

    def test_missing_optional_model_is_skipped_when_disconnected(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(utilities, "is_online", return_value=False), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            utilities.conditional_download(directory, [
                "https://model.test/optional.onnx"
            ], required=False)
            self.assertIn("[OFFLINE]", output.getvalue())
            self.assertFalse((Path(directory) / "optional.onnx").exists())

    def test_missing_required_model_reports_recoverable_offline_error(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(utilities, "is_online", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "not available locally"):
                utilities.conditional_download(directory, [
                    "https://model.test/required.onnx"
                ])

    def test_download_commits_atomically_after_connected_transfer(self):
        response = mock.MagicMock()
        response.headers = {"Content-Length": "10"}
        response.__enter__.return_value = response

        def write_partial(_url, path, reporthook=None):
            Path(path).write_bytes(b"0123456789")

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(utilities, "is_online", return_value=True), \
                mock.patch.object(utilities.urllib.request, "urlopen",
                                   return_value=response), \
                mock.patch.object(utilities.urllib.request, "urlretrieve",
                                   side_effect=write_partial):
            utilities.conditional_download(directory, [
                "https://model.test/connected.onnx"
            ])
            self.assertEqual((Path(directory) / "connected.onnx").read_bytes(), b"0123456789")
            self.assertFalse((Path(directory) / "connected.onnx.part").exists())


class OfflineUiContractTests(unittest.TestCase):
    def test_hud_does_not_claim_unmeasured_internet_or_runtime_values(self):
        source = (Path(__file__).resolve().parents[2] / "react-ui" / "src" / "App.jsx").read_text(
            encoding="utf-8")
        self.assertNotIn("Online (0.2 ms latency)", source)
        self.assertIn("Local engine connected", source)
        self.assertIn("Local engine unavailable", source)
        self.assertIn("hudValue", source)

    def test_startup_precheck_does_not_gate_local_models_on_global_internet(self):
        source = (Path(__file__).resolve().parents[1] / "roop" / "core.py").read_text(
            encoding="utf-8")
        self.assertNotIn("elif util.is_online()", source)
        self.assertIn("util.conditional_download", source)


if __name__ == "__main__":
    unittest.main()
