"""model_integrity: resumable, SHA256-verified model downloads and the startup check."""

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from roop import model_integrity as mi  # noqa: E402

PAYLOAD = bytes(range(256)) * 40          # 10240 bytes
SHA = hashlib.sha256(PAYLOAD).hexdigest()
URL = "https://model.test/m.onnx"


class _Response(io.BytesIO):
    def __init__(self, body, status, headers):
        super().__init__(body)
        self.status = status
        self.headers = headers


class FakeServer:
    """urlopen stand-in: honours Range, can drop a connection after N bytes."""

    def __init__(self, payload=PAYLOAD, honour_range=True, cut_after=None, status=None):
        self.payload = payload
        self.honour_range = honour_range
        self.cut_after = list(cut_after or [])
        self.status = status
        self.ranges = []

    def __call__(self, request, timeout=None, context=None):
        if self.status:
            raise urllib.error.HTTPError(request.full_url, self.status, "err", {}, None)
        header = request.headers.get("Range")
        self.ranges.append(header)
        start = int(header.split("=")[1].rstrip("-")) if header and self.honour_range else 0
        if start >= len(self.payload) and header:
            raise urllib.error.HTTPError(request.full_url, 416, "range", {}, None)
        body = self.payload[start:]
        headers = {"Content-Length": str(len(body))}
        if self.cut_after:
            body = body[:self.cut_after.pop(0)]
        return _Response(body, 206 if (header and self.honour_range) else 200, headers)


def _entry(**kw):
    base = dict(file="m.onnx", sha256=SHA, size=len(PAYLOAD),
                urls=("https://model.test/m.onnx",))
    base.update(kw)
    return mi.ModelEntry(**base)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.dest = os.path.join(self.dir, "m.onnx")

    def test_resumes_with_range_after_a_dropped_connection(self):
        server = FakeServer(cut_after=[3000, 4000])
        mi.download_resumable(URL, self.dest, expected_size=len(PAYLOAD),
                              expected_sha256=SHA, opener=server, sleep=lambda s: None)
        self.assertEqual(Path(self.dest).read_bytes(), PAYLOAD)
        self.assertEqual(server.ranges, [None, "bytes=3000-", "bytes=7000-"])
        self.assertFalse(os.path.exists(self.dest + ".part"))

    def test_resumes_a_part_left_by_a_previous_process(self):
        Path(self.dest + ".part").write_bytes(PAYLOAD[:5000])
        server = FakeServer()
        mi.download_resumable(URL, self.dest, expected_size=len(PAYLOAD),
                              expected_sha256=SHA, opener=server, sleep=lambda s: None)
        self.assertEqual(server.ranges, ["bytes=5000-"])
        self.assertEqual(Path(self.dest).read_bytes(), PAYLOAD)

    def test_server_ignoring_range_restarts_from_zero(self):
        Path(self.dest + ".part").write_bytes(b"x" * 5000)
        mi.download_resumable(URL, self.dest, expected_size=len(PAYLOAD),
                              expected_sha256=SHA, opener=FakeServer(honour_range=False),
                              sleep=lambda s: None)
        self.assertEqual(Path(self.dest).read_bytes(), PAYLOAD)

    def test_complete_part_is_verified_without_refetching(self):
        Path(self.dest + ".part").write_bytes(PAYLOAD)
        server = FakeServer()
        mi.download_resumable(URL, self.dest, expected_size=len(PAYLOAD),
                              expected_sha256=SHA, opener=server)
        self.assertEqual(server.ranges, [])
        self.assertTrue(os.path.exists(self.dest))

    def test_hash_mismatch_never_becomes_the_model(self):
        bad = FakeServer(payload=b"y" * len(PAYLOAD))
        with self.assertRaisesRegex(mi.DownloadError, "sha256"):
            mi.download_resumable(URL, self.dest, expected_size=len(PAYLOAD),
                                  expected_sha256=SHA, opener=bad)
        self.assertFalse(os.path.exists(self.dest))
        self.assertFalse(os.path.exists(self.dest + ".part"))

    def test_404_is_not_retried(self):
        server = FakeServer(status=404)
        slept = []
        with self.assertRaises(mi.DownloadError):
            mi.download_resumable(URL, self.dest, opener=server, sleep=slept.append)
        self.assertEqual(slept, [])

    def test_progress_reports_bytes_and_total(self):
        seen = []
        mi.download_resumable(URL, self.dest, expected_size=len(PAYLOAD),
                              opener=FakeServer(), progress=lambda d, t: seen.append((d, t)))
        self.assertEqual(seen[-1], (len(PAYLOAD), len(PAYLOAD)))


class VerifyAndRepairTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "m.onnx")
        self.status = mi.IntegrityStatus()
        self.logs = []

    def _run(self, opener, download=True):
        return mi.verify_and_repair(self.dir, [_entry()], download=download,
                                    status=self.status, log=self.logs.append,
                                    opener=opener, sleep=lambda s: None)

    def test_missing_model_is_downloaded_and_status_is_ready(self):
        report = self._run(FakeServer())
        self.assertEqual(report["results"], {"m.onnx": "downloaded"})
        self.assertEqual(self.status.snapshot()["phase"], "ready")

    def test_good_model_is_hashed_once_then_cached(self):
        Path(self.path).write_bytes(PAYLOAD)
        self._run(FakeServer())
        cache = json.loads(Path(self.dir, mi.CACHE_NAME).read_text())
        self.assertEqual(cache["m.onnx"]["sha256"], SHA)
        calls = []
        original = mi.sha256_file
        mi.sha256_file = lambda *a, **k: calls.append(a) or original(*a, **k)
        try:
            self.assertEqual(self._run(FakeServer())["results"], {"m.onnx": "ok"})
        finally:
            mi.sha256_file = original
        self.assertEqual(calls, [])

    def test_corrupt_model_is_replaced(self):
        Path(self.path).write_bytes(b"truncated")
        report = self._run(FakeServer())
        self.assertEqual(report["results"], {"m.onnx": "repaired"})
        self.assertEqual(Path(self.path).read_bytes(), PAYLOAD)
        self.assertFalse(os.path.exists(self.path + ".corrupt"))

    def test_corrupt_model_is_kept_when_replacement_fails(self):
        Path(self.path).write_bytes(b"suspect but loadable")
        report = self._run(FakeServer(status=503))
        self.assertEqual(report["results"], {"m.onnx": "corrupt"})
        self.assertEqual(Path(self.path).read_bytes(), b"suspect but loadable")
        self.assertEqual(self.status.snapshot()["phase"], "degraded")

    def test_verify_only_mode_never_downloads(self):
        def refuse(*a, **k):
            raise AssertionError("health mode touched the network")
        self.assertEqual(self._run(refuse, download=False)["results"], {"m.onnx": "missing"})

    def test_offline_missing_model_degrades_without_raising(self):
        report = self._run(FakeServer(status=404))
        self.assertFalse(report["ok"])
        self.assertIn("m.onnx", self.status.snapshot()["message"])


class ManifestTests(unittest.TestCase):
    def test_manifest_covers_the_requested_families_with_pinned_hashes(self):
        entries = mi.load_manifest()
        self.assertTrue({"inswapper", "retinaface", "gpen", "bisenet"}
                        <= {e.family for e in entries})
        for e in entries:
            self.assertRegex(e.sha256, r"^[0-9a-f]{64}$", e.file)
            self.assertGreater(e.size, 0)
            self.assertTrue(e.urls and all(u.startswith("https://") for u in e.urls))
            self.assertTrue(all(u.endswith("/" + e.file) for u in e.urls), e.file)

    def test_manifest_names_the_files_the_loaders_open(self):
        # A manifest entry whose name drifted from the loader's constant would
        # verify a file nothing reads while the real one goes unchecked.
        sources = "".join(p.read_text(encoding="utf-8") for p in [
            APP / "roop" / "retinaface.py",
            APP / "roop" / "processors" / "Mask_FaceParser.py",
            APP / "roop" / "processors" / "Enhance_GPEN.py",
            APP / "roop" / "core.py",
        ])
        for e in mi.load_manifest():
            self.assertIn(e.file, sources)

    def test_startup_runs_the_integrity_check(self):
        source = (APP / "roop" / "core.py").read_text(encoding="utf-8")
        self.assertIn("model_integrity.verify_and_repair", source)


if __name__ == "__main__":
    unittest.main()
