"""Test and verify backend server static serving, HTTP 206 range requests, web URLs, and CORS."""
import os
import sys
import tempfile
import json
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")
sys.path.insert(0, APP)

from fastapi.testclient import TestClient
import roop.globals as roop_globals
from api import app

class TestBackendServerEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.tmpdir = tempfile.TemporaryDirectory()
        roop_globals.output_path = cls.tmpdir.name

        # Create a sample MP4 file of 1000 bytes in the output folder
        cls.test_filename = "sample_render.mp4"
        cls.test_filepath = os.path.join(cls.tmpdir.name, cls.test_filename)
        with open(cls.test_filepath, "wb") as f:
            f.write(b"0123456789" * 100) # 1000 bytes

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_dedicated_outputs_endpoint(self):
        """1. Server exposes dedicated /outputs/{filename} static endpoint."""
        resp = self.client.get(f"/outputs/{self.test_filename}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.content), 1000)
        self.assertEqual(resp.headers.get("accept-ranges"), "bytes")
        self.assertIn("video/mp4", resp.headers.get("content-type", ""))

    def test_api_media_alias_endpoint(self):
        """1b. Server exposes /api/media/{filename} alias."""
        resp = self.client.get(f"/api/media/{self.test_filename}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.content), 1000)
        self.assertEqual(resp.headers.get("accept-ranges"), "bytes")

    def test_http_206_byte_range_request(self):
        """2. HTTP 206 Partial Content for smooth video scrubbing."""
        # Request bytes 0-99
        headers = {"Range": "bytes=0-99"}
        resp = self.client.get(f"/outputs/{self.test_filename}", headers=headers)
        self.assertEqual(resp.status_code, 206)
        self.assertEqual(len(resp.content), 100)
        self.assertEqual(resp.content, b"0123456789" * 10)
        self.assertEqual(resp.headers.get("content-range"), "bytes 0-99/1000")
        self.assertEqual(resp.headers.get("content-length"), "100")
        self.assertEqual(resp.headers.get("accept-ranges"), "bytes")

        # Request bytes 500-
        headers2 = {"Range": "bytes=500-"}
        resp2 = self.client.get(f"/outputs/{self.test_filename}", headers=headers2)
        self.assertEqual(resp2.status_code, 206)
        self.assertEqual(len(resp2.content), 500)
        self.assertEqual(resp2.headers.get("content-range"), "bytes 500-999/1000")
        self.assertEqual(resp2.headers.get("content-length"), "500")

    def test_api_file_resolves_outputs_path_with_206(self):
        """2b. /api/file supports web URL path and byte range."""
        headers = {"Range": "bytes=10-29"}
        resp = self.client.get(f"/api/file?path=/outputs/{self.test_filename}", headers=headers)
        self.assertEqual(resp.status_code, 206)
        self.assertEqual(len(resp.content), 20)
        self.assertEqual(resp.headers.get("content-range"), "bytes 10-29/1000")

    def test_cors_preflight_and_range_headers(self):
        """3. CORS permits Origin, Methods, Range, and exposes Range headers."""
        # Preflight OPTIONS request
        cors_headers = {
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Range, Content-Type",
        }
        resp = self.client.options(f"/outputs/{self.test_filename}", headers=cors_headers)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("access-control-allow-origin", resp.headers)
        allow_headers = resp.headers.get("access-control-allow-headers", "").lower()
        self.assertTrue("range" in allow_headers or "*" in allow_headers)

        # GET request CORS headers
        get_resp = self.client.get(f"/outputs/{self.test_filename}", headers={"Origin": "http://localhost:5173"})
        self.assertIn("access-control-allow-origin", get_resp.headers)
        expose_headers = get_resp.headers.get("access-control-expose-headers", "").lower()
        self.assertIn("content-range", expose_headers)
        self.assertIn("accept-ranges", expose_headers)

    def test_web_url_response_payload(self):
        """4. Response payload returns web-accessible URL path."""
        from api import _record_last_output, _last_output
        _record_last_output()
        self.assertTrue(_last_output["path"].startswith("/outputs/"), f"Path {_last_output['path']} is not web-accessible")
        self.assertTrue(_last_output["url"].startswith("/outputs/"), f"URL {_last_output['url']} is not web-accessible")
        self.assertEqual(_last_output["name"], self.test_filename)
        self.assertEqual(_last_output["absolute_path"], self.test_filepath)

if __name__ == "__main__":
    unittest.main()
