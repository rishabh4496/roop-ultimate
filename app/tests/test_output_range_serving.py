"""HTTP 206 serving of output media, and the compare view's source endpoint.

The output player seeks by Range request, and three of the shapes a media
element sends were answered wrongly before this: a SUFFIX range (`bytes=-N`,
how a player fetches an MP4's trailing moov atom) came back as the first N+1
bytes, a range past EOF got 206 with the last byte instead of 416, and an
inverted range was "repaired". These pin the RFC 9110 behaviour, the
validators that make a re-opened player revalidate instead of re-download, and
that /api/output/source can serve only the file the server recorded.
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
import routes_output  # noqa: E402
from routes_output import parse_byte_range  # noqa: E402


class ParseByteRangeTests(unittest.TestCase):
    def test_absent_or_foreign_units_serve_the_whole_file(self):
        for h in (None, "", "items=0-1", "bytes", "bytes=abc-def", "bytes=5"):
            self.assertIsNone(parse_byte_range(h, 100), h)

    def test_closed_and_open_ranges(self):
        self.assertEqual(parse_byte_range("bytes=0-9", 100), (0, 9))
        self.assertEqual(parse_byte_range("bytes=90-", 100), (90, 99))
        self.assertEqual(parse_byte_range("bytes=90-500", 100), (90, 99))
        self.assertEqual(parse_byte_range("BYTES = 3-4", 100), (3, 4))

    def test_suffix_range_is_the_tail(self):
        self.assertEqual(parse_byte_range("bytes=-10", 100), (90, 99))
        self.assertEqual(parse_byte_range("bytes=-1000", 100), (0, 99))

    def test_unsatisfiable(self):
        self.assertEqual(parse_byte_range("bytes=100-", 100), "unsatisfiable")
        self.assertEqual(parse_byte_range("bytes=-0", 100), "unsatisfiable")
        self.assertEqual(parse_byte_range("bytes=0-", 0), "unsatisfiable")

    def test_inverted_range_is_ignored_not_repaired(self):
        self.assertIsNone(parse_byte_range("bytes=9-3", 100))

    def test_first_of_several_ranges(self):
        self.assertEqual(parse_byte_range("bytes=0-1, 5-6", 100), (0, 1))


class ServingTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.name = "range_probe.mp4"
        self.path = os.path.join(self.dir, self.name)
        self.body = bytes(range(256)) * 40          # 10,240 bytes
        with open(self.path, "wb") as f:
            f.write(self.body)
        self._orig_out = api.roop_globals.output_path
        api.roop_globals.output_path = self.dir
        self.addCleanup(setattr, api.roop_globals, "output_path", self._orig_out)
        self.client = TestClient(api.app)
        self.url = f"/outputs/{self.name}"

    def test_full_get_carries_validators(self):
        r = self.client.get(self.url)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content, self.body)
        self.assertEqual(r.headers["accept-ranges"], "bytes")
        self.assertTrue(r.headers["etag"].startswith('W/"'))
        self.assertIn("last-modified", r.headers)
        self.assertEqual(r.headers["cache-control"], "no-cache")

    def test_no_wildcard_cors(self):
        r = self.client.get(self.url)
        self.assertNotEqual(r.headers.get("access-control-allow-origin"), "*")

    def test_range_and_suffix(self):
        r = self.client.get(self.url, headers={"Range": "bytes=100-199"})
        self.assertEqual(r.status_code, 206)
        self.assertEqual(r.content, self.body[100:200])
        self.assertEqual(r.headers["content-range"], f"bytes 100-199/{len(self.body)}")
        r = self.client.get(self.url, headers={"Range": "bytes=-16"})
        self.assertEqual(r.status_code, 206)
        self.assertEqual(r.content, self.body[-16:])

    def test_past_eof_is_416(self):
        r = self.client.get(self.url, headers={"Range": f"bytes={len(self.body)}-"})
        self.assertEqual(r.status_code, 416)
        self.assertEqual(r.headers["content-range"], f"bytes */{len(self.body)}")

    def test_head_range(self):
        r = self.client.head(self.url, headers={"Range": "bytes=0-9"})
        self.assertEqual(r.status_code, 206)
        self.assertEqual(r.headers["content-length"], "10")

    def test_revalidation_is_304(self):
        etag = self.client.get(self.url).headers["etag"]
        r = self.client.get(self.url, headers={"If-None-Match": etag})
        self.assertEqual(r.status_code, 304)
        self.assertEqual(r.content, b"")

    def test_if_range_mismatch_serves_the_whole_new_file(self):
        r = self.client.get(self.url, headers={"Range": "bytes=0-9", "If-Range": 'W/"stale"'})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.content), len(self.body))
        etag = r.headers["etag"]
        r = self.client.get(self.url, headers={"Range": "bytes=0-9", "If-Range": etag})
        self.assertEqual(r.status_code, 206)

    def test_file_version_changes_with_contents(self):
        v1 = routes_output.file_version(self.path)
        with open(self.path, "ab") as f:
            f.write(b"more")
        self.assertNotEqual(routes_output.file_version(self.path), v1)
        self.assertEqual(routes_output.file_version(os.path.join(self.dir, "nope")), "")


class OutputSourceTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        saved = dict(routes_output._output_source)
        self.addCleanup(lambda: routes_output._output_source.update(saved))

    def test_404_without_a_recorded_source(self):
        routes_output._output_source["path"] = ""
        self.assertEqual(self.client.get("/api/output/source").status_code, 404)

    def test_serves_only_the_recorded_file_with_ranges(self):
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.write(fd, b"0123456789")
        os.close(fd)
        routes_output._output_source["path"] = path
        r = self.client.get("/api/output/source?path=C:/Windows/win.ini",
                            headers={"Range": "bytes=2-4"})
        self.assertEqual(r.status_code, 206)
        self.assertEqual(r.content, b"234", "a query path must not redirect the endpoint")

    def test_record_carries_version_and_source(self):
        from roop.ProcessEntry import ProcessEntry
        out_dir = tempfile.mkdtemp()
        with open(os.path.join(out_dir, "result.mp4"), "wb") as f:
            f.write(b"render")
        fd, src = tempfile.mkstemp(suffix=".mp4")
        os.write(fd, b"source")
        os.close(fd)
        orig = api.roop_globals.output_path
        saved = dict(api._last_output)
        self.addCleanup(setattr, api.roop_globals, "output_path", orig)
        self.addCleanup(lambda: (api._last_output.clear(), api._last_output.update(saved)))
        api.roop_globals.output_path = out_dir
        api._record_last_output(source_entry=ProcessEntry(src, 48, 600, 24.0))
        rec = api._last_output
        self.assertEqual(rec["name"], "result.mp4")
        self.assertTrue(rec["version"])
        self.assertEqual(rec["source"]["start_frame"], 48)
        self.assertEqual(rec["source"]["fps"], 24.0)
        self.assertTrue(rec["source"]["url"].startswith("/api/output/source?v="))
        self.assertNotIn(src, str(rec["source"]), "the absolute source path leaked to the client")
        self.assertEqual(routes_output._output_source["path"], src)
        # A multi-target run records no source rather than guessing.
        api._record_last_output(source_entry=None)
        self.assertIsNone(api._last_output["source"])
        self.assertEqual(routes_output._output_source["path"], "")


if __name__ == "__main__":
    unittest.main()
