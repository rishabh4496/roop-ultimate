"""Every endpoint that takes a path, a filename or an upload stays inside the
app's own folders.

Reviewed 2026-09-22: /api/file and /outputs/ used abspath, so a symlink inside
the output folder reached anything on disk; /api/reveal opened ANY path;
/api/output/delete followed symlinks; uploads went to disk under the client's
basename with no extension, content, size or count check; /api/target/add_path
took UNC paths (opening one sends the machine's credentials to that host).
safe_paths is now the one boundary; these pin it, through the real ASGI app
where the endpoint is cheap enough to drive without loading a model.
"""

import io
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
os.environ.setdefault('ROOP_REACT_CLIENT', '1')

import safe_paths  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64
MKV = b"\x1aE\xdf\xa3" + b"\x00" * 64
ZIP = b"PK\x03\x04" + b"\x00" * 64
WAV = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 64
TEXT = b"hello, not media at all\n" * 4

if sys.platform == "win32":
    ABS_ELSEWHERE = r"C:\Windows\win.ini"
    OTHER_DRIVE = r"Z:\anything\at\all.png"
    UNC = r"\\attacker.example\share\evil.png"
else:
    ABS_ELSEWHERE = "/etc/passwd"
    OTHER_DRIVE = "/anything/at/all.png"
    UNC = "//attacker.example/share/evil.png"


def _write(path, data=PNG):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _link_dir(link, target):
    """Directory symlink, or a junction on Windows without the privilege."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        if sys.platform != "win32":
            return False
        r = subprocess.run(["cmd", "/c", "mklink", "/J", link, target], capture_output=True)
        return r.returncode == 0 and os.path.isdir(link)


def _link_file(link, target):
    try:
        os.symlink(target, link)
        return True
    except (OSError, NotImplementedError):
        return False


class Sanitize(unittest.TestCase):
    def test_filename_is_never_a_path(self):
        cases = {
            "../../etc/passwd": "passwd",
            "..\\..\\Windows\\win.ini": "win.ini",
            "C:\\Users\\x\\a.PNG": "a.png",
            "\\\\server\\share\\a.png": "a.png",
            "/abs/dir/b.jpg": "b.jpg",
            "..": "upload",
            ".": "upload",
            "": "upload",
            None: "upload",
            ".hidden.png": "hidden.png",
            "CON.png": "CON_file.png",
            "nul": "nul_file",
            "a\x00b.png": "ab.png",
            "we ird$name;rm -rf.png": "we ird_name_rm -rf.png",
        }
        for raw, want in cases.items():
            with self.subTest(raw=raw):
                got = safe_paths.sanitize_filename(raw)
                self.assertEqual(got, want)
                self.assertNotIn("/", got)
                self.assertNotIn("\\", got)
                self.assertNotIn("..", got)

    def test_long_name_is_capped(self):
        got = safe_paths.sanitize_filename("x" * 500 + ".png")
        self.assertLessEqual(len(got), safe_paths.MAX_NAME)
        self.assertTrue(got.endswith(".png"))


class Confine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = os.path.join(self.tmp, "root")
        self.outside = os.path.join(self.tmp, "outside")
        self.inside = _write(os.path.join(self.root, "sub", "a.png"))
        self.secret = _write(os.path.join(self.outside, "secret.png"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_inside_is_returned_canonical(self):
        got = safe_paths.confine(self.inside, roots=[self.root])
        self.assertEqual(got, safe_paths.canonical(self.inside))
        # `..` that stays inside is fine
        self.assertIsNotNone(safe_paths.confine(os.path.join(self.root, "sub", "..", "sub", "a.png"), roots=[self.root]))

    def test_traversal_absolute_drive_and_unc_are_refused(self):
        bad = [
            os.path.join(self.root, "..", "outside", "secret.png"),   # ../
            os.path.join(self.root, "sub", "..", "..", "outside", "secret.png"),
            self.secret,                                                # absolute elsewhere
            ABS_ELSEWHERE,
            OTHER_DRIVE,                                                # another drive letter
            UNC,                                                        # UNC
            "\\\\?\\" + self.secret if sys.platform == "win32" else "//" + self.secret,
            "",
            "   ",
            None,
            self.root + "_sibling/x.png",                               # prefix trick: root_sibling
        ]
        for path in bad:
            with self.subTest(path=path):
                self.assertIsNone(safe_paths.confine(path, roots=[self.root]))

    def test_symlink_out_of_the_root_is_refused(self):
        link = os.path.join(self.root, "escape")
        if not _link_dir(link, self.outside):
            self.skipTest("cannot create a directory link here")
        self.assertTrue(os.path.isfile(os.path.join(link, "secret.png")))  # it IS reachable on disk
        self.assertIsNone(safe_paths.confine(os.path.join(link, "secret.png"), roots=[self.root]))
        flink = os.path.join(self.root, "flink.png")
        if _link_file(flink, self.secret):
            self.assertIsNone(safe_paths.confine_file(flink, roots=[self.root]))

    def test_dangling_and_missing_are_refused(self):
        self.assertIsNone(safe_paths.confine(os.path.join(self.root, "nope.png"), roots=[self.root]))
        self.assertIsNone(safe_paths.confine_file(os.path.join(self.root, "sub"), roots=[self.root]))  # a dir is not a file

    def test_is_unc(self):
        self.assertTrue(safe_paths.is_unc(UNC))
        self.assertTrue(safe_paths.is_unc("//host/share"))
        self.assertFalse(safe_paths.is_unc(self.inside))


class Magic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_content_must_match_extension(self):
        ok = [("a.png", PNG), ("a.jpg", JPG), ("a.mp4", MP4), ("a.mkv", MKV), ("a.fsz", ZIP), ("a.wav", WAV)]
        bad = [("a.png", TEXT), ("a.mp4", PNG), ("a.png", MP4), ("a.fsz", PNG), ("a.wav", TEXT),
               ("a.exe", b"MZ" + b"\x00" * 64), ("a.txt", TEXT), ("a", PNG)]
        for name, data in ok:
            with self.subTest(name=name):
                self.assertTrue(safe_paths.check_magic(_write(os.path.join(self.tmp, name), data)))
        for name, data in bad:
            with self.subTest(name=name):
                self.assertFalse(safe_paths.check_magic(_write(os.path.join(self.tmp, name), data)))


def _upload(name, data):
    return types.SimpleNamespace(filename=name, file=io.BytesIO(data))


class SaveUpload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dest = os.path.join(self.tmp, "uploads")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _nothing_outside_dest(self):
        stray = [p for p in os.listdir(self.tmp) if p != "uploads"]
        self.assertEqual(stray, [], "an upload escaped the upload folder")

    def test_traversal_names_land_in_the_upload_folder(self):
        for name in ("../../evil.png", "..\\..\\evil.png", "C:\\x\\evil.png", "\\\\h\\s\\evil.png", "/etc/evil.png"):
            with self.subTest(name=name):
                path = safe_paths.save_upload(_upload(name, PNG), self.dest, ("image",))
                self.assertEqual(os.path.dirname(safe_paths.canonical(path)), safe_paths.canonical(self.dest))
                self.assertTrue(os.path.basename(path).startswith("evil"))
        self._nothing_outside_dest()

    def test_same_name_never_overwrites(self):
        a = safe_paths.save_upload(_upload("a.png", PNG), self.dest, ("image",))
        b = safe_paths.save_upload(_upload("a.png", PNG), self.dest, ("image",))
        self.assertNotEqual(a, b)
        self.assertTrue(os.path.isfile(a) and os.path.isfile(b))

    def test_wrong_kind_content_empty_and_oversize_are_refused_and_removed(self):
        with self.assertRaises(safe_paths.UploadRejected):
            safe_paths.save_upload(_upload("a.mp4", MP4), self.dest, ("image",))       # kind not accepted here
        with self.assertRaises(safe_paths.UploadRejected):
            safe_paths.save_upload(_upload("a.exe", b"MZ" * 40), self.dest, ("image", "video"))
        with self.assertRaises(safe_paths.UploadRejected):
            safe_paths.save_upload(_upload("a.png", TEXT), self.dest, ("image",))       # magic mismatch
        with self.assertRaises(safe_paths.UploadRejected):
            safe_paths.save_upload(_upload("a.png", b""), self.dest, ("image",))        # empty
        with mock.patch.dict(safe_paths.UPLOAD_LIMITS, {"image": (32, 200)}):
            with self.assertRaises(safe_paths.UploadRejected) as cm:
                safe_paths.save_upload(_upload("big.png", PNG + b"\x00" * 100), self.dest, ("image",))
            self.assertIn("limit", cm.exception.detail)
        self.assertEqual(os.listdir(self.dest), [], "a refused upload left a file behind")

    def test_streams_in_chunks_not_whole(self):
        # The source is read in CHUNK pieces; a single read of the whole body
        # would show up as one call.
        src = mock.MagicMock()
        body = [PNG + b"\x00" * (safe_paths.CHUNK - len(PNG)), b"\x01" * 10, b""]
        src.read.side_effect = body
        path = safe_paths.save_upload(types.SimpleNamespace(filename="s.png", file=src), self.dest, ("image",))
        self.assertEqual(src.read.call_count, 3)
        self.assertEqual(os.path.getsize(path), safe_paths.CHUNK + 10)

    def test_count_cap(self):
        with mock.patch.dict(safe_paths.UPLOAD_LIMITS, {"image": (1, 2)}):
            with self.assertRaises(safe_paths.UploadRejected):
                safe_paths.check_count([1, 2, 3], ("image",))
            safe_paths.check_count([1, 2], ("image",))


class Endpoints(unittest.TestCase):
    """Through the real app. output_path points at a temp folder for the test;
    the model-loading paths are never reached (every request here is refused
    before them, or serves a static file)."""

    @classmethod
    def setUpClass(cls):
        import api
        import api_access
        import roop.globals as roop_globals
        from fastapi.testclient import TestClient
        cls.api, cls.roop_globals = api, roop_globals
        api_access.set_policy(api_access.AccessPolicy(share=False))
        cls.client = TestClient(api.app)

    @classmethod
    def tearDownClass(cls):
        import api_access
        api_access.set_policy(None)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.out = os.path.join(self.tmp, "output")
        self.elsewhere = os.path.join(self.tmp, "elsewhere")
        self.result = _write(os.path.join(self.out, "result.png"))
        self.secret = _write(os.path.join(self.elsewhere, "secret.png"))
        self._saved_out = getattr(self.roop_globals, "output_path", None)
        self.roop_globals.output_path = self.out

    def tearDown(self):
        self.roop_globals.output_path = self._saved_out
        shutil.rmtree(self.tmp, ignore_errors=True)

    # /api/file ------------------------------------------------------------
    def test_file_serves_output_and_refuses_everything_else(self):
        self.assertEqual(self.client.get("/api/file", params={"path": "result.png"}).status_code, 200)
        self.assertEqual(self.client.get("/api/file", params={"path": self.result}).status_code, 200)
        for bad in ("../elsewhere/secret.png", "..\\elsewhere\\secret.png", self.secret, ABS_ELSEWHERE,
                    OTHER_DRIVE, UNC, os.path.join(APP, "api.py"), "/outputs/../elsewhere/secret.png"):
            with self.subTest(path=bad):
                self.assertEqual(self.client.get("/api/file", params={"path": bad}).status_code, 403)

    def test_file_refuses_a_symlink_out_of_the_output_folder(self):
        link = os.path.join(self.out, "escape")
        if not _link_dir(link, self.elsewhere):
            self.skipTest("cannot create a directory link here")
        self.assertEqual(self.client.get("/api/file", params={"path": "escape/secret.png"}).status_code, 403)
        self.assertEqual(self.client.get("/api/file", params={"path": os.path.join(link, "secret.png")}).status_code, 403)
        self.assertEqual(self.client.get("/outputs/escape/secret.png").status_code, 404)

    def test_outputs_route_traversal(self):
        self.assertEqual(self.client.get("/outputs/result.png").status_code, 200)
        self.assertEqual(self.client.get("/outputs/%2e%2e/elsewhere/secret.png").status_code, 404)
        self.assertEqual(self.client.get("/outputs/..%5celsewhere%5csecret.png").status_code, 404)

    # /api/reveal ----------------------------------------------------------
    def test_reveal_only_opens_allowed_folders(self):
        with mock.patch.object(self.api._routes_output.subprocess, "Popen") as popen, \
                mock.patch.object(self.api._routes_output.os, "startfile", create=True) as startfile:
            for bad in (self.elsewhere, self.secret, ABS_ELSEWHERE, UNC, os.path.join(self.out, "..", "elsewhere")):
                with self.subTest(path=bad):
                    self.assertEqual(self.client.post("/api/reveal", json={"path": bad}).status_code, 403)
            popen.assert_not_called()
            startfile.assert_not_called()
            self.assertEqual(self.client.post("/api/reveal", json={"path": self.result}).status_code, 200)
            self.assertEqual(self.client.post("/api/reveal", json={}).status_code, 200)

    # /api/output/delete ---------------------------------------------------
    def test_delete_cannot_leave_the_output_folder(self):
        for bad in ("../elsewhere/secret.png", "..\\elsewhere\\secret.png", self.secret, UNC, "..", "."):
            with self.subTest(name=bad):
                self.assertEqual(self.client.post("/api/output/delete", json={"name": bad}).status_code, 404)
        self.assertTrue(os.path.isfile(self.secret))
        link = os.path.join(self.out, "linked.png")
        if _link_file(link, self.secret):
            self.assertEqual(self.client.post("/api/output/delete", json={"name": "linked.png"}).status_code, 404)
            self.assertTrue(os.path.isfile(self.secret))
        self.assertEqual(self.client.post("/api/output/delete", json={"name": "result.png"}).status_code, 200)
        self.assertFalse(os.path.exists(self.result))

    # /api/storage/delete --------------------------------------------------
    def test_storage_delete_takes_ids_not_paths(self):
        # item_id is looked up in a fresh server-side inventory; a path is not
        # an id. The inventory scan walks the whole project, so stub it.
        import routes_storage
        with mock.patch.object(routes_storage.MANAGER, "scan", return_value={"items": []}):
            for bad in ("../../app/api.py", self.secret, UNC):
                r = self.client.post("/api/storage/delete", json={"item_id": bad, "confirm": True})
                self.assertGreaterEqual(r.status_code, 400)
        self.assertTrue(os.path.isfile(self.secret))

    # /api/target/add_path -------------------------------------------------
    def test_add_path_rejects_unc_and_wrong_content(self):
        text_png = _write(os.path.join(self.elsewhere, "fake.png"), TEXT)
        txt = _write(os.path.join(self.elsewhere, "notes.txt"), TEXT)
        r = self.client.post("/api/target/add_path", json={"paths": [UNC, text_png, txt, os.path.join(self.tmp, "missing.png"), 42]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["added"], [])
        whys = [x["why"] for x in r.json()["rejected"]]
        self.assertEqual(len(whys), 5)
        self.assertIn("UNC", whys[0])
        self.assertIn("not a supported", whys[1])
        self.assertIn("not a supported", whys[2])
        self.assertIn("not a file", whys[3])

    def test_add_path_is_confined_in_share_mode(self):
        import api_access
        api_access.set_policy(api_access.AccessPolicy(share=True, token="t"))
        try:
            r = self.client.post("/api/target/add_path", json={"paths": [self.secret]},
                                 headers={"Authorization": "Bearer t"})
            self.assertEqual(r.json()["added"], [])
            self.assertIn("outside the allowed folders", r.json()["rejected"][0]["why"])
        finally:
            api_access.set_policy(api_access.AccessPolicy(share=False))

    # uploads --------------------------------------------------------------
    def _upload_dir_and_parent(self):
        up = self.api.API_TEMP
        return up, os.path.dirname(up)

    def test_uploads_are_sanitized_and_checked(self):
        up, parent = self._upload_dir_and_parent()
        before = set(os.listdir(parent)) if os.path.isdir(parent) else set()
        # traversal name + wrong content: refused, and nothing lands anywhere
        r = self.client.post("/api/target/add", files=[("files", ("../../evil.mp4", TEXT, "video/mp4"))])
        self.assertEqual(r.status_code, 400)
        self.assertIn("does not look like", r.json()["detail"])
        r = self.client.post("/api/target/add", files=[("files", ("tool.exe", b"MZ" * 40, "application/octet-stream"))])
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/lipsync/audio/add", files=[("file", ("song.mp4", MP4, "video/mp4"))])
        self.assertEqual(r.status_code, 400)   # audio endpoint, video kind
        r = self.client.post("/api/source/add-folder", files=[("files", ("../x.png", TEXT, "image/png"))])
        self.assertEqual(r.status_code, 400)
        # source/add reports per file instead of failing the batch
        r = self.client.post("/api/source/add", files=[("files", ("../../evil.png", TEXT, "image/png")),
                                                       ("files", ("bad.fsz", PNG, "application/zip"))])
        self.assertEqual(r.status_code, 200)
        self.assertEqual([e["error"] for e in r.json().get("errors", [])], ["rejected", "rejected"])
        self.assertEqual(set(os.listdir(parent)) - before if os.path.isdir(parent) else set(), set())
        for name in ("evil.mp4", "evil.png", "x.png", "tool.exe", "song.mp4", "bad.fsz"):
            self.assertFalse(os.path.exists(os.path.join(parent, name)))
            self.assertFalse(os.path.exists(os.path.join(up, name)))

    def test_upload_count_cap(self):
        with mock.patch.dict(safe_paths.UPLOAD_LIMITS, {"video": (1024, 2), "image": (1024, 2)}):
            files = [("files", (f"c{i}.mp4", MP4, "video/mp4")) for i in range(3)]
            r = self.client.post("/api/target/add", files=files)
        self.assertGreaterEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
