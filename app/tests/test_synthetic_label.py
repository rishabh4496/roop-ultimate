"""Output labelling and the intended-use gate (NOTICE.md's asks, made real).

Audited 2026-09-22: app/roop had no content or consent safeguard of any kind.
These pin the three that now exist -- the metadata tag (default on), the
visible watermark (default off), and the first-run acceptance of NOTICE.md's
"Intended use" section, which /api/swap enforces as well as the UI.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
os.environ.setdefault('ROOP_REACT_CLIENT', '1')

import intended_use  # noqa: E402
from roop import synthetic_label as sl  # noqa: E402


def _ffmpeg():
    try:
        from roop.util_ffmpeg import ffmpeg_binary
        exe = ffmpeg_binary()
        return exe if exe and (os.path.isfile(exe) or shutil.which(exe)) else None
    except Exception:
        return shutil.which("ffmpeg")


class MetadataTag(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        import cv2
        self.cv2 = cv2
        self.img = np.full((64, 96, 3), 90, np.uint8)
        self.img[10:30, 10:40] = (30, 200, 60)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_png_gets_a_text_chunk_and_the_pixels_are_untouched(self):
        p = os.path.join(self.tmp, "out.png")
        self.cv2.imwrite(p, self.img)
        self.assertIsNone(sl.read_label(p))
        self.assertTrue(sl.label_file(p, "hello synthetic"))
        self.assertEqual(sl.read_label(p), "hello synthetic")
        back = self.cv2.imread(p)
        np.testing.assert_array_equal(back, self.img)      # lossless, byte-level insert

    def test_jpeg_gets_exif_description_and_a_comment_without_reencoding(self):
        p = os.path.join(self.tmp, "out.jpg")
        self.cv2.imwrite(p, self.img, [self.cv2.IMWRITE_JPEG_QUALITY, 90])
        before = self.cv2.imread(p)
        raw_before = open(p, "rb").read()
        self.assertTrue(sl.label_file(p, "hello synthetic"))
        raw_after = open(p, "rb").read()
        self.assertEqual(sl.read_label(p), "hello synthetic")
        self.assertIn(b"Exif\x00\x00", raw_after)
        self.assertIn(b"hello synthetic\x00", raw_after)   # EXIF ImageDescription
        # the entropy-coded image data is the same bytes, only shifted
        self.assertTrue(raw_after.endswith(raw_before[2:]))
        np.testing.assert_array_equal(self.cv2.imread(p), before)

    def test_jpeg_with_existing_exif_gets_only_a_comment(self):
        p = os.path.join(self.tmp, "out.jpg")
        self.cv2.imwrite(p, self.img)
        raw = open(p, "rb").read()
        fake_exif = b"\xff\xe1\x00\x08Exif\x00\x00"
        open(p, "wb").write(raw[:2] + fake_exif + raw[2:])
        self.assertTrue(sl.label_file(p, "x"))
        self.assertEqual(open(p, "rb").read().count(b"Exif\x00\x00"), 1)
        self.assertEqual(sl.read_label(p), "x")

    def test_unsupported_and_missing_files_are_reported_not_raised(self):
        gif = os.path.join(self.tmp, "a.gif")
        open(gif, "wb").write(b"GIF89a" + b"\x00" * 20)
        self.assertFalse(sl.label_file(gif))
        self.assertFalse(sl.label_file(os.path.join(self.tmp, "missing.png")))
        self.assertFalse(sl.label_file(""))
        self.assertFalse(sl.label_file(None))

    def test_a_corrupt_png_is_left_alone(self):
        p = os.path.join(self.tmp, "bad.png")
        open(p, "wb").write(b"not a png at all")
        self.assertFalse(sl.label_file(p))
        self.assertEqual(open(p, "rb").read(), b"not a png at all")

    @unittest.skipUnless(_ffmpeg(), "ffmpeg not available")
    def test_video_gets_the_tag_after_a_stream_copy_remux(self):
        p = os.path.join(self.tmp, "out.mp4")
        r = subprocess.run([_ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=64x64:r=10:d=0.5",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-metadata", "title=kept", p],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIsNone(sl.read_label(p))
        self.assertTrue(sl.label_file(p, "hello synthetic"))
        self.assertEqual(sl.read_label(p), "hello synthetic")
        # the other tags survive, and it was a copy, not a re-encode
        probe = subprocess.run([os.path.join(os.path.dirname(_ffmpeg()), "ffprobe"), "-v", "error",
                                "-show_entries", "format_tags=title:stream=codec_name", "-of", "csv=p=0", p],
                               capture_output=True, text=True)
        self.assertIn("h264", probe.stdout)
        self.assertIn("kept", probe.stdout)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "out.labelling.mp4")))


class Watermark(unittest.TestCase):
    def test_stamp_is_a_copy_confined_to_the_corner(self):
        frame = np.full((360, 640, 3), 120, np.uint8)
        out = sl.stamp_frame(frame, "AI face swap")
        self.assertIsNot(out, frame)
        np.testing.assert_array_equal(frame, 120)                   # original untouched
        changed = np.argwhere((out != 120).any(axis=2))
        self.assertGreater(len(changed), 0)
        ys, xs = changed[:, 0], changed[:, 1]
        self.assertGreater(ys.min(), 360 * 0.8)                    # bottom
        self.assertGreater(xs.min(), 640 * 0.5)                    # right
        self.assertLess(len(changed) / (360 * 640), 0.05)          # small

    def test_off_by_default_and_a_no_op_when_off(self):
        stub = types.SimpleNamespace(synthetic_watermark=False, synthetic_label=True)
        import roop.globals as g
        with mock.patch.object(g, "CFG", stub):
            frame = np.zeros((32, 32, 3), np.uint8)
            self.assertIs(sl.maybe_stamp(frame), frame)
            self.assertTrue(sl.label_enabled())
            stub.synthetic_watermark = True
            self.assertIsNot(sl.maybe_stamp(frame), frame)

    def test_settings_defaults(self):
        import settings
        s = settings.Settings(os.path.join(APP, "__no_such_settings_file__.yaml"))
        self.assertTrue(s.synthetic_label)
        self.assertFalse(s.synthetic_watermark)
        self.assertEqual(s.synthetic_watermark_text, "AI face swap")
        self.assertEqual(s.intended_use_acknowledged, "")
        keys = {k for k, _, _ in settings.UI_SETTINGS}
        self.assertTrue({"synthetic_label", "synthetic_watermark", "synthetic_watermark_text"} <= keys)


class IntendedUseGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import api
        import api_access
        from fastapi.testclient import TestClient
        cls.api = api
        api_access.set_policy(api_access.AccessPolicy(share=False))
        cls.client = TestClient(api.app)

    @classmethod
    def tearDownClass(cls):
        import api_access
        api_access.set_policy(None)

    def _cfg(self, ack=""):
        return types.SimpleNamespace(intended_use_acknowledged=ack, save=lambda: None)

    def test_terms_come_from_notice_md(self):
        text = intended_use.terms_text()
        self.assertIn("informed consent", text)
        self.assertIn("label", text)
        self.assertEqual(len(intended_use.terms_version()), 16)
        with open(intended_use.NOTICE_PATH, encoding="utf-8") as fh:
            self.assertIn(text.splitlines()[0], fh.read())

    def test_get_terms_reports_state(self):
        import roop.globals as g
        with mock.patch.object(g, "CFG", self._cfg("")):
            r = self.client.get("/api/terms").json()
            self.assertFalse(r["acknowledged"])
            self.assertEqual(r["version"], intended_use.terms_version())
        with mock.patch.object(g, "CFG", self._cfg(intended_use.terms_version())):
            self.assertTrue(self.client.get("/api/terms").json()["acknowledged"])

    def test_swap_is_refused_until_accepted(self):
        import roop.globals as g
        with mock.patch.object(g, "CFG", self._cfg("")), \
                mock.patch.object(self.api, "_configuration_ready", return_value=True):
            r = self.client.post("/api/swap", json={})
            self.assertEqual(r.status_code, 403)
            self.assertTrue(r.json()["terms_required"])

    def test_acknowledge_requires_the_shown_version_and_an_explicit_accept(self):
        import roop.globals as g
        cfg = self._cfg("")
        saved = []
        cfg.save = lambda: saved.append(True)
        with mock.patch.object(g, "CFG", cfg):
            self.assertEqual(self.client.post("/api/terms/acknowledge", json={"version": "stale", "accept": True}).status_code, 409)
            self.assertEqual(self.client.post("/api/terms/acknowledge", json={"version": intended_use.terms_version()}).status_code, 400)
            self.assertEqual(cfg.intended_use_acknowledged, "")
            r = self.client.post("/api/terms/acknowledge", json={"version": intended_use.terms_version(), "accept": True})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(cfg.intended_use_acknowledged, intended_use.terms_version())
            self.assertTrue(saved)
            self.assertTrue(self.client.get("/api/terms").json()["acknowledged"])

    def test_changed_terms_ask_again(self):
        stale = self._cfg(intended_use.terms_version())
        with mock.patch.object(intended_use, "terms_text", return_value="new words"):
            self.assertFalse(intended_use.acknowledged(stale))


if __name__ == "__main__":
    unittest.main()
