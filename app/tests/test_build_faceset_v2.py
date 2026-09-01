"""A V2 archive is not automatically a working identity-detail source.

`format_version == 2` is the check the handoff named, and on its own it is not
sufficient: `faceset_v2.migrate_legacy_fsz` produces a perfectly valid V2
archive whose `identity_details` are empty, so the feature stays inert while
every version assertion passes. These contracts pin the difference.
"""

import os
import sys
import unittest
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import build_faceset_v2 as bfv
from roop.faceset_v2 import migrate_legacy_fsz, write_faceset_v2


def _png(seed=0):
    import cv2
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes(), img


def _legacy(path, count=3):
    with zipfile.ZipFile(path, "w") as z:
        for i in range(count):
            z.writestr("%d.png" % i, _png(i)[0])
    return path


class _Face(dict):
    """Minimal insightface-shaped face; attribute access returns None."""

    def __getattr__(self, name):
        return self.get(name)


class VerifyRejectsInertArchivesTest(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.dir = tempfile.mkdtemp(prefix="fsv2-")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_legacy_archive_is_not_v2_at_all(self):
        p = _legacy(os.path.join(self.dir, "legacy.fsz"))
        r = bfv.verify(p)
        self.assertFalse(r["ok"])
        self.assertEqual(r["why"], "no V2 metadata member")

    def test_detectionless_migration_is_refused_despite_version_2(self):
        """The trap: version 2, valid schema, and the feature still cannot run."""
        src = _legacy(os.path.join(self.dir, "legacy.fsz"))
        dst = os.path.join(self.dir, "migrated.fsz")
        migrate_legacy_fsz(src, dst)

        r = bfv.verify(dst)
        # The naive gate the handoff asked for passes...
        self.assertEqual(r["format_version"], 2)
        self.assertEqual(r["schema"], "roop.fsz")
        # ...and the archive is still inert, so verify() must refuse it.
        self.assertTrue(r["migrated_without_detection"])
        self.assertEqual(r["identity_detail_ok"], 0)
        self.assertFalse(r["ok"])


class MainRefusesToClobberTest(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.dir = tempfile.mkdtemp(prefix="fsv2-")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_empty_suffix_would_overwrite_the_locked_archive(self):
        """An empty --suffix aims the writer at its own input."""
        orig = bfv.LIB
        try:
            bfv.LIB = self.dir
            _legacy(os.path.join(self.dir, "src.fsz"))
            with self.assertRaises(SystemExit):
                bfv.build("src", "", None, None, None, None, 0, overwrite=True)
        finally:
            bfv.LIB = orig

    def test_existing_output_is_not_overwritten_without_the_flag(self):
        orig = bfv.LIB
        try:
            bfv.LIB = self.dir
            _legacy(os.path.join(self.dir, "src.fsz"))
            _legacy(os.path.join(self.dir, "src_v2.fsz"))
            with self.assertRaises(SystemExit):
                bfv.build("src", "_v2", None, None, None, None, 0, overwrite=False)
        finally:
            bfv.LIB = orig


if __name__ == "__main__":
    unittest.main()
