"""B-01 / B-02: the paste-mask offsets are read from the live settings when used, not from a per-face snapshot.

Before: ingest stamped `face.mask_offsets` from the config of that moment, the sliders wrote `faces[0]` of the
SELECTED faceset only, and the paste read `fs.faces[canonical_face_index()]` -- a different face for V2 sets, and a
stale one for every other faceset. A faceset built by a route that never stamped (face manager, identity lab,
project restore) had nothing at all: `insightface.Face.__getattr__` answers None, so `faces[0].mask_offsets[4]` raised
in the manual-mask parser and the user's painted mask was dropped.
"""

import os
import re
import sys
import types
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.environ.setdefault("ROOP_SKIP_STARTUP", "1")

import roop.globals as roop_globals  # noqa: E402
from roop import mask_offsets  # noqa: E402


def _cfg(**over):
    values = dict(mask_top=0.0, mask_bottom=0.0, mask_left=0.0, mask_right=0.0,
                  face_mask_blend=25.0, mouth_mask_blend=10.0,
                  mouth_top_scale=1.0, mouth_bottom_scale=1.0,
                  mouth_left_scale=1.0, mouth_right_scale=1.0)
    values.update(over)
    cfg = types.SimpleNamespace(**values)
    cfg.saved = 0
    cfg.save = lambda: setattr(cfg, "saved", cfg.saved + 1)
    return cfg


class _Restore(unittest.TestCase):
    def setUp(self):
        self._cfg = roop_globals.CFG

    def tearDown(self):
        roop_globals.CFG = self._cfg


class ResolveTests(_Restore):
    def test_a_face_without_the_attribute_gets_the_live_settings(self):
        roop_globals.CFG = _cfg(face_mask_blend=25.0)
        face = types.SimpleNamespace()          # no mask_offsets at all
        self.assertEqual(mask_offsets.resolve(face)[4], 25.0)

    def test_the_real_insightface_face_that_used_to_crash_the_manual_mask(self):
        try:
            from insightface.app.common import Face
        except Exception as exc:
            self.skipTest("insightface not importable here: %s" % type(exc).__name__)
        roop_globals.CFG = _cfg(face_mask_blend=17.0)
        face = Face()
        self.assertIsNone(face.mask_offsets)    # the premise of B-01
        self.assertEqual(mask_offsets.resolve(face)[4], 17.0)

    def test_a_slider_change_is_seen_by_the_next_call_with_no_per_face_copy(self):
        cfg = roop_globals.CFG = _cfg(face_mask_blend=12.0)
        faces = [types.SimpleNamespace(), types.SimpleNamespace()]   # "faces[0]" and the canonical face
        self.assertEqual([mask_offsets.resolve(f)[4] for f in faces], [12.0, 12.0])
        cfg.face_mask_blend = 30.0
        self.assertEqual([mask_offsets.resolve(f)[4] for f in faces], [30.0, 30.0])

    def test_an_explicit_per_face_override_still_wins(self):
        # What the benchmark harnesses do (runner.py, stage0_harness.py, regression.py).
        roop_globals.CFG = _cfg(face_mask_blend=25.0)
        face = types.SimpleNamespace(mask_offsets=[0, 0, 0, 0, 0.85])
        self.assertEqual(mask_offsets.resolve(face), [0, 0, 0, 0, 0.85])

    def test_a_malformed_override_is_ignored(self):
        roop_globals.CFG = _cfg(face_mask_blend=25.0)
        for bad in ([], [1, 2], "abc", 7):
            face = types.SimpleNamespace(mask_offsets=bad)
            self.assertEqual(mask_offsets.resolve(face)[4], 25.0, repr(bad))

    def test_no_face_at_all(self):
        roop_globals.CFG = _cfg(face_mask_blend=25.0)
        self.assertEqual(mask_offsets.resolve(None)[4], 25.0)

    def test_without_settings_the_historic_fallback_is_used(self):
        roop_globals.CFG = None
        self.assertEqual(list(mask_offsets.resolve(types.SimpleNamespace())), [0.0, 0.0, 0.0, 0.0, 20.0, 10.0])
        self.assertEqual(mask_offsets.from_cfg(), [0.0, 0.0, 0.0, 0.0, 12.0, 10.0, 1.0, 1.0, 1.0, 1.0])

    def test_dict_faces_are_read_too(self):
        roop_globals.CFG = _cfg()
        self.assertEqual(mask_offsets.resolve({"mask_offsets": [0, 0, 0, 0, 3.0, 4.0]})[4], 3.0)


class SliderWriteTests(_Restore):
    def setUp(self):
        super().setUp()
        try:
            import api
        except Exception as exc:
            self.skipTest("api module not importable here: %s" % type(exc).__name__)
        self.api = api
        self.old_sets = list(roop_globals.INPUT_FACESETS)

    def tearDown(self):
        roop_globals.INPUT_FACESETS[:] = self.old_sets
        super().tearDown()

    def test_sliders_update_the_settings_and_leave_faces_alone(self):
        cfg = roop_globals.CFG = _cfg(face_mask_blend=25.0)
        face = types.SimpleNamespace()
        fs = types.SimpleNamespace(faces=[face, types.SimpleNamespace()])
        roop_globals.INPUT_FACESETS[:] = [fs]
        self.api._update_mask_offsets_from_payload({"face_mask_blend": 9, "mouth_left_scale": "1.5"})
        self.assertEqual(cfg.face_mask_blend, 9.0)
        self.assertEqual(cfg.mouth_left_scale, 1.5)
        self.assertEqual(cfg.saved, 1)
        self.assertFalse(hasattr(face, "mask_offsets"), "a per-face copy would go stale again")
        self.assertEqual(mask_offsets.resolve(fs.faces[1])[4], 9.0)

    def test_unchanged_values_do_not_hit_the_disk(self):
        cfg = roop_globals.CFG = _cfg(face_mask_blend=25.0)
        self.api._update_mask_offsets_from_payload({"face_mask_blend": 25})
        self.assertEqual(cfg.saved, 0)

    def test_a_non_number_never_reaches_the_paste(self):
        cfg = roop_globals.CFG = _cfg(face_mask_blend=25.0)
        self.api._update_mask_offsets_from_payload({"face_mask_blend": "wide", "mask_top": None})
        self.assertEqual(cfg.face_mask_blend, 25.0)
        self.assertEqual(cfg.saved, 0)


class NothingStampsTheSnapshotAnyMore(unittest.TestCase):
    """Guards the call sites, because a stamped snapshot is exactly the thing that went stale."""

    def _source(self, *parts):
        with open(os.path.join(APP, *parts), encoding="utf-8") as handle:
            return handle.read()

    def test_ingest_does_not_stamp(self):
        for parts in (("source_gallery.py",), ("api.py",)):
            text = self._source(*parts)
            self.assertIsNone(re.search(r"\.mask_offsets\s*=\s*_mask_offsets_from_cfg\(\)", text), parts)

    def test_processmgr_reads_through_the_resolver(self):
        text = self._source("roop", "ProcessMgr.py")
        self.assertIsNone(re.search(r"\.mask_offsets\s*\[", text))
        self.assertIsNone(re.search(r"getattr\([^)]*'mask_offsets'", text))
        self.assertIn("resolve as _resolve_mask_offsets", text)


if __name__ == "__main__":
    unittest.main()
