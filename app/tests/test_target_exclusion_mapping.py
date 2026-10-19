"""Stage 19: "Exclude (never swap)" survives the backend.

The React UI writes an exclusion as an EXPLICIT `null` for the person
(`react-ui/src/components/faceswap/faceRoles.js`). The backend has to treat that as
"drop this person's binding", including when it is the ONLY binding: the canonical
mapping is replaced only `if stable:`, so excluding the last mapped person used to leave
the old binding in place and the person kept swapping.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_target_person_identity import TargetPersonIdentityTests as _Base  # noqa: E402
import api_state as state  # noqa: E402


class ExcludeMapping(unittest.TestCase):
    setUp = _Base.setUp
    tearDown = _Base.tearDown
    _add = _Base._add
    _configure = _Base._configure

    def _write(self, mapping):
        import api
        response = api.target_context({
            "target_media_id": state.active_target_media_id,
            "target_person_source_mapping": mapping,
        })
        # an error comes back as a JSONResponse, success as the target payload (a dict)
        self.assertNotIsInstance(response, api.JSONResponse, getattr(response, "body", response))

    def test_excluding_one_of_two_drops_only_that_person(self):
        self._add("a.mp4")
        records = self._configure(0, ["A", "B"], {})
        a, b = (r["target_person_id"] for r in records)
        self._write({a: "source-a", b: "source-b"})
        self.assertEqual(state.active_target_person_source_mapping, {a: "source-a", b: "source-b"})
        self._write({a: "source-a", b: None})
        self.assertEqual(state.active_target_person_source_mapping, {a: "source-a"})

    def test_excluding_the_only_mapped_person_clears_the_binding(self):
        self._add("a.mp4")
        records = self._configure(0, ["A", "B"], {})
        a, b = (r["target_person_id"] for r in records)
        self._write({b: "source-b"})
        self.assertEqual(state.active_target_person_source_mapping, {b: "source-b"})
        self._write({b: None})
        self.assertEqual(state.active_target_person_source_mapping, {},
                         "the exclusion was ignored: B still has a source")

    def test_the_explicit_null_is_kept_for_the_round_trip(self):
        self._add("a.mp4")
        records = self._configure(0, ["A", "B"], {})
        a, b = (r["target_person_id"] for r in records)
        self._write({a: "source-a", b: None})
        self.assertIn(b, state.active_target_source_mapping)
        self.assertIsNone(state.active_target_source_mapping[b])


if __name__ == "__main__":
    unittest.main()
