"""A requested identity-detail restoration that cannot run must say so.

`identity_detail_strength` is a strength dial, so a non-zero value is a
statement of intent. `FaceSet.identity_detail_for` returns None for every V1
archive -- which is what the shipped facesets are -- and the render path used to
skip in complete silence: return code 0, valid output, swap audit 100%, and the
`identity_detail` stage simply absent from the ROOP_PROFILE table.

That was measured on the RTX 4070, not inferred. A 30-frame render at
`--identity-detail-strength 0.35` produced no `identity_detail` stage row, and
its pixel difference against strength 0 (mean 0.766/255) sat inside the
pipeline's own run-to-run noise floor (mean 0.747/255 for two renders of one
unchanged configuration). Both the "did it run" and the "did it change
anything" instruments read clean while the feature did nothing.

These tests pin the reporting, not the restoration.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from roop import identity_detail as idet   # noqa: E402


class _FakeFaceSet(object):
    def __init__(self, version):
        self.format_version = version


class IdentityDetailVisibility(unittest.TestCase):
    def setUp(self):
        idet.reset_identity_detail_warnings()
        self.addCleanup(idet.reset_identity_detail_warnings)

    def test_v1_faceset_is_named_as_the_cause(self):
        idet.warn_identity_detail_unavailable(source_index=0,
                                              faceset=_FakeFaceSet(1),
                                              strength=0.35)
        causes = idet.identity_detail_unavailable_causes()
        self.assertEqual(1, len(causes))
        self.assertIn("v1", causes[0])
        self.assertIn("FaceSet V2", causes[0])

    def test_v2_without_a_residual_is_a_different_cause(self):
        """A V2 archive that carries no residual must not read as "V1".

        The two need different remedies -- rebuild versus re-ingest that source
        -- so collapsing them would send the user to the wrong fix.
        """
        idet.warn_identity_detail_unavailable(source_index=2,
                                              faceset=_FakeFaceSet(2))
        causes = idet.identity_detail_unavailable_causes()
        self.assertEqual(1, len(causes))
        self.assertIn("no high-frequency residual", causes[0])
        self.assertIn("2", causes[0])

    def test_missing_faceset_is_a_third_cause(self):
        idet.warn_identity_detail_unavailable(source_index=0, faceset=None)
        self.assertIn("no source FaceSet",
                      idet.identity_detail_unavailable_causes()[0])

    def test_repeated_identical_causes_report_once(self):
        """Bounded on purpose: this sits on the per-face path.

        A 60,000-frame render with two faces would otherwise emit 120,000
        identical lines, which is its own failure.
        """
        for _ in range(50):
            idet.warn_identity_detail_unavailable(source_index=0,
                                                  faceset=_FakeFaceSet(1))
        self.assertEqual(1, len(idet.identity_detail_unavailable_causes()))

    def test_distinct_causes_each_report(self):
        idet.warn_identity_detail_unavailable(source_index=0,
                                              faceset=_FakeFaceSet(1))
        idet.warn_identity_detail_unavailable(source_index=0, faceset=None)
        self.assertEqual(2, len(idet.identity_detail_unavailable_causes()))


class RenderPathCallsTheReporter(unittest.TestCase):
    """The reporter only helps if the render path reaches it.

    Asserted against the source rather than by rendering, because the failing
    population is "a V1 faceset with a non-zero strength" and constructing a
    full ProcessMgr face pass for that is far more fragile than reading the one
    branch that matters.
    """

    def test_processmgr_warns_on_the_none_branch(self):
        with open(os.path.join(APP, "roop", "ProcessMgr.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("warn_identity_detail_unavailable", src,
                      "ProcessMgr no longer imports or calls the reporter")
        anchor = src.index("_detail = (_detail_fs.identity_detail_for")
        window = src[anchor:anchor + 900]
        self.assertIn("if _detail is None:", window,
                      "the unavailable branch disappeared from ProcessMgr")
        self.assertIn("warn_identity_detail_unavailable", window,
                      "the unavailable branch no longer reports; identity "
                      "detail can silently do nothing again")


if __name__ == "__main__":
    unittest.main()
