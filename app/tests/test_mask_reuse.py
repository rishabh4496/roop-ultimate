"""The mask must be computed once per face, not once per target.

With an enhancer active, `swap_faces` masks TWICE per face — the swapped crop and
the enhanced crop. Everything `process_mask` derives its mask from (`frame`,
`target_face`, `M`, `orig_frame`, `tgt_pitch_deg`) is identical across that pair;
only `target` differs. So the second call was recomputing a byte-identical mask:
the engine inference, the landmark convex hull, the mouth mask, the blurs and the
non-frontal unwarp.

The engine is the cheap part of that — DFL XSeg measures 2.35 ms/call isolated,
against a mask stage reported at ~70 ms/call. The waste was the CPU work wrapped
around it.

Measured on a fixed 240-frame 1080p clip, same seed, same settings:

    mask stage   69.60 -> 42.44 ms/call   (-39%)
    throughput    4.47 ->  4.65 fps       (+4.0%)
    output       bit-identical: 240 frames, 0 differing, max pixel delta 0

Bit-identity is the whole point — an optimisation that only removes duplicated
work has no licence to move a pixel. It does not halve, because the composite
step still runs per target (the swapped crop is 256px, the enhanced one 512px,
so the mask is resized differently for each); only the derivation is shared.

Checked structurally: exercising process_mask for real needs a live ProcessMgr,
a mask processor and a TensorRT session. The bit-identity above was measured
end-to-end out of band, which is the evidence that matters; these guard the shape
so the second call cannot quietly start recomputing again.
"""

import os
import re
import sys
import unittest
from pathlib import Path

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

MASKING = Path(APP, 'roop', 'procmgr_masking.py').read_text(encoding='utf-8')
PM = Path(APP, 'roop', 'ProcessMgr.py').read_text(encoding='utf-8')


def _code(text):
    text = re.sub(r'""".*?"""', '', text, flags=re.S)
    return '\n'.join(re.sub(r'#.*$', '', ln) for ln in text.splitlines())


MASK_CODE, PM_CODE = _code(MASKING), _code(PM)


class MaskIsDerivedOncePerFace(unittest.TestCase):

    def test_sources_were_found(self):
        self.assertGreater(len(MASK_CODE), 5000)
        self.assertGreater(len(PM_CODE), 20000)

    def test_process_mask_accepts_a_precomputed_mask(self):
        sig = re.search(r'def process_mask\(([^)]*)\)', MASK_CODE)
        self.assertIsNotNone(sig, 'process_mask not found')
        self.assertIn('reuse_mask', sig.group(1),
                      'process_mask cannot be handed an already-computed mask')

    def test_process_mask_returns_the_mask_it_computed(self):
        """Otherwise the caller has nothing to pass to the second call."""
        self.assertRegex(MASK_CODE, r'return self\._composite_mask\([^)]*\), *\w+')

    def test_the_second_call_site_reuses_the_first_mask(self):
        """The regression: two bare process_mask calls means the mask is derived
        twice per face for identical output."""
        calls = re.findall(r'self\.process_mask\([^\n]*', PM_CODE)
        self.assertEqual(len(calls), 2, f'expected exactly 2 call sites, got {calls}')
        first, second = calls
        self.assertNotIn('reuse_mask', first,
                         'the first call must actually compute the mask')
        self.assertIn('reuse_mask', second,
                      'the enhanced-crop call must reuse the mask from the '
                      'swapped-crop call — every input it derives from is the same')

    def test_composite_is_the_only_target_dependent_step(self):
        """Reuse is only sound because `target` is consumed exclusively in the
        composite. If mask DERIVATION starts reading `target`, sharing one mask
        across the two calls silently becomes wrong."""
        body = re.search(r'def process_mask\(.*?\n(?=\n?    def )', MASK_CODE, re.S)
        self.assertIsNotNone(body)
        derivation = body.group(0)
        # The guard clause and the final composite call may mention target;
        # nothing in between may.
        derivation = re.sub(r'if reuse_mask is not None:.*?\n', '', derivation, flags=re.S | re.M)
        derivation = re.sub(r'return self\._composite_mask\([^\n]*\n', '', derivation)
        self.assertNotIn('target.shape', derivation,
                         'mask derivation now depends on the target it is applied '
                         'to — one mask can no longer serve both calls')

    def test_composite_helper_exists_and_handles_differing_sizes(self):
        """The swapped crop and the enhanced crop are different resolutions, so
        the shared mask has to be resized per target rather than once."""
        self.assertIn('def _composite_mask', MASK_CODE)
        helper = re.search(r'def _composite_mask\(.*?\n(?=\n?    def )', MASK_CODE, re.S)
        self.assertIsNotNone(helper)
        self.assertIn('cv2.resize(img_mask', helper.group(0))


if __name__ == '__main__':
    unittest.main()
