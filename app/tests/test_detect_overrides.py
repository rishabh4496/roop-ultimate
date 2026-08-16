"""A per-call det_size/det_thresh override must reach the engine actually in use.

`_rescue_downscaled` is the close-up rescue, and it is the only rescue in the
ladder that works by changing detector PARAMETERS rather than the image — the
other three hand the detector a different picture (upscaled / padded / rotated).
So if its override does not arrive, it is not merely weaker: it re-runs the pass
that has just returned nothing, with the same numbers, and can only return
nothing again.

That is what happened. The overrides were written onto `fa.det_model`, which is
SCRFD's detector. A hybrid engine does not own one — `_build_face_analyser` pops
it and leaves None — and the hybrid branches then re-read the unchanged globals,
so on four of the five engines the rescue was a byte-identical repeat costing a
full extra detection per empty frame.

Measured on 8 real close-up frames per zoom level, before -> after (recovered by
the rescue, of the frames the 640 pass missed):

    retinaface   z=1.6   0/8 -> 8/8        yunet  z=2.8  0/2 -> 2/2
    retinaface   z=2.2   0/8 -> 8/8        yunet  z=3.4  0/5 -> 5/5

Primary-pass hit counts were identical across all five engines and all five zoom
levels, i.e. nothing about the normal path moved.

Checked structurally rather than by running a detect: importing face_util pulls
roop.globals and torch (~4s), which would roughly double a suite that runs in
four seconds. The functional A/B above was run out-of-band against the real
models, and the pre-fix column is what makes it non-vacuous.
"""

import os
import re
import sys
import unittest
from pathlib import Path

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

FU = Path(APP, 'roop', 'face_util.py').read_text(encoding='utf-8')
HYBRIDS = ('_hybrid_yolo_faces', '_hybrid_yunet_faces', '_hybrid_retinaface_faces')


def _code(text):
    """Source with docstrings and comments stripped — this file discusses the
    defect at length, so a substring search would match the prose describing it."""
    text = re.sub(r'""".*?"""', '', text, flags=re.S)
    return '\n'.join(re.sub(r'#.*$', '', ln) for ln in text.splitlines())


def _body(name, src):
    m = re.search(r'def %s\(.*?\n(?=\n\ndef |\Z)' % name, src, re.S)
    assert m is not None, f'{name} not found'
    return m.group(0)


CODE = _code(FU)


class OverridesReachTheSelectedEngine(unittest.TestCase):

    def test_source_was_found(self):
        """Guard the parsing — an empty source would pass everything below."""
        self.assertGreater(len(CODE), 5000)
        for name in HYBRIDS + ('_detect_faces_raw', '_rescue_downscaled'):
            self.assertIn(f'def {name}(', CODE, name)

    def test_each_hybrid_takes_the_override_as_an_argument(self):
        for name in HYBRIDS:
            sig = re.search(r'def %s\(([^)]*)\)' % name, CODE).group(1)
            with self.subTest(hybrid=name):
                for param in ('det_size', 'det_thresh'):
                    self.assertIn(param, sig,
                                  f'{name} does not accept {param}, so a per-call '
                                  f'override cannot reach it')

    def test_no_hybrid_re_reads_the_globals(self):
        """The actual defect. Re-deriving these inside the helper silently
        discards whatever the caller asked for — the helper still *looks* like it
        honours a det_size, because it has one."""
        for name in HYBRIDS:
            body = _body(name, CODE)
            with self.subTest(hybrid=name):
                self.assertNotIn('_desired_det_size', body,
                                 f'{name} re-derives det_size from globals, '
                                 f'overriding the caller')
                self.assertNotIn('face_detector_threshold', body,
                                 f'{name} re-derives det_thresh from globals, '
                                 f'overriding the caller')

    def test_every_dispatch_forwards_the_effective_values(self):
        """Each of the five engine branches has to be handed them; one branch
        left behind is one engine where the rescue is dead again.

        Only the four ENGINE dispatches are in scope. `_hybrid_detector_faces`
        is also reachable from here (the detector-only SCRFD path, for callers
        that read nothing but bbox/kps), but it is handed boxes that have
        already been detected — there is no detector left inside it for a
        det_size to steer, so requiring one would be requiring a dead argument.
        """
        raw = _body('_detect_faces_raw', CODE)
        self.assertIn('eff_size', raw)
        self.assertIn('eff_thresh', raw)
        calls = [c for c in re.findall(r'_hybrid_\w+\([^)]*\)', raw)
                 if not c.startswith('_hybrid_detector_faces(')]
        self.assertEqual(len(calls), 4, f'expected 4 hybrid dispatches, got {calls}')
        for call in calls:
            with self.subTest(call=call):
                self.assertIn('eff_size', call)
                self.assertIn('eff_thresh', call)

    def test_effective_values_fall_back_to_the_configured_ones(self):
        """With no override the behaviour must be exactly what it was."""
        raw = _body('_detect_faces_raw', CODE)
        self.assertRegex(raw, r'eff_size\s*=\s*det_size if det_size is not None')
        self.assertRegex(raw, r'eff_thresh\s*=\s*\(?det_thresh if det_thresh is not None')
        self.assertIn('_desired_det_size()', raw)
        self.assertIn('face_detector_threshold', raw)

    def test_the_close_up_rescue_still_asks_for_something_different(self):
        """If it ever asks for the configured size/threshold again it is once
        more a duplicate of the pass that just failed, override plumbing or not."""
        body = _body('_rescue_downscaled', CODE)
        call = re.search(r'_detect_faces_raw\((.*?)\)', body, re.S)
        self.assertIsNotNone(call, '_rescue_downscaled no longer detects')
        args = call.group(1)
        self.assertIn('det_size=320', args,
                      'the close-up rescue must drop the detector canvas')
        self.assertIn('det_thresh=', args,
                      'the close-up rescue must lower the confidence floor')
        self.assertIn('lowered_thresh', args)


if __name__ == '__main__':
    unittest.main()
