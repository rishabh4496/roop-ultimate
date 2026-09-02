"""What a resumed project reports as its progress.

`ApiProgress` maps a run's own 0..1 onto `[base, 1]`, where `base` is the share
of the frame range the run will NOT redo. `_start_existing_project` used to
compute that from the checkpoint's `safe_frame`.

`safe_frame` is not that quantity. `_checkpoint_segment` advances it from the
frame index alone whenever no writer exists yet, so it routinely records a
prefix with nothing on disk behind it. Any run interrupted at or past the end
of its range therefore produced base = 1.0 -- and `base + fraction * (1 - base)`
with base 1.0 is 1.0 for every fraction. The resumed render sat at 100% from
its first frame to its last while re-rendering the whole segment. Observed
live: "Processing frame 777 / 899" with a reported progress of 1.0.

The frames that actually survive an interruption are the ones inside COMMITTED
segments, so that is what `base` is now taken from.
"""

import unittest


def resume_base(record):
    """The computation under test, as `api._start_existing_project` performs it."""
    inputs = record.get("inputs") or {}
    total = max(0, int(inputs.get("frame_end", 0) or 0) -
                int(inputs.get("frame_start", 0) or 0))
    committed = sum(int(item.get("frames", 0) or 0)
                    for item in ((record.get("checkpoint") or {}).get("segments") or []))
    return min(1.0, max(0, committed) / total) if total else 0.0


def reported(base, fraction):
    """`ApiProgress.__call__`'s mapping."""
    return max(0.0, min(1.0, base + fraction * (1.0 - base)))


def record(*, start, end, safe, segments=()):
    return {
        "inputs": {"frame_start": start, "frame_end": end},
        "checkpoint": {"safe_frame": safe, "next_frame": safe,
                       "segments": [{"frames": n} for n in segments]},
    }


class ResumeBase(unittest.TestCase):
    def test_nothing_committed_reports_real_progress(self):
        """The regression: safe_frame at the end of the range, zero segments."""
        base = resume_base(record(start=1, end=900, safe=900))
        self.assertEqual(base, 0.0)
        # The whole point — a mid-render fraction must not read as finished.
        self.assertAlmostEqual(reported(base, 777 / 899), 777 / 899)
        self.assertLess(reported(base, 0.5), 1.0)

    def test_committed_segments_are_the_prefix_that_survives(self):
        base = resume_base(record(start=0, end=1000, safe=1000, segments=(300, 200)))
        self.assertAlmostEqual(base, 0.5)
        self.assertAlmostEqual(reported(base, 0.0), 0.5)
        self.assertAlmostEqual(reported(base, 1.0), 1.0)

    def test_a_fresh_project_starts_at_zero(self):
        self.assertEqual(resume_base(record(start=0, end=500, safe=0)), 0.0)

    def test_fully_committed_range_is_complete(self):
        self.assertEqual(resume_base(record(start=0, end=400, safe=400, segments=(400,))), 1.0)

    def test_committed_beyond_the_range_is_clamped(self):
        self.assertEqual(resume_base(record(start=0, end=100, safe=100, segments=(90, 90))), 1.0)

    def test_an_image_job_has_no_range(self):
        self.assertEqual(resume_base(record(start=0, end=0, safe=0)), 0.0)

    def test_safe_frame_alone_no_longer_moves_the_base(self):
        """Two records differing only in safe_frame must report the same base."""
        self.assertEqual(resume_base(record(start=1, end=900, safe=900)),
                         resume_base(record(start=1, end=900, safe=1)))


if __name__ == "__main__":
    unittest.main()
