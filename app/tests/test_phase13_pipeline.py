"""Dependency-light contracts for the Phase 13 output pipeline."""
import os
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)


def _read(name):
    with open(os.path.join(APP, "roop", name), encoding="utf-8") as fh:
        return fh.read()


class Phase13WriterContract(unittest.TestCase):
    def setUp(self):
        self.fw = _read("ffmpeg_writer.py")
        self.seg = _read("segment_writer.py")
        self.pm = _read("ProcessMgr.py")
        with open(os.path.join(HERE, "phase13_benchmark.py"), encoding="utf-8") as fh:
            self.bench = fh.read()

    def test_explicit_codec_is_passed_through_the_controlled_pipeline(self):
        with open(os.path.join(HERE, "two_face_video.py"), encoding="utf-8") as fh:
            two_face = fh.read()
        with open(os.path.join(HERE, "baseline_controlled.py"), encoding="utf-8") as fh:
            controlled = fh.read()
        self.assertIn('"--codec"', controlled)
        self.assertIn('g.video_encoder = args.codec', two_face)
        self.assertIn('"--codec", codec', self.bench)

    def test_colorspace_conversion_has_an_explicit_no_conversion_path(self):
        self.assertIn("ROOP_FFMPEG_COLORSPACE", self.fw)
        self.assertIn("'off', 'none', 'passthrough'", self.fw)
        self.assertIn("colorspace=bt709:iall=bt601-6-625:fast=1", self.fw)

    def test_segments_use_dynamic_duration_but_keep_explicit_chunk_authority(self):
        self.assertIn("ROOP_RESUME_SEGMENT_SECONDS", self.seg)
        self.assertIn('raw_chunk = os.environ.get("ROOP_RESUME_CHUNK")', self.seg)
        self.assertIn("self.chunk = max(50, int(raw_chunk))", self.seg)

    def test_segment_writer_forwards_encoder_options(self):
        self.assertIn("**self._writer_options", self.seg)
        for key in ("preset", "bitrate", "threads", "ffmpeg_params", "colorspace"):
            self.assertIn('"%s"' % key, self.seg)

    def test_single_segment_skips_redundant_concat(self):
        self.assertIn("self._promote_single()", self.seg)
        self.assertIn("os.replace(source, self.target_video)", self.seg)
        self.assertIn("preserve_parts = not completed and keep_after_stop", self.seg)

    def test_finalize_time_is_included_in_encoder_profile(self):
        self.assertIn("with _prof('encode_finalize')", self.pm)

    def test_output_queue_override_is_bounded(self):
        self.assertIn("ROOP_OUTPUT_QUEUE_DEPTH", self.pm)
        self.assertIn("min(4, output_qdepth)", self.pm)

    def test_report_contains_end_to_end_encoder_fields(self):
        for field in ("wall_seconds", "encode_total_seconds", "encode_share_pct",
                      "encode_throughput_fps", "rotation_count", "stability",
                      "output_quality"):
            self.assertIn('"%s"' % field, self.bench)


if __name__ == "__main__":
    unittest.main()
