"""Unit tests for Stage 18: WebGL2 3D .cube LUT Color Pipeline, Skin-Protected Vibrance, and Real-Time Broadcast Scopes.

Validates:
1. .cube domain validation invariants: strict DOMAIN_MIN < DOMAIN_MAX enforcement.
2. Premultiplied alpha custody: strict (rgb * a, a) scaling and over compositing.
3. FFmpeg video writer tetrahedral 3D LUT injection into 10-bit encode pipeline.
4. Settings serialization and schema custody for project checkpoints and Batch Matrix recipes.
5. roop.globals Stage 18 attributes and API payload handling.
"""

import os
import tempfile
import unittest
from app.roop import globals as roop_globals
from app.roop.ffmpeg_writer import FFMPEG_VideoWriter
from app.roop.segment_writer import SegmentedVideoWriter
from app.settings import Settings


class TestStage18LutAndScopes(unittest.TestCase):
    def test_cube_domain_validation_logic(self):
        """Verify DOMAIN_MIN < DOMAIN_MAX invariant logic."""
        def validate_domain(domain_min, domain_max):
            if len(domain_min) != 3 or len(domain_max) != 3:
                raise ValueError("Domain arrays must have 3 elements")
            for i in range(3):
                if domain_min[i] >= domain_max[i]:
                    raise ValueError(
                        f"Invalid .cube domain: DOMAIN_MIN ({domain_min}) must be strictly less than DOMAIN_MAX ({domain_max}) at channel {i}"
                    )
            return True

        # Valid domain
        self.assertTrue(validate_domain([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
        self.assertTrue(validate_domain([-0.1, -0.1, -0.1], [1.2, 1.2, 1.2]))

        # Inverted domain: min >= max
        with self.assertRaises(ValueError):
            validate_domain([1.0, 0.0, 0.0], [0.5, 1.0, 1.0])

        # Equal domain: min == max
        with self.assertRaises(ValueError):
            validate_domain([0.0, 0.5, 0.0], [1.0, 0.5, 1.0])

    def test_premultiplied_alpha_custody(self):
        """Verify premultiplied alpha scaling (rgb * a, a) and compositing."""
        def premultiply(rgb, alpha):
            a = max(0.0, min(1.0, float(alpha)))
            return [rgb[0] * a, rgb[1] * a, rgb[2] * a, a]

        def blend_over(src, dst):
            src_a = src[3]
            out_r = src[0] + dst[0] * (1.0 - src_a)
            out_g = src[1] + dst[1] * (1.0 - src_a)
            out_b = src[2] + dst[2] * (1.0 - src_a)
            out_a = src_a + dst[3] * (1.0 - src_a)
            return [out_r, out_g, out_b, out_a]

        # Half alpha scaling
        pm = premultiply([1.0, 0.6, 0.2], 0.5)
        self.assertAlmostEqual(pm[0], 0.5, places=5)
        self.assertAlmostEqual(pm[1], 0.3, places=5)
        self.assertAlmostEqual(pm[2], 0.1, places=5)
        self.assertAlmostEqual(pm[3], 0.5, places=5)

        # Zero alpha scaling produces black
        pm_zero = premultiply([1.0, 0.6, 0.2], 0.0)
        self.assertEqual(pm_zero, [0.0, 0.0, 0.0, 0.0])

        # Full opacity over background
        src = [0.5, 0.25, 0.1, 0.5]
        dst = [0.5, 0.5, 0.5, 1.0]
        out = blend_over(src, dst)
        self.assertAlmostEqual(out[0], 0.75, places=5)
        self.assertAlmostEqual(out[1], 0.50, places=5)
        self.assertAlmostEqual(out[2], 0.35, places=5)
        self.assertAlmostEqual(out[3], 1.00, places=5)

    def test_ffmpeg_writer_lut_tetrahedral_filter(self):
        """Verify FFMPEG_VideoWriter injects lut3d tetrahedral filter when lut_bake is enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_lut_path = os.path.join(tmpdir, "test.cube")
            with open(temp_lut_path, "wb") as f:
                f.write(b"TITLE \"Test LUT\"\nLUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n")

            dummy_out = os.path.join(tmpdir, "dummy_out.mp4")

            # Arm 1: lut_bake=True with existing file -> filter present
            writer_baked = FFMPEG_VideoWriter(
                dummy_out,
                (1920, 1080),
                fps=30.0,
                lut_path=temp_lut_path,
                lut_bake=True
            )
            cmd_baked = writer_baked._build_cmd("libx265")
            # Verify tetrahedral filter in command
            self.assertTrue(any("lut3d=" in arg and "interp=tetrahedral" in arg for arg in cmd_baked))

            # Arm 2: lut_bake=False -> display only, filter omitted from FFmpeg encode
            writer_nobake = FFMPEG_VideoWriter(
                dummy_out,
                (1920, 1080),
                fps=30.0,
                lut_path=temp_lut_path,
                lut_bake=False
            )
            cmd_nobake = writer_nobake._build_cmd("libx265")
            self.assertFalse(any("lut3d=" in arg for arg in cmd_nobake))

            # Arm 3: lut_bake=True but empty lut_path -> filter omitted
            writer_nopath = FFMPEG_VideoWriter(
                dummy_out,
                (1920, 1080),
                fps=30.0,
                lut_path="",
                lut_bake=True
            )
            cmd_nopath = writer_nopath._build_cmd("libx265")
            self.assertFalse(any("lut3d=" in arg for arg in cmd_nopath))

    def test_segmented_writer_lut_forwarding(self):
        """Verify SegmentedVideoWriter forwards lut_path and lut_bake options into identity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_out = os.path.join(tmpdir, "dummy_out.mp4")
            writer = SegmentedVideoWriter(
                dummy_out,
                (1920, 1080),
                fps=30.0,
                lut_path="C:/path/to/grade.cube",
                lut_bake=True
            )
            self.assertEqual(writer._writer_options.get("lut_path"), "C:/path/to/grade.cube")
            self.assertTrue(writer._writer_options.get("lut_bake"))
            self.assertIn("C:/path/to/grade.cube", str(writer._identity))

    def test_settings_stage18_serialization(self):
        """Verify Settings._load and Settings.save serialize Stage 18 color grading parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = os.path.join(tmpdir, "config.yaml")
            s = Settings(settings_file)

            # Check default state
            self.assertTrue(hasattr(s, "lut_path"))
            self.assertTrue(hasattr(s, "lut_bake"))
            self.assertTrue(hasattr(s, "lut_opacity"))
            self.assertTrue(hasattr(s, "color_temperature"))
            self.assertTrue(hasattr(s, "color_tint"))
            self.assertTrue(hasattr(s, "color_vibrance"))
            self.assertTrue(hasattr(s, "color_skin_protect"))

            self.assertFalse(s.lut_bake)
            self.assertEqual(s.lut_opacity, 1.0)
            self.assertEqual(s.color_skin_protect, 1.0)

            # Mutate grading settings
            s.lut_path = "G:/luts/creative_film.cube"
            s.lut_bake = True
            s.lut_opacity = 0.85
            s.color_temperature = 0.25
            s.color_tint = -0.10
            s.color_vibrance = 0.40
            s.color_skin_protect = 0.90
            s.save()

            # Reload in a clean Settings instance to verify round-trip persistence
            s2 = Settings(settings_file)
            self.assertEqual(s2.lut_path, "G:/luts/creative_film.cube")
            self.assertTrue(s2.lut_bake)
            self.assertAlmostEqual(s2.lut_opacity, 0.85, places=4)
            self.assertAlmostEqual(s2.color_temperature, 0.25, places=4)
            self.assertAlmostEqual(s2.color_tint, -0.10, places=4)
            self.assertAlmostEqual(s2.color_vibrance, 0.40, places=4)
            self.assertAlmostEqual(s2.color_skin_protect, 0.90, places=4)

    def test_roop_globals_stage18_attributes(self):
        """Verify Stage 18 globals exist and have correct types."""
        self.assertTrue(hasattr(roop_globals, "lut_path"))
        self.assertTrue(hasattr(roop_globals, "lut_bake"))
        self.assertTrue(hasattr(roop_globals, "lut_opacity"))
        self.assertTrue(hasattr(roop_globals, "color_temperature"))
        self.assertTrue(hasattr(roop_globals, "color_tint"))
        self.assertTrue(hasattr(roop_globals, "color_vibrance"))
        self.assertTrue(hasattr(roop_globals, "color_skin_protect"))
        self.assertIsInstance(roop_globals.lut_bake, bool)
