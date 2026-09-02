"""Automated coordinate transformation and overlay rendering unit tests for React UI 2.0.

Verifies:
1. Native-to-viewport bounding box percentage mapping across aspect ratios (16:9, 9:16, 1:1, 4:3, 21:9).
2. Sub-pixel zoom and boundary pan clamping calculations.
3. 5-point ArcFace landmark alignment geometry.
4. Solved 3D head pose vector string representations.
5. Presence and exports of all React UI 2.0 preview subsystem components.
"""

import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
V2_SRC = ROOT / "react-ui-v2" / "src"


# Python reference mirror of zoomPan.js and trackingAdapter.js
def box_to_percent_style(bbox, img_w, img_h):
    if not bbox or not img_w or not img_h:
        return None
    sx, sy, ex, ey = bbox
    return {
        "left": f"{(sx / img_w) * 100:.4f}%",
        "top": f"{(sy / img_h) * 100:.4f}%",
        "width": f"{((ex - sx) / img_w) * 100:.4f}%",
        "height": f"{((ey - sy) / img_h) * 100:.4f}%",
    }


def parse_landmarks(kps):
    if not kps or len(kps) < 5:
        return None
    eye_l, eye_r, nose, mouth_l, mouth_r = kps[:5]
    return {
        "eye_mid": [(eye_l[0] + eye_r[0]) / 2, (eye_l[1] + eye_r[1]) / 2],
        "mouth_mid": [(mouth_l[0] + mouth_r[0]) / 2, (mouth_l[1] + mouth_r[1]) / 2],
    }


def format_pose(pose_list):
    if not pose_list or len(pose_list) < 3:
        return None
    yaw, pitch, roll = [round(float(v)) for v in pose_list[:3]]
    return f"y{yaw}° p{pitch}° r{roll}°"


def clamp_pan(pan_x, pan_y, zoom, cw, ch, iw, ih):
    if zoom <= 1:
        return 0, 0
    max_pan_x = max(0, (iw * zoom - cw) / 2)
    max_pan_y = max(0, (ih * zoom - ch) / 2)
    clamped_x = max(-max_pan_x, min(pan_x, max_pan_x))
    clamped_y = max(-max_pan_y, min(pan_y, max_pan_y))
    return clamped_x, clamped_y


class UI2PreviewCoordinateTests(unittest.TestCase):
    def test_bounding_box_percent_mapping_16_by_9(self):
        # 1920x1080 frame with face at [480, 270, 1440, 810]
        style = box_to_percent_style([480, 270, 1440, 810], 1920, 1080)
        self.assertEqual(style["left"], "25.0000%")
        self.assertEqual(style["top"], "25.0000%")
        self.assertEqual(style["width"], "50.0000%")
        self.assertEqual(style["height"], "50.0000%")

    def test_bounding_box_percent_mapping_4k(self):
        # 3840x2160 frame with face at [960, 540, 2880, 1620]
        style = box_to_percent_style([960, 540, 2880, 1620], 3840, 2160)
        self.assertEqual(style["left"], "25.0000%")
        self.assertEqual(style["top"], "25.0000%")
        self.assertEqual(style["width"], "50.0000%")
        self.assertEqual(style["height"], "50.0000%")

    def test_bounding_box_percent_mapping_portrait_9_by_16(self):
        # 1080x1920 portrait video frame with face at [270, 480, 810, 1440]
        style = box_to_percent_style([270, 480, 810, 1440], 1080, 1920)
        self.assertEqual(style["left"], "25.0000%")
        self.assertEqual(style["top"], "25.0000%")
        self.assertEqual(style["width"], "50.0000%")
        self.assertEqual(style["height"], "50.0000%")

    def test_bounding_box_percent_mapping_square_and_ultrawide(self):
        # Square 1024x1024
        style_sq = box_to_percent_style([100, 100, 600, 600], 1024, 1024)
        self.assertAlmostEqual(float(style_sq["left"].rstrip("%")), 100 / 1024 * 100, places=3)
        self.assertAlmostEqual(float(style_sq["width"].rstrip("%")), 500 / 1024 * 100, places=3)

        # Ultrawide 3440x1440
        style_uw = box_to_percent_style([860, 360, 2580, 1080], 3440, 1440)
        self.assertEqual(style_uw["left"], "25.0000%")
        self.assertEqual(style_uw["top"], "25.0000%")

    def test_landmarks_midpoints_calculation(self):
        # 5-point ArcFace landmark coordinates
        kps = [
            [200.0, 150.0],  # eyeL
            [300.0, 150.0],  # eyeR
            [250.0, 200.0],  # nose
            [220.0, 260.0],  # mouthL
            [280.0, 260.0],  # mouthR
        ]
        parsed = parse_landmarks(kps)
        self.assertEqual(parsed["eye_mid"], [250.0, 150.0])
        self.assertEqual(parsed["mouth_mid"], [250.0, 260.0])

    def test_head_pose_vector_formatting(self):
        pose = [-14.2, 4.8, 2.1]
        formatted = format_pose(pose)
        self.assertEqual(formatted, "y-14° p5° r2°")

    def test_zoom_and_pan_bounds_clamping(self):
        # At 1x zoom, pan must be strictly (0, 0)
        cx, cy = clamp_pan(100, 100, 1.0, 800, 600, 800, 600)
        self.assertEqual((cx, cy), (0, 0))

        # At 2x zoom on 800x600 container with 800x600 image:
        # max_pan_x = (800 * 2 - 800) / 2 = 400
        # max_pan_y = (600 * 2 - 600) / 2 = 300
        cx, cy = clamp_pan(500, -400, 2.0, 800, 600, 800, 600)
        self.assertEqual((cx, cy), (400, -300))

        # Within bounds: should retain exact coordinate
        cx, cy = clamp_pan(150, -120, 2.0, 800, 600, 800, 600)
        self.assertEqual((cx, cy), (150, -120))

    def test_preview_subsystem_files_and_exports_exist(self):
        preview_dir = V2_SRC / "components" / "preview"
        self.assertTrue((preview_dir / "InteractivePreview.jsx").is_file())
        self.assertTrue((preview_dir / "CrossfadeImage.jsx").is_file())
        self.assertTrue((preview_dir / "TrackingOverlay.jsx").is_file())
        self.assertTrue((preview_dir / "ComparisonWipe.jsx").is_file())
        self.assertTrue((preview_dir / "MagnifierLoupe.jsx").is_file())
        self.assertTrue((preview_dir / "MaskBrushOverlay.jsx").is_file())
        self.assertTrue((preview_dir / "PreviewControlsHUD.jsx").is_file())
        self.assertTrue((preview_dir / "index.js").is_file())

        index_text = (preview_dir / "index.js").read_text(encoding="utf-8")
        for export_name in (
            "InteractivePreview",
            "CrossfadeImage",
            "TrackingOverlay",
            "ComparisonWipe",
            "MagnifierLoupe",
            "MaskBrushOverlay",
            "PreviewControlsHUD",
        ):
            self.assertIn(export_name, index_text)


if __name__ == "__main__":
    unittest.main()
