"""Manual brush mask — that what the preview box paints survives to the render.

The brush was end-to-end dead before this: the canvas was exported nowhere, and
both /api/preview and /api/swap passed a literal `None` for `imagemask`, so even
a hand-built payload could not have reached ProcessMgr. These tests pin the two
halves that are easy to break again silently:

  * the ENCODING contract — the frontend exports solid white on transparent, and
    the backend decodes with cv2.IMREAD_GRAYSCALE. Exporting the translucent
    accent-coloured layer the user actually sees would decode to grey ~121 and be
    applied as a half-strength mask everywhere, which looks like a blending bug
    rather than a wiring bug.
  * the WIRING — neither endpoint may go back to hardcoding the mask off.

Nothing here needs a GPU, a model or a running server.
"""

import base64
import io as _io
import json
import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
API_PY = REPO / "app" / "api.py"
FACESWAP_JSX = REPO / "react-ui" / "src" / "components" / "FaceSwap.jsx"
PREVIEW_JSX = REPO / "react-ui" / "src" / "components" / "faceswap" / "InteractivePreview.jsx"


def _png_data_url(bgra):
    ok, buf = cv2.imencode(".png", bgra)
    assert ok
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _decode_like_processmgr(data_url):
    """The exact decode ProcessMgr._decode_mask performs on a mask data URL."""
    _, b64 = data_url.split(",", 1)
    arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


class TestMaskEncoding(unittest.TestCase):
    def test_white_on_transparent_decodes_to_full_strength(self):
        """A painted pixel must read as ~1.0, not a partial mask."""
        h = w = 32
        bgra = np.zeros((h, w, 4), dtype=np.uint8)
        bgra[8:24, 8:24] = (255, 255, 255, 255)      # solid white, opaque
        img = _decode_like_processmgr(_png_data_url(bgra))

        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (h, w))
        self.assertGreaterEqual(int(img[16, 16]), 250)   # painted -> full
        self.assertLessEqual(int(img[0, 0]), 5)          # untouched -> zero

    def test_translucent_accent_export_would_be_a_half_mask(self):
        """Why the export canvas is a separate white one.

        This is the bug the two-canvas split exists to prevent: exporting the
        visible rgba(233,69,96,0.45) layer yields a mid-grey, which the pipeline
        would apply as a ~50% blend of the original over the whole painted area.
        """
        bgra = np.zeros((32, 32, 4), dtype=np.uint8)
        bgra[8:24, 8:24] = (96, 69, 233, 115)        # BGRA of the visible paint
        img = _decode_like_processmgr(_png_data_url(bgra))
        self.assertLess(int(img[16, 16]), 200, "should NOT be full strength")

    def test_empty_canvas_decodes_to_nothing(self):
        """ProcessMgr treats an all-zero mask as absent (`not np.any`)."""
        bgra = np.zeros((32, 32, 4), dtype=np.uint8)
        img = _decode_like_processmgr(_png_data_url(bgra))
        self.assertFalse(np.any(img))


class TestMaskJsonShape(unittest.TestCase):
    """ProcessMgr picks its parser by whether every top-level key is a digit."""

    def test_frontend_shape_selects_the_per_faceset_parser(self):
        payload = json.loads(json.dumps({
            "0": {"exclude": "data:image/png;base64,AA==",
                  "canonical": False,
                  "ref_kps": [[1.0, 2.0]] * 5},
        }))
        keys = list(payload.keys())
        self.assertTrue(keys and all(k.isdigit() for k in keys))
        entry = payload["0"]
        self.assertIn("exclude", entry)
        self.assertFalse(entry["canonical"])
        self.assertEqual(len(entry["ref_kps"]), 5)


class TestMaskWiring(unittest.TestCase):
    def test_api_reads_imagemask_from_both_payloads(self):
        src = API_PY.read_text(encoding="utf-8")
        self.assertEqual(
            src.count('payload.get("imagemask") or None'), 2,
            "preview() and _run_swap() must both forward the brush mask; a "
            "literal None in either one silently discards it")

    def test_preview_returns_keypoints(self):
        """ref_kps is what maps a frame-space mask onto the aligned crop."""
        src = API_PY.read_text(encoding="utf-8")
        self.assertIn('"kps": kps_list', src)

    def test_frontend_sends_imagemask_everywhere_it_renders(self):
        """Every render path must carry the brush mask.

        This counted three literal `imagemask: maskJson` occurrences — one per
        payload built by hand. The two swap payloads are now one builder, so the
        count is 2 and a count is the wrong measure anyway: it passes just as
        happily if a third call site is added that forgets the mask. Assert the
        structure instead — the preview builder spells it out, and EVERY
        /api/swap body goes through buildSwapPayload, which spells it out once.
        """
        src = FACESWAP_JSX.read_text(encoding="utf-8")
        # The preview path and the swap builder each state it literally.
        self.assertIn("imagemask: maskJson", src)
        builder = re.search(r"const buildSwapPayload = .*?\n  \};", src, re.S)
        self.assertIsNotNone(builder, "buildSwapPayload must exist")
        self.assertIn("imagemask: maskJson", builder.group(0),
                      "the shared swap payload builder must forward the brush mask")
        # …and no /api/swap call may hand-roll a body around it.
        for body in re.findall(r"postJSON\(\s*'/api/swap'\s*,\s*(.*?)\)\s*;", src, re.S):
            self.assertIn("buildSwapPayload()", body,
                          f"an /api/swap call bypasses the shared builder: {body[:120]}")

    def test_export_canvas_is_white(self):
        src = PREVIEW_JSX.read_text(encoding="utf-8")
        self.assertIn("'#ffffff'", src,
                      "the exported twin canvas must paint solid white")
        self.assertIn("maskExportRef", src)

    def test_brush_size_is_converted_to_canvas_space(self):
        """brushSize is a screen measurement; the canvas is at native res."""
        src = PREVIEW_JSX.read_text(encoding="utf-8")
        self.assertIn("canvas.width / rect.width", src)


if __name__ == "__main__":
    unittest.main()
