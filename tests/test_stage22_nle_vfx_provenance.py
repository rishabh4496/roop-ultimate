import json
import math
import os
import socket
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import cv2

from app.roop.nle_interchange import (
    frames_to_timecode,
    timecode_to_frames,
    catalog_active_face_timecodes,
    export_cmx3600_edl,
    parse_cmx3600_edl,
    export_otio_json,
    parse_otio_json,
)
from app.roop.vfx_multipass import (
    affine_2x3_to_3x3,
    invert_affine_matrix,
    compute_corner_pin,
    create_combined_matte,
    VfxMultiPassSession,
)
from face_engine.pipeline.provenance import (
    create_provenance_manifest,
    get_stage14_hardware_receipt_hash,
    verify_provenance_manifest,
    sign_and_embed_file,
    extract_and_verify_file,
    _embed_png_provenance,
    _read_png_provenance,
    embed_frequency_watermark,
    extract_frequency_watermark,
    verify_frequency_watermark,
)


class TestStage22NleInterchange(unittest.TestCase):
    """Test frame-accurate SMPTE timecode math across all broadcast and cinema frame rates."""

    def test_timecode_math_all_fps(self):
        fps_list = [23.976, 24.0, 25.0, 29.97, 30.0, 59.94, 60.0]
        test_frames = [0, 1, 24, 30, 60, 100, 1000, 1798, 1800, 5000, 17982, 35964]

        for fps in fps_list:
            for f in test_frames:
                tc_ndf = frames_to_timecode(f, fps=fps, drop_frame=False)
                parsed_f = timecode_to_frames(tc_ndf, fps=fps, drop_frame=False)
                self.assertEqual(
                    f, parsed_f,
                    f"NDF round-trip mismatch at fps={fps}, frame={f}: tc={tc_ndf}, parsed={parsed_f}"
                )

        # Drop-Frame 29.97 testing
        df_fps = 29.97
        for f in test_frames:
            tc_df = frames_to_timecode(f, fps=df_fps, drop_frame=True)
            self.assertIn(";", tc_df, "Drop frame timecode must contain semicolon delimiter")
            parsed_df = timecode_to_frames(tc_df, fps=df_fps, drop_frame=True)
            self.assertEqual(
                f, parsed_df,
                f"DF round-trip mismatch at frame={f}: tc={tc_df}, parsed={parsed_df}"
            )

    def test_catalog_active_face_timecodes(self):
        # Face ranges with scene cuts
        face_ranges = [(0, 120), (200, 350)]
        scene_cuts = [60, 250]
        catalog = catalog_active_face_timecodes(
            video_path="source_reel.mp4",
            fps=30.0,
            face_ranges=face_ranges,
            scene_cuts=scene_cuts,
            speaker_cluster_id="person_1",
            min_run_length=5,
        )
        self.assertEqual(len(catalog), 4)
        self.assertEqual(catalog[0]["frame_in"], 0)
        self.assertEqual(catalog[0]["frame_out"], 60)
        self.assertEqual(catalog[1]["frame_in"], 60)
        self.assertEqual(catalog[1]["frame_out"], 120)
        self.assertEqual(catalog[2]["frame_in"], 200)
        self.assertEqual(catalog[2]["frame_out"], 250)
        self.assertEqual(catalog[3]["frame_in"], 250)
        self.assertEqual(catalog[3]["frame_out"], 350)
        for entry in catalog:
            self.assertIn("tc_in", entry)
            self.assertIn("tc_out", entry)
            self.assertEqual(entry["speaker_cluster_id"], "person_1")

    def test_edl_cmx3600_export_and_import_roundtrip(self):
        doc = {
            "name": "STUDIO_REEL_01",
            "fps": 24.0,
            "source_media": "raw_plates.mov",
            "swap_media_name": "faceswap_beauty_v2.mov",
            "total_frames": 240,
            "timeline": {
                "fps": 24.0,
                "frame_start": 0,
                "frame_end": 240,
                "face_segments": [
                    {"in": 24, "out": 72, "speaker_cluster_id": "person_0", "clip_name": "swap_shot1.mov"},
                    {"in": 120, "out": 192, "speaker_cluster_id": "person_1", "clip_name": "swap_shot2.mov"},
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            edl_file = os.path.join(tmpdir, "cutlist.edl")
            edl_text = export_cmx3600_edl(doc, output_path=edl_file, fps=24.0)
            self.assertTrue(os.path.exists(edl_file))
            self.assertIn("TITLE: STUDIO_REEL_01", edl_text)
            self.assertIn("001  AX       V     C", edl_text)
            self.assertIn("* TRACK: V1", edl_text)
            self.assertIn("002  SWAP     V     C", edl_text)
            self.assertIn("* TRACK: V2", edl_text)

            cuts = parse_cmx3600_edl(edl_text, fps=24.0)
            self.assertEqual(len(cuts), 3)  # V1 source + 2 V2 cuts
            self.assertEqual(cuts[1]["frame_in"], 24)
            self.assertEqual(cuts[1]["frame_out"], 72)
            self.assertEqual(cuts[1]["speaker_cluster_id"], "person_0")
            self.assertEqual(cuts[2]["frame_in"], 120)
            self.assertEqual(cuts[2]["frame_out"], 192)
            self.assertEqual(cuts[2]["speaker_cluster_id"], "person_1")

    def test_otio_json_export_and_import_roundtrip(self):
        doc = {
            "name": "OTIO_STUDIO_TIMELINE",
            "fps": 25.0,
            "source_media": "source_master.mp4",
            "total_frames": 500,
            "timeline": {
                "fps": 25.0,
                "frame_start": 0,
                "frame_end": 500,
                "face_segments": [
                    {"in": 50, "out": 150, "speaker_cluster_id": "person_0"},
                    {"in": 300, "out": 450, "speaker_cluster_id": "person_2"},
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            otio_file = os.path.join(tmpdir, "timeline.otio.json")
            otio_text = export_otio_json(doc, output_path=otio_file, fps=25.0)
            self.assertTrue(os.path.exists(otio_file))

            cuts = parse_otio_json(otio_text)
            self.assertEqual(len(cuts), 3)  # 1 source clip on V1 + 2 swap clips on V2
            # V2 clips (track named "V2_FaceSwap")
            v2_cuts = [c for c in cuts if "V2" in c.get("track", "")]
            self.assertEqual(len(v2_cuts), 2)
            self.assertEqual(v2_cuts[0]["frame_in"], 50)
            self.assertEqual(v2_cuts[0]["frame_out"], 150)
            self.assertEqual(v2_cuts[0]["speaker_cluster_id"], "person_0")
            self.assertEqual(v2_cuts[1]["frame_in"], 300)
            self.assertEqual(v2_cuts[1]["frame_out"], 450)
            self.assertEqual(v2_cuts[1]["speaker_cluster_id"], "person_2")


class TestStage22VfxMultiPass(unittest.TestCase):
    """Test VFX Multi-Pass outputs, inverse affine matrices, corner pins, and combined mattes."""

    def test_affine_inversion_and_corner_pin(self):
        # Create a standard 2x3 affine matrix (scale 1.5, translation 10, 20)
        M_2x3 = np.array([
            [1.5, 0.0, 10.0],
            [0.0, 1.5, 20.0],
        ], dtype=np.float64)

        M_3x3_list = affine_2x3_to_3x3(M_2x3)
        self.assertEqual(len(M_3x3_list), 3)
        self.assertEqual(len(M_3x3_list[0]), 3)
        self.assertEqual(M_3x3_list[2][2], 1.0)

        # Invert matrix
        M_inv = invert_affine_matrix(M_2x3)
        self.assertEqual(M_inv.shape, (2, 3))

        # Corner pins computation
        pin = compute_corner_pin(M_2x3, crop_w=512, crop_h=512)
        self.assertEqual(len(pin["to1"]), 2)
        self.assertEqual(len(pin["to2"]), 2)
        self.assertEqual(len(pin["to3"]), 2)
        self.assertEqual(len(pin["to4"]), 2)
        self.assertEqual(pin["from1"], [0.0, 0.0])
        self.assertEqual(pin["from2"], [512.0, 0.0])
        self.assertEqual(pin["from3"], [512.0, 512.0])
        self.assertEqual(pin["from4"], [0.0, 512.0])

    def test_combined_matte_generation(self):
        # 100x100 frame
        base_matte = np.full((100, 100), 255, dtype=np.uint8)
        xseg_mask = np.full((100, 100), 200, dtype=np.uint8)
        roto_mask = np.zeros((100, 100), dtype=np.uint8)
        roto_mask[20:50, 20:50] = 255

        combined = create_combined_matte(
            base_matte,
            xseg_mask=xseg_mask,
            roto_polygons_mask=roto_mask,
            feather_radius=1.5,
        )
        self.assertEqual(combined.shape, (100, 100))
        self.assertTrue(np.all(combined >= 0.0) and np.all(combined <= 1.0))
        # Inside roto region should have significant intensity
        self.assertGreater(combined[30, 30], 0.5)
        # Outside roto region should be near zero
        self.assertLess(combined[10, 10], 0.05)

    def test_vfx_multi_pass_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = VfxMultiPassSession(output_dir=tmpdir, fps=30.0)
            crop_img = np.full((64, 64, 3), 128, dtype=np.uint8)
            M_2x3 = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]])

            rec1 = session.record_frame(
                frame_index=0,
                faces_data=[{"crop": crop_img, "matrix": M_2x3}],
            )
            self.assertEqual(rec1["frame_index"], 0)
            self.assertEqual(len(rec1["faces"]), 1)

            rec2 = session.record_frame(
                frame_index=1,
                faces_data=[{"crop": crop_img, "matrix": M_2x3}],
            )
            self.assertEqual(rec2["frame_index"], 1)

            manifest_path = session.finalize()
            self.assertTrue(os.path.exists(manifest_path))
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["version"], "1.0")
            self.assertEqual(len(data["frames"]), 2)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "vfx_crops", "crop_000000_0.png")))


class TestStage22CryptographicProvenance(unittest.TestCase):
    """Test 100% offline local cryptographic provenance signer, DCT watermarking, and zero network calls."""

    def test_zero_external_network_calls(self):
        """Strictly verify that provenance signing and metadata extraction perform zero network calls."""
        with patch("socket.socket.connect") as mock_connect, \
             patch("socket.getaddrinfo") as mock_getaddr:
            mock_connect.side_effect = RuntimeError("Network call attempted during offline provenance signing!")
            mock_getaddr.side_effect = RuntimeError("DNS resolution attempted during offline provenance signing!")

            # Execute full provenance pipeline
            manifest = create_provenance_manifest(
                payload_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                disclosure="100% Offline Synthetic Media Verification",
            )
            self.assertEqual(manifest["schema"], "roop.provenance.v1")
            self.assertIn("signature", manifest)
            valid, reason = verify_provenance_manifest(manifest, payload_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
            self.assertTrue(valid, f"Verification failed: {reason}")

            hw_hash = get_stage14_hardware_receipt_hash()
            self.assertIsInstance(hw_hash, str)
            self.assertEqual(len(hw_hash), 64)

            # Assert network was never touched
            mock_connect.assert_not_called()
            mock_getaddr.assert_not_called()

    def test_png_metadata_embedding_and_extraction(self):
        dummy_img = np.zeros((16, 16, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = os.path.join(tmpdir, "test.png")
            cv2.imwrite(png_path, dummy_img)

            # Embed provenance
            success, manifest = sign_and_embed_file(
                png_path,
                disclosure="Roop Ultimate Studio Verification",
            )
            self.assertTrue(success)
            self.assertIsNotNone(manifest)

            # Extract provenance
            valid, ext_manifest, reason = extract_and_verify_file(png_path)
            self.assertTrue(valid, f"Verification failed: {reason}")
            self.assertEqual(ext_manifest["schema"], "roop.provenance.v1")
            self.assertEqual(ext_manifest["synthetic_media_disclosure"], "Roop Ultimate Studio Verification")

    def test_dct_frequency_watermark_psnr_and_recovery(self):
        # 128x128 image gives 16x16 = 256 blocks of 8x8 (enough for 32 bytes = 256 bits)
        y, x = np.mgrid[0:128, 0:128]
        img = np.uint8((np.sin(x / 8.0) + np.cos(y / 8.0) + 2.0) * 50)
        bgr = cv2.merge([img, img, img])

        payload_bytes = b"ROOP_STAGE22_PROVENANCE_HMAC_123"  # 32 bytes = 256 bits
        watermarked = embed_frequency_watermark(bgr, payload_bytes, delta=4.0)

        # Compute PSNR between original and watermarked
        mse = np.mean((bgr.astype(np.float64) - watermarked.astype(np.float64)) ** 2)
        psnr = 10.0 * math.log10((255.0 ** 2) / max(1e-9, mse))
        # Requirement: PSNR > 42 dB (imperceptible)
        self.assertGreater(psnr, 42.0, f"Watermark PSNR {psnr:.2f} dB is below required 42.0 dB threshold")

        # Extract and verify watermark payload
        extracted_bytes = extract_frequency_watermark(watermarked, length_bytes=len(payload_bytes))
        self.assertEqual(extracted_bytes, payload_bytes)
        self.assertTrue(verify_frequency_watermark(watermarked, payload_bytes))


if __name__ == "__main__":
    unittest.main()
