"""Phase 4 tests for FaceSet V2 serialization, migration, and lookup."""

import hashlib
import os
import sys
import tempfile
import unittest
import zipfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.FaceSet import FaceSet  # noqa: E402
from roop.faceset_v2 import (  # noqa: E402
    FORMAT_VERSION,
    POSE_BINS,
    POSE_MATRIX_CELLS,
    migrate_legacy_fsz,
    parse_pose_matrix_key,
    pose_matrix_cell,
    pose_matrix_key,
    prepare_faceset_v2,
    read_faceset_archive,
    select_reference_index,
    validate_metadata,
    write_faceset_v2,
)


def _face(x, pose=(0.0, 0.0, 0.0), size=180.0, identity=0, score=0.96):
    # A valid 68-point shape is sufficient for deterministic geometry and
    # expression extraction; this test does not need an inference provider.
    lm = np.zeros((68, 2), dtype=np.float32)
    lm[:, 0] = x + size * 0.5
    lm[:, 1] = 80.0 + size * 0.5
    lm[36:42] = [[x + size * .28, 80 + size * .34],
                 [x + size * .34, 80 + size * .31],
                 [x + size * .40, 80 + size * .34],
                 [x + size * .40, 80 + size * .39],
                 [x + size * .34, 80 + size * .41],
                 [x + size * .28, 80 + size * .39]]
    lm[42:48] = lm[36:42] + np.array([size * .32, 0], dtype=np.float32)
    lm[48] = [x + size * .30, 80 + size * .70]
    lm[51] = [x + size * .50, 80 + size * .68]
    lm[54] = [x + size * .70, 80 + size * .70]
    lm[57] = [x + size * .50, 80 + size * .76]
    emb = np.zeros(8, dtype=np.float32)
    emb[identity] = 1.0
    pitch, yaw, roll = pose
    return {
        "bbox": np.array([x, 80, x + size, 80 + size], dtype=np.float32),
        "landmark_2d_68": lm,
        "landmark_3d_68": np.column_stack([lm, np.linspace(-2, 2, 68)]).astype(np.float32),
        "pose": np.array([pitch, yaw, roll], dtype=np.float32),
        "embedding": emb,
        "det_score": score,
        "landmark_confidence": 0.94,
    }


def _image(seed, blur=False):
    rng = np.random.default_rng(seed)
    img = rng.integers(20, 235, (320, 320, 3), dtype=np.uint8)
    cv2.rectangle(img, (80, 60), (240, 260), (150, 125, 110), -1)
    cv2.circle(img, (135, 135), 12, (20, 20, 20), -1)
    cv2.circle(img, (185, 135), 12, (20, 20, 20), -1)
    if blur:
        img = cv2.GaussianBlur(img, (31, 31), 0)
    return img


class FaceSetV2Test(unittest.TestCase):

    def test_prepare_contains_required_identity_geometry_quality_appearance_expression_details(self):
        metadata, selected = prepare_faceset_v2(
            [_face(70, pose=(0, -55, 0))], [_image(1)], min_quality=0.0)
        self.assertEqual(selected, [0])
        self.assertEqual(metadata["version"], FORMAT_VERSION)
        entry = metadata["sources"][0]
        for section in ("identity", "geometry", "quality", "appearance", "expression", "identity_details"):
            self.assertIn(section, entry)
        self.assertEqual(len(entry["geometry"]["landmarks_68"]), 68)
        self.assertEqual(len(entry["geometry"]["landmarks_68_3d"]), 68)
        self.assertIn("profile_left", metadata["pose_bank"])
        self.assertEqual(metadata["pose_bank"]["strong_left"], [0])
        self.assertIsNotNone(metadata["identity"]["normalized_embedding"])
        detail = entry["identity_details"]["high_frequency"]
        self.assertEqual(detail["shape"], [64, 64])
        self.assertEqual(len(detail["residual_q"]), 64 * 64)
        self.assertEqual(metadata["identity_details"]["high_frequency"]["source_count"], 1)

    def test_pose_specific_embeddings_are_not_replaced_by_global_average(self):
        faces = [_face(70, identity=0), _face(70, pose=(0, 60, 0), identity=1)]
        metadata, _ = prepare_faceset_v2(faces, [_image(2), _image(3)], min_quality=0.0)
        vectors = [entry["identity"]["normalized_embedding"] for entry in metadata["sources"]]
        self.assertNotEqual(vectors[0], vectors[1])
        self.assertNotEqual(vectors[0], metadata["identity"]["normalized_embedding"])

    def test_quality_filter_and_redundancy_keep_complementary_high_quality_sources(self):
        faces = [
            _face(70, pose=(0, 0, 0), identity=0),
            _face(70, pose=(0, 0, 0), identity=0),
            _face(70, pose=(0, 55, 0), identity=0),
            _face(70, pose=(0, -55, 0), identity=0, score=0.1),
        ]
        images = [_image(10), _image(10), _image(11), _image(12, blur=True)]
        metadata, selected = prepare_faceset_v2(
            faces, images, min_quality=0.35, max_entries=8, max_per_bin=3)
        self.assertLess(len(selected), 4)
        self.assertEqual(len(metadata["pose_bank"]["frontal"]), 1)
        self.assertTrue(metadata["rejected"])

    def test_round_trip_is_deterministic_and_checksum_validated(self):
        fs = FaceSet()
        fs.faces = [_face(70), _face(70, pose=(0, 35, 0))]
        images = [_image(20), _image(21)]
        with tempfile.TemporaryDirectory() as td:
            a, b = os.path.join(td, "a.fsz"), os.path.join(td, "b.fsz")
            first = write_faceset_v2(a, fs, images, source_name="person", min_quality=0.0)
            second = write_faceset_v2(b, fs, images, source_name="person", min_quality=0.0)
            self.assertEqual(first, second)
            with open(a, "rb") as handle:
                hash_a = hashlib.sha256(handle.read()).digest()
            with open(b, "rb") as handle:
                hash_b = hashlib.sha256(handle.read()).digest()
            self.assertEqual(hash_a, hash_b)
            loaded = read_faceset_archive(a)
            self.assertEqual(validate_metadata(loaded)["version"], FORMAT_VERSION)
            with zipfile.ZipFile(a) as zf:
                self.assertIn("metadata.json", zf.namelist())
                self.assertIn("0.png", zf.namelist())
                self.assertIn("1.png", zf.namelist())

    def test_faceset_attach_and_lookup_use_v2_metadata(self):
        fs = FaceSet()
        fs.faces = [_face(70, pose=(0, -50, 0)), _face(70, pose=(0, 50, 0))]
        metadata, _ = prepare_faceset_v2(fs.faces, [_image(30), _image(31)], min_quality=0.0)
        fs.attach_v2_metadata(metadata)
        self.assertEqual(fs.format_version, 2)
        self.assertEqual(fs.select_reference_index(pose=(50, 0)), 1)
        self.assertEqual(len(fs.face_poses), 2)
        self.assertEqual(select_reference_index(metadata, pose=(-50, 0)), 0)

    def test_legacy_archive_is_detected_and_migrated_without_losing_pngs(self):
        with tempfile.TemporaryDirectory() as td:
            old, new = os.path.join(td, "old.fsz"), os.path.join(td, "new.fsz")
            with zipfile.ZipFile(old, "w") as zf:
                for i, image in enumerate((_image(40), _image(41))):
                    ok, buf = cv2.imencode(".png", image)
                    self.assertTrue(ok)
                    zf.writestr(f"{i}.png", bytes(buf))
            self.assertIsNone(read_faceset_archive(old))
            metadata = migrate_legacy_fsz(old, new)
            self.assertEqual(metadata["version"], 2)
            self.assertIsNotNone(read_faceset_archive(new))
            with zipfile.ZipFile(new) as zf:
                self.assertEqual(sorted(n for n in zf.namelist() if n.endswith(".png")), ["0.png", "1.png"])

    def test_corrupt_archive_fails_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            bad = os.path.join(td, "bad.fsz")
            with open(bad, "wb") as handle:
                handle.write(b"not a zip")
            with self.assertRaises(ValueError):
                read_faceset_archive(bad)

    def test_checksum_corruption_is_rejected_even_when_zip_crc_is_rewritten(self):
        fs = FaceSet()
        fs.faces = [_face(70)]
        with tempfile.TemporaryDirectory() as td:
            good, bad = os.path.join(td, "good.fsz"), os.path.join(td, "bad.fsz")
            write_faceset_v2(good, fs, [_image(50)], min_quality=0.0)
            with zipfile.ZipFile(good, "r") as source, zipfile.ZipFile(bad, "w") as target:
                for name in source.namelist():
                    data = source.read(name)
                    if name == "0.png":
                        data = bytes([data[0] ^ 0x01]) + data[1:]
                    target.writestr(name, data)
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_faceset_archive(bad)

    def test_v1_average_stays_legacy_and_v2_average_is_noop(self):
        v1 = FaceSet()
        v1.faces = [_face(70, identity=0), _face(70, identity=1)]
        before = v1.faces[0]["embedding"].copy()
        v1.AverageEmbeddings()
        self.assertIsNotNone(v1.embeddings_backup)
        self.assertFalse(np.array_equal(before, v1.faces[0]["embedding"]))

        v2 = FaceSet()
        v2.faces = [_face(70, identity=0), _face(70, identity=1)]
        v2.format_version = 2
        before = v2.faces[0]["embedding"].copy()
        v2.AverageEmbeddings()
        np.testing.assert_array_equal(before, v2.faces[0]["embedding"])


# ---------------------------------------------------------------------------
# V2 quality pre-screening, 3x3 pose matrix, dermal patch and the specified
# top-level surface (version / default_embedding / pose_bins / dermal_patch),
# plus V1 fall-back.
# ---------------------------------------------------------------------------

def _posed_face(x, yaw=0.0, pitch=0.0, size=180.0, identity=0, score=0.96):
    """A `_face` whose insightface-convention pose encodes yaw/pitch."""
    return _face(x, pose=(pitch, yaw, 0.0), size=size, identity=identity, score=score)


def _identity_face(x, vector, yaw=0.0, size=180.0):
    """A face carrying an explicit embedding, for the clustering tests.

    `yaw` exists so a fixture can spread its references across pose bins. The
    selector removes near-duplicates WITHIN a bin (cosine distance < 0.08), so
    without that spread a cluster of deliberately-similar genuine references
    gets deduped down to one and the identity gate under test is not what is
    being measured.
    """
    face = _face(x, pose=(0.0, yaw, 0.0), size=size)
    face["embedding"] = np.asarray(vector, dtype=np.float32)
    return face


# Three genuine references, spread far enough apart to survive near-duplicate
# removal (pairwise cosine ~0.83, i.e. distance ~0.17 > the 0.08 threshold)
# while every one still sits ~0.91 from their median centroid, well above the
# 0.70 identity floor. The impostor is orthogonal to all three.
_CLUSTER = ([1.0, 0.45, 0.0, 0.0], [1.0, 0.0, 0.45, 0.0], [1.0, 0.0, 0.0, 0.45])
_IMPOSTOR = [0.0, 0.45, 0.45, 1.0]
_SPREAD_YAWS = (-45.0, 0.0, 45.0)


class FaceSetV2QualityScreeningTest(unittest.TestCase):
    """The two absolute pre-screens: motion blur, then identity outliers."""

    def test_blurred_reference_is_rejected_and_sharp_ones_survive(self):
        images = [_image(2), _image(3), _image(4, blur=True)]
        faces = [_posed_face(70, yaw=y) for y in _SPREAD_YAWS]
        metadata, selected = prepare_faceset_v2(faces, images, min_quality=0.0)
        # Selection order follows pose-bin coverage, not input order.
        self.assertEqual(sorted(selected), [0, 1])
        self.assertEqual(metadata["gates"]["rejected_motion_blur"], 1)
        blur = [r for r in metadata["rejected"] if r["reason"] == "motion_blur"]
        self.assertEqual([r["index"] for r in blur], [2])
        self.assertLess(blur[0]["laplacian_variance"], 100.0)
        self.assertEqual(blur[0]["threshold"], 100.0)

    def test_native_variance_is_recorded_separately_from_the_resized_score(self):
        # The two are computed at different resolutions and must not be
        # conflated: a floor calibrated on one is wrong on the other.
        metadata, _ = prepare_faceset_v2([_posed_face(70)], [_image(5)], min_quality=0.0)
        quality = metadata["sources"][0]["quality"]
        self.assertGreater(quality["laplacian_variance"], 100.0)
        self.assertNotAlmostEqual(quality["laplacian_variance"],
                                  quality["sharpness_variance"], places=3)

    def test_every_reference_blurred_fails_loudly_rather_than_silently(self):
        images = [_image(6, blur=True), _image(7, blur=True)]
        faces = [_posed_face(70, yaw=-45.0), _posed_face(70, yaw=45.0)]
        with self.assertRaises(ValueError) as ctx:
            prepare_faceset_v2(faces, images, min_quality=0.0)
        self.assertIn("motion-blur floor", str(ctx.exception))

    def test_blur_floor_is_configurable_and_zero_disables_it(self):
        images = [_image(8), _image(9, blur=True)]
        faces = [_posed_face(70, yaw=-45.0), _posed_face(70, yaw=45.0)]
        _, selected = prepare_faceset_v2(faces, images, min_quality=0.0,
                                         laplacian_floor=0.0)
        self.assertEqual(sorted(selected), [0, 1])

    def test_wrong_person_is_rejected_against_the_cluster_median(self):
        faces = [_identity_face(70, v, yaw=y) for v, y in zip(_CLUSTER, _SPREAD_YAWS)]
        faces.append(_identity_face(70, _IMPOSTOR, yaw=0.0))   # a different person
        images = [_image(10 + i) for i in range(4)]
        metadata, selected = prepare_faceset_v2(faces, images, min_quality=0.0)
        self.assertEqual(sorted(selected), [0, 1, 2])
        outliers = [r for r in metadata["rejected"] if r["reason"] == "identity_outlier"]
        self.assertEqual([r["index"] for r in outliers], [3])
        self.assertLess(outliers[0]["cosine_similarity"], 0.70)
        self.assertEqual(metadata["gates"]["min_identity_cosine"], 0.70)

    def test_outlier_gate_uses_the_median_so_one_impostor_cannot_drag_it(self):
        # With a mean centroid a single distant vector pulls the reference
        # towards itself; the median is unmoved by it. Three genuine references
        # plus one impostor is exactly the case that distinguishes them.
        faces = [_identity_face(70, v, yaw=y) for v, y in zip(_CLUSTER, _SPREAD_YAWS)]
        faces.append(_identity_face(70, _IMPOSTOR, yaw=0.0))
        images = [_image(20 + i) for i in range(4)]
        metadata, selected = prepare_faceset_v2(faces, images, min_quality=0.0)
        self.assertEqual(sorted(selected), [0, 1, 2])
        # The mean of the four is dragged towards the impostor; the median is
        # not, which is the only reason index 3 is caught here.
        self.assertEqual(metadata["gates"]["rejected_identity_outlier"], 1)

    def test_two_references_skip_outlier_rejection_entirely(self):
        # A median over two points is their midpoint and both are equidistant
        # from it, so "outlier" is undefined. Declining to reject beats
        # guessing which of the two is the impostor.
        faces = [_identity_face(70, [1.0, 0.0, 0.0], yaw=-45.0),
                 _identity_face(70, [0.0, 1.0, 0.0], yaw=45.0)]
        images = [_image(30), _image(31)]
        metadata, selected = prepare_faceset_v2(faces, images, min_quality=0.0)
        self.assertEqual(sorted(selected), [0, 1])
        self.assertEqual(metadata["gates"]["rejected_identity_outlier"], 0)

    def test_no_cluster_at_all_declines_to_reject_everything(self):
        # Three mutually orthogonal vectors: every one disagrees with the
        # median, which means the median describes no cluster. Rejecting the
        # whole set on that basis would be a worse answer than not rejecting.
        faces = [_identity_face(70, v, yaw=y) for v, y in zip(
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), _SPREAD_YAWS)]
        images = [_image(40), _image(41), _image(42)]
        metadata, selected = prepare_faceset_v2(faces, images, min_quality=0.0)
        self.assertEqual(sorted(selected), [0, 1, 2])
        self.assertEqual(metadata["gates"]["rejected_identity_outlier"], 0)

    def test_blur_screen_runs_before_identity_clustering(self):
        # A smeared frame must not be allowed to shift the median it would then
        # be measured against, so it has to be gone before clustering starts.
        faces = [_identity_face(70, v, yaw=y) for v, y in zip(_CLUSTER, _SPREAD_YAWS)]
        faces.append(_identity_face(70, _IMPOSTOR, yaw=0.0))
        images = [_image(50), _image(51), _image(52), _image(53, blur=True)]
        metadata, selected = prepare_faceset_v2(faces, images, min_quality=0.0)
        self.assertEqual(sorted(selected), [0, 1, 2])
        reasons = {r["index"]: r["reason"] for r in metadata["rejected"]}
        self.assertEqual(reasons[3], "motion_blur")


class FaceSetV2PoseMatrixTest(unittest.TestCase):
    """The specified 3x3 yaw/pitch matrix, built beside the 9-way yaw ladder."""

    def test_every_cell_of_the_matrix_is_reachable(self):
        cells = {pose_matrix_cell((yaw, pitch))
                 for yaw in (-40.0, 0.0, 40.0) for pitch in (-30.0, 0.0, 30.0)}
        self.assertEqual(cells, set(POSE_MATRIX_CELLS))
        self.assertEqual(len(POSE_MATRIX_CELLS), 9)

    def test_bin_boundaries_match_the_specified_edges(self):
        self.assertEqual(pose_matrix_cell((-25.1, 0.0))[0], "left")
        self.assertEqual(pose_matrix_cell((-25.0, 0.0))[0], "center")
        self.assertEqual(pose_matrix_cell((25.0, 0.0))[0], "center")
        self.assertEqual(pose_matrix_cell((25.1, 0.0))[0], "right")
        self.assertEqual(pose_matrix_cell((0.0, -15.1))[1], "down")
        self.assertEqual(pose_matrix_cell((0.0, -15.0))[1], "center")
        self.assertEqual(pose_matrix_cell((0.0, 15.0))[1], "center")
        self.assertEqual(pose_matrix_cell((0.0, 15.1))[1], "up")

    def test_unknown_pose_resolves_to_the_centre_cell(self):
        self.assertEqual(pose_matrix_cell(None), ("center", "center"))
        self.assertEqual(pose_matrix_cell(()), ("center", "center"))

    def test_key_round_trips_and_rejects_unknown_keys(self):
        for cell in POSE_MATRIX_CELLS:
            self.assertEqual(parse_pose_matrix_key(pose_matrix_key(cell)), cell)
        self.assertIsNone(parse_pose_matrix_key("sideways_center"))
        self.assertIsNone(parse_pose_matrix_key("center"))
        self.assertIsNone(parse_pose_matrix_key(""))

    def test_only_non_empty_cells_are_stored_and_centroids_are_normalized(self):
        faces = [_posed_face(70, yaw=-45.0, pitch=0.0),
                 _posed_face(70, yaw=0.0, pitch=0.0),
                 _posed_face(70, yaw=45.0, pitch=30.0)]
        images = [_image(60), _image(61), _image(62)]
        metadata, _ = prepare_faceset_v2(faces, images, min_quality=0.0)
        bins = metadata["pose_bins"]
        self.assertEqual(set(bins), {"left_center", "center_center", "right_up"})
        for key, cell in bins.items():
            self.assertEqual(parse_pose_matrix_key(key), (cell["yaw_bin"], cell["pitch_bin"]))
            self.assertGreaterEqual(cell["support"], 1)
            self.assertEqual(len(cell["members"]), cell["support"])
            norm = float(np.linalg.norm(np.asarray(cell["embedding"], dtype=np.float32)))
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_matrix_members_index_into_the_selected_sources(self):
        faces = [_posed_face(70, yaw=y) for y in (-45.0, 0.0, 45.0)]
        images = [_image(70), _image(71), _image(72)]
        metadata, _ = prepare_faceset_v2(faces, images, min_quality=0.0)
        count = len(metadata["sources"])
        seen = sorted(i for cell in metadata["pose_bins"].values() for i in cell["members"])
        self.assertEqual(seen, list(range(count)))

    def test_nine_way_pose_bank_is_retained_beside_the_matrix(self):
        # `final_quality_gate` requires `pose_bank`, and the runtime selector
        # drives off `sources`; the 3x3 matrix is additive, never a swap-in.
        faces = [_posed_face(70, yaw=y) for y in (-45.0, 0.0, 45.0)]
        images = [_image(80), _image(81), _image(82)]
        metadata, _ = prepare_faceset_v2(faces, images, min_quality=0.0)
        self.assertIn("pose_bank", metadata)
        self.assertEqual(set(metadata["pose_bank"]) & set(POSE_BINS), set(POSE_BINS))
        self.assertIn("pose_bins", metadata)


class FaceSetV2SurfaceTest(unittest.TestCase):
    """version / default_embedding / pose_bins / dermal_patch."""

    def test_metadata_carries_the_specified_top_level_keys(self):
        faces = [_posed_face(70, yaw=y) for y in (-45.0, 0.0, 45.0)]
        images = [_image(90), _image(91), _image(92)]
        metadata, _ = prepare_faceset_v2(faces, images, min_quality=0.0)
        self.assertEqual(metadata["version"], 2)
        self.assertEqual(metadata["version"], FORMAT_VERSION)
        for key in ("default_embedding", "pose_bins", "dermal_patch"):
            self.assertIn(key, metadata)
        norm = float(np.linalg.norm(np.asarray(metadata["default_embedding"], dtype=np.float32)))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_default_embedding_matches_the_nested_identity_vector(self):
        faces = [_posed_face(70, yaw=y) for y in (-45.0, 0.0)]
        metadata, _ = prepare_faceset_v2(faces, [_image(100), _image(101)], min_quality=0.0)
        np.testing.assert_allclose(
            np.asarray(metadata["default_embedding"], dtype=np.float32),
            np.asarray(metadata["identity"]["normalized_embedding"], dtype=np.float32),
            rtol=0, atol=0)

    def test_dermal_patch_prefers_a_frontal_reference_and_keeps_shape(self):
        faces = [_posed_face(70, yaw=-60.0), _posed_face(70, yaw=0.0), _posed_face(70, yaw=60.0)]
        images = [_image(110), _image(111), _image(112)]
        metadata, _ = prepare_faceset_v2(faces, images, min_quality=0.0)
        patch = metadata["dermal_patch"]
        self.assertTrue(patch["is_frontal"])
        self.assertEqual(patch["frontal_basis"], "pose_cell_center")
        self.assertEqual(metadata["sources"][patch["source_index"]]["pose_cell"], "center_center")
        self.assertEqual(patch["texture"]["shape"], [64, 64])
        for key in ("residual_q", "confidence_q", "mask_q"):
            self.assertEqual(len(patch["texture"][key]), 64 * 64)
        self.assertGreaterEqual(patch["laplacian_variance"], 100.0)

    def test_dermal_patch_accepts_a_nodding_frontal_reference(self):
        # yaw 0 / pitch -20 is `center_down`, not `center_center`. A strict
        # cell test handed the patch to a profile on real material; yaw is what
        # decides whether a face is presented to camera.
        faces = [_posed_face(70, yaw=-60.0), _posed_face(70, yaw=0.0, pitch=-20.0),
                 _posed_face(70, yaw=60.0)]
        images = [_image(114), _image(115), _image(116)]
        metadata, _ = prepare_faceset_v2(faces, images, min_quality=0.0)
        patch = metadata["dermal_patch"]
        self.assertTrue(patch["is_frontal"])
        self.assertEqual(patch["frontal_basis"], "yaw_center")
        self.assertEqual(metadata["sources"][patch["source_index"]]["pose_cell"], "center_down")

    def test_dermal_patch_falls_back_when_nothing_is_frontal(self):
        faces = [_posed_face(70, yaw=-60.0), _posed_face(70, yaw=60.0)]
        metadata, _ = prepare_faceset_v2(faces, [_image(120), _image(121)], min_quality=0.0)
        patch = metadata["dermal_patch"]
        self.assertTrue(patch)
        self.assertFalse(patch["is_frontal"])
        self.assertEqual(patch["frontal_basis"], "none")

    def test_dermal_uv_anchors_come_from_the_projects_own_template(self):
        # Reconstructing these as `arcface_dst * 128/112` is the documented
        # trap; 128 takes the `% 128` branch, which also shifts x by 8.
        from roop.face_util import swap_template_points
        metadata, _ = prepare_faceset_v2([_posed_face(70)], [_image(130)], min_quality=0.0)
        anchors = metadata["dermal_patch"]["uv_anchors"]
        self.assertEqual(anchors["space"], "arcface_128")
        expected = np.asarray(swap_template_points(128, "arcface"), dtype=np.float32) / 128.0
        np.testing.assert_allclose(np.asarray(anchors["uv"], dtype=np.float32),
                                   expected, rtol=0, atol=1e-5)
        self.assertEqual(len(anchors["order"]), 5)

    def test_round_trip_through_the_archive_preserves_the_new_surface(self):
        faces = [_posed_face(70, yaw=y) for y in (-45.0, 0.0, 45.0)]
        images = [_image(140), _image(141), _image(142)]
        faceset = FaceSet()
        faceset.faces = list(faces)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "surface.fsz")
            write_faceset_v2(path, faceset, images, source_name="surface")
            metadata = read_faceset_archive(path)
        self.assertEqual(metadata["version"], 2)
        self.assertIsNotNone(metadata["default_embedding"])
        self.assertTrue(metadata["pose_bins"])
        self.assertTrue(metadata["dermal_patch"])
        validate_metadata(metadata)

    def test_faceset_exposes_the_surface_after_attach(self):
        faces = [_posed_face(70, yaw=y) for y in (-45.0, 0.0, 45.0)]
        images = [_image(150), _image(151), _image(152)]
        metadata, _ = prepare_faceset_v2(faces, images, min_quality=0.0)
        faceset = FaceSet()
        faceset.faces = list(faces)
        faceset.attach_v2_metadata(metadata)
        self.assertEqual(faceset.format_version, 2)
        self.assertEqual(faceset.default_embedding.shape, (8,))
        self.assertAlmostEqual(float(np.linalg.norm(faceset.default_embedding)), 1.0, places=5)
        self.assertTrue(faceset.pose_bins)
        for cell, vector in faceset.pose_bins.items():
            self.assertIn(cell, POSE_MATRIX_CELLS)
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)
        self.assertIsNotNone(faceset.dermal_patch)

    def test_pose_bin_lookup_widens_instead_of_returning_none(self):
        faces = [_posed_face(70, yaw=-45.0, pitch=0.0), _posed_face(70, yaw=0.0, pitch=0.0)]
        metadata, _ = prepare_faceset_v2(faces, [_image(160), _image(161)], min_quality=0.0)
        faceset = FaceSet()
        faceset.faces = list(faces)
        faceset.attach_v2_metadata(metadata)
        exact = faceset.pose_bin_embedding((-45.0, 0.0))
        self.assertIsNotNone(exact)
        np.testing.assert_allclose(faceset.pose_bins[("left", "center")], exact)
        # Same yaw column, a pitch cell that was never populated.
        np.testing.assert_allclose(faceset.pose_bin_embedding((-45.0, 40.0)), exact)
        # An entirely unpopulated column falls all the way back.
        self.assertIsNotNone(faceset.pose_bin_embedding((60.0, 40.0)))
        self.assertIsNone(faceset.pose_bin_embedding((60.0, 40.0), fallback=False))


class FaceSetV1CompatibilityTest(unittest.TestCase):
    """A V1 FaceSet answers the same calls without raising."""

    def _v1_faceset(self):
        faceset = FaceSet()
        faceset.faces = [_identity_face(70, [1.0, 0.0, 0.0], yaw=-45.0),
                         _identity_face(70, [0.0, 1.0, 0.0], yaw=45.0)]
        return faceset

    def test_v1_faceset_reports_version_1_and_an_empty_matrix(self):
        faceset = self._v1_faceset()
        self.assertEqual(faceset.format_version, 1)
        self.assertEqual(faceset.pose_bins, {})
        self.assertIsNone(faceset.dermal_patch)

    def test_v1_default_embedding_is_derived_and_normalized(self):
        faceset = self._v1_faceset()
        vector = faceset.default_embedding
        self.assertEqual(vector.shape, (3,))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)

    def test_v1_default_embedding_is_stable_across_averageembeddings(self):
        # `AverageEmbeddings` overwrites faces[0].embedding in place and stashes
        # the original in `embeddings_backup`; the reported centroid must not
        # change because of that mutation.
        faceset = self._v1_faceset()
        before = faceset.default_embedding.copy()
        faceset.AverageEmbeddings()
        self.assertIsNotNone(faceset.embeddings_backup)
        np.testing.assert_allclose(faceset.default_embedding, before, rtol=0, atol=1e-6)

    def test_v1_pose_lookup_falls_back_to_the_default_embedding(self):
        faceset = self._v1_faceset()
        np.testing.assert_allclose(faceset.pose_bin_embedding((-45.0, 0.0)),
                                   faceset.default_embedding)

    def test_empty_faceset_reports_none_without_raising(self):
        faceset = FaceSet()
        self.assertIsNone(faceset.default_embedding)
        self.assertIsNone(faceset.pose_bin_embedding((0.0, 0.0)))

    def test_legacy_archive_migrates_and_gains_the_new_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = os.path.join(tmp, "legacy.fsz")
            with zipfile.ZipFile(legacy, "w") as zf:
                for i in range(3):
                    zf.writestr(f"{i}.png", cv2.imencode(".png", _image(170 + i))[1].tobytes())
            self.assertIsNone(read_faceset_archive(legacy))
            faceset = FaceSet()
            faceset.faces = [_posed_face(70, yaw=y) for y in (-45.0, 0.0, 45.0)]
            # `migrate_legacy_fsz` returns the metadata and rewrites the
            # archive at `output_path or path`, so the file to read back is the
            # original path, not the return value.
            migrate_legacy_fsz(legacy, faceset=faceset,
                               images=[_image(170 + i) for i in range(3)])
            metadata = read_faceset_archive(legacy)
        self.assertEqual(metadata["version"], 2)
        self.assertIsNotNone(metadata["default_embedding"])
        self.assertTrue(metadata["pose_bins"])


class FaceSetV2NewKeyValidationTest(unittest.TestCase):
    """The new keys are validated, and stay optional for older V2 archives."""

    def _metadata(self):
        faces = [_posed_face(70, yaw=y) for y in (-45.0, 0.0, 45.0)]
        metadata, _ = prepare_faceset_v2(
            faces, [_image(180), _image(181), _image(182)], min_quality=0.0)
        metadata["integrity"] = {"sha256": {
            s["reference_member"]: "0" * 64 for s in metadata["sources"]}}
        return metadata

    def test_an_archive_without_the_new_keys_still_validates(self):
        metadata = self._metadata()
        for key in ("default_embedding", "pose_bins", "dermal_patch"):
            metadata.pop(key)
        validate_metadata(metadata)

    def test_unknown_pose_bin_key_is_rejected(self):
        metadata = self._metadata()
        metadata["pose_bins"]["sideways_center"] = {
            "yaw_bin": "left", "pitch_bin": "center", "members": [0],
            "support": 1, "embedding": None}
        with self.assertRaises(ValueError):
            validate_metadata(metadata)

    def test_out_of_range_pose_bin_member_is_rejected(self):
        metadata = self._metadata()
        next(iter(metadata["pose_bins"].values()))["members"] = [99]
        with self.assertRaises(ValueError):
            validate_metadata(metadata)

    def test_degenerate_default_embedding_is_rejected(self):
        metadata = self._metadata()
        metadata["default_embedding"] = [0.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            validate_metadata(metadata)

    def test_out_of_range_dermal_source_index_is_rejected(self):
        metadata = self._metadata()
        metadata["dermal_patch"]["source_index"] = 99
        with self.assertRaises(ValueError):
            validate_metadata(metadata)

    def test_truncated_dermal_texture_is_rejected(self):
        metadata = self._metadata()
        metadata["dermal_patch"]["texture"]["residual_q"] = [0] * 10
        with self.assertRaises(ValueError):
            validate_metadata(metadata)

    def test_malformed_dermal_uv_anchors_are_rejected(self):
        metadata = self._metadata()
        metadata["dermal_patch"]["uv_anchors"]["uv"] = [[0.0, 0.0]]
        with self.assertRaises(ValueError):
            validate_metadata(metadata)


if __name__ == "__main__":
    unittest.main()
