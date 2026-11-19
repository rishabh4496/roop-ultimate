"""Exercise the files and API contracts consumed by the Studio delivery UI."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from roop import provenance
from roop.ffmpeg_path import ffmpeg_binary
from roop.lut_utils import ffmpeg_lut_filter, validate_cube_file
from roop.nle_interchange import export_cmx3600_edl, export_otio_json
from roop.procmgr_masking import MaskingMixin
from roop.video_stream import NVHardwareVideoWriter
from roop.vfx_multipass import VfxMultiPassSession
from routes_lut import router


CUBE = 'LUT_3D_SIZE 2\n' + '0.0 0.0 0.0\n' * 8


def test_lut_shelf_round_trip_and_path_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', classmethod(lambda _cls: tmp_path))
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        uploaded = client.post('/api/lut/upload', files={'file': ('test.cube', CUBE)})
        assert uploaded.status_code == 200
        path = uploaded.json()['path']
        assert client.get('/api/lut/shelf').json()['files'] == [{'path': path, 'name': 'test.cube'}]
        assert client.post('/api/lut/read', json={'path': path}).json()['cube_text'] == CUBE
        outside = tmp_path / 'private.cube'
        outside.write_text(CUBE)
        assert client.post('/api/lut/read', json={'path': str(outside)}).status_code == 400
        assert client.post('/api/lut/upload', files={'file': ('bad.cube', 'LUT_3D_SIZE 2')}).status_code == 400


def test_lut_mix_uses_red_fastest_order_and_real_ffmpeg(tmp_path):
    cube = tmp_path / "grade's [v1], final.cube"
    cube.write_text(CUBE)
    assert ffmpeg_lut_filter(str(cube), False, 1.0) == (None, None)
    assert ffmpeg_lut_filter(str(cube), True, 0.0) == (None, None)
    for strength, expected in [(0.5, 128), (1.0, 0)]:
        filter_text, mixed = ffmpeg_lut_filter(str(cube), True, strength)
        try:
            if mixed:
                _, _, _, entries = validate_cube_file(mixed)
                assert entries[1] == (0.5, 0, 0)
                assert entries[4] == (0, 0, 0.5)
            completed = subprocess.run([
                ffmpeg_binary(), '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                '-video_size', '16x16', '-i', '-', '-vf', filter_text,
                '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-',
            ], input=bytes([255, 0, 0]) * 256, capture_output=True, check=True, timeout=20)
            assert abs(completed.stdout[0] - expected) <= 1
            assert completed.stdout[1:3] == bytes(2)
        finally:
            if mixed:
                Path(mixed).unlink()


def test_nv_writer_applies_blend_on_software_fallback_path(tmp_path):
    cube = tmp_path / 'black.cube'
    cube.write_text(CUBE)
    output = tmp_path / 'graded.mp4'
    with NVHardwareVideoWriter(str(output), 16, 16, 24, codec='libx264', crf=0,
                               lut_path=str(cube), lut_bake=True, lut_opacity=0.5) as writer:
        mixed = Path(writer._lut_mix_path)
        assert mixed.is_file()
        writer.write_frame(np.full((16, 16, 3), 240, dtype=np.uint8))
    assert not mixed.exists()
    video = cv2.VideoCapture(str(output))
    try:
        ok, frame = video.read()
        assert ok and 116 <= frame.mean() <= 124
    finally:
        video.release()


def test_delivery_matte_png_and_manifest(tmp_path):
    session = VfxMultiPassSession(str(tmp_path), fps=24)
    matte = np.array([[0, 0.5], [0.75, 1]], dtype=np.float32)
    requested = ('xseg3', 'face_parser', 'difference')
    entry = session.record_mattes(21, {'xseg3': matte, 'difference': 1 - matte}, requested=requested)
    image = cv2.imread(str(tmp_path / entry['layers']['xseg3']), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(image, [[0, 128], [191, 255]])
    manifest = json.loads(Path(session.finalize_mattes(requested=requested)).read_text())
    assert manifest['frames'] == [entry]
    assert manifest['unavailable_layers'] == ['face_parser']


def test_matte_sequence_fills_frames_without_faces(tmp_path):
    session = VfxMultiPassSession(str(tmp_path), fps=24, resume_offset=2)
    requested = ('face_parser',)
    session.record_mattes(3, {'face_parser': np.ones((4, 4))}, requested=requested)
    manifest = json.loads(Path(session.finalize_mattes(
        requested=requested, frame_count=5, frame_shape=(4, 4), source_start=50)).read_text())
    assert [row['frame_index'] for row in manifest['frames']] == [2, 3, 4]
    assert manifest['resume_offset'] == 2 and manifest['source_start_frame'] == 50
    assert manifest['frame_counts'] == {'face_parser': 1}
    images = [cv2.imread(str(tmp_path / row['layers']['face_parser']), cv2.IMREAD_GRAYSCALE)
              for row in manifest['frames']]
    assert [int(image.max()) for image in images] == [0, 255, 0]


def test_live_matte_capture_aligns_and_combines_faces_in_frame_space():
    capture = MaskingMixin()
    capture._tls = SimpleNamespace(vfx_mattes={}, vfx_face_channels=set())
    for offset in (0, 1):
        matrix = np.array([[1, 0, -offset], [0, 1, -offset]], dtype=np.float32)
        capture._capture_vfx_mask('mask_xseg3', np.zeros((2, 2)), (2, 2, 3), matrix, (4, 4, 3))
    expected = np.zeros((4, 4))
    expected[:2, :2] = 1
    expected[1:3, 1:3] = 1
    np.testing.assert_array_equal(capture._tls.vfx_mattes['xseg3'], expected)
    assert capture._tls.vfx_face_channels == {'mask_xseg3'}


def test_interchange_marker_uses_exact_trim_relative_position():
    doc = {'timeline': {'fps': 24, 'frame_start': 24, 'frame_end': 72,
                        'markers': [{'frame': 11}, {'frame': 36, 'note': 'Cut here'}, {'frame': 72}],
                        'face_segments': [{'in': 36, 'out': 48}]}}
    edl = export_cmx3600_edl(doc)
    assert edl.count('* MARKER:') == 1
    assert '* MARKER: 00:00:01:12 Cut here' in edl
    otio = json.loads(export_otio_json(doc))
    marker, = otio['tracks']['markers']
    assert marker['marked_range']['start_time']['value'] == 12
    assert marker['metadata']['roop']['note'] == 'Cut here'


def test_provenance_checks_finished_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, 'is_provenance_enabled', lambda: True)
    monkeypatch.setattr(provenance.Path, 'home', classmethod(lambda _cls: tmp_path))
    path = tmp_path / 'beauty.png'
    assert cv2.imwrite(str(path), np.zeros((4, 4, 3), dtype=np.uint8))
    signed = provenance.sign_render_output(str(path))
    assert signed['embedded'] and provenance.verify_render_output(str(path))['file_verified']
    assert not provenance.verify_provenance_manifest(signed['manifest'])[0]
    with path.open('ab') as stream:
        stream.write(b'tampered')
    assert provenance.verify_render_output(str(path))['reason'] == 'Payload hash mismatch'
