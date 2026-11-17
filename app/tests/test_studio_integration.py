"""Offline integration contracts for the final Studio wiring.

Real metadata/image/color operations and provider-option construction; injected
memory telemetry tests admission without pretending to be a physical 3060.
"""
import concurrent.futures
import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from roop import studio_parameters as sp, timeline_state as ts


def activate(payload):
    ts.set_active(ts.TimelineState.from_dict(payload))


@pytest.fixture(autouse=True)
def clear_timeline():
    ts.clear_active()
    yield
    ts.clear_active()


def test_all_studio_parameters_roundtrip_and_hold_categories():
    values = dict(detector_model='gnkps', multi_angle_detect=True, stabilization_strength=.7,
                  pose_adaptive_faceset=True, mask_engine='bisenet+xseg', yaw_erosion_bias=-.4,
                  expression_restorer_weight=.3, preserve_inner_mouth=False, pixel_boost=768,
                  ultramax_ramp=384, temporal_restore_alpha=.2, restore_identity_weight=.8,
                  color_transfer_mode='cdf', grain_match_strength=.5, motion_blur_match=.6)
    key = ts.ParameterKeyframe.from_dict({'frame': 20, **values})
    assert key.values() == values
    state = ts.TimelineState.from_dict({'keyframes': [key.to_dict(), {'frame': 40, 'pixel_boost': 256, 'detector_model': 'scrfd'}]})
    assert ts.evaluate_keyframes(state.keyframes, 39)['pixel_boost'] == 768
    assert ts.evaluate_keyframes(state.keyframes, 40)['pixel_boost'] == 256
    assert ts.evaluate_keyframes(state.keyframes, 39)['detector_model'] == 'gnkps'
    assert ts.TimelineState.from_dict(state.to_dict()).to_dict() == state.to_dict()


@pytest.mark.parametrize('field,value', [('multi_angle_detect', 1), ('preserve_inner_mouth', 'false'),
    ('pixel_boost', 384), ('pixel_boost', 512.0), ('detector_model', 'typo'),
    ('temporal_restore_alpha', float('nan')), ('motion_blur_match', 2)])
def test_invalid_parameters_are_rejected(field, value):
    with pytest.raises(ts.TimelineError):
        sp.normalize_payload({field: value})


def test_curves_keep_individual_interpolation_and_thread_contexts():
    activate({'keyframes': [
        {'frame': 0, 'restore_identity_weight': 0., 'grain_match_strength': 0.,
         'interpolations': {'restore_identity_weight': 'hold', 'grain_match_strength': 'ease_in_out'}},
        {'frame': 100, 'restore_identity_weight': 1., 'grain_match_strength': 1.}]})
    def observe(index):
        with sp.frame_scope(index, (2160, 3840, 3)):
            return sp.value('codeformer_fidelity'), sp.value('grain_match_strength'), sp.frame_shape()
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        values = list(pool.map(observe, [0, 25, 50, 100]))
    assert values[1] == (0., .15625, (2160, 3840, 3))
    assert values[3][:2] == (1., 1.)
    assert sp.frame_shape() is None


def test_disabled_binding_and_active_ranges_reach_the_swap_gate():
    activate({'bindings': [{'track_id': 7, 'faceset_path': 'person.fsz', 'enabled': False}]})
    face = {'_track_id': 7}
    kept, dropped = ts.gate_pending([(0, face), (0, {'_track_id': 8})], 1)
    assert dropped == [7] and len(kept) == 1


def test_engine_filenames_change_for_model_precision_shape_profile_and_boost(tmp_path):
    from roop.engine_cache import engine_cache_prefix
    path = tmp_path / 'model.onnx'
    path.write_bytes(b'graph-one')
    base = engine_cache_prefix(path, 'fp32', (1, 3, 256, 256), (1, 2, 8), 256)
    alternatives = [
        engine_cache_prefix(path, 'fp16', (1, 3, 256, 256), (1, 2, 8), 256),
        engine_cache_prefix(path, 'fp32', (1, 3, 512, 512), (1, 2, 8), 256),
        engine_cache_prefix(path, 'fp32', (1, 3, 256, 256), (1, 4, 8), 256),
        engine_cache_prefix(path, 'fp32', (1, 3, 256, 256), (1, 2, 8), 768)]
    path.write_bytes(b'graph-two')
    alternatives.append(engine_cache_prefix(path, 'fp32', (1, 3, 256, 256), (1, 2, 8), 256))
    assert len({base, *alternatives}) == 6
    assert hashlib.sha256(b'graph-one').hexdigest()[:16] in base


def test_provider_owned_engine_prefix_is_applied_and_stable(tmp_path):
    import onnx
    from roop.engine_cache import qualify_providers
    tensor = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, ['B', 3, 256, 256])
    graph = onnx.helper.make_graph([onnx.helper.make_node('Identity', ['x'], ['y'])], 'cache', [tensor],
        [onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, ['B', 3, 256, 256])])
    path = tmp_path / 'model.onnx'
    onnx.save(onnx.helper.make_model(graph), path)
    source = [('TensorrtExecutionProvider', {'trt_fp16_enable': True, 'trt_engine_cache_path': str(tmp_path)})]
    result = qualify_providers(source, str(path))
    assert 'trt_engine_cache_prefix' not in source[0][1]
    assert result == qualify_providers(result, str(path))
    assert 'fp16' in result[0][1]['trt_engine_cache_prefix']
    source[0][1]['trt_fp16_enable'] = '0'
    assert '_fp32_' in qualify_providers(source, str(path))[0][1]['trt_engine_cache_prefix']


def test_offline_parser_alias_keeps_the_existing_fp32_lock(monkeypatch):
    from roop.precision_policy import precision_override_for
    monkeypatch.delenv('ROOP_TRT_MODEL_PRECISION', raising=False)
    assert precision_override_for('face_parser', 'models/face_parser.onnx') == 'fp32'


@pytest.mark.parametrize('size', [256, 512, 768])
@pytest.mark.parametrize('native', [128, 256, 512])
def test_pixel_boost_roundtrips_complete_grids_with_the_matching_model_mask(size, native):
    from roop.procmgr_tiling import PixelBoostMixin
    from roop.ProcessMgr import ProcessMgr
    size = max(size, native)
    grid = (size + native - 1) // native
    yy, xx = np.mgrid[:size, :size]
    image = np.stack([xx*255//size, yy*255//size, (xx+yy)*127//size], axis=-1).astype(np.uint8)
    tiling = PixelBoostMixin()
    tiles = tiling.implode_pixel_boost(image, native, grid)
    result = tiling.explode_pixel_boost(tiles, native, grid, size)
    assert result.shape == image.shape
    assert np.abs(result.astype(float) - image).mean() < .3
    masks = [tile[..., 0].astype(np.float32)/255 for tile in tiles]
    mask = ProcessMgr._explode_mask(masks, native, grid, size)
    assert mask.shape == image.shape[:2]
    assert np.abs(mask - image[..., 0]/255).mean() < .003


def test_vram_batch_cap_accounts_for_pressure_4k_and_small_gpu():
    from roop.hardware_streamer import VramSessionManager, VramBudgetError
    gib = 1024**3
    memory = [10*gib, 12*gib]
    manager = VramSessionManager(memory_query=lambda: memory, reserve_bytes=gib, bytes_per_face=gib)
    assert manager.cap_batch_size(8, (2160, 3840, 3)) == 8
    with manager.lease(8):
        assert manager.cap_batch_size(8) == 1
    assert manager._reserved == 0
    memory[:] = [4*gib, 6*gib]
    assert manager.cap_batch_size(8) == 1
    memory[:] = [gib//2, 12*gib]
    with pytest.raises(VramBudgetError):
        with manager.lease(1):
            pytest.fail('must refuse GPU work before inference')


def test_vram_oom_retry_keeps_all_outputs_in_order_and_releases_leases():
    from roop.hardware_streamer import VramSessionManager
    manager = VramSessionManager(memory_query=lambda: (12*1024**3, 12*1024**3))
    calls = []
    def infer(items):
        calls.append(len(items))
        if len(items) > 2:
            raise RuntimeError('CUDA out of memory')
        return [i * 3 for i in items]
    assert manager.run_batches(list(range(11)), infer, 8) == [i*3 for i in range(11)]
    assert calls[:3] == [8, 4, 2]
    assert manager._active == manager._reserved == 0


def test_offline_model_resolution_never_downloads_and_checks_alias_hash(tmp_path, monkeypatch):
    from roop import model_integrity as mi
    monkeypatch.setattr(mi.urllib.request, 'urlopen', lambda *a, **kw: pytest.fail('offline resolver used network'))
    data = b'local graph'
    (tmp_path/'parser.onnx').write_bytes(data)
    entry = mi.ModelEntry('face_parser.onnx', hashlib.sha256(data).hexdigest(), len(data), (),
                          offline_only=True, aliases=('parser.onnx',))
    assert mi.require_local_model('face_parser.onnx', str(tmp_path), [entry]).endswith('parser.onnx')
    (tmp_path/'parser.onnx').write_bytes(b'wrong graph')
    assert mi.offline_presence(str(tmp_path), [entry])['face_parser.onnx']['status'] == 'corrupt'
    (tmp_path/'parser.onnx').unlink()
    assert mi.offline_presence(str(tmp_path), [entry])['face_parser.onnx']['status'] == 'missing'


def test_lazy_faceset_analysis_is_cached_without_changing_v1_archive(tmp_path, monkeypatch):
    from roop import faceset_builder as builder
    from roop.FaceSet import FaceSet
    import zipfile
    rng = np.random.default_rng(17)
    image = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    face = {'bbox': np.array([0, 0, 128, 128]), 'det_score': .9,
            'kps': np.array([[40, 45], [88, 45], [64, 65], [45, 90], [83, 90]]),
            'embedding': np.ones(512, np.float32)}
    archive = tmp_path/'legacy.fsz'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('0.png', cv2.imencode('.png', image)[1].tobytes())
    original = archive.read_bytes()
    fs = FaceSet(); fs.faces = [face]; fs.ref_images = [image]
    result = builder.cache_missing_metadata(archive, fs, tmp_path/'cache')
    assert result[0]['det_score'] == .9
    assert result[0]['quality']['laplacian_variance'] > 0
    assert all(np.isfinite(result[0]['pose']))
    assert fs.format_version == 1
    monkeypatch.setattr(builder, '_entry', lambda *a: pytest.fail('cache miss on unchanged archive'))
    assert builder.cache_missing_metadata(archive, fs, tmp_path/'cache') == result
    assert archive.read_bytes() == original


@pytest.mark.parametrize('mode', ['lab', 'cdf', 'adain'])
def test_color_modes_execute_real_masked_operations(mode):
    from roop.procmgr_color import ColorTransferMixin
    rng = np.random.default_rng(18)
    source = rng.integers(40, 160, (64, 64, 3), dtype=np.uint8)
    target = np.clip(source.astype(np.int16) + [10, 30, 5], 0, 255).astype(np.uint8)
    activate({'keyframes': [{'frame': 0, 'color_transfer_mode': mode}]})
    with sp.frame_scope(0):
        result = ColorTransferMixin().apply_color_transfer(source, target, mask=np.ones((64, 64), np.float32))
    assert result.dtype == np.uint8 and result.shape == source.shape
    assert np.abs(result.astype(float) - source).mean() > 1


def test_color_transfer_alias_and_hdr_to_sdr_tags():
    from roop.video_muxer import extract_color_tags, output_plan
    tags = extract_color_tags({'color_space': 'bt709', 'color_trc': 'smpte2084',
                               'color_primaries': 'bt2020', 'color_range': 'pc'})
    assert tags['color_transfer'] == 'smpte2084'
    plan = output_plan({'color': tags}, hdr_to_sdr=True)
    assert plan.transfer == plan.matrix == plan.primaries == 'bt709'
    assert plan.range == 'tv'


def test_scalar_fidelity_input_is_warmed_by_a_real_cpu_session(tmp_path):
    import onnx
    import onnxruntime as ort
    from roop.predictor import warmup_session
    h = onnx.helper
    inputs = [h.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1, 3, 16, 16]),
              h.make_tensor_value_info('w', onnx.TensorProto.DOUBLE, [])]
    graph = h.make_graph([h.make_node('Cast', ['w'], ['wf'], to=onnx.TensorProto.FLOAT),
                          h.make_node('Add', ['x', 'wf'], ['y'])], 'scalar', inputs,
                         [h.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1, 3, 16, 16])])
    model = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)], ir_version=8)
    path = tmp_path/'scalar.onnx'; onnx.save(model, path)
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    original, feeds = session.run, []
    def observe(names, feed):
        feeds.append(feed)
        return original(names, feed)
    session.run = observe
    assert warmup_session(session, 'studio-scalar', once=False)
    assert len(feeds) == 1 and feeds[0]['w'].shape == ()


def test_animated_stabilization_updates_existing_filters_and_zero_resets_history():
    from roop.temporal_stabilizer import FaceTrackStabilizer
    stabilizer = FaceTrackStabilizer()
    points = np.array([[40, 45], [88, 45], [64, 65], [45, 90], [83, 90]], np.float32)
    stabilizer.update(1, frame_index=0, landmarks=points, bbox=[0, 0, 128, 128])
    state = stabilizer._tracks[1]
    stabilizer.set_strength(1)
    assert state.landmarks_filter.min_cutoff == pytest.approx(.05)
    stabilizer.set_strength(.25)
    assert state.landmarks_filter.min_cutoff == pytest.approx(.3875)
    stabilizer.set_strength(0)
    assert not stabilizer._tracks


def test_real_hdr_decode_to_segmented_sdr_preserves_the_working_view_and_sdr_tags(tmp_path, monkeypatch):
    import subprocess
    from roop import hdr_pipeline as hp
    from roop.segment_writer import SegmentedVideoWriter
    from roop.video_muxer import probe_color
    from roop.capturer import _probe_video
    source, output = tmp_path/'pq.mp4', tmp_path/'sdr.mp4'
    monkeypatch.setenv('ROOP_HDR', 'tonemap')
    monkeypatch.setenv('ROOP_RESUME_CHUNK', '4')
    command = [hp.ffmpeg_binary(), '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'lavfi', '-i', 'testsrc2=size=128x96:rate=24', '-frames:v', '12',
        '-vf', 'setparams=range=tv:color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc,format=yuv420p10le',
        '-c:v', 'libx265', '-preset', 'ultrafast', '-x265-params', 'lossless=1:log-level=error', str(source)]
    subprocess.run(command, check=True, capture_output=True, timeout=30)
    reader = writer = None
    try:
        spec = hp.begin_session(str(source), str(output))
        assert spec is not None and hp.output_to_sdr()
        assert hp.session_descriptor()['output_mode'] == 'tonemap'
        assert probe_color(source)['hdr']
        reader = hp.HdrFrameReader(str(source), spec, fps=24, hwaccel='')
        writer = SegmentedVideoWriter(str(output), (128, 96), 24, codec='libx264', crf=0,
                                      source_video=str(source), frame_end=12)
        expected = []
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            expected.append(frame.copy())
            writer.write_frame(frame)
        reader.release(); reader = None
        writer.close(); writer = None
        tags = probe_color(output)
        assert (tags['color_space'], tags['color_trc'], tags['color_primaries'], tags['color_range']) == ('bt709', 'bt709', 'bt709', 'tv')
        assert not tags['hdr'] and _probe_video(str(output))['frames'] == 12
        decoded = subprocess.run([hp.ffmpeg_binary(), '-v', 'error', '-i', str(output),
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], check=True, capture_output=True, timeout=30).stdout
        actual = np.frombuffer(decoded, np.uint8).reshape(-1, 96, 128, 3)
        assert np.abs(actual.astype(float) - np.stack(expected)).mean() < 4
    finally:
        if reader is not None:
            reader.release()
        if writer is not None:
            writer.abort()
        hp.end_session()
