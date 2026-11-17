"""Run Studio's scan through the render's actual ordered tracking pre-pass."""
from roop import globals as runtime
from roop.capturer import _probe_video
from roop.ProcessMgr import ProcessMgr
from roop.ProcessOptions import ProcessOptions
from roop.video_muxer import probe_color


def scan(path, payload):
    info = _probe_video(path)
    if not info:
        raise ValueError('Cannot probe the selected video')
    # Full-source, absolute decoder numbering: trimming later never shifts the
    # track IDs or changes the meaning of saved keys/bindings.
    count = int(info.get('frames') or 0)
    if count < 1:
        raise ValueError('Pre-Scan needs a video with a known frame count')
    cfg = runtime.CFG
    get = lambda key, default: payload.get(key, getattr(cfg, key, default))
    options = ProcessOptions({}, get('max_face_distance', 0.75), 1.0, 'all', 0,
                             '', None, 1, 256, False, False,
                             stabilize_face=get('stabilize_face', True),
                             stabilize_method=get('stabilize_method', 'one_euro'),
                             stabilization_strength=get('stabilization_strength', 0.5),
                             pose_adaptive_faceset=get('pose_adaptive_faceset', False))
    options.disable_occlusion_injection = True
    manager = ProcessMgr(None)
    manager.initialize(runtime.INPUT_FACESETS, runtime.TARGET_FACES, options)
    manager._video_fps = float(info.get('fps') or 30)
    manager._temporal_frame_shape = (int(info['h']), int(info['w']), 3)
    manager._temporal_mode = True
    manager._precompute_temporal(path, None, 0, count, count)
    if not runtime.processing:
        raise RuntimeError('Pre-Scan cancelled')
    cache = manager._trajectory
    if cache is None:
        raise RuntimeError('Pre-Scan produced no face trajectory; inspect detector/model diagnostics')
    return {'available': True, 'summary': cache.summary(), 'color': probe_color(path)}
