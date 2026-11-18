"""Stage 11 flow bar + contact sheet, backend half.

What each group proves (and what it does not):

* ``SoloIds``            the pure id / alias / bypass parsing in ``roop.preview_solo``.
* ``DiffHeatmap``        the difference pass: black where nothing changed, noise floor respected, stats exact.
* ``SweepBindings``      every one of the 17 ``SWEEP_PARAMETERS`` ids is bound to something that READS it (or is
                         explicitly 'none' with a reason). The frontend's list is parsed from source, so a parameter
                         added there without a binding fails here.
* ``PreviewSoloApi``     the real ASGI app and the real ``preview()``: ``live_swap`` is replaced by a recorder, so the
                         test sees which ``roop.globals`` each solo / bypass leaves in force at the moment the swap
                         runs. The GPU models are NOT run - this proves the routing, not the pixels of a real restorer.
* ``SweepApi``           the real ``/api/preview/sweep`` with a recording ``live_swap``: each axis reaches the global
                         that consumes it, a ``CFG`` value is restored and never saved, an identical sweep is reported.

NOT covered: a real XSeg matte, a real BiSeNet parse and a real Restore Ultra difference need the GPU stack and the
weights; they were not run here.
"""
import os
import re
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
os.environ.setdefault("ROOP_SKIP_STARTUP", "1")

from roop import preview_solo as solo  # noqa: E402
from roop import sweep_bindings as bindings  # noqa: E402

REACT_SWEEP_PARAMS = os.path.join(os.path.dirname(APP), 'react-ui', 'src', 'components', 'faceswap', 'sweepParameters.js')


class TestSoloIds:
    def test_every_ui_id_and_legacy_alias_resolves(self):
        want = {'source': 'source', 'detect_align': 'detect_align', '1': 'detect_align', 'detect': 'detect_align',
                'hyperswap': 'hyperswap', '2': 'hyperswap', 'swap': 'hyperswap', 'gaze_mouth': 'gaze_mouth', '3': 'gaze_mouth',
                'xseg_roto': 'xseg_roto', 'mask': 'xseg_roto', '4': 'xseg_roto', 'xseg_3_mask': 'xseg_roto',
                'restore_ultra': 'restore_ultra', 'restore': 'restore_ultra', '5': 'restore_ultra',
                'restore_diff': 'restore_diff', 'relight_lut': 'relight_lut', '6': 'relight_lut',
                'relight_diff': 'relight_diff', 'bisenet_parsing': 'bisenet_parsing',
                'occlusion_alpha': 'occlusion_alpha', 'depth_z_order': 'depth_z_order'}
        for raw, canonical in want.items():
            assert solo.canonical_solo(raw) == canonical, raw
        assert solo.canonical_solo(None) is None and solo.canonical_solo('nonsense') is None

    def test_bypass_accepts_the_react_state_shape_and_keeps_pipeline_order(self):
        assert solo.parse_bypass({'relight_lut': True, 'hyperswap': True, 'xseg_roto': False}) == ('hyperswap', 'relight_lut')
        assert solo.parse_bypass(['restore_ultra', 'gaze_mouth']) == ('gaze_mouth', 'restore_ultra')
        assert solo.parse_bypass('relight_lut, swap') == ('hyperswap', 'relight_lut')

    def test_detect_and_unknown_stages_cannot_be_bypassed(self):
        assert solo.parse_bypass(['detect_align', 'source', 'writer', 'nope']) == ()
        assert solo.parse_bypass(None) == () and solo.parse_bypass('') == () and solo.parse_bypass(5) == ()

    def test_bypass_overrides_force_the_globals_the_stage_reads(self):
        assert solo.bypass_overrides(('restore_ultra',)) == {'selected_enhancer': 'None'}
        assert solo.bypass_overrides(('gaze_mouth',)) == {'expression_restore_strength': 0.0}
        assert solo.bypass_overrides(('relight_lut',))['light_harmonizer'] is False
        assert solo.bypass_overrides(()) == {}

    def test_a_solo_belongs_to_the_stage_it_inspects(self):
        assert solo.bypassed_solo('restore_diff', ('restore_ultra',)) and solo.bypassed_solo('relight_diff', ('relight_lut',))
        assert solo.bypassed_solo('bisenet_parsing', ('xseg_roto',)) and solo.bypassed_solo('xseg_roto', ('xseg_roto',))
        assert not solo.bypassed_solo('restore_ultra', ('xseg_roto',)) and not solo.bypassed_solo(None, ('xseg_roto',))
        assert not solo.bypassed_solo('detect_align', ('restore_ultra',)) and not solo.bypassed_solo('source', ('hyperswap',))

    def test_the_two_unbuilt_passes_say_why(self):
        assert set(solo.UNAVAILABLE) == {'occlusion_alpha', 'depth_z_order'}
        assert all(len(reason) > 20 for reason in solo.UNAVAILABLE.values())

    def test_response_fields_flag_a_refused_solo(self):
        refused = solo.response_fields('depth_z_order', 'depth_z_order', (), applied=False, note='no')
        assert refused['solo_applied'] is False and refused['solo_note'] == 'no' and refused['solo_id'] == 'depth_z_order'
        assert solo.response_fields(None, None, ('restore_ultra',)) == {'bypassed': ['restore_ultra']}


class TestDiffHeatmap:
    def test_identical_frames_are_black_and_report_zero(self):
        a = np.full((40, 60, 3), 120, np.uint8)
        heat, stats = solo.diff_heatmap(a, a.copy())
        assert stats == {'changed_pct': 0.0, 'peak': 0, 'mean_changed': 0.0}
        assert heat.shape == a.shape and int(heat.max()) <= 5          # INFERNO(0) is black

    def test_noise_at_or_below_the_floor_is_ignored(self):
        a = np.full((20, 20, 3), 100, np.uint8)
        b = a.copy()
        b[:, :10] += solo.DIFF_NOISE_FLOOR            # exactly the floor: not counted
        assert solo.diff_heatmap(a, b)[1]['changed_pct'] == 0.0

    def test_a_changed_patch_is_counted_exactly_and_is_bright(self):
        a = np.full((40, 40, 3), 100, np.uint8)
        b = a.copy()
        b[0:10, 0:10] = 160                           # 100 of 1600 px, delta 60
        heat, stats = solo.diff_heatmap(a, b)
        assert stats['changed_pct'] == pytest.approx(6.25) and stats['peak'] == 60 and stats['mean_changed'] == 60.0
        assert heat[0:10, 0:10].mean() > heat[20:, 20:].mean() + 40

    def test_a_size_mismatch_is_resized_not_crashed(self):
        a = np.zeros((30, 30, 3), np.uint8)
        b = np.zeros((60, 60, 3), np.uint8)
        assert solo.diff_heatmap(a, b)[1]['changed_pct'] == 0.0

    def test_empty_difference_notes(self):
        assert 'No restorer' in solo.solo_note_for('restore_diff', enhancer='None')
        assert solo.solo_note_for('restore_diff', enhancer='restore_ultra') == ''
        assert 'Relight is off' in solo.solo_note_for('relight_diff', relight_on=False)
        assert solo.solo_note_for('relight_diff', relight_on=True) == ''


def _react_sweep_ids():
    with open(REACT_SWEEP_PARAMS, encoding='utf-8') as fh:
        return re.findall(r"^\s{4}id:\s*'([a-z_0-9]+)'", fh.read(), flags=re.M)


class TestSweepBindings:
    def test_the_parse_found_the_frontend_list(self):
        assert len(_react_sweep_ids()) >= 17

    def test_every_sweepable_id_is_bound_or_says_why_not(self):
        missing = [i for i in _react_sweep_ids() if i not in bindings.BINDINGS]
        assert missing == [], f'sweepParameters.js lists ids with no backend binding: {missing}'
        for name, b in bindings.BINDINGS.items():
            if b.kind == 'none':
                assert len(b.reason) > 15, name
            else:
                assert b.target, name

    def test_a_global_binding_names_a_real_global(self):
        import roop.globals as g
        for name, b in bindings.BINDINGS.items():
            if b.kind == 'global':
                assert hasattr(g, b.target), f'{name} -> roop.globals.{b.target} does not exist'

    def test_an_alias_drives_the_canonical_global_not_a_phantom(self):
        rt = types.SimpleNamespace(expression_restore_strength=0.0, temporal_enhancer_blend=0.0, codeformer_fidelity=0.5,
                                   merger_motion_blur=0.0, subsample_size=256)
        pl = {}
        for name, value, target in (('expression_restorer_weight', 0.9, 'expression_restore_strength'),
                                    ('temporal_restore_alpha', 0.4, 'temporal_enhancer_blend'),
                                    ('restore_identity_weight', 0.7, 'codeformer_fidelity'),
                                    ('motion_blur_match', 0.3, 'merger_motion_blur')):
            out = bindings.apply(name, value, runtime=rt, cfg=None, tile_payload=pl)
            assert out['bound'] and out['target'] == target and getattr(rt, target) == value, name
        assert not hasattr(rt, 'expression_restorer_weight')

    def test_pixel_boost_sets_the_subsample_size_and_the_upscale_the_options_read(self):
        rt = types.SimpleNamespace(subsample_size=256)
        pl = {}
        bindings.apply('pixel_boost', 512, runtime=rt, cfg=None, tile_payload=pl)
        assert rt.subsample_size == 512 and pl['upscale'] == '512px'

    def test_distance_and_blend_override_the_payload_the_options_read_first(self):
        rt = types.SimpleNamespace(distance_threshold=0.75, blend_ratio=0.8)
        pl = {'face_distance': 0.75, 'blend_ratio': 0.8}
        bindings.apply('max_face_distance', 0.4, runtime=rt, cfg=None, tile_payload=pl)
        bindings.apply('blend_ratio', 0.5, runtime=rt, cfg=None, tile_payload=pl)
        assert rt.distance_threshold == 0.4 and pl['face_distance'] == 0.4 and rt.blend_ratio == 0.5 and pl['blend_ratio'] == 0.5

    def test_a_cfg_axis_is_restored_and_never_saved(self):
        class Cfg:
            face_mask_blend = 20.0
            saved = 0

            def save(self):
                Cfg.saved += 1
        cfg, saved = Cfg(), {}
        bindings.apply('face_mask_blend', 5.0, runtime=types.SimpleNamespace(), cfg=cfg, tile_payload={}, saved_cfg=saved)
        bindings.apply('face_mask_blend', 30.0, runtime=types.SimpleNamespace(), cfg=cfg, tile_payload={}, saved_cfg=saved)
        assert cfg.face_mask_blend == 30.0 and saved == {'face_mask_blend': 20.0}      # the FIRST original is the one kept
        bindings.restore_cfg(cfg, saved)
        assert cfg.face_mask_blend == 20.0 and Cfg.saved == 0

    def test_a_stabilizer_axis_is_honestly_unbound(self):
        info = bindings.describe('stabilization_strength')
        assert info['bound'] is False and 'frames' in info['reason']
        assert bindings.describe('not_a_param')['bound'] is False

    def test_a_bad_value_is_an_error_not_a_crash(self):
        out = bindings.apply('blend_ratio', 'abc', runtime=types.SimpleNamespace(blend_ratio=1.0), cfg=None, tile_payload={})
        assert out['bound'] is False and out['error']


# ══════════════════════════════════════════════════════════════════════════════════════════════
# the real ASGI app
from tests.test_preview_and_timeline_api import api_module, client, _make_video, _ffmpeg  # noqa: E402,F401  (fixtures)

PLUGINS = [types.SimpleNamespace(type='swap'), types.SimpleNamespace(type='mask'), types.SimpleNamespace(type='enhance')]


def _arm(monkeypatch, api_module, record):
    """live_swap replaced by a recorder that returns a frame whose pixels depend on the globals in force."""
    g = api_module.roop_globals

    def fake_live_swap(frame, options, input_facesets=None, *, frame_idx=None, source_video=None):
        record.append({
            'enhancer': str(g.selected_enhancer), 'relight': bool(g.light_harmonizer),
            'expr': float(g.expression_restore_strength), 'types': [getattr(p, 'type', None) for p in options.processors],
            'detail': float(getattr(g, 'restore_ultra_detail_weight', -1)),
            'distance': float(g.distance_threshold), 'blend': float(g.blend_ratio),
            'subsample': int(g.subsample_size), 'mask_blend': float(g.CFG.face_mask_blend),
            'fidelity': float(g.codeformer_fidelity), 'upscale_ok': True,
        })
        out = frame.copy()
        out[8:30, 8:30] = np.clip(out[8:30, 8:30].astype(int) + 40, 0, 255).astype(np.uint8)        # the "swap"
        if str(g.selected_enhancer) != 'None':
            out[10:20, 10:20] = 255 - out[10:20, 10:20]                                              # the "restorer"
        if bool(g.light_harmonizer):
            out[:, :, 2] = np.clip(out[:, :, 2].astype(int) + 25, 0, 255).astype(np.uint8)           # the "relight"
        return out

    monkeypatch.setattr("roop.core.live_swap", fake_live_swap)
    monkeypatch.setattr("roop.core.get_processing_plugins", lambda *a, **k: list(PLUGINS))
    monkeypatch.setattr(api_module.roop_globals, "INPUT_FACESETS", [types.SimpleNamespace(faces=[object()])], raising=False)
    face = {"bbox": np.array([8, 6, 40, 40]), "kps": np.array([[14, 15], [32, 15], [23, 24], [16, 33], [30, 33]])}
    monkeypatch.setattr(api_module, "_faces_in_target_selection_order", lambda frame: [face])


BASE = {"frame": 3, "fake_preview": True, "swap_mode": "all", "enhancer": "restore_ultra", "light_harmonizer": True,
        "expression_restore_strength": 0.6, "mask_engine": "xseg"}


class TestPreviewSoloApi:
    def _post(self, client, **extra):
        r = client.post("/api/preview", json={**BASE, **extra})
        assert r.status_code == 200, r.text
        return r.json()

    def test_no_solo_runs_the_whole_chain_and_reports_no_solo_fields(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        body = self._post(client)
        assert rec[-1]['enhancer'] == 'restore_ultra' and rec[-1]['relight'] is True and rec[-1]['expr'] == pytest.approx(0.6)
        assert 'solo_applied' not in body and body['bypassed'] == []

    def test_swapper_solo_strips_everything_after_the_swap(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        body = self._post(client, solo_stage='hyperswap')
        assert rec[-1]['enhancer'] == 'None' and rec[-1]['relight'] is False and rec[-1]['expr'] == 0.0
        assert rec[-1]['types'] == ['swap'] and body['solo_applied'] is True and body['solo_id'] == 'hyperswap'

    def test_restore_solo_keeps_the_restorer_but_drops_relight(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        self._post(client, solo_stage='restore_ultra')
        assert rec[-1]['enhancer'] == 'restore_ultra' and rec[-1]['relight'] is False

    def test_gaze_solo_drops_restorer_and_relight(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        self._post(client, solo_stage='gaze_mouth')
        assert rec[-1]['enhancer'] == 'None' and rec[-1]['relight'] is False and rec[-1]['expr'] == pytest.approx(0.6)

    def test_restore_difference_is_two_renders_and_lights_the_restorer_box_only(self, client, api_module, monkeypatch):
        import cv2
        rec = []
        _arm(monkeypatch, api_module, rec)
        n = len(rec)
        body = self._post(client, solo_stage='restore_diff')
        assert len(rec) - n == 2                                        # before (restorer off) and after (on)
        assert rec[-2]['enhancer'] == 'None' and rec[-1]['enhancer'] == 'restore_ultra'
        assert rec[-2]['relight'] is False and rec[-1]['relight'] is False        # the rest of the chain held equal
        assert body['solo_applied'] is True and body['solo_stats']['changed_pct'] > 0
        raw = __import__('base64').b64decode(body['image'].split(',', 1)[1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        assert img[12:18, 12:18].mean() > img[40:, 40:].mean() + 30      # bright where the fake restorer inverted pixels
        assert api_module.roop_globals.selected_enhancer == 'restore_ultra'  # and the globals were put back

    def test_restore_difference_with_no_restorer_says_so_and_renders_nothing(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        n = len(rec)
        body = self._post(client, solo_stage='restore_diff', enhancer='None')
        assert len(rec) == n and body['solo_applied'] is False and 'No restorer' in body['solo_note']

    def test_relight_difference_lights_only_when_relight_is_on(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        on = self._post(client, solo_stage='relight_diff')
        assert on['solo_applied'] is True and rec[-2]['relight'] is False and rec[-1]['relight'] is True
        off = self._post(client, solo_stage='relight_diff', light_harmonizer=False)
        assert off['solo_applied'] is False and 'Relight is off' in off['solo_note']

    def test_source_solo_is_the_untouched_frame_and_runs_no_swap(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        body = self._post(client, solo_stage='source')
        assert rec == [] and body['solo_applied'] is True and body['solo_id'] == 'source'

    def test_the_two_unbuilt_passes_are_refused_not_faked(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        for sid in ('occlusion_alpha', 'depth_z_order'):
            body = self._post(client, solo_stage=sid)
            assert body['solo_applied'] is False and len(body['solo_note']) > 20, sid
        assert rec == []

    def test_an_unknown_solo_id_is_refused(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        body = self._post(client, solo_stage='warp_drive')
        assert body['solo_applied'] is False and 'Unknown' in body['solo_note'] and rec == []

    def test_xseg_solo_without_a_mask_engine_is_refused_not_silently_the_swap(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        monkeypatch.setattr("roop.core.get_processing_plugins", lambda *a, **k: [types.SimpleNamespace(type='swap')])
        body = self._post(client, solo_stage='xseg_roto')
        assert body['solo_applied'] is False and 'mask engine' in body['solo_note'].lower() and rec == []

    def test_solo_with_live_swap_off_is_refused(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        body = self._post(client, solo_stage='restore_ultra', fake_preview=False)
        assert body['solo_applied'] is False and 'Live Swap' in body['solo_note'] and rec == []

    # ── bypass ───────────────────────────────────────────────────────────────────────────────
    def test_bypassing_the_restorer_reaches_the_swap_as_no_enhancer(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        body = self._post(client, bypass_stages=['restore_ultra'])
        assert rec[-1]['enhancer'] == 'None' and rec[-1]['relight'] is True and body['bypassed'] == ['restore_ultra']
        self._post(client)
        assert rec[-1]['enhancer'] == 'restore_ultra'                  # nothing leaked into the next, un-bypassed preview

    def test_bypassing_relight_and_gaze_reach_the_swap(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        self._post(client, bypass_stages={'relight_lut': True, 'gaze_mouth': True})
        assert rec[-1]['relight'] is False and rec[-1]['expr'] == 0.0 and rec[-1]['enhancer'] == 'restore_ultra'

    def test_bypassing_the_matte_removes_the_mask_processors(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        self._post(client, bypass_stages=['xseg_roto'])
        assert 'mask' not in rec[-1]['types'] and 'swap' in rec[-1]['types']

    def test_bypassing_the_swapper_returns_the_original_and_refuses_a_downstream_solo(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        plain = self._post(client, bypass_stages=['hyperswap'])
        assert rec == [] and plain['bypassed'] == ['hyperswap']
        refused = self._post(client, bypass_stages=['hyperswap'], solo_stage='restore_ultra')
        assert refused['solo_applied'] is False and 'bypassed' in refused['solo_note'] and rec == []

    def test_a_solo_of_a_stage_that_is_also_bypassed_is_refused(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        for sid, off in (('xseg_roto', 'xseg_roto'), ('restore_ultra', 'restore_ultra'), ('restore_diff', 'restore_ultra'),
                         ('relight_diff', 'relight_lut'), ('gaze_mouth', 'gaze_mouth'), ('bisenet_parsing', 'xseg_roto')):
            body = self._post(client, solo_stage=sid, bypass_stages=[off])
            assert body['solo_applied'] is False and 'bypassed' in body['solo_note'], (sid, body.get('solo_note'))
        assert rec == []

    def test_detect_cannot_be_bypassed(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        body = self._post(client, bypass_stages=['detect_align'])
        assert body['bypassed'] == [] and rec                           # the swap still ran


class TestSweepApi:
    def _sweep(self, client, x, xv, y, yv, **extra):
        r = client.post('/api/preview/sweep', json={**BASE, 'x_param': x, 'x_values': xv, 'y_param': y, 'y_values': yv, **extra})
        return r

    def test_each_axis_reaches_the_global_that_reads_it(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        r = self._sweep(client, 'restore_ultra_detail_weight', [0.1, 0.5, 0.9], 'blend_ratio', [0.3, 0.6, 0.9])
        assert r.status_code == 200, r.text
        assert len(rec) == 9
        assert [round(x['detail'], 2) for x in rec[:3]] == [0.1, 0.5, 0.9]
        assert [round(rec[i]['blend'], 2) for i in (0, 3, 6)] == [0.3, 0.6, 0.9]

    def test_alias_and_payload_axes_are_no_longer_dead(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        r = self._sweep(client, 'restore_identity_weight', [0.1, 0.5, 0.9], 'pixel_boost', [256, 512, 768])
        assert r.status_code == 200, r.text
        assert [round(x['fidelity'], 2) for x in rec[:3]] == [0.1, 0.5, 0.9]
        assert [rec[i]['subsample'] for i in (0, 3, 6)] == [256, 512, 768]

    def test_similarity_threshold_reaches_the_matcher_threshold(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        self._sweep(client, 'max_face_distance', [0.4, 0.6, 0.8], 'blend_ratio', [0.5, 0.5, 0.5])
        assert [round(x['distance'], 2) for x in rec[:3]] == [0.4, 0.6, 0.8]

    def test_mask_feather_reaches_the_cfg_and_is_put_back_unsaved(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        cfg = api_module.roop_globals.CFG
        cfg.face_mask_blend = 22.0
        saves = []
        monkeypatch.setattr(cfg, 'save', lambda *a, **k: saves.append(1))
        r = self._sweep(client, 'face_mask_blend', [5.0, 15.0, 30.0], 'blend_ratio', [0.5, 0.5, 0.5], face_mask_blend=22.0)
        assert r.status_code == 200, r.text
        assert [x['mask_blend'] for x in rec[:3]] == [5.0, 15.0, 30.0]
        assert cfg.face_mask_blend == 22.0 and saves == []

    def test_a_dead_axis_makes_identical_tiles_and_the_response_says_so(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        r = self._sweep(client, 'stabilization_strength', [0.2, 0.5, 0.8], 'stabilization_strength', [0.2, 0.5, 0.8])
        body = r.json()
        assert r.status_code == 200 and body['identical'] is True and body['distinct_tiles'] == 1
        assert body['axes']['x']['bound'] is False and 'frames' in body['axes']['x']['reason']

    def test_a_live_sweep_is_not_flagged_identical(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)

        def varying(frame, options, input_facesets=None, *, frame_idx=None, source_video=None):
            out = frame.copy()
            out[8:30, 8:30] = int(api_module.roop_globals.blend_ratio * 200)
            return out
        monkeypatch.setattr("roop.core.live_swap", varying)
        body = self._sweep(client, 'blend_ratio', [0.2, 0.5, 0.9], 'blend_ratio', [0.2, 0.5, 0.9]).json()
        assert body['identical'] is False and body['distinct_tiles'] >= 3
        assert body['axes']['x']['bound'] is True and body['axes']['x']['target'] == 'blend_ratio'

    def test_an_unknown_axis_is_422_and_lists_what_is_sweepable(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        r = self._sweep(client, 'warp_factor', [1, 2, 3], 'blend_ratio', [0.1, 0.2, 0.3])
        assert r.status_code == 422 and 'warp_factor' in r.json()['message'] and 'blend_ratio' in r.json()['sweepable'] and rec == []

    def test_a_non_numeric_value_is_422_not_500(self, client, api_module, monkeypatch):
        rec = []
        _arm(monkeypatch, api_module, rec)
        r = self._sweep(client, 'blend_ratio', ['x', 0.5, 0.9], 'blend_ratio', [0.2, 0.5, 0.9])
        assert r.status_code == 422 and 'rejected' in r.json()['message']


# ══════════════════════════════════════════════════════════════════════════════════════════════
# the three copies of the solo ids (React model, backend, mock) cannot drift apart

FLOW_MODEL = os.path.join(os.path.dirname(APP), 'react-ui', 'src', 'components', 'faceswap', 'flowModel.js')
MOCK_SERVER = os.path.join(os.path.dirname(APP), 'react-ui', 'mock-server', 'server.ts')


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


class TestIdsAgreeAcrossTheStack:
    def test_every_id_the_flow_bar_can_send_is_one_the_backend_knows(self):
        src = _read(FLOW_MODEL)
        sent = set(re.findall(r"(?:soloId|diffId):\s*'([a-z_0-9]+)'", src))
        passes = re.search(r'BUFFER_SOLO_PASSES = Object\.freeze\(\[(.*?)\]\);', src, flags=re.S).group(1)
        sent |= set(re.findall(r"\bid:\s*'([a-z_0-9]+)'", passes))
        assert {'source', 'detect_align', 'hyperswap', 'gaze_mouth', 'xseg_roto', 'restore_ultra', 'relight_lut',
                'restore_diff', 'relight_diff', 'bisenet_parsing', 'xseg_3_mask', 'occlusion_alpha', 'depth_z_order'} <= sent
        unknown = sorted(i for i in sent if solo.canonical_solo(i) is None)
        assert unknown == [], f'the flow bar sends solo ids the backend does not accept: {unknown}'

    def test_a_node_id_is_its_own_canonical_id(self):
        """The signal path and the backend must call a node the same thing."""
        src = _read(FLOW_MODEL)
        for node_solo in re.findall(r"soloId:\s*'([a-z_0-9]+)'", src):
            assert solo.canonical_solo(node_solo) == node_solo, node_solo

    def test_the_bypassable_nodes_are_exactly_the_backends(self):
        src = _read(FLOW_MODEL)
        blocks = re.findall(r"\{ id: '([a-z_0-9]+)', n: \d+,.*?bypassable: (true|false)", src, flags=re.S)
        bypassable = tuple(i for i, flag in blocks if flag == 'true')
        assert bypassable == solo.BYPASSABLE, (bypassable, solo.BYPASSABLE)

    def test_the_unavailable_passes_match(self):
        src = _read(FLOW_MODEL)
        passes = re.search(r'BUFFER_SOLO_PASSES = Object\.freeze\(\[(.*?)\]\);', src, flags=re.S).group(1)
        off = set(re.findall(r"\bid:\s*'([a-z_0-9]+)'[^}]*?available:\s*false", passes, flags=re.S))
        assert off == set(solo.UNAVAILABLE), (off, set(solo.UNAVAILABLE))

    def test_the_mock_servers_alias_table_is_the_backends(self):
        mock = _read(MOCK_SERVER)
        table = re.search(r'const SOLO_ALIASES: Record<string, string> = \{(.*?)\n\};', mock, flags=re.S).group(1)
        pairs = dict(re.findall(r"'?([a-z_0-9]+)'?:\s*'([a-z_0-9]+)'", table))
        assert pairs == solo._ALIASES
        assert tuple(re.search(r'const BYPASSABLE = \[(.*?)\];', mock).group(1).replace("'", '').replace(' ', '').split(',')) == solo.BYPASSABLE
        unavailable = set(re.findall(r"^\s{2}([a-z_]+):\s*'", re.search(r'const SOLO_UNAVAILABLE.*?\{(.*?)\n\};', mock, flags=re.S).group(1), flags=re.M))
        assert unavailable == set(solo.UNAVAILABLE)

    def test_the_mocks_sweepable_list_and_unbound_axes_are_the_backends(self):
        mock = _read(MOCK_SERVER)
        listed = re.search(r"const sweepable = \[(.*?)\];", mock, flags=re.S).group(1)
        assert set(re.findall(r"'([a-z_0-9]+)'", listed)) == set(bindings.BINDINGS)
        mock_unbound = set(re.findall(r"^\s{2}([a-z_]+):\s*'", re.search(r'const SWEEP_UNBOUND.*?\{(.*?)\n\};', mock, flags=re.S).group(1), flags=re.M))
        assert mock_unbound == {n for n, b in bindings.BINDINGS.items() if b.kind == 'none'}


# ══════════════════════════════════════════════════════════════════════════════════════════════
# the flow bar's "native bypass" pill is the backend's ramp, not a guess

class TestNativeBypassPillMatchesTheBackend:
    def test_the_js_ramp_and_conditions_are_the_python_ones(self):
        from roop import smart_restorer_bypass as sb
        src = _read(FLOW_MODEL)
        lo = int(re.search(r'BYPASS_RAMP_LO_PX = (\d+)', src).group(1))
        hi = int(re.search(r'BYPASS_RAMP_HI_PX = (\d+)', src).group(1))
        assert (lo, hi) == (sb.RAMP_LO_PX, sb.RAMP_HI_PX), 'flowModel.js ramp drifted from smart_restorer_bypass.py'
        # the weight at the points the pill text is built from
        for h, w in ((200, 1.0), (256, 1.0), (288, 0.75), (320, 0.5), (352, 0.25), (384, 0.0), (500, 0.0)):
            assert sb.ramp_weight(h) == pytest.approx(w), h
        # the swap-model and restorer sets: the JS regexes accept exactly what the Python predicates accept
        swap_rx = re.compile(re.search(r'isBypassSwapModel = \(m\) => /([^/]+)/i', src).group(1), re.I)
        rest_rx = re.compile(re.search(r'isBypassRestorer = \(name\) => /([^/]+)/i', src).group(1), re.I)
        for name in ('hyperswap', 'HyperSwap', 'realswap', 'alpha_256', 'inswapper', 'simswap', 'ghost', 'hififace', ''):
            assert bool(swap_rx.search(name)) == sb.is_bypass_swap_model(name), name
        for name in ('restore_ultra', 'Restore Ultra', 'UltraMax', 'codeformer', 'gfpgan', 'restoreformer++', 'rf++', 'gpen_256', 'None', ''):
            assert bool(rest_rx.search(name)) == sb.is_bypass_restorer(name), name

    def test_the_legacy_320_px_step_is_the_ramps_midpoint(self):
        from roop import smart_restorer_bypass as sb
        assert sb.ramp_weight(320.0) == 0.5 and (sb.RAMP_LO_PX + sb.RAMP_HI_PX) / 2 == 320.0
