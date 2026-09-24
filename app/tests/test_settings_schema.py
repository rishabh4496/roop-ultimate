"""app/settings.py is the single source for settings; the generated files
must match it, and the shared env mapping must behave exactly as the old
hand-written block in run.py did.

Three generated/derived artefacts hang off settings.py:
  * app/settings.schema.json and react-ui/src/components/settingsCatalog.js are
    rendered by tools/gen_settings.py -- stale is a failure here, not a drift;
  * run.py and the comparison benches export ROOP_* through settings.apply_env.

The oracle below is the pre-registry mapping from run.py, kept verbatim, so
the refactor is proven equivalent on every value shape rather than assumed.
"""

import os
import random
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
for p in (APP, os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import settings  # noqa: E402
import gen_settings  # noqa: E402


def legacy_apply(cfg, environ):
    """The pre-registry mapping from run.py (as of commit 52f43b7), verbatim
    except os.environ -> environ. The oracle: apply_env must produce the
    same environment for any config."""
    def _set(var, val):
        # A caller (including a controlled benchmark) owns an explicit
        # process environment value.  Config is only the fallback; otherwise
        # an A/B arm can be silently replaced before modules import it.
        if var in environ:
            return
        if val is None:
            return
        s = str(val).strip()
        if s and s.lower() != 'auto':
            environ[var] = s

    _set('ROOP_TRT_POOL', cfg.get('perf_trt_pool'))
    _set('ROOP_TRT_BUILDER_OPT_LEVEL', cfg.get('trt_builder_optimization_level'))
    _set('ROOP_TRT_AUX_STREAMS', cfg.get('trt_auxiliary_streams'))
    if cfg.get('trt_cuda_graph') is not None and 'ROOP_TRT_CUDA_GRAPH' not in environ:
        graph = cfg.get('trt_cuda_graph')
        graph_on = graph is True or str(graph).strip().lower() in ('1', 'true', 'yes', 'on')
        environ['ROOP_TRT_CUDA_GRAPH'] = '1' if graph_on else '0'
    _set('ROOP_CV_THREADS', cfg.get('cpu_opencv_threads'))
    _set('ROOP_ORT_INTRA_THREADS', cfg.get('cpu_ort_intra_threads'))
    _set('ROOP_ORT_INTER_THREADS', cfg.get('cpu_ort_inter_threads'))
    _set('ROOP_FFMPEG_THREADS', cfg.get('cpu_ffmpeg_threads'))
    _set('ROOP_DETMASK_POOL', cfg.get('perf_detmask_pool'))
    _set('ROOP_DETECTOR_POOL', cfg.get('perf_detector_pool'))
    _set('ROOP_EXPR_POOL', cfg.get('perf_expr_pool'))
    _set('ROOP_ENCODER_PRESET', cfg.get('perf_encoder_preset'))
    _set('ROOP_STAB_CHUNK_MB', cfg.get('perf_stab_chunk_mb'))
    _set('ROOP_STAB_STREAMING', cfg.get('perf_stab_streaming'))
    # These three names are core.py's, not invented here: it reads
    # ROOP_CUDA_ARENA_STRATEGY and ROOP_CUDA_MEM_LIMIT directly when building
    # the CUDA provider options, and ROOP_CUDNN_CONV_ALGO overrides the
    # otherwise-hardcoded conv planner. Exporting under any other name would
    # produce a setting that saves, displays, and does nothing.
    _set('ROOP_CUDA_ARENA_STRATEGY', cfg.get('perf_ort_arena_strategy'))
    _set('ROOP_CUDNN_CONV_ALGO', cfg.get('perf_cudnn_conv_algo'))
    _mem_limit = cfg.get('perf_gpu_mem_limit')
    if _mem_limit is not None and str(_mem_limit).strip().lower() not in ('', 'auto'):
        # core.py wants BYTES; the benchmark and the UI both speak MiB.
        try:
            environ['ROOP_CUDA_MEM_LIMIT'] = str(
                int(float(str(_mem_limit).strip()) * 1024 * 1024))
        except (TypeError, ValueError):
            pass
    for var, key in (('ROOP_PROFILE', 'perf_profile'), ('ROOP_BATCH_SWAP', 'perf_batch_swap'),
                     ('ROOP_NVDEC', 'perf_nvdec'),
                     # Identity/tracking features that used to be reachable only
                     # by editing a launcher's environment. Same 'auto' contract:
                     # leave the env alone and let each module keep its own
                     # default, so exposing them changed no shipped behaviour.
                     ('ROOP_FACE_DEMARCATE', 'face_demarcate'),
                     ('ROOP_TRACK_STITCH', 'track_stitch'),
                     ('ROOP_VERIFY_SWAP', 'verify_swap'),
                     ('ROOP_UPRIGHT_REMEASURE', 'upright_remeasure')):
        if var in environ:
            continue
        v = str(cfg.get(key, 'auto')).strip().lower()
        if v == 'on' or (v == 'auto' and var == 'ROOP_BATCH_SWAP'):
            environ[var] = '1'
            if var == 'ROOP_BATCH_SWAP' and 'ROOP_BATCH_SWAP_XFRAME' not in environ:
                environ['ROOP_BATCH_SWAP_XFRAME'] = '1'
        elif v == 'off':
            environ[var] = '0'
            if var == 'ROOP_BATCH_SWAP' and 'ROOP_BATCH_SWAP_XFRAME' not in environ:
                environ['ROOP_BATCH_SWAP_XFRAME'] = '0'

    # Not tri-state: a model choice and a priority class.
    _rec = str(cfg.get('recognizer', 'default')).strip().lower()
    if _rec == 'adaface' and 'ROOP_ADAFACE' not in environ:
        environ['ROOP_ADAFACE'] = '1'
    elif _rec == 'default' and 'ROOP_ADAFACE' not in environ:
        environ['ROOP_ADAFACE'] = '0'
    # Only the names keep_awake._PRIORITY_CLASSES accepts; it falls back to
    # 'high' for anything else, so passing a value it does not know through
    # would present as a working setting that does nothing.
    _pri = str(cfg.get('process_priority', 'auto')).strip().lower()
    if _pri in ('high', 'above_normal', 'normal') and 'ROOP_PRIORITY' not in environ:
        environ['ROOP_PRIORITY'] = _pri


VALUES = [None, "", "  ", "auto", "AUTO", "on", "off", "On", "1", "0", "true", "false", "yes",
          True, False, 0, 1, 2, 3, 7.5, "2", "512", "4096.5", "notanumber",
          "adaface", "default", "high", "above_normal", "normal", "realtime", "x"]
PRESET = ["ROOP_TRT_POOL", "ROOP_TRT_CUDA_GRAPH", "ROOP_BATCH_SWAP", "ROOP_BATCH_SWAP_XFRAME",
          "ROOP_CUDA_MEM_LIMIT", "ROOP_ADAFACE", "ROOP_PRIORITY", "ROOP_PROFILE", "ROOP_NVDEC"]


class GeneratedFilesAreCurrent(unittest.TestCase):
    def test_schema_and_catalog_match_settings_py(self):
        stale = gen_settings.stale()
        self.assertEqual(stale, [], "run `python tools/gen_settings.py` and commit the result: "
                         + ", ".join(os.path.relpath(p, ROOT) for p in stale))

    def test_check_mode_fails_on_a_stale_file(self):
        # The guard has to actually fail -- verified, not assumed.
        real = gen_settings._read
        try:
            gen_settings._read = lambda p: "// edited by hand\n" if p == gen_settings.CATALOG_PATH else real(p)
            self.assertEqual([os.path.basename(p) for p in gen_settings.stale()], ["settingsCatalog.js"])
        finally:
            gen_settings._read = real

    def test_catalog_js_carries_exactly_ui_settings(self):
        with open(gen_settings.CATALOG_PATH, encoding="utf-8") as fh:
            src = fh.read()
        entries = re.findall(r"\{ key: '([a-z0-9_]+)', label: '((?:[^'\\]|\\.)*)', section: '([^']+)' \}", src)
        self.assertEqual([(k, l.replace("\\'", "'"), s) for k, l, s in entries], list(settings.UI_SETTINGS))
        self.assertIn("export const focusSetting", src)   # the palette helpers survive generation

    def test_schema_names_defaults_and_mappings(self):
        schema = gen_settings.build_schema()["settings"]
        for key, label, section in settings.UI_SETTINGS:
            self.assertEqual(schema[key]["ui"], {"label": label, "section": section})
        for key, var, kind in settings.ENV_SETTINGS:
            self.assertEqual(schema[key]["env"], {"var": var, "kind": kind})
        probe = settings.Settings(os.path.join(APP, "__no_such_settings_file__.yaml"))
        for key, entry in schema.items():
            if entry.get("config_only"):
                self.assertFalse(hasattr(probe, key), key)
                continue
            self.assertTrue(hasattr(probe, key), key)
            if "default" in entry:
                self.assertEqual(entry["default"], getattr(probe, key), key)
        # the machine's own facts never become a "default"
        for key in gen_settings.DERIVED:
            self.assertTrue(schema[key].get("derived"), key)


# Keys added to ENV_SETTINGS after the registry replaced run.py's block. The
# oracle above cannot know them; each instead has to behave exactly like an
# existing key of the same kind (test_new_keys_behave_like_their_kind).
ADDED_AFTER_REGISTRY = ('perf_batch_max', 'perf_nvenc_preset', 'perf_gpu_affine',
                        'perf_pinned_buffers', 'temporal_step', 'identity_confidence_threshold')


class EnvMappingMatchesTheOldRunPy(unittest.TestCase):
    """apply_env(cfg, env) == the pre-registry block, for any config values."""

    def _keys(self):
        return [k for k, _, _ in settings.ENV_SETTINGS if k not in ADDED_AFTER_REGISTRY]

    def test_new_keys_behave_like_their_kind(self):
        rows = {k: (var, kind) for k, var, kind in settings.ENV_SETTINGS}
        reference = {'value': 'perf_trt_pool', 'tristate': 'perf_nvdec'}
        for key in ADDED_AFTER_REGISTRY:
            var, kind = rows[key]
            ref_key = reference[kind]
            ref_var = rows[ref_key][0]
            for value in VALUES:
                a, b = {}, {}
                settings.apply_env({key: value}, a, keys=(key,))
                settings.apply_env({ref_key: value}, b, keys=(ref_key,))
                self.assertEqual(a.get(var), b.get(ref_var), f"{key}={value!r}")

    def test_every_single_value_shape(self):
        for key in self._keys():
            for value in VALUES:
                cfg = {key: value}
                a, b = {}, {}
                settings.apply_env(cfg, a)
                legacy_apply(cfg, b)
                self.assertEqual(a, b, f"{key}={value!r}")

    def test_environment_wins_the_same_way(self):
        for var in PRESET:
            for key in self._keys():
                for value in ("on", "off", "2", True, "adaface", "high", "1024"):
                    cfg = {key: value}
                    a, b = {var: "preset"}, {var: "preset"}
                    settings.apply_env(cfg, a)
                    legacy_apply(cfg, b)
                    self.assertEqual(a, b, f"{var} preset, {key}={value!r}")

    def test_random_full_configs(self):
        rng = random.Random(20260922)
        keys = self._keys()
        for _ in range(400):
            cfg = {k: rng.choice(VALUES) for k in rng.sample(keys, rng.randint(0, len(keys)))}
            env = {v: "preset" for v in rng.sample(PRESET, rng.randint(0, 3))}
            a, b = dict(env), dict(env)
            settings.apply_env(cfg, a)
            legacy_apply(cfg, b)
            self.assertEqual(a, b, f"cfg={cfg} env={env}")

    def test_applied_names_are_reported(self):
        env = {}
        applied = settings.apply_env({"perf_trt_pool": 2, "perf_batch_swap": "auto"}, env)
        # ROOP_ADAFACE is '0' whenever `recognizer` is absent: the old block read
        # cfg.get('recognizer', 'default') and 'default' exports '0'. Same here.
        self.assertEqual(sorted(applied), ["ROOP_ADAFACE", "ROOP_BATCH_SWAP", "ROOP_BATCH_SWAP_XFRAME", "ROOP_TRT_POOL"])
        self.assertEqual(env, {"ROOP_TRT_POOL": "2", "ROOP_BATCH_SWAP": "1", "ROOP_BATCH_SWAP_XFRAME": "1",
                               "ROOP_ADAFACE": "0"})


class RunPyAndBenchesUseTheSharedMapping(unittest.TestCase):
    """Source-level: no hand-written `_set('ROOP_...` block may come back."""

    FILES = ("run.py", os.path.join("tests", "compare_enhancers_video.py"))

    def test_no_private_env_blocks(self):
        for rel in self.FILES:
            with open(os.path.join(APP, rel), encoding="utf-8") as fh:
                src = fh.read()
            self.assertIn("apply_env(", src, rel)
            self.assertNotRegex(src, r"_set\('ROOP_", rel)
            self.assertNotIn("for var, key in (('ROOP_", src, rel)


if __name__ == "__main__":
    unittest.main()
