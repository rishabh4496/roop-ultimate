"""The runtime layer never changes which face is selected.

A provider failure is allowed to change the RUNTIME (TensorRT -> CUDA -> CPU,
batch -> sequential, or a controlled render error).  It is never allowed to
change the SELECTION (which target person is eligible).  Three checks:

1. The runtime modules do not even name a selection symbol, so there is no
   code path by which a fallback could rewrite one.
2. When a batched swap fails and the swapper falls back to B=1, every crop is
   re-run with the SAME (source, target) pair it was submitted with, in order.
   A fallback that re-paired sources and targets would be a selection change
   dressed as a runtime one.
3. The ``[Runtime]`` banner reads provider / precision / batch path off live
   objects and re-checks the initialize-time selection snapshot, so a moved
   selection is reported as VIOLATED rather than passing as a runtime event.
"""
import os
import re
import sys
import threading
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from roop import runtime_banner  # noqa: E402
from roop.processors.FaceSwapInsightFace import FaceSwapInsightFace  # noqa: E402
from roop.target_selection import (normalize_target_selection,  # noqa: E402
                                   selection_group_ids)

APP = Path(__file__).resolve().parents[1]

# Every module that decides HOW a model runs.  `swap_model` is deliberately not
# in the forbidden list: it names the network, not the person.
RUNTIME_MODULES = [
    "roop/predictor.py", "roop/backend_manager.py", "roop/swap_batcher.py",
    "roop/session_pool.py", "roop/render_guard.py", "roop/precision_policy.py",
    "roop/trt_session_builder.py", "roop/gpu_preflight.py", "roop/trt_engine.py",
    "roop/trt_probe.py", "roop/ort_support.py", "roop/cudnn_algo.py",
    "roop/processors/FaceSwapInsightFace.py",
]
SELECTION_SYMBOLS = re.compile(
    r"\b(selection_state|swap_mode|selected_index|TARGET_FACE_GROUP|TARGET_FACES|"
    r"target_face_groups|selected_target_groups|person_id|person_ids|"
    r"distance_threshold|max_face_distance|face_distance_threshold|"
    r"normalize_target_selection|selection_group_ids)\b")


class RuntimeModulesDoNotNameSelection(unittest.TestCase):
    def test_no_runtime_module_references_a_selection_symbol(self):
        for rel in RUNTIME_MODULES:
            source = (APP / rel).read_text(encoding="utf-8")
            hits = sorted({m.group(1) for m in SELECTION_SYMBOLS.finditer(source)})
            self.assertEqual(hits, [], f"{rel} references selection symbols: {hits}")


class _PairRecordingSwapper(FaceSwapInsightFace):
    """Real RunBatch/RunBatchMulti/_sequential_fallback over a stub net whose
    batched inference always fails.  `Run` records the (source, target) pair it
    was handed so the fallback's pairing can be checked exactly."""

    def __init__(self):
        self.pool = None
        self._mask_tls = threading.local()
        self.image_input_name = 'target'
        self.embed_input_name = 'source'
        self.loaded_model_key = 'stub'
        self._batch_unsupported = False
        self.pairs = []

    def _compute_source_input(self, source_face):
        return np.zeros((1, 512), dtype=np.float32)

    def _infer(self, feed):
        if feed[self.image_input_name].shape[0] > 1:
            raise RuntimeError('static batch: expected 1')
        return [np.zeros((1, 3, 8, 8), dtype=np.float32)]

    def Run(self, source_face, target_face, temp_frame):
        self.pairs.append((source_face, target_face))
        self._mask_tls.masks = None
        return temp_frame[0]


class BatchFallbackKeepsEveryPair(unittest.TestCase):
    def test_run_batch_multi_fallback_replays_the_same_pairs_in_order(self):
        swapper = _PairRecordingSwapper()
        sources = [object() for _ in range(3)]
        targets = [object() for _ in range(3)]
        crops = [np.full((1, 3, 8, 8), i, dtype=np.float32) for i in range(3)]
        requests = list(zip(sources, targets, crops))

        out = swapper.RunBatchMulti(requests)

        self.assertEqual(len(out), 3)
        self.assertTrue(swapper._batch_unsupported, "the failure must be recorded")
        self.assertEqual([id(s) for s, _ in swapper.pairs], [id(s) for s in sources])
        self.assertEqual([id(t) for _, t in swapper.pairs], [id(t) for t in targets])

    def test_run_batch_fallback_keeps_one_source_and_target_for_every_tile(self):
        swapper = _PairRecordingSwapper()
        src, tgt = object(), object()
        crops = [np.full((1, 3, 8, 8), i, dtype=np.float32) for i in range(4)]
        swapper.RunBatch(src, tgt, crops)
        self.assertEqual(len(swapper.pairs), 4)
        self.assertTrue(all(s is src and t is tgt for s, t in swapper.pairs))


class _Session:
    def __init__(self, providers):
        self._providers = providers

    def get_providers(self):
        return list(self._providers)


class _Swap:
    type = 'swap'

    def __init__(self, providers, requested, trt_disabled=False, batch_unsupported=False):
        self.model_swap_insightface = _Session(providers)
        self._swap_providers = requested
        self._trt_disabled = trt_disabled
        self._batch_unsupported = batch_unsupported
        self.loaded_model_key = 'hyperswap'

    def RunBatch(self, *_a):
        return []


class _Options:
    swap_mode = 'selected'
    swap_model = 'hyperswap'


class _Mgr:
    def __init__(self, swap):
        self.processors = [swap]
        self.options = _Options()
        self.target_face_groups = [0, 1, 0]
        self.target_selection = normalize_target_selection(
            {"selection_mode": "selected", "person_id": 1}, person_count=2)
        self.selected_target_groups = selection_group_ids(
            self.target_face_groups, self.target_selection)
        self._swap_batcher = None


_TRT = [("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
        "CUDAExecutionProvider", "CPUExecutionProvider"]


def _fields(line):
    return dict(kv.split("=", 1) for kv in line.split(" ")[1:])


class BannerReadsLiveState(unittest.TestCase):
    def test_banner_carries_every_required_key(self):
        mgr = _Mgr(_Swap(["TensorrtExecutionProvider", "CUDAExecutionProvider"], _TRT))
        line = runtime_banner.runtime_selection_line(mgr, 'init', batch_mode='sequential')
        f = _fields(line)
        for key in ("provider_active", "swap_model", "batch_mode",
                    "target_selection", "selected_person"):
            self.assertIn(key, f, line)
        self.assertEqual(f["provider_active"], "tensorrt")
        self.assertEqual(f["precision"], "mixed")
        self.assertEqual(f["swap_model"], "hyperswap")
        self.assertEqual(f["target_selection"], "selected/selected")
        self.assertEqual(f["selected_person"], "1")
        self.assertEqual(f["selected_groups"], "[1]")
        self.assertNotIn("selection_invariant", f)

    def test_a_silent_cuda_fallback_is_named_as_cuda_not_as_the_request(self):
        mgr = _Mgr(_Swap(["CUDAExecutionProvider", "CPUExecutionProvider"], _TRT))
        f = _fields(runtime_banner.runtime_selection_line(mgr, 'init'))
        self.assertEqual(f["provider_active"], "cuda")
        self.assertEqual(f["requested"], "tensorrt>cuda>cpu")

    def test_a_runtime_trt_rebuild_is_visible(self):
        mgr = _Mgr(_Swap(["CUDAExecutionProvider"], _TRT, trt_disabled=True))
        f = _fields(runtime_banner.runtime_selection_line(mgr, 'video'))
        self.assertEqual(f["provider_active"], "cuda(trt-rebuilt)")

    def test_a_model_that_declined_batching_says_so(self):
        mgr = _Mgr(_Swap(["CPUExecutionProvider"], ["CPUExecutionProvider"],
                         batch_unsupported=True))
        f = _fields(runtime_banner.runtime_selection_line(mgr, 'video'))
        self.assertEqual(f["batch_mode"], "sequential(model-declined-batch)")
        self.assertEqual(f["precision"], "fp32")

    def test_selection_unchanged_across_a_provider_fallback_reads_ok(self):
        swap = _Swap(["TensorrtExecutionProvider"], _TRT)
        mgr = _Mgr(swap)
        mgr._selection_snapshot = runtime_banner.snapshot_selection(mgr)
        # The runtime moves: TensorRT rebuilt on CUDA, batching declined.
        swap.model_swap_insightface = _Session(["CUDAExecutionProvider"])
        swap._trt_disabled = True
        swap._batch_unsupported = True
        f = _fields(runtime_banner.runtime_selection_line(mgr, 'video'))
        self.assertEqual(f["provider_active"], "cuda(trt-rebuilt)")
        self.assertEqual(f["selection_invariant"], "OK")
        self.assertEqual(f["selected_person"], "1")

    def test_a_moved_selection_is_reported_as_violated(self):
        mgr = _Mgr(_Swap(["CPUExecutionProvider"], ["CPUExecutionProvider"]))
        mgr._selection_snapshot = runtime_banner.snapshot_selection(mgr)
        mgr.target_selection = normalize_target_selection(
            {"selection_mode": "selected", "person_id": 0}, person_count=2)
        mgr.selected_target_groups = selection_group_ids(
            mgr.target_face_groups, mgr.target_selection)
        line = runtime_banner.runtime_selection_line(mgr, 'video')
        self.assertIn("selection_invariant=VIOLATED(", line)
        self.assertIn("person_id", line)


class ProcessMgrEmitsTheBanner(unittest.TestCase):
    def test_initialize_snapshots_and_prints_and_video_phase_rechecks(self):
        pm = (APP / "roop" / "ProcessMgr.py").read_text(encoding="utf-8")
        self.assertIn("_runtime_banner.snapshot_selection(self)", pm)
        self.assertIn("runtime_selection_line(self, 'init')", pm)
        batch = (APP / "roop" / "procmgr_batch.py").read_text(encoding="utf-8")
        self.assertIn("runtime_selection_line(self, 'video')", batch)


if __name__ == "__main__":
    unittest.main()
