"""Process-pool workers that each own an independent recognition engine.

NO RENDER USES THIS. The app's pipeline is one process with worker THREADS, and the standalone
``face_engine`` package has its own spawn-based segment pool (``face_engine/media/worker_pool.py``)
that already builds its processor inside each worker and assigns GPUs round-robin. This module is
the tested recipe for anyone who puts ``RecognitionInferenceEngine`` behind a ``ProcessPoolExecutor``
(e.g. embedding thousands of crops offline). It exists because three things go wrong, each verified:

  * An engine holds an ONNX Runtime session and CANNOT BE PICKLED (``TypeError``). It must never
    travel as an argument or return value: build it INSIDE the worker, in the pool initializer, and
    send only crops in and (embedding, quality) out.
  * CUDA must not be initialised in the parent and then inherited by a ``fork``ed child. Use the
    ``spawn`` context (the only one on Windows; ``make_pool`` always uses it). A spawn caller needs
    the usual ``if __name__ == "__main__":`` guard.
  * Each worker's CPU-tier session defaults to one thread per physical core, so N workers would run
    N x cores threads. ``make_pool`` divides the cores between the workers.

GPUs are assigned round-robin by the order in which workers claim an index from a shared counter,
so ``gpu_ids`` can name any subset of the machine's devices.

Measured on an RTX 4070 (2 spawned workers, a cold TensorRT cache built by both at once): both
initialised, a second run read the shared cache, and every embedding matched a parent-process CPU
reference at >= 0.99997 cosine (TensorRT FP16) / 0.999996 (CUDA).
"""

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_engine = None            # this worker process's own engine (module global: one per process)
_gpu_id: Optional[int] = None


def pick_gpu(worker_index: int, gpu_ids: Sequence[int]) -> int:
    """Round-robin: worker i -> gpu_ids[i % len(gpu_ids)]."""
    if not gpu_ids:
        raise ValueError("gpu_ids must name at least one device")
    return int(gpu_ids[worker_index % len(gpu_ids)])


def default_gpu_ids() -> List[int]:
    """Every CUDA device torch can see, else [0] (the CPU provider ignores the id)."""
    try:
        import torch
        count = torch.cuda.device_count()
        if count:
            return list(range(count))
    except ImportError:
        pass
    return [0]


def init_worker_process(model_name: str, models_dir: str, provider: str, gpu_ids: Sequence[int],
                        counter, cpu_threads: Optional[int] = None) -> None:
    """``ProcessPoolExecutor`` initializer: build THIS process's engine, once.

    `counter` is a ``Value('i')`` from the same spawn context; each worker takes the next index
    under its lock, which is what spreads workers across `gpu_ids`. A failure here (unknown model,
    failed download, no usable provider) kills the worker and surfaces as ``BrokenProcessPool`` on
    the pool's first result -- it does not hang.
    """
    global _engine, _gpu_id
    if _engine is not None:
        return
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    _gpu_id = pick_gpu(index, gpu_ids)
    from roop.recognition_engine import RecognitionInferenceEngine      # imported here: the child stays light
    _engine = RecognitionInferenceEngine(model_name, models_dir, provider, _gpu_id, cpu_threads=cpu_threads)


def worker_engine():
    if _engine is None:
        raise RuntimeError("init_worker_process has not run in this process (pid %d)" % os.getpid())
    return _engine


def embed_crop(crop: np.ndarray) -> Tuple[np.ndarray, float]:
    """Pool task: aligned BGR crop in, (unit embedding, quality) out. Both are plain, picklable values."""
    return worker_engine().compute_embedding(crop)


def worker_info(pause: float = 0.0) -> Dict[str, Any]:
    """Pool task: what this worker actually runs (the pid, the GPU it was given, the verified providers).

    `pause` keeps the worker busy so a short batch of calls reaches every worker.
    """
    if pause:
        import time
        time.sleep(pause)
    engine = worker_engine()
    return {"pid": os.getpid(), "gpu_id": _gpu_id, "providers": list(engine.active_providers),
            "degraded": engine.degraded,
            "cpu_threads": engine.session.get_session_options().intra_op_num_threads}


def make_pool(n_workers: int, model_name: str, models_dir: str, provider: str = "auto",
              gpu_ids: Optional[Sequence[int]] = None) -> ProcessPoolExecutor:
    """A spawn-context pool whose workers each build their own engine.

    On CPU the cores are divided between the workers (at least one thread each).
    """
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    ctx = mp.get_context("spawn")
    ids = list(gpu_ids) if gpu_ids else default_gpu_ids()
    from roop.recognition_engine import _physical_cores              # the same core count the engine uses
    cores = _physical_cores()
    return ProcessPoolExecutor(
        n_workers, mp_context=ctx, initializer=init_worker_process,
        initargs=(model_name, models_dir, provider, ids, ctx.Value("i", 0), max(1, cores // n_workers)))
