"""Why is this machine slow? One command, one verdict.

Written for the report "I set Max Threads to 7 and it runs ONE thread at 1-2
fps". That symptom has a specific cause and the app used to reach it in
silence: torch cannot see the CUDA device, `ui/main.run()` quietly rewrites the
provider to dml/rocm/cpu, and `core.batch_process` then forces
execution_threads to 1 for dml/rocm -- discarding Max Threads without a word.

Everything below is read the way the app reads it, IN ORDER, so the first line
that disagrees with what you configured is the answer.

THE LAST SECTION IS THE ONE THAT MATTERS. Provider lists are what onnxruntime
was ASKED for, not what it ran: ORT lists TensorrtExecutionProvider as
"available" with the runtime DLLs missing and falls back to CPU per node,
silently. Only a timed inference through the app's own init can tell the
difference -- which is why this runs one.

    env/Scripts/python.exe tests/diag_device.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(APP)

VERDICTS = []


def head(t):
    print()
    print(t)
    print("-" * len(t))


def main():
    print("=" * 72)
    print("  ROOP ULTIMATE - DEVICE DIAGNOSTIC")
    print("=" * 72)

    # 1. the card, as torch sees it
    head("1. torch / CUDA")
    cuda_ok = False
    try:
        import torch
        print("  torch            %s" % torch.__version__)
        print("  built for CUDA   %s" % torch.version.cuda)
        cuda_ok = torch.cuda.is_available()
        print("  cuda.is_available() -> %s" % cuda_ok)
        if cuda_ok:
            pr = torch.cuda.get_device_properties(0)
            print("  device           %s" % pr.name)
            print("  VRAM             %.2f GB" % (pr.total_memory / 1024 ** 3))
        else:
            VERDICTS.append(
                "torch CANNOT SEE THE GPU. This is the whole problem: the app "
                "rewrites the provider to dml/rocm/cpu, and dml/rocm are then "
                "forced to ONE worker thread. Reinstall the CUDA build of torch "
                "(a CPU-only wheel reports exactly this) and check the NVIDIA "
                "driver is new enough for it.")
    except Exception as e:
        print("  torch import FAILED: %s" % e)
        VERDICTS.append("torch will not import (%s). Nothing else can work." % e)

    # 2. what onnxruntime offers
    head("2. onnxruntime")
    avail = []
    try:
        import onnxruntime as ort
        print("  onnxruntime      %s" % ort.__version__)
        avail = ort.get_available_providers()
        print("  available        %s" % avail)
        print("  (what ORT was BUILT with, NOT what it can actually run)")
    except Exception as e:
        print("  onnxruntime import FAILED: %s" % e)

    try:
        import tensorrt
        print("  tensorrt         %s" % tensorrt.__version__)
    except Exception:
        print("  tensorrt         NOT IMPORTABLE -> 'tensorrt' silently becomes 'cuda'")

    # 3. the provider the app will actually choose
    head("3. provider resolution (the order ui/main.run uses)")
    from settings import Settings
    cfg = Settings("config.yaml")
    asked = cfg.provider
    resolved = asked
    print("  config.yaml asks for            '%s'" % asked)
    if asked in ("cuda", "tensorrt") and not cuda_ok:
        if 'DmlExecutionProvider' in avail:
            resolved = "dml"
        elif 'ROCMExecutionProvider' in avail:
            resolved = "rocm"
        else:
            resolved = "cpu"
        print("  no CUDA device -> rewritten to  '%s'" % resolved)
    if resolved == "tensorrt":
        try:
            import tensorrt  # noqa: F811
        except Exception:
            resolved = "cuda"
            print("  tensorrt not importable -> 'cuda'")
    print("  EFFECTIVE PROVIDER              '%s'" % resolved)
    if resolved != asked:
        VERDICTS.append("The provider you configured ('%s') is NOT what will "
                        "run ('%s')." % (asked, resolved))

    # 4. threads: what you set vs what runs
    head("4. threads")
    print("  config max_threads              %s" % cfg.max_threads)
    print("  auto_thread_selection           %s" % cfg.auto_thread_selection)
    if cfg.auto_thread_selection:
        for mode in ("standard", "enhanced", "heavy"):
            print("    auto would pick (%-8s)    %s" % (mode, cfg.resolve_threads(mode)))
        VERDICTS.append(
            "auto_thread_selection is ON, so your Max Threads is IGNORED and "
            "the numbers above are what run. Turn it OFF to use your own value. "
            "NOTE: it resets to ON by itself whenever the config lands on a "
            "different GPU than it was saved on.")
    else:
        print("  -> execution_threads            %s" % cfg.max_threads)
    if resolved in ("dml", "rocm"):
        print("  BUT: batch_process forces this provider to 1 worker thread.")
        VERDICTS.append("'%s' is FORCED TO ONE THREAD by "
                        "core.suggest_execution_threads, whatever Max Threads "
                        "says. That is your 'processing on 1 thread only'."
                        % resolved)

    # 5. pools
    head("5. session pools")
    from roop import session_pool
    gb = session_pool._detect_vram_gb()
    auto_trt, auto_detmask = session_pool._auto_pool_defaults()
    print("  detected VRAM                   %.1f GB" % gb)
    print("  auto pools (trt / detmask)      %s / %s" % (auto_trt, auto_detmask))
    for key in ('perf_trt_pool', 'perf_detmask_pool',
                'perf_detector_pool', 'perf_expr_pool'):
        v = getattr(cfg, key, 'auto')
        pinned = str(v).lower() != 'auto'
        print("  %-24s %8s%s" % (key, v, '   <- PINNED, not auto' if pinned else ''))
        if pinned:
            VERDICTS.append(
                "%s is pinned to '%s'. Set it back to 'auto' unless a "
                "counterbalanced A/B on THIS card says otherwise -- a pinned "
                "pool size is the usual cause of a render collapsing to 1-2 "
                "fps (TensorRT context thrashing, which looks like a hang)."
                % (key, v))

    # 6. THE ONE THAT MATTERS: is the GPU actually doing the work?
    head("6. real inference through the app's own init")
    print("  (loading models -- the first run builds TensorRT engines, be patient)")
    try:
        import numpy as np
        import cv2
        from angle_bench import init_pipeline
        init_pipeline(cfg.provider, cfg.swap_model, 'GPEN 256 Pro', 'RealityUX')
        import roop.globals as g
        print("  execution_providers  %s" % (g.execution_providers,))

        from roop.processors.Enhance_GPEN256Pro import Enhance_GPEN256Pro
        p = Enhance_GPEN256Pro()
        p.Initialize({'devicename': 'cuda' if cuda_ok else 'cpu',
                      'pool_size': None})
        rng = np.random.default_rng(0)
        crop = cv2.GaussianBlur(
            rng.integers(60, 200, (256, 256, 3), dtype=np.uint8), (0, 0), 1.4)
        for _ in range(3):
            p.Run(None, None, crop.copy())          # warm / build engines
        t = time.perf_counter()
        for _ in range(20):
            p.Run(None, None, crop.copy())
        ms = (time.perf_counter() - t) / 20 * 1000
        print("  GPEN 256 Pro         %.1f ms/face" % ms)
        print("  reference            ~25 ms on an RTX 4070 (TensorRT)")
        if ms > 120:
            VERDICTS.append(
                "The restorer takes %.0f ms/face against ~25 ms on a working "
                "GPU. Whatever the provider list says, THIS IS RUNNING ON THE "
                "CPU." % ms)
        else:
            print("  -> the GPU is executing this. Slowness is elsewhere.")
    except Exception as e:
        print("  inference FAILED: %s: %s" % (type(e).__name__, e))
        VERDICTS.append("A real inference could not run at all: %s" % e)

    print()
    print("=" * 72)
    if VERDICTS:
        print("  FINDINGS")
        for i, v in enumerate(VERDICTS, 1):
            print("  %d. %s" % (i, v))
    else:
        print("  Nothing wrong found: real GPU provider, threads as configured,")
        print("  pools on auto. If it is still slow, measure with")
        print("  tests/ab_small_card_pools.py rather than changing settings.")
    print("=" * 72)


if __name__ == '__main__':
    main()
