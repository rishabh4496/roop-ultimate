"""Phase 11 item 5: cost of ProcessMgr.update_progress in isolation (600 calls, 1 and 20 threads, chunk and bar styles).

Measured 2026-10-21 (4070 box): 5-71 us per call, 3.6-82 ms for 600 frames. The 0.3-0.5 ms per call seen inside a real
render (probe_phase11_interconnect.py) is GIL / lock wait, which a display throttle does not remove."""
import sys, time, threading, types, io, contextlib
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings('ignore')
from roop.ProcessMgr import ProcessMgr
from roop.procmgr_runtime import ChunkedProgress
import psutil
def run(nthreads, per, style):
    import os; os.environ['ROOP_PROGRESS_STYLE'] = style
    pm = types.SimpleNamespace(_pipeline_monitor=None, _tls=threading.local(), _psutil_proc=psutil.Process(), num_threads=nthreads,
                               progress_gradio=None, total_frames=nthreads*per, _active_inference_workers=None)
    out = io.StringIO()
    with contextlib.redirect_stderr(out):
        with ChunkedProgress(total=nthreads*per, desc='Processing', unit='frames') as bar:
            tot = [0]*nthreads
            def w(i):
                t = 0
                for _ in range(per):
                    a = time.perf_counter_ns(); ProcessMgr.update_progress(pm, bar); t += time.perf_counter_ns() - a
                tot[i] = t
            ths = [threading.Thread(target=w, args=(i,)) for i in range(nthreads)]
            t0 = time.perf_counter()
            [t.start() for t in ths]; [t.join() for t in ths]
            wall = time.perf_counter() - t0
    n = nthreads*per
    print(f'style={style} threads={nthreads} calls={n} mean {sum(tot)/n/1e3:.1f} us/call, wall {wall*1e3:.1f} ms, bar.n={bar.n}')
for style in ('chunk', 'bar'):
    for nt in (1, 20):
        run(nt, 600 // nt if nt > 1 else 600, style)
