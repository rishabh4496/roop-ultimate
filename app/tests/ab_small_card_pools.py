"""What the small-card pool guard costs, with GPU utilisation sampled.

`session_pool._auto_pool_defaults` returns 0/0 below 7GB, so on an RTX 3060 6GB
BOTH pools are off -- and `_gpu_guard(pooled=False)` then hands detect, mask,
swap and enhance the SAME global lock. Every worker thread queues on it, so the
card idles while threads wait on each other rather than on the GPU.

This renders the same clip under different pool policies IN SEPARATE PROCESSES
(the pool size is cached per process and read from the env at import), sampling
`nvidia-smi` throughout, and reports fps beside mean GPU utilisation.

COUNTERBALANCED. The first arm of a process pays the TensorRT engine build for
any geometry it is first to use, which has read as a false +21.8% here before.
Arms run A,B then B,A and the two orders are averaged.

    env/Scripts/python.exe tests/ab_small_card_pools.py --vram 6 --threads 8
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def sample_gpu(stop, out, pid_box):
    """Mean GPU utilisation, whole-card VRAM, and THIS PROCESS's own VRAM.

    The per-process figure is the one that decides whether a config fits a 6GB
    card: whole-card `memory.used` on a desktop machine also carries the
    compositor and the browser, so it overstates the requirement by a GB or more
    and would condemn a config that actually fits."""
    while not stop.is_set():
        try:
            r = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=utilization.gpu,memory.used,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=4)
            u, m, w = [x.strip() for x in r.stdout.strip().splitlines()[0].split(',')]
            own = 0.0
            pid = pid_box.get('pid')
            if pid:
                a = subprocess.run(
                    ['nvidia-smi', '--query-compute-apps=pid,used_memory',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=4)
                for line in a.stdout.strip().splitlines():
                    parts = [x.strip() for x in line.split(',')]
                    if len(parts) == 2 and parts[0] == str(pid):
                        own = float(parts[1])
            out.append((float(u), float(m), float(w), own))
        except Exception:
            pass
        stop.wait(0.5)


CHILD = r'''
import os, sys, time
APP = %(app)r
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(APP, "tests"))
os.chdir(APP)
import cv2
import angle_bench as ab
from settings import Settings
cfg = Settings("config.yaml")
_MASK = {"RealityUX": "mask_realityux", "DFL XSeg": "mask_xseg",
         "Face Parser (BiSeNet)": "mask_faceparser", "Clip2Seg": "mask_clip2seg",
         "Face Occluder": "mask_occluder", "Face Occluder v3 (XSeg-3)": "mask_xseg3"}
mask = _MASK.get(cfg.mask_engine, cfg.mask_engine)
g = ab.init_pipeline(cfg.provider, cfg.swap_model, cfg.selected_enhancer, mask,
                     float(cfg.swap_model_mask_strength))
g.codeformer_fidelity = float(cfg.codeformer_fidelity)
g.execution_threads = %(threads)d
from roop.core import live_swap
from roop import session_pool
src = ab.load_faceset(os.path.join(APP, "facesets", "harjot.fsz"))
g.INPUT_FACESETS = [src]
g.TARGET_FACES = []
opts = ab.build_options(g, cfg.swap_model, mask)

cap = cv2.VideoCapture(%(clip)r)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()

from roop.ProcessMgr import ProcessMgr
pm = ProcessMgr(None)
pm.initialize([src], [], opts)

import concurrent.futures as cf
# Warm every engine before the clock starts, so the measurement is throughput
# and not the TensorRT build the first arm would otherwise pay for.
pm.process_frame(frames[0].copy())
t0 = time.perf_counter()
with cf.ThreadPoolExecutor(max_workers=%(threads)d) as ex:
    list(ex.map(lambda f: pm.process_frame(f.copy()), [f for f in frames]))
el = time.perf_counter() - t0
print("RESULT " + repr({"frames": len(frames), "sec": el,
                        "fps": len(frames) / el,
                        "trt": session_pool.pool_size(),
                        "detmask": session_pool.detmask_pool_size()}))
'''


def run_arm(name, env_extra, clip, threads):
    env = dict(os.environ)
    env.update(env_extra)
    code = CHILD % {'app': APP, 'threads': threads, 'clip': clip}
    # BASELINE FIRST. nvidia-smi's per-process query returns nothing under
    # Windows' WDDM driver model, so the process's own footprint is taken as the
    # RISE in card usage over an idle reading taken just before it starts. That
    # is the number that decides whether a config fits a 6GB card; whole-card
    # `memory.used` on a desktop also carries the compositor and the browser.
    def _card_mb():
        try:
            r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                                '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=4)
            return float(r.stdout.strip().splitlines()[0])
        except Exception:
            return 0.0
    baseline = min(_card_mb() for _ in range(3))
    samples, stop, pid_box = [], threading.Event(), {}
    t = threading.Thread(target=sample_gpu, args=(stop, samples, pid_box), daemon=True)
    t.start()
    proc = subprocess.Popen([sys.executable, '-c', code], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    pid_box['pid'] = proc.pid
    out_s, err_s = proc.communicate()

    class _R:
        stdout, stderr = out_s, err_s
    r = _R()
    stop.set()
    t.join(timeout=3)
    line = [l for l in r.stdout.splitlines() if l.startswith('RESULT ')]
    if not line:
        print(f"  {name}: FAILED\n{r.stdout[-2500:]}\n{r.stderr[-2500:]}")
        return None
    d = eval(line[0][len('RESULT '):])
    d['baseline'] = baseline
    busy = [s for s in samples if s[1] > baseline + 400]
    d['util'] = sum(s[0] for s in busy) / len(busy) if busy else 0.0
    d['vram'] = max((s[1] for s in samples), default=0)
    d['own'] = max((s[1] for s in samples), default=0) - baseline
    d['watt'] = sum(s[2] for s in busy) / len(busy) if busy else 0.0
    print(f"  {name:34s} {d['fps']:6.2f} fps  util {d['util']:5.1f}%  "
          f"own VRAM {d['own']:6.0f} MB (card {d['vram']:.0f}, idle {baseline:.0f})  "
          f"{d['watt']:5.1f} W  "
          f"(pools trt {d['trt']} / detmask {d['detmask']})")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', default=os.path.join(
        r'G:/pinokio/cache/TEMP/claude/G--pinokio-api-roop-ultimate'
        r'/93920599-2f75-4ddd-a101-ca937f9978fc/scratchpad', 's1_400.mp4'))
    ap.add_argument('--vram', default='6',
                    help='ROOP_VRAM_GB for both arms (simulates the card)')
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()

    base = {'ROOP_VRAM_GB': str(args.vram)}
    arms = [
        ('A the 6GB policy (pools OFF)', dict(base)),
        ('B pools 2/2', dict(base, ROOP_TRT_POOL='2', ROOP_DETMASK_POOL='2')),
    ]
    print(f"clip={os.path.basename(args.clip)} threads={args.threads} "
          f"simulated VRAM={args.vram}GB")
    acc = {}
    for order in (arms, list(reversed(arms))):
        print(f"\n  -- pass {'forward' if order is arms else 'reversed'} --")
        for name, env in order:
            d = run_arm(name, env, args.clip, args.threads)
            if d:
                acc.setdefault(name, []).append(d)
    print(f"\n  {'arm':34s} {'fps (mean of both orders)':>26s} {'util':>7s}")
    for name in [a[0] for a in arms]:
        r = acc.get(name, [])
        if not r:
            continue
        print(f"  {name:34s} {sum(x['fps'] for x in r) / len(r):26.2f} "
              f"{sum(x['util'] for x in r) / len(r):6.1f}%   "
              f"own VRAM {max(x['own'] for x in r):.0f} MB")


if __name__ == '__main__':
    main()
