"""Counterbalanced A/B: one inference owner (streaming) vs parallel blocks.

WHY THIS EXISTS. A terminal profile of a stabilized render reads like a
threading problem -- hundreds of OS threads, most cores idle, GPU short of
saturation, and the scheduler's input queue permanently full at `in_flight 1`.
It is not. `ProcessMgr` sets `_inference_workers = 1` on the streaming
stabilization path, and every one of those symptoms follows from that single
line: decode fills a shallow queue that one consumer drains.

This harness measures the thing that actually gates the pipeline. It varies
ROOP_STAB_STREAMING and reports, per arm, wall clock beside GPU utilisation,
OS THREAD COUNT and swap rate -- because:

  - a configuration that goes faster by finding fewer faces has not got
    faster, so the swap counts are printed and must be compared;
  - the block path re-processes warm-up frames, so its `faces seen` is
    legitimately ~28% higher than the stream's for the SAME output. Compare
    OUTPUT FRAME COUNTS, not face counts, to check for lost work;
  - thread count is reported so the GIL hypothesis can be tested rather than
    assumed. On the 2026-09-02 run the FASTER arm had FEWER threads.

Each arm is a separate process: the TensorRT engine cache, the sticky path
flags and the CUDA context must not leak between arms. Arms are ordered ABBA
because the first arm of a process pays the engine build -- read without
counterbalancing, comparisons on this pipeline have been seen to invert.

    env/Scripts/python.exe tests/ab_stab_owner.py
    env/Scripts/python.exe tests/ab_stab_owner.py --end 600 --reps 2

Measured on an RTX 4070, production stabilizer settings, 141 output frames:

    streaming (cuda_owner=1)   155.5 / 154.8 s   GPU peak 68-73%   256 threads
    parallel blocks (8 wkrs)   126.4 / 139.0 s   GPU peak 92-97%   198 threads
                               -> blocks +16.9%
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures  # noqa: E402

PY = os.path.join(APP, 'env', 'Scripts', 'python.exe')
SEEN_RE = re.compile(r'faces seen\s+(\d+)')
SWAP_RE = re.compile(r'swapped \([^)]*\)\s+(\d+)\s+([\d.]+)%')
MODE_RE = re.compile(r'unified coordinator ON: mode=(\w+), workers=(\d+)')
PAR_RE = re.compile(r'parallel stabilization ON \(threads=(\d+)')


def _live_config():
    import yaml
    with open(os.path.join(APP, 'config.yaml')) as fh:
        return yaml.safe_load(fh) or {}


def _monitor(proc, stop, gpu, threads, cpu):
    """Sample GPU, whole-tree thread count and system CPU once a second."""
    try:
        import psutil
        handle = psutil.Process(proc.pid)
    except Exception:
        return
    while not stop.is_set():
        try:
            count = handle.num_threads()
            for child in handle.children(recursive=True):
                try:
                    count += child.num_threads()
                except Exception:
                    pass
            threads.append(count)
            cpu.append(psutil.cpu_percent(interval=None))
            res = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5)
            gpu.append(int(res.stdout.strip().splitlines()[0]))
        except Exception:
            pass
        time.sleep(1.0)


def run_arm(args, cfg, tag, streaming, out_dir):
    env = dict(os.environ)
    env['ROOP_STAB_STREAMING'] = '1' if streaming else '0'
    mask = {'RealityUX': 'mask_realityux', 'DFL XSeg': 'mask_xseg',
            'None': 'None'}.get(cfg.get('mask_engine', 'RealityUX'), 'None')
    cmd = [PY, os.path.join('tests', 'two_face_video.py'),
           '--tag', tag, '--video', args.clip, '--sources', args.sources,
           '--end', str(args.end), '--provider', cfg.get('provider', 'tensorrt'),
           '--swap-model', cfg.get('swap_model', 'realswap'),
           '--enhancer', cfg.get('selected_enhancer', 'UltraMax'),
           '--mask-engine', mask,
           '--threads', str(int(cfg.get('max_threads', 10))),
           '--codec', cfg.get('output_video_codec', 'hevc_nvenc'),
           # Stated explicitly: two_face_video defaults these OFF, and with
           # them off NEITHER path under test is selected -- the run then
           # silently measures something else entirely.
           '--stabilize-face', '1', '--stabilize-enhancer', '1',
           '--stabilize-mask', '1', '--out', out_dir]
    gpu, threads, cpu = [], [], []
    stop = threading.Event()
    started = time.time()
    proc = subprocess.Popen(cmd, cwd=APP, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            errors='ignore')
    watcher = threading.Thread(target=_monitor,
                               args=(proc, stop, gpu, threads, cpu),
                               daemon=True)
    watcher.start()
    text = proc.communicate()[0]
    wall = time.time() - started
    stop.set()
    watcher.join(timeout=3)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, tag + '.log')
    with open(log_path, 'w', encoding='utf-8', errors='ignore') as fh:
        fh.write(text)

    seen = SEEN_RE.search(text)
    swapped = SWAP_RE.search(text)
    mode = MODE_RE.search(text)
    par = PAR_RE.search(text)
    return dict(
        tag=tag, streaming=bool(streaming), rc=proc.returncode, wall=wall,
        seen=int(seen.group(1)) if seen else None,
        swapped=int(swapped.group(1)) if swapped else None,
        swap_pct=float(swapped.group(2)) if swapped else None,
        mode=mode.group(1) if mode else ('parallel' if par else '?'),
        workers=(int(mode.group(2)) if mode
                 else (int(par.group(1)) if par else None)),
        gpu_mean=statistics.mean(gpu) if gpu else None,
        gpu_max=max(gpu) if gpu else None,
        threads_mean=statistics.mean(threads) if threads else None,
        threads_max=max(threads) if threads else None,
        cpu_mean=statistics.mean(cpu) if cpu else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', default=fixtures.clip('double/d1.mp4'))
    ap.add_argument('--sources', default='harjot,gargee')
    ap.add_argument('--end', type=int, default=600)
    ap.add_argument('--reps', type=int, default=1,
                    help='ABBA passes; one pass is already counterbalanced')
    ap.add_argument('--out',
                    default=os.path.join(APP, 'output', 'ab_stab_owner'))
    args = ap.parse_args()

    cfg = _live_config()
    os.makedirs(args.out, exist_ok=True)
    print('=' * 78)
    print('[ab] {}  {} / {} / {} / threads={} / {}'.format(
        os.path.basename(args.clip), cfg.get('selected_enhancer'),
        cfg.get('swap_model'), cfg.get('mask_engine'),
        cfg.get('max_threads'), cfg.get('provider')))
    print('[ab] ROOP_STAB_STREAMING 1 (one inference owner) vs 0 (parallel '
          'blocks), counterbalanced ABBA')

    plan = []
    for rep in range(max(1, args.reps)):
        plan += [('stream_%da' % rep, True), ('block_%da' % rep, False),
                 ('block_%db' % rep, False), ('stream_%db' % rep, True)]

    results = []
    for tag, streaming in plan:
        row = run_arm(args, cfg, tag, streaming, args.out)
        results.append(row)
        # A failed arm still has a wall clock, and a wall clock still divides
        # into a plausible frame rate. Refuse to carry one forward.
        if row['rc'] != 0 or not row['seen'] or not row['swapped']:
            print('  !! {} DID NOT RENDER (rc={} seen={} swapped={}) -- '
                  'see {}.log'.format(tag, row['rc'], row['seen'],
                                      row['swapped'], tag))
            return 2
        print('  {:11s} stream={} mode={:9s} workers={}  wall {:7.1f}s  '
              'gpu {:.0f}%/{}%  threads {:.0f}/{}  swapped {} ({}%)'.format(
                  tag, int(streaming), row['mode'], row['workers'],
                  row['wall'], row['gpu_mean'] or 0, row['gpu_max'],
                  row['threads_mean'] or 0, row['threads_max'],
                  row['swapped'], row['swap_pct']), flush=True)

    with open(os.path.join(args.out, 'results.json'), 'w') as fh:
        json.dump(results, fh, indent=2)

    stream = [r['wall'] for r in results if r['streaming']]
    block = [r['wall'] for r in results if not r['streaming']]
    s_mean, b_mean = statistics.mean(stream), statistics.mean(block)
    print()
    print('STREAM (one owner) : {}  mean {:.1f}s'.format(
        ['%.1f' % x for x in stream], s_mean))
    print('BLOCKS (N workers) : {}  mean {:.1f}s'.format(
        ['%.1f' % x for x in block], b_mean))
    print('DELTA blocks vs stream: {:+.1f}% faster'.format(
        (s_mean / b_mean - 1) * 100))
    print('worst block {:.1f}s vs best stream {:.1f}s -> {}'.format(
        max(block), min(stream),
        'separated' if max(block) < min(stream)
        else 'OVERLAPPING, not resolvable'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
