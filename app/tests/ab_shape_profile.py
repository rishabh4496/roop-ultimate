"""A/B the TensorRT shape profile end to end, counterbalanced.

WHAT IS BEING MEASURED. `roop/trt_shape_profile.py` gives TensorRT an explicit
min/opt/max optimization profile for the models whose inputs are genuinely
dynamic.  THREE models get a profile in the live stack, established by
inspecting the engine cache a real render wrote (2026-09-03): `retinaface_r50`
(`_sp12x512input8x3x1280x1280`), the batch-dynamic `hififace` second net inside
`realswap` (`_spet8x3x256x256source8x512`), and `_spx512x512input8x3x512x512`.
An earlier note in `97d562a` claimed the detector was the only one; that was
true of the models sampled by hand and WRONG for the running stack.  So this
A/B asks "does pinning the shape range of the dynamic models help, hurt, or do
nothing" -- a broader question than the detector alone.

WHY IT MIGHT HELP. Without a profile TensorRT picks its own range from the
graph, and a shape outside what it assumed forces a rebuild or a slower
generic kernel.  With `opt` pinned at 512 -- the det_size config.yaml ships --
tactics are selected for the shape actually fed.

WHY IT MIGHT NOT. The pipeline feeds ONE det_size for a whole render, so
TensorRT already sees a single stable shape after the first frame.  The
profile may simply restate what it would have inferred.  Detection is also
~10 ms of a ~245 ms frame on this machine, and this project has now measured
three separate stage-level wins that were neutral end to end, because stage
share is thread time summed across workers, not a speedup budget.

TWO THINGS THIS HARNESS DOES THAT A NAIVE ONE WOULD NOT.

1.  EACH ARM IS ITS OWN PROCESS.  `baseline_controlled.py` already launches
    the render as a child, so the provider list is rebuilt per arm.  Running
    the arms in one process would let the first arm's cached ONNX sessions
    serve the second, and the A/B would measure nothing while looking fine.

2.  BOTH ENGINE CACHES ARE WARMED FIRST.  A profile changes the engine cache
    namespace, so the profiled arm starts with no engine while the unprofiled
    one may have been built weeks ago.  Measured cold, that is a ~213 s
    handicap on one side only -- larger than any effect being looked for.  The
    warm-up pass builds both, and is discarded.

Counterbalanced because end-to-end position alone moves this machine by more
than the effect looked for, and the null control is run first because the
resolution of the rig has to be known before a delta is believed.

    env/Scripts/python.exe tests/ab_shape_profile.py
    env/Scripts/python.exe tests/ab_shape_profile.py --reps 3 --null
"""

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PY = os.path.join(APP, 'env', 'Scripts', 'python.exe')
BASELINE = os.path.join(HERE, 'baseline_controlled.py')
OUT = os.path.join(APP, 'output', 'ab_shape_profile')

FLAG = 'ROOP_TRT_SHAPE_PROFILE'


def run_arm(tag, profile_on, end, extra_env=None):
    """One render in its own process; returns the harness's own result JSON."""
    os.makedirs(OUT, exist_ok=True)
    cmd = [PY, BASELINE, '--tag', tag, '--out', OUT, '--end', str(end),
           '--env', f'{FLAG}={1 if profile_on else 0}']
    for pair in (extra_env or []):
        cmd.extend(['--env', pair])
    started = time.time()
    proc = subprocess.run(cmd, cwd=APP, capture_output=True, text=True,
                          encoding='utf-8', errors='ignore')
    wall = time.time() - started
    path = os.path.join(OUT, tag + '.json')
    result = {}
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as handle:
            result = json.load(handle)
    result['_wall'] = wall
    result['_rc'] = proc.returncode
    log = os.path.join(OUT, tag + '.log')
    with open(log, 'w', encoding='utf-8', errors='ignore') as handle:
        handle.write(proc.stdout or '')
        handle.write('\n==== STDERR ====\n')
        handle.write(proc.stderr or '')
    return result


def metric(result, key, default=None):
    """Read a run metric.

    `baseline_controlled.py` nests its measurements under a `run` object.
    Reading them at the top level returns None for every arm, which prints
    `0.00 fps  swapped ?/?`, makes the mean zero so the delta guard
    (`if off and on and st.mean(off)`) silently prints NO result, and reduces
    the face-count guard to `[(None, None)]` -- the check that is supposed to
    catch an arm that "got faster" by finding fewer faces.  Everything still
    exits 0.  Found on 2026-09-03, after four arms had already printed zeros.
    """
    run = result.get('run')
    if isinstance(run, dict) and key in run:
        return run[key]
    return result.get(key, default)


def line(tag, result):
    return ('  %-26s %7.2f fps  %6.1fs  swapped %s/%s  rc=%s'
            % (tag, metric(result, 'fps') or 0.0, result.get('_wall', 0.0),
               metric(result, 'faces_swapped', '?'),
               metric(result, 'faces_seen', '?'),
               result.get('_rc')))


def summarise(title, values):
    if len(values) < 2:
        return
    spread = (max(values) - min(values)) / max(1e-9, st.mean(values)) * 100.0
    print('  %-18s n=%d  mean %.2f  min %.2f  max %.2f  spread %.1f%%'
          % (title, len(values), st.mean(values), min(values), max(values), spread))


def revisions(results):
    """Map each arm tag -> the source revision it ran against."""
    out = {}
    for tag, result in results:
        rev = result.get('source_revision') or {}
        head = rev.get('head')
        out[tag] = (head[:9] if head else 'unstamped',
                    'dirty' if rev.get('dirty') else 'clean')
    return out


def assert_one_tree(results):
    """REFUSE to summarise arms that did not run the same code.

    A counterbalanced design assumes the treatment is the only thing that
    varies.  On 2026-09-03 two feature commits landed mid-run, both touching
    face_swapper.py, and this harness averaged six arms across three versions
    of the swapper: the "identical config" null pair stepped 5.48 -> 7.68 fps
    and was read as machine noise.  Nothing objected because nothing recorded
    the tree.

    Prints its verdict either way -- a clean set must say so, or the guard is
    invisible until the day it fires.
    """
    revs = revisions(results)
    distinct = set(revs.values())
    print('')
    print('[ab] source revision per arm')
    for tag, rev in revs.items():
        print('  %-26s %s (%s)' % (tag, rev[0], rev[1]))
    if len(distinct) > 1:
        print('  !! ARMS RAN DIFFERENT CODE -- this comparison is VOID.')
        print('  !! %d distinct revisions across %d arms: %s'
              % (len(distinct), len(revs), sorted(distinct)))
        print('  !! A commit landing mid-run splits a counterbalanced set. '
              'Re-run on a quiescent tree.')
        return False
    if any(rev[1] == 'dirty' for rev in revs.values()):
        print('  !  arms share a revision but the tree was DIRTY: an '
              'uncommitted edit moves the code without moving HEAD.')
    elif 'unstamped' in {rev[0] for rev in revs.values()}:
        print('  !  arms are unstamped (older baseline_controlled.py); '
              'sameness of code is NOT established.')
    else:
        print('  OK all arms ran the same committed tree.')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps', type=int, default=2,
                        help='counterbalanced passes; 2 gives 4 measured arms')
    parser.add_argument('--end', type=int, default=600,
                        help='frames per arm; 600 is the locked window because '
                             '120 measures warm-up on this pipeline')
    parser.add_argument('--null', action='store_true',
                        help='run the null control (same config twice) first')
    parser.add_argument('--skip-warmup', action='store_true')
    args = parser.parse_args()

    print('=' * 74)
    every_arm = [(tag, r) for on in (False, True)
                 for tag, r in zip(['profile%d_rep%d' % (int(on), k)
                                    for k in range(len(results[on]))],
                                   results[on])]
    one_tree = assert_one_tree(every_arm)
    print('[ab] TensorRT shape profile, %d frames/arm, %d counterbalanced pass(es)'
          % (args.end, args.reps))
    print('=' * 74, flush=True)

    if not args.skip_warmup:
        print('\n[ab] warm-up: building BOTH engine caches (discarded)', flush=True)
        for on in (True, False):
            tag = 'warmup_profile%d' % int(on)
            result = run_arm(tag, on, min(args.end, 120))
            print(line(tag, result), flush=True)

    if args.null:
        print('\n[ab] null control: identical config, back to back', flush=True)
        nulls = []
        for index in range(2):
            tag = 'null_%d' % index
            result = run_arm(tag, True, args.end)
            nulls.append(metric(result, 'fps') or 0.0)
            print(line(tag, result), flush=True)
        summarise('null control', nulls)

    print('\n[ab] measured arms', flush=True)
    results = {True: [], False: []}
    for rep in range(args.reps):
        order = [True, False] if rep % 2 == 0 else [False, True]
        for on in order:
            tag = 'profile%d_rep%d' % (int(on), rep)
            result = run_arm(tag, on, args.end)
            results[on].append(result)
            print(line(tag, result), flush=True)

    print('\n' + '=' * 74)
    for on in (False, True):
        fps = [metric(r, 'fps') or 0.0 for r in results[on]]
        summarise('profile=%s' % ('ON' if on else 'OFF'), fps)
    off = [metric(r, 'fps') or 0.0 for r in results[False]]
    on = [metric(r, 'fps') or 0.0 for r in results[True]]
    if off and on and st.mean(off):
        delta = (st.mean(on) - st.mean(off)) / st.mean(off) * 100.0
        if one_tree:
            print('  profile ON vs OFF: %+.1f%%' % delta)
        else:
            print('  profile ON vs OFF: %+.1f%% -- VOID, arms ran '
                  'different code (see above); do not quote it.' % delta)
    swapped = {(metric(r, 'faces_swapped'), metric(r, 'faces_seen'))
               for group in results.values() for r in group}
    print('  face counts across all arms: %s' % sorted(swapped))
    print('  (identical counts are the guard: an arm that goes faster by '
          'finding fewer faces has not got faster)')
    print('=' * 74)


if __name__ == '__main__':
    main()
