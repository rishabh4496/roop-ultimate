#!/usr/bin/env python3
"""Stage 19 integration verifier: what happens when a second face or an object crosses the target.

Renders three deterministic 30-frame synthetic scenes through the real pipeline (lossless,
un-trimmed, realswap + RealityUX) and prints measurements next to the thresholds the
Stage 19 brief asked for and the ratchets `tests/test_occlusion_interaction.py` enforces:

  1. crossing   a bystander slides across the target, sprite IoU peaking at 0.75, drawn in
                front of the target and then behind it
  2. bar        a skin-coloured bar sweeps across the target's eyes, nose and mouth:
                alignment of the swap crop vs the same frames without the bar
  3. flicker    second-order luma derivative and LAB colour of the swapped skin

Exit code 0 means every RATCHET held (the measured behaviour has not got worse). Brief
thresholds that the pipeline does not meet are printed as `MISS` and are tracked as strict
xfails in the test file, so they cannot silently start or stop passing.

Needs a CUDA/TensorRT device and the model files; roughly 8 minutes.
Usage (repo root):  app/env/Scripts/python.exe scripts/verification/verify_integration.py [--workdir DIR]
"""
import argparse
import os
import sys
import tempfile

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/verification/ -> repo root
for p in (os.path.join(ROOT, 'app'), os.path.join(ROOT, 'app', 'tests')):
    sys.path.insert(0, p)

import occlusion_interaction as oi      # noqa: E402
from occlusion_gates import BRIEF, RATCHET, check   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', default=None)
    ap.add_argument('--reuse', action='store_true', help='re-grade renders already in --workdir')
    args = ap.parse_args()
    work = args.workdir or tempfile.mkdtemp(prefix='s19_')
    print(f'[verify_integration] working in {work}')
    m = oi.run_all(work, reuse=args.reuse)
    import json
    print('\n== measurements ==')
    print(json.dumps({k: {a: b for a, b in v.items() if a != 'per_frame'} for k, v in m.items()}, indent=1))

    failed = 0
    print('\n== ratchets (must hold) ==')
    for name, ok, line in check(m, RATCHET):
        print(('  PASS  ' if ok else '  FAIL  ') + line)
        failed += 0 if ok else 1
    print('\n== brief thresholds (MISS = documented gap, strict xfail in the test file) ==')
    for name, ok, line in check(m, BRIEF):
        print(('  MET   ' if ok else '  MISS  ') + line)
    print(f'\n[verify_integration] {"ALL RATCHETS HELD" if not failed else f"{failed} RATCHET(S) FAILED"}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
