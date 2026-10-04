"""Hold N GiB of GPU memory in a separate process (another application's footprint).

    python ballast.py 5.0 <stopfile>

Allocates in 256 MiB blocks (a single huge allocation can fail on a fragmented card), touches
each so the driver commits it, prints the device-wide used/total, then sleeps until <stopfile>
exists. Exit code 3 if it could not get what it was asked for.
"""
import os
import sys
import time

import torch

gib = float(sys.argv[1])
stop = sys.argv[2]
block_mb = 256
blocks = []
want = int(gib * 1024 / block_mb)
for _ in range(want):
    try:
        t = torch.empty(block_mb * 1024 * 1024, dtype=torch.uint8, device='cuda')
        t.fill_(1)
        blocks.append(t)
    except Exception as exc:                      # noqa: BLE001
        print('ballast: could not allocate block %d/%d: %s' % (len(blocks) + 1, want, exc), flush=True)
        sys.exit(3)
torch.cuda.synchronize()
free, total = torch.cuda.mem_get_info()
print('ballast: holding %.2f GiB; device used %.0f / %.0f MiB' % (
    len(blocks) * block_mb / 1024, (total - free) / 2**20, total / 2**20), flush=True)
while not os.path.exists(stop):
    time.sleep(0.5)
print('ballast: released', flush=True)
