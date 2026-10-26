"""One cancelled render, run in its OWN process so the exit code is observable (Windows 0xC0000005 is a process death).

Run by tests/test_release_signoff_vfr_color_cancel.py; prints one JSON line on stdout prefixed ``RESULT `` and exits through
the same teardown a normal run uses. Not a test module (leading underscore).

    python tests/_cancel_scenario.py <workdir> <stage> <stop_after> <clip> [frames]

* ``stage``      3: stop once ``stop_after`` frames were retired by the Stage 3 writer;
                 inflight: stop once Stage 3 has been RUNNING for ``stop_after`` seconds (GPU batches in flight, whether or
                 not a frame was written yet: the teardown the brief worries about);
                 2: stop during the pre-pass once ``stop_after`` frames were analysed.
* ``clip``       ``faces`` = the 180-frame clip with a face in every frame (a real swap render);
                 ``plain`` = ``frames`` face-free frames (the same reader, writer and finalisation, no model work).
"""
import json
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_pipeline_3stage_sync_and_fps as T  # noqa: E402


def main() -> int:
    workdir, stage, stop_after, clip_kind = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    n_plain = int(sys.argv[5]) if len(sys.argv) > 5 else 600
    os.makedirs(workdir, exist_ok=True)
    import roop.globals as g
    from roop import stage_telemetry as st
    import torch

    rig = T.Rig(workdir)
    rig.build()
    clip = rig.clip
    if clip_kind == 'plain':
        base = rig.shots()[0]
        clip = rig.write_clip([np.roll(base, i * 3, axis=1) for i in range(n_plain)], os.path.join(workdir, 'plain.mp4'))
    holder, t_stop = {}, {}

    def render():
        try:
            holder['run'] = rig.render('cancel', clip, rng=(0, 0), total=n_plain if clip_kind == 'plain' else T.N_FRAMES)
        except BaseException as exc:                 # a render that raises is a finding
            holder['error'] = f'{type(exc).__name__}: {exc}'

    th = threading.Thread(target=render, name='render', daemon=False)
    th.start()
    deadline = time.time() + 1500
    asked = False
    while th.is_alive() and time.time() < deadline:
        time.sleep(0.05)
        if asked:
            continue
        stages = st.snapshot().get('stages') or {}
        s2 = stages.get('STAGE_2_ACTIVE_PREPASS') or {}
        retired = int((st.retired_view() or {}).get('frames') or 0)
        s3 = stages.get('STAGE_3_MAIN_SWAP') or {}
        s3_running_s = float(s3.get('wall_s') or 0.0) if s3.get('state') == 'RUNNING' else 0.0
        if (stage == '3' and retired >= stop_after) or \
                (stage == 'inflight' and s3_running_s >= stop_after) or \
                (stage == '2' and s2.get('state') == 'RUNNING' and float(s2.get('stage_progress_pct') or 0) >= stop_after):
            # exactly what POST /api/stop does to the worker (api.py stop_swap)
            from roop import procmgr_runtime
            procmgr_runtime.pause_controller.cancel()
            g.pause = False
            g.processing = False
            t_stop.update(t=time.time(), retired=retired, s2_pct=s2.get('stage_progress_pct'))
            asked = True
    th.join(timeout=300)
    result = {'asked': asked, 'alive_after_join': th.is_alive(), 'error': holder.get('error'),
              'retired_at_stop': t_stop.get('retired'), 's2_pct_at_stop': t_stop.get('s2_pct')}
    run = holder.get('run')
    if run is not None:
        result.update(output=run.get('output'), render_error=run.get('error'),
                      finalize=(run.get('finalize') or [None])[-1],
                      retired=((run.get('report') or {}).get('stages') or {}).get('STAGE_3_MAIN_SWAP', {}).get('frames_retired'),
                      seconds_to_stop=round(time.time() - t_stop['t'], 2) if 't' in t_stop else None)
    # the cleanup the app runs between jobs, then what the GPU still holds
    try:
        import roop.core as core
        core.release_resources()
    except Exception as exc:
        result['release_error'] = f'{type(exc).__name__}: {exc}'
    try:
        torch.cuda.synchronize()
        result['cuda_allocated_mb'] = round(torch.cuda.memory_allocated() / 2 ** 20, 1)
    except Exception as exc:
        result['cuda_error'] = f'{type(exc).__name__}: {exc}'
    print('RESULT ' + json.dumps(result, default=str), flush=True)
    return 0


if __name__ == '__main__':
    code = main()
    sys.stdout.flush()
    sys.exit(code)      # interpreter teardown (CUDA contexts, DLPack, ffmpeg pipes) is part of what is being checked
