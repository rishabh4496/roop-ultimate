"""Reproduce the live outage: Gradio fails, does the API survive?

Occupies the Gradio port so ui/main.py's launch fails, starts run.py, and then
asks the ONE question that matters: is /api/meta still answering after Gradio
has gone?

ARM=1 (default) simulates a React launcher and must PASS.
ARM=0 simulates the legacy launcher, where Gradio owns the process, and must
show the process exiting -- that is the control arm proving the fix is
conditional and did not just pin every process open forever.
"""
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(APP, 'env', 'Scripts', 'python.exe')
ARM = os.environ.get('ARM', '1')


def probe(port, timeout=8.0):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/meta',
                                    timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
blocker.bind(('127.0.0.1', 0))
blocker.listen(8)
gradio_port = blocker.getsockname()[1]

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('127.0.0.1', 0))
api_port = s.getsockname()[1]
s.close()

env = dict(os.environ)
env.update(ROOP_API_PORT=str(api_port), ROOP_GRADIO_PORT=str(gradio_port))
if ARM == '1':
    env['ROOP_REACT_CLIENT'] = '1'
else:
    env.pop('ROOP_REACT_CLIENT', None)

print(f'[test] ARM={ARM}  gradio {gradio_port} OCCUPIED  api {api_port}', flush=True)
proc = subprocess.Popen([PY, 'run.py'], cwd=APP, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding='utf-8', errors='replace')

# DRAIN THE PIPE.  A child writing to a full pipe BLOCKS -- including the API
# thread's own logging -- so an undrained probe wedges the process it is
# measuring and then reports it as dead.  This repository has already been
# bitten by exactly this once, in update_health's launch probe.
_lines = []
def _drain():
    for line in proc.stdout:
        _lines.append(line)
threading.Thread(target=_drain, daemon=True).start()

# 1) wait for the API to come up at all
up = False
deadline = time.time() + 300
while time.time() < deadline and proc.poll() is None:
    if probe(api_port):
        up = True
        break
    time.sleep(2.0)
print(f'[test] API came up: {up}', flush=True)

# 2) let Gradio fail and run.py decide, then re-probe with retries -- the
#    backend is initialising models and can be slow, which is not the same as
#    being dead.
time.sleep(30)
exited = proc.poll() is not None
still = False
for _ in range(6):
    if probe(api_port, timeout=15.0):
        still = True
        break
    time.sleep(5.0)
    exited = proc.poll() is not None

print(f'[test] process exited: {exited}   /api/meta still 200: {still}', flush=True)

proc.terminate()
try:
    proc.wait(timeout=45)
except Exception:
    proc.kill()
time.sleep(1.0)
out = ''.join(_lines)
blocker.close()

def _safe(t):
    return t.encode('ascii', 'replace').decode('ascii')
print('--- Gradio / Backend lines ---')
for line in out.splitlines():
    if any(k in line for k in ('Gradio', '[Backend]', 'Closing server')):
        print('   |', _safe(line.strip())[:170])

if ARM == '1':
    ok = up and still and not exited
    print('[test] RESULT:', 'PASS' if ok else 'FAIL',
          '- a React client must outlive a Gradio failure', flush=True)
else:
    ok = exited
    print('[test] RESULT:', 'PASS' if ok else 'FAIL',
          '- the legacy arm must still exit with Gradio (control)', flush=True)
sys.exit(0 if ok else 1)
