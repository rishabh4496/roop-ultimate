"""The backend serves the built React client, and Node is build-time only.

This guards the cross-device failure the previous topology caused.

The launcher used to run `vite preview` as a SECOND server and point the
webview at it, with a Vite proxy forwarding /api and /ws back to the backend.
That worked on the machine the UI was built on and broke elsewhere, because it
put a Node toolchain on the RUNTIME path of a Python app:

1. `vite preview` refuses to start when `react-ui/dist` is missing, with
   "The directory "dist" does not exist. Did you build your project?". `dist/`
   is gitignored, so a fresh clone has one only if `npm run build` just
   succeeded. Any build failure -- wrong Node major (Vite 8 needs
   ^20.19 || >=22.12), a missing per-platform rolldown/oxlint binary, a bad
   npm cache -- therefore took down the SERVER, not just the build. The
   launcher's URL matcher never fired and the user saw a Vite error and no UI.

2. `vite preview` does NOT inherit `server.proxy`, so the proxy had to be
   declared twice; missing the `preview` copy made every fetch in the built
   client 404 against the static server, which the UI reported as "Cannot
   reach backend" while the backend was healthy.

Both classes of failure are structural to having two servers. `app/api.py` now
serves `react-ui/dist` itself, so there is one server, one port and one origin,
no proxy, and nothing Node-related running at render time. These tests pin that
shape down.

They evaluate the real files through node rather than scanning for substrings:
`start_react.js` exports a function, and a launcher can be broken in ways that
leave every keyword present.
"""

import json
import os
import shutil
import subprocess
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)


def _find_node():
    """node, without naming one machine's drive.

    Pinokio ships node in its own bin, which is on PATH inside a Pinokio shell
    but not necessarily in a bare terminal. Derive it from the environment or
    from this repo's own location (an app launcher lives at
    <PINOKIO_HOME>/api/<name>), never from a literal path -- this project runs
    on two machines with different drives, and `test_fixture_paths` exists
    because harnesses keep forgetting that.
    """
    found = shutil.which('node')
    if found:
        return found
    roots = [os.environ.get('PINOKIO_HOME'),
             os.path.abspath(os.path.join(ROOT, os.pardir, os.pardir))]
    for root in filter(None, roots):
        for rel in (('bin', 'miniforge', 'node.exe'), ('bin', 'miniforge', 'node'),
                    ('bin', 'miniconda', 'node.exe'), ('bin', 'miniconda', 'node'),
                    ('bin', 'nodejs', 'node.exe'), ('bin', 'nodejs', 'node')):
            candidate = os.path.join(root, *rel)
            if os.path.isfile(candidate):
                return candidate
    return None


NODE = _find_node()


def _node(script, cwd):
    out = subprocess.run([NODE, '--input-type=module', '-e', script],
                         capture_output=True, text=True, timeout=120, cwd=cwd)
    if out.returncode != 0:
        raise AssertionError(f'node failed: {out.stderr.strip()[:2000]}')
    return json.loads(out.stdout.strip().splitlines()[-1])


@unittest.skipUnless(NODE, 'node is not on PATH')
class LauncherServesTheBuild(unittest.TestCase):

    def setUp(self):
        script = f"""
        const mod = await import({json.dumps(
            'file:///' + os.path.join(ROOT, 'start_react.js').replace(os.sep, '/'))});
        const cfg = await mod.default({{ port: async () => 42003 }});
        console.log(JSON.stringify(cfg));
        """
        self.cfg = _node(script, ROOT)
        self.run_steps = self.cfg['run']
        self.ui_steps = [s for s in self.run_steps
                         if s.get('method') == 'shell.run'
                         and s.get('params', {}).get('path') == 'react-ui']
        self.build_steps = [s for s in self.ui_steps
                            if 'npm run build' in ' '.join(
                                s.get('params', {}).get('message', []))]
        self.backend_steps = [s for s in self.run_steps
                              if s.get('method') == 'shell.run'
                              and s.get('params', {}).get('path') == 'app'
                              and 'python run.py' in ' '.join(
                                  s.get('params', {}).get('message', [])
                                  if isinstance(s.get('params', {}).get('message', []), list)
                                  else [s.get('params', {}).get('message', '')])]

    def test_the_ui_is_built_not_served_by_node(self):
        """One UI step, and it builds. Nothing long-running from Node.

        A `preview`/`dev` step here would reintroduce the second server whose
        startup failure mode is the bug this file documents.
        """
        self.assertEqual(len(self.build_steps), 1,
                         'expected exactly one react-ui build step')
        message = ' ; '.join(self.build_steps[0]['params']['message'])
        self.assertIn('npm run build', message)
        self.assertNotIn('npm run preview', message)
        self.assertNotIn('npm run dev', message)

    def test_the_build_runs_before_the_backend(self):
        """The backend picks up dist/ at import time, so it must exist first."""
        ui_index = self.run_steps.index(self.build_steps[0])
        backend_index = self.run_steps.index(self.backend_steps[0])
        self.assertLess(ui_index, backend_index)

    def test_a_failed_build_stops_the_launch_loudly(self):
        """Without dist/ the backend serves the API but has no UI to show.

        A silent build failure would therefore present as a blank or 404 page
        rather than as the build error it actually is.
        """
        events = [o.get('event') for o in self.build_steps[0]['params'].get('on', [])]
        self.assertTrue(events, 'the build step watches for nothing')
        breaking = [o for o in self.build_steps[0]['params']['on'] if o.get('break')]
        self.assertTrue(breaking, 'a failing build does not stop the launch')

    def test_only_the_backend_publishes_a_url(self):
        """One server means one URL, and it is the backend's."""
        self.assertEqual(len(self.backend_steps), 1)
        events = [o.get('event') for o in self.backend_steps[0]['params']['on']]
        self.assertIn('/(http:\\/\\/[0-9.:]+)/', events)

    def test_backend_preflight_failure_breaks_before_url_local_set(self):
        events = self.backend_steps[0]['params']['on']
        breaking = [o for o in events if o.get('break')]
        self.assertTrue(breaking)
        self.assertTrue(any('FATAL' in o.get('event', '') for o in breaking))

    def test_the_backend_gets_the_api_port_and_a_clear_gradio_port(self):
        env = self.backend_steps[0]['params']['env']
        self.assertEqual(env['ROOP_API_PORT'], '42003')
        self.assertEqual(env['ROOP_REACT_CLIENT'], '1')
        # A Gradio/API port collision previously killed the whole backend.
        self.assertNotEqual(env['ROOP_GRADIO_PORT'], env['ROOP_API_PORT'])

    def test_a_url_capture_is_read_by_the_very_next_step(self):
        """`input` is the return value of the IMMEDIATELY PREVIOUS step.

        A `when` that evaluates false makes Pinokio skip the step and pass
        nothing on, so a `local.set` placed after a conditional branch can read
        the skipped one and leave `{{input.event[1]}}` unresolved. Pinokio then
        treats the literal template as a path:

            ENOENT: no such file or directory, stat '<app>\\{{input.event[1]}}'

        That is a real regression this launcher shipped once. So: every step
        that reads `input.event` must sit directly after a step that captures
        it, under the SAME `when`.
        """
        run = self.run_steps
        readers = [(i, s) for i, s in enumerate(run)
                   if 'input.event' in json.dumps(s.get('params', {}))]
        self.assertTrue(readers, 'nothing captures the UI url any more')
        for i, step in readers:
            self.assertGreater(i, 0, 'a capture reader cannot be the first step')
            prev = run[i - 1]
            self.assertEqual(prev.get('method'), 'shell.run',
                             f'step {i} reads input.event but follows '
                             f'{prev.get("method")!r}, which captures nothing')
            events = [o.get('event') for o in prev.get('params', {}).get('on', [])]
            self.assertTrue(any(e and '(http' in e for e in events),
                            f'step {i} reads input.event but the step before '
                            f'it has no capturing pattern')
            self.assertEqual(step.get('when'), prev.get('when'),
                             f'step {i} and its capture run under different '
                             f'conditions, so one can be skipped alone')

    def test_the_launcher_sets_both_url_and_api_url(self):
        """pinokio.js needs `url` for the tab and `api_url` for stop/pause."""
        sets = [s for s in self.run_steps if s.get('method') == 'local.set']
        self.assertEqual(len(sets), 1, 'exactly one local.set should run')
        params = sets[0]['params']
        self.assertIn('url', params)
        self.assertIn('42003', params['api_url'])

    def test_the_launcher_stays_a_daemon(self):
        """The backend must outlive the run array; it IS the app."""
        self.assertTrue(self.cfg.get('daemon'))


class InstallAndUpdateProduceABuild(unittest.TestCase):
    """dist/ is gitignored, so every path that prepares the app must build it.

    `npm install` alone leaves no UI for the backend to serve on a fresh
    clone, and leaves a STALE one after an update -- a pulled UI change would
    simply not appear.
    """

    def _read(self, name):
        with open(os.path.join(ROOT, name), encoding='utf-8') as handle:
            return handle.read()

    def test_install_builds_the_ui(self):
        install = self._read('install.js')
        self.assertIn('npm ci --no-audit --no-fund', install)
        self.assertIn('npm run build', install)
        self.assertIn('install_state.py commit --manifest', install)
        self.assertIn('.runtime-verification.json', install)
        self.assertIn('path.resolve(cwd, \'../../bin/miniforge\')', install)

    def test_update_rebuilds_the_ui(self):
        update = self._read('update.js')
        self.assertIn('npm ci --no-audit --no-fund', update)
        self.assertIn('npm run build', update)
        self.assertIn('install_state.py commit --manifest', update)
        self.assertIn('.runtime-verification.json', update)
        self.assertIn('path.resolve(cwd, \'../../bin/miniforge\')', update)

    def test_reset_removes_the_build(self):
        """Otherwise a "reset" app still boots the previous UI."""
        reset = self._read('reset.js')
        self.assertIn('react-ui/dist', reset)
        self.assertIn('.pinokio-install-complete.json', reset)

    def test_start_repairs_old_numpy_installations(self):
        start = self._read('start_react.js')
        self.assertIn('uv pip install numpy==1.26.4', start)


if __name__ == '__main__':
    unittest.main()
