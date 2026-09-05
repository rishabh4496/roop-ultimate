"""The React client is served as a BUILD, and both servers proxy /api.

Two failures this guards, both of which present as something other than what
they are:

1. `npm run dev` ships development React (StrictMode double-invokes every
   render and effect) as unbundled ESM. Measured on this repo: a cold load
   pulls 13.15 MB across 94 module requests against 0.85 MB for the built
   client, and Pinokio RELOADS this webview on every tab switch, so it is not
   a one-off startup cost. The production build takes ~1 s, so there is
   nothing to trade.

2. `vite preview` does NOT inherit `server.proxy`. A proxy defined only under
   `server` leaves every fetch in the built client hitting the static server
   and 404ing -- which the UI reports as "Cannot reach backend" while the
   backend is up and healthy. That exact misreport has already cost this
   project a debugging session once (2026-08-25 Part 2, via a different
   route), so it gets a test rather than a comment.

These evaluate the real files through node rather than scanning them for
substrings: `start_react.js` exports a function, and a launcher can be broken
in ways that leave every keyword present.
"""

import json
import os
import shutil
import subprocess
import sys
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
        self.steps = [s for s in self.cfg['run']
                      if s.get('method') == 'shell.run'
                      and s.get('params', {}).get('path') == 'react-ui']

    def test_there_is_a_default_step_and_an_opt_in_dev_step(self):
        self.assertEqual(len(self.steps), 2)
        default = [s for s in self.steps if s.get('when') == '{{!envs.ROOP_UI_DEV}}']
        dev = [s for s in self.steps if s.get('when') == '{{envs.ROOP_UI_DEV}}']
        self.assertEqual(len(default), 1, 'no unconditional-by-default UI step')
        self.assertEqual(len(dev), 1, 'the dev server is not reachable at all')
        self.default, self.dev = default[0], dev[0]

    def test_the_default_step_builds_and_previews(self):
        default = next(s for s in self.steps
                       if s.get('when') == '{{!envs.ROOP_UI_DEV}}')
        message = ' ; '.join(default['params']['message'])
        self.assertIn('npm run build', message)
        self.assertIn('npm run preview', message)
        self.assertNotIn('npm run dev', message)

    def test_the_dev_server_is_opt_in_only(self):
        dev = next(s for s in self.steps if s.get('when') == '{{envs.ROOP_UI_DEV}}')
        self.assertIn('npm run dev', ' ; '.join(dev['params']['message']))

    def test_both_ui_steps_are_given_the_api_port_and_their_own_port(self):
        for step in self.steps:
            env = step['params']['env']
            self.assertEqual(env['ROOP_API_PORT'], '42003')
            # The client's own port must not collide with the backend's.
            self.assertNotEqual(env['PORT'], env['ROOP_API_PORT'])

    def test_every_ui_step_reports_a_url_the_launcher_can_capture(self):
        """`local.set` reads `input.event[1]`, so each step must carry the
        capturing pattern -- otherwise the tab never gets its URL."""
        for step in self.steps:
            events = [o['event'] for o in step['params']['on']]
            self.assertIn('/(http:\\/\\/[0-9.:]+)/', events)


@unittest.skipUnless(NODE, 'node is not on PATH')
class BothViteServersProxyTheApi(unittest.TestCase):

    def setUp(self):
        ui = os.path.join(ROOT, 'react-ui')
        script = """
        process.env.ROOP_API_PORT = '42003';
        process.env.PORT = '42004';
        const mod = await import('./vite.config.js');
        const c = typeof mod.default === 'function' ? mod.default({}) : mod.default;
        const r = await c;
        console.log(JSON.stringify({
          server: r.server?.proxy?.['/api'] ?? null,
          preview: r.preview?.proxy?.['/api'] ?? null,
          serverPort: r.server?.port ?? null,
          previewPort: r.preview?.port ?? null,
        }));
        """
        self.cfg = _node(script, ui)

    def test_preview_has_an_api_proxy(self):
        """Without this the built client 404s every fetch and the UI blames
        the backend."""
        self.assertIsNotNone(self.cfg['preview'],
                             'vite preview has no /api proxy')

    def test_server_has_an_api_proxy(self):
        self.assertIsNotNone(self.cfg['server'])

    def test_both_proxies_point_at_the_same_backend(self):
        self.assertEqual(self.cfg['preview']['target'],
                         self.cfg['server']['target'])
        self.assertIn('42003', self.cfg['server']['target'])

    def test_both_servers_honour_the_launcher_assigned_port(self):
        self.assertEqual(self.cfg['serverPort'], 42004)
        self.assertEqual(self.cfg['previewPort'], 42004)


if __name__ == '__main__':
    unittest.main()
