"""The backend serves the built UI without shadowing the API.

`app/api.py` serves `react-ui/dist` so the app needs no Vite process at
runtime (see test_react_launcher_serves_build for why that matters). Attaching
a catch-all to a router is easy to get subtly wrong, and every way of getting
it wrong is quiet:

* `app.mount("/", StaticFiles(...))` matches EVERY path, so it permanently
  shadows any route registered after it. Observed while building this: a route
  added below the mount returned 404 despite plainly existing in the source.
  The fix is to serve the SPA from the 404 handler, which by construction runs
  only after the router has already failed to match.
* Handing `index.html` to an unknown `/api` path turns a removed or mistyped
  endpoint into `200 text/html`, which reaches the client as an
  unintelligible JSON parse error rather than a clean 404.
* A naive path join lets `..` escape `dist/` and read arbitrary files.

These drive the real ASGI app through Starlette's TestClient, so they exercise
the actual routing table rather than asserting on the source text.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

os.environ.setdefault('ROOP_REACT_CLIENT', '1')

import api  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


DIST = os.path.join(os.path.dirname(APP), 'react-ui', 'dist')
BUILT = api.ui_dist_ready()


class UiStatusIsAlwaysAvailable(unittest.TestCase):
    """Reported whether or not a build exists, so 'not built' is
    distinguishable from 'not running' without guessing."""

    def test_status_reports_the_dist_location_and_state(self):
        with TestClient(api.app) as client:
            body = client.get('/api/ui/status').json()
        self.assertEqual(os.path.normcase(body['dist']), os.path.normcase(DIST))
        self.assertIsInstance(body['built'], bool)
        self.assertEqual(body['built'], BUILT)


@unittest.skipUnless(BUILT, 'react-ui/dist has not been built')
class TheSpaDoesNotShadowTheApi(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(api.app)

    def test_the_spa_is_served_at_the_root(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/html', response.headers['content-type'])
        self.assertIn('<div id="root">', response.text)

    def test_a_client_side_route_falls_back_to_the_app(self):
        """A deep link or a reload on an in-app view has no file behind it.
        Answering 404 shows the user a blank page."""
        response = self.client.get('/some/deep/client/route')
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/html', response.headers['content-type'])

    def test_an_unknown_api_route_is_still_a_real_404(self):
        response = self.client.get('/api/definitely_not_a_route')
        self.assertEqual(response.status_code, 404)
        self.assertNotIn('text/html', response.headers.get('content-type', ''))

    def test_an_unknown_ws_route_is_still_a_real_404(self):
        response = self.client.get('/ws/definitely_not_a_socket')
        self.assertEqual(response.status_code, 404)
        self.assertNotIn('text/html', response.headers.get('content-type', ''))

    def test_a_real_api_route_still_answers(self):
        response = self.client.get('/api/ui/status')
        self.assertEqual(response.status_code, 200)
        self.assertIn('application/json', response.headers['content-type'])

    def test_a_route_registered_after_the_spa_is_not_shadowed(self):
        """The ordering trap, pinned.

        A catch-all MOUNT would swallow this; a 404-handler fallback cannot,
        because it only runs after the router fails to match.
        """
        @api.app.get('/api/__shadow_probe__')
        def _probe():
            return {'reached': True}

        try:
            with TestClient(api.app) as client:
                response = client.get('/api/__shadow_probe__')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {'reached': True})
        finally:
            api.app.router.routes = [
                r for r in api.app.router.routes
                if getattr(r, 'path', None) != '/api/__shadow_probe__'
            ]

    def test_hashed_build_assets_are_served(self):
        assets = os.path.join(DIST, 'assets')
        entries = [f for f in os.listdir(assets) if f.endswith('.js')]
        self.assertTrue(entries, 'the build produced no JS assets')
        response = self.client.get(f'/assets/{entries[0]}')
        self.assertEqual(response.status_code, 200)

    def test_a_traversal_attempt_cannot_escape_dist(self):
        """`..` must not read files outside the build directory.

        A hit would return the requested file's contents; the SPA fallback
        returns index.html instead, so assert on the body rather than only on
        the status code.
        """
        for attempt in ('/../api.py', '/../../app/api.py', '/assets/../../api.py'):
            response = self.client.get(attempt)
            self.assertNotIn('uvicorn.run', response.text,
                             f'{attempt} escaped react-ui/dist')


if __name__ == '__main__':
    unittest.main()
