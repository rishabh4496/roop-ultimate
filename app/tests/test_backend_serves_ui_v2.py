"""React UI 2.0 is served at /v2/ by the same backend, without disturbing 1.0 or the API.

2.0 (react-ui-v2/) is 1.0's client with a new visual layer, built with Vite `base: '/v2/'`. It is
served from the same 404 fallback as 1.0 (see test_backend_serves_ui.py for why nothing is
mounted), so these drive the real ASGI app through Starlette's TestClient and check the four ways
a second SPA can go quietly wrong: it shadows 1.0 at "/", it turns a missing asset into 200 HTML,
it lets `..` out of its dist/, or it makes an unknown /api path answer with HTML.
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

ROOT = os.path.dirname(APP)
DIST2 = os.path.join(ROOT, 'react-ui-v2', 'dist')
BUILT2 = api.ui2_dist_ready()
BUILT1 = api.ui_dist_ready()


class StatusReportsBothClients(unittest.TestCase):
    def test_status_names_the_v2_build_and_its_path(self):
        with TestClient(api.app) as client:
            body = client.get('/api/ui/status').json()
        self.assertEqual(os.path.normcase(body['v2']['dist']), os.path.normcase(DIST2))
        self.assertEqual(body['v2']['built'], BUILT2)
        self.assertEqual(body['v2']['path'], '/v2/')
        # 1.0's fields are unchanged in shape.
        self.assertIn('dist', body)
        self.assertIsInstance(body['built'], bool)


@unittest.skipUnless(BUILT2, 'react-ui-v2/dist has not been built')
class V2IsServedUnderItsPrefix(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(api.app)

    def test_v2_index_is_served(self):
        response = self.client.get('/v2/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/html', response.headers['content-type'])
        self.assertIn('Roop Ultimate 2.0', response.text)
        self.assertIn('/v2/assets/', response.text)

    def test_bare_prefix_redirects_to_the_slash_form(self):
        response = self.client.get('/v2', follow_redirects=False)
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers['location'], '/v2/')

    def test_v2_hashed_assets_are_served(self):
        assets = os.listdir(os.path.join(DIST2, 'assets'))
        js = next(a for a in assets if a.endswith('.js'))
        response = self.client.get(f'/v2/assets/{js}')
        self.assertEqual(response.status_code, 200)
        self.assertIn('javascript', response.headers['content-type'])

    def test_v2_bundled_fonts_are_served(self):
        woff = [a for a in os.listdir(os.path.join(DIST2, 'assets')) if a.endswith('.woff2')]
        self.assertTrue(woff, 'the v2 build bundles its fonts (offline rule)')
        response = self.client.get(f'/v2/assets/{woff[0]}')
        self.assertEqual(response.status_code, 200)

    def test_missing_v2_asset_is_a_404_not_html(self):
        response = self.client.get('/v2/assets/definitely-missing-123.js')
        self.assertEqual(response.status_code, 404)
        self.assertNotIn('text/html', response.headers.get('content-type', ''))

    def test_a_v2_deep_path_falls_back_to_the_v2_app(self):
        response = self.client.get('/v2/some/deep/route')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Roop Ultimate 2.0', response.text)

    def test_traversal_cannot_escape_v2_dist(self):
        for probe in ('/v2/../app/api.py', '/v2/..%2f..%2fapp%2fapi.py', '/v2/assets/../../../app/api.py'):
            response = self.client.get(probe)
            self.assertNotIn('def preview', response.text, probe)

    def test_api_is_not_shadowed_by_v2(self):
        response = self.client.get('/api/ui/status')
        self.assertEqual(response.status_code, 200)
        self.assertIn('application/json', response.headers['content-type'])

    def test_unknown_api_route_still_404_json(self):
        response = self.client.get('/api/v2/definitely_not_a_route')
        self.assertEqual(response.status_code, 404)
        self.assertNotIn('text/html', response.headers.get('content-type', ''))


@unittest.skipUnless(BUILT1 and BUILT2, 'both builds are needed to tell them apart')
class V1StaysAtTheRoot(unittest.TestCase):
    def test_root_is_still_1_0(self):
        with TestClient(api.app) as client:
            response = client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('Roop Ultimate 2.0', response.text)
        self.assertNotIn('/v2/assets/', response.text)


if __name__ == '__main__':
    unittest.main()
