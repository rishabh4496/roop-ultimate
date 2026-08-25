"""Route-registration guards for `api.app`.

Written after `GET /api/settings` spent a release answering 422. The helper
`_public_settings(cfg)` was extracted and inserted directly beneath the
existing `@app.get("/api/settings")` decorator, so it took the decorator over
and the real handler below it was left undecorated. FastAPI read the untyped
`cfg` parameter as a REQUIRED QUERY STRING and rejected every bare GET.

Nothing in the suite noticed, because every test called `_public_settings`
and `get_settings` as plain Python functions -- which is exactly what they
still were. The defect lived entirely in the routing table, so that is what
these tests read.

The symptom is worth recording too: the React boot fetches /api/meta and
/api/settings together, so a 422 on either one lands in the same catch as a
dead socket and paints "Cannot reach backend". A healthy server looked down.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GetRoutesAreCallableWithoutQueryString(unittest.TestCase):
    """No GET route may demand a query parameter.

    This is the general form of the bug. Every GET this UI issues is a bare
    URL, so a required query param is unreachable by definition -- and the way
    one appears is never deliberate: an undecorated helper inherits a
    decorator, and its ordinary positional arguments silently become required
    query params.
    """

    # The one deliberate exception: /api/file IS a lookup, and the thing being
    # looked up is the query string. Listed by (path, param) rather than by
    # path so a second required param appearing on it still fails.
    ALLOWED = {('/api/file', 'path')}

    def test_no_get_route_has_a_required_query_param(self):
        import api
        offenders = []
        for route in api.app.routes:
            methods = getattr(route, 'methods', None) or set()
            dependant = getattr(route, 'dependant', None)
            if 'GET' not in methods or dependant is None:
                continue
            for field in dependant.query_params:
                # fastapi's ModelField wrapper dropped `.required` between
                # pydantic v1 and v2; the FieldInfo answer is the one that
                # survives both, and `.required` is tried first so this keeps
                # working if the venv is ever rolled back.
                required = getattr(field, 'required', None)
                if required is None:
                    required = field.field_info.is_required()
                if required and (route.path, field.name) not in self.ALLOWED:
                    offenders.append(f'{route.path} -> '
                                     f'{route.endpoint.__name__}({field.name})')
        self.assertEqual(
            [], offenders,
            'GET routes requiring a query parameter (usually a decorator that '
            'landed on a helper instead of its handler): ' + ', '.join(offenders))


class SettingsRouteIsTheHandler(unittest.TestCase):
    """The specific regression, pinned by name."""

    def _endpoint(self, path, method):
        import api
        for route in api.app.routes:
            if getattr(route, 'path', None) == path and method in (
                    getattr(route, 'methods', None) or set()):
                return route.endpoint
        return None

    def test_get_settings_is_registered(self):
        self.assertIsNotNone(self._endpoint('/api/settings', 'GET'),
                             'GET /api/settings is not registered at all')

    def test_get_settings_is_not_the_private_helper(self):
        endpoint = self._endpoint('/api/settings', 'GET')
        self.assertEqual('get_settings', endpoint.__name__)

    def test_public_settings_is_not_a_route(self):
        import api
        named = [r.path for r in api.app.routes
                 if getattr(r, 'endpoint', None) is api._public_settings]
        self.assertEqual([], named,
                         '_public_settings is a helper, not an endpoint')

    def test_save_settings_is_still_the_post(self):
        endpoint = self._endpoint('/api/settings', 'POST')
        self.assertIsNotNone(endpoint)
        self.assertEqual('save_settings', endpoint.__name__)


if __name__ == '__main__':
    unittest.main()
