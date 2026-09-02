"""`GET /api/update/check` is read-only, and must stay that way.

`update_manager.py` carries two very different halves: `check()` collects
compatibility evidence and changes nothing, while `apply()` stages a candidate,
runs a health check and can roll back. Exposing the first to a browser is what
lets a UI answer "is there an update, and is it safe?"; exposing the second
would let a page mutate the installation.

These tests pin that boundary. The route must exist, must be a GET, must be
callable with no query string, and the module must not reference `apply` at
all -- a future edit that wires the installer to an HTTP handler fails here.
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import routes_diagnostics


class RouteShape(unittest.TestCase):
    def setUp(self):
        self.routes = [r for r in routes_diagnostics.router.routes
                       if getattr(r, "path", "") == "/api/update/check"]

    def test_the_route_is_registered_exactly_once(self):
        self.assertEqual(len(self.routes), 1, "expected one /api/update/check route")

    def test_it_is_a_get(self):
        self.assertEqual(set(self.routes[0].methods), {"GET"})

    def test_it_is_bound_to_the_check_handler_not_a_helper(self):
        self.assertEqual(self.routes[0].endpoint.__name__, "update_check")

    def test_no_required_query_parameter(self):
        """The UI issues a bare URL. `refresh` must be optional."""
        for field in (self.routes[0].dependant.query_params or []):
            self.assertFalse(field.field_info.is_required(),
                             f"{field.name} is a required query param")


class ItCannotInstallAnything(unittest.TestCase):
    SOURCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "routes_diagnostics.py")

    def test_the_module_never_calls_the_updater_apply_path(self):
        with open(self.SOURCE, encoding="utf-8") as handle:
            source = handle.read()
        for forbidden in (r"update_manager\.apply", r"\bapply\(\)",
                          r"_stage_candidate", r"_rollback", r"_create_snapshot"):
            self.assertIsNone(
                re.search(forbidden, source),
                f"routes_diagnostics.py must not reach the installer ({forbidden}); "
                "applying an update is Pinokio's update.js, not an HTTP handler")

    def test_the_response_names_where_an_update_is_actually_applied(self):
        """A client must be able to say so without hardcoding it."""
        with open(self.SOURCE, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn('"apply_channel": "pinokio"', source)


class DegradesHonestly(unittest.TestCase):
    def test_a_failed_check_is_unverified_not_up_to_date(self):
        """Offline, `git ls-remote` fails. That must never read as
        'no update available' with a clean bill of health."""
        original = sys.modules.get("update_manager")

        class Boom:
            @staticmethod
            def check():
                raise RuntimeError("remote unreachable")

        sys.modules["update_manager"] = Boom
        try:
            routes_diagnostics._UPDATE_CHECK_CACHE.update({"at": 0.0, "value": None})
            result = routes_diagnostics.update_check(refresh=True)
        finally:
            if original is None:
                sys.modules.pop("update_manager", None)
            else:
                sys.modules["update_manager"] = original
            routes_diagnostics._UPDATE_CHECK_CACHE.update({"at": 0.0, "value": None})

        self.assertEqual(result["classification"], "UNVERIFIED")
        self.assertFalse(result["available"])
        self.assertTrue(any("remote unreachable" in r for r in result["reasons"]),
                        result["reasons"])


if __name__ == "__main__":
    unittest.main()
