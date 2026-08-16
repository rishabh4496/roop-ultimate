"""What /api/file is allowed to hand out.

The endpoint takes an absolute path straight off the query string and streams
it back, so its root list is the entire access-control boundary. Anything that
can reach the port can read anything under a root — and "anything that can
reach the port" stops meaning localhost the moment the "public server (share)"
setting is on.

The roots drifted once already, while fixing something unrelated: reaching a
`.pinokio-temp` directory that sits beside `app/` was done by allowing the
working directory AND its parent, which published config.yaml, .git, the source
tree and the logs along with it. Nothing failed — the media it was chasing
loaded fine — which is exactly why this needs a test rather than a review.

So: name directories of media, never an ancestor of the project.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT = os.path.dirname(_APP)


class _Req:
    headers = {}


class FileEndpointRoots(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from api import get_file
        cls.get_file = staticmethod(get_file)

    def _status(self, path):
        res = self.get_file(path=path, request=_Req())
        return getattr(res, 'status_code', 200)

    def _assert_forbidden(self, path, why):
        if not os.path.isfile(path):
            self.skipTest(f'{path} not present in this checkout')
        self.assertEqual(self._status(path), 403, why)

    def test_the_project_config_is_not_served(self):
        self._assert_forbidden(
            os.path.join(_APP, 'config.yaml'),
            'config.yaml holds the saved settings and is not media',
        )

    def test_the_source_tree_is_not_served(self):
        self._assert_forbidden(os.path.join(_APP, 'api.py'),
                               'the endpoint must not serve its own source')

    def test_the_git_directory_is_not_served(self):
        self._assert_forbidden(os.path.join(_PROJECT, '.git', 'HEAD'),
                               '.git carries history and remote credentials')

    def test_the_project_root_is_not_a_root(self):
        self._assert_forbidden(os.path.join(_PROJECT, 'README.md'),
                               'an ancestor of the project must never be a root')

    def test_a_neighbouring_app_is_not_served(self):
        """cwd's parent is other Pinokio apps when launched from api/<name>."""
        outside = os.path.join(os.path.dirname(_PROJECT), 'CLAUDE.md')
        self._assert_forbidden(outside, 'sibling installs are out of bounds')

    def test_pinokio_temp_media_is_still_served(self):
        """The fix must not re-break what it was reaching for."""
        d = os.path.join(_PROJECT, '.pinokio-temp')
        os.makedirs(d, exist_ok=True)
        probe = os.path.join(d, '_roots_test_probe.bin')
        with open(probe, 'wb') as fh:
            fh.write(b'media')
        try:
            self.assertNotEqual(
                self._status(probe), 403,
                '.pinokio-temp beside app/ is the directory this list exists to reach',
            )
        finally:
            try:
                os.remove(probe)
            except OSError:
                pass

    def test_a_missing_file_under_a_root_is_refused(self):
        self.assertEqual(
            self._status(os.path.join(_PROJECT, '.pinokio-temp', 'nope.bin')), 403)


if __name__ == '__main__':
    unittest.main()
