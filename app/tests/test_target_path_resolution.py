"""Where a RELATIVE target path is resolved from.

The backend's working directory is `app/` — start_react.js runs `python run.py`
with `path: "app"` — but everything a person is likely to name relatively sits
one level up at the project root: the launcher scripts, `output/`, and
`.pinokio-temp/`. A path typed into the "add by path" box as
`.pinokio-temp/image_10.png` was resolved with a bare `os.path.abspath`, landed
on `app/.pinokio-temp/...`, and came back "not a file on this machine" while
the file sat on disk a directory away. Confirmed against the running backend.

This is a latent wrong-root defect, NOT the cause of a reported failure — the
report it was chased from ("cannot create a recoverable project") was
`_json_default` in project_checkpoint.py, and the `.pinokio-temp` path in it
was Pinokio's name for a pasted SCREENSHOT rather than anything the UI sent.
Whether the drop zone can produce a relative path is unverified; these tests
cover the path-box route only, which is the one that was measured.

The rejection is an ordinary 200 response, so neither the terminal log nor a
return code showed it — hence a test rather than a review.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT = os.path.dirname(_APP)


class TargetPathResolution(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from api import _resolve_user_path, target_add_path, list_files_process
        cls.resolve = staticmethod(_resolve_user_path)
        cls.add_path = staticmethod(target_add_path)
        cls.queue = list_files_process

    def test_absolute_path_is_returned_unchanged(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as fh:
            fh.write(b'x')
        self.addCleanup(os.unlink, fh.name)
        self.assertEqual(self.resolve(fh.name), os.path.abspath(fh.name))

    def test_relative_path_resolves_against_the_project_root(self):
        # The real shape: a file under the project's .pinokio-temp, named the
        # way Pinokio names it, asked for relative to a working directory that
        # is app/ rather than the project root.
        directory = os.path.join(_PROJECT, '.pinokio-temp')
        os.makedirs(directory, exist_ok=True)
        target = os.path.join(directory, '_roop_resolution_probe.png')
        with open(target, 'wb') as fh:
            fh.write(b'x')
        self.addCleanup(os.unlink, target)
        previous = os.getcwd()
        os.chdir(_APP)
        try:
            resolved = self.resolve('.pinokio-temp/_roop_resolution_probe.png')
        finally:
            os.chdir(previous)
        self.assertEqual(os.path.normcase(resolved),
                         os.path.normcase(os.path.abspath(target)),
                         'a project-root-relative path must not be resolved under app/')

    def test_missing_relative_path_still_reports_a_path(self):
        # The rejection message names what comes back, so it has to stay a
        # path rather than becoming empty or None when nothing matches.
        resolved = self.resolve('.pinokio-temp/no_such_file_here.png')
        self.assertTrue(os.path.isabs(resolved))
        self.assertFalse(os.path.isfile(resolved))

    def test_add_path_accepts_a_pinokio_temp_drop(self):
        directory = os.path.join(_PROJECT, '.pinokio-temp')
        os.makedirs(directory, exist_ok=True)
        target = os.path.join(directory, '_roop_add_path_probe.png')
        with open(target, 'wb') as fh:
            fh.write(b'x')
        self.addCleanup(os.unlink, target)
        before = list(self.queue)
        self.addCleanup(lambda: (self.queue.clear(), self.queue.extend(before)))
        previous = os.getcwd()
        os.chdir(_APP)
        try:
            result = self.add_path({'paths': ['.pinokio-temp/_roop_add_path_probe.png']})
        finally:
            os.chdir(previous)
        self.assertEqual(result['rejected'], [])
        self.assertEqual([os.path.normcase(p) for p in result['added']],
                         [os.path.normcase(os.path.abspath(target))])


if __name__ == '__main__':
    unittest.main()
