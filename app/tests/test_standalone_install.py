"""Roop Ultimate must be physically self-contained.

Until 2026-08-23 this was not true, and nothing said so. `app/env`, `app/models`
and `app/facesets` were NTFS junctions into a different working copy on the same
machine — 9.3 GB, 39.3 GB and 0.07 GB of virtual environment, model weights and
the user's own face libraries, all owned by another folder. Everything ran
perfectly, so the coupling was invisible: deleting or moving that folder would
have taken the whole application down, and the project could not have been moved
to another machine or handed to anyone without reproducing it.

The three directories are gitignored, so git could never have caught this. These
tests are the check that did not exist.

They are deliberately tolerant about ABSENCE — a fresh clone has no `env` or
`models` until the installer runs, and that is correct. What they refuse is a
directory that exists but is a link somewhere else.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)

# Everything a run needs that lives outside git and must therefore be real,
# local and owned by this project rather than borrowed from elsewhere.
LOCAL_DIRS = ('env', 'models', 'facesets')


def _is_reparse_point(path):
    """True for an NTFS junction or a symlink. `os.path.islink` alone is not
    enough on Windows: it reports False for a JUNCTION, which is exactly the
    kind of link this project was using."""
    if os.path.islink(path):
        return True
    if sys.platform != 'win32':
        return False
    try:
        FILE_ATTRIBUTE_REPARSE_POINT = 0x400
        return bool(os.stat(path, follow_symlinks=False).st_file_attributes
                    & FILE_ATTRIBUTE_REPARSE_POINT)
    except (OSError, AttributeError):
        return False


class TestStandaloneInstall(unittest.TestCase):
    def test_runtime_dirs_are_real_and_local(self):
        for name in LOCAL_DIRS:
            p = os.path.join(APP, name)
            if not os.path.exists(p):
                continue        # not installed yet — fine
            with self.subTest(dir=name):
                self.assertFalse(
                    _is_reparse_point(p),
                    f"app/{name} is a junction or symlink. This project must own "
                    f"its own environment, weights and facesets — if it points at "
                    f"another folder, deleting that folder breaks the install and "
                    f"the project cannot be moved or handed to anyone.")

    def test_the_venv_belongs_to_this_project(self):
        """A venv reached THROUGH a junction reports the junction's target as
        sys.prefix, so this is what actually catches a borrowed environment."""
        env = os.path.join(APP, 'env')
        if not os.path.exists(env):
            self.skipTest('no venv installed')
        prefix = os.path.realpath(sys.prefix)
        if os.path.realpath(os.path.dirname(sys.executable)) not in (
                os.path.realpath(os.path.join(env, 'Scripts')),
                os.path.realpath(os.path.join(env, 'bin'))):
            self.skipTest('not running under this project\'s venv')
        self.assertEqual(
            prefix, os.path.realpath(env),
            "the interpreter's sys.prefix is not this project's env/ — the "
            "virtual environment physically lives somewhere else")

    def test_no_runtime_dir_is_tracked_by_git(self):
        """The flip side: now that they are real directories holding ~49 GB,
        nothing may ever stage them.

        Asked of git itself rather than by reading a .gitignore, because the
        three are ignored from TWO different files — env and models from
        app/.gitignore, facesets from the root one — and a test that reads only
        one of them reports a false failure. (It did. That is why it asks git.)
        Also note `git check-ignore` resolves paths against the CWD, so it has to
        be run from the repository root or it silently answers about a path that
        does not exist.
        """
        import subprocess
        for name in LOCAL_DIRS:
            with self.subTest(dir=name):
                r = subprocess.run(['git', 'check-ignore', '-q', f'app/{name}'],
                                   cwd=ROOT, capture_output=True)
                if r.returncode == 128:
                    self.skipTest('not a git checkout')
                self.assertEqual(
                    r.returncode, 0,
                    f"app/{name} is not gitignored, and it now holds real "
                    f"multi-GB content that must never be staged")


class TestNoUpstreamCoupling(unittest.TestCase):
    """The project is independent; the shipped tree should read that way.

    Attribution belongs in NOTICE.md, where the AGPL requires it. Anywhere else
    an upstream name appears it is either stale branding or, worse, an install
    step that reaches into somebody else's repository — this project shipped
    installers that cloned upstream and downloaded a wheel from their releases.
    """

    NEEDLES = ('roop-unleashed', 'roop_unleashed', 'C0untFloyd', 's0md3v')

    # What this guard is FOR: the product surface — code, launcher, UI, package
    # metadata, READMEs. A name appearing there is either stale branding or an
    # install step reaching into somebody else's repository, and this project
    # shipped both.
    #
    # What it is NOT for: prose that talks ABOUT the separation. NOTICE.md
    # carries the attribution the licence requires. This file names the strings
    # it searches for. The development session logs record, among other things,
    # the work of removing those references — a changelog saying "X was removed"
    # is history, not an affiliation claim, and gagging it would make the record
    # worse. They are development notes rather than product surface, so a stale
    # mention there is harmless in a way one in `metadata.py` is not.
    ALLOWED = {'NOTICE.md', 'app/tests/test_standalone_install.py',
               'CLAUDE.md', 'GEMINI.md', 'QWEN.md', 'AGENTS.md', 'facegemini.md',
               '.clinerules', '.cursorrules', '.windsurfrules'}
    EXTS = {'.py', '.js', '.jsx', '.ts', '.tsx', '.json', '.md', '.sh', '.bat',
            '.yaml', '.yml', '.html', '.css'}

    def test_no_upstream_references_outside_notice(self):
        """Scans TRACKED files only — that is the surface a recipient receives.

        A filesystem walk instead flags local, untracked editor state
        (.claude/settings.local.json and friends), which is neither shipped nor
        anyone else's business.
        """
        import subprocess
        r = subprocess.run(['git', 'ls-files'], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest('not a git checkout')
        hits = []
        for rel in r.stdout.splitlines():
            rel = rel.strip()
            if not rel or rel in self.ALLOWED:
                continue
            if os.path.splitext(rel)[1].lower() not in self.EXTS:
                continue
            try:
                with open(os.path.join(ROOT, rel), encoding='utf-8',
                          errors='ignore') as f:
                    body = f.read().lower()
            except OSError:
                continue
            for n in self.NEEDLES:
                if n.lower() in body:
                    hits.append(f"{rel}: {n}")
                    break
        self.assertEqual(hits, [],
                         "upstream references in tracked files outside "
                         "NOTICE.md: " + "; ".join(hits))


if __name__ == '__main__':
    unittest.main()
