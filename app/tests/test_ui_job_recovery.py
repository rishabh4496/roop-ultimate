"""Persistent job recovery: what may be remembered, and in what order.

The failure this design exists to avoid has two symmetric halves, and getting
the ORDER wrong picks one of them:

  * restore from localStorage, then poll  -> a window that confidently shows a
    progress bar for a job that ended half an hour ago;
  * poll, then merge                       -> a window that briefly shows "idle"
    over a live render, which on this app means the user starts a second one.

So the store opens in an explicit `reconciling` phase that is neither, and only
the reconcile may resolve it. These tests assert that the persisted partition
and the fallback path stay that way, because every one of these is invisible to
the build: a persisted `phase` would look completely correct until the day a run
happened to be interrupted.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')


def read(*parts):
    with open(os.path.join(SRC, *parts), encoding='utf-8') as fh:
        return fh.read()


# Comments here QUOTE the code they explain away ("`runTabOpen` is declared
# further down"), which is exactly the string some of these assertions look for.
# Strip them, or a test fails on its own documentation.
LINE_COMMENT = re.compile('//.*')
BLOCK_COMMENT = re.compile(r'/\*.*?\*/', re.S)


def code_only(text):
    return LINE_COMMENT.sub('', BLOCK_COMMENT.sub('', text))


class OnlyClientKnowledgeIsPersisted(unittest.TestCase):
    def test_conclusions_are_not_written_to_localStorage(self):
        """`phase`, `server` and `reattached` are conclusions drawn from a live
        server answer. Persisting one means restoring a conclusion about a
        world that has moved on."""
        src = read('store', 'jobStore.js')
        block = src[src.index('partialize:'):]
        block = block[:block.index('}),')]
        for persisted in ('jobId', 'startedAt', 'settings', 'label', 'queuedIds'):
            self.assertIn(persisted, block)
        for conclusion in ('phase', 'server', 'reattached', 'lastError'):
            self.assertNotIn(conclusion + ':', block, conclusion)

    def test_it_starts_in_neither_state(self):
        src = read('store', 'jobStore.js')
        self.assertIn("phase: 'unknown'", src)
        self.assertIn("beginReconcile: () => set({ phase: 'reconciling' })", src)

    def test_a_stale_memory_is_dropped_on_rehydrate(self):
        src = read('store', 'jobStore.js')
        self.assertIn('MAX_AGE_MS', src)
        self.assertIn('onRehydrateStorage', src)

    def test_storage_failure_does_not_take_the_app_down(self):
        """A webview with storage blocked must still boot: this store is
        imported at module scope by App.jsx."""
        src = read('store', 'jobStore.js')
        self.assertIn('createJSONStorage', src)
        self.assertIn('catch {', src)


class ReconcileIsTheOnlyResolver(unittest.TestCase):
    def test_unreachable_is_not_idle(self):
        """A backend restarting under a live render must not make this window
        declare the render over."""
        src = read('store', 'jobStore.js')
        block = src[src.index('if (!snap) {'):]
        block = block[:block.index('const remembered')]
        self.assertIn("phase: get().jobId ? 'active' : 'idle'", block)
        self.assertIn('backend unreachable', block)

    def test_a_finished_job_retires_the_memory(self):
        src = read('store', 'jobStore.js')
        block = src[src.index('if (!running) {'):]
        block = block[:block.index('// Running.')]
        self.assertIn("phase: 'idle'", block)
        self.assertIn('jobId: null', block)

    def test_the_server_clock_wins(self):
        src = read('store', 'jobStore.js')
        self.assertIn('snap.started_at ? snap.started_at * 1000', src)


class RecoveryPollsOnMountAndSurvivesAnOldBackend(unittest.TestCase):
    def test_it_asks_the_endpoint_first(self):
        src = read('components', 'faceswap', 'useJobRecovery.js')
        self.assertIn("getJSON('/api/jobs/active'", src)
        self.assertIn('beginReconcile();', src)

    def test_only_a_404_downgrades_to_the_legacy_path(self):
        """`app/api.py` runs in a NON-RELOADING uvicorn thread, so a user who
        pulls this change without restarting the backend gets a 404 — the
        documented failure mode for every route added to this app. But a
        TIMEOUT must not be read as "this backend is old", or one stalled
        request would permanently downgrade the client.
        """
        src = read('components', 'faceswap', 'useJobRecovery.js')
        block = src[src.index('legacyRef.current = true'):]
        self.assertIn("getJSON('/api/progress'", src)
        guard = src[src.index('} catch (e) {'):src.index('legacyRef.current = true')]
        self.assertIn('404|not found', guard)
        self.assertIn('throw e', guard)
        self.assertTrue(block)

    def test_the_fallback_produces_the_same_shape(self):
        src = read('components', 'faceswap', 'useJobRecovery.js')
        block = src[src.index('const normalise ='):]
        block = block[:block.index('});') + 3]
        for key in ('processing', 'job_id', 'started_at', 'label', 'queued'):
            self.assertIn(key, block)

    def test_reattachment_is_announced_once(self):
        """Announcing per poll would toast every fifteen seconds for the length
        of a forty-minute render."""
        src = read('components', 'faceswap', 'useJobRecovery.js')
        self.assertIn('announcedRef', src)
        self.assertIn('!announcedRef.current && hadMemory', src)

    def test_a_job_this_window_started_is_not_announced_as_recovered(self):
        src = read('components', 'faceswap', 'useJobRecovery.js')
        self.assertIn('after.reattached', src)

    def test_the_heartbeat_is_slow_and_pauses_when_hidden(self):
        """App's own 1 s progress poll drives the bar while attached; this
        exists only to notice a job that started elsewhere and to retire a dead
        memory. Polling it hard would take GPU away from the render, which is
        the whole reason render-lite mode exists."""
        src = read('components', 'faceswap', 'useJobRecovery.js')
        self.assertIn('HEARTBEAT_MS = 15000', src)
        self.assertIn('if (!document.hidden) poll()', src)


class AppWiresItWithoutReachingForwards(unittest.TestCase):
    def test_recovery_resumes_the_progress_poll(self):
        src = read('App.jsx')
        block = src[src.index('useJobRecovery({'):]
        block = block[:block.index('});')]
        self.assertIn('startPolling()', block)
        self.assertIn('setStartTime(startedAt)', block)

    def test_it_does_not_touch_state_declared_below_it(self):
        """`runTabOpen` is declared further down App.jsx. Reading a `const`
        before its declaration executes is a hard throw on first render, and
        this codebase has shipped that bug before (see test_ui_hook_order)."""
        src = read('App.jsx')
        callback = src[src.index('useJobRecovery({'):]
        callback = code_only(callback[:callback.index('});')])
        self.assertNotIn('setRunTabOpen', callback)
        self.assertNotIn('runTabOpen', callback)

    def test_the_store_is_updated_on_the_edge_not_the_level(self):
        """Writing on every poll would re-stamp `startedAt` forever, so the
        remembered start time would always be 'now'."""
        src = read('App.jsx')
        block = src[src.index('const prevPhaseRef = useRef(false);'):]
        block = block[:block.index('// ── The Processing tab')]
        self.assertIn('if (progress.processing && !was)', block)
        self.assertIn('beginJob({', block)
        self.assertIn('endJob()', block)


class ZustandIsActuallyADependency(unittest.TestCase):
    def test_declared_in_package_json(self):
        pkg = os.path.join(os.path.dirname(APP), 'react-ui', 'package.json')
        with open(pkg, encoding='utf-8') as fh:
            text = fh.read()
        self.assertTrue(re.search(r'"zustand":\s*"', text),
                        'the store imports zustand; it must be a declared '
                        'dependency or a clean checkout will not build')


if __name__ == '__main__':
    unittest.main()
