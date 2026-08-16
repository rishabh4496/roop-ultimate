"""The run lives on its own tab, and every half of that has to stay wired.

A running job used to take the Face Swap tab over: the settings sidebar, the
asset rail, the timeline, the preview controls and the output section were all
hidden for the duration, and a progress panel took their place. So the tab you
set a job up on was the tab you could not use while one was running — queueing
the next clip meant waiting for the current one, or stopping it.

The progress panel is now the Processing tab, and Face Swap keeps its layout at
all times. Three things hold that up, and each of them fails quietly:

  * The tab is a lazy chunk. `lazy(() => import(...))` that is never RENDERED
    builds fine, ships a chunk nobody fetches, and shows an empty tab body.
  * The tab is transient — it is filtered out of the strip when no run is on —
    so a stale `TABS` reference in the nav or the command palette would either
    show it permanently or lose the filter's effect entirely.
  * The hiding in Face Swap was a `progress.processing ? 'hidden' : ''` on each
    panel. Re-adding one puts a panel back in the hole this change dug it out
    of, and nothing else notices: the tab still renders, the tests still pass,
    the panel is simply not there during a run.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')

APP_JSX = os.path.join(SRC, 'App.jsx')
FACESWAP_JSX = os.path.join(SRC, 'components', 'FaceSwap.jsx')
PROCESSING_JSX = os.path.join(SRC, 'components', 'Processing.jsx')


def _code(path):
    """The file with its comments removed.

    Every claim below is about what the code DOES, and the comments here talk
    about exactly the patterns being searched for — a raw substring search
    would match the prose explaining why the pattern is gone and report clean
    on a live regression.
    """
    with open(path, encoding='utf-8') as fh:
        src = fh.read()
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)          # block + JSX comments
    return '\n'.join(l.split('//', 1)[0] for l in src.split('\n'))


class ProcessingTabWiring(unittest.TestCase):
    def test_tab_is_declared_and_rendered(self):
        code = _code(APP_JSX)
        self.assertIn("id: 'processing'", code,
                      "the Processing tab is not in the tab list")
        self.assertIn("transient: true", code,
                      "the Processing tab must be transient — it exists only for a run")
        self.assertRegex(code, r"tab === 'processing'\s*&&\s*\(?\s*<Processing",
                         "the Processing tab is declared but never rendered")
        self.assertIn("const Processing = lazy(loadProcessing)", code,
                      "Processing is not code-split like the other tab panels")

    def test_props_app_passes_are_the_props_processing_reads(self):
        app = _code(APP_JSX)
        m = re.search(r'<Processing\b(.*?)/>', app, re.S)
        self.assertIsNotNone(m, 'no <Processing .../> render found in App.jsx')
        passed = set(re.findall(r'(\w+)=\{', m.group(1)))

        proc = _code(PROCESSING_JSX)
        d = re.search(r'export default function Processing\(\s*\{([^}]*)\}', proc)
        self.assertIsNotNone(d, 'Processing does not destructure its props')
        read = {b.strip().split(':')[0].split('=')[0].strip()
                for b in d.group(1).split(',') if b.strip()}

        missing = read - passed
        self.assertFalse(missing,
                         f'Processing reads props App never passes: {sorted(missing)}')

    def test_no_tabs_constant_survives_the_rename(self):
        # ALL_TABS is the full list; visibleTabs is what the strip and the
        # palette iterate. A leftover bare `TABS` would be an undefined name at
        # runtime, or worse, resolve to something else later.
        code = _code(APP_JSX)
        self.assertNotRegex(code, r'(?<![A-Z_])TABS\b(?!\s*=)',
                            'a bare TABS reference is left over; use ALL_TABS or visibleTabs')

    def test_faceswap_hides_nothing_while_a_run_is_in_flight(self):
        code = _code(FACESWAP_JSX)
        offenders = [l.strip() for l in code.split('\n')
                     if 'progress.processing' in l and "'hidden'" in l]
        self.assertFalse(offenders,
                         'Face Swap is hiding a panel during a run again:\n  '
                         + '\n  '.join(offenders))

    def test_the_run_controls_moved_rather_than_being_duplicated(self):
        # Pause/Resume and the live diagnostics belong to the Processing tab.
        # Two copies would mean two sets of controls drifting apart, and the
        # peek/terminal are the expensive ones to have mounted twice.
        faceswap = _code(FACESWAP_JSX)
        for gone in ('ProcessingDock', 'DiagnosticsPanel', 'LiveProcessingPeek',
                     'ProcessingTerminal'):
            self.assertNotIn(gone, faceswap,
                             f'{gone} is still mounted by Face Swap as well as Processing')

        proc = _code(PROCESSING_JSX)
        for needed in ('ProcessingDock', 'DiagnosticsPanel', 'LiveProcessingPeek',
                       'ProcessingTerminal'):
            self.assertIn(needed, proc, f'{needed} is not on the Processing tab')

    def test_run_start_selects_the_processing_tab(self):
        # Starting a job and then having to go and find where it went would be
        # the whole point of the split, undone.
        code = _code(APP_JSX)
        self.assertIn("setTab('processing')", code,
                      'nothing switches to the Processing tab when a run starts')


if __name__ == '__main__':
    unittest.main()
