"""The four comparison grids must not share a localStorage key.

Enhancers, mask engines, swapper models and AI upscalers each remember which
variants you last put side by side. They now share one hook, which is what the
call sites should look like — but it also means each one is a near-identical
four-line block, and the storage key is the only thing distinguishing two of
them.

A copy-paste that leaves the key behind is invisible: both grids still work,
each still remembers a selection, and they silently overwrite each other's.
You would see it as "my swapper grid keeps resetting to mask engines", which
does not obviously point at a duplicated string.

The test also holds the extraction itself in place. Re-inlining a grid's state
would work fine and simply restore the four-way duplication the hook removed —
including the drift that had already started, where the enhancer grid's timer
state is still called `liveRenderingTimers` while the other three follow one
naming scheme.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')
FACESWAP = os.path.join(SRC, 'components', 'FaceSwap.jsx')
HOOK = os.path.join(SRC, 'components', 'faceswap', 'useCompareGrid.js')

STORAGE_KEY = re.compile(r"storageKey:\s*'([^']+)'")


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


class CompareGrids(unittest.TestCase):
    def setUp(self):
        self.keys = STORAGE_KEY.findall(_read(FACESWAP))

    def test_all_four_grids_use_the_shared_hook(self):
        self.assertEqual(
            len(self.keys), 4,
            'expected the four comparison grids (enhancers, masks, swappers, '
            f'upscalers) to each call useCompareGrid; found {len(self.keys)}: '
            + ', '.join(self.keys))

    def test_grid_storage_keys_are_distinct(self):
        dupes = sorted({k for k in self.keys if self.keys.count(k) > 1})
        self.assertEqual(
            dupes, [],
            'two comparison grids share a localStorage key, so they overwrite '
            "each other's saved selection: " + ', '.join(dupes))

    def test_the_hook_bounds_the_selection(self):
        """An empty or oversized list renders a broken grid, so it is rejected."""
        src = _read(HOOK)
        self.assertRegex(
            src, r'length\s*>=\s*1',
            'useCompareGrid must reject an empty stored selection, or the grid '
            'renders a comparison with nothing in it')
        self.assertRegex(
            src, r'length\s*<=\s*4',
            'useCompareGrid must cap the stored selection at four, which is '
            'what CompareGrid lays out')


if __name__ == '__main__':
    unittest.main()
