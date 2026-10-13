"""`roop.core` must stay the MODULE roop/core.py.

9b18807 added a `roop/core/` package beside it. A package beats a same-named
module in one directory, so `import roop.core` silently became an empty package:
`live_swap`, `batch_process_regular`, `get_processing_plugins`,
`decode_execution_providers`, ... all vanished, and every lazy
`from roop.core import ...` (api.py, bench.py, the benchmark runners) failed at
call time. The Stage 7 tests only imported the submodule, so the suite stayed
green. Nothing new may live under `roop/core/`.
"""
import os

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOP = os.path.join(APP, "app", "roop")


def test_no_core_package_shadows_core_module():
    assert os.path.isfile(os.path.join(ROOP, "core.py"))
    assert not os.path.exists(os.path.join(ROOP, "core")), (
        "roop/core/ shadows roop/core.py; put new modules directly under roop/")


def test_core_module_exports_survive():
    import roop.core as core
    assert core.__file__.endswith("core.py"), core.__file__
    for name in ("live_swap", "batch_process_regular",
                 "get_processing_plugins", "decode_execution_providers"):
        assert hasattr(core, name), name
