"""Every name the repository imports from `roop.core` must exist on `roop.core`.

`test_core_module_not_shadowed.py` guards the cause of 5609800 (a `roop/core/` package beside
`roop/core.py`) with a short list of names. This guards the EFFECT for every caller: it reads the
source of the whole repository, collects each `from roop.core import X`, `import roop.core as c`
+ `c.X`, and `roop.core.X`, and fails on a name the module does not define. A package that
swallows `core.py`, a rename, or a deleted entry point all fail here, naming the file and line
that would break at call time (these imports are mostly lazy, inside functions, so nothing else
notices until a render or an API call reaches them).

`roop.core` stays one module on purpose: `core.py` is the CLI/headless entry, 2,000 lines of
module-level state that tests and the API monkeypatch by attribute (`roop.core.process_mgr`, ...);
turning it into a package would move that state behind an `__init__` re-export and is exactly the
shape that broke the app once.
"""
import ast
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(REPO, "app")
SKIP_DIRS = {"env", "node_modules", ".git", "__pycache__", "output", "models", "react-ui", "dist"}


def _python_files():
    for root, dirs, files in os.walk(APP):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(root, f)


def _uses():
    """[(name, file, line)] for every attribute the repo reads off roop.core."""
    found = []
    for path in _python_files():
        try:
            tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
        except SyntaxError:
            continue
        found.extend(_uses_in_tree(tree, path))
    return found


def _uses_in_tree(tree, path):
    found = []
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "roop.core":
            for a in node.names:
                if a.name != "*":
                    found.append((a.name, path, node.lineno))
        elif isinstance(node, ast.ImportFrom) and node.module == "roop":
            for a in node.names:
                if a.name == "core":
                    aliases.add(a.asname or "core")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "roop.core" and a.asname:
                    aliases.add(a.asname)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            base = node.value
            if isinstance(base, ast.Name) and base.id in aliases:
                found.append((node.attr, path, node.lineno))
            elif (isinstance(base, ast.Attribute) and base.attr == "core"
                    and isinstance(base.value, ast.Name) and base.value.id == "roop"):
                found.append((node.attr, path, node.lineno))
    return found


def test_every_name_the_repo_takes_from_roop_core_exists():
    import roop.core as core
    uses = _uses()
    assert len(uses) > 20, "the scan found almost nothing; the scanner is broken, not the repo"
    missing = sorted({(n, os.path.relpath(p, REPO), ln) for n, p, ln in uses if not hasattr(core, n)})
    assert not missing, "roop.core is missing names its callers import:\n" + "\n".join(
        f"  {n}  ({p}:{ln})" for n, p, ln in missing)


def test_the_entry_points_named_by_the_brief_are_callable():
    import roop.core as core
    for name in ("live_swap", "batch_process_regular", "get_processing_plugins",
                 "decode_execution_providers", "release_resources", "final_video_destination"):
        assert callable(getattr(core, name)), name
    assert core.__file__.endswith("core.py"), core.__file__
    assert not os.path.exists(os.path.join(APP, "roop", "core")), "roop/core/ would shadow core.py"


def test_the_scanner_would_notice_a_missing_name():
    """A scanner that cannot fail proves nothing: feed it every spelling of a name that is absent."""
    import roop.core as core
    source = (
        "from roop.core import definitely_missing_a\n"
        "import roop.core as c\n"
        "from roop import core as d\n"
        "def f():\n"
        "    c.definitely_missing_b()\n"
        "    d.definitely_missing_c\n"
        "    import roop\n"
        "    roop.core.definitely_missing_d\n"
    )
    names = {n for n, _, _ in _uses_in_tree(ast.parse(source), "synthetic.py")}
    assert names == {"definitely_missing_a", "definitely_missing_b", "definitely_missing_c",
                     "definitely_missing_d"}
    assert not any(hasattr(core, n) for n in names)
    assert "batch_process_regular" in {n for n, _, _ in _uses()}, \
        "the scan must see at least one real use of batch_process_regular"


class _Globals:
    pass


@pytest.fixture
def core_globals(tmp_path):
    import roop.core as core
    import roop.globals as g
    saved = {k: getattr(g, k, None) for k in ("_segment_spec", "_project_output_file", "output_path", "CFG")}

    class Cfg:
        output_template = "{file}"

    g.CFG, g.output_path = Cfg(), str(tmp_path)
    g._segment_spec = g._project_output_file = None
    yield core, g
    for k, v in saved.items():
        setattr(g, k, v)


def test_destination_for_an_ordinary_video_uses_the_template(core_globals):
    core, g = core_globals
    assert os.path.basename(core.final_video_destination("clip__temp.mp4", "clip.mp4", 0)) == "clip.mp4"


def test_destination_for_a_project_uses_its_fixed_name_for_the_first_file_only(core_globals):
    core, g = core_globals
    g._project_output_file = "fixed.mp4"
    assert os.path.basename(core.final_video_destination("clip__temp.mp4", "clip.mp4", 0)) == "fixed.mp4"
    assert os.path.basename(core.final_video_destination("clip__temp.mp4", "clip.mp4", 1)) == "clip.mp4"


def test_destination_for_a_slice_is_numbered_ranged_and_beats_the_template_and_project_name(core_globals):
    core, g = core_globals
    from roop.segment_output import SegmentSpec
    g._project_output_file = "fixed.mp4"
    names = set()
    for i, (s, e) in enumerate([(1, 200), (201, 400)], 1):
        g._segment_spec = SegmentSpec("G", i, 2, s, e)
        names.add(os.path.basename(core.final_video_destination("clip__temp.mp4", "C:/media/clip.mp4", 0)))
    assert names == {"clip_segment0001_000001_000200.mp4", "clip_segment0002_000201_000400.mp4"}
