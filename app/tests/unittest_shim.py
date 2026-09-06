"""Make module-level ``test_*`` functions visible to ``unittest discover``.

WHY THIS EXISTS. This project's standing verification command -- the one in
CLAUDE.md, and the one every session log quotes a green count from -- is::

    env/Scripts/python.exe -m unittest discover -s tests -t . -p "test_*.py"

``unittest`` only collects ``TestCase`` subclasses. Eight files here are written
in the bare-function style (``def test_x(): assert ...``), which pytest collects
and unittest does not. Run individually they reported::

    Ran 0 tests in 0.000s
    OK

That is the exact failure shape this repository keeps re-discovering: a check
that reports success having done no work. Thirty real, passing assertions --
including the Phase 11 inventory, the Phase 15 cross-hardware audit and the
Phase 16 final quality gate -- were absent from every "suite green" figure on
record, and a regression in any of them would not have failed the suite.

HOW. Each affected module gains a three-line ``load_tests`` hook. unittest calls
it during discovery; pytest ignores it entirely, so both runners keep working
and neither double-counts (the generated TestCase is built inside the call and
never exists at module scope for pytest to find).

``tmp_path`` is supported because three of these tests take pytest's fixture of
that name. Nothing else from pytest's fixture system is provided: a test that
needs more than a temporary directory should be written as a TestCase.
"""

import inspect
import pathlib
import tempfile
import unittest


def _wrap(func):
    """Adapt a bare test function to a TestCase method."""
    try:
        params = list(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        params = []
    unsupported = [name for name in params if name != "tmp_path"]
    if unsupported:
        def method(self, _func=func, _names=tuple(unsupported)):
            self.skipTest(
                "%s needs pytest fixture(s) %s, which the unittest shim does "
                "not provide; run it under pytest or rewrite it as a TestCase"
                % (_func.__name__, ", ".join(_names)))
        method.__doc__ = func.__doc__
        return method

    if "tmp_path" in params:
        def method(self, _func=func):
            with tempfile.TemporaryDirectory() as tmp:
                _func(tmp_path=pathlib.Path(tmp))
    else:
        def method(self, _func=func):
            _func()
    method.__doc__ = func.__doc__
    return method


def module_function_suite(namespace, loader=None):
    """A TestSuite over every module-level ``test_*`` function in ``namespace``.

    Pass ``globals()``. Functions imported from elsewhere are skipped, so a
    helper pulled in from another test module is not counted twice.
    """
    module_name = namespace.get("__name__", "module_function_tests")
    own = []
    for name, value in namespace.items():
        if not name.startswith("test_") or not inspect.isfunction(value):
            continue
        if value.__module__ != module_name:
            continue
        own.append((name, value))

    attrs = {name: _wrap(func) for name, func in sorted(own)}
    case = type("ModuleFunctionTests", (unittest.TestCase,), attrs)
    case.__module__ = module_name
    return (loader or unittest.TestLoader()).loadTestsFromTestCase(case)


def load_tests_for(namespace):
    """The whole ``load_tests`` hook, so call sites stay three lines.

        def load_tests(loader, tests, pattern):
            from tests.unittest_shim import load_tests_for
            return load_tests_for(globals())
    """
    return module_function_suite(namespace)
