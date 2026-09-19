#!/usr/bin/env python3
"""Root entry point forwarding execution to app/run.py."""
import os
import runpy
import sys

_APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)
os.chdir(_APP_DIR)

if __name__ == "__main__":
    # Importing app/run.py as a module skips its __main__ startup path.  Execute
    # it with the script entry-point name so diagnostics, preflight, and UI
    # startup behave exactly like running the app launcher directly.
    runpy.run_path(os.path.join(_APP_DIR, "run.py"), run_name="__main__")
