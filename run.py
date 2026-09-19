#!/usr/bin/env python3
"""Root entry point forwarding execution to app/run.py."""
import os
import sys

_APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)
os.chdir(_APP_DIR)

if __name__ == "__main__":
    import run
