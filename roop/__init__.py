import os
import pkgutil
import sys

# Ensure app directory is in sys.path so top-level imports like 'settings' resolve
_app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if os.path.isdir(_app_dir) and _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

# Extend package path so 'app/roop' modules are resolved when importing from root roop
_app_roop = os.path.join(_app_dir, "roop")
if os.path.isdir(_app_roop) and _app_roop not in __path__:
    __path__.append(_app_roop)
__path__ = pkgutil.extend_path(__path__, __name__)
