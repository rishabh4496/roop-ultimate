import os
import pkgutil

# Extend package path so repo root 'roop' (including benchmark) is resolved when importing from app/roop
_root_roop = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "roop"))
if os.path.isdir(_root_roop) and _root_roop not in __path__:
    __path__.append(_root_roop)
__path__ = pkgutil.extend_path(__path__, __name__)
