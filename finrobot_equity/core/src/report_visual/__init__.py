"""Visual report formats (HTML and PDF) built from one validated report model. See SCHEMA.md."""
import os as _os
import sys as _sys

_SRC = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # core/src, so `modules` imports resolve
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)
