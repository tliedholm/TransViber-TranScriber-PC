"""Offline regression tests for TransViber TransScriber.

Run from the project root:

    .venv\\Scripts\\python.exe -m unittest discover -s tests -t .

Everything here runs without a GPU, a model download, or a network
connection, and in a few seconds. The app's modules import each other by
bare name (``import merger``, ``from models import Segment``) because they
run as a package directory rather than an installed package, so ``app/``
goes on the path here once for every test module.
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent / "app"
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))
