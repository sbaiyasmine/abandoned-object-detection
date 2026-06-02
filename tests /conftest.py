import sys
import os
import pytest

TEAM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _isolate_team_modules():
    """Ensure each test uses THIS team's src/common modules."""
    _purge = [m for m in sys.modules
              if m in ("src", "common")
              or m.startswith(("src.", "common."))]
    for m in _purge:
        del sys.modules[m]
    if TEAM_ROOT in sys.path:
        sys.path.remove(TEAM_ROOT)
    sys.path.insert(0, TEAM_ROOT)
    yield
    _purge = [m for m in sys.modules
              if m in ("src", "common")
              or m.startswith(("src.", "common."))]
    for m in _purge:
        del sys.modules[m]
