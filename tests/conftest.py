import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def refs():
    """A test that runs the pi0-quant reference numerics (they import numba) requests this fixture, per test:
    it skips, as refutil.need_refs does, when the reference submodules are not at their pins or numba is missing.
    Modules that need it hold tests that need neither, so it is not a module-level skip."""
    from refutil import need_refs

    need_refs()
