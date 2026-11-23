import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from helpers import make_store  # noqa: E402


@pytest.fixture(scope="session")
def store(tmp_path_factory):
    """One synthetic token store shared by all tests that only read from it."""
    return make_store(tmp_path_factory.mktemp("store"))
