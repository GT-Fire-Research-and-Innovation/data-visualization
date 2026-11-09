import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def mirror(tmp_path):
    """A throwaway mirror folder, so tests never write into the repo."""
    return tmp_path / "hf_mirror"
