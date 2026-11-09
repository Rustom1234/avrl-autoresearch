import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


@pytest.fixture(scope="session")
def base():
    return (HERE / "fixtures" / "train_base.py").read_text()


@pytest.fixture(scope="session")
def edit():
    def _edit(src, old, new):
        assert src.count(old) == 1, "fixture anchor must be unique: %r" % old
        return src.replace(old, new)
    return _edit
