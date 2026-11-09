"""pytest path setup for the P3 tests: puts p3_tests/, avrl/exp/ and loop/models/ (the call layer) on sys.path."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parents[2] / "loop" / "models"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
