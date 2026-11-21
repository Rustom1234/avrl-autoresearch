"""Shared constants. T is part of the task statement: curators may import it."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]            # active-audit/
DATA = ROOT / "data" / "curation"
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(DATA / "tiktoken"))

SEQ = 512                  # context length
BATCH_TOKENS = 32768       # tokens per optimizer step (64 sequences)
STEPS = 300                # optimizer steps, fixed
T = STEPS * BATCH_TOKENS   # training token budget: 9,830,400
MAX_CHARS = 8000           # every document is truncated to this many characters
