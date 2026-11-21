"""Filtered curator: drop repetitive and junk-like documents, keep the rest in a seeded random order.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os, random

# A line counts as a "long string without spaces" if it is at least this many
# characters and contains no whitespace at all.
LONG_NOSPACE_MIN_LEN = 40
# Drop a document if more than this fraction of its non-empty lines are long no-space strings.
LONG_NOSPACE_MAX_FRAC = 0.30
# Drop a document if its unique-line ratio is below this value.
MIN_UNIQUE_LINE_RATIO = 0.5


def _non_empty_lines(text):
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _is_repetitive(lines):
    if not lines:
        return False
    unique_ratio = len(set(lines)) / len(lines)
    return unique_ratio < MIN_UNIQUE_LINE_RATIO


def _is_junk_nospace(lines):
    if not lines:
        return False
    long_nospace = sum(
        1 for ln in lines
        if len(ln) >= LONG_NOSPACE_MIN_LEN and not any(ch.isspace() for ch in ln)
    )
    return long_nospace / len(lines) > LONG_NOSPACE_MAX_FRAC


def select(docs, target_sample):
    keep = []
    for i, doc in enumerate(docs):
        lines = _non_empty_lines(doc if isinstance(doc, str) else str(doc))
        if _is_repetitive(lines) or _is_junk_nospace(lines):
            continue
        keep.append(i)

    random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(keep)
    return keep
