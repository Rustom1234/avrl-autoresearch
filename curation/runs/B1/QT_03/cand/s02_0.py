"""Default curator ("smart-curate"): Pre-filter pool to remove obvious junk (empty, single-char, or all-special)
before shuffling. This ensures high-quality content dominates early training tokens.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens (T = 9,830,400, see common.py).
If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import string


# Common ASCII special characters (ASCII punctuation, space, tab, newline)
ASCII_SPECIAL = frozenset(string.punctuation + " \t\n")


def select(docs, target_sample):
    idx = list(range(len(docs)))
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    filtered_idx = []

    for doc_idx, doc in enumerate(docs):
        # Skip empty strings
        if not doc:
            continue
        # Skip strings consisting only of whitespace
        if not doc.strip():
            continue
        # Skip strings consisting entirely of ASCII special characters (excluding whitespace)
        doc_clean = doc.strip()
        if all(ch in ASCII_SPECIAL for ch in doc_clean):
            continue
        
        # Replace just one character, such as only one type of character
        if len(set(doc_clean)) == 1:
            ch = doc_clean[0]
            if ch.isspace() or ch in ASCII_SPECIAL:
                continue

        filtered_idx.append(doc_idx)

    rng.shuffle(idx)
    return idx
