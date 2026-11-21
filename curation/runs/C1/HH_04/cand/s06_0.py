"""Vocabulary-coverage curator: exact-dedupe, drop damaged docs by vocabulary coverage, then seeded shuffle.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Coverage threshold: first try the strict threshold (_STRICT_COVERAGE). If the
survivors' estimated token count (total characters / 4) is below T, fall back to
the looser _MIN_COVERAGE threshold. If that is still short, the looser survivors
are used as they are, and the list is cycled by the trainer as before.
"""
import os, random, re, hashlib
from collections import Counter

from common import T

_WORD_RE = re.compile(r"[A-Za-z]+")
_MIN_DF = 20
_MIN_COVERAGE = 0.6
_STRICT_COVERAGE = 0.75
_MIN_WORDS = 5


def _words(text):
    return [w.lower() for w in _WORD_RE.findall(text)]


def select(docs, target_sample):
    # (1) exact dedupe on hash of stripped text, keeping first occurrence
    seen = set()
    unique_idx = []
    for i, d in enumerate(docs):
        h = hashlib.sha1(d.strip().encode("utf-8", errors="replace")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        unique_idx.append(i)

    # (2) vocabulary: target-sample words + pool words with document frequency >= _MIN_DF
    vocab = set()
    for s in target_sample:
        vocab.update(_words(s))

    df = Counter()
    doc_words = {}
    for i in unique_idx:
        ws = _words(docs[i])
        doc_words[i] = ws
        df.update(set(ws))
    vocab.update(w for w, c in df.items() if c >= _MIN_DF)

    # (3) per-doc vocabulary coverage; drop docs with too few alphabetic words
    coverage = {}
    for i in unique_idx:
        ws = doc_words[i]
        if len(ws) < _MIN_WORDS:
            continue
        covered = sum(1 for w in ws if w in vocab)
        coverage[i] = covered / len(ws)

    # (4) apply the strict coverage threshold; fall back to the looser one if short of T
    strict = [i for i in unique_idx
              if i in coverage and coverage[i] >= _STRICT_COVERAGE]
    est_tokens = sum(len(docs[i]) for i in strict) / 4
    if est_tokens < T:
        # Relaxed threshold. If still short of T, these survivors are used as-is.
        survivors = [i for i in unique_idx
                     if i in coverage and coverage[i] >= _MIN_COVERAGE]
    else:
        survivors = strict

    # (5) seeded shuffle of survivors, no truncation
    random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(survivors)
    return survivors
