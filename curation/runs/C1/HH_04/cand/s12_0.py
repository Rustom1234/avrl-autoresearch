"""Vocabulary-coverage curator: exact-dedupe, prefix-dedupe, drop damaged docs by vocabulary coverage, then seeded shuffle.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os, random, re, hashlib
from collections import Counter

_WORD_RE = re.compile(r"[A-Za-z]+")
_WS_RE = re.compile(r"\s+")
_MIN_DF = 20
_MIN_COVERAGE = 0.6
_MIN_WORDS = 5
_PREFIX_CHARS = 200


def _words(text):
    return [w.lower() for w in _WORD_RE.findall(text)]


def _prefix_hash(text):
    norm = _WS_RE.sub(" ", text.strip().lower())
    return hashlib.sha1(norm[:_PREFIX_CHARS].encode("utf-8", errors="replace")).hexdigest()


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

    # (1b) prefix dedupe: hash of first 200 chars (lowercased, whitespace collapsed),
    # keeping first occurrence
    seen_prefix = set()
    prefix_unique_idx = []
    for i in unique_idx:
        ph = _prefix_hash(docs[i])
        if ph in seen_prefix:
            continue
        seen_prefix.add(ph)
        prefix_unique_idx.append(i)
    unique_idx = prefix_unique_idx

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

    # (3) drop docs with low vocabulary coverage or too few alphabetic words
    survivors = []
    for i in unique_idx:
        ws = doc_words[i]
        if len(ws) < _MIN_WORDS:
            continue
        covered = sum(1 for w in ws if w in vocab)
        if covered / len(ws) < _MIN_COVERAGE:
            continue
        survivors.append(i)

    # (4) seeded shuffle of survivors, no truncation
    random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(survivors)
    return survivors
