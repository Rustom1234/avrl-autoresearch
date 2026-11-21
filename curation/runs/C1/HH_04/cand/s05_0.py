"""Vocabulary-coverage curator with wiki-style format-match filter.

Pipeline: exact-dedupe, drop damaged docs by vocabulary coverage, drop docs whose
wiki-style token ratio is below 0.5 * median of the target sample, then seeded shuffle.
No truncation.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os, random, re, hashlib, string, statistics
from collections import Counter

_WORD_RE = re.compile(r"[A-Za-z]+")
_MIN_DF = 20
_MIN_COVERAGE = 0.6
_MIN_WORDS = 5
_FORMAT_FACTOR = 0.5
_PUNCT = frozenset(string.punctuation)


def _words(text):
    return [w.lower() for w in _WORD_RE.findall(text)]


def _format_ratio(text):
    # Fraction of whitespace-separated tokens that are standalone punctuation or
    # markers (",", ".", "(", ")", ";", ":", "@-@", "@.@", "@,@", "=", ...).
    toks = text.split()
    if not toks:
        return 0.0
    n = sum(1 for t in toks if all(c in _PUNCT for c in t))
    return n / len(toks)


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

    # (4) format-match filter: drop pool docs whose wiki-style token ratio is below
    # 0.5 * median ratio over the target sample
    if target_sample:
        target_ratios = [_format_ratio(s) for s in target_sample]
        threshold = _FORMAT_FACTOR * statistics.median(target_ratios)
        survivors = [i for i in survivors if _format_ratio(docs[i]) >= threshold]

    # (5) seeded shuffle of survivors, no truncation
    random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(survivors)
    return survivors
