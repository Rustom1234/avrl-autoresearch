"""Curator: rank pool documents by bigram coherence with the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Exact dedup on whitespace-normalised text (first occurrence kept).
  2. Drop documents under 50 tokens.
  3. Score = fraction of adjacent whitespace-separated word pairs (first 2000
     words, case preserved) that also occur in the target sample, plus a 0.05
     bonus if the first line looks like a WikiText title ("= Title =").
  4. Sort by score descending (seeded random tie-break) and take documents
     until the token total reaches T.
"""
import hashlib
import os
import random
import re

import tiktoken

from common import T

_MIN_TOKENS = 50
_MIN_WORDS = 20
_MAX_WORDS = 2000
_HEADER_BONUS = 0.05
_HEADER_RE = re.compile(r"^= .+ =$")


def _token_counts(texts):
    enc = tiktoken.get_encoding("gpt2")
    counts = []
    batch = 2048
    threads = max(1, min(8, os.cpu_count() or 1))
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        try:
            encoded = enc.encode_ordinary_batch(chunk, num_threads=threads)
            counts.extend(len(e) for e in encoded)
        except Exception:
            for d in chunk:
                try:
                    counts.append(len(enc.encode_ordinary(d)))
                except Exception:
                    counts.append(max(1, len(d) // 4))
    return counts


def _has_title_header(doc):
    # First non-empty line, stripped (WikiText lines carry leading/trailing spaces).
    for line in doc.split("\n"):
        line = line.strip()
        if line:
            return _HEADER_RE.match(line) is not None
    return False


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # Target bigram set: adjacent whitespace-separated word pairs, case preserved.
    bigrams = set()
    for t in target_sample:
        w = t.split()
        bigrams.update(zip(w, w[1:]))

    # Exact dedup on whitespace-normalised text, and coherence score.
    seen = set()
    kept = []
    scores = {}
    for i, d in enumerate(docs):
        if not isinstance(d, str):
            continue
        words = d.split()
        if not words:
            continue
        key = hashlib.md5(
            " ".join(words).encode("utf-8", errors="replace")
        ).digest()
        if key in seen:
            continue
        seen.add(key)

        if len(words) < _MIN_WORDS:
            c = 0.0
        else:
            w = words[:_MAX_WORDS]
            n_pairs = len(w) - 1
            hits = 0
            for p in zip(w, w[1:]):
                if p in bigrams:
                    hits += 1
            c = hits / n_pairs if n_pairs > 0 else 0.0

        if _has_title_header(d):
            c += _HEADER_BONUS

        kept.append(i)
        scores[i] = c

    # Token counts for the deduplicated documents only; drop very short ones.
    counts = _token_counts([docs[i] for i in kept])
    tok = {}
    candidates = []
    for i, n in zip(kept, counts):
        if n >= _MIN_TOKENS:
            tok[i] = n
            candidates.append(i)

    # Seeded random tie-breaker, highest score first.
    tiebreak = {i: rng.random() for i in candidates}
    order = sorted(candidates, key=lambda i: (-scores[i], tiebreak[i]))

    # Take the best documents until the T-token budget is filled.
    selected = []
    total = 0
    for i in order:
        selected.append(i)
        total += tok[i]
        if total >= T:
            break

    if selected:
        return selected
    if kept:
        return kept
    return list(range(len(docs)))
