"""Curator: two-tier ranking on top of target-frequency weighted shared word count.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Ranking:
  * tier 0 docs are read first, then tier 1 docs;
  * within a tier, docs are sorted by the score (sum of target word frequencies
    over the doc's unique lowercased words, divided by its unique word count),
    highest first, with a seeded random tie-break.

A doc is demoted to tier 1 if it is an exact duplicate (whitespace-normalised)
of an earlier doc, or if it fails a word-order check calibrated on the target
sample (adjacent word pair hit rate below a threshold, or fewer than 30 words).
Nothing is dropped.
"""
import hashlib
import os
import random
import re
import statistics
from collections import Counter

import tiktoken

from common import T

_WORD_RE = re.compile(r"\w+")

_MAX_WORDS = 1500   # words inspected per doc for the word-order check
_MIN_WORDS = 30     # docs with fewer words are demoted


def _words(text):
    return set(_WORD_RE.findall(text.lower()))


def _token_counts(docs):
    enc = tiktoken.get_encoding("gpt2")
    counts = []
    batch = 2048
    threads = max(1, min(8, os.cpu_count() or 1))
    for start in range(0, len(docs), batch):
        chunk = docs[start:start + batch]
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


def _loo_rate(words, pair_counts, own_counts):
    """Leave-one-out hit rate of the adjacent pairs in the first _MAX_WORDS words.

    A pair is a hit if it still appears in pair_counts after subtracting the
    doc's own pair counts. Returns None if there are no pairs.
    """
    head = words[:_MAX_WORDS]
    n = len(head) - 1
    if n <= 0:
        return None
    hits = 0
    for p in zip(head, head[1:]):
        if pair_counts.get(p, 0) - own_counts.get(p, 0) > 0:
            hits += 1
    return hits / n


def _calibrate(target_sample, seed):
    """Build the target pair Counter and the word-order threshold."""
    target_words = [t.split() for t in target_sample]

    pair_counts = Counter()
    own_counts_list = []
    for w in target_words:
        own = Counter(zip(w, w[1:]))
        own_counts_list.append(own)
        pair_counts.update(own)

    shuffle_rng = random.Random(seed)
    clean_rates = []
    shuffled_rates = []
    for w, own in zip(target_words, own_counts_list):
        r = _loo_rate(w, pair_counts, own)
        if r is not None:
            clean_rates.append(r)

        shuffled = list(w)
        shuffle_rng.shuffle(shuffled)
        r = _loo_rate(shuffled, pair_counts, own)
        if r is not None:
            shuffled_rates.append(r)

    if clean_rates and shuffled_rates:
        threshold = (statistics.median(clean_rates)
                     + statistics.median(shuffled_rates)) / 2.0
    else:
        threshold = 0.0

    return pair_counts, threshold


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # Token count for each document in the pool.
    tok_counts = _token_counts(docs)

    # Calculate target word frequencies (TF-like weighting).
    target_word_freqs = Counter()
    for t in target_sample:
        words = _WORD_RE.findall(t.lower())
        target_word_freqs.update(words)

    # Score each document:
    # Sum of frequencies of shared words normalized by unique word count.
    scores = []
    for d in docs:
        w_set = _words(d)
        if not w_set:
            scores.append(0.0)
            continue

        shared_score = 0.0
        for w in w_set:
            shared_score += target_word_freqs.get(w, 0)

        scores.append(shared_score / len(w_set))

    # Word-order check calibrated on the target sample.
    pair_counts, threshold = _calibrate(target_sample, seed)

    # Tier assignment: 0 = read first, 1 = demoted.
    tiers = []
    seen = set()
    for d in docs:
        ws = d.split()
        tier = 0

        # (a) Exact duplicate of an earlier doc (whitespace-normalised).
        h = hashlib.blake2b(" ".join(ws).encode("utf-8", "surrogatepass"),
                            digest_size=16).digest()
        if h in seen:
            tier = 1
        else:
            seen.add(h)

        # (b) Word-order check.
        if tier == 0:
            if len(ws) < _MIN_WORDS:
                tier = 1
            else:
                head = ws[:_MAX_WORDS]
                n = len(head) - 1
                hits = 0
                for p in zip(head, head[1:]):
                    if p in pair_counts:
                        hits += 1
                if n <= 0 or (hits / n) < threshold:
                    tier = 1

        tiers.append(tier)

    # Seeded random tie-breaker, then sort: tier first, then highest score first.
    tiebreak = [rng.random() for _ in docs]
    order = sorted(range(len(docs)),
                   key=lambda i: (tiers[i], -scores[i], tiebreak[i]))

    # Keep the most relevant documents up to the T-token budget.
    selected = []
    total = 0
    for i in order:
        if tok_counts[i] <= 0:
            continue
        selected.append(i)
        total += tok_counts[i]
        if total >= T:
            break

    return selected if selected else order
