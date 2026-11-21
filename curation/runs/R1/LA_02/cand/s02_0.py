"""Curator: dedupe, drop obvious junk, rank by target-vs-pool unigram LLR, fill to T tokens.

select(docs, target_sample) -> ordered list of indices into docs.
Pipeline:
  1. exact dedupe (whitespace-normalised, lowercased), keep first occurrence;
  2. hard-drop obvious junk (too short, highly compressible, non-prose, word-shuffled);
  3. score survivors by mean per-token log-likelihood ratio (target vs pool unigram),
     with a +1.0 bonus for WikiText markers;
  4. take top-scored docs until the cumulative gpt2 token count reaches T (no repeats;
     cycling happens only if all survivors together hold fewer than T tokens);
  5. shuffle the selection with CURATE_SEED.
"""
import hashlib
import math
import os
import random
import re
import time
import zlib
from collections import Counter
from itertools import repeat

from common import T

MIN_CHARS = 200
MIN_ZLIB_RATIO = 0.25
MIN_ALPHA_SPACE = 0.60
SHUFFLE_TOKENS = 400
SHUFFLE_REL_THRESHOLD = 0.25
SCORE_TOKENS = 1000
HEADER_BONUS = 1.0
POOL_ALPHA = 0.5          # additive smoothing for the pool unigram model
TARGET_MU_FRAC = 0.05     # target model is smoothed towards the pool (Dirichlet prior)
TOKENISE_CHUNK = 1000
TIME_BUDGET = 150.0       # seconds; after this, token counts are estimated from chars

_HEADER_RE = re.compile(r"^\s*= [^=\n]+ =\s*\n")


def _bigram_stats(text, target_bigrams):
    """Over the first 400 whitespace tokens: (fraction of bigrams seen in the target
    sample, rate of common function bigrams 'of the' / 'in the' / 'to the' / ', and')."""
    toks = text[:8000].lower().split()[:SHUFFLE_TOKENS]
    n = len(toks) - 1
    if n < 1:
        return 0.0, 0.0
    func = 0
    hit = 0
    for a, b in zip(toks, toks[1:]):
        if b == "the":
            if a == "of" or a == "in" or a == "to":
                func += 1
        elif b == "and" and a.endswith(","):
            func += 1
        if (a, b) in target_bigrams:
            hit += 1
    return hit / n, func / n


def _median(values):
    if not values:
        return 0.0
    v = sorted(values)
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


def _alpha_space_fraction(text):
    n = len(text)
    if n == 0:
        return 0.0
    return (sum(map(str.isalpha, text)) + sum(map(str.isspace, text))) / n


def _zlib_ratio(text):
    raw = text.encode("utf-8", "replace")
    if not raw:
        return 0.0
    return len(zlib.compress(raw)) / len(raw)


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    n_docs = len(docs)
    if n_docs == 0:
        return []

    # ---- target statistics --------------------------------------------------
    target_counts = Counter()
    target_bigrams = set()
    target_func_rates = []
    for text in target_sample:
        toks = text.lower().split()
        target_counts.update(toks)
        target_bigrams.update(zip(toks, toks[1:]))
    for text in target_sample:
        if len(text.split()) >= 2:
            target_func_rates.append(_bigram_stats(text, target_bigrams)[1])
    func_threshold = SHUFFLE_REL_THRESHOLD * _median(target_func_rates)

    # ---- pass 1: pool unigram counts (whole pool) + exact dedupe -------------
    pool_counts = Counter()
    seen = set()
    unique = []
    for i, text in enumerate(docs):
        toks = text.lower().split()
        pool_counts.update(toks)
        key = hashlib.blake2b(
            " ".join(toks).encode("utf-8", "replace"), digest_size=16
        ).digest()
        if key in seen:
            continue
        seen.add(key)
        unique.append(i)
    del seen

    # ---- unigram log-likelihood ratio table ----------------------------------
    n_target = sum(target_counts.values())
    n_pool = sum(pool_counts.values())
    vocab = len(pool_counts) + sum(1 for w in target_counts if w not in pool_counts)
    llr = {}
    unseen_llr = 0.0
    if n_target > 0:
        mu = max(1.0, TARGET_MU_FRAC * n_target)
        pool_den = n_pool + POOL_ALPHA * max(vocab, 1)
        tgt_den = n_target + mu
        # Words absent from the target sample: p_t = mu * p_pool / tgt_den.
        unseen_llr = math.log(mu / tgt_den)
        for w, c in target_counts.items():
            p_pool = (pool_counts.get(w, 0) + POOL_ALPHA) / pool_den
            llr[w] = math.log((c + mu * p_pool) / (tgt_den * p_pool))
    del pool_counts

    # ---- pass 2: junk filters + scoring --------------------------------------
    get = llr.get
    scored = []
    for i in unique:
        text = docs[i]
        if len(text) < MIN_CHARS:
            continue
        if _alpha_space_fraction(text) < MIN_ALPHA_SPACE:
            continue
        if _zlib_ratio(text) < MIN_ZLIB_RATIO:
            continue
        # The target-bigram hit fraction is computed for reference; only the
        # function-bigram rate gates the drop.
        _hit_frac, func_rate = _bigram_stats(text, target_bigrams)
        if func_rate < func_threshold:
            continue

        toks = text[:16000].lower().split()[:SCORE_TOKENS]
        if not toks:
            continue
        score = sum(map(get, toks, repeat(unseen_llr))) / len(toks)
        if _HEADER_RE.match(text) or " @-@ " in text:
            score += HEADER_BONUS
        scored.append((-score, i))

    if not scored:
        # Nothing survived: fall back to the deduped pool in a seeded random order.
        idx = list(unique)
        random.Random(seed).shuffle(idx)
        return idx

    scored.sort()
    ranked = [i for _, i in scored]

    # ---- selection: fill from the top until T tokens -------------------------
    enc = None
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    selected = []
    total = 0
    seen_chars = 0
    seen_tokens = 0
    pos = 0
    done = False
    while pos < len(ranked) and not done:
        chunk = ranked[pos:pos + TOKENISE_CHUNK]
        pos += TOKENISE_CHUNK
        counts = None
        if enc is not None and (time.time() - t0) < TIME_BUDGET:
            try:
                counts = [len(x) for x in enc.encode_ordinary_batch([docs[i] for i in chunk])]
                seen_chars += sum(len(docs[i]) for i in chunk)
                seen_tokens += sum(counts)
            except Exception:
                counts = None
        if counts is None:
            ratio = (seen_tokens / seen_chars) if seen_chars > 0 else 0.25
            counts = [max(1, int(len(docs[i]) * ratio)) for i in chunk]
        for i, c in zip(chunk, counts):
            selected.append(i)
            total += c
            if total >= T:
                done = True
                break

    # If all survivors together are under T, `selected` holds every survivor and
    # the trainer cycles it; otherwise it holds >= T tokens with no repeats.
    random.Random(seed).shuffle(selected)
    return selected
