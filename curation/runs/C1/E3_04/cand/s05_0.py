"""Curator: order-sensitive in-domain filter, then a seeded shuffle.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

1. Drop exact duplicates (normalised first 500 chars + length).
2. Score every doc against the target sample:
     u = fraction of words in the target unigram set,
     b = fraction of adjacent word bigrams in the target bigram set,
     r = distinct word trigrams / total trigrams.
3. Thresholds tu, tb = 5th percentile of leave-one-out u, b over the target docs.
4. Tier A = clean in-domain docs (with a relaxed bigram bar for WikiText-marked docs).
5. Tier B = clean general-English filler, sorted by b descending.
6. All of Tier A once, then Tier B up to ~1.02*T tokens, then repeats of Tier A.
7. Shuffle the final list with random.Random(CURATE_SEED).
"""
import os
import random
from collections import Counter

import numpy as np

from common import T

MARKERS = (" @-@ ", " @,@ ", " , ", " . ")
R_MIN = 0.5
CHUNK = 2000

def _words(text):
    return text.lower().split()

def _has_markers(doc):
    if doc.lstrip().startswith("= "):
        return True
    for m in MARKERS:
        if m in doc:
            return True
    return False

def _token_counter():
    """Returns a function mapping a list of strings to a list of token counts."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        threads = max(1, min(8, os.cpu_count() or 1))

        def count(texts):
            if not texts:
                return []
            return [len(x) for x in enc.encode_ordinary_batch(texts, num_threads=threads)]

        count(["warm up"])
        return count
    except Exception:
        def count(texts):
            return [max(1, len(t) // 4) for t in texts]
        return count

def _calibrate(target_sample):
    """Build V, B and the leave-one-out 5th-percentile thresholds tu, tb."""
    t_words = [_words(t) for t in target_sample]
    t_bigrams = [list(zip(w, w[1:])) for w in t_words]

    df_u = Counter()
    df_b = Counter()
    for w, bg in zip(t_words, t_bigrams):
        df_u.update(set(w))
        df_b.update(set(bg))

    V = set(df_u)
    B = set(df_b)

    loo_u, loo_b = [], []
    for w, bg in zip(t_words, t_bigrams):
        if w:
            loo_u.append(sum(1 for x in w if df_u[x] >= 2) / len(w))
        if bg:
            loo_b.append(sum(1 for x in bg if df_b[x] >= 2) / len(bg))

    tu = float(np.percentile(loo_u, 5)) if loo_u else 0.0
    tb = float(np.percentile(loo_b, 5)) if loo_b else 0.0
    return V, B, tu, tb

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    V, B, tu, tb = _calibrate(target_sample)
    in_V = V.__contains__
    in_B = B.__contains__

    seen = set()
    kept = []
    tier_a = []
    tier_b = []

    tu_lo = 0.8 * tu
    tb_marked = 0.8 * tb
    tb_lo = 0.5 * tb

    for i, doc in enumerate(docs):
        w = _words(doc)
        key = (" ".join(w[:500])[:500], len(doc))
        if key in seen:
            continue
        seen.add(key)
        kept.append(i)

        n = len(w)
        if n < 2:
            continue

        u = sum(map(in_V, w)) / n
        if u < tu_lo:
            continue

        b = sum(map(in_B, zip(w, w[1:]))) / (n - 1)
        if b < tb_lo:
            continue

        if n >= 3:
            r = len(set(zip(w, w[1:], w[2:]))) / (n - 2)
        else:
            r = 1.0
        
        # Diversity penalty: 1 - (unique_unigrams / total_unigrams)
        rep_penalty = 1.0 - (len(set(w)) / n)
        
        # Threshold check with diversity penalty
        if u >= tu and (b >= tb or (b >= tb_marked and _has_markers(doc))):
            # We apply a more stringent r check that accounts for internal repetition
            if r > R_MIN + rep_penalty:
                tier_a.append(i)
        else:
            tier_b.append((b, i))

    tier_b.sort(key=lambda x: (-x[0], x[1]))
    tier_b = [i for _, i in tier_b]

    if not tier_a and not tier_b:
        out = kept if kept else list(range(len(docs)))
        out = list(out)
        rng.shuffle(out)
        return out

    count_func = _token_counter()
    budget = 1.02 * T

    selected = list(tier_a)
    a_tokens = []
    for s in range(0, len(tier_a), CHUNK):
        a_tokens.extend(count_func([docs[i] for i in tier_a[s:s + CHUNK]]))
    total = sum(a_tokens)

    if total < T:
        done = False
        for s in range(0, len(tier_b), CHUNK):
            chunk = tier_b[s:s + CHUNK]
            for i, n_tok in zip(chunk, count_func([docs[i] for i in chunk])):
                selected.append(i)
                total += n_tok
                if total >= budget:
                    done = True
                    break
            if done:
                break

    if total < T and tier_a and sum(a_tokens) > 0:
        order = list(range(len(tier_a)))
        rep_rng = random.Random(seed + 1)
        while total < budget:
            rep_rng.shuffle(order)
            for j in order:
                selected.append(tier_a[j])
                total += a_tokens[j]
                if total >= budget:
                    break

    rng.shuffle(selected)
    return selected
