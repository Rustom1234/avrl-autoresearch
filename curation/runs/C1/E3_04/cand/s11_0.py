"""Curator: WikiText-format priority tier, truncated to T by priority.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
from collections import Counter

import numpy as np

from common import T

MARKERS = (" @-@ ", " @,@ ", " , ", " . ")
PUNCT = frozenset({",", ".", "@-@", "@,@", "@.@", "(", ")", ";", ":", '"', "'s"})
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


def _punct_frac(w):
    n = len(w)
    if n == 0:
        return 0.0
    return sum(map(PUNCT.__contains__, w)) / n


def _token_counter():
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
    t_words = [_words(t) for t in target_sample]
    t_bigrams = [list(zip(w, w[1:])) for w in t_words]
    df_u = Counter()
    df_b = Counter()
    for w, bg in zip(t_words, t_bigrams):
        df_u.update(set(w))
        df_b.update(set(bg))
    V = set(df_u)
    B = set(df_b)
    loo_u, loo_b, ps = [], [], []
    for w, bg in zip(t_words, t_bigrams):
        if w:
            loo_u.append(sum(1 for x in w if df_u[x] >= 2) / len(w))
            ps.append(_punct_frac(w))
        if bg:
            loo_b.append(sum(1 for x in bg if df_b[x] >= 2) / len(bg))
    tu = float(np.percentile(loo_u, 5)) if loo_u else 0.0
    tb = float(np.percentile(loo_b, 5)) if loo_b else 0.0
    tp = float(np.percentile(ps, 5)) if ps else 0.0
    return V, B, tu, tb, tp


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    V, B, tu, tb, tp = _calibrate(target_sample)
    in_V = V.__contains__
    in_B = B.__contains__
    in_P = PUNCT.__contains__

    seen = set()
    kept = []
    tier_w = []
    tier_a = []
    tier_b = []

    # Thresholds
    tu_lo = 0.8 * tu
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

        # Hard gate for repetitive garbage
        if r < R_MIN:
            continue

        p = sum(map(in_P, w)) / n

        b_score = b / (tb if tb > 0 else 1.0)
        u_score = u / (tu if tu > 0 else 1.0)
        combined_score = (u_score + b_score) / 2.0

        if p >= tp and u >= tu and b >= 0.8 * tb:
            # Tier W: target-format, clean
            tier_w.append((combined_score, i))
        elif u >= tu and (b >= tb or (b >= 0.8 * tb and _has_markers(doc))):
            # Tier A': in-domain vocabulary but not in target format
            tier_a.append((combined_score, i))
        else:
            # Tier B: lower-confidence candidates
            tier_b.append((b, i))

    tier_w.sort(key=lambda x: (-x[0], x[1]))
    tier_w = [x[1] for x in tier_w]
    tier_a.sort(key=lambda x: (-x[0], x[1]))
    tier_a = [x[1] for x in tier_a]
    tier_b.sort(key=lambda x: (-x[0], x[1]))
    tier_b = [x[1] for x in tier_b]

    if not tier_w and not tier_a and not tier_b:
        out = kept if kept else list(range(len(docs)))
        rng.shuffle(out)
        return out

    count_fn = _token_counter()
    budget = 1.0 * T
    selected = []
    sel_tokens = []
    total_tokens = 0

    def take(tier):
        # Append docs from tier in priority order until the budget is reached.
        nonlocal total_tokens
        for s in range(0, len(tier), CHUNK):
            if total_tokens >= budget:
                return
            chunk = tier[s:s + CHUNK]
            counts = count_fn([docs[i] for i in chunk])
            for i, c in zip(chunk, counts):
                selected.append(i)
                sel_tokens.append(c)
                total_tokens += c
                if total_tokens >= budget:
                    return

    # Tier W first (truncated to the best-scored docs if it exceeds T)
    take(tier_w)
    n_w = len(selected)

    # Then Tier A' by score, then Tier B by b
    if total_tokens < budget:
        take(tier_a)
    n_wa = len(selected)
    if total_tokens < budget:
        take(tier_b)

    # Only if everything together falls short: repeat Tier W docs
    if total_tokens < budget:
        if n_w > 0:
            pool = list(range(0, n_w))
        elif n_wa > 0:
            pool = list(range(0, n_wa))
        else:
            pool = list(range(len(selected)))
        pool_tokens = sum(sel_tokens[j] for j in pool)
        if pool and pool_tokens > 0:
            rep_rng = random.Random(seed + 1)
            order = list(pool)
            while total_tokens < budget:
                rep_rng.shuffle(order)
                for j in order:
                    selected.append(selected[j])
                    total_tokens += sel_tokens[j]
                    if total_tokens >= budget:
                        break

    rng.shuffle(selected)
    return selected
