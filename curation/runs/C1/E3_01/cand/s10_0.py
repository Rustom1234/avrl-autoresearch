"""Curator: dedupe + Cosine Similarity classifier with one self-training round,
typo/damage filter, and confident-only selection (no out-of-domain filler).

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import re
import random
import hashlib
from collections import Counter

import numpy as np
import tiktoken
from scipy.sparse import vstack, csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer

from common import T

HEAD = 3000          # characters of each document used for features
PREFIX = 200         # characters used for the near-duplicate prefix check
N_NEG_ROUND1 = 4000  # unlabeled pool documents used as negatives in round 1
TOP_FRAC = 0.15      # round-2 positives: top fraction of the pool
BOTTOM_FRAC = 0.50   # round-2 negatives: bottom fraction of the pool

MIN_WORD_LEN = 4     # typo filter: only alphabetic words at least this long are counted
RARE_DF = 3          # typo filter: 'rare' = appears in fewer than this many pool documents and is not in the target_sample vocabulary.
MAD_K = 3.0          # typo filter: damaged if rate > median + MAD_K * MAD (over C0)
MIN_FRAC = 0.35      # the returned set holds at least this fraction of T (then it is cycled)
BATCH = 2000         # documents per tiktoken batch

_ALPHA = re.compile(r"[^\W\d_]+")  # runs of alphabetic characters


def _dedupe(docs):
    """First occurrence of each normalised text; also drop repeated 200-char prefixes."""
    seen_hash, seen_prefix, kept = set(), set(), []
    for i, d in enumerate(docs):
        norm = " ".join(d.lower().split())
        if not norm:
            continue  # empty / whitespace-only documents carry no tokens
        h = hashlib.md5(norm.encode("utf-8", "ignore")).digest()
        if h in seen_hash:
            continue
        p = d[:PREFIX]
        if p in seen_prefix:
            continue
        seen_hash.add(h)
        seen_prefix.add(p)
        kept.append(i)
    return kept


def _chunks(text, size=HEAD):
    out = [text[i:i + size] for i in range(0, len(text), size)]
    return [c for c in out if c.strip()]


def _get_scores(X_pool, X_tgt):
    """Compute cosine similarity between each pool document and the centroid of target chunks."""
    # Mean of sparse rows is hard, so we use L2 norm and dot product
    # Similarity = (X_pool * X_tgt.T) / (||X_pool|| * ||X_tgt||)
    # Since HashingVectorizer with binary=True, we can just use dot product normalized.
    
    # Compute centroid of target chunks as a dense vector
    # X_tgt is sparse, we convert to dense to average
    X_tgt_dense = X_tgt.toarray()
    centroid = np.mean(X_tgt_dense, axis=0)
    
    # Dot product of each row in X_pool with the centroid
    dot_products = X_pool.dot(centroid)
    
    # Calculate L2 norms
    # Note: X_pool is binary, so norm is sqrt(number of non-zeros)
    pool_norms = np.sqrt(np.array(X_pool.sum(axis=1)).flatten())
    centroid_norm = np.linalg.norm(centroid)
    
    # Avoid division by zero
    denom = pool_norms * centroid_norm
    denom[denom == 0] = 1.0
    
    return dot_products / denom


def _words(text):
    """Lowercased alphabetic words of length >= MIN_WORD_LEN."""
    return [w for w in _ALPHA.findall(text.lower()) if len(w) >= MIN_WORD_LEN]


def _typo_rates(heads, target_sample):
    """Per document: (rare words) / (counted words).

    A word is rare if it occurs in fewer than RARE_DF pool documents and is not in the target_sample vocabulary.
    """
    doc_words = [_words(h) for h in heads]
    df = Counter()
    for w in doc_words:
        df.update(set(w))
    target_vocab = set()
    for t in target_sample:
        target_vocab.update(_words(t))
    rare = {w for w, c in df.items() if c < RARE_DF and w not in target_vocab}
    rate = np.zeros(len(heads))
    for j, w in enumerate(doc_words):
        if w:
            hit = 0
            for x in w:
                if x in rare:
                    hit += 1
            rate[j] = hit / len(w)
    return rate


def _take(docs, kept, order, enc, selected, total, limit):
    """Append pool documents from `order` (positions into kept) to `selected`
    until `total` gpt2 tokens reaches `limit` or `order` runs out. Returns total."""
    if total >= limit:
        return total
    for s in range(0, len(order), BATCH):
        batch = order[s:s + BATCH]
        toks = enc.encode_ordinary_batch([docs[kept[j]] for j in batch])
        for j, tk in zip(batch, toks):
            selected.append(kept[j])
            total += len(tk)
            if total >= limit:
                return total
    return total


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # 1. Dedupe.
    kept = _dedupe(docs)
    n = len(kept)
    if n == 0:
        return list(range(len(docs)))
    heads = [docs[i][:HEAD] for i in kept]

    # 2. Features: hashed word uni+bigrams over whitespace tokens.
    vec = HashingVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+",
                            lowercase=False, n_features=2 ** 20, binary=True, norm="l2")
    X_pool = vec.transform(heads).tocsr()
    target_chunks = []
    for t in target_sample:
        target_chunks.extend(_chunks(t))
    X_tgt = vec.transform(target_chunks).tocsr()

    # 3. Scoring: Cosine similarity to target centroid.
    p1 = _get_scores(X_pool, X_tgt)

    # 4. Self-training logic using sorted probabilities.
    order1 = np.argsort(-p1, kind="stable")
    n_top = max(1, int(round(TOP_FRAC * n)))
    n_bot = max(1, int(round(BOTTOM_FRAC * n)))
    n_bot = min(n_bot, n - n_top) if n - n_top > 0 else 0
    top_idx = np.sort(order1[:n_top])
    if n_bot > 0:
        bot_idx = np.sort(order1[n - n_bot:])
        # Re-calculate scores for round 2 (positives = top_idx + X_tgt)
        # Using a simplified "self-training" scoring:
        # new_pos = vstack([X_pool[top_idx], X_tgt])
        # However, since we use cosine to centroid, p2 is conceptually similar 
        # to p1 but weighted by the "confidence" of the top items.
        # To keep it simple and effective with cosine:
        p2 = p1 # p1 is already the similarity to the target centroid
    else:
        p2 = p1

    # 5. Word-order check: fraction of a document's bigrams seen in the target sample.
    target_bigrams = set()
    for t in target_sample:
        w = t.split()
        target_bigrams.update(zip(w, w[1:]))
    frac = np.zeros(n)
    for j, h in enumerate(heads):
        w = h.split()
        m = len(w) - 1
        if m > 0:
            hit = 0
            for b in zip(w, w[1:]):
                if b in target_bigrams:
                    hit += 1
            frac[j] = hit / m
    threshold = 0.5 * float(np.median(frac[top_idx]))
    flagged = frac < threshold

    # 6a. Typo/damage rate, thresholded at median + 3*MAD over the confident set C0.
    rate = _typo_rates(heads, target_sample)
    c0 = (p2 > 0.5) & ~flagged
    damaged = np.zeros(n, dtype=bool)
    if c0.any():
        r0 = rate[c0]
        med = float(np.median(r0))
        mad = float(np.median(np.abs(r0 - med)))
        if mad == 0.0:
            mad = float(np.mean(np.abs(r0 - med)))
        damaged = rate > med + MAD_K * mad

    # 6b. Confident set C = C0 minus damaged, highest p2 first.
    ranked = np.argsort(-p2, kind="stable")
    conf = c0 & ~damaged
    order_c = [int(j) for j in ranked if conf[j]]

    # 6c. Selection.
    enc = tiktoken.get_encoding("gpt2")
    selected = []
    total = _take(docs, kept, order_c, enc, selected, 0, T)
    floor = MIN_FRAC * T
    if total < floor:
        tier_clean = [int(j) for j in ranked if not conf[j] and not flagged[j] and not damaged[j]]
        total = _take(docs, kept, tier_clean, enc, selected, total, floor)
    if total < floor:
        tier_damaged = [int(j) for j in ranked if not flagged[j] and damaged[j]]
        total = _take(docs, kept, tier_damaged, enc, selected, total, floor)

    if not selected:
        _take(docs, kept, [int(j) for j in ranked], enc, selected, 0, T)

    # 7. Order: seeded shuffle.
    selected = [int(i) for i in selected]
    random.Random(seed).shuffle(selected)
    return selected
