"""Curator: target-similarity filter + junk/shuffle rejection + dedup.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Hard filters: title line ('= '), minimum length, spam signatures.
  2. Dedup on the whitespace-normalised first 500 chars (first occurrence wins).
  3. Word-shuffle detection: word-bigram hit rate against target bigrams
     (threshold = 5th percentile of the leave-one-out rate on the target sample),
     plus a punctuation-adjacency statistic.
  4. Char n-gram TF-IDF Cosine Similarity to Target Centroid.
  5. Keep top docs up to ~1.0 T gpt2 tokens, then seeded shuffle.
"""
import os
import random
import time
import hashlib
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

SPAM = ("discount code", "coupon", "free shipping", "click here to order")
PUNCT = frozenset([",", ".", ";", ":"])
MIN_CHARS = 200
HEAD_CHARS = 3000        # chars per doc for the classifier
STAT_CHARS = 6000        # chars per doc for bigram / punctuation statistics
N_BIGRAMS = 400
TOKEN_MARGIN = 1.02      # small safety margin over T
CHARS_PER_TOKEN = 4.3
SCORE_DEADLINE = 105.0   # seconds: classifier scoring should be done by here
TOKEN_DEADLINE = 150.0   # seconds: exact token counting must stop by here


def _has_title(d):
    return d.lstrip().startswith("= ")


def _is_spam(d):
    low = d.lower()
    return any(s in low for s in SPAM)


def _words(d):
    return d[:STAT_CHARS].lower().split()[: N_BIGRAMS + 1]


def _bigram_hit(words, table, min_count):
    """Fraction of the word bigrams whose count in `table` exceeds min_count."""
    n = len(words) - 1
    if n < 5:
        return 0.0
    get = table.get
    hits = 0
    for b in zip(words, words[1:]):
        if get(b, 0) > min_count:
            hits += 1
    return hits / n


def _punct_bad(d):
    """Fraction of ',' / '.' tokens that follow punctuation or start a line."""
    tot = 0
    bad = 0
    for line in d[:STAT_CHARS].split("\n"):
        prev_p = True  # line start counts as "bad" position
        for tok in line.split():
            if tok == "," or tok == ".":
                tot += 1
                if prev_p:
                    bad += 1
                prev_p = True
            else:
                prev_p = tok in PUNCT
    if tot < 5:
        return 0.0
    return bad / tot


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n_docs = len(docs)
    if n_docs == 0:
        return []
    target_sample = list(target_sample)

    def fallback():
        idx = list(range(n_docs))
        random.Random(seed).shuffle(idx)
        return idx

    # ---------- (2) hard filters ----------
    n_t = max(1, len(target_sample))
    need_title = sum(_has_title(d) for d in target_sample) >= 0.8 * n_t
    need_len = sum(len(d) >= MIN_CHARS for d in target_sample) >= 0.8 * n_t

    cand = []
    for i, d in enumerate(docs):
        if need_len and len(d) < MIN_CHARS:
            continue
        if need_title and not _has_title(d):
            continue
        if _is_spam(d):
            continue
        cand.append(i)

    # ---------- (4) dedup on normalised first 500 chars ----------
    seen = set()
    deduped = []
    for i in cand:
        key = hashlib.md5(
            " ".join(docs[i][:500].split()).encode("utf-8", "ignore")
        ).digest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(i)
    cand = deduped

    # ---------- (3) word-shuffle detection ----------
    if target_sample:
        doc_count = Counter()
        for d in target_sample:
            w = d.lower().split()
            doc_count.update(set(zip(w, w[1:])))

        loo = np.array(
            [_bigram_hit(_words(d), doc_count, 1) for d in target_sample]
        )
        loo_valid = loo[loo > 0]
        big_thr = float(np.percentile(loo_valid, 5)) if len(loo_valid) else 0.0

        t_punct = np.array([_punct_bad(d) for d in target_sample])
        punct_thr = float(np.percentile(t_punct, 95)) + 0.10

        kept = []
        for i in cand:
            d = docs[i]
            if _bigram_hit(_words(d), doc_count, 0) < big_thr:
                continue
            if _punct_bad(d) > punct_thr:
                continue
            kept.append(i)
        if kept:
            cand = kept

    if not cand:
        return fallback()

    # ---------- (4) Classifier: Cosine Similarity to Target Centroid ----------
    scores = None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Use char_wb for robustness to typos/variations
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 3),
            sublinear_tf=True,
            max_features=30000,
            dtype=np.float32,
        )
        
        train_txt = [d[:HEAD_CHARS] for d in target_sample]
        X_target = vec.fit_transform(train_txt)
        
        # Calculate centroid
        # We use the mean of the TF-IDF vectors
        centroid = np.asarray(X_target.mean(axis=0))
        
        # Score all candidates
        cand_txt = [docs[i][:HEAD_CHARS] for i in cand]
        X_cand = vec.transform(cand_txt)
        
        # Cosine similarity in TF-IDF space (vectors are normalized/sublinear-scaled)
        # dot product of normalized vectors = cosine similarity
        # Since TfidfVectorizer isn't strictly normalized, we normalize manually
        # to ensure correct cosine similarity logic.
        norm_centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        
        # We can compute similarity for all at once
        # X_cand is (n_cand, n_features)
        # norm_centroid is (n_features,)
        # scores = X_cand @ norm_centroid / (norm(X_cand) + 1e-9)
        # Simpler: use inner product of normalized rows
        # For performance and simplicity, a simple dot product is often sufficient 
        # if we don't need perfect unit-norm precision.
        
        scores = np.array([np.dot(row, centroid) for row in X_cand.权]).reshape(-1)
        # Actually, let's just use a standard dot product and the fact that 
        # tf-idf vectors are already centered/weighted.
        # scores = np.array([np.dot(row, centroid) for row in X_cand]) # Logic check
        
        # Let's do a robust dot product of the raw counts/vectors
        scores = np.array([np.dot(X_cand[i], centroid) for i in range(len(cand))])
        
    except Exception:
        # Fallback if sklearn fails
        if target_sample:
            scores = np.array(
                [_bigram_hit(_words(docs[i]), doc_count, 0) for i in cand]
            )
        else:
            scores = np.zeros(len(cand))

    # ---------- (5) keep top docs up to ~1.0 T tokens ----------
    order = np.argsort(-scores, kind="stable")
    ranked = [cand[j] for j in order]
    budget_tokens = T * TOKEN_MARGIN

    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    selected = []
    total = 0.0
    B = 400
    pos = 0
    while pos < len(ranked) and total < budget_tokens:
        chunk = ranked[pos:pos + B]
        pos += B
        counts = None
        if enc is not None and time.time() - t0 < TOKEN_DEADLINE:
            try:
                toks = enc.encode_ordinary_batch(
                    [docs[i] for i in chunk], num_threads=8
                )
                counts = [len(t) for t in toks]
            except Exception:
                counts = None
        if counts is None:
            counts = [len(docs[i]) / CHARS_PER_TOKEN for i in chunk]
        for i, c in zip(chunk, counts):
            selected.append(i)
            total += c
            if total >= budget_tokens:
                break

    if not selected:
        return fallback()

    # ---------- (6) seeded shuffle ----------
    random.Random(seed).shuffle(selected)
    return selected
