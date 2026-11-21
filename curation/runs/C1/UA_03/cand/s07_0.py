"""Curator: target-similarity filter + Wikipedia-style structural density + dedup.

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
  4. Wikipedia-style structural markers: score based on '==', '---', 'See also', etc.
  5. Char n-gram TF-IDF + logistic regression (target vs. random pool docs).
  6. Ranking: Classifier scores boosted by structural markers.
  7. Keep top docs up to ~1.0 T gpt2 tokens, then seeded shuffle.
"""
import os
import random
import time
import hashlib
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: do cover
    T = 9_830_400

SPAM = ("discount code", "coupon", "free shipping", "click here to order")
PUNCT = frozenset([",", ".", ";", ":"])
MIN_CHARS = 200
HEAD_CHARS = 3000        # chars per doc for the classifier
STAT_CHARS = 6000        # chars per doc for bigram / punctuation statistics
N_BIGRAMS = 400
N_NEG = 3000
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

def _get_struct_score(d):
    """Calculate density of Wikipedia-style structural elements."""
    content = d[:2000]
    score = 0.0
    lines = content.splitlines()
    if not lines:
        return 0.0
    
    markers = 0
    for line in lines:
        l_stripped = line.strip()
        if l_stripped.startswith("==") or l_stripped.startswith("---"):
            markers += 1
        if any(m in l_stripped for m in ["See also", "References", "Categories"]):
            markers += 1
    
    # Density is relative to the number of lines or just a raw count boost
    # Here we use a normalized count to keep it stable
    score = markers / (len(lines) + 1)
    return score

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

    # ---------- (3) dedup ----------
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

    # ---------- (4) word-shuffle detection ----------
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

    # ---------- (5) classifier + structural boosting ----------
    scores = None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        neg_idx = rng.sample(range(n_docs), min(N_NEG, n_docs))
        train_txt = [d[:HEAD_CHARS] for d in target_sample] + [
            docs[i][:HEAD_CHARS] for i in neg_idx
        ]
        y = np.array([1] * len(target_sample) + [0] * len(neg_idx))

        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 3),
            sublinear_tf=True,
            max_features=50000,
            dtype=np.float32,
        )
        tf0 = time.time()
        X = vec.fit_transform(train_txt)
        fit_time = max(time.time() - tf0, 1e-3)
        rate = sum(len(s) for s in train_txt) / fit_time

        clf = LogisticRegression(C=1.0, max_iter=200, random_state=seed)
        clf.fit(X, y)

        budget = max(8.0, SCORE_DEADLINE - (time.time() - t0))
        window = int(rate * budget * 0.8 / max(1, len(cand)))
        window = max(300, min(HEAD_CHARS, window))

        scores = np.full(len(cand), -1e9, dtype=np.float64)
        B = 2000
        for s in range(0, len(cand), B):
            if time.time() - t0 > SCORE_DEADLINE + 25:
                break
            chunk = [docs[i][:window] for i in cand[s:s + B]]
            scores[s:s + B] = clf.decision_function(vec.transform(chunk))
            
        # Structural boosting: add a weight based on Wikipedia-style markers
        # This acts as a secondary feature for the final ranking
        for idx, doc_idx in enumerate(cand):
            scores[idx] += _get_struct_score(docs[doc_idx]) * 2.0

    except Exception:
        if target_sample:
            scores = np.array(
                [_bigram_hit(_words(docs[i]), doc_count, 0) for i in cand]
            )
        else:
            scores = np.zeros(len(cand))

    # ---------- (6) keep top docs ----------
    order = np.argsort(-scores, kind="stable")
    ranked = [cand[j] for j in order]
    budget_tokens = T * TOKEN_MARGIN

    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        pass

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

    random.Random(seed).shuffle(selected)
    return selected
