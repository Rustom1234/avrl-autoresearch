"""Curator: Two-stage ranking (Quality Score filter + Logistic Regression).

Pipeline:
  1. Hard filters: title line, min length, spam.
  2. Dedup on normalized first 500 chars.
  3. Quality Score filter: weighted sum of bigram hits, punctuation penalty, 
     and structural markers (top 50%).
  4. Logistic Regression ranking on the survivors.
  5. Token-budgeted selection and seeded shuffle.
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
HEAD_CHARS = 3000
STAT_CHARS = 6000
N_BIGRAMS = 400
N_NEG = 3000
TOKEN_MARGIN = 1.02
CHARS_PER_TOKEN = 4.3
SCORE_DEADLINE = 105.0
TOKEN_DEADLINE = 150.0

def _has_title(d):
    return d.lstrip().startswith("= ")

def _is_spam(d):
    low = d.lower()
    return any(s in low for s in SPAM)

def _words(d):
    return d[:STAT_CHARS].lower().split()[: N_BIGRAMS + 1]

def _bigram_hit(words, table, min_count):
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
    tot = 0
    bad = 0
    for line in d[:STAT_CHARS].split("\n"):
        prev_p = True
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

def _structural_bonus(d):
    # Markers like Wikipedia sections, horizontal rules, or see also
    markers = ["== ", "---", "See also"]
    score = 0.0
    for m in markers:
        if m in d[:STAT_CHARS]:
            score += 0.1
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

    # ---------- (1) Hard filters ----------
    n_t = max(1, len(target_sample))
    need_title = sum(_has_title(d) for d in target_sample) >= 0.8 * n_t
    need_len = sum(len(d) >= MIN_CHARS for d in target_sample) >= 0.8 * n_t

    cand_indices = []
    for i, d in enumerate(docs):
        if need_len and len(d) < MIN_CHARS:
            continue
        if need_title and not _has_title(d):
            continue
        if _is_spam(d):
            continue
        cand_indices.append(i)

    # ---------- (2) Dedup ----------
    seen = set()
    deduped = []
    for i in cand_indices:
        key = hashlib.md5(
            " ".join(docs[i][:500].split()).encode("utf-8", "ignore")
        ).digest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(i)
    cand_indices = deduped

    # ---------- (3) Quality Score Ranking ----------
    if target_sample:
        doc_count = Counter()
        for d in target_sample:
            w = d.lower().split()
            doc_count.update(set(zip(w, w[1:])))

        q_scores = []
        for i in cand_indices:
            d = docs[i]
            hit_rate = _bigram_hit(_words(d), doc_count, 0)
            punct_pen = _punct_bad(d)
            struct_bonus = _structural_bonus(d)
            # Weighted Score: Hits are primary, penalized by punctuation, boosted by structure
            score = (hit_rate * 10.0) - (punct_pen * 2.0) + (struct_bonus * 5.0)
            q_scores.append(score)
        
        q_scores = np.array(q_scores)
        threshold = np.percentile(q_scores, 50)
        # Filter to top 50% of quality scores
        mask = q_scores >= threshold
        cand_indices = [cand_indices[j] for j in range(len(cand_indices)) if mask[j]]
    else:
        cand_indices = list(range(len(docs)))

    if not cand_indices:
        return fallback()

    # ---------- (4) Logistic Regression Ranking ----------
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
        X = vec.fit_transform(train_txt)
        clf = LogisticRegression(C=1.0, max_iter=200, random_state=seed)
        clf.fit(X, y)

        budget = max(8.0, SCORE_DEADLINE - (time.time() - t0))
        # Simple heuristic for rate: assume 100 chars per sec if unknown
        rate = sum(len(s) for s in train_txt) / max(1, time.time() - t0)
        window = int(rate * budget * 0.8 / max(1, len(cand_indices)))
        window = max(300, min(HEAD_CHARS, window))

        scores = np.full(len(cand_indices), -1e9, dtype=np.float64)
        B = 2000
        for s in range(0, len(cand_indices), B):
            if time.time() - t0 > SCORE_DEADLINE + 25:
                break
            chunk = [docs[i][:window] for i in cand_indices[s:s + B]]
            scores[s:s + B] = clf.decision_function(vec.transform(chunk))
    except Exception:
        if target_sample:
            doc_count = Counter()
            for d in target_sample:
                w = d.lower().split()
                doc_count.update(set(zip(w, w[1:])))
            scores = np.array([_bigram_hit(_words(docs[i]), doc_count, 0) for i in cand_indices])
        else:
            scores = np.zeros(len(cand_indices))

    # ---------- (5) Selection ----------
    order = np.argsort(-scores, kind="stable")
    ranked = [cand_indices[j] for j in order]
    budget_tokens = T * TOKEN_MARGIN

    selected = []
    total = 0.0
    B = 400
    pos = 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        enc = None

    while pos < len(ranked) and total < budget_tokens:
        chunk = ranked[pos:pos + B]
        pos += B
        counts = None
        if enc is not None and time.time() - t0 < TOKEN_DEADLINE:
            try:
                toks = enc.encode_ordinary_batch([docs[i] for i in chunk], num_threads=8)
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
