"""Curator: Refined quality scoring + LogisticRegression."""
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
HEAD_CHARS = 3000
STAT_CHARS = 4000
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
            if tok in ("," , "."):
                tot += 1
                if prev_p:
                    bad += 1
                prev_p = True
            else:
                prev_p = tok in PUNCT
    if tot < 5:
        return 0.0
    return bad / tot

def _quality_score(d, doc_count, punct_thr):
    # 1. Bigram Hit
    words = _words(d)
    score = _bigram_hit(words, doc_count, 0)
    
    # 2. Punctuation Penalty
    pb = _punct_bad(d)
    if pb > punct_thr:
        score -= 0.5 * (pb - punct_thr)
        
    # 3. Structural Bonus
    body = d[:2000]
    markers = ('==', '---', 'See also', 'References', 'Note:', 'Summary')
    for m in markers:
        if m in body:
            score += 0.05
            
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

    # --- Hard Filters ---
    n_t = max(1, len(target_sample))
    need_title = sum(_has_title(d) for d in target_sample) >= 0.8 * n_t
    need_len = sum(len(d) >= MIN_CHARS for d in target_sample) >= 0.8 * n_t

    cand = []
    for i, d in enumerate(docs):
        if need_len and len(d) < MIN_CHARS: continue
        if need_title and not _has_title(d): continue
        if _is_spam(d): continue
        cand.append(i)

    # --- Dedup ---
    seen = set()
    deduped = []
    for i in cand:
        key = hashlib.md5(" ".join(docs[i][:500].split()).encode("utf-8", "ignore")).digest()
        if key not in seen:
            seen.add(key)
            deduped.append(i)
    cand = deduped

    # --- Stats & Quality Scoring ---
    if target_sample:
        doc_count = Counter()
        for d in target_sample:
            w = d.lower().split()
            doc_count.update(set(zip(w, w[1:])))

        loo = np.array([_bigram_hit(_words(d), doc_count, 1) for d in target_sample])
        loo_valid = loo[loo > 0]
        big_thr = float(np.percentile(loo_valid, 5)) if len(loo_valid) else 0.0
        
        t_punct = np.array([_punct_bad(d) for d in target_sample])
        punct_thr = float(np.percentile(t_punct, 95)) + 0.10

        quality_scores = []
        for i in cand:
            # Use big_thr as a baseline filter
            if _bigram_hit(_words(docs[i]), doc_count, 0) < big_thr:
                continue
            # Calculate score
            qs = _quality_score(docs[i], doc_count, punct_thr)
            quality_scores.append((i, qs))
        
        # Filter by 10th percentile of quality scores
        if quality_scores:
            scores_only = [x[1] for x in quality_scores]
            q_threshold = np.percentile(scores_only, 10)
            cand = [x[0] for x in quality_scores if x[1] >= q_threshold]
        else:
            return fallback()
    else:
        cand = deduped

    # --- Classifier ---
    scores = None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        
        neg_idx = rng.sample(range(n_docs), min(N_NEG, n_docs))
        train_txt = [d[:HEAD_CHARS] for d in target_sample] + [docs[i][:HEAD_CHARS] for i in neg_idx]
        y = np.array([1] * len(target_sample) + [0] * len(neg_idx))

        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), sublinear_tf=True, max_features=50000, dtype=np.float32)
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
            if time.time() - t0 > SCORE_DEADLINE + 25: break
            chunk = [docs[i][:window] for i in cand[s:s + B]]
            scores[s:s + B] = clf.decision_function(vec.transform(chunk))
    except Exception:
        if target_sample:
            scores = np.array([_quality_score(docs[i], doc_count, punct_thr) for i in cand])
        else:
            scores = np.zeros(len(cand))

    # --- Ranking & Selection ---
    order = np.argsort(-scores, kind="stable")
    ranked = [cand[j] for j in order]
    budget_tokens = T * TOKEN_MARGIN

    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except:
        enc = None

    selected = []
    total = 0.0
    for i in ranked:
        if enc is not None and time.time() - t0 < TOKEN_DEADLINE:
            try:
                # Sample check: 1 doc at a time is safer for memory/speed in loops
                t_len = len(enc.encode_ordinary([docs[i]])[0])
            except:
                t_len = len(docs[i]) / CHARS_PER_TOKEN
        else:
            t_len = len(docs[i]) / CHARS_PER_TOKEN
            
        if total + t_len > budget_tokens and total > 0:
            break
        selected.append(i)
        total += t_len

    if not selected: return fallback()
    
    random.Random(seed).shuffle(selected)
    return selected
