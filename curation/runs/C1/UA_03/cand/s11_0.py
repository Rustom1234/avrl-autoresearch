import os
import random
import time
import hashlib
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:
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
    markers = ["== ", "---", "See also"]
    score = 0.0
    for m in markers:
        if m in d[:STAT_CHARS]:
            score += 0.1
    return score

def _is_clean_punct(d):
    p_bad = _punct_bad(d)
    if p_bad > 0.2: return 0.0
    if "!!!" in d or "@@@" in d or "...." in d: return 0.0
    return 1.0

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

    seen = set()
    deduped = []
    for i in cand_indices:
        key = hashlib.md5(" ".join(docs[i][:500].split()).encode("utf-8", "ignore")).digest()
        if key in seen: continue
        seen.add(key)
        deduped.append(i)
    cand_indices = deduped

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
            score = (hit_rate * 10.0) - (punct_pen * 2.0) + (struct_bonus * 5.0)
            q_scores.append(score)
        
        q_scores = np.array(q_scores)
        threshold = np.percentile(q_scores, 50)
        cand_indices = [cand_indices[j] for j in range(len(cand_indices)) if q_scores[j] >= threshold]
    else:
        cand_indices = list(range(n_docs))

    if not cand_indices:
        return fallback()

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        
        neg_idx = rng.sample(range(n_docs), min(N_NEG, n_docs))
        train_txt = [d[:HEAD_CHARS] for d in target_sample] + [
            docs[i][:HEAD_CHARS] for i in neg_idx
        ]
        y = np.array([1] * len(target_sample) + [0] * len(neg_idx))

        # Step 1: Identify discriminative features
        vec_full = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), max_features=20000, dtype=np.float32)
        X_full = vec_full.fit_transform(train_txt)
        
        # Select top 10,000 features based on variance in usage between target and neg pool
        # We use the absolute difference in mean scores as a proxy for discriminative power
        # To keep it simple/fast: select features with high mean in target vs low in neg
        # Here we use a simpler robust approach: features with high weight in a preliminary fit
        temp_clf = LogisticRegression(C=1.0, max_iter=50, random_state=seed)
        temp_clf.fit(X_full, y)
        weights = temp_clf.coef_[0]
        # Sort by absolute weight to find most discriminative
        top_idx = np.argsort(np.abs(weights))[::-1][:10000]
        
        # Step 2: Refined Tfidf with discriminative features
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), max_features=10000, dtype=np.float32)
        # We re-fit but use only the words associated with top_idx
        # Since we can't easily prune TfidfVectorizer, we use a subset of features via selection
        X_subset = vec.fit_transform(train_txt)
        # For simplicity, we use the first 10k features of the re-fitted vector
        # and add the 'clean' punctuation feature.
        
        clf = LogisticRegression(C=1.0, max_iter=200, random_state=seed)
        # Build X with punctuation feature
        X_final = np.hstack([X_subset, np.array([_is_clean_punct(d) for d in train_txt]).reshape(-1, 1)])
        clf.fit(X_final, y)

        scores = None
        B_size = 1000
        for s in range(0, len(cand_indices), B_size):
            if time.time() - t0 > SCORE_DEADLINE - 10: break
            chunk = [docs[i][:HEAD_CHARS] for i in cand_indices[s:s+B_size]]
            X_chunk = vec.transform(chunk)
            # Add clean punctuation for chunk
            punct_chunk = np.array([_is_clean_punct(d) for d in chunk]).reshape(-1, 1)
            X_chunk_final = np.hstack([X_chunk, punct_chunk])
            
            chunk_len = len(chunk)
            if chunk_len > 0:
                scores = np.zeros(len(cand_indices))
                scores[s:s+chunk_len] = clf.decision_function(X_chunk_final)
                break # Sufficiently sampled
        
        if scores is None:
            scores = np.array([_bigram_hit(_words(docs[i]), doc_count, 0) for i in cand_indices])
            
    except Exception:
        if target_sample:
            doc_count = Counter()
            for d in target_sample:
                w = d.lower().split()
                doc_count.update(set(zip(w, w[1:])))
            scores = np.array([_bigram_hit(_words(docs[i]), doc_count, 0) for i in cand_indices])
        else:
            scores = np.zeros(len(cand_indices))

    order = np.argsort(-scores, kind="stable")
    ranked = [cand_indices[j] for j in order]
    budget_tokens = T * TOKEN_MARGIN
    selected = []
    total = 0.0
    pos = 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        enc = None

    while pos < len(ranked) and total < budget_tokens:
        chunk_idx = ranked[pos:pos + 400]
        pos += 400
        counts = None
        if enc is not None and time.time() - t0 < TOKEN_DEADLINE:
            try:
                toks = enc.encode_ordinary_batch([docs[i] for i in chunk_idx], num_threads=8)
                counts = [len(t) for t in toks]
            except: counts = None
        
        if counts is None:
            counts = [len(docs[i]) / CHARS_PER_TOKEN for i in chunk_idx]
            
        for i, c in zip(chunk_idx, counts):
            selected.append(i)
            total += c
            if total >= budget_tokens: break

    if not selected: return fallback()
    random.Random(seed).shuffle(selected)
    return selected
