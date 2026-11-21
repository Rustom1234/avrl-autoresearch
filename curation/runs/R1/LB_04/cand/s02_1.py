"""Curator: target-similarity filter + dedup + junk removal.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Char n-gram TF-IDF + logistic regression (target vs. random pool subsample),
     followed by one round of self-training on the pool.
  2. Drop exact / near duplicates (hash of normalized first 500 chars).
  3. Drop junk (too short, low alphabetic fraction, low in-target-vocabulary fraction).
  4. Filter for high confidence (score >= 0.5), with a safety floor of 0.25*T.
  5. Construct output: if kept < T, cycle with varied shuffles; else top-scored up to T.
"""
import os
import random
import re
import hashlib
from collections import Counter
import numpy as np
from common import T

HEAD_CHARS = 3000          # chars of each doc used for featurization
POOL_CHAR_BUDGET = 60_000_000  # deterministic cap on total chars vectorized (time safety)
NEG_SUBSAMPLE = 4000
MAX_SELF_POS = 6000        # caps on self-training set size (time safety)
MAX_SELF_NEG = 12000
TOKEN_BUDGET = 1.0 * T

_WORD_RE = re.compile(r"[^\W\d_]+")
_ALPHA_SPACE_RE = re.compile(r"[^\W\d_]|\s")
_WS_RE = re.compile(r"\s+")

def _seed():
    try:
        return int(os.environ.get("CURATE_SEED", 0))
    except ValueError:
        return 0

def _score(docs, target_sample, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    n = len(docs)
    rng = random.Random(seed)

    cap = int(max(600, min(HEAD_CHARS, POOL_CHAR_BUDGET // max(n, 1))))

    tgt_heads = [t[:HEAD_CHARS] for t in target_sample]
    sub_idx = rng.sample(range(n), min(NEG_SUBSAMPLE, n))
    sub_heads = [docs[i][:HEAD_CHARS] for i in sub_idx]

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 4),
        sublinear_tf=True,
        max_features=200_000,
        dtype=np.float32,
    )
    X_fit = vec.fit_transform(tgt_heads + sub_heads)
    y_fit = np.array([1] * len(tgt_heads) + [0] * len(sub_heads))
    X_tgt = X_fit[: len(tgt_heads)]

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=300)
    clf.fit(X_fit, y_fit)

    X_pool = vec.transform([d[:cap] for d in docs])
    scores = clf.predict_proba(X_pool)[:, 1]

    try:
        from scipy.sparse import vstack
        order = np.argsort(-scores, kind="stable")
        n_pos = int(0.15 * n)
        n_neg = int(0.50 * n)
        if n_pos >= 10 and n_neg >= 10:
            pos = order[:n_pos].tolist()
            neg = order[n - n_neg:].tolist()
            if len(pos) > MAX_SELF_POS:
                pos = rng.sample(pos, MAX_SELF_POS)
            if len(neg) > MAX_SELF_NEG:
                neg = rng.sample(neg, MAX_SELF_NEG)
            X2 = vstack([X_tgt, X_pool[pos], X_pool[neg]]).tocsr()
            y2 = np.array([1] * (X_tgt.shape[0] + len(pos)) + [0] * len(neg))
            clf2 = LogisticRegression(C=1.0, class_weight="balanced", max_iter=300)
            clf2.fit(X2, y2)
            scores = clf2.predict_proba(X_pool)[:, 1]
    except Exception:
        pass

    return np.asarray(scores, dtype=np.float64)

def _target_vocab(target_sample):
    cnt = Counter()
    for t in target_sample:
        cnt.update(_WORD_RE.findall(t.lower()))
    return {w for w, c in cnt.items() if c >= 2}

def _is_junk(doc, vocab):
    if len(doc.strip()) < 200:
        return True
    head = doc[:5000]
    if not head:
        return True
    alpha_space = len(_ALPHA_SPACE_RE.findall(head))
    if alpha_space / len(head) < 0.7:
        return True
    if vocab:
        words = _WORD_RE.findall(head.lower())
        if not words:
            return True
        in_vocab = sum(1 for w in words if w in vocab)
        if in_vocab / len(words) < 0.5:
            return True
    return False

def _dedup_key(doc):
    norm = _WS_RE.sub(" ", doc[:2000].lower()).strip()[:500]
    return hashlib.md5(norm.encode("utf-8", "ignore")).digest()

def _token_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        def count(texts):
            # Fix: encode strings individually to avoid TypeError
            return [len(enc.encode_ordinary(text)) for text in texts]
        return count
    except Exception:
        def count(texts):
            return [max(1, len(t) // 4) for t in texts]
        return count

def select(docs, target_sample):
    seed = _seed()
    n = len(docs)
    if n == 0:
        return []

    try:
        scores = _score(docs, target_sample, seed)
    except Exception:
        scores = np.zeros(n, dtype=np.float64)

    vocab = _target_vocab(target_sample)
    seen = set()
    keep_mask = np.zeros(n, dtype=bool)
    for i, d in enumerate(docs):
        if _is_junk(d, vocab):
            continue
        k = _dedup_key(d)
        if k in seen:
            continue
        seen.add(k)
        keep_mask[i] = True

    cand = np.nonzero(keep_mask)[0]
    if len(cand) == 0:
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        return idx

    sorted_cand = cand[np.argsort(-scores[cand], kind="stable")]
    high_conf_indices = [c for c in sorted_cand if scores[c] >= 0.5]
    
    count_fn = _token_counter()
    
    def get_token_count(indices):
        if not indices: return 0
        return sum(count_fn([docs[i] for i in indices]))

    high_conf_tokens = get_token_count(high_conf_indices)
    min_tokens = 0.25 * T
    
    final_indices = []
    
    if high_conf_tokens >= TOKEN_BUDGET:
        total = 0
        chunk_size = 2000
        for s in range(0, len(sorted_cand), chunk_size):
            part = sorted_cand[s:s + chunk_size]
            counts = count_fn([docs[i] for i in part])
            for i, c in zip(part, counts):
                final_indices.append(int(i))
                total += c
                if total >= TOKEN_BUDGET:
                    break
            if total >= TOKEN_BUDGET:
                break
    elif high_conf_tokens >= min_tokens:
        final_indices = [int(c) for c in high_conf_indices]
    else:
        remaining_needed = min_tokens - high_conf_tokens
        # combine high conf and others
        pool = high_conf_indices + [int(c) for c in sorted_cand if scores[c] < 0.5]
        
        total_kept_tokens = high_conf_tokens
        final_indices = [int(c) for c in high_conf_indices]
        
        # Add next best until min_tokens
        for c in sorted_cand:
            if scores[c] < 0.5:
                c_tokens = count_fn([docs[c]])[0]
                if total_kept_tokens < min_tokens:
                    final_indices.append(int(c))
                    total_kept_tokens += c_tokens
                else:
                    break
        
        if total_kept_tokens < TOKEN_BUDGET:
            kept_set = final_indices
            current_total = total_kept_tokens
            epoch = 0
            while current_total < TOKEN_BUDGET:
                shuffled = list(kept_set)
                random.Random(seed + epoch).shuffle(shuffled)
                pass_tokens = get_token_count(shuffled)
                
                if current_total + pass_tokens > TOKEN_BUDGET:
                    partial_count = 0
                    for idx in shuffled:
                        c = count_fn([docs[idx]])[0]
                        if current_total + partial_count + c <= TOKEN_BUDGET:
                            final_indices.append(idx)
                            partial_count += c
                        else:
                            break
                    current_total += partial_count
                    break
                else:
                    final_indices.extend(shuffled)
                    current_total += pass_tokens
                epoch += 1

    return final_indices
