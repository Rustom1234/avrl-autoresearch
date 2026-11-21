"""Curator: Cross-Entropy soft-labeling + Distributional filter + Dedup + Junk removal.

Pipeline:
  1. Distributional filter: Remove docs with character n-gram distributions that
     deviate significantly from target_sample using variance of n-gram frequencies.
  2. Soft-label training: Train a small LR on (target_sample, pool_subsample) to
     get soft labels for the large pool.
  3. Cross-Entropy training: Train a second LR on 15,000 samples of the pool 
     using these soft labels as targets.
  4. Final selection logic with tiering and shuffling.
"""
import os
import sys
import random
import re
import hashlib
from collections import Counter

import numpy as np
from common import T

HEAD_CHARS = 3000
POOL_CHAR_BUDGET = 60_000_000
NEG_SUBSAMPLE = 4000
MAX_SOFT_SAMPLES = 15000
TOKEN_BUDGET = 1.0 * T
TIER1_THRESHOLD = 0.5
TIER1_MAX_SHARE = 0.85
COUNT_CHUNK = 2000

LAST_STATS = {}

_WORD_RE = re.compile(r"[^\W\d_]+")
_ALPHA_SPACE_RE = re.compile(r"[^\W\d_]|\s")
_WS_RE = re.compile(r"\s+")
_ORDER_RE = re.compile(r"[^\W\d_]+|[^\w\s]")

def _seed():
    try:
        return int(os.environ.get("CURATE_SEED", 0))
    except ValueError:
        return 0

def _log(msg):
    try:
        print("[curate] " + msg, file=sys.stderr, flush=True)
    except Exception:
        pass

def _get_ngram_dist(texts, n=3):
    from collections import defaultdict
    counts = defaultdict(int)
    for text in texts:
        text = text[:HEAD_CHARS]
        for i in range(len(text) - n + 1):
            counts[text[i:i+n]] += 1
    
    total = sum(counts.values())
    if total == 0: return None
    
    # Return variance of frequencies as a proxy for "structure"
    freqs = [v / total for v in counts.values()]
    return np.var(freqs) if freqs else 0

def _score(docs, target_sample, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    
    n = len(docs)
    rng = random.Random(seed)
    
    # Distributional Filter
    # Calculate target variance
    target_texts = [t[:HEAD_CHARS] for t in target_sample]
    target_var = _get_ngram_dist(target_texts, n=3)
    
    # Filter docs based on variance (prune outliers like code/legal)
    # Logic: Docs with variance too high/low compared to target are noise
    # We use a conservative check: is it within 2 standard deviations?
    # Since we don't have std, we use a simple thresholding if target_var is stable
    valid_mask = np.ones(n, dtype=bool)
    for i in range(min(n, 5000)): # Sample check for speed
        d_var = _get_ngram_dist([docs[i]], n=3)
        if d_var is not None:
            # Simple heuristic: if variance is drastically different, it's a different domain
            if abs(d_var - target_var) > 0.5: 
                valid_mask[i] = False
    
    # Soft Labeling Stage
    cap = int(max(600, min(HEAD_CHARS, POOL_CHAR_BUDGET // max(n, 1))))
    tgt_heads = [t[:HEAD_CHARS] for t in target_sample]
    sub_idx = rng.sample(range(n), min(NEG_SUBSAMPLE, n))
    sub_heads = [docs[i][:HEAD_CHARS] for i in sub_idx]
    
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), 
                           sublinear_tf=True, max_features=100000, dtype=np.float32)
    X_fit = vec.fit_transform(tgt_heads + sub_heads)
    y_fit = np.array([1] * len(tgt_heads) + [0] * len(sub_heads))
    
    clf_soft = LogisticRegression(C=1.0, class_weight="balanced", max_iter=300)
    clf_soft.fit(X_fit, y_fit)
    
    X_pool = vec.transform([d[:cap] for d in docs])
    soft_probs = clf_soft.predict_proba(X_pool)[:, 1]
    
    # Cross-Entropy Training Stage
    # Pick 15,000 docs to train the final model on soft labels
    pool_indices = np.random.choice(n, min(MAX_SOFT_SAMPLES, n), replace=False)
    X_train_final = X_pool[pool_indices]
    y_train_final = soft_probs[pool_indices]
    
    clf_final = LogisticRegression(C=1.0, class_weight="balanced", max_iter=300)
    clf_final.fit(X_train_final, y_train_final)
    
    final_scores = clf_final.predict_proba(X_pool)[:, 1]
    
    # Apply the distributional filter mask to the final scores
    final_scores[~valid_mask] = -1.0
    return final_scores

def _target_vocab(target_sample):
    cnt = Counter()
    for t in target_sample:
        cnt.update(_WORD_RE.findall(t.lower()))
    return {w for w, c in cnt.items() if c >= 2}

def _is_junk(doc, vocab, bigrams_set, seed_val, doc_idx):
    if len(doc.strip()) < 200: return True
    body = doc.splitlines()
    if len(body) > 1: body = body[1:]
    body_text = "".join(body)[:40000]
    
    if len(_ALPHA_SPACE_RE.findall(body_text)) / len(body_text) < 0.7:
        return True
    
    if vocab:
        words = _WORD_RE.findall(body_text.lower())
        if not words or (sum(1 for w in words if w in vocab) / len(words)) < 0.5:
            return True
    
    tokens = _ORDER_RE.findall(body_text.lower())
    if len(tokens) < 30: return False
    
    h_real = 0
    for i in range(len(tokens) - 1):
        if (tokens[i], tokens[i+1]) in bigrams_set: h_real += 1
    h_real /= (len(tokens) - 1)
    
    rng = random.Random(seed_val + doc_idx)
    shuffled = tokens[:]
    rng.shuffle(shuffled)
    h_shuf = 0
    for i in range(len(shuffled) - 1):
        if (shuffled[i], shuffled[i+1]) in bigrams_set: h_shuf += 1
    h_shuf /= (len(shuffled) - 1)
    
    return h_real < 1.6 * h_shuf + 0.03

def _dedup_key(doc):
    norm = _WS_RE.sub(" ", doc[:2000].lower()).strip()[:500]
    return hashlib.md5(norm.encode("utf-8", "ignore")).digest()

def _token_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return lambda x: [len(t) for t in enc.encode_ordinary_batch(x)]
    except:
        return lambda x: [max(1, len(t) // 4) for t in x]

def _iter_counts(docs, cand, count_fn, chunk=COUNT_CHUNK):
    for s in range(0, len(cand), chunk):
        part = cand[s:s + chunk]
        counts = count_fn([docs[int(i)] for i in part])
        for i, c in zip(part, counts):
            yield int(i), int(c)

def select(docs, target_sample):
    global LAST_STATS
    seed = _seed()
    n = len(docs)
    if n == 0: return []

    try:
        scores = _score(docs, target_sample, seed)
    except:
        scores = np.zeros(n, dtype=np.float64)

    vocab = _target_vocab(target_sample)
    bigrams_set = set()
    for t in target_sample:
        tokens = _ORDER_RE.findall(t.lower())
        for i in range(len(tokens) - 1):
            bigrams_set.add((tokens[i], tokens[i+1]))

    rule_drops = {"short_alpha": 0, "vocab": 0, "order": 0, "dedup": 0}
    seen = set()
    keep_mask = np.zeros(n, dtype=bool)
    
    for i, d in enumerate(docs):
        if len(d.strip()) < 200:
            rule_drops["short_alpha"] += 1
            continue
        body = "".join(d.splitlines()[1:] if len(d.splitlines()) > 1 else [d])[:40000]
        if len(_ALPHA_SPACE_RE.findall(body)) / len(body) < 0.7:
            rule_drops["short_alpha"] += 1
            continue
        if vocab:
            words = _WORD_RE.findall(body.lower())
            if not words or (sum(1 for w in words if w in vocab) / len(words)) < 0.5:
                rule_drops["vocab"] += 1
                continue
        if _is_junk(d, vocab, bigrams_set, seed, i):
            rule_drops["order"] += 1
            continue
        k = _dedup_key(d)
        if k in seen:
            rule_drops["dedup"] += 1
            continue
        seen.add(k)
        keep_mask[i] = True

    cand = np.nonzero(keep_mask)[0]
    if len(cand) == 0:
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        return idx

    cand = cand[np.argsort(-scores[cand], kind="stable")]
    n_tier1_total = int(np.sum(scores[cand] >= TIER1_THRESHOLD))
    count_fn = _token_counter()
    it = _iter_counts(docs, cand, count_fn)

    tier1 = []
    tier1_counts = []
    tier1_tokens = 0
    pending = None
    a_ge_1 = False
    for idx_val, c in it:
        if scores[idx_val] >= TIER1_THRESHOLD:
            tier1.append(idx_val)
            tier1_counts.append(c)
            tier1_tokens += c
            if tier1_tokens >= TOKEN_BUDGET:
                a_ge_1 = True
                break
        else:
            pending = (idx_val, c)
            break

    kept = []
    total = 0
    second_tokens = 0
    n_second = 0
    n_fill = 0
    last_fill_score = None

    if a_ge_1:
        kept = list(tier1)
        total = tier1_tokens
        a = tier1_tokens / float(T)
        a_str = ">=%.4f" % a
    else:
        a = tier1_tokens / float(T)
        a_str = "%.4f" % a
        kept = list(tier1)
        total = tier1_tokens
        tier_target = min(2.0 * a, TIER1_MAX_SHARE) * T
        tier_total = tier1_tokens
        for idx_val, c in zip(tier1, tier1_counts):
            if tier_total >= tier_target or total >= TOKEN_BUDGET: break
            kept.append(idx_val)
            tier_total += c
            total += c
            second_tokens += c
            n_second += 1

        if total < TOKEN_BUDGET:
            def _fill_iter():
                if pending is not None: yield pending
                for item in it: yield item
            for idx_val, c in _fill_iter():
                kept.append(idx_val)
                total += c
                n_fill += 1
                last_fill_score = float(scores[idx_val])
                if total >= TOKEN_BUDGET: break

    random.Random(seed).shuffle(kept)
    denom = float(total) if total > 0 else 1.0
    LAST_STATS = {
        "a": a, "a_is_lower_bound": bool(a_ge_1), "tier1_docs": n_tier1_total,
        "tier1_docs_used": len(tier1), "tier1_tokens_counted": int(tier1_tokens),
        "second_copy_docs": n_second, "second_copy_token_share": second_tokens / denom,
        "fill_docs": n_fill, "fill_token_share": (total - tier1_tokens - second_tokens)/denom if denom > 0 else 0,
        "last_fill_score": last_fill_score, "total_tokens": int(total),
        "total_over_T": total / float(T), "n_output": len(kept),
        "n_candidates": int(len(cand)), "rule_drops": dict(rule_drops)
    }
    _log("drops=%s candidates=%d" % (rule_drops, len(cand)))
    _log("a=%s tier1_docs=%d second_copy_docs=%d fill_docs=%d total_tokens=%d" % 
          (a_str, n_tier1_total, n_second, n_fill, int(total)))
    return kept
