"""Curator: target-aware cross-entropy + diversity penalty + dedup + junk removal.

Pipeline:
  1. Train LogisticRegression on (target_sample, pool_subsample) to get soft labels.
  2. Train a second LogisticRegression on a large pool_subsample (10k) using those soft labels.
  3. Score the pool using the second model.
  4. Apply a diversity penalty: subtract a factor based on cosine similarity to the target mean
     if a doc is "too" similar (avoids overfitting to a specific target example).
  5. Junk removal, dedup, and multi-pass tiering logic.
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
MAX_SELF_POS = 6000
MAX_SELF_NEG = 12000
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

def _score(docs, target_sample, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics.pairwise import cosine_similarity

    n = len(docs)
    rng = random.Random(seed)
    cap = int(max(600, min(HEAD_CHARS, POOL_CHAR_BUDGET // max(n, 1))))

    tgt_heads = [t[:HEAD_CHARS] for t in target_sample]
    
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 4),
        sublinear_tf=True,
        max_features=100_000,
        dtype=np.float32,
    )
    
    # Step 1: Initial model on (Target, Small Neg)
    sub_idx = rng.sample(range(n), min(NEG_SUBSAMPLE, n))
    sub_heads = [docs[i][:cap] for i in sub_idx]
    X_init = vec.fit_transform(tgt_heads + sub_heads)
    y_init = np.array([1] * len(tgt_heads) + [0] * len(sub_heads))
    
    clf_init = LogisticRegression(C=1.0, class_weight="balanced", max_iter=300)
    clf_init.fit(X_init, y_init)
    
    # Step 2: Cross-entropy/Soft-label training
    # Use a larger chunk of the pool to refine
    pool_sub_size = min(10000, n)
    pool_sub_idx = rng.sample(range(n), pool_sub_size)
    pool_sub_heads = [docs[i][:cap] for i in pool_sub_idx]
    X_pool_sub = vec.transform(pool_sub_heads)
    
    # Soft labels from the first model
    soft_labels = clf_init.predict_proba(X_pool_sub)[:, 1]
    # Train second model with soft labels
    clf_final = LogisticRegression(C=1.0, class_weight="balanced", max_iter=300)
    clf_final.fit(X_pool_sub, soft_labels)
    
    # Final Scoring
    X_all = vec.transform([d[:cap] for d in docs])
    raw_probs = clf_final.predict_proba(X_all)[:, 1]
    
    # Diversity Penalty
    # Subtract score if it's extremely similar to the mean target vector
    X_tgt_vec = vec.transform(tgt_heads)
    target_mean_vec = X_tgt_vec.mean(axis=0).reshape(1, -1)
    
    # Cosine similarity to the average target
    sims = cosine_similarity(X_all, target_mean_vec).flatten()
    # Penalty: if sim > 0.9, reduce score slightly to favor diversity
    # We want to avoid picking documents that are essentially identical to the target sample
    penalties = np.where(sims > 0.9, (sims - 0.9) * 0.2, 0.0)
    
    scores = raw_probs - penalties
    return np.clip(scores, 0, 1)

def _target_vocab(target_sample):
    cnt = Counter()
    for t in target_sample:
        cnt.update(_WORD_RE.findall(t.lower()))
    return {w for w, c in cnt.items() if c >= 2}

def _is_junk(doc, vocab, bigrams_set, seed_val, doc_idx):
    if len(doc.strip()) < 200:
        return True
    body_text_full = doc.splitlines()
    if len(body_text_full) > 1:
        body_text_full = body_text_full[1:]
    body_text_full = "".join(body_text_full)[:40000]
    if len(_ALPHA_SPACE_RE.findall(body_text_full)) / len(body_text_full) < 0.7:
        return True
    if vocab:
        words = _WORD_RE.findall(body_text_full.lower())
        if words:
            in_vocab = sum(1 for w in words if w in vocab)
            if in_vocab / len(words) < 0.5:
                return True
        else:
            return True
    tokens = _ORDER_RE.findall(body_text_full.lower())
    if len(tokens) < 30:
        return False
    h_real = 0
    for i in range(len(tokens) - 1):
        if (tokens[i], tokens[i+1]) in bigrams_set:
            h_real += 1
    h_real /= (len(tokens) - 1)
    rng = random.Random(seed_val + doc_idx)
    shuffled_tokens = tokens[:]
    rng.shuffle(shuffled_tokens)
    h_shuf = 0
    for i in range(len(shuffled_tokens) - 1):
        if (shuffled_tokens[i], shuffled_tokens[i+1]) in bigrams_set:
            h_shuf += 1
    h_shuf /= (len(shuffled_tokens) - 1)
    if h_real < 1.6 * h_shuf + 0.03:
        return True
    window_count = 0
    low_window_count = 0
    for i in range(0, len(tokens), 150):
        win = tokens[i : i + 150]
        if len(win) < 60: break
        window_count += 1
        h_real_w = 0
        for j in range(len(win) - 1):
            if (win[j], win[j+1]) in bigrams_set:
                h_real_w += 1
        h_real_w /= (len(win) - 1)
        if h_real_w < 1.6 * h_shuf + 0.03:
            low_window_count += 1
    if window_count > 0 and (low_window_count / window_count) > 0.25:
        return True
    return False

def _dedup_key(doc):
    norm = _WS_RE.sub(" ", doc[:2000].lower()).strip()[:500]
    return hashlib.md5(norm.encode("utf-8", "ignore")).digest()

def _token_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return lambda texts: [len(x) for x in enc.encode_ordinary_batch(texts)]
    except:
        return lambda texts: [max(1, len(t) // 4) for t in texts]

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
        body_text_full = d.splitlines()
        if len(body_text_full) > 1:
            body_text_full = body_text_full[1:]
        body_text_full = "".join(body_text_full)[:40000]
        
        if len(body_text_full) > 0:
            if len(_ALPHA_SPACE_RE.findall(body_text_full)) / len(body_text_full) < 0.7:
                rule_drops["short_alpha"] += 1
                continue
            if vocab:
                words = _WORD_RE.findall(body_text_full.lower())
                if words:
                    in_vocab = sum(1 for w in words if w in vocab)
                    if in_vocab / len(words) < 0.5:
                        rule_drops["vocab"] += 1
                        continue
                else:
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

    second_tokens = 0
    fill_tokens = 0
    n_second = 0
    n_fill = 0
    last_fill_score = None
    kept = []
    total = 0

    if a_ge_1:
        kept = list(tier1)
        total = tier1_tokens
        a = tier1_tokens / float(T)
        a_str = ">=%.4f (tier 1 covers budget)" % a
    else:
        a = tier1_tokens / float(T)
        a_str = "%.4f" % a
        kept = list(tier1)
        total = tier1_tokens
        tier_target = min(2.0 * a, TIER1_MAX_SHARE) * T
        tier_total = tier1_tokens
        for idx_val, c in zip(tier1, tier1_counts):
            if tier_total >= tier_target or total >= TOKEN_BUDGET:
                break
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
                fill_tokens += c
                n_fill += 1
                last_fill_score = float(scores[idx_val])
                if total >= TOKEN_BUDGET:
                    break

    random.Random(seed).shuffle(kept)
    denom = float(total) if total > 0 else 1.0
    LAST_STATS = {
        "a": a,
        "a_is_lower_bound": bool(a_ge_1),
        "tier1_docs": n_tier1_total,
        "tier1_docs_used": len(tier1),
        "tier1_tokens_counted": int(tier1_tokens),
        "second_copy_docs": n_second,
        "second_copy_token_share": second_tokens / denom,
        "fill_docs": n_fill,
        "fill_token_share": fill_tokens / denom,
        "last_fill_score": last_fill_score,
        "total_tokens": int(total),
        "total_over_T": total / float(T),
        "n_output": len(kept),
        "n_candidates": int(len(cand)),
        "rule_drops": dict(rule_drops),
    }
    _log("drops=%s candidates=%d" % (rule_drops, len(cand)))
    _log("a=%s tier1_docs=%d second_copy_docs=%d second_copy_token_share=%.4f fill_docs=%d fill_token_share=%.4f last_fill_score=%s total_tokens=%d (%.4f*T) n_output=%d" % (
        a_str, n_tier1_total, n_second, second_tokens / denom, n_fill, fill_tokens / denom,
        ("%.6f" % last_fill_score) if last_fill_score is not None else "n/a", int(total), total / float(T), len(kept)))
    return kept
