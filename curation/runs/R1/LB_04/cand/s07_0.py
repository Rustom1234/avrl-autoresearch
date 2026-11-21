"""Curator: target-similarity filter + dedup + junk removal + word-order check,
with mixed repetition of target-like docs plus fill.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Char n-gram TF-IDF + logistic regression (target vs. random pool subsample),
     followed by one round of self-training on the pool.
  2. Drop junk (too short, low alphabetic fraction, low in-target-vocabulary fraction,
     and low word-order coherence compared to a random shuffle).
  3. Drop exact / near duplicates (hash of normalized first 500 chars).
  4. Tier 1 = surviving docs with self-trained score >= 0.5; a = tier1_tokens / T.
       a >= 1: keep the top-scoring documents until ~1.0 * T tokens (as before).
       a <  1: (a) every tier-1 doc once,
               (b) second copies of tier-1 docs in descending score order until
                   tier-1 tokens (first + second copies) reach min(2a, 0.85) * T,
               (c) fill up to 1.0 * T with the next-best docs below 0.5, once each.
  5. One seeded shuffle of the whole list.

Diagnostics are written to stderr and stored in LAST_STATS (no stdout, no files).
"""
import os
import sys
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
TIER1_THRESHOLD = 0.5      # self-trained score defining tier 1
TIER1_MAX_SHARE = 0.85     # cap on tier-1 tokens (first + second copies) as a share of T
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

def _is_junk(doc, vocab, bigrams_set, seed_val, doc_idx):
    if len(doc.strip()) < 200:
        return True
    
    body_text_full = doc.splitlines()
    if len(body_text_full) > 1:
        body_text_full = body_text_full[1:]
    body_text_full = "".join(body_text_full)[:40000]
    
    alpha_space_count = len(_ALPHA_SPACE_RE.findall(body_text_full))
    if alpha_space_count / len(body_text_full) < 0.7:
        return True
        
    if vocab:
        words = _WORD_RE.findall(body_text_full.lower())
        if not words:
            return True
        in_vocab = sum(1 for w in words if w in vocab)
        if in_vocab / len(words) < 0.5:
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
        
    # Window test
    window_count = 0
    low_window_count = 0
    for i in range(0, len(tokens), 150):
        win = tokens[i : i + 150]
        if len(win) < 60:
            break
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
        def count(texts):
            return [len(x) for x in enc.encode_ordinary_batch(texts)]
        return count
    except Exception:
        def count(texts):
            return [max(1, len(t) // 4) for t in texts]
        return count

def _iter_counts(docs, cand, count_fn, chunk=COUNT_CHUNK):
    """Lazily yield (doc_index, token_count) in the order of cand."""
    for s in range(0, len(cand), chunk):
        part = cand[s:s + chunk]
        counts = count_fn([docs[int(i)] for i in part])
        for i, c in zip(part, counts):
            yield int(i), int(c)

def select(docs, target_sample):
    global LAST_STATS
    seed = _seed()
    n = len(docs)
    if n == 0:
        return []

    try:
        scores = _score(docs, target_sample, seed)
    except Exception:
        scores = np.zeros(n, dtype=np.float64)

    vocab = _target_vocab(target_sample)
    bigrams_set = set()
    for t in target_sample:
        tokens = _ORDER_RE.findall(t.lower())
        for i in range(len(tokens) - 1):
            bigrams_set.add((tokens[i], tokens[i+1]))

    # Counting drops
    rule_drops = {"short_alpha": 0, "vocab": 0, "order": 0, "dedup": 0}
    seen = set()
    keep_mask = np.zeros(n, dtype=bool)
    
    for i, d in enumerate(docs):
        # Manual checks for logging
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

    # ---- Pass over tier 1 (score >= threshold), in descending score order ----
    tier1 = []
    tier1_counts = []
    tier1_tokens = 0
    pending = None          # first below-threshold doc already pulled from the iterator
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

    if a_ge_1:
        # Tier 1 alone covers the budget: identical to the previous behaviour
        # (top-scored docs up to 1.0*T). Counting stops at the budget, so a is
        # only known as a lower bound here.
        kept = list(tier1)
        total = tier1_tokens
        a = tier1_tokens / float(T)
        a_str = ">=%.4f (tier 1 covers budget; counting stopped early)" % a
    else:
        a = tier1_tokens / float(T)
        a_str = "%.4f" % a

        # (a) every tier-1 doc once
        kept = list(tier1)
        total = tier1_tokens

        # (b) second copies, descending score, until tier-1 total reaches min(2a, 0.85)*T
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

        # (c) fill with next-best docs below the threshold, once each
        if total < TOKEN_BUDGET:
            def _fill_iter():
                if pending is not None:
                    yield pending
                for item in it:
                    yield item
            for idx_val, c in _fill_iter():
                kept.append(idx_val)
                total += c
                fill_tokens += c
                n_fill += 1
                last_fill_score = float(scores[idx_val])
                if total >= TOKEN_BUDGET:
                    break

    # One uniform shuffle of the whole list, so two copies of a doc sit far apart.
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
    _log(
        "a=%s tier1_docs=%d second_copy_docs=%d second_copy_token_share=%.4f "
        "fill_docs=%d fill_token_share=%.4f last_fill_score=%s total_tokens=%d (%.4f*T) n_output=%d"
        % (
            a_str,
            n_tier1_total,
            n_second,
            second_tokens / denom,
            n_fill,
            fill_tokens / denom,
            ("%.6f" % last_fill_score) if last_fill_score is not None else "n/a",
            int(total),
            total / float(T),
            len(kept),
        )
    )

    return kept
