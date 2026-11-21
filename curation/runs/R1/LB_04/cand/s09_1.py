import os
import sys
import random
import re
import hashlib
from collections import Counter
import numpy as np
from common import T

# Configuration and Constants
HEAD_CHARS = 3000          
POOL_CHAR_BUDGET = 60_000_000  
NEG_SUBSAMPLE = 4000
MAX_SELF_POS = 6000        
MAX_SELF_NEG = 12000
TOKEN_BUDGET = 1.0 * T
TIER1_THRESHOLD = 0.5      
TIER1_MAX_SHARE = 0.85     
COUNT_CHUNK = 2000

_WORD_RE = re.compile(r"[^\W\d_]+")
_ALPHA_SPACE_RE = re.compile(r"[^\W\d_]|\s")
_WS_RE = re.compile(r"\s+")
_ORDER_RE = re.compile(r"[^\W\d_]+|[^\w\s]")

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
    for s in range(0, len(cand), chunk):
        part = cand[s:s + chunk]
        counts = count_fn([docs[int(i)] for i in part])
        for i, c in zip(part, counts):
            yield int(i), int(c)

def select(docs, target_sample):
    seed = _seed()
    n = len(docs)
    if n == 0:
        return []

    try:
        raw_scores = _score(docs, target_sample, seed)
    except Exception:
        raw_scores = np.zeros(n, dtype=np.float64)

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

    final_scores = np.zeros(len(cand), dtype=np.float64)
    for i, idx_val in enumerate(cand):
        doc_body = docs[idx_val].splitlines()
        if len(doc_body) > 1:
            doc_body = doc_body[1:]
        doc_body_str = "".join(doc_body)[:40000]
        word_count = len(_WORD_RE.findall(doc_body_str.lower()))
        
        raw_s = raw_scores[idx_val]
        w_num = max(0.0, 1.0 - (word_count / 10000.0))
        w_den = max(0.01, 1.0 - (word_count / 50000.0))
        final_scores[i] = raw_s * (w_num / w_den)

    cand = cand[np.argsort(-final_scores, kind="stable")]
    n_tier1_total = int(np.sum(final_scores >= TIER1_THRESHOLD))
    count_fn = _token_counter()
    it = _iter_counts(docs, cand, count_fn)

    tier1 = []
    tier1_counts = []
    tier1_tokens = 0
    pending = None
    a_ge_1 = False
    for idx_val, c in it:
        # Find index in sorted cand to check score
        pos = np.where(cand == idx_val)[0][0]
        if final_scores[pos] >= TIER1_THRESHOLD:
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
        kept = list(tier1)
        total = tier1_tokens
        a = tier1_tokens / float(T)
    else:
        a = tier1_tokens / float(T)
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
                if pending is not None:
                    yield pending
                for item in it:
                    yield item
            for idx_val, c in _fill_iter():
                kept.append(idx_val)
                total += c
                fill_tokens += c
                n_fill += 1
                pos = np.where(cand == idx_val)[0][0]
                last_fill_score = float(final_scores[pos])
                if total >= TOKEN_BUDGET:
                    break

    random.Random(seed).shuffle(kept)
    return kept
