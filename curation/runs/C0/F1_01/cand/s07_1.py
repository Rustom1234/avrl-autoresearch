import hashlib
import os
import random
import string
import zlib
import numpy as np

try:
    from common import T
except Exception:
    T = 9_830_400

MIN_CHARS = 200
MIN_ZLIB_RATIO = 0.2
MIN_ALPHA_FRAC = 0.70
SCORE_CHARS = 3000
SURFACE_RATE = 3.0
BIGRAM_WINDOW = 400
RATE_WINDOW = 2000
MIN_COMMON_RATE = 1.0
COVERAGE_FACTOR = 0.5

_ALPHA_SPACE = (string.ascii_letters + " ").encode("ascii")
_COMMON_BIGRAMS = {("of", "the"), ("in", "the"), (",", "and"), (".", "The")}

def _is_junk(d):
    n = len(d)
    if n < MIN_CHARS:
        return True
    b = d.encode("latin-1", "replace")
    non_alpha = len(b.translate(None, _ALPHA_SPACE))
    if (n - non_alpha) / n < MIN_ALPHA_FRAC:
        return True
    raw = d.encode("utf-8", "replace")
    if len(zlib.compress(raw)) / max(1, len(raw)) < MIN_ZLIB_RATIO:
        return True
    return False

def _surface_ok(d):
    d_strip = d.lstrip()
    if d_strip.startswith("= "):
        return True
    cnt = d.count(" , ") + d.count(" . ")
    return cnt * 1000.0 / max(1, len(d)) >= SURFACE_RATE

def _damage_stats(d, target_bigrams):
    words = d.split()
    nw = len(words)
    if nw < 2:
        return 0.0, 0.0
    
    # Limited window to keep it O(N)
    idx_limit = min(nw, RATE_WINDOW)
    words_subset = words[:idx_limit]
    
    head_limit = min(idx_limit, BIGRAM_WINDOW + 1)
    head = words_subset[:head_limit]
    nb = len(head) - 1
    
    hits = 0
    for bg in zip(head, head[1:]):
        if bg in target_bigrams:
            hits += 1
    cov = hits / nb if nb > 0 else 0.0
    
    common = 0
    for bg in zip(words_subset, words_subset[1:]):
        if bg in _COMMON_BIGRAMS:
            common += 1
    rate = common * 100.0 / idx_limit
    return cov, rate

def _make_token_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        def count(texts):
            return [len(x) for x in enc.encode_ordinary_batch(texts, num_threads=1)]
        return count
    except Exception:
        def count(texts):
            return [max(1, len(t) // 4) for t in texts]
        return count

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0: return []

    # 1 & 2: Dedup and Junk
    seen = set()
    clean = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < MIN_CHARS:
            continue
        norm = " ".join(d.lower().split())[:2000]
        h = hashlib.md5(norm.encode("utf-8", "replace")).digest()
        if h in seen: continue
        seen.add(h)
        if _is_junk(d): continue
        clean.append(i)

    if not clean:
        indices = list(range(n))
        rng.shuffle(indices)
        return indices

    # 3: Domain Similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    pos_texts = [t[:SCORE_CHARS] for t in target_sample if isinstance(t, str) and t.strip()]
    
    if pos_texts:
        vec = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, 
                               ngram_range=(1, 2), max_features=20000, sublinear_tf=True, dtype=np.float32)
        X_pos = vec.fit_transform(pos_texts)
        centroid = np.mean(X_pos, axis=0)
        
        pool_texts = [docs[i][:SCORE_CHARS] for i in clean]
        X_pool = vec.transform(pool_texts)
        
        scores = np.dot(X_pool, centroid)
        norms = np.linalg.norm(X_pool, axis=1)
        c_norm = np.linalg.norm(centroid)
        
        valid_scores = np.zeros(len(scores))
        mask = (norms > 0) & (c_norm > 0)
        valid_scores[mask] = scores[mask] / (norms[mask] * c_norm)
        
        # Keep top 30%
        threshold = int(len(scores) * 0.7)
        top_idx_sorted = np.argsort(-valid_scores)[:threshold]
        current_clean = [clean[k] for k in top_idx_sorted]
    else:
        current_clean = clean

    # Surface Filter
    refined = [i for i in current_clean if _surface_ok(docs[i])]
    if len(refined) < len(clean) * 0.01 and len(clean) > 10:
        refined = [i for i in clean if _surface_ok(docs[i])]

    # 4: Damage Filter
    target_bigrams = set()
    for t in target_sample:
        if isinstance(t, str):
            w = t.split()
            target_bigrams.update(zip(w, w[1:]))

    covs = []
    rates = []
    for i in refined:
        c, r = _damage_stats(docs[i], target_bigrams)
        covs.append(c)
        rates.append(r)

    if covs:
        med_cov = np.median(covs)
        damage_ok = np.array([c >= COVERAGE_FACTOR * med_cov and r >= MIN_COMMON_RATE 
                               for c, r in zip(covs, rates)])
        survivors = [refined[k] for k in np.nonzero(damage_ok)[0]]
    else:
        survivors = refined

    # 5: Token Budget and Shuffle
    count_tokens = _make_token_counter()
    current_total = 0
    
    # Simple check to see if we need to pull from original clean
    # We only want to keep enough to satisfy the training loop
    # but we don't need to over-select. 
    # The trainer will cycle the list, so we just need a good representative set.
    
    # Ensure we have a decent amount of content
    # If survivors are too few, supplement from 'clean'
    if len(survivors) == 0:
        survivors = refined
    
    # Shuffle and return
    rng.shuffle(survivors)
    
    # Final safety: if we have nothing, return clean
    if not survivors:
        return clean
        
    return survivors
