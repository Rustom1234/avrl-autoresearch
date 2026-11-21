"""Curator: target-similarity filter + heuristics + dedup.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import string
import hashlib
import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

SPAM_MARKERS = ("click here to order", "best price guaranteed", "coupon")
_LETTERS = string.ascii_letters.encode()
_WS = b" \t\n\r\x0b\x0c"

_CHAR_BUDGET = 24_000_000

def _ascii_letter_ratio(d):
    b = d.encode("utf-8", "replace")
    non_ws = len(b) - (len(b) - len(b.translate(None, _WS)))
    if non_ws <= 0:
        return 0.0
    letters = len(b) - len(b.translate(None, _LETTERS))
    return letters / non_ws

def _bonus(d):
    b = 0.0
    nl = d.find("\n")
    first = (d if nl < 0 else d[:nl]).strip()
    if first.startswith("= ") and first.endswith(" ="):
        b += 1.0
    if " @-@ " in d:
        b += 1.0
    if " , " in d or " . " in d:
        b += 0.5
    return b

def _hard_drop(d):
    if len(d) < 200:
        return True
    low = d.lower()
    for m in SPAM_MARKERS:
        if m in low:
            return True
    if _ascii_letter_ratio(d) < 0.6:
        return True
    return False

def _h(s):
    return hashlib.md5(s.encode("utf-8", "replace")).digest()

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    def fallback():
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        return idx

    keep = np.zeros(n, dtype=bool)
    seen_full, seen_prefix = set(), set()
    for i, d in enumerate(docs):
        if not isinstance(d, str) or _hard_drop(d):
            continue
        norm = " ".join(d.split())
        hf = _h(norm)
        if hf in seen_full:
            continue
        hp = _h(norm[:300])
        if hp in seen_prefix:
            continue
        seen_full.add(hf)
        seen_prefix.add(hp)
        keep[i] = True
    
    if not keep.any():
        return fallback()

    scores = np.zeros(n, dtype=np.float64)
    tgt_scores = np.zeros(len(target_sample), dtype=np.float64)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        snip = int(max(300, min(2000, _CHAR_BUDGET // max(1, n + len(target_sample)))))
        pool_txt = [d[:snip] if isinstance(d, str) else "" for d in docs]
        tgt_txt = [t[:snip] if isinstance(t, str) else "" for t in target_sample]

        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 3),
            sublinear_tf=True,
            max_features=50000,
            dtype=np.float32,
        )
        
        X = vec.fit_transform(pool_txt + tgt_txt)
        Xp = X[:n]
        Xt = X[n:]

        if Xt.shape[0] > 0:
            # Calculate centroid of target set
            centroid = Xt.mean(axis=0)
            # Similarity of all pool docs to centroid
            sims = cosine_similarity(Xp, centroid.reshape(1, -1)).flatten()
            scores = sims
            # Similarity of targets to centroid (for heuristic calc)
            tgt_scores = cosine_similarity(Xt, centroid.reshape(1, -1)).flatten()
    except Exception:
        pass

    for i in range(n):
        if keep[i]:
            scores[i] += _bonus(docs[i])
    for j, t in enumerate(target_sample):
        if isinstance(t, str):
            tgt_scores[j] += _bonus(t)

    kept_idx = np.flatnonzero(keep)
    kept_scores = scores[kept_idx]
    order = kept_idx[np.argsort(-kept_scores, kind="stable")]

    if len(target_sample) > 0:
        # Thresholding logic from previous version
        thr = 0.5 * (float(np.mean(tgt_scores)) + float(np.median(kept_scores)))
    else:
        thr = float(np.median(kept_scores))
    
    n_in = int(np.sum(kept_scores > thr))
    if n_in > len(order): n_in = len(order)

    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        def count_batch(texts):
            return [len(x) for x in enc.encode_ordinary_batch(texts)]
    except Exception:
        def count_batch(texts):
            return [max(1, len(x) // 4) for x in texts]

    target_tokens = int(T * 1.01)
    counts = []
    cum = 0
    B = 1000
    pos = 0
    while pos < len(order) and cum < target_tokens:
        batch = [docs[int(i)] for i in order[pos:pos + B]]
        c = count_batch(batch)
        counts.extend(c)
        cum += sum(c)
        pos += B
        
    counts = np.asarray(counts, dtype=np.int64)
    csum = np.cumsum(counts)
    n_counted = len(counts)

    if n_counted and csum[-1] >= target_tokens:
        n_fill = int(np.searchsorted(csum, target_tokens, side="left")) + 1
    else:
        n_fill = n_counted

    n_sel = n_fill
    if n_in <= n_counted:
        in_tokens = int(csum[n_in - 1]) if n_in > 0 else 0
        if in_tokens < T and in_tokens >= 0.4 * T:
            n_sel = n_in

    selected = [int(i) for i in order[:n_sel]]
    if not selected:
        return fallback()

    random.Random(seed).shuffle(selected)
    return selected
