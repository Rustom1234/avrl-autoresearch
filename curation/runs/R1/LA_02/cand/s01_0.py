"""Curator: filter-then-shuffle pipeline that keeps clean, target-like documents.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. dedupe (hash of normalised text and of its first 300 characters)
  2. cheap junk rules (length, non-alphabetic ratio, zlib compression ratio)
  3. domain score p_domain (TF-IDF word 1-2grams + logistic regression)
  4. word-order score (target bigram coverage + function-bigram rate)
  5. keep clean docs with p_domain >= 0.5, cap at T tokens by p_domain
  6. seeded shuffle
"""
import hashlib
import os
import random
import zlib

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover - common.py is always present in the real setup
    T = 9_830_400

MIN_CHARS = 200
PREFIX_CHARS = 300
N_NEG = 4000
ORDER_BIGRAMS = 400
ZLIB_MIN_RATIO = 0.2
P_KEEP = 0.5

# Lookup table: 1 where a latin-1 byte is alphabetic or whitespace.
_ALPHA_TABLE = np.array(
    [1 if (chr(i).isalpha() or chr(i).isspace()) else 0 for i in range(256)],
    dtype=np.uint8,
)


def _h(s):
    return hashlib.md5(s.encode("utf-8", "replace")).digest()


def _nonalpha_ratio(text):
    """Share of characters that are neither letters nor whitespace (first 4000 chars)."""
    chunk = text[:4000]
    if not chunk:
        return 1.0
    arr = np.frombuffer(chunk.encode("latin-1", "replace"), dtype=np.uint8)
    return 1.0 - float(_ALPHA_TABLE[arr].mean())


def _zlib_ratio(text):
    raw = text[:20000].encode("utf-8", "replace")
    if not raw:
        return 0.0
    return len(zlib.compress(raw)) / len(raw)


def _order_stats(text, bigram_set):
    """Return (fraction of bigrams in bigram_set, function-bigram rate, n_bigrams)
    over the first ORDER_BIGRAMS whitespace-token bigrams."""
    toks = text[:12000].split()[: ORDER_BIGRAMS + 1]
    n = len(toks) - 1
    if n <= 0:
        return 0.0, 0.0, 0
    hit = 0
    fn = 0
    prev = toks[0]
    for cur in toks[1:]:
        if (prev, cur) in bigram_set:
            hit += 1
        if cur == "the":
            if prev == "of" or prev == "in" or prev == "to":
                fn += 1
        elif cur == "and":
            if prev.endswith(","):
                fn += 1
        elif cur == "The":
            if prev.endswith("."):
                fn += 1
        prev = cur
    return hit / n, fn / n, n


def _two_means_threshold(x):
    """1-D 2-means. Returns the midpoint between the centres if the two clusters
    are clearly separated, else None."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 20:
        return None
    lo, hi = np.percentile(x, 10), np.percentile(x, 90)
    if hi - lo < 1e-9:
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        a = x[x < mid]
        b = x[x >= mid]
        if a.size == 0 or b.size == 0:
            return None
        nlo, nhi = a.mean(), b.mean()
        if abs(nlo - lo) < 1e-9 and abs(nhi - hi) < 1e-9:
            break
        lo, hi = nlo, nhi
    mid = 0.5 * (lo + hi)
    a = x[x < mid]
    b = x[x >= mid]
    if a.size == 0 or b.size == 0:
        return None
    pooled = np.sqrt((a.var() * a.size + b.var() * b.size) / x.size) + 1e-9
    # A unimodal distribution split in two gives a ratio of about 2.7; require more.
    if (hi - lo) / pooled > 4.0 and lo < 0.6 * hi:
        return float(mid)
    return None


def _count_tokens(texts):
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        counts = []
        step = 2000
        for i in range(0, len(texts), step):
            batch = enc.encode_ordinary_batch(texts[i : i + step], num_threads=8)
            counts.extend(len(t) for t in batch)
        return counts
    except Exception:
        return [max(1, len(t) // 4) for t in texts]


def _domain_scores(docs, cand, target_sample, rng):
    """p_domain for the candidate indices; returns an array aligned with cand."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    n = len(docs)
    # Cap per-document text so vectorising stays well inside the time budget.
    cap = int(max(800, min(4000, 3e7 / max(1, n))))

    neg_idx = rng.sample(range(n), min(N_NEG, n))
    pos_txt = [t[:cap] for t in target_sample]
    neg_txt = [docs[i][:cap] for i in neg_idx]

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_features=200_000,
        dtype=np.float32,
    )
    X = vec.fit_transform(pos_txt + neg_txt)
    y = np.array([1] * len(pos_txt) + [0] * len(neg_txt))
    clf = LogisticRegression(class_weight="balanced", C=10.0, max_iter=1000)
    clf.fit(X, y)

    p = np.zeros(len(cand), dtype=np.float64)
    step = 5000
    for s in range(0, len(cand), step):
        chunk = [docs[i][:cap] for i in cand[s : s + step]]
        p[s : s + len(chunk)] = clf.predict_proba(vec.transform(chunk))[:, 1]
    return p


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []
    target_sample = [t for t in target_sample if isinstance(t, str) and t.strip()]

    # ---- 1. dedupe (keep first occurrence) --------------------------------
    seen_full = set()
    seen_prefix = set()
    is_dup = np.zeros(n, dtype=bool)
    for i, d in enumerate(docs):
        norm = " ".join(d.lower().split())
        hf = _h(norm)
        hp = _h(norm[:PREFIX_CHARS])
        if hf in seen_full or hp in seen_prefix:
            is_dup[i] = True
        else:
            seen_full.add(hf)
            seen_prefix.add(hp)

    # ---- 4. cheap junk rules ----------------------------------------------
    t_na = [_nonalpha_ratio(t) for t in target_sample]
    na_thr = max(0.2, 1.5 * max(t_na) + 0.05) if t_na else 0.35
    is_junk = np.zeros(n, dtype=bool)
    for i, d in enumerate(docs):
        if is_dup[i]:
            continue
        if len(d) < MIN_CHARS or len(d.strip()) < MIN_CHARS:
            is_junk[i] = True
        elif _nonalpha_ratio(d) > na_thr:
            is_junk[i] = True
        elif _zlib_ratio(d) < ZLIB_MIN_RATIO:
            is_junk[i] = True

    cand = [i for i in range(n) if not is_dup[i] and not is_junk[i]]
    if not cand:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # ---- 2. domain score ---------------------------------------------------
    if target_sample:
        p = _domain_scores(docs, cand, target_sample, rng)
    else:
        p = np.ones(len(cand), dtype=np.float64)

    # ---- 3. word-order score -----------------------------------------------
    damaged = np.zeros(len(cand), dtype=bool)
    if target_sample:
        df = {}
        for t in target_sample:
            toks = t.split()
            for b in set(zip(toks, toks[1:])):
                df[b] = df.get(b, 0) + 1
        full_set = set(df)
        # Leave-one-out view for the target docs: a bigram counts only if it
        # occurs in at least one other target doc.
        loo_set = {b for b, c in df.items() if c >= 2}

        t_frac, t_func = [], []
        for t in target_sample:
            f, g, m = _order_stats(t, loo_set)
            if m >= 50:
                t_frac.append(f)
            if m >= 200:
                t_func.append(g)
        frac_floor = 0.5 * min(t_frac) if t_frac else 0.0
        func_thr = 0.5 * min(t_func) if t_func else 0.0

        frac = np.zeros(len(cand))
        func = np.zeros(len(cand))
        nbig = np.zeros(len(cand), dtype=np.int64)
        for k, i in enumerate(cand):
            frac[k], func[k], nbig[k] = _order_stats(docs[i], full_set)

        in_dom = (p > 0.5) & (nbig >= 50)
        km = _two_means_threshold(frac[in_dom]) if in_dom.any() else None
        if km is not None:
            frac_thr = km
            if t_frac:
                # Never set the cut above what real target docs reach.
                frac_thr = max(frac_floor, min(frac_thr, min(t_frac)))
        else:
            frac_thr = frac_floor

        damaged = (frac < frac_thr) | ((nbig >= 200) & (func < func_thr))

    # ---- 5. selection -------------------------------------------------------
    keep_pos = [k for k in range(len(cand)) if not damaged[k] and p[k] >= P_KEEP]
    if not keep_pos:
        # Safety net so the trainer always gets something: fall back to the
        # undamaged clean docs (or all clean docs), best p_domain first.
        keep_pos = [k for k in range(len(cand)) if not damaged[k]] or list(range(len(cand)))

    counts = _count_tokens([docs[cand[k]] for k in keep_pos])
    total = sum(counts)
    if total > T:
        order = sorted(range(len(keep_pos)), key=lambda j: (-p[keep_pos[j]], keep_pos[j]))
        chosen = []
        acc = 0
        for j in order:
            chosen.append(keep_pos[j])
            acc += counts[j]
            if acc >= T:
                break
        keep_pos = chosen
    # If total <= T: do not pad with off-domain text; the kept set is cycled.

    # ---- 6. order -----------------------------------------------------------
    final = sorted(cand[k] for k in keep_pos)
    rng.shuffle(final)
    return final
