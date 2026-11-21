"""Curator: filter-then-shuffle. Keeps only clean, in-domain, unique documents.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. exact/near-duplicate removal (hash of first 2000 normalised characters)
  2. junk filter (length, zlib compression ratio, ASCII letter/space fraction)
  3. similarity filter (TF-IDF cosine similarity to target centroid > 70th percentile)
  4. shuffled-word damage filter (target bigram coverage, common bigram rate)
  5. seeded shuffle, with a top-up to 0.5 T tokens if too few documents survive
"""
import hashlib
import os
import random
import string
import zlib
import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover - fallback if common.py is unavailable
    T = 9_830_400

MIN_CHARS = 200
MIN_ZLIB_RATIO = 0.2
MIN_ALPHA_FRAC = 0.70
SCORE_CHARS = 3000
SURFACE_RATE = 3.0          # (' , ' + ' . ') per 1000 characters
BIGRAM_WINDOW = 400         # first 400 word bigrams for target coverage
RATE_WINDOW = 2000          # words used for the common-bigram rate
MIN_COMMON_RATE = 1.0       # per 100 words
COVERAGE_FACTOR = 0.5       # times the median coverage of in-domain docs
TOPUP_FRACTION = 0.5        # of T

_ALPHA_SPACE = (string.ascii_letters + " ").encode("ascii")
_COMMON_BIGRAMS = {("of", "the"), ("in", "the"), (",", "and"), (".", "The")}


def _is_junk(d):
    """Step 2: too short, too repetitive, or too few letters/spaces."""
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
    """Wikitext surface form: heading start, or spaced punctuation."""
    if d.lstrip().startswith("= "):
        return True
    cnt = d.count(" , ") + d.count(" . ")
    return cnt * 1000.0 / max(1, len(d)) >= SURFACE_RATE


def _damage_stats(d, target_bigrams):
    """Step 4: (a) target-bigram coverage, (b) common-bigram rate per 100 words."""
    words = d.split(None, RATE_WINDOW)[:RATE_WINDOW]
    nw = len(words)
    if nw < 2:
        return 0.0, 0.0
    head = words[: BIGRAM_WINDOW + 1]
    nb = len(head) - 1
    hits = 0
    for bg in zip(head, head[1:]):
        if bg in target_bigrams:
            hits += 1
    cov = hits / nb
    common = 0
    for bg in zip(words, words[1:]):
        if bg in _COMMON_BIGRAMS:
            common += 1
    rate = common * 100.0 / nw
    return cov, rate


def _make_token_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        def count(texts):
            return [len(x) for x in enc.encode_ordinary_batch(texts, num_threads=8)]
        return count
    except Exception:
        def count(texts):
            return [max(1, len(t) // 4) for t in texts]
        return count


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)

    def fallback():
        indices = list(range(n))
        rng.shuffle(indices)
        return indices

    if n == 0:
        return []

    # ---- Steps 1 + 2: dedup, then junk filter ----
    seen = set()
    clean_indices = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < MIN_CHARS:
            continue
        norm = " ".join(d.lower().split())[:2000]
        key = hashlib.md5(norm.encode("utf-8", "replace")).digest()
        if key in seen:
            continue
        seen.add(key)
        if _is_junk(d):
            continue
        clean_indices.append(i)
    del seen

    if not clean_indices:
        return fallback()

    # ---- Step 3: Dense representation similarity ----
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    pos_texts = [t[:SCORE_CHARS] for t in target_sample if isinstance(t, str) and t.strip()]
    
    # Pick a random subset of pool to build the TF-IDF space
    subset_size = min(len(clean_indices), 5000)
    subset_indices = rng.sample(clean_indices, subset_size)
    pool_texts = [docs[i][:SCORE_CHARS] for i in subset_indices]

    vec = TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        max_features=100000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    
    # Fit on target + random pool
    all_texts = pos_texts + pool_texts
    if all_texts:
        X = vec.fit_transform(all_texts)
        # Centroid of target
        target_vecs = X[:len(pos_texts)]
        if target_vecs.shape[0] > 0:
            centroid = np.mean(target_vecs, axis=0)
            # Similarity of every clean doc
            X_clean = vec.transform([docs[i][:SCORE_CHARS] for i in clean_indices])
            sims = cosine_similarity(X_clean, centroid.reshape(1, -1)).flatten()
        else:
            sims = np.ones(len(clean_indices))
    else:
        sims = np.ones(len(clean_indices))

    # Filter by 70th percentile
    threshold = np.percentile(sims, 70) if sims.size > 0 else 0
    
    # Apply domain_ok (similarity + surface)
    domain_ok = np.zeros(len(clean_indices), dtype=bool)
    for k, i in enumerate(clean_indices):
        if sims[k] >= threshold and _surface_ok(docs[i]):
            domain_ok[k] = True

    # ---- Step 4: shuffled-word damage filter ----
    target_bigrams = set()
    for t in target_sample:
        if not isinstance(t, str): continue
        w = t.split()
        target_bigrams.update(zip(w, w[1:]))

    cov = np.empty(len(clean_indices), dtype=np.float64)
    rate = np.empty(len(clean_indices), dtype=np.float64)
    for k, i in enumerate(clean_indices):
        cov[k], rate[k] = _damage_stats(docs[i], target_bigrams)

    # Median coverage only of those passing domain_ok
    med_cov = 0.0
    if np.any(domain_ok):
        med_cov = float(np.median(cov[domain_ok]))
    
    damage_ok = (cov >= COVERAGE_FACTOR * med_cov) & (rate >= MIN_COMMON_RATE)
    keep = domain_ok & damage_ok
    survivors = [clean_indices[k] for k in np.nonzero(keep)[0]]

    # ---- Step 5: token budget check, top-up, shuffle ----
    target_tokens = int(TOPUP_FRACTION * T)
    count_tokens = _make_token_counter()
    BATCH = 512

    total = 0
    for s in range(0, len(survivors), BATCH):
        if total >= target_tokens: break
        batch_indices = survivors[s : s + BATCH]
        total += sum(count_tokens([docs[i] for i in batch_indices]))

    if total < target_tokens:
        # Fill from docs that passed similarity/surface but failed damage
        # Or just any clean docs in order of similarity
        rest_indices = [k for k in range(len(clean_indices)) if keep[k] == False]
        # Sort rest by similarity score
        rest_indices = sorted(rest_indices, key=lambda x: sims[x], reverse=True)
        
        for idx in rest_indices:
            survivors.append(clean_indices[idx])
            total += count_tokens([docs[clean_indices[idx]]])[0]
            if total >= target_tokens:
                break

    if not survivors:
        return fallback()

    rng.shuffle(survivors)
    return survivors
