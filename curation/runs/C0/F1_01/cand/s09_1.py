"""Curator: filter-then-shuffle. Keeps only clean, in-domain, unique documents.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. exact/near-duplicate removal (hash of first 2000 normalised characters)
  2. junk filter (length, zlib compression ratio, ASCII letter/space fraction)
  3. domain filter (TF-IDF Centroid Cosine Similarity + wikitext surface form)
  4. shuffled-word damage filter (target bigram coverage, common bigram rate)
  5. seeded shuffle, with a top-up to 0.5 T tokens if too few documents survive
"""
import hashlib
import os
import random
import string
import zlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from common import T
except Exception:  # pragma: no cover - fallback if common.py is unavailable
    T = 9_830_400

MIN_CHARS = 200
MIN_ZLIB_RATIO = 0.2
MIN_ALPHA_FRAC = 0.70
SCORE_CHARS = 3000
PROB_THRESHOLD = 0.4  # Used as a percentile threshold
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
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    if n == 0:
        return []

    # ---- Steps 1 + 2: dedup, then junk filter ----
    seen = set()
    clean = []
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
        clean.append(i)
    del seen

    if not clean:
        return fallback()

    # ---- Step 3: TF-IDF Centroid Similarity ----
    pos = [t[:SCORE_CHARS] for t in target_sample if isinstance(t, str) and t.strip()]
    
    vec = TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        max_features=100000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    
    X_pos = vec.fit_transform(pos)
    if X_pos.shape[1] > 0:
        centroid = np.asarray(X_pos.mean(axis=0))
        # Score all candidates
        chunk_size = 1000
        probs = np.empty(len(clean), dtype=np.float32)
        for s in range(0, len(clean), chunk_size):
            chunk_indices = clean[s : s + chunk_size]
            chunk_texts = [docs[i][:SCORE_CHARS] for i in chunk_indices]
            X_chunk = vec.transform(chunk_texts)
            # scores is (num_chunks, num_features), centroid is (num_features,)
            # Dot product returns (num_chunks,)
            scores = X_chunk.dot(centroid)
            
            norm_chunk = np.linalg.norm(X_chunk, axis=1)
            norm_centroid = np.linalg.norm(centroid)
            probs[s : s + len(chunk_indices)] = scores / (np.maximum(norm_chunk, 1e-8) * max(norm_centroid, 1e-8))
    else:
        probs = np.ones(len(clean), dtype=np.float32)

    # Percentile Cutoff (Top 40%)
    threshold_val = np.percentile(probs, 60)
    domain_ok = np.array([probs[k] >= threshold_val and _surface_ok(docs[i]) 
                           for k, i in enumerate(clean)], dtype=bool)

    # ---- Step 4: shuffled-word damage filter ----
    target_bigrams = set()
    for t in target_sample:
        if not isinstance(t, str): continue
        w = t.split()
        target_bigrams.update(zip(w, w[1:]))

    cov = np.empty(len(clean), dtype=np.float64)
    rate = np.empty(len(clean), dtype=np.float64)
    for k, i in enumerate(clean):
        cov[k], rate[k] = _damage_stats(docs[i], target_bigrams)

    # Filter only on docs that passed domain_ok
    valid_indices = np.where(domain_ok)[0]
    if valid_indices.size > 0:
        med_cov = np.median(cov[valid_indices])
        damage_ok = (cov >= COVERAGE_FACTOR * med_cov) & (rate >= MIN_COMMON_RATE)
    else:
        damage_ok = np.zeros(len(clean), dtype=bool)
    
    # Apply domain_ok AND damage_ok
    keep = domain_ok & damage_ok
    survivors = [clean[k] for k in np.nonzero(keep)[0]]

    # ---- Step 5: token budget check ----
    target_tokens = int(TOPUP_FRACTION * T)
    count_tokens = _make_token_counter()
    
    total = 0
    # We process survivors first
    BATCH = 512
    for s in range(0, len(survivors), BATCH):
        batch_idx = survivors[s : s + BATCH]
        counts = count_tokens([docs[i] for i in batch_idx])
        for i, c in zip(batch_idx, counts):
            total += c
            if total >= target_tokens: break
        if total >= target_tokens: break

    if total < target_tokens:
        # Get rest that passed domain_ok but not the full keep (failed damage or were not in domain_ok)
        rest_candidates = [clean[k] for k in np.nonzero(domain_ok & ~keep)[0]]
        # Sort by score
        # Note: probs corresponds to clean, so we need the index in clean
        def get_score(idx):
            try:
                return probs[clean.index(idx)]
            except:
                return 0.0
        rest_candidates.sort(key=get_score, reverse=True)
        
        for i in rest_candidates:
            total += count_tokens([docs[i]])[0]
            survivors.append(i)
            if total >= target_tokens: break

    if not survivors:
        return fallback()

    rng.shuffle(survivors)
    return survivors
