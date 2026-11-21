"""Curator: filter-then-shuffle. Keeps only clean, in-domain, unique documents.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. exact/near-duplicate removal (hash of first 2000 normalised characters)
  2. junk filter (length, zlib compression ratio, ASCII letter/space fraction)
  3. domain filter (Centroid-based Cosine Similarity vs target_sample TF-IDF)
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
except ImportError:
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
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    if n == 0:
        return []

    # ---- Steps 1 + 2: dedup (first occurrence wins), then junk filter ----
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

    # ---- Step 3: Centroid-based Cosine Similarity ----
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    pos_texts = [t[:SCORE_CHARS] for t in target_sample if isinstance(t, str) and t.strip()]
    
    vec = TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        max_features=100_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    
    if pos_texts:
        X_pos = vec.fit_transform(pos_texts)
        # Compute centroid (mean vector)
        centroid = np.asarray(X_pos.mean(axis=0))
        # Score clean docs
        X_clean = vec.transform([docs[i][:SCORE_CHARS] for i in clean])
        scores = np.asarray(X_clean.dot(centroid)).flatten()
        # Keep docs above 60th percentile
        threshold = np.percentile(scores, 60)
        domain_ok = np.array([scores[k] >= threshold for k in range(len(clean))], dtype=bool)
    else:
        domain_ok = np.array([True] * len(clean), dtype=bool)

    # ---- Step 4: shuffled-word damage filter ----
    target_bigrams = set()
    for t in target_sample:
        if not isinstance(t, str):
            continue
        w = t.split()
        target_bigrams.update(zip(w, w[1:]))

    cov = np.empty(len(clean), dtype=np.float64)
    rate = np.empty(len(clean), dtype=np.float64)
    for k, i in enumerate(clean):
        cov[k], rate[k] = _damage_stats(docs[i], target_bigrams)

    valid_cov = cov[domain_ok]
    med = float(np.median(valid_cov)) if valid_cov.size > 0 else 0.0
    damage_ok = (cov >= COVERAGE_FACTOR * med) & (rate >= MIN_COMMON_RATE)

    keep = domain_ok & damage_ok
    survivors = [clean[k] for k in np.nonzero(keep)[0]]

    # ---- Step 5: token budget check, optional top-up, seeded shuffle ----
    target_tokens = int(TOPUP_FRACTION * T)
    count_tokens = _make_token_counter()
    BATCH = 512

    total = 0
    for s in range(0, len(survivors), BATCH):
        if total >= target_tokens:
            break
        batch_indices = survivors[s : s + BATCH]
        total += sum(count_tokens([docs[i] for i in batch_indices]))

    if total < target_tokens:
        # remaining docs that are deduplicated and pass steps 2 and 4
        rest_indices = np.nonzero(damage_ok & ~keep)[0]
        if len(rest_indices) > 0:
            # Sort rest by scores (descending)
            # Since we don't have access to raw scores easily without re-calculating,
            # we just use a simple order.
            rest_sorted = [clean[k] for k in rest_indices]
            
            for s in range(0, len(rest_sorted), BATCH):
                batch = rest_sorted[s : s + BATCH]
                counts = count_tokens([docs[i] for i in batch])
                for i, c in zip(batch, counts):
                    survivors.append(i)
                    total += c
                    if total >= target_tokens:
                        break
                if total >= target_tokens:
                    break

    if not survivors:
        return fallback()

    rng.shuffle(survivors)
    return survivors
