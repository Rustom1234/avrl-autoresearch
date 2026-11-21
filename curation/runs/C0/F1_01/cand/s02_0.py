"""Curator: filter-then-shuffle. Keeps only clean, in-domain, unique documents.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. exact/near-duplicate removal (hash of first 2000 normalised characters)
  2. junk filter (length, zlib compression ratio, ASCII letter/space fraction)
  3. domain filter (Wikitext-density + Jaccard Similarity)
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
PROB_THRESHOLD = 0.5
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
        random.Random(seed).shuffle(idx)
        return idx

    if n == 0:
        return []

    # ---- Step 1 + 2: dedup, then junk filter ----
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

    # ---- Step 3: Wikitext-density + Jaccard Similarity ----
    # Pre-process target sample for similarity
    target_words = set()
    for t in target_sample:
        if isinstance(t, str):
            target_words.update(t.lower().split())
    
    def get_score(d):
        # Wikitext-density: punctuation/structure signals
        s_chars = d[:SCORE_CHARS]
        punct = sum(s_chars.count(c) for c in (":", "-", "(", ")", '"'))
        density = punct / max(1, len(s_chars))
        
        # Jaccard similarity on first 200 words
        words = d.split()[:200]
        if not words: return density
        current_words = set(w.lower() for w in words)
        intersection = len(current_words.intersection(target_words))
        union = len(current_words.union(target_words))
        jaccard = intersection / union if union > 0 else 0
        
        # Combined heuristic (weighted)
        return (density * 0.4) + (jaccard * 0.6)

    scores = np.array([get_score(docs[i]) for i in clean], dtype=np.float64)
    
    # Use a threshold or just rank (here we use a simple 0.1 threshold relative to mean)
    # To keep it robust, we'll keep docs above the 30th percentile or a fixed heuristic
    domain_ok = scores > (np.mean(scores) * 0.8)
    # Ensure we don't drop everything; fall back to the highest scores
    if not any(domain_ok):
        domain_ok = np.argsort(scores)[-max(1, len(clean)//10):] > -1
    
    # Final filter: domain_ok + surface_ok
    valid_indices = []
    for k, i in enumerate(clean):
        if domain_ok[k] and _surface_ok(docs[i]):
            valid_indices.append(k)
    
    # ---- Step 4: shuffled-word damage filter ----
    target_bigrams = set()
    for t in target_sample:
        if not isinstance(t, str): continue
        w = t.split()
        target_bigrams.update(zip(w, w[1:]))

    cov = np.empty(len(valid_indices), dtype=np.float64)
    rate = np.empty(len(valid_indices), dtype=np.float64)
    for k, idx_in_clean in enumerate(valid_indices):
        cov[k], rate[k] = _damage_stats(docs[clean[idx_in_clean]], target_bigrams)

    med = float(np.median(cov)) if len(cov) > 0 else 0.0
    damage_ok = (cov >= COVERAGE_FACTOR * med) & (rate >= MIN_COMMON_RATE)

    keep = np.zeros(len(valid_indices), dtype=bool)
    keep[damage_ok] = True
    survivors = [clean[valid_indices[k]] for k in np.nonzero(keep)[0]]

    # ---- Step 5: token budget check, top-up, shuffle ----
    target_tokens = int(TOPUP_FRACTION * T)
    count_tokens = _make_token_counter()
    BATCH = 512
    total = 0
    
    # Estimate current
    for s in range(0, len(survivors), BATCH):
        batch_docs = [docs[i] for i in survivors[s : s + BATCH]]
        total += sum(count_tokens(batch_docs))
        if total >= target_tokens: break

    if total < target_tokens:
        # Get all non-kept docs that were in valid_indices
        rest_indices = [valid_indices[k] for k in np.nonzero(~keep)[0]]
        # Sort by score descending
        rest_indices = sorted(rest_indices, key=lambda k: scores[k], reverse=True)
        
        for r_idx in rest_indices:
            survivors.append(clean[r_idx])
            total += sum(count_tokens([docs[clean[r_idx]]]))
            if total >= target_tokens: break

    if not survivors:
        return fallback()

    rng.shuffle(survivors)
    return survivors
