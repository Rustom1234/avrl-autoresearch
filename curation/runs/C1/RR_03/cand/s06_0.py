"""Curator: diversity-based sampling using bigram score quantiles.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
import time
from collections import Counter
import numpy as np

try:
    from common import T
except Exception:
    T = 9_830_400

MAX_WORDS = 1500
MIN_CHARS = 200
MAX_NONASCII_LETTER_FRAC = 0.05
MAX_TOP_SHINGLE_FRAC = 0.05
SPAM_PATTERNS = ("discount code", "coupon", "free shipping", "best price guaranteed")

def _words(text):
    return text.lower().split()

def _bigram_set(texts):
    B = set()
    for t in texts:
        w = _words(t)[:MAX_WORDS + 1]
        B.update(zip(w, w[1:]))
    return B

def _score_words(words, B):
    w = words[:MAX_WORDS]
    n = len(w) - 1
    if n <= 0: return 0.0
    hits = sum(1 for b in zip(w, w[1:]) if b in B)
    return hits / n

def _calibrate_tau(target_sample):
    n = len(target_sample)
    if n < 2: return 0.0
    half = n // 2
    a, b = target_sample[:half], target_sample[half:]
    scores = []
    for build, held in ((a, b), (b, a)):
        B = _bigram_set(build)
        for t in held:
            w = _words(t)
            if len(w) >= 2:
                scores.append(_score_words(w, B))
    if not scores: return 0.0
    return 0.8 * float(np.percentile(np.asarray(scores), 5))

def _nonascii_letter_frac(text):
    n = len(text)
    if n == 0: return 0.0
    non_ascii = n - len(text.encode("ascii", "ignore"))
    if non_ascii > MAX_NONASCII_LETTER_FRAC * n:
        cnt = sum(1 for c in text if ord(c) > 127 and c.isalpha())
        return cnt / n
    return 0.0

def _repetitive(words):
    n = len(words) - 4
    if n <= 0: return False
    shingles = list(zip(words, words[1:], words[2:], words[3:], words[4:]))
    if not shingles: return False
    top_count = Counter(shingles).most_common(1)[0][1]
    return top_count >= 3 and top_count > MAX_TOP_SHINGLE_FRAC * n

def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    target_sample = [t for t in target_sample if isinstance(t, str)]
    B = _bigram_set(target_sample)
    tau = _calibrate_tau(target_sample)

    hdr_frac = sum(1 for t in target_sample if t.lstrip().startswith("= ")) / len(target_sample) if target_sample else 0.0
    require_header = hdr_frac >= 0.9

    # Phase 1: Filtering and Scoring
    scored_docs = []
    seen_full = set()
    seen_prefix = set()

    for i, text in enumerate(docs):
        if not isinstance(text, str) or len(text) < MIN_CHARS: continue
        if require_header and not text.lstrip().startswith("= "): continue
        if _nonascii_letter_frac(text) > MAX_NONASCII_LETTER_FRAC: continue
        
        low = text.lower()
        if any(p in low for p in SPAM_PATTERNS): continue

        words = low.split()
        score = _score_words(words, B)
        if score < tau: continue
        if _repetitive(words): continue

        norm = " ".join(words)
        prefix = norm[:300]
        if norm in seen_full or prefix in seen_prefix: continue
        
        seen_full.add(norm)
        seen_prefix.add(prefix)
        scored_docs.append((i, score))

    if not scored_docs:
        return list(range(len(docs)))

    # Phase 2: Diversity Sampling via Quantiles
    # Sort by score to determine quantiles
    scored_docs.sort(key=lambda x: x[1])
    scores = [x[1] for x in scored_docs]
    
    # Use 5 quantiles to ensure diversity
    num_quantiles = 5
    indices_by_quantile = [[] for _ in range(num_quantiles)]
    
    # Group indices into 5 equal-sized buckets based on score rank
    for rank, (idx, score) in enumerate(scored_docs):
        q_idx = min(rank // (len(scored_docs) // num_quantiles + 1), num_quantiles - 1)
        indices_by_quantile[q_idx].append(idx)

    # Selection logic: 
    # Try to pick from every bucket equally to ensure no score range dominates
    # but maintain enough volume to approach T.
    final_indices = []
    # Initial pass: take a portion from each bucket
    for bucket in indices_by_quantile:
        if bucket:
            # Take 20% from each bucket initially for diversity
            take_count = max(1, len(bucket) // 5)
            final_indices.extend(rng.sample(bucket, take_count))
    
    # Fill the rest of the budget by picking from buckets that are under-represented
    # relative to their size in the full ranked list.
    remaining_scored = []
    for bucket in indices_by_quantile:
        # This is simplified: just take remaining documents from buckets randomly
        # to reach a healthy volume.
        remaining_scored.extend(bucket)
    
    # Re-shuffle everything to maintain randomness
    # Since we want to reach T, if we have few docs, we cycle/repeat.
    # But the prompt asks to sample from the pool. We'll take a large enough subset.
    # To be safe and diverse, we take a balanced number from each quantile.
    final_indices = []
    bucket_counts = [len(b) for b in indices_by_quantile]
    avg_bucket_size = sum(bucket_counts) / num_quantiles
    
    for i in range(num_quantiles):
        # Take a balanced amount from each quantile based on the average bucket size
        # weighted by how large that bucket actually is.
        count = max(1, int(avg_bucket_size * (bucket_counts[i] / sum(bucket_counts) * 2)))
        # Ensure we don't take more than available
        count = min(count, len(indices_by_quantile[i]))
        final_indices.extend(rng.sample(indices_by_quantile[i], count))

    # If we have nothing, just take the first one
    if not final_indices:
        final_indices = [scored_docs[0][0]]

    rng.shuffle(final_indices)

    # Report
    est_tokens = sum(len(docs[i]) for i in final_indices) // 4
    print(f"[curate] pool={len(docs)} kept={len(final_indices)} kept_tokens~{est_tokens} tau={tau:.4f} time={time.time()-t0:.2f}s")
    
    return final_indices
