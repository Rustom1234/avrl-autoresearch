"""Curator: Hard Filter Pipeline.
Pipeline:
  1. Dedup and Junk Density check.
  2. Wikitext-Structure score.
  3. Damage Detection (median bigram coverage).
  4. Frequency-based ranking and top-40% selection for budget filling.
"""
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
MIN_ALPHA_FRAC = 0.70
MIN_ZLIB_RATIO = 0.2
TOPUP_FRACTION = 0.4

_ALPHA_SPACE = (string.ascii_letters + " ").encode("ascii")

def _is_junk(d):
    """Step 1: Junk Density check."""
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

def _wikitext_score(d):
    """Step 2: Wikitext-Structure score."""
    if d.lstrip().startswith("= "):
        return 1.0
    # Score based on presence of punctuation patterns
    cnt = d.count(" , ") + d.count(" . ")
    if len(d) == 0: return 0.0
    return min(1.0, cnt * 1000.0 / len(d))

def _get_bigram_coverage(d, target_bigrams):
    """Step 3: Bigram coverage calculation."""
    words = d.split()
    if len(words) < 2:
        return 0.0
    hits = 0
    for i in range(len(words) - 1):
        bg = (words[i], words[i+1])
        if bg in target_bigrams:
            hits += 1
    return hits / (len(words) - 1)

def _make_token_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return lambda texts: [len(x) for x in enc.encode_ordinary_batch(texts, num_threads=1)]
    except:
        return lambda texts: [max(1, len(t) // 4) for t in texts]

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0: return []

    # 1. Dedup and Junk Filter
    seen = set()
    initial_clean = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < MIN_CHARS:
            continue
        norm = " ".join(d.lower().split())[:2000]
        key = hashlib.md5(norm.encode("utf-8", "replace")).digest()
        if key in seen:
            continue
        seen.add(key)
        if not _is_junk(d):
            initial_clean.append(i)

    if not initial_clean:
        indices = list(range(n))
        rng.shuffle(indices)
        return indices

    # 2. Wikitext Score
    scores_wt = np.array([_wikitext_score(docs[i]) for i in initial_clean])
    
    # 3. Damage Detection
    target_bigrams = set()
    for t in target_sample:
        if isinstance(t, str):
            w = t.split()
            target_bigrams.update(zip(w, w[1:]))
    
    coverages = np.array([_get_bigram_coverage(docs[i], target_bigrams) for i in initial_clean])
    
    # Filter: Wikitext > 0.1 (simple threshold) and Coverage >= Median
    med_cov = float(np.median(coverages)) if len(coverages) > 0 else 0.0
    mask = (scores_wt > 0.1) & (coverages >= med_cov)
    survivors = [initial_clean[k] for k, m in enumerate(mask) if m]

    # 4. Frequency-based ranking and top-40% selection
    # Use the target sample to find "common" words to rank current docs
    # But the prompt asks for frequency-based ranking of the remaining pool
    # We'll use word frequency in survivors to rank
    word_counts = {}
    for idx in survivors:
        for word in docs[idx].split():
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Rank by total word counts (proxy for info density)
    doc_weights = np.array([sum(1 for w in docs[idx].split() if w in word_counts) for idx in survivors])
    
    # Get top 40% of the survivors
    num_to_keep = max(1, int(len(survivors) * TOPUP_FRACTION))
    top_indices = np.argsort(-doc_weights)[:num_to_keep]
    final_selection = [survivors[k] for k in top_indices]
    
    # If still under budget, add remaining survivors in order
    if sum(1 for _ in final_selection) < len(survivors):
        remaining = [survivors[k] for k in np.argsort(doc_weights) for k in [k] if k not in top_indices]
        # This logic is slightly simplified to ensure we don't double count
        # Let's do a cleaner final construction:
        sorted_survivors = [survivors[k] for k in np.argsort(-doc_weights)]
        final_selection = sorted_survivors[:num_to_keep]
        # Logic above is messy, let's rewrite the final block:
        
    # RE-WRITE FINAL SELECTION LOGIC
    sorted_survivors = [survivors[k] for k in np.argsort(-doc_weights)]
    final_selection = sorted_survivors[:max(1, int(len(survivors) * TOPUP_FRACTION))]
    
    # Token Check / Top-up
    count_tokens = _make_token_counter()
    total_tokens = 0
    # Use a simple first pass to see if we need more
    for idx in final_selection:
        total_tokens += count_tokens([docs[idx]])[0]
    
    if total_tokens < int(TOPUP_FRACTION * T):
        # Add from sorted_survivors that aren't in final_selection
        set_final = set(final_selection)
        for idx in sorted_survivors:
            if idx not in set_final:
                final_selection.append(idx)
                total_tokens += count_tokens([docs[idx]])[0]
                if total_tokens >= int(TOPUP_FRACTION * T):
                    break
    
    rng.shuffle(final_selection)
    return final_selection
