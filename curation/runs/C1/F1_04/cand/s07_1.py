import os
import re
import time
import random
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from common import T

# Parameters
MAX_CHARS = 2000
MIN_CHARS = 200
MAX_FEATURES = 20000
FILL = 1.01

_GOOD_RE = re.compile(r"[A-Za-z\s]+")

def _norm_hash(text):
    norm = " ".join(text.lower().split())
    return hashlib.md5(norm.encode("utf-8", "ignore")).digest()

def _bad_frac(text):
    s = text[:MAX_CHARS]
    if not s: return 1.0
    return len(_GOOD_RE.sub("", s)) / len(s)

def _uniq_ratio(text):
    words = text[:MAX_CHARS].lower().split()
    if not words: return 0.0
    return len(set(words)) / len(words)

def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Fast Deduplication
    seen_hashes = set()
    unique_indices = []
    for i, d in enumerate(docs):
        h = _norm_hash(d)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(i)

    # 2. Fast Quality Filter
    keep = []
    for i in unique_indices:
        d = docs[i]
        if len(d) < MIN_CHARS: continue
        if _bad_frac(d) > 0.4: continue
        if _uniq_ratio(d) < 0.2: continue
        keep.append(i)
    
    if not keep:
        res = list(unique_indices)
        rng.shuffle(res)
        return res

    # 3. Vectorized Scoring (Memory/Time Efficient)
    # Use a smaller feature set to ensure it fits in memory/time limits
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=MAX_FEATURES,
        dtype=np.float32
    )

    targets = [t[:MAX_CHARS] for t in target_sample if len(t) > 10]
    if targets:
        X_targets = vectorizer.fit_transform(targets)
        centroid = np.mean(X_targets, axis=0)
        
        # Process in chunks to prevent memory spikes
        scores = []
        chunk_size = 5000
        for i in range(0, len(keep), chunk_size):
            chunk_idx = keep[i : i + chunk_size]
            X_chunk = vectorizer.transform([docs[idx][:MAX_CHARS] for idx in chunk_idx])
            scores.extend(X_chunk.dot(centroid).tolist())
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        ranked_docs = [keep[idx] for idx in sorted_indices]
    else:
        ranked_docs = keep
        rng.shuffle(ranked_docs)

    # 4. Selection until T
    needed_tokens = T * FILL
    current_tokens = 0
    selection = []
    
    # Use a conservative estimate (1 char ≈ 0.25 tokens)
    # but cap the max tokens a single doc can contribute to prevent skew
    for idx in ranked_docs:
        d = docs[idx]
        # Fast heuristic for tokens
        tokens_est = len(d) // 4
        selection.append(idx)
        current_tokens += tokens_est
        if current_tokens >= needed_tokens:
            break
            
    # Fallback if no docs were added
    if not selection:
        selection = ranked_docs[:min(len(ranked_docs), 1000)]

    rng.shuffle(selection)
    return selection
