import os
import re
import time
import random
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from common import T

# Constraints: 180s limit, 2T tokens pool. 
# The SIGKILL error indicates a Memory Error (OOM) because X_pool 
# for 2T tokens with char_wb (1,4) is too large for memory.
# Strategy: Use a more memory-efficient word-level TF-IDF and 
# a sampling-based approach to keep the matrix size manageable.

MAX_CHARS = 2000
MIN_CHARS = 100
MIN_UNIQ_RATIO = 0.2
MAX_BAD_FRAC = 0.4

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
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Dedupe
    seen = set()
    uniq_idx = []
    for i, d in enumerate(docs):
        h = _norm_hash(d)
        if h not in seen:
            seen.add(h)
            uniq_idx.append(i)

    # 2. Basic Quality Filter
    keep = []
    for i in uniq_idx:
        d = docs[i]
        if len(d) < MIN_CHARS: continue
        if _bad_frac(d) > MAX_BAD_FRAC: continue
        if _uniq_ratio(d) < MIN_UNIQ_RATIO: continue
        keep.append(i)
    
    if not keep:
        indices = list(range(len(docs)))
        rng.shuffle(indices)
        return indices[:int(T/4)]

    # 3. Memory-Efficient Ranking
    # To avoid OOM with char_wb, we use word-level and limit features.
    # We also sample the pool if it's too large to process in one matrix.
    vectorizer = TfidfVectorizer(
        analyzer="word",
        max_features=10000,
        dtype=np.float32,
    )
    
    target_texts = [t[:MAX_CHARS] for t in target_sample if isinstance(t, str)]
    if not target_texts:
        indices = list(keep)
        rng.shuffle(indices)
        return indices

    X_target = vectorizer.fit_transform(target_texts)
    centroid = np.asarray(X_target.mean(axis=0))
    
    # Memory Management: Process pool in chunks or sample to find best docs
    # If pool is huge, we take a representative sample to rank.
    num_to_score = min(len(keep), 10000)
    sample_indices = rng.sample(keep, num_to_score)
    pool_texts = [docs[i][:MAX_CHARS] for i in sample_indices]
    
    X_pool = vectorizer.transform(pool_texts)
    
    # Dot product for similarity
    dot = np.dot(X_pool, centroid)
    norm_pool = np.linalg.norm(X_pool, axis=1)
    norm_centroid = np.linalg.norm(centroid)
    
    similarities = np.divide(dot, norm_pool * norm_centroid, 
                               out=np.zeros_like(dot), 
                               where=(norm_pool * norm_centroid) != 0)
    
    # Rank the sampled indices
    ranked_sampled = sorted(range(num_to_score), key=lambda i: similarities[i], reverse=True)
    best_docs = [sample_indices[i] for i in ranked_sampled]
    
    # 4. Final Selection
    # Use the "best_docs" as a high-quality seed to fill the budget
    final_selection = []
    current_tokens = 0
    target_tokens = T * 1.01
    
    # Use a set to avoid duplicates if a doc was in the sample
    selected_set = set()
    
    # First, add top high-quality samples
    for idx in best_docs:
        if idx not in selected_set:
            final_selection.append(idx)
            selected_set.add(idx)
            current_tokens += len(docs[idx]) // 4
            if current_tokens >= target_tokens:
                break
                
    # If we still need more, add from the rest of the 'keep' pool
    remaining = [i for i in keep if i not in selected_set]
    rng.shuffle(remaining)
    for idx in remaining:
        if current_tokens >= target_tokens:
            break
        final_selection.append(idx)
        current_tokens += len(docs[idx]) // 4
        
    return final_selection
