import hashlib
import os
import random
import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import T

MIN_CHARS = 500
MAX_FEATURES = 30000

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # (1) Length filter + exact dedup
    seen = set()
    valid_indices = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < MIN_CHARS:
            continue
        norm = " ".join(d.split())
        if len(norm) < MIN_CHARS:
            continue
        h = hashlib.blake2b(norm.encode("utf-8", "ignore"), digest_size=16).digest()
        if h in seen:
            continue
        seen.add(h)
        valid_indices.append(i)

    if not valid_indices:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    targets = [t for t in target_sample if isinstance(t, str) and t.strip()]
    if not targets:
        out = list(valid_indices)
        rng.shuffle(out)
        return out

    # (2) Char n-gram TF-IDF
    pool_texts = [docs[i] for i in valid_indices]
    
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 3),
        sublinear_tf=True,
        max_features=MAX_FEATURES,
        dtype=np.float32,
    )
    
    X_pool = vec.fit_transform(pool_texts)
    X_targets = vec.transform(targets)

    # (3) Calculate Scores: Mean of Top 5 Similarities
    # cos_sim shape: (num_pool_docs, num_targets)
    cos_sim = cosine_similarity(X_pool, X_targets)
    
    # Sort each row to find top 5
    # We use a partition/sort approach to get top 5 scores per doc
    sorted_sim = np.sort(cos_sim, axis=1)
    # Take the 5 largest scores (last 5 columns)
    top_5_sim = sorted_sim[:, -5:]
    # If target sample is smaller than 5, use all available
    if cos_sim.shape[1] < 5:
        top_5_sim = sorted_sim
    
    mean_top_sim = np.mean(top_5_sim, axis=1)
    
    # Filter: remove documents with max similarity score below 0.1
    # The prompt implies we should check the max similarity score for the noise pruning.
    max_sim_scores = np.max(cos_sim, axis=1)
    
    # Weighting: (len(doc) / 1500) capped at 1.8
    weights = np.array([min(len(docs[i]) / 1500.0, 1.8) for i in valid_indices])
    final_scores = mean_top_sim * weights

    # Filter indices based on max_sim_scores > 0.1
    qualified_mask = max_sim_scores > 0.1
    qualified_indices_in_valid = np.where(qualified_mask)[0]
    
    # Rank the qualified documents
    qualified_scores = final_scores[qualified_indices_in_valid]
    order = np.argsort(-qualified_scores, kind="stable")
    
    candidates = [valid_indices[i] for i in qualified_indices_in_valid[order]]
    
    if not candidates:
        # Fallback if pruning is too aggressive
        candidates = [valid_indices[i] for i in np.argsort(-final_scores, kind="stable")]

    enc = tiktoken.get_encoding("gpt2")
    kept = []
    total_tokens = 0
    
    for idx in candidates:
        tk_count = len(enc.encode_ordinary(docs[idx]))
        if total_tokens + tk_count <= T:
            kept.append(idx)
            total_tokens += tk_count
        else:
            # Try to add part of the document if we have space
            # (Optional, but ensures we get as close to T as possible)
            break
            
        if total_tokens >= T:
            break

    if not kept and candidates:
        kept = [candidates[0]]

    rng.shuffle(kept)
    return kept
