import hashlib
import os
import random
import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import T

MIN_CHARS = 500
MAX_FEATURES = 40000

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

    # (2) Char n-gram TF-IDF with requested parameters
    # Use n-gram range (3, 5) as requested
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        max_features=MAX_FEATURES,
        dtype=np.float32,
    )
    
    pool_texts = [docs[i] for i in valid_indices]
    
    # Fit on both to ensure same vocabulary, though pool is larger
    X_pool = vec.fit_transform(pool_texts)
    X_targets = vec.transform(targets)

    # (3) Compute Centroid Similarity
    # Calculate the mean vector of the target sample (the centroid)
    if X_targets.shape[0] > 0:
        centroid = np.asarray(X_targets.mean(axis=0))
        # Reshape to 2D for cosine_similarity
        centroid_reshaped = centroid.reshape(1, -1)
        centroid_sims = cosine_similarity(X_pool, centroid_reshaped).flatten()
    else:
        centroid_sims = np.zeros(len(valid_indices))

    # (4) Density Score: ratio of unique n-grams to total characters
    # We use the same analyzer/logic as the vectorizer to define "n-grams"
    # To keep it efficient within 180s, we use a simplified unique count
    # of substrings of lengths 3, 4, and 5
    densities = []
    for idx in valid_indices:
        d = docs[idx]
        # Simple heuristic for n-gram density to maintain performance
        # unique_ngrams / total_chars
        # We sample a few points to estimate density if text is huge
        unique_ng = set()
        sample_text = d[:2000] # Use first 2000 chars for density check
        for n in [3, 4, 5]:
            for i in range(len(sample_text) - n + 1):
                unique_ng.add(sample_text[i:i+n])
        densities.append(len(unique_ng) / len(d))

    # (5) Final Scoring
    # Score = Similarity * Density
    final_scores = centroid_sims * np.array(densities)
    
    # Rank and select
    order = np.argsort(-final_scores, kind="stable")
    candidates = [int(r) for r in order]
    
    enc = tiktoken.get_encoding("gpt2")
    kept = []
    total_tokens = 0
    
    for i in candidates:
        idx = valid_indices[i]
        tk_count = len(enc.encode_ordinary(docs[idx]))
        
        if total_tokens + tk_count <= T:
            kept.append(idx)
            total_tokens += tk_count
        else:
            break
            
    if not kept and candidates:
        kept = [valid_indices[i] for i in candidates[:10]]

    rng.shuffle(kept)
    return kept
