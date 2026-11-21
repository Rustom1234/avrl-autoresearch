import hashlib
import os
import random
import numpy as np
import tiktoken
import re
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

    # (2) Clean text for word-based TF-IDF
    def clean_text(text):
        # Remove special characters but keep basic punctuation/spacing
        return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()

    pool_texts = [clean_text(docs[i]) for i in valid_indices]
    target_texts = [clean_text(t) for t in targets]

    # Use word-based n-grams as requested
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        sublinear_tf=True,
        max_features=MAX_FEATURES,
        dtype=np.float32,
    )
    
    try:
        X_pool = vec.fit_transform(pool_texts)
        X_targets = vec.transform(target_texts)
    except ValueError:
        # Fallback if TF-IDF fails due to lack of vocabulary
        scores = np.zeros(len(valid_indices))

    # (3) Calculate Scores with Refined Quality Weighting
    # Weight = min(len/800, 1.5) * max(len/2000, 0.5)
    cos_sim = cosine_similarity(X_pool, X_targets) if 'X_pool' in locals() else np.zeros((len(valid_indices), len(targets)))
    max_sim = np.max(cos_sim, axis=1)
    
    weights = []
    for i in valid_indices:
        l = len(docs[i])
        w = min(l / 800.0, 1.5) * max(l / 2000.0, 0.5)
        weights.append(w)
    
    weights = np.array(weights)
    final_scores = max_sim * weights

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
            # If this doc alone doesn't fit, we stop 
            # (or could take smaller ones, but following previous structure)
            pass
            
        if total_tokens >= T:
            break

    if not kept and candidates:
        kept = [valid_indices[i] for i in candidates[:10]]

    # (4) Seeded shuffle
    rng.shuffle(kept)
    return kept
