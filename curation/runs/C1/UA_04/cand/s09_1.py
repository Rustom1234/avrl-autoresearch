import hashlib
import os
import random
import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion
from common import T

# Configuration
HEAD_CHARS = 2000
MIN_FEAT_CHARS = 200
FEAT_CHAR_BUDGET = 10_000_000
DEDUP_CHARS = 400
N_NEG = 1000  # Reduced for speed
MIN_CHARS = 150
MIN_UNIQUE_RATIO = 0.15
MIN_ALPHA_FRAC = 0.6
BUDGET = T

def _is_junk(doc):
    if len(doc) < MIN_CHARS:
        return True
    head = doc[:HEAD_CHARS]
    words = head.lower().split()
    if not words:
        return True
    if len(set(words)) / len(words) < MIN_UNIQUE_RATIO:
        return True
    good = sum(1 for c in head if c.isalpha() or c.isspace())
    if good / len(head) < MIN_ALPHA_FRAC:
        return True
    return False

def _dedup_key(doc):
    # Use a very small prefix to stay within time limits
    norm = " ".join(doc[:DEDUP_CHARS].lower().split())
    return hashlib.md5(norm.encode("utf-8", "ignore")).hexdigest()

def _features(all_head):
    # Reduced max_features significantly to ensure speed within 180s
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 3), sublinear_tf=True,
        max_features=30000, dtype=np.float32,
    )
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        max_features=30000, dtype=np.float32,
    )
    X = FeatureUnion([("char", char_vec), ("word", word_vec)]).fit_transform(all_head)
    return X

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []
    m = len(target_sample)

    scores = np.zeros(n, dtype=np.float64)
    if m > 0 and n > 0:
        try:
            # Dynamic budget based on docs count to avoid OOM or Timeout
            feat_chars = min(HEAD_CHARS, max(MIN_FEAT_CHARS, FEAT_CHAR_BUDGET // (n + m)))
            all_head = [d[:feat_chars] for d in docs] + [d[:feat_chars] for d in target_sample]
            
            X = _features(all_head)
            X_pool = X[:n]

            neg_idx = rng.sample(range(n), min(N_NEG, n))
            train_rows = np.concatenate([np.arange(n, n + m), neg_idx])
            X_train = X[train_rows]
            y_train = np.concatenate([np.ones(m), np.zeros(len(neg_idx))])
            
            clf = MLPClassifier(
                hidden_layer_sizes=(128,), 
                activation='relu', 
                solver='sgd', # SGD is faster than Adam for this scale
                max_iter=100, 
                random_state=seed
            )
            clf.fit(X_train, y_train)
            scores = np.asarray(clf.decision_function(X_pool), dtype=np.float64)
        except:
            pass

    # Dedup + Junk Filter
    seen = set()
    survivors = []
    for i, d in enumerate(docs):
        if _is_junk(d):
            continue
        key = _dedup_key(d)
        if key in seen:
            continue
        seen.add(key)
        survivors.append(i)

    if not survivors:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # Ranking
    survivors = np.array(survivors, dtype=np.int64)
    order = survivors[np.argsort(-scores[survivors], kind="stable")]

    # Token counting (optimized loop)
    enc = tiktoken.get_encoding("gpt2")
    selected = []
    current_tokens = 0
    
    for idx in order:
        # Estimate tokens to avoid heavy encoding inside the loop
        # Using a heuristic: 1 word ~ 1.3 tokens, or just encode if budget allows
        # To stay safe and fast, encode only when necessary or encode all small chunks
        t_count = len(enc.encode(docs[idx]))
        selected.append(int(idx))
        current_tokens += t_count
        if current_tokens >= BUDGET:
            break
            
    # If we didn't reach budget, cycle the survivors
    if current_tokens < BUDGET and len(selected) < len(order):
        remaining = BUDGET - current_tokens
        # Approximate how many more we need (rough estimate to avoid encoding overhead)
        needed = int(remaining / (len(docs[order[0]]) / 4)) 
        for i in range(needed):
            selected.append(int(order[i % len(order)]))

    rng.shuffle(selected)
    return selected
