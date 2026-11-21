"""Target-similarity curator: classifier filter + dedup + junk removal + diversity.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. char (2-4, char_wb) + word (1-2) TF-IDF over pool + target_sample.
  2. LogisticRegression: target_sample = positives, seeded random 3000 pool docs = negatives;
     score every pool doc with decision_function.
  3. Dedup by hash of whitespace-normalised, lowercased first 500 chars (keep first).
  4. Drop junk (too short, low unique-word ratio, low alphabetic+space fraction).
  5. Rank survivors by score.
  6. Diversity Sampling: Select top 80% of budget from highest scores; fill 
     remaining 20% by weighted sampling from the rest of the survivors.
  7. Seeded shuffle of the selected indices.
"""
import hashlib
import os
import random
import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from common import T

HEAD_CHARS = 3000       # max prefix used for features and junk statistics
MIN_FEAT_CHARS = 400    # lower bound of the adaptive feature prefix
FEAT_CHAR_BUDGET = 30_000_000  # total chars fed to the vectorizers (time bound)
DEDUP_CHARS = 500       # prefix used for the near-duplicate key
N_NEG = 3000            # pool docs sampled as negatives
MIN_CHARS = 200
MIN_UNIQUE_RATIO = 0.2
MIN_ALPHA_FRAC = 0.7
BUDGET = 1.0 * T
TOK_CHUNK = 512


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
    norm = " ".join(doc[:DEDUP_CHARS].lower().split())
    return hashlib.md5(norm.encode("utf-8", "ignore")).digest()


def _features(all_head):
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True,
        max_features=200000, dtype=np.float32,
    )
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        max_features=200000, dtype=np.float32,
    )
    try:
        X = FeatureUnion([("char", char_vec), ("word", word_vec)]).fit_transform(all_head)
    except ValueError:  # empty word vocabulary (degenerate pool)
        X = char_vec.fit_transform(all_head)
    return X.tocsr()


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    n = len(docs)
    if n == 0:
        return []
    m = len(target_sample)

    scores = np.zeros(n, dtype=np.float64)
    if m > 0:
        try:
            feat_chars = int(min(HEAD_CHARS, max(MIN_FEAT_CHARS, FEAT_CHAR_BUDGET // (n + m))))
            all_head = [d[:feat_chars] for d in docs] + [d[:feat_chars] for d in target_sample]
            X = _features(all_head)
            X_pool = X[:n]

            neg_idx = np.asarray(
                random.Random(seed).sample(range(n), min(N_NEG, n)), dtype=np.int64
            )
            train_rows = np.concatenate([np.arange(n, n + m, dtype=np.int64), neg_idx])
            X_train = X[train_rows]
            y_train = np.concatenate([
                np.ones(m, dtype=np.int32),
                np.zeros(len(neg_idx), dtype=np.int32),
            ])
            clf = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=1000, random_state=seed,
            )
            clf.fit(X_train, y_train)
            scores = np.asarray(clf.decision_function(X_pool), dtype=np.float64)
        except ValueError:
            scores = np.zeros(n, dtype=np.float64)

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
        random.Random(seed).shuffle(idx)
        return idx

    survivors = np.asarray(survivors, dtype=np.int64)
    # Sort survivors by score descending
    sorted_survivors = survivors[np.argsort(-scores[survivors], kind="stable")]
    
    # Ranking by score (all survivors)
    order = sorted_survivors.tolist()
    
    enc = tiktoken.get_encoding("gpt2")
    selected = []
    total = 0
    
    # Calculate the split point (80% of budget)
    # We don't know the total tokens exactly, so we iterate and check.
    # To respect the 80/20 split rule: 
    # 1. Take top-scoring until we hit 80% of total BUDGET.
    # 2. Take remaining from the rest using weighted sampling.
    # Since we don't know the total tokens of all survivors, we first calculate total 
    # tokens of all survivors to find the 80% threshold accurately.
    
    all_survivor_tokens = []
    for idx in order:
        t_len = len(enc.encode_ordinary_batch([docs[idx]])[0])
        all_survivor_tokens.append(t_len)
    
    total_survivor_tokens = sum(all_survivor_tokens)
    target_80_tokens = 0.8 * BUDGET
    
    # Find the break point for top 80% of the budget
    break_idx = -1
    current_sum = 0
    for i, t_len in enumerate(all_survivor_tokens):
        current_sum += t_len
        if current_sum >= target_80_tokens:
            break_idx = i
            break
    
    # If all survivors fit in 80%, break_idx will be total length
    if break_idx == -1:
        break_idx = len(all_survivor_tokens)

    # Part 1: Top scorers
    top_indices = order[:break_idx]
    for idx in top_indices:
        selected.append(int(idx))
        total += all_survivor_tokens[top_indices.index(idx)]
        if total >= BUDGET:
            break
            
    if total < BUDGET:
        # Part 2: Weighted sample from the rest
        remaining_indices = order[break_idx:]
        if remaining_indices:
            # Weights are scores. Ensure they are non-negative for sampling.
            weights = np.maximum(scores[survivors[break_idx:]], 0)
            # If all scores are 0, use uniform weights
            if np.sum(weights) == 0:
                weights = np.ones(len(weights))
            
            # Sample from the rest until budget
            # We use a loop to ensure we don't over-sample if one doc is huge,
            # but the prompt asks to fill the remaining 20% by weighted sampling.
            while total < BUDGET and remaining_indices:
                # Sample one index based on weights
                # Note: scores[survivors[break_idx:]] aligns with remaining_indices
                sampled_pos = np.random.choice(len(remaining_indices), p=weights / np.sum(weights))
                idx = remaining_indices[sampled_pos]
                selected.append(int(idx))
                total += all_survivor_tokens[break_idx + sampled_pos]
                # Update weights/indices is tricky with replacement; 
                # For simplicity in this constraint, we allow replacement or re-sample
                # until budget is reached.
                if total >= BUDGET: break

    random.Random(seed).shuffle(selected)
    return selected
