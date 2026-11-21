"""Target-similarity curator: Word Filtering + Dedup + Junk removal.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. TF-IDF over target_sample to identify high-value words.
  2. For each pool doc, score based on the weighted intersection of words with target high-value words.
  3. Dedup by hash of whitespace-normalised, lowercased first 500 chars (keep first).
  4. Drop junk (too short, low unique-word ratio, low alphabetic+space fraction).
  5. Rank survivors by score, take best until the cumulative gpt2 token count reaches T.
  6. Seeded shuffle of the selected indices.
"""
import hashlib
import os
import random
import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from common import T

HEAD_CHARS = 3000
DEDUP_CHARS = 500
MIN_CHARS = 200
MIN_UNIQUE_RATIO = 0.2
MIN_ALPHA_FRAC = 0.7
BUDGET = T
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

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    n = len(docs)
    if n == 0:
        return []

    # 1. Extract high-value words from target_sample
    # Use a word-based TF-IDF to get weights
    try:
        tfidf_target = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 1), 
            min_df=1, max_features=5000
        ).fit(target_sample)
        
        # Map words to their TF-IDF weights
        feature_names = tfidf_target.get_feature_names_out()
        weights = tfidf_target.transform(target_sample).mean(axis=0).A1
        # Get top 2000 weights
        top_indices = np.argsort(-weights)[:2000]
        high_value_vocab = dict(zip(feature_names[top_indices], weights[top_indices]))
    except Exception:
        high_value_vocab = {}

    # 2. Score pool documents
    scores = np.zeros(n, dtype=np.float64)
    if high_value_vocab:
        for i, doc in enumerate(docs):
            words = doc.lower().split()
            score = 0.0
            count = 0
            for w in words:
                if w in high_value_vocab:
                    score += high_value_vocab[w]
                    count += 1
            # Score is weighted sum of weights divided by total words (or normalized)
            scores[i] = score / (len(words) + 1)

    # 3 & 4. Dedup + Junk Filter
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

    # 5. Rank survivors by score
    survivors = np.asarray(survivors, dtype=np.int64)
    # Sort indices of survivors by their scores in the original docs array
    sorted_survivor_indices = survivors[np.argsort(-scores[survivors], kind="stable")]

    enc = tiktoken.get_encoding("gpt2")
    selected = []
    total_tokens = 0
    
    for i, doc_idx in enumerate(sorted_survivor_indices):
        doc = docs[doc_idx]
        # Estimate tokens to avoid excessive calls, then verify
        tokens = enc.encode_ordinary_batch([doc])[0]
        num_tokens = len(tokens)
        
        selected.append(int(doc_idx))
        total_tokens += num_tokens
        
        if total_tokens >= BUDGET:
            break

    # 6. Seeded shuffle
    random.Random(seed).shuffle(selected)
    return selected
