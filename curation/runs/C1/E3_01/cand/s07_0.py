"""Curator: Pointwise Cosine Similarity + Density Score selection.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import hashlib
import numpy as np
import tiktoken
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.sparse import vstack

from common import T

HEAD = 3000          # characters of each document used for features
PREFIX = 200         # characters used for the near-duplicate prefix check

def _dedupe(docs):
    """First occurrence of each normalised text; also drop repeated 200-char prefixes."""
    seen_hash, seen_prefix, kept = set(), set(), []
    for i, d in enumerate(docs):
        norm = " ".join(d.lower().split())
        if not norm:
            continue  # empty or whitespace-only documents carry no tokens
        h = hashlib.md5(norm.encode("utf-8", "ignore")).digest()
        if h in seen_hash:
            continue
        p = d[:PREFIX]
        if p in seen_prefix:
            continue
        seen_hash.add(h)
        seen_prefix.add(p)
        kept.append(i)
    return kept

def _chunks(text, size=HEAD):
    out = [text[i:i + size] for i in range(0, len(text), size)]
    return [c for c in out if c.strip()]

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # 1. Dedupe.
    kept_indices = _dedupe(docs)
    n = len(kept_indices)
    if n == 0:
        return list(range(len(docs)))
    
    heads = [docs[i][:HEAD] for i in kept_indices]

    # 2. Feature Extraction.
    # Use HashingVectorizer for both pool and target sample.
    vec = HashingVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+",
                            lowercase=False, n_features=2**18, binary=True, norm="l2")
    X_pool = vec.transform(heads)
    
    target_chunks = []
    for t in target_sample:
        target_chunks.extend(_chunks(t))
    X_tgt = vec.transform(target_chunks)

    # 3. Pointwise Cosine Similarity.
    # For each pool document, calculate the mean similarity to all target chunks.
    # Since vectors are normalized in HashingVectorizer(norm="l2"), 
    # dot product is the cosine similarity.
    # X_pool is (n_pool, n_feats), X_tgt is (n_tgt_chunks, n_feats)
    # Dot product: (n_pool, n_tgt_chunks)
    sim_matrix = X_pool.dot(X_tgt.T).toarray()
    # Mean similarity across all target chunks for each document
    mean_sim = np.mean(sim_matrix, axis=1)

    # 4. Density Score.
    # Reward documents with a high ratio of unique, high-frequency target tokens 
    # to total token count.
    # Identify "target tokens" as those appearing in target_sample.
    target_words = set()
    for t in target_sample:
        target_words.update(t.lower().split())
    
    # We use the "heads" to calculate density.
    density = np.zeros(n)
    for i, h in enumerate(heads):
        words = h.lower().split()
        if not words:
            density[i] = 0
            continue
        # Intersection of words in head and target words
        target_hits = [w for w in words if w in target_words]
        # Count unique target hits (unique tokens from target sample appearing in this doc)
        unique_target_hits = len(set(target_hits))
        density[i] = unique_target_hits / len(words)

    # 5. Combine Scores.
    # Pointwise score = mean_sim + density_contribution.
    # We scale density to be comparable to similarity values.
    final_scores = mean_sim + (density * 0.5)
    
    # Rank documents.
    ranked_indices = np.argsort(-final_scores, kind="stable")
    
    # 6. Selection.
    enc = tiktoken.get_encoding("gpt2")
    selected = []
    total_tokens = 0
    
    # We iterate through the ranked indices and pick until we hit T.
    for idx_in_ranked in ranked_indices:
        doc_idx = kept_indices[idx_in_ranked]
        doc_text = docs[doc_idx]
        
        # Encode to count tokens accurately.
        tokens = enc.encode_ordinary_batch([doc_text])[0]
        
        # If this doc alone exceeds T, we need to handle it. 
        # However, T is large and docs are short, so this is rare.
        # The problem states we stop after T tokens.
        if total_tokens + len(tokens) <= T:
            selected.append(doc_idx)
            total_tokens += len(tokens)
        else:
            # If this doc would put us over, but we haven't reached T,
            # the logic "stops after T" means we include only the part of the doc
            # that completes the T requirement. Since we can't split docs easily 
            # in this selection step without complexity, we add it and let the 
            # trainer's "stop after T" logic handle the truncation.
            # But to be safe with the "stop after T" rule, we check if total_tokens < T.
            if total_tokens < T:
                selected.append(doc_idx)
                total_tokens += len(tokens)
            else:
                break
                
    # 7. Final Shuffle.
    # The trainer stops at T, but the order matters for the "first" T tokens.
    # However, the instruction says "Returns an ordered list... trainer reads front to back".
    # To maximize quality, we want the best documents first. 
    # A random shuffle at the very end would destroy the ranking.
    # We keep the ranking order to ensure highest quality is seen first.
    # The prompt asks to "Replace... while incorporating...", and the original
    # code shuffled at the end. If the trainer is "fixed", we assume the 
    # ranking is our primary tool. To respect the original structure:
    # final_shuffle = random.Random(seed).shuffle(selected)
    # But the prompt implies the ranking is the optimization.
    # Given "order: seeded shuffle" was in original, but the logic is 
    # to improve the data, we will keep the sorted order.
    
    # Re-reading: "select(docs, target_sample) -> ordered list of indices".
    # If we shuffle, we lose the clustering of high-quality text.
    # We'll maintain the sorted order.
    
    return selected
