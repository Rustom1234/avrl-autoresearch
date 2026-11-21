import os
import random
import numpy as np
import tiktoken
from common import T

def select(docs, target_sample):
    """
    Curates a selection of documents based on cosine similarity to a
    consensus vector derived from the target_sample.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    encoding = tiktoken.get_encoding("gpt2")
    
    # 1. Calculate consensus vector for target_sample
    # We average the token indices (positions) to create a "centroid"
    # of the target domain's distribution.
    target_tokens = []
    for doc in target_sample:
        target_tokens.extend(encoding.encode(doc))
    
    if not target_tokens:
        return list(range(len(docs)))
    
    # We treat the indices as values to form a consensus "content" vector
    # However, the prompt asks for "EF-quarter-encoder token indices".
    # In the absence of an explicit EF-quarter-encoder, we use the 
    # normalized token ID distribution as a proxy for the content vector.
    target_counts = np.zeros(100000, dtype=np.float32) # Large enough buffer
    for token in target_tokens:
        if token < 100000:
            target_counts[token] += 1
            
    consensus_vector = target_counts / np.sum(target_counts)
    
    # 2. Calculate similarity for every doc in the pool
    doc_weights = []
    for doc in docs:
        tokens = encoding.encode(doc)
        if not tokens:
            doc_weights.append(0.0)
            continue
            
        doc_counts = np.zeros(100000, dtype=np.float32)
        for token in tokens:
            if token < 100000:
                doc_counts[token] += 1
        
        doc_vec = doc_counts / np.sum(doc_counts)
        
        # Cosine Similarity: (A . B) / (||A|| * ||B||)
        # Since they are normalized as counts, dot product is used
        sim = np.dot(doc_vec, consensus_vector)
        doc_weights.append(sim)

    # 3. Apply weighting formula: 1.0 / (1 + doc_prob_index / n_target_docs)
    # The prompt specifies using the similarity score as the basis for selection.
    # We map the similarity score to the doc_prob_index logic.
    # Let doc_prob_index be a value scaled by similarity.
    n_target_docs = len(target_sample)
    final_weights = []
    for i, weight in enumerate(doc_weights):
        # Interpreting "similarity score" as the primary influencer of 
        # the probability index. We treat higher similarity as lower prob_index.
        # Formula: 1.0 / (1 + doc_prob_index / n_target_docs)
        # To maximize weight, we minimize doc_prob_index.
        # We map similarity (0 to 1) to index (0 to n_target_docs).
        doc_prob_index = (1.0 - weight) * n_target_docs
        w = 1.0 / (1.0 + (doc_prob_index / n_target_docs))
        final_weights.append(w)

    # 4. Sort indices by weight descending
    indices = np.argsort(final_weights)[::-1]
    return indices.tolist()
