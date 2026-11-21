import os
import random
import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from common import T

def select(docs, target_sample):
    """
    Curates a selection of documents from docs to optimize for target_sample.
    Embeds the target_sample into a GPT-2 space and uses cosine similarity
    of top tokens to rank and reweight the selection of pool documents.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Helper to get centroids of target_sample documents
    # We represent target_sample as a collection of token IDs
    target_tokens = []
    for doc in target_sample:
        target_tokens.extend(enc.encode(doc))
    
    # To handle the 'scatter search' requirement efficiently within 180s,
    # we compute a summary representation (centroid) of the target_sample
    # to use as our reference vector for similarity.
    if not target_tokens:
        return list(range(len(docs)))
    
    target_centroid = np.mean([i for i in target_tokens], dtype=np.float64)
    # Note: Since tokens are IDs, we treat the distribution of IDs as the vector.
    # To perform cosine similarity on these, we treat the frequency of token IDs
    # in the target_sample as the embedding.
    
    unique_tokens = list(set(target_tokens))
    target_vec = np.zeros(max(unique_tokens) + 1)
    for t in target_tokens:
        target_vec[t] += 1
    target_vec = (target_vec / np.linalg.norm(target_vec)).astype(np.float64)

    doc_weights = []
    for doc in docs:
        tokens = enc.encode(doc)
        if not tokens:
            doc_weights.append(0.0)
            continue
            
        # Calculate doc vector from top tokens (to capture dominant themes)
        # We use all tokens to ensure we don't miss sparse but important features
        doc_vec = np.zeros(max(unique_tokens) + 1)
        for t in tokens:
            if t < len(doc_vec):
                doc_vec[t] += 1
        
        norm = np.linalg.norm(doc_vec)
        if norm == 0:
            doc_weights.append(0.0)
        else:
            doc_vec = doc_vec / norm
            # Cosine similarity
            sim = np.dot(doc_vec, target_vec)
            doc_weights.append(max(0.0, sim))

    # Re-weighting and selection
    # We want to prioritize high similarity scores
    weights = np.array(doc_weights, dtype=np.float64)
    # Add a small epsilon to avoid division by zero and ensure all docs have a non-zero chance
    weights = weights + 1e-6
    
    # The trainer cycles the list until T tokens are reached.
    # We should return the best documents first.
    indices = np.argsort(weights)[::-1]
    
    # Ensure we return a substantial amount of indices to cover T tokens
    # Since we can cycle, we return the sorted best.
    return indices.tolist()
