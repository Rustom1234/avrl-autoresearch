import os
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from common import T
import tiktoken

def select(docs, target_sample):
    """
    Curates a selection of documents using importance weighting.
    Optimizes for the target domain by prioritizing documents with 
    high semantic similarity to the target_sample.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def get_weights(pool_docs, target_docs):
        """
        Calculates weights for pool_docs based on relevance to target_docs.
        Uses TF-IDF to estimate importance.
        """
        if not pool_docs or not target_docs:
            return np.ones(len(pool_docs))
        
        # Use a small vocabulary to keep it fast and memory efficient
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
        try:
            # Fit on target to establish 'important' vocabulary
            target_tfidf = vectorizer.fit_transform(target_docs).ddf
            pool_tfidf = vectorizer.transform(pool_docs).ddf
            # Dot product to get similarity scores
            scores = (pool_tfidf * target_tfidf.T).sum(axis=1)
            # Normalize scores to weights
            weights = scores / (1.0 + scores)
            return weights
        except Exception:
            # Fallback to uniform if TF-IDF fails
            return np.ones(len(pool_docs))

    # --- First Pass ---
    # Initial weights based on the provided target_sample
    weights = get_weights(docs, target_sample)
    
    # Normalize weights to probabilities
    sum_w = weights.sum()
    if sum_w > 0:
        probs = weights / sum_w
    else:
        probs = np.ones(len(docs)) / len(docs)
    
    # Sample a subset of docs to form a 'refined sub-target'
    # To avoid the error, we use the manual sampling approach or standard random.choices
    sample_size = min(len(docs), 500)
    # Using random.choices which is the correct way to sample with probabilities in the stdlib
    sub_target_indices = random.choices(range(len(docs)), weights=probs, k=sample_size)
    sub_target = [docs[i] for i in sub_target_indices]

    # --- Second Pass ---
    # Re-calculate weights using the refined sub-target as the reference
    refined_weights = get_weights(docs, sub_target)
    
    # Sort indices by descending weight to prioritize high-value content
    sorted_indices = np.argsort(refined_weights)[::-1]
    
    # Return the sorted indices as the ordered list
    return sorted_indices.tolist()
