import os
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from common import T
import tiktoken

def select(docs, target_sample):
    """
    Curates a selection of documents using recursive probability selection.
    1. Estimate importance weights based on target_sample.
    2. Sample a refined 'sub-target' from the pool.
    3. Re-calculate weights based on the refined sub-target.
    4. Return the final ordered selection.
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
        
        # Use a small vocabulary to keep it fast
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
        try:
            # Fit on target to establish 'important' vocabulary
            target_tfidf = vectorizer.fit_transform(target_docs).ddf
            pool_tfidf = vectorizer.transform(pool_docs).ddf
            # Dot product to get similarity scores
            scores = (pool_tfidf * target_tfidf.T).sum(axis=1)
            # Normalize scores to probabilities/weights
            weights = scores / (1.0 + scores)
            return weights
        except:
            # Fallback to uniform if TF-IDF fails
            return np.ones(len(pool_docs))

    # --- First Pass ---
    # Initial weights based on the provided target_sample
    initial_weights = get_weights(docs, target_sample)
    probs = initial_weights / initial_weights.sum()
    
    # Sample a subset of docs to form a 'refined sub-target'
    # We want enough docs to represent the domain but not so many it exceeds 180s
    sample_size = min(len(docs), 500)
    sub_target_indices = rng.choice(len(docs), size=sample_size, p=probs, replace=True)
    sub_target = [docs[i] for i in sub_target_indices]

    # --- Second Pass (Recursive Probability Selection) ---
    # Re-calculate weights using the refined sub-target as the reference
    refined_weights = get_weights(docs, sub_target)
    
    # Final selection logic:
    # We want the most 'valuable' documents first to optimize the 
    # prefix of the training run.
    # Sort indices by descending weight.
    sorted_indices = np.argsort(refined_weights)[::-1]
    
    # Because the trainer cycles the list, we want to prioritize the best 
    # documents but ensure variety. We can repeat the best ones or 
    # simply return the sorted list. The prompt asks for an ordered list.
    # To ensure we hit T tokens efficiently, we return the sorted indices.
    # If the sum of tokens in docs[sorted_indices] < T, the trainer cycles it.
    
    return sorted_indices.tolist()
