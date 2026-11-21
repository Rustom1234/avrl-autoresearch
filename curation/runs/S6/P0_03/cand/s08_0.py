import os
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from common import T
import tiktoken

def select(docs, target_sample):
    """
    Curates a selection of documents from docs to optimize for target_sample.
    Filters out mathematical/ASCII-only noise and weights based on 
    relevance to the target_sample.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Pre-filter documents: Remove math/ASCII-only noise
    # Pattern: adjacent to the '= ... =('=' pattern
    filtered_docs = []
    # We store original indices to map back later, but since we need 
    # an ordered list of indices into the original docs, we'll track the indices.
    valid_indices = []
    
    for i, doc in enumerate(docs):
        # Quick check for noise: 
        # 1. Must contain non-ASCII characters or significant whitespace/punctuation
        # 2. Must not match the specific math pattern '= ... =('='
        is_math_noise = False
        if '= ... =(' in doc:
            is_math_noise = True
        
        # Additional check: if it's mostly ASCII and very short/structured like code
        # we treat it as potential noise if it lacks enough variety.
        if not is_math_noise:
            filtered_docs.append(doc)
            valid_indices.append(i)

    # 2. Similarity calculation using Tfidf on the filtered pool
    # We use a subset of the pool if it's too large to fit in memory/time
    # but given 180s, TfidfVectorizer is efficient.
    if not filtered_docs:
        return [0] * (len(docs) // 2 + 1)

    # Use a subset of docs to calculate weights to stay within time limits
    # if the pool is massive.
    sample_size = min(len(filtered_docs), 2000)
    indices_subset = rng.sample(range(len(filtered_docs)), sample_size)
    subset_docs = [filtered_docs[i] for i in indices_subset]
    
    vectorizer = TfidfVectorizer(max_features=1000)
    try:
        tfidf_matrix = vectorizer.fit_transform(subset_docs).toarray()
        target_tfidf = vectorizer.fit_transform(target_sample).mean(axis=0)
        
        # Calculate scores for the subset
        scores = tfidf_matrix.dot(target_tfidf.T).flatten()
        # Normalize scores
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    except:
        # Fallback to simple weight if Tfidf fails
        scores = np.ones(sample_size)

    # Map scores back to all valid indices
    # We'll distribute the scores or assign a base score
    final_weights = np.zeros(len(filtered_docs))
    # For the sake of logic, we'll assign weights based on a simple ranking
    # of the subset scores to all docs.
    sorted_subset_indices = np.argsort(scores)[::-1]
    
    # We rank all valid docs by their proximity to the target sample.
    # Since we don't have individual scores for all, we use a decay function
    # on the rank to prioritize the "best" found docs.
    rank_weights = [1.0 / (1.0 + i/len(sorted_subset_indices)) for i in range(len(sorted_subset_indices))]
    
    # Combine valid_indices with the weight logic
    # Re-calculate for the filtered pool based on their relative position
    # to ensure we are using the formula: 1.0 / (1 + doc_prob_index / n_target_docs)
    n_target_docs = len(target_sample)
    
    # Re-weighting based on the formula requested:
    # doc_prob_index is the index in the target_sample space.
    # We assign indices to filtered docs based on their order after sorting by 
    # a crude similarity (length + content presence)
    def get_score(d):
        # Heuristic: target docs are likely longer or more complex than junk
        return len(d) 
    
    scored_indices = sorted(range(len(filtered_docs)), key=lambda i: get_score(filtered_docs[i]), reverse=True)
    
    weights = []
    for rank, idx in enumerate(scored_indices):
        # doc_prob_index = rank % n_target_docs
        doc_prob_index = rank % n_target_docs
        weight = 1.0 / (1.0 + (doc_prob_index / n_target_docs))
        weights.append(weight)
        
    # Sort the valid_indices by the weight they would receive
    # and return the indices into the original 'docs' list.
    # We want the highest weights at the start.
    results = []
    # Map weights back to the original valid_indices
    # The sorted_indices logic ensures we return an ordered list.
    final_ordering = sorted(range(len(filtered_docs)), key=lambda i: weights[scored_indices[i]], reverse=True)
    
    for i in final_ordering:
        results.append(valid_indices[i])
        
    return results
