import os
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def select(docs, target_sample):
    """
    Implements a priority strategy:
    1. Use the target_sample as the primary source of high-quality signal.
    2. Rank the rest of the pool using cosine similarity to the target_sample.
    3. Return the ordered indices.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    random.seed(seed)
    np.random.seed(seed)

    # We cannot use docs that are in target_sample because target_sample 
    # is defined as not being in the pool. 
    # However, we want to model the domain of target_sample.
    
    # To calculate similarity efficiently within the time limit:
    # We use a subset of target_sample if it's too large, 
    # but 200 is small enough for TF-IDF.
    vectorizer = TfidfVectorizer(max_features=1000)
    
    # Combine target_sample for the 'centroid' of the domain
    # and docs for the pool.
    # Note: We need to find which docs are most similar to the target domain.
    
    try:
        # Vectorize target sample to get a domain representation
        target_vectors = vectorizer.fit_transform(target_sample)
        # Average vector for the target domain
        domain_centroid = np.mean(target_vectors.toarray(), axis=0).reshape(1, -1)
        
        # Vectorize the docs pool
        doc_vectors = vectorizer.transform(docs)
        
        # Calculate similarities
        similarities = cosine_similarity(domain_centroid, doc_vectors).flatten()
        
        # Sort doc indices by similarity descending
        # We'll use a stable sort to maintain consistency
        doc_indices = list(range(len(docs)))
        doc_indices.sort(key=lambda i: similarities[i], reverse=True)
        
        # Because the target_sample isn't in the pool, we can't "copy" its 
        # indices directly as requested (they aren't in docs).
        # The prompt asks to "copies target_sample indices first" - 
        # since target_sample are strings and docs is a pool of strings, 
        # if any doc in the pool matches a string in target_sample, 
        # we should prioritize it.
        
        # Find indices in docs that are identical to any string in target_sample
        target_set = set(target_sample)
        priority_indices = []
        remaining_indices = []
        
        for i, d in enumerate(docs):
            if d in target_set:
                priority_indices.append(i)
            else:
                remaining_indices.append(i)
        
        # Sort the remaining by similarity
        # We need to re-map the similarities to the remaining_indices
        rem_similarities = [similarities[i] for i in remaining_indices]
        rem_indices_sorted = sorted(remaining_indices, key=lambda i: rem_similarities[docs.index(docs[i])], reverse=True)
        
        # The prompt specifically asked for a strategy that copies target_sample indices first.
        # Since target_sample is a list of strings, and we need indices into docs:
        # We will prioritize docs that match target_sample strings, then the rest sorted by similarity.
        
        return priority_indices + rem_indices_sorted

    except Exception:
        # Fallback to simple random if TF-IDF fails
        idx = list(range(len(docs)))
        random.Random(seed).shuffle(idx)
        return idx
