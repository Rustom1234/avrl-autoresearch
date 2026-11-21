import os
import random
import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Identify documents passing WikiText token filters
    # We look for common separators and structure tokens.
    valid_indices = []
    wiki_tokens = [' @-@ ', ' , ', ' . ']
    
    for i, doc in enumerate(docs):
        if any(token in doc for token in wiki_tokens):
            valid_indices.append(i)
            
    # 2. Similarity Filter
    # We compare the first 300 chars of valid docs against target_sample 
    # using character n-grams to ensure relevance.
    if valid_indices and target_sample:
        # Prepare target_sample for vectorization
        # We take the strings as they are to represent "known WikiText fragments"
        corpus = target_sample
        
        # Prepare the first 300 chars of candidate docs
        candidates = [docs[i][:300] for i in valid_indices]
        
        # Use CountVectorizer with character n-grams
        # Note: We need to keep track of the mapping back to valid_indices
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 5), max_features=500)
        
        try:
            # Fit on both to ensure same vocabulary
            combined = corpus + candidates
            vectorizer.fit(combined)
            
            target_vectors = vectorizer.transform(corpus)
            candidate_vectors = vectorizer.transform(candidates)
            
            # Calculate similarity of each candidate to the target set
            # (Mean similarity to any document in target_sample)
            sim_matrix = cosine_similarity(candidate_vectors, target_vectors)
            max_sims = sim_matrix.max(axis=1)
            
            # Only keep docs with at least a baseline similarity (e.g., > 0.1)
            # This filters out junk/noise that doesn't share structure with target
            filtered_indices = []
            for idx, score in enumerate(max_sims):
                if score > 0.1:
                    filtered_indices.append(valid_indices[idx])
            
            # Update valid_indices to be the filtered ones
            valid_indices = filtered_indices
        except:
            # Fallback if vectorization fails for some reason
            pass

    # 3. Exact Deduplication
    # We want to remove exact duplicates from the selected pool
    unique_indices = []
    seen_docs = set()
    # Shuffle to preserve randomness while deduplicating
    rng.shuffle(valid_indices)
    for idx in valid_indices:
        if docs[idx] not in seen_docs:
            unique_indices.append(idx)
            seen_docs.add(docs[idx])
            
    # Fallback if filter was too aggressive
    if not unique_indices:
        unique_indices = list(range(len(docs)))
        rng.shuffle(unique_indices)

    return unique_indices
