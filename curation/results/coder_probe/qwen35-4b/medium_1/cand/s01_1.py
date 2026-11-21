"""Curator for tuning training data allocation.

select(docs, target_sample) -> ordered list of integers (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import numpy as np
from common import T

def select(docs, target_sample):
    target_vocab = set()
    for doc in target_sample:
        words = doc.lower().split()
        for word in words:
            if word.isalpha():
                target_vocab.add(word)
    
    scored_docs = []
    for i, doc in enumerate(docs):
        doc_lower = doc.lower()
        words = doc_lower.split()
        
        words_to_check = words[:300]
        
        bigram_match_count = 0
        for j in range(len(words_to_check) - 1):
            word1 = words_to_check[j]
            word2 = words_to_check[j+1]
            if word1 in target_vocab and word2 in target_vocab:
                bigram_match_count += 1
        
        total_bigrams = len(words_to_check) - 1
        if total_bigrams == 0:
            score = 0
        else:
            score = bigram_match_count / total_bigrams
        
        scored_docs.append((i, score))
    
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    selected_indices = []
    tokens_collected = 0
    
    for doc_idx, score in scored_docs:
        if tokens_collected >= T:
            break
        
        doc = docs[doc_idx]
        est_tokens = abs(len(doc.split()) / 4.0)
        tokens_collected += est_tokens
        
        if tokens_collected > T:
            tokens_collected -= est_tokens
            break
            
        selected_indices.append(doc_idx)
    
    if not selected_indices:
        return selected_indices
        
    shuffled_indices = []
    selection_dict = {doc_idx: ans for ans, doc_idx in enumerate(selected_indices)}
    # Realign to avoid using the subset of selected indices as index map.
    # We need to shuffle *the original selected_indices*, then map back to the original docs indices.
    docs_to_shuffle = selected_indices
    random_seed = int(os.environ.get("CURATE_SEED", 0))
    rng = np.random.RandomState(random_seed)
    
    indices = list(range(len(docs_to_shuffle)))
    rng.shuffle(indices)
    
    shuffled_indices = [docs_to_shuffle[i] for i in indices]
    
    return shuffled_indices
