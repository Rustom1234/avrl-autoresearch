"""Curator for tuning training data allocation.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import numpy as np
from common import T

def select(docs, target_sample):
    # Step 1: Create target vocabulary from target_sample
    # We only need the first character of each word? Or the whole word?
    # The prompt says "target sample ... in the pool". Usually, we check if a word exists in the target.
    # "word bigrams ... occur in the target sample"
    target_vocab = set()
    for doc in target_sample:
        words = doc.lower().split()
        for word in words:
            if word.isalpha():
                target_vocab.add(word)
    
    # Step 2: Score each document
    scored_docs = []
    for i, doc in enumerate(docs):
        doc_lower = doc.lower()
        words = doc_lower.split()
        
        # Filter first 300 words (approx)
        # To be precise about tokens, we can't easily do that without a tokenizer,
        # but the prompt says "first 300 words". So we just take first 300 words.
        words_to_check = words[:300]
        
        bigram_match_count = 0
        for j in range(len(words_to_check) - 1):
            word1 = words_to_check[j]
            word2 = words_to_check[j+1]
            if word1 in target_vocab and word2 in target_vocab:
                bigram_match_count += 1
        
        # Score is fraction of matches
        total_bigrams = len(words_to_check) - 1
        if total_bigrams == 0:
            score = 0
        else:
            score = bigram_match_count / total_bigrams
        
        scored_docs.append((i, score))
    
    # Step 3: Sort by score descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Step 4: Select top documents until we hold about T tokens
    selected_indices = []
    tokens_collected = 0
    
    # Re-cap score calculation:
    # Score = bigram matches / (total bigrams in first 300 words)
    # The requirement says: "Score each document by the fraction of its word bigrams ... that also occur in the target sample"
    # And "Keep the highest-scoring documents until they hold about T tokens"
    
    while tokens_collected < T:
        doc_idx, score = scored_docs.pop(0)
        selected_indices.append(doc_idx)
        doc = docs[doc_idx]
        estimated_tokens = abs(len(doc.split()) / 4.0)
        tokens_collected += estimated_tokens
        
        # Safety check to prevent infinite loop or selecting all just in case
        if len(selected_indices) >= len(docs):
            break
            
    # Step 5: Shuffle the selected documents
    docs_to_shuffle = docs[selected_indices]
    random_seed = int(os.environ.get("CURATE_SEED", 0))
    rng = np.random.RandomState(random_seed)
    
    indices = list(range(len(docs_to_shuffle)))
    rng.shuffle(indices)
    shuffled_indices = [selected_indices[i] for i in indices]
    
    return shuffled_indices
