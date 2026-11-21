import os
import random
import tiktoken
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Pre-calculate bigrams in the target sample
    target_bigrams = set()
    for text in target_sample:
        words = text.lower().split()
        for i in range(len(words) - 1):
            target_bigrams.add((words[i], words[i+1]))
            
    # Score documents and collect info
    scored_docs = []
    for i, doc in enumerate(docs):
        words = doc.lower().split()
        # Use first 300 words for scoring
        score_words = words[:300]
        match_count = 0
        total_bigrams = 0
        
        for j in range(len(score_words) - 1):
            bigram = (score_words[j], score_words[j+1])
            total_bigrams += 1
            if bigram in target_bigrams:
                match_count += 1
        
        score = match_count / total_bigrams if total_bigrams > 0 else 0
        # Estimate tokens as len(text) / 4
        token_count = len(doc) / 4
        scored_docs.append((score, token_count, i))
        
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Select documents until we reach T tokens
    selected_indices = []
    current_tokens = 0
    T = 9830400 # Value from common.py
    
    for score, tokens, idx in scored_docs:
        selected_indices.append(idx)
        current_tokens += tokens
        if current_tokens >= T:
            break
            
    # Shuffle the selection to provide variety
    rng.shuffle(selected_indices)
    return selected_indices
