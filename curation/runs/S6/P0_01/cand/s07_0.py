import os
import random
import re
from collections import Counter
import numpy as np
import tiktoken

def select(docs, target_sample):
    """
    Selects indices from docs to optimize model training on a target domain.
    Implements a stratified sampling schedule that favors high-quality, short 
    documents (<= 500 tokens) to maximize signal density in a cycled training loop.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    T = 9830400
    enc = tiktoken.get_encoding("gpt2")
    
    def get_tokens(text):
        return enc.encode(text)

    # 1. Profile the target domain
    target_words = []
    stop_words = {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'were', 'they', 'have', 'are', 'be', 'an', 'is'}
    for doc in target_sample:
        words = re.findall(r'\w+', doc.lower())
        target_words.extend([w for w in words if len(w) > 3 and w not in stop_words])
    
    target_counts = Counter(target_words)
    
    # 2. Score the noisy docs and filter by token length
    # We strictly prioritize docs <= 500 tokens to ensure high turnover of 
    # high-quality content in the repeating cycle.
    pool_scores = []
    pool_token_counts = []
    valid_indices = []

    for i, doc in enumerate(docs):
        tokens = get_tokens(doc)
        count = len(tokens)
        
        # Rule: Strictly limit to max 500 tokens for high signal density
        if count > 500:
            continue
            
        # Score based on vocabulary overlap
        words = re.findall(r'\w+', doc.lower())
        score = 0
        for w in words:
            if w in target_counts:
                score += target_counts[w]
        
        pool_scores.append(score)
        pool_token_counts.append(count)
        valid_indices.append(i)

    # 3. Rank the filtered candidates
    scores = np.array(pool_scores)
    counts = np.array(pool_token_counts)
    
    # Combine score and token count into a ranking (prioritize high score, then high density)
    # Rank = score_normalized * 0.9 + density_normalized * 0.1
    max_score = scores.max() if scores.size > 0 else 1.0
    max_count = counts.max() if counts.size > 0 else 1.0
    
    ranking = (scores / max_score) * 0.9 + (counts / max_count) * 0.1
    sorted_meta = np.argsort(ranking)[::-1]
    
    # 4. Deterministic Stratified Sampling
    # We select the top N documents that provide a robust variety.
    # Since the trainer cycles, we want a high-density block of the best content.
    selected_indices = []
    
    # Selection: Take the top 20% of the best-scoring short documents
    # to ensure variety while maintaining strict quality.
    num_to_select = max(1, len(valid_indices) // 5)
    # Ensure we don't exceed a reasonable number of unique docs to keep cycle fast
    num_to_select = min(num_to_select, 500)
    
    top_indices = sorted_meta[:num_to_select]
    for idx_in_valid in top_indices:
        selected_indices.append(valid_indices[idx_in_valid])

    # Fallback if no documents were < 500 tokens
    if not selected_indices:
        # Re-run without the 500 limit but take very small ones if available
        # and sort by score
        for i, doc in enumerate(docs):
            tokens = get_tokens(doc)
            if len(tokens) <= 1000: # Slightly relaxed for fallback
                words = re.findall(r'\w+', doc.lower())
                score = sum(target_counts[w] for w in words)
                pool_scores.append(score)
                valid_indices.append(i)
        
        if valid_indices:
            scores = np.array(pool_scores)
            sorted_meta = np.argsort(scores)[::-1]
            selected_indices = [valid_indices[i] for i in sorted_meta[:200]]
        else:
            # Final fallback: just take the first few
            selected_indices = list(range(min(len(docs), 100)))

    return selected_indices
