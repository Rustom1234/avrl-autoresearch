import os
import random
import re
from collections import Counter
import numpy as np
import tiktoken

def select(docs, target_sample):
    """
    Selects indices from docs to optimize model training on a target domain.
    Implements a term-overlap score weighted by structural markers like
    Markdown headers or equation blocks.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    T = 9830400
    enc = tiktoken.get_encoding("gpt2")
    
    def get_tokens(text):
        return enc.encode(text)

    # 1. Profile the target domain
    # Build a profile of significant words from the target domain.
    target_words = []
    stop_words = {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'were', 'they', 'have', 'are', 'be', 'an', 'is'}
    for doc in target_sample:
        words = re.findall(r'\w+', doc.lower())
        target_words.extend([w for w in words if len(w) > 3 and w not in stop_words])
    
    target_counts = Counter(target_words)
    
    # 2. Score the noisy docs based on term overlap and structural signals
    pool_scores = []
    pool_token_counts = []
    
    for doc in docs:
        # Term-overlap score
        words = re.findall(r'\w+', doc.lower())
        score = 0
        for w in words:
            if w in target_counts:
                score += target_counts[w]
        
        # Structural Marker Weighting:
        # Check for Markdown headers (#, ##, etc.) or equation blocks (= ... =)
        # The rule: apply a multiplicative weight if these are present.
        weight = 1.0
        if doc.strip().startswith('#') or (doc.strip().startswith('=') and doc.strip().endswith('=')):
            weight = 2.5
        elif re.search(r'^#+ ', doc, re.MULTILINE) or re.search(r'=\s*.+\s*=', doc, re.MULTILINE):
            # Also check for internal blocks to catch multi-line structures
            weight = 1.8
            
        pool_scores.append(score * weight)
        pool_token_counts.append(len(get_tokens(doc)))
    
    pool_scores = np.array(pool_scores)
    pool_token_counts = np.array(pool_token_counts)
    
    # 3. Rank indices by score
    ranked_indices = np.argsort(pool_scores)[::-1]
    
    # 4. Selection Strategy
    # Prioritize high-score docs. We want to build a dense list of high-quality 
    # training data that the trainer will cycle through.
    selected_indices = []
    current_tokens = 0
    
    # Filter out extremely long strings that might be garbage but keep the signal
    # (heuristic: avoid strings > 100k tokens if they are low quality)
    for idx in ranked_indices:
        count = pool_token_counts[idx]
        if count > 100000:
            continue
            
        selected_indices.append(int(idx))
        current_tokens += count
        
        # Break early if we have a very high-quality subset to ensure 
        # the cycling happens on the best available data.
        if current_tokens > T * 0.5 and len(selected_indices) > 200:
            break

    # Ensure we have at least some content
    if len(selected_indices) == 0:
        # Fallback to top ranked if filter was too aggressive
        selected_indices = [int(idx) for idx in ranked_indices[:100]]

    return selected_indices
