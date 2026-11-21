import os
import random
import math
import re
import numpy as np
from collections import Counter
import tiktoken
from common import T

def select(docs, target_sample):
    """
    Selects documents from the pool based on a weighted score of Jaccard Similarity
    and Cosine Similarity of token-frequency vectors, with a fallback for low
    Jaccard scores based on token length median.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.encoding.get_encoding("gpt2")
    
    def get_stats(texts):
        token_counts = []
        word_sets = []
        token_freqs = []
        total_tokens = 0
        
        for text in texts:
            tokens = enc.encode(text)
            token_counts.append(len(tokens))
            total_tokens += len(tokens)
            
            # Word set for Jaccard
            words = set(re.findall(r'\w+', text.lower()))
            word_sets.append(words)
            
            # Token frequency for Cosine
            token_freqs.append(Counter(tokens))
            
        return token_counts, word_sets, token_freqs, total_tokens

    # Analyze target sample
    t_counts, t_word_sets, t_token_freqs, _ = get_stats(target_sample)
    
    # Aggregate target word set and token frequency vector
    target_words_all = set().union(*t_word_sets)
    
    # Build a global token frequency vector for the target domain
    # We use a dictionary to store sums of counts for each token
    target_token_vec = Counter()
    for freq_map in t_token_freqs:
        target_token_vec.update(freq_map)
    
    # Calculate median token length of target sample
    median_t_len = np.median(t_counts)

    # Analyze pool docs
    p_counts, p_word_sets, p_token_freqs, _ = get_stats(docs)
    
    scores = []
    for i in range(len(docs)):
        # Jaccard Similarity
        doc_words = p_word_sets[i]
        intersection = len(doc_words.intersection(target_words_all))
        union = len(doc_words.union(target_words_all))
        jaccard = intersection / union if union > 0 else 0.0
        
        # Cosine Similarity of token-frequency vectors
        doc_token_vec = p_token_freqs[i]
        # We need the intersection of keys to compute dot product
        common_tokens = set(doc_token_vec.keys()).intersection(target_token_vec.keys())
        
        dot_product = sum(doc_token_vec[t] * target_token_vec[t] for t in common_tokens)
        
        # Magnitudes
        mag_doc = math.sqrt(sum(v**2 for v in doc_token_vec.values()))
        mag_target = math.sqrt(sum(v**2 for v in target_token_vec.values()))
        
        cosine = dot_product / (mag_doc * mag_target) if mag_doc > 0 and mag_target > 0 else 0.0
        
        # Weighted Score
        score = (jaccard * 0.7) + (cosine * 0.3)
        scores.append((score, jaccard, p_counts[i]))

    # Ranking logic
    # Primary: Score (desc)
    # Secondary: For low Jaccard (e.g. < 0.1), prioritize closer to median token length
    # We handle this by sorting by score first, and then using a custom key logic
    
    indices = list(range(len(docs)))
    
    # To strictly follow "Among documents with low Jaccard scores, prioritize...",
    # we can use a sort key that weights the median distance for the lower Jaccard bracket.
    def ranking_key(idx):
        score, jaccard, count = scores[idx]
        # If Jaccard is "low" (arbitrary threshold 0.1), we factor in the length difference
        # otherwise we just use the score.
        if jaccard < 0.1:
            # Distance from median (smaller is better, so we negate it)
            dist_score = -abs(count - median_t_len)
            # We return a tuple. The first element is the score.
            # To ensure the "prioritize" logic works correctly with the score, 
            # we can slightly nudge the score of low-jaccard docs based on length.
            return (score, dist_score)
        else:
            # For high jaccard, we don't care about length as much, 
            # so we use a large constant for dist_score to keep them above low-jaccard docs
            return (score, 1e9)

    indices.sort(key=ranking_key, reverse=True)
    
    return indices
