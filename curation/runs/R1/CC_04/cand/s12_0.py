import os
import random
import re
import math
import tiktoken
from common import T

def select(docs, target_sample):
    """
    Selects documents from the pool based on a weighted Jaccard similarity
    score calculated using tiktoken and adjusted for content density.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Define noise patterns to strip
    junk_patterns = [
        r'click here', r'posted by', r'subscribe', r'follow us',
        r'share this', r'related articles', r'advertisement',
        r'click to see more', r'terms and conditions'
    ]
    
    # Helper to clean and tokenize
    def get_clean_tokens(text):
        # Remove junk patterns
        cleaned = text.lower()
        for pattern in junk_patterns:
            cleaned = re.sub(pattern, '', cleaned)
        return enc.encode(cleaned)

    # Build target token set
    target_tokens = set()
    for text in target_sample:
        target_tokens.update(get_clean_tokens(text))

    scores = []
    for doc in docs:
        # Full tokenization
        doc_tokens_raw = enc.encode(doc)
        total_count = len(doc_tokens_raw)
        
        if total_count == 0:
            scores.append(-1.0)
            continue
            
        # Content-bearing tokenization
        doc_tokens_clean = set(get_clean_tokens(doc))
        
        # Jaccard Similarity
        intersection = len(doc_tokens_clean.intersection(target_tokens))
        union = len(doc_tokens_clean.union(target_tokens))
        jaccard = intersection / union if union > 0 else 0
        
        # Content Density: Ratio of unique clean tokens to total raw tokens
        # This rewards documents that are "dense" with unique info per token
        content_density = len(doc_tokens_clean) / total_count
        
        # Weighted score
        # We use log1p on the jaccard and weight it by density to ensure 
        # we prioritize relevant and high-quality text.
        score = math.log1p(jaccard) * (1.0 + content_density)
        scores.append(score)

    # Rank indices: score descending, then index ascending for stability
    indices = list(range(len(docs)))
    indices.sort(key=lambda i: (scores[i], -i), reverse=True)

    return indices
