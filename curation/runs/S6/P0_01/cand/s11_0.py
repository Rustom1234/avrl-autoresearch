import os
import random
import re
from collections import Counter
import numpy as np
import tiktoken

def select(docs, target_sample):
    """
    Selects indices from docs by first filtering for high-quality English/Wikipedia-like
    text and then ranking based on term overlap with the target_sample.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    T = 9830400
    enc = tiktoken.get_encoding("gpt2")
    
    def get_tokens(text):
        return enc.encode(text)

    # 1. Profile the target domain
    target_words = []
    # Basic stop words to filter noise in target profile
    stop_words = {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'were', 'they', 'have', 'are', 'be', 'an', 'is'}
    for doc in target_sample:
        words = re.findall(r'\w+', doc.lower())
        target_words.extend([w for w in words if len(w) > 3 and w not in stop_words])
    
    target_counts = Counter(target_words)
    
    # 2. Score and Filter
    # Scoring parameters
    # We want to filter out junk (URLs, hex, money patterns, etc.)
    # and favor coherent, sentence-rich text.
    
    # Regex patterns for "junk" content
    junk_patterns = [
        r'http\://\S+|https?://\S+',  # URLs
        r'\b0x[0-9a-fA-F]{40,}\b',      # Long hex strings
        r'\$\d+(?:\.\d+)?',             # Money
        r'[a-zA-Z0-9]{20,}',            # Long alphanumeric strings (possible hashes/IDs)
    ]
    junk_regex = re.compile('|'.join(junk_patterns))
    
    # Regex for sentences (basic)
    sentence_regex = re.compile(r'[.!?]+')
    
    pool_scores = []
    valid_docs = []
    
    for i, doc in enumerate(docs):
        # Basic cleanliness check
        if junk_regex.search(doc):
            # Significant junk found
            pass
        else:
            # Check for coherent structure
            sentences = len(sentence_regex.findall(doc))
            words = re.findall(r'\w+', doc.lower())
            
            # Filter: needs some sentences and enough words to be a "snippet"
            if sentences >= 1 and len(words) > 5:
                # Calculate term overlap score
                score = 0
                for w in words:
                    if w in target_counts:
                        score += target_counts[w]
                
                # Heuristic: Reward docs that are not too short but have good overlap
                # and penalize very small word counts to avoid "fragments"
                final_score = score * (1.0 + (sentences / 10.0))
                pool_scores.append(final_score)
                valid_docs.append(i)
            else:
                # This doc is either too short or lacks structure
                pass

    # 3. Rank the valid documents
    # We sort by score descending
    scores = np.array(pool_scores)
    sorted_indices = np.argsort(scores)[::-1]
    
    # Map back to original doc indices
    ranked_indices = [valid_docs[idx] for idx in sorted_indices]
    
    # 4. Selection
    # We need to fill a sequence that will be cycled.
    # We pick the best documents until we have a good pool.
    # Since the trainer cycles the list, having a dense high-quality block is best.
    
    selected_indices = []
    current_tokens = 0
    
    # Limit the maximum size to avoid extreme cycles while ensuring good variety
    # but prioritize the absolute highest quality.
    for idx in ranked_indices:
        doc_tokens = len(get_tokens(docs[idx]))
        
        # Skip extremely long junk-like blobs (oversized documents)
        if doc_tokens > 10000:
            continue
            
        selected_indices.append(int(idx))
        current_tokens += doc_tokens
        
        # If we've reached a healthy chunk of T, we can stop.
        # Because the list is cycled, a shorter, high-quality list is 
        # often better than a long, diluted one.
        if current_tokens > T * 0.7:
            break
            
    # Fallback to ensure we have a baseline if filtering was too aggressive
    if len(selected_indices) == 0 and len(docs) > 0:
        # If no "valid" docs found, just take some from the top of original
        selected_indices = [i for i in range(min(100, len(docs)))]
        
    return selected_indices
