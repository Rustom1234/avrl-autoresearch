import os
import random
import math
import re
import tiktoken
from common import T

def select(docs, target_sample):
    """
    Selects documents from the pool based on vocabulary overlap with the 
    target_sample, normalized by length and prioritized by similarity to 
    the average target sample length.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Fix: The correct way to access the encoding in tiktoken is 
    # tiktoken.get_encoding() or using the specific method.
    enc = tiktoken.get_encoding("gpt2")
    
    # Pre-process target sample
    target_tokens = []
    target_word_set = set()
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at', 
        'by', 'from', 'to', 'in', 'for', 'with', 'on', 'as', 'is', 'are', 'was', 
        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 
        'its', 'our', 'their', 'this', 'that', 'which', 'who', 'whom', 'whose'
    }

    target_token_counts = []
    for text in target_sample:
        t_tokens = enc.encode(text.lower())
        target_tokens.extend(t_tokens)
        target_token_counts.append(len(t_tokens))
        words = re.findall(r'\w+', text.lower())
        for w in words:
            if w not in stop_words and len(w) > 1:
                target_word_set.add(w)

    avg_target_len = sum(target_token_counts) / len(target_sample)

    scores = []
    for doc in docs:
        doc_tokens = enc.encode(doc)
        doc_len = len(doc_tokens)
        if doc_len == 0:
            scores.append(-float('inf'))
            continue
            
        # Vocabulary overlap (using words to define domain relevance)
        doc_words = set(re.findall(r'\w+', doc.lower()))
        overlap = len(doc_words.intersection(target_word_set))
        
        # Normalized score: log1p(overlap / doc_len)
        # Note: Using doc_len as a divisor to favor information density
        norm_overlap = overlap / doc_len
        base_score = math.log1p(norm_overlap)
        
        # Proximity to average target length: 
        # Use a similarity score: exp(-|doc_len - avg_target_len| / constant)
        length_diff = abs(doc_len - avg_target_len)
        # Using a smoothing factor based on average length
        length_similarity = math.exp(-length_diff / (avg_target_len + 10))
        
        final_score = base_score * length_similarity
        scores.append(final_score)

    indices = list(range(len(docs)))
    # Sort by score descending, then by original index for stability
    indices.sort(key=lambda i: (scores[i], -i), reverse=True)

    return indices
