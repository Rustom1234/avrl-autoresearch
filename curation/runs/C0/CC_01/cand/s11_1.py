import os
import random
import math
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Corrected initialization of tiktoken encoding
    enc = tiktoken.get_encoding("gpt2")
    
    # Pre-process target sample to get token counts
    target_tokens = []
    target_vocab_counts = {}
    
    for doc in target_sample:
        tokens = enc.encode(doc)
        target_tokens.extend(tokens)
        for t in tokens:
            target_vocab_counts[t] = target_vocab_counts.get(t, 0) + 1
            
    valid_indices = []
    scores = []
    
    for i, doc in enumerate(docs):
        # Filter 1: Ratio of non-alphanumeric characters > 15%
        non_alnum_count = sum(1 for char in doc if not char.isalnum())
        total_chars = len(doc)
        
        if total_chars > 0:
            ratio = non_alnum_count / total_chars
            if ratio > 0.15:
                continue
        else:
            continue

        # Tokenize the document
        doc_tokens = enc.encode(doc)
        doc_token_count = len(doc_tokens)
        
        # Filter 2: Fewer than 100 tokens
        if doc_token_count < 100:
            continue
            
        # Calculate Score: (sum of target_vocabulary_token_counts) * (log(1 + total_tokens))
        sum_target_counts = 0
        for t in doc_tokens:
            if t in target_vocab_counts:
                sum_target_counts += target_vocab_counts[t]
        
        score = sum_target_counts * math.log(1 + doc_token_count)
        
        valid_indices.append(i)
        scores.append(score)
        
    if not valid_indices:
        return list(range(len(docs)))
    
    # Sort indices by score descending
    combined = sorted(zip(valid_indices, scores), key=lambda x: x[1], reverse=True)
    final_indices = [x[0] for x in combined]
    
    # Ensure we return enough tokens to fill the target size (T)
    # The trainer cycles the list if it's shorter than T.
    # We provide the sorted list; the trainer handles the cycling.
    return final_indices
