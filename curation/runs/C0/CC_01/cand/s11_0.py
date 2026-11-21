import os
import random
import math
import re
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.encoding.Tokenizer.from_pretrained("gpt2")
    
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
        # We count characters that are NOT [a-zA-Z0-9]
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
        # Here, "target_vocabulary_token_counts" refers to the counts of tokens 
        # in the current doc that appear in the target sample's vocabulary.
        sum_target_counts = 0
        for t in doc_tokens:
            if t in target_vocab_counts:
                sum_target_counts += target_vocab_counts[t]
        
        score = sum_target_counts * math.log(1 + doc_token_count)
        
        valid_indices.append(i)
        scores.append(score)
        
    if not valid_indices:
        return list(range(len(docs)))
    
    # Sort indices by score descending, then shuffle slightly or keep order?
    # The prompt asks to select indices based on score. Usually implies sorting.
    # To maintain variety while following score, we sort by score.
    combined = sorted(zip(valid_indices, scores), key=lambda x: x[1], reverse=True)
    final_indices = [x[0] for x in combined]
    
    return final_indices
