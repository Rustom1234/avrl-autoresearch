import os
import random
import string
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initialize tiktoken encoder for gpt2
    enc = tiktoken.get_encoding("gpt2")
    
    # Keywords to filter out unless they are part of a valid document
    # Since we don't have specific "topic" labels, we exclude these specific 
    # distracting noise sources as requested.
    blacklist = {'monster energy', 'mit media lab'}
    
    valid_indices = []
    
    for i, doc in enumerate(docs):
        # 1. Length check: skip if less than 25 tokens
        tokens = enc.encode(doc)
        if len(tokens) < 25:
            continue
            
        # 2. Special character density check (>80% special characters)
        special_count = 0
        total_chars = 0
        for char in doc:
            total_chars += 1
            if char not in string.ascii_letters and char not in string.digits and char not in " \n\r\t":
                special_count += 1
        
        if total_chars > 0:
            density = special_count / total_chars
            if density > 0.8:
                continue
        else:
            continue

        # 3. Keyword blacklist check
        doc_lower = doc.lower()
        is_blacklisted = any(kw in doc_lower for kw in blacklist)
        # The prompt implies we exclude them unless it's actually the topic.
        # Given no labels on docs, we treat blacklist matches as noise.
        if is_blacklisted:
            continue
            
        valid_indices.append(i)
        
    # Shuffle the valid indices found
    rng.shuffle(valid_indices)
    
    # Return the list of indices. The trainer will cycle this if it's shorter than T.
    return valid_indices
