import os
import random
import zlib
import re
from collections import Counter
import numpy as np
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    T = 9830400
    
    # Tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # 1. Basic stats and Deduplication
    # We use a simple set of hashes to skip duplicates
    seen_hashes = set()
    doc_data = []
    
    # To estimate "goodness", we first build a vocabulary of the target sample
    target_vocab = set()
    for s in target_sample:
        words = re.findall(r'[a-zA-Z]+', s.lower())
        target_vocab.update(words)

    # 2. Process documents and collect features
    # We do a light pass to calculate some global stats if possible, 
    # but to stay under 180s, we'll process docs in one pass.
    
    for i, doc in enumerate(docs):
        # Fast deduplication
        h = hash(doc)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        
        # Basic quality filters
        # 1. Mojibake detection
        mojibake_chars = ('Ã', 'â', '€', 'Â', '§', '°')
        has_mojibake = any(c in doc for c in mojibake_chars)
        
        # 2. Compression ratio (simple proxy for entropy/repetition)
        raw_bytes = doc.encode('utf-8')
        z_ratio = len(zlib.compress(raw_bytes)) / len(raw_bytes) if len(raw_bytes) > 0 else 1.0
        
        # 3. Token count
        tokens = enc.encode(doc)
        num_tokens = len(tokens)
        
        # 4. Junk text detection
        is_junk = any(junk in doc.lower() for junk in ['click here', 'best price', 'free shipping', 'login to', 'powered by'])
        
        # Score: 1.0 for high quality, lower for lower quality
        # We want text that isn't junk, isn't mojibake, and has a reasonable compression ratio
        score = 0.0
        if not has_mojibake and not is_junk:
            score = 1.0
            if z_ratio > 0.3 and z_ratio < 0.8:
                score += 0.5
        
        doc_data.append({
            'id': i,
            'tokens': num_tokens,
            'score': score,
            'z_ratio': z_ratio
        })

    # Sort by score descending
    doc_data.sort(key=lambda x: x['score'], reverse=True)

    # 3. Selection
    selected_indices = []
    current_tokens = 0
    # Goal is to fill roughly T tokens. We aim for slightly more (e.g., 1.05*T) 
    # because the trainer cycles the list if it's shorter than T.
    target_limit = int(1.05 * T)
    
    for item in doc_data:
        selected_indices.append(item['id'])
        current_tokens += item['tokens']
        if current_tokens >= target_limit:
            break
            
    # If we didn't reach the limit, we add more random high-score docs 
    # or just cycle what we have.
    if current_tokens < target_limit:
        remaining = target_limit - current_tokens
        # Find docs not already selected
        remaining_docs = [d for d in doc_data if d['id'] not in selected_indices]
        if remaining_docs:
            # Simple selection of remaining
            for item in remaining_docs:
                selected_indices.append(item['id'])
                current_tokens += item['tokens']
                if current_tokens >= target_limit:
                    break
        else:
            # Fallback: Cycle the current selection to reach the target
            while current_tokens < target_limit:
                for idx in selected_indices:
                    # In reality, we need to know the token count of the original doc
                    # So we find it again
                    doc_idx = selected_indices.index(idx)
                    current_tokens += doc_data[doc_idx]['tokens']
                    selected_indices.append(idx)
                    if current_tokens >= target_limit:
                        break
                    if len(selected_indices) > 100000: # Safety break
                        break
                if len(selected_indices) > 100000: break

    # The instructions say "ordered list of indices". 
    # The trainer cycles the list, so order matters (initial position).
    # We return the indices directly.
    return selected_indices
