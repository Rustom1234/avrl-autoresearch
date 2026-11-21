import os
import random
import re
from collections import Counter
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Analyze target sample to build a bigram profile
    target_tokens = []
    for s in target_sample:
        target_tokens.extend(enc.encode(s))
    
    target_bigrams = Counter()
    for i in range(len(target_tokens) - 1):
        target_bigrams[(target_tokens[i], target_tokens[i+1])] += 1
    
    # We need a threshold for what constitutes a "similar" bigram
    # Using a simple density estimate from the target_sample
    total_target_bigrams = sum(target_bigrams.values())
    
    scored_docs = []
    seen_hashes = set()

    for i, doc in enumerate(docs):
        if not doc or len(doc) < 100:
            continue
        
        # 1. Purity Metric (Strict)
        purity = sum(1 for char in doc if char.isalnum() or char == ' ') / len(doc)
        if purity < 0.9:
            continue
            
        # 2. Structural Filtering
        # Must have '= Title =' style header
        has_title = bool(re.search(r'^\s*=\s*\w+\s*=', doc, re.MULTILINE))
        # WikiText-style punctuation count > 50
        punctuation_count = sum(1 for char in doc if char in '.,;!?:"()')
        
        if not (has_title and punctuation_count > 50):
            continue

        # 3. Similarity check via bigrams
        doc_tokens = enc.encode(doc)
        doc_bigrams = Counter()
        for j in range(len(doc_tokens) - 1):
            doc_bigrams[(doc_tokens[j], doc_tokens[j+1])] += 1
            
        # Calculate similarity score based on overlap with target bigrams
        match_score = 0
        for b, count in doc_bigrams.items():
            if b in target_bigrams:
                # Weighted by target frequency
                match_score += count * (target_bigrams[b] / total_target_bigrams)
        
        # Normalize score by document size (rough proxy for density)
        similarity = match_score / (len(doc_tokens) / 100.0 + 1)
        
        # 4. Deduplication
        # Use a simple hash of the first 200 chars of the cleaned text
        clean_chunk = "".join(c.lower() for c in doc[:200] if c.isalnum()).encode('ascii', 'ignore')
        h = hash(clean_chunk)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        
        # Store (similarity_score, index)
        scored_docs.append((similarity, i))

    # Sort by similarity descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Select indices
    selected_indices = [idx for score, idx in scored_docs]
    
    if not selected_indices:
        # Fallback to any high-purity docs
        purity_docs = []
        for i, doc in enumerate(docs):
            if len(doc) > 0 and sum(1 for c in doc if c.isalnum()) / len(doc) > 0.8:
                purity_docs.append(i)
        if purity_docs:
            rng.shuffle(purity_docs)
            return purity_docs[:len(docs)]
        return list(range(len(docs)))

    # Ranking/Truncation Logic:
    # We want enough tokens to fill T. 
    # Since docs are cycled, we pick enough to ensure variety, then the cycle takes over.
    # However, the instructions ask to rank and truncate to T tokens.
    # Because the trainer cycles the list, we want to provide the highest quality docs 
    # until we hit the token limit T.
    
    total_tokens_count = 0
    final_indices = []
    
    # Pre-calculate token counts for selected
    doc_token_counts = {}
    for idx in selected_indices:
        # Optimization: only count if we haven't reached T
        if total_tokens_count < T:
            # We don't have to encode everything, just estimate
            doc_token_counts[idx] = len(enc.encode(docs[idx]))
            
    for idx in selected_indices:
        count = doc_token_counts.get(idx, 0)
        if total_tokens_count + count <= T:
            final_indices.append(idx)
            total_tokens_count += count
        else:
            # Add partial or just stop if we are very close
            if total_tokens_count < T:
                final_indices.append(idx)
                total_tokens_count += count
            else:
                break
                
    return final_indices
