"""Curated curator: filters junk and prioritizes content similar to the target sample.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
import re
import hashlib
from collections import Counter
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    scored_indices = []
    seen_hashes = set()
    
    # Pre-calculate target vocabulary
    target_words = set()
    for ts in target_sample:
        target_words.update(re.findall(r'\w{4,}', ts.lower()))

    for i, doc in enumerate(docs):
        # Deduplication using 32-bit hash of first 500 characters
        prefix = doc[:500]
        h = hashlib.md5(prefix.encode('utf-8')).digest()
        h_32 = int.from_bytes(h[:4], byteorder='big')
        if h_32 in seen_hashes:
            continue
        seen_hashes.add(h_32)

        # Word extraction
        words = re.findall(r'\w+', doc.lower())
        num_words = len(words)
        if num_words < 10:
            continue

        # Repetition Filter
        sample_words = words if num_words < 500 else words[:500]
        word_counts = Counter(sample_words)
        max_freq = max(word_counts.values()) if word_counts else 0
        if max_freq / len(sample_words) > 0.4:
            continue

        # Scoring
        score = 0.0
        
        # Similarity to target
        overlap_count = sum(1 for w in words[:200] if w in target_words)
        score += (overlap_count / (num_words ** 0.5))

        # Wiki-style Structure Density
        if doc.startswith('= '):
            # Markers: '@-@', ' ( ', ', ', ' . '
            # We count occurrences of these specific substrings
            markers = ['@-@', ' ( ', ', ', ' . ']
            marker_count = sum(doc.count(m) for m in markers)
            
            # Tokens in first 500 words (approximation of "tokens in first 500 tokens")
            first_500_tokens = enc.encode(doc[:10000])[:500] # Safety cap
            token_count = len(first_500_tokens)
            
            structure_density = marker_count / max(token_count, 1)
            score += structure_density
        
        scored_indices.append((score, i))

    # Sort by score descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Select until T is reached
    selected_indices = []
    current_tokens = 0
    
    # We want to prioritize the best ones but don't need to cycle a tiny number
    # If we run out of indices, we'll cycle the ones we have.
    for score, idx in scored_indices:
        # Estimate length using a small sample of docs to get an average 
        # or just calculate for the doc
        doc_len_tokens = len(enc.encode(docs[idx]))
        
        if current_tokens + doc_len_tokens <= T:
            selected_indices.append(idx)
            current_tokens += doc_len_tokens
        else:
            # Fill remaining space with the best available docs (cycling)
            remaining = T - current_tokens
            # Pick indices that best fit the remaining space
            for i in range(len(scored_indices)):
                idx_to_add = scored_indices[i][1]
                doc_len_tokens = len(enc.encode(docs[idx_to_add]))
                if current_tokens + doc_len_tokens <= remaining:
                    selected_indices.append(idx_to_add)
                    current_tokens += doc_len_tokens
                if current_tokens >= remaining:
                    break
            break

    # If we somehow have 0 indices (all docs filtered), return first doc
    if not selected_indices:
        return [0]
        
    # Ensure we have enough to cover T by cycling the best ones if necessary
    # The logic above ensures we fill as much as possible. 
    # If current_tokens < T, the trainer will cycle them.
    
    return selected_indices
