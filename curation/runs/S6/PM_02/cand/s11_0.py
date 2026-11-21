import os
import random
import hashlib
import zlib
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use gpt2 encoding for token counts as the trainer uses gpt2
    try:
        encoder = tiktoken.get_encoding("gpt2")
    except:
        # Fallback if tiktoken environment is weird
        encoder = None

    def get_bigrams(text, limit=1000):
        words = text.split()
        if len(words) > limit:
            words = words[:limit]
        bgams = set()
        for i in range(len(words) - 1):
            bgams.add((words[i], words[i+1]))
        return bgams

    ref_bgrams = set()
    for doc in target_sample:
        ref_bgrams.update(get_bigrams(doc))

    doc_data = [] # (hash, score, token_count, original_idx)
    
    for i, doc in enumerate(docs):
        doc_bytes = doc.encode('utf-8')
        if len(doc_bytes) < 10:
            continue
            
        bgams = get_bigrams(doc)
        if not bgams:
            continue
            
        hits = len(bgams.intersection(ref_bgrams))
        hit_rate = hits / len(bgams)
        
        compressed = zlib.compress(doc_bytes)
        comp_ratio = len(compressed) / len(doc_bytes)
        
        # Basic filter for junk/extreme compression
        if comp_ratio < 0.10:
            continue
            
        h = hashlib.sha1(doc_bytes).hexdigest()
        
        # Calculate tokens
        if encoder:
            tokens = len(encoder.encode(doc))
        else:
            tokens = len(doc)
            
        doc_data.append((h, hit_rate, tokens, i))

    # Sort by hit_rate descending
    doc_data.sort(key=lambda x: x[1], reverse=True)
    
    # Filter for high-quality unique documents
    unique_docs = []
    seen_hashes = set()
    for h, score, tokens, idx in doc_data:
        if h not in seen_hashes and score > 0.001:
            unique_docs.append((tokens, idx))
            seen_hashes.add(h)

    if not unique_docs:
        # Fallback to some docs if none pass filters
        return [i % len(docs) for i in range(int(T // 100))]

    # Sort unique docs by token count descending to pick high-value content
    unique_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Extract indices
    best_indices = [item[1] for item in unique_docs]
    
    # Cyclic padding strategy:
    # We need to return indices such that total tokens >= T.
    # Since the trainer cycles the list, we provide a sequence 
    # that, when cycled, covers T.
    # The trainer stops after exactly T tokens.
    
    # We want the model to see the highest quality unique docs first.
    # The trainer cycles the list provided by select().
    # If the sum of tokens in best_indices < T, the trainer's cycle will handle it.
    # If we want to ensure specific ordering, we return best_indices.
    # The trainer reads them in order and cycles.
    
    # To optimize the 300 steps / T tokens constraint:
    # We return the best unique docs. The trainer will cycle through them.
    # This ensures the model sees the best content repeatedly.
    
    rng.shuffle(best_indices) # Shuffle to avoid ordering bias in cyclic repetition
    return best_indices
