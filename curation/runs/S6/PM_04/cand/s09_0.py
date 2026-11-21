import os
import random
import zlib
import hashlib
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Pre-calculate target statistics for a basic vocabulary check if needed
    # but the prompt emphasizes hashing, compression, and length.
    
    seen_hashes = set()
    candidates = []
    
    # Heuristic thresholds
    # Minimum tokens to be considered "content"
    MIN_TOKENS = 20
    # Max compression ratio (compressed_size / original_size)
    # High ratio means high entropy (good), low ratio means highly repetitive (junk)
    # We want to avoid documents that compress too well (highly repetitive junk)
    # and documents that are just random noise.
    # A ratio < 0.4 often suggests extreme repetition (e.g., "aaaaa...")
    MIN_COMPRESSION_RATIO = 0.4 

    for i, doc in enumerate(docs):
        if not doc:
            continue
            
        # 1. Exact text hashing for deduplication
        doc_hash = hashlib.md5(doc.encode('utf-8')).hexdigest()
        if doc_hash in seen_hashes:
            continue
            
        # 2. Tokenize to check length
        tokens = enc.encode(doc)
        if len(tokens) < MIN_TOKENS:
            continue
            
        # 3. zlib compression ratio check
        # We use a small chunk to avoid overhead, but the doc is the unit
        data = doc.encode('utf-8')
        compressed = zlib.compress(data)
        ratio = len(compressed) / len(data)
        
        if ratio < MIN_COMPRESSION_RATIO:
            continue
            
        # If we passed the pragmatic checks
        seen_hashes.add(doc_hash)
        candidates.append(i)

    # Fallback if filtering is too aggressive
    if not candidates:
        indices = list(range(len(docs)))
        rng.shuffle(indices)
        return indices

    # Shuffle the filtered indices as requested
    rng.shuffle(candidates)
    return candidates
