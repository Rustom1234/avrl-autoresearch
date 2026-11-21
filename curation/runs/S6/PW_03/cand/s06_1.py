import os
import random
import zlib
import numpy as np
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.encoding_for_model("gpt2")
    
    # 1. Build c_set from target_sample
    c_set = set()
    target_text = "".join(target_sample).lower()
    for i in range(len(target_text) - 1):
        b = target_text[i:i+2]
        c_set.add(zlib.adler32(b.encode('utf-8')) & 0xFFFFFFFF)
        
    seen_hashes = set()
    processed_docs = []
    
    # Filter and score
    for doc in docs:
        if len(doc) < 200:
            continue
            
        norm_text = doc.lower().split()
        norm_text = " ".join(norm_text)
        norm_text = norm_text[:4000]
        h_hash = zlib.adler32(norm_text.encode('utf-8')) & 0xFFFFFFFF
        
        if h_hash in seen_hashes:
            continue
        seen_hashes.add(h_hash)
        
        doc_low = doc.lower()
        count = 0
        total = 0
        for i in range(len(doc_low) - 1):
            b = doc_low[i:i+2]
            total += 1
            if zlib.adler32(b.encode('utf-8')) & 0xFFFFFFFF in c_set:
                count += 1
        h = count / total if total > 0 else 0.0
        processed_docs.append((doc, h))
        
    # 2. Filter by lo
    scores = [x[1] for x in processed_docs]
    if not scores:
        return []
    
    scores_sorted = sorted(scores)
    p2_5 = scores_sorted[int(len(scores_sorted) * 0.025)]
    p95 = scores_sorted[int(len(scores_sorted) * 0.95)]
    lo = (p2_5 + p95) / 2
    
    filtered_docs = [x for x in processed_docs if x[1] >= lo]
    # Sort by descending h
    filtered_docs.sort(key=lambda x: x[1], reverse=True)
    
    # 3. Token count and selection
    potential_docs = []
    for i, (doc, h) in enumerate(filtered_docs):
        tokens = enc.encode(doc)
        # We store the original index in docs
        # Since processed_docs was derived from docs, we need original index.
        # Let's refine the loop to keep track of original index.
        pass
    
    # Re-doing logic to maintain correct original indices
    processed_with_idx = []
    seen_hashes = set()
    for i, doc in enumerate(docs):
        if len(doc) < 200: continue
        norm_text = doc.lower().split()
        norm_text = " ".join(norm_text)
        norm_text = norm_text[:4000]
        h_hash = zlib.adler32(norm_text.encode('utf-8')) & 0xFFFFFFFF
        if h_hash in seen_hashes: continue
        seen_hashes.add(h_hash)
        
        doc_low = doc.lower()
        count = 0
        total = 0
        for j in range(len(doc_low) - 1):
            b = doc_low[j:j+2]
            total += 1
            if zlib.adler32(b.encode('utf-8')) & 0xFFFFFFFF in c_set:
                count += 1
        h = count / total if total > 0 else 0.0
        processed_with_idx.append((i, doc, h))
        
    scores = [x[2] for x in processed_with_idx]
    if not scores: return []
    scores_sorted = sorted(scores)
    lo = (scores_sorted[int(len(scores_sorted) * 0.025)] + scores_sorted[int(len(scores_sorted) * 0.95)]) / 2
    
    filtered = [x for x in processed_with_idx if x[2] >= lo]
    filtered.sort(key=lambda x: x[2], reverse=True)
    
    potential_docs = []
    for idx, doc, h in filtered:
        tokens = enc.encode(doc)
        potential_docs.append((idx, doc, len(tokens)))
        
    final_indices = []
    current_tokens = 0
    for i in range(len(potential_docs)):
        idx, doc, count = potential_docs[i]
        if current_tokens + count <= T:
            final_indices.append(idx)
            current_tokens += count
        else:
            # If it's the very last one and we are under T, take it
            if i == len(potential_docs) - 1 and current_tokens < T:
                final_indices.append(idx)
            break

    rng.shuffle(final_indices)
    return final_indices
