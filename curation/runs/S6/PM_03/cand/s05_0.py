import os
import random
import re
import hashlib
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Pre-process target sample to build a bigram profile
    target_bigrams = set()
    target_words = []
    for doc in target_sample:
        words = re.findall(r'\w+', doc.lower())
        target_words.extend(words)
        for i in range(len(words) - 1):
            target_bigrams.add((words[i], words[i+1]))
    
    # Scoring and Deduplication
    seen_hashes = set()
    unique_docs_indices = []
    
    # Helper to normalize and hash
    def get_doc_hash(d):
        norm = " ".join(d.split())
        return hashlib.md5(norm[:200].encode('utf-8')).hexdigest()

    # First pass: Deduplicate and Filter for "Quality"
    # We evaluate quality by bigram hit rate against target_sample
    for i, doc in enumerate(docs):
        h = get_doc_hash(doc)
        if h in seen_hashes:
            continue
        
        # Simple heuristic: doc must be long enough to have bigrams
        words = re.findall(r'\w+', doc.lower())
        if len(words) < 2:
            continue
            
        # Calculate bigram hit rate
        hits = 0
        for j in range(len(words) - 1):
            if (words[j], words[j+1]) in target_bigrams:
                hits += 1
        
        # Filter: remove documents with 0 bigram hits (highly likely word-shuffled)
        # unless the doc is very short (where shuffling is hard to detect)
        if hits > 0 or len(words) < 5:
            seen_hashes.add(h)
            unique_docs_indices.append(i)

    if not unique_docs_indices:
        unique_docs_indices = list(range(len(docs)))

    # Shuffle the filtered pool
    rng.shuffle(unique_docs_indices)
    
    # Expand to ensure we cover T tokens (the trainer handles cycling, 
    # but we return indices for the duration of T). 
    # Since the trainer stops at T and cycles, returning the unique pool 
    # is sufficient as the trainer will handle the wrap-around.
    # However, to be safe on length, we return the list.
    return unique_docs_indices
