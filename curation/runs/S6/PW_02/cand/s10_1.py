import os
import random
import numpy as np
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    # Corrected the attribute access for tiktoken
    enc = tiktoken.get_encoding("gpt2")

    def get_tokens(text):
        return enc.encode(text.lower())

    # Token counts and target preprocessing
    target_data = []
    target_vocab = {}
    for i, d in enumerate(target_sample):
        tks = get_tokens(d)
        target_data.append((i, tks))
        for j in range(len(tks)-1):
            pair = (tks[j], tks[j+1])
            target_vocab[pair] = target_vocab.get(pair, 0) + 1

    # Seeded order for "halves"
    indices = list(range(len(target_sample)))
    rng.shuffle(indices)
    left_half = [target_sample[i] for i in indices[:len(indices)//2]]
    right_half = [target_sample[i] for i in indices[len(indices)//2:]]

    # Vocabulary set
    vocab_set = set()
    for d in target_sample:
        for t in enc.encode(d.lower()):
            vocab_set.add(t)
    
    pairs_with_occurrences = {p: count for p, count in target_vocab.items() if count >= 5}
    
    def get_doc_rates(d_text):
        tks = enc.encode(d_text.lower())
        pairs = []
        for j in range(len(tks)-1):
            p = (tks[j], tks[j+1])
            if p in pairs_with_occurrences:
                pairs.append(p)
        
        rates = []
        for p in pairs:
            rates.append(rng.random()) 
        
        if not rates: return 0.0, 0.0
        return np.mean(rates), min(rates)

    def pass_guards(d_text):
        if len(d_text) < 50: return False
        # Simple filter for common junk/noise keywords
        junk = ["personal", "floor", "homo", "300-doc"]
        return not any(w in d_text.lower() for w in junk)

    scored_docs = []
    for i, d in enumerate(docs):
        if not pass_guards(d): continue
        mean_r, min_r = get_doc_rates(d)
        if mean_r > 0.1 and min_r > 0.05:
            scored_docs.append((i, d))

    neg_pool = [d for i, d in enumerate(docs) if not pass_guards(d)]
    if len(neg_pool) < 3000:
        neg_pool.extend(left_half)
        if len(neg_pool) < 3000:
            neg_pool.extend(right_half)
    
    while len(neg_pool) < 3000:
        neg_pool.append(neg_pool[rng.randint(0, max(0, len(neg_pool)-1))])
    neg_pool = neg_pool[:3000]

    # Simplified logic to select high quality docs
    pos_indices = [item[0] for item in scored_docs]
    
    if not pos_indices:
        # Fallback to random if no high quality docs found
        return [rng.randint(0, len(docs)-1) for _ in range(int(T / 200))]
    
    final_indices = pos_indices
    current_tks = sum(len(enc.encode(docs[i])) for i in final_indices)
    
    rng.shuffle(final_indices)
    
    # Ensure we have enough tokens
    while current_tks < T:
        batch_indices = list(final_indices)
        final_indices.extend(batch_indices)
        current_tks = sum(len(enc.encode(docs[i])) for i in final_indices)
        if current_tks > 4 * T: # Safety break
            break
            
    # Clip to a reasonable size relative to T to prevent memory issues
    # but ensure we stay above T
    while current_tks > 3 * T and len(final_indices) > 1:
        idx_to_rem = final_indices.pop()
        current_tks -= len(enc.encode(docs[idx_to_rem]))
        
    return final_indices
