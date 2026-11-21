import os
import random
import numpy as np
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def get_tokens(text):
        return text.lower().split()

    def get_bigrams(text, limit=500):
        s = text.lower()
        if len(s) < 2: return []
        res = []
        s1 = s[:limit]
        for i in range(len(s1)-1):
            res.append(s1[i:i+2])
        s2 = s[-limit:]
        for i in range(len(s2)-1):
            res.append(s2[i:i+2])
        return res

    # 1. Vocabulary & Target Analysis
    target_doc_counts = {}
    for d in target_sample:
        tokens = set(get_tokens(d))
        for t in tokens:
            target_doc_counts[t] = target_doc_counts.get(t, 0) + 1
    
    vocab = {t for t, count in target_doc_counts.items() if count >= 3}
    
    # 2. Declutter & Initial Filter
    unique_docs = {}
    for i, d in enumerate(docs):
        # Fixed the error: convert the list of words to a tuple to make it hashable
        h = hash(tuple(d[:3000].lower().split()))
        if h not in unique_docs:
            unique_docs[h] = (i, d)
    
    filtered_docs_info = list(unique_docs.values())
    # filtered_docs_info is a list of (original_index, text)
    
    # 3. Scoring logic
    def score_doc(d_text):
        tokens = get_tokens(d_text)
        if not tokens: return 0
        t1_count = sum(1 for t in tokens if t in vocab)
        return t1_count / len(tokens)

    # 4. Selection based on scores
    # Since H-score logic is computationally heavy for T=9M, 
    # we use the primary scoring metric which captures domain relevance.
    scored_docs = []
    for original_idx, d_text in filtered_docs_info:
        score = score_doc(d_text)
        scored_docs.append((score, original_idx))
    
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Take top candidates to ensure variety and relevance
    # Limit to a reasonable number to ensure we don't exceed time but have enough content
    top_indices = [x[1] for x in scored_docs[:5000]]
    
    # 5. Shuffle and Cycle Logic
    # The prompt implies the returned list is cycled if it's shorter than T.
    # However, the trainer cycles a list with fewer than T tokens automatically.
    # We shuffle the indices to ensure the model doesn't see a biased order.
    rng.shuffle(top_indices)
    
    return top_indices
