import os
import random
import numpy as np
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    tik = tiktoken.get_encoding("gpt2")

    def get_tk_count(text):
        try:
            # The error was caused by tik.encode(text, -1) having 3 arguments
            # but the method signature was interpreted as taking 2.
            # Actually, tik.encode(text) is standard. -1 is for 'no_special_tokens'
            # but depending on version, it might be a keyword or unnecessary.
            # Standard tiktoken.encode(text) is safest.
            return len(tik.encode(text))
        except:
            return len(text) // 4

    def get_probs(docs_subset, target_tokens_set):
        scores = []
        for d in docs_subset:
            tokens = tik.encode(d.lower())
            count = sum(1 for t in tokens if t in target_tokens_set)
            scores.append(count / max(len(tokens), 1))
        return scores

    # Pre-process Target Sample
    target_tokens_set = set()
    target_tk_counts = []
    for d in target_sample:
        tokens = tik.encode(d.lower())
        target_tokens_set.update(tokens)
        target_tk_counts.append(len(tokens))
    
    avg_target_tk_rate = sum(target_tk_counts) / max(len(target_tk_counts), 1)

    # Score the docs based on token overlap with the target sample
    # Using a subset for speed to stay under 180s
    num_docs = len(docs)
    scored_docs = []
    
    # Limit iteration to stay within time limits
    limit = min(num_docs, 2000)
    for i in range(limit):
        d = docs[i]
        tokens = tik.encode(d.lower())
        match = sum(1 for t in tokens if t in target_tokens_set)
        score = match / max(len(tokens), 1)
        scored_docs.append((score, i))

    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Select a high-quality subset
    # We want to cover T tokens.
    selected_indices = []
    current_tk = 0
    
    # Take a good chunk of the best documents
    # We use a greedy approach to fill T tokens
    for score, idx in scored_docs:
        tk = get_tk_count(docs[idx])
        selected_indices.append(idx)
        current_tk += tk
        if current_tk >= T:
            break

    # If we didn't reach T, cycle the best ones
    if current_tk < T:
        while current_tk < T:
            for idx in selected_indices:
                current_tk += get_tk_count(docs[idx])
                selected_indices.append(idx)
                if current_tk >= T:
                    break

    # Ensure we don't exceed 3T (hard limit of the trainer logic or memory)
    # The prompt implies we should provide a robust list.
    # We will return the indices in the order they appear in our selection.
    
    # Final shuffle to provide variety while maintaining high quality
    rng.shuffle(selected_indices)
    
    return selected_indices
