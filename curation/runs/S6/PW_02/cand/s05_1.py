import os
import random
import numpy as np
from collections import Counter
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def get_tokens(text):
        return text.lower().split()

    # (1) Tokens
    target_token_lists = [get_tokens(d) for d in target_sample]
    
    # (2) Vocabulary
    # Tokens that occur in at least 3 target docs
    vocab_counts = Counter()
    for tokens in target_token_lists:
        unique_tokens = set(tokens)
        for t in unique_tokens:
            vocab_counts[t] += 1
    vocabulary = {t for t, count in vocab_counts.items() if count >= 3}
    
    # Find top 20 frequent tokens in target
    target_all_tokens = []
    for tokens in target_token_lists:
        target_all_tokens.extend(tokens)
    
    # Fix: Counter.most_common(20) returns a list of tuples [(token, count), ...]
    # We need to iterate through the list and check if the token is in vocabulary
    common_pairs = Counter(target_all_tokens).most_common(20)
    top_20_tokens = [t for t, count in common_pairs if t in vocabulary]
    top_20_set = set(top_20_tokens)

    def score_doc(doc_tokens):
        doc_vocab_tokens = [t for t in doc_tokens if t in vocabulary]
        if not doc_vocab_tokens:
            return 0.0
        
        # Tier 1: fraction of doc's vocabulary tokens among the 20 most frequent tokens, times 2
        overlap = len(set(doc_vocab_tokens) & top_20_set)
        t1 = (overlap / len(doc_vocab_tokens)) * 2.0

        # Tier 2: count of adjacent pairs whose first token is in doc's tier 1 tokens
        # We look at a window of 200 tokens
        window = doc_tokens[:200] if len(doc_tokens) > 200 else doc_tokens
        if len(window) < 2:
            return t1
        
        doc_t1_set = set(doc_vocab_tokens) & top_20_set
        count = 0
        for i in range(len(window) - 1):
            if window[i] in doc_t1_set:
                count += 1
        
        return t1 + float(count)

    # (5) Scoring and Filtering
    scored_docs = []
    for i, d in enumerate(docs):
        tokens = get_tokens(d)
        # Basic length and variety check
        if len(d) < 100:
            continue
        if len(tokens) > 0:
            unique_ratio = len(set(tokens)) / len(tokens)
            if unique_ratio < 0.25:
                continue
        
        score = score_doc(tokens)
        scored_docs.append((score, i))

    if not scored_docs:
        return []

    # Split into two halves for the logic requested
    # Using index parity to split the scored population
    scorer_indices = [scored_docs[i] for i in range(0, len(scored_docs), 2)]
    other_half_indices = [scored_docs[i] for i in range(1, len(scored_docs), 2)]
    
    if not scorer_indices:
        scorer_scores = [s[0] for s in other_half_indices]
    else:
        scorer_scores = [s[0] for s in scorer_indices]
        
    if not scorer_scores:
        # Fallback to all if no scores available
        final_indices = [s[1] for s in scored_docs]
    else:
        # Determine threshold (5th percentile * 0.5)
        threshold = np.percentile(scorer_scores, 5) * 0.5
        final_indices = [s[1] for s in other_half_indices if s[0] > threshold]
        
        # If filtering is too aggressive, fallback to original scored order
        if not final_indices:
            final_indices = [s[1] for s in scored_docs]

    # Deduplicate and shuffle
    unique_final = list(set(final_indices))
    rng.shuffle(unique_final)
    return unique_final
