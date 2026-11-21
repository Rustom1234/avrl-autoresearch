import os
import random
import re
from collections import Counter
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Pre-process target_sample for bigram likelihood
    # Use tiktoken to get token IDs for the target sample
    enc = tiktoken.get_encoding("gpt2")
    target_tokens = []
    for s in target_sample:
        target_tokens.extend(enc.encode(s))
    
    # Build a frequency map of tokens in target_sample
    target_counts = Counter(target_tokens)
    # Build a bigram map for target_sample
    target_bigrams = {}
    for i in range(len(target_tokens) - 1):
        b = (target_tokens[i], target_tokens[i+1])
        target_bigrams[b] = target_bigrams.get(b, 0) + 1

    scored_indices = []
    seen_normalized = set()
    
    # Heuristic parameters
    # We want to find documents that "look" like the target domain
    # and have high structural integrity.
    
    for i, doc in enumerate(docs):
        # 1. Purity Metric (ASCII letters and spaces)
        if not doc: continue
        purity = sum(1 for char in doc if (char.isalnum() or char == ' ')) / len(doc)
        if purity < 0.8: continue

        # 2. WikiText-style structural heuristics
        # Check for title lines and punctuation balance
        has_title = bool(re.search(r'^\s*=\s*\w+\s*=', doc, re.MULTILINE))
        punctuation_count = sum(1 for char in doc if char in '.,;!?:"()')
        # A heuristic: some punctuation is good, but not too much (noise)
        # and not zero (non-informative).
        is_structured = has_title and (0 < punctuation_count < len(doc) * 0.1)

        # 3. Bigram Likelihood Check
        doc_tokens = enc.encode(doc)
        bigram_score = 0
        total_bigrams = 0
        for j in range(len(doc_tokens) - 1):
            b = (doc_tokens[j], doc_tokens[j+1])
            total_bigrams += 1
            if b in target_bigrams:
                bigram_score += target_bigrams[b]
        
        bigram_likelihood = bigram_score / total_bigrams if total_bigrams > 0 else 0

        # Composite Score
        # Weighted toward bigram likelihood (relevance) and purity (quality)
        # We filter for a "high" threshold: 
        # purity > 0.85 AND (bigram_likelihood > 0.01 OR is_structured)
        if purity > 0.85 and (bigram_likelihood > 0.01 or is_structured):
            # Deduplication: Normalize text (lower, strip non-alphanumeric)
            normalized = "".join(c.lower() for c in doc if c.isalnum()).replace(" ", "")
            if normalized not in seen_normalized and len(normalized) > 10:
                seen_normalized.add(normalized)
                scored_indices.append(i)

    # If we found nothing, fallback to original logic or just the first few
    if not scored_indices:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    # Shuffle the filtered indices
    rng.shuffle(scored_indices)
    return scored_indices
