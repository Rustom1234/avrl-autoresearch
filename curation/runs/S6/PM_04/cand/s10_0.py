import os
import random
import re
import zlib
from collections import Counter
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    target_tokens = []
    for s in target_sample:
        target_tokens.extend(enc.encode(s))
    
    target_counts = Counter(target_tokens)
    target_bigrams = {}
    for i in range(len(target_tokens) - 1):
        b = (target_tokens[i], target_tokens[i+1])
        target_bigrams[b] = target_bigrams.get(b, 0) + 1

    scored_docs = []

    for i, doc in enumerate(docs):
        if not doc or len(doc) < 50:
            continue

        # 1. Boilerplate and Spam Filtering
        # Compression ratio check (higher ratio = more repetitive)
        compressed = zlib.compress(doc.encode('utf-8'))
        compression_ratio = len(compressed) / len(doc)
        
        # Line uniqueness ratio
        lines = [l.strip() for l in doc.splitlines() if l.strip()]
        if not lines:
            continue
        line_counts = Counter(lines)
        unique_lines = len(line_counts)
        line_uniqueness = unique_lines / len(lines)

        # Filter out highly repetitive content or spam templates
        if compression_ratio > 0.5 or line_uniqueness < 0.6:
            continue

        # 2. Quality Scoring
        # ASCII Letter Density
        letters = sum(1 for char in doc if char.isalpha())
        letter_density = letters / len(doc)
        
        # WikiText structural purity (Title detection and punctuation balance)
        has_title = bool(re.search(r'^\s*=\s*\w+\s*=', doc, re.MULTILINE))
        punctuation = sum(1 for char in doc if char in '.,;!?:"()')
        struct_score = 0
        if has_title:
            struct_score += 0.5
        if 0 < punctuation < len(doc) * 0.15:
            struct_score += 0.5

        # Bigram Overlap
        doc_tokens = enc.encode(doc)
        bigram_score = 0
        total_bigrams = 0
        for j in range(len(doc_tokens) - 1):
            b = (doc_tokens[j], doc_tokens[j+1])
            total_bigrams += 1
            bigram_score += target_bigrams.get(b, 0)
        
        bigram_likelihood = bigram_score / total_bigrams if total_bigrams > 0 else 0

        # Weighted Scoring
        # Weightings: Bigrams (0.5), Letter Density (0.3), Structure (0.2)
        score = (bigram_likelihood * 100 * 0.5) + (letter_density * 100 * 0.3) + (struct_score * 100 * 0.2)
        
        # Basic threshold for quality
        if letter_density > 0.7 and score > 1.0:
            scored_docs.append((score, i))

    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Deduplicate by normalized content
    seen_normalized = set()
    final_indices = []
    
    for score, idx in scored_docs:
        doc = docs[idx]
        normalized = "".join(c.lower() for c in doc if c.isalnum())
        if len(normalized) > 20 and normalized not in seen_normalized:
            seen_normalized.add(normalized)
            final_indices.append(idx)
        
        if len(final_indices) >= 1000: # Safety break
            break

    if not final_indices:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    rng.shuffle(final_indices)
    return final_indices
