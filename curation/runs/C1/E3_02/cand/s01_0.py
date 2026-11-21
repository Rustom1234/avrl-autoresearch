"""Curated curator: filters junk and prioritizes content similar to the target sample.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
import re
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Calculate target vocabulary
    # Flatten target sample to get a sense of common meaningful words
    target_words = set()
    for doc in target_sample:
        # Basic cleanup for word extraction
        words = re.findall(r'\w{4,}', doc.lower())
        target_words.update(words)
    
    # 2. Score and Filter documents
    # We want to avoid "junk" (repetitive, short, or spammy)
    # and prioritize "useful" (high keyword overlap with target).
    doc_scores = []
    valid_indices = []
    
    for i, doc in enumerate(docs):
        # Rule 1: Length filter (ignore very short fragments)
        if len(doc.split()) < 10:
            continue
            
        # Rule 2: Repetition/Spam filter
        # If a document is heavily repetitive (e.g., "Click here" many times), score it low
        words = re.findall(r'\w+', doc.lower())
        word_counts = Counter(words)
        if len(words) > 0:
            most_common_count = word_counts.most_common(1)[0][1]
            if most_common_count / len(words) > 0.4: # High repetition threshold
                continue
        
        # Rule 3: Similarity score
        # Count overlap with target sample words
        score = 0
        for w in words:
            if w in target_words:
                score += 1
        
        # Normalize score by document length to prioritize "dense" content
        # and add a small bias for longer documents to get more data
        score = score / (len(words) ** 0.5) if len(words) > 0 else 0
        doc_scores.append((score, i))
        valid_indices.append(i)

    # Sort by score descending
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    # 3. Selection
    # Take the best documents. Since we need a lot of tokens (T), 
    # we take a large portion of the high-scoring ones.
    # The trainer cycles the list, so we want a high-quality repeating sequence.
    best_indices = [idx for score, idx in doc_scores[:len(docs)//2]]
    
    # Shuffle the high-quality subset to maintain some variety
    rng.shuffle(best_indices)
    
    return best_indices
