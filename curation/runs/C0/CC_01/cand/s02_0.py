import os
import random
import numpy as np
from collections import Counter
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initialize tokenizer
    encoder = tiktoken.get_encoding("gpt2")
    
    # Analyze target sample: build a global word frequency map
    # We use word-level to capture semantic overlap as requested
    target_words = Counter()
    for doc in target_sample:
        target_words.update(doc.lower().split())
        
    # Pre-calculate token counts for docs and target sample
    doc_token_counts = []
    doc_scores = []
    
    # We want to score each doc in the pool
    # Score = (Token Count Weight) * (Target Overlap Weight)
    # Since we want to prioritize higher average token count AND higher frequency
    # we will use a scoring mechanism.
    
    for i, doc in enumerate(docs):
        # Get tokens
        tokens = encoder.encode(doc)
        count = len(tokens)
        doc_token_counts.append(count)
        
        # Calculate overlap frequency
        # We count how many tokens in this doc are also in the target_words
        doc_words = doc.lower().split()
        overlap_score = sum(target_words[w] for w in doc_words)
        
        # Scoring formula:
        # High token count provides "meat", high overlap provides "relevance"
        # We normalize slightly to ensure neither dominates completely
        # A score of (Count * log(Overlap + 1)) or similar could work, 
        # but keeping it linear for speed/simplicity:
        score = count * (1 + overlap_score / (len(doc_words) + 1))
        doc_scores.append(score)

    # Create list of (score, index)
    scored_indices = []
    for i in range(len(docs)):
        # We weight high token counts and high overlap
        # Adding a small epsilon to avoid zeros
        score = doc_scores[i]
        scored_indices.append((score, i))
        
    # Sort by score descending to prioritize the best documents
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # To ensure we have enough variety and don't just repeat the best 10 docs 
    # (unless necessary), we take a large chunk of the best docs.
    # However, the prompt asks to "prioritize", so we keep the best.
    # We'll take the top 70% of the sorted list to maintain some diversity 
    # while heavily front-loading the best content.
    best_indices = [idx for score, idx in scored_indices[:int(len(docs) * 0.7)]]
    
    # Shuffle the selection slightly to prevent over-fitting to the exact same 
    # order if many docs have identical scores, but keep the high-quality front.
    # Since the trainer cycles, the order matters.
    
    # Re-shuffle the whole list but keep the high-score ones towards the front
    # Actually, the most effective way to "prioritize" for a cyclic trainer 
    # is to put the best ones first.
    
    # Final indices: top performers first
    return [idx for score, idx in scored_indices]
