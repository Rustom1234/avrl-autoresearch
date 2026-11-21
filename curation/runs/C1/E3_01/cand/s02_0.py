import os
import random
import re
import tiktoken
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    def get_tokens(text):
        return enc.encode(text)

    def get_content_set(tokens):
        # Simple content filter: remove common punctuation and single-char tokens
        # and treat tokens starting with '.' or ending with '.' as likely punctuation
        content = set()
        stop_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "of", "in", "for", "with", "on", "at", "by", "from", "this", "that", "it", "as", "be", "has", "had", "have"}
        for t in tokens:
            decoded = enc.decode([t]).strip().lower()
            if len(decoded) > 1 and decoded not in stop_words and not decoded.isnumeric():
                # Basic check to filter out purely punctuation-like strings
                if not re.match(r'^[^\w\s]+$', decoded):
                    content.add(decoded)
        return content

    # Analyze target sample
    target_tokens_sets = []
    total_target_tokens = 0
    for doc in target_sample:
        tokens = get_tokens(doc)
        target_tokens_sets.append(get_content_set(tokens))
        total_target_tokens += len(tokens)
    
    # Create a global content vocabulary for the target domain
    target_vocab = set()
    for s in target_tokens_sets:
        target_vocab.update(s)

    doc_scores = []
    for i, doc in enumerate(docs):
        tokens = get_tokens(doc)
        n = len(tokens)
        
        # Filter out extremely short or empty docs
        if n < 50:
            doc_scores.append((0, i))
            continue
            
        # Junk detection: high repetition (simple heuristic)
        if len(set(tokens)) < n * 0.3 and n > 100:
            doc_scores.append((0, i))
            continue

        content = get_content_set(tokens)
        
        # Jaccard-like similarity against the target vocabulary
        intersection = len(content.intersection(target_vocab))
        union = len(content.union(target_vocab))
        similarity = intersection / union if union > 0 else 0
        
        # Score is a mix of similarity and length (prefer informative content)
        # Weighting similarity heavily
        score = similarity * 1000 + (n / total_target_tokens)
        doc_scores.append((score, i))

    # Sort by score descending
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    # To ensure we don't just repeat the best doc forever, 
    # we use a weighted distribution or a simple top-K variety.
    # Given the constraints, a sorted list with high-quality docs first is best.
    # We return the sorted indices.
    
    indices = [item[1] for item in doc_scores]
    return indices
