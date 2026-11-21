import os
import random
import zlib
import re
import math
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Build vocabulary from target_sample
    V = set()
    for doc in target_sample:
        words = re.findall(r'[a-zA-Z]+', doc.lower())
        for w in words:
            V.add(w)
            
    unique_docs = []
    seen_hashes = set()
    
    for i, doc in enumerate(docs):
        # Basic length and compression filters
        if len(doc) < 200:
            continue
        
        compressed_ratio = len(zlib.compress(doc.encode('utf-8'))) / len(doc)
        if compressed_ratio < 0.30:
            continue
            
        doc_lower = doc.lower()
        spam_phrases = [
            'click here to order', 'buy now', 'free shipping', 
            'discount code', 'coupon', 'for sale |', 
            'limited offer', 'best price'
        ]
        if any(phrase in doc_lower for phrase in spam_phrases):
            continue

        # Scramble check
        scramble_tokens = re.findall(r'[a-zA-Z]{4,}', doc_lower)
        if len(scramble_tokens) >= 20:
            match_count = sum(1 for w in scramble_tokens if w in V)
            if match_count / len(scramble_tokens) < 0.5:
                continue

        # Word-shuffle checks
        tokens = doc.split()
        if len(tokens) > 0:
            # 'the' preceding token check
            pre_the_count = 0
            the_count = 0
            pre_words = {'of', 'in', 'to', 'on', 'and', 'at', 'for', 'by', 'from', 'with', 'as', 'that', 'was', 'is'}
            for j in range(len(tokens)):
                if tokens[j].lower() == 'the':
                    the_count += 1
                    if j > 0 and tokens[j-1].lower() in pre_words:
                        pre_the_count += 1
            if the_count >= 10 and (pre_the_count / the_count) < 0.25:
                continue

            # Punctuation check
            dot_count = 0
            dot_followed_lower = 0
            for j in range(len(tokens) - 1):
                t = tokens[j]
                if t == '.' or t.endswith('.'):
                    dot_count += 1
                    next_t = tokens[j+1]
                    if next_t and next_t[0].islower():
                        dot_followed_lower += 1
            if dot_count >= 8 and (dot_followed_lower / dot_count) > 0.4:
                continue
            
            # Adjacent identical function-word pairs
            pair_count = 0
            if len(tokens) >= 4:
                for j in range(len(tokens) - 3):
                    chunk = [t.lower() for t in tokens[j:j+4]]
                    if chunk == ['the', 'of', 'the', 'the']:
                        pair_count += 1
            for j in range(len(tokens) - 1):
                pair = (tokens[j].lower(), tokens[j+1].lower())
                if pair in [('the', 'the'), ('of', 'of'), ('a', 'the'), ('the', 'a')]:
                    pair_count += 1
            if (pair_count / len(tokens)) > 0.005:
                continue

        # Deduplication
        norm_text = " ".join(tokens)
        h = hash(norm_text)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # Scoring Calculation
        # Wiki markers
        markers = doc.count('= ') + doc.count(' , ') + doc.count(' . ') + doc.count(' @-@ ')
        
        # Vocabulary overlap
        doc_words = re.findall(r'[a-zA-Z]+', doc_lower)
        overlap_count = sum(1 for w in doc_words if w in V)
        vocab_overlap = overlap_count / len(doc_words) if doc_words else 0
        
        # Formula: (Wiki markers * 2 + Vocabulary Overlap * 1.5) / log10(length)
        # Using max(1, log10) to avoid division by zero or negative numbers
        log_len = math.log10(len(doc))
        score = (markers * 2 + vocab_overlap * 1.5) / max(1.0, log_len)
        
        unique_docs.append((score, i, len(doc)))

    # Sort by score descending, then original index
    unique_docs.sort(key=lambda x: (-x[0], x[1]))
    
    kept_indices = []
    current_tokens = 0
    
    for score, idx, length in unique_docs:
        if current_tokens < T:
            kept_indices.append(idx)
            current_tokens += length
        else:
            break
            
    return kept_indices
