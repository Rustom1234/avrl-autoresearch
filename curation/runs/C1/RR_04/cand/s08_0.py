import os
import random
import zlib
import re
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Prep V: set of lowercased words from target_sample
    V = set()
    for doc in target_sample:
        words = re.findall(r'[a-zA-Z]+', doc.lower())
        for w in words:
            V.add(w)
            
    unique_docs = []
    seen_hashes = set()
    
    for i, doc in enumerate(docs):
        # Initial filters
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

        # Word-shuffle check
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
            
            # Adjacent function-word pairs
            pair_count = 0
            if len(tokens) >= 4:
                for j in range(len(tokens) - 3):
                    chunk = [t.lower() for t in tokens[j:j+3]] # simplified check
                    if chunk == ['the', 'of', 'the']:
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

        # Scoring
        # Wiki score
        markers = doc.count('= ') + doc.count(' , ') + doc.count(' . ') + doc.count(' @-@ ')
        wiki_score = markers / len(doc)
        
        # Vocabulary Similarity Score
        # Ratio of tokens (first 500) that appear in V
        first_500_tokens = tokens[:500]
        if first_500_tokens:
            v_matches = sum(1 for t in first_500_tokens if t.lower() in V)
            vocab_sim = v_matches / len(first_500_tokens)
        else:
            vocab_sim = 0.0
            
        unique_docs.append((wiki_score, vocab_sim, len(doc), i))

    # Sort primarily by wiki_score (desc), then vocab_sim (desc), then original index (asc)
    unique_docs.sort(key=lambda x: (-x[0], -x[1], x[3]))
    
    kept_indices = []
    current_tokens = 0
    
    for score, v_sim, length, idx in unique_docs:
        if current_tokens < T:
            kept_indices.append(idx)
            current_tokens += length
            
    return kept_indices
