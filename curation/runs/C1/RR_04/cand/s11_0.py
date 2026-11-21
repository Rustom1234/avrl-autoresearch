import os
import random
import zlib
import re
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Prep Target Vocabulary
    V = set()
    for doc in target_sample:
        words = re.findall(r'[a-zA-Z]+', doc.lower())
        for w in words:
            V.add(w)
            
    # 2. Scoring and Filtering
    # We use a dictionary or list to store (score, overlap, original_index, length)
    candidates = []
    seen_hashes = set()
    
    # Markers from requirements
    markers = [' , ', ' . ', ' @-@ ']
    # Weights for markers (Higher weight to specific target markers)
    # Since ' , ' and ' . ' are standard, we can weight them slightly lower than ' @-@ '
    marker_weights = {' , ': 1.0, ' . ': 1.0, ' @-@ ': 3.0}
    
    for i, doc in enumerate(docs):
        # Basic Pre-filtering
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

        # Letter-scramble check
        scramble_tokens = re.findall(r'[a-zA-Z]{4,}', doc_lower)
        if len(scramble_tokens) >= 20:
            match_count = sum(1 for w in scramble_tokens if w in V)
            if match_count / len(scramble_tokens) < 0.5:
                continue

        # Word-shuffle check
        tokens = doc.split()
        if len(tokens) > 0:
            # (a) 'the' preceding token check
            the_count = 0
            pre_the_count = 0
            pre_words = {'of', 'in', 'to', 'on', 'and', 'at', 'for', 'by', 'from', 'with', 'as', 'that', 'was', 'is'}
            for j in range(len(tokens)):
                if tokens[j].lower() == 'the':
                    the_count += 1
                    if j > 0 and tokens[j-1].lower() in pre_words:
                        pre_the_count += 1
            if the_count >= 10 and (pre_the_count / the_count) < 0.25:
                continue

            # (b) '.' punctuation check
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
            
            # (c) Adjacent identical function-word pairs
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
        nh = hash(norm_text)
        if nh in seen_hashes:
            continue
        seen_hashes.add(nh)

        # Weighted Wiki Density
        density_score = 0
        for m_str, weight in marker_weights.items():
            density_score += doc.count(m_str) * weight
        density_score /= len(doc)

        # Vocabulary Overlap Score (first 300 chars)
        prefix = doc[:300].lower()
        prefix_tokens = re.findall(r'[a-z]+', prefix)
        overlap_count = sum(1 for w in prefix_tokens if w in V)
        overlap_score = overlap_count / len(prefix_tokens) if prefix_tokens else 0
        
        candidates.append({
            'score': density_score,
            'overlap': overlap_score,
            'idx': i,
            'len': len(doc)
        })

    # Sort by Weighted Wiki Density (desc), then Vocabulary Overlap (desc)
    candidates.sort(key=lambda x: (-x['score'], -x['overlap']))
    
    kept_indices = []
    current_tokens = 0
    
    for cand in candidates:
        if current_tokens < T:
            kept_indices.append(cand['idx'])
            current_tokens += cand['len']
            
    return kept_indices
