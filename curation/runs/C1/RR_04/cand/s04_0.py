import os
import random
import zlib
import re
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Prep V (alphabetical words in target_sample)
    V = set()
    for doc in target_sample:
        words = re.findall(r'[a-zA-Z]+', doc.lower())
        for w in words:
            V.add(w)
            
    kept_indices = []
    seen_full_hashes = set()
    seen_prefix_hashes = set()

    for i, doc in enumerate(docs):
        # 3. Spam and template check
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

        # 1. Letter-scramble check
        # Use regex for alphabetic tokens >= 4
        scramble_tokens = re.findall(r'[a-zA-Z]{4,}', doc.lower())
        if len(scramble_tokens) >= 20:
            match_count = sum(1 for w in scramble_tokens if w in V)
            if match_count / len(scramble_tokens) < 0.5:
                continue

        # 2. Word-shuffle check
        tokens = doc.split()
        if len(tokens) > 0:
            # (a) 'the' preceding token check
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
            # Note: The prompt asks for specific pairs: 'the the', 'of of', 'a the', 'the a', 'the of the the'
            # However, 'the of the the' is 4 tokens, 'a the' is 2.
            # The rule says "adjacent identical function-word pairs ... make up more than 0.5% of its tokens"
            # We interpret "identical function-word pairs" as the specific list provided.
            pair_count = 0
            bad_patterns = [('the', 'the'), ('of', 'of'), ('a', 'the'), ('the', 'a'), ('the', 'of'), ('of', 'the'), ('the', 'the')]
            # Re-reading: "adjacent identical function-word pairs ('the the', 'of of', 'a the', 'the a', 'the of the the')"
            # Since "identical" usually means same word, but the list includes 'a the', we follow the list provided.
            
            # Let's check specific strings to simplify the "adjacent" logic
            if len(tokens) >= 4:
                # Check for "the of the the" (4 tokens)
                for j in range(len(tokens) - 3):
                    chunk = [t.lower() for t in tokens[j:j+4]]
                    if chunk == ['the', 'of', 'the', 'the']:
                        pair_count += 1
            
            # Check for 2-token sequences
            for j in range(len(tokens) - 1):
                pair = (tokens[j].lower(), tokens[j+1].lower())
                if pair in [('the', 'the'), ('of', 'of'), ('a', 'the'), ('the', 'a')]:
                    pair_count += 1
            
            if (pair_count / len(tokens)) > 0.005:
                continue

        # 4. Dedupe
        norm_text = " ".join(tokens)
        h1 = hash(norm_text)
        h2 = hash(norm_text[:300])
        if h1 in seen_full_hashes or h2 in seen_prefix_hashes:
            continue
        
        seen_full_hashes.add(h1)
        seen_prefix_hashes.add(h2)
        kept_indices.append(i)

    rng.shuffle(kept_indices)
    return kept_indices
