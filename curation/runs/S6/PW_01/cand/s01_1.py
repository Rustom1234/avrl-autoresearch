import os
import random
import zlib
import re
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.encoding_for_model("gpt2")
    
    # 1. Basic Filtering and Deduplication
    processed_keys = set()
    valid_candidates = []
    
    for i, doc in enumerate(docs):
        if len(doc) < 200:
            continue
            
        # Clean/Filter junk
        doc_lower = doc.lower()
        if any(x in doc_lower for x in ['coupon', 'discount code', 'free shipping']):
            continue
            
        # Check for meaningful text (basic alpha/space ratio)
        alpha_count = sum(1 for c in doc if (('a' <= c <= 'z') or ('A' <= c <= 'Z')) or c == ' ')
        if alpha_count / len(doc) < 0.05:
            continue
            
        # Compression check
        data = doc.encode('utf-8', 'ignore')
        if not data: continue
        compressed = zlib.compress(data)
        if len(compressed) / len(data) > 0.3:
            continue

        # Deduplication key
        norm = doc_lower.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
        first_500 = norm[:500]
        try:
            first_token = enc.encode(doc, shortcuts=False)[0]
        except:
            first_token = 0
        
        key = hash((first_500, first_token))
        if key in processed_keys:
            continue
        processed_keys.add(key)
        
        valid_candidates.append((i, doc))

    # 2. Target Similarity Scoring
    def get_bigrams(text):
        words = re.findall(r'\w+', text.lower())
        return set(zip(words, words[1:])) if len(words) > 1 else set()

    target_bigrams = set()
    for ts in target_sample:
        target_bigrams.update(get_bigrams(ts))

    def get_score(text):
        b_list = get_bigrams(text)
        if not b_list: return 0.0
        hits = sum(1 for b in b_list if b in target_bigrams)
        return hits / len(b_list)

    scored_candidates = []
    for idx, doc in valid_candidates:
        score = get_score(doc)
        tokens = len(enc.encode(doc))
        scored_candidates.append({
            'idx': idx,
            'score': score,
            'tokens': tokens
        })

    # Sort by score descending
    scored_candidates.sort(key=lambda x: x['score'], reverse=True)

    # 3. Selection
    selected_indices = []
    current_tokens = 0
    
    # Take the best documents until we hit the token limit T
    # We use a greedy approach for the highest quality text
    for c in scored_candidates:
        if current_tokens + c['tokens'] <= T:
            selected_indices.append(c['idx'])
            current_tokens += c['tokens']
        else:
            # If one doc is too big, try to find a smaller one that fits
            continue

    # If we have space left, fill it with lower-scoring docs
    if current_tokens < T * 0.9:
        remaining = [c for c in scored_candidates if c['idx'] not in selected_indices]
        for c in remaining:
            if current_tokens + c['tokens'] <= T:
                selected_indices.append(c['idx'])
                current_tokens += c['tokens']
            if current_tokens >= T:
                break

    # Final shuffle to avoid ordering bias
    rng.shuffle(selected_indices)
    return selected_indices
