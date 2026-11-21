import os
import random
import zlib
import re
import collections
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Deduplication
    unique_docs = {}
    processed_keys = set()
    doc_indices_map = []
    
    def get_dedup_key(text):
        norm = text.lower().replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
        # first 500 chars + first token hash
        first_500 = norm[:500]
        enc = tiktoken.encoding_for_model("gpt2")
        first_token = enc.encode(text,添加=True)[0] if text else 0
        return hash((first_500, first_token))

    # Intermediate storage for validation
    valid_candidates = []
    
    for i, doc in enumerate(docs):
        # Step 2: Drop junk (Basic filters)
        if len(doc) < 200: continue
        
        # Non-ASCII / Web Junk / Non-Alpha check
        alpha_count = sum(1 for c in doc if ('a' <= c <= 'z') or ('A' <= c <= 'Z') or c == ' ')
        if alpha_count / len(doc) < 0.05: continue
        
        if any(x in doc.lower() for x in ['coupon', 'discount code', 'free shipping']):
            continue

        # Vocab complexity
        words = doc.split()
        unique_words = len(set(w for w in words if w))
        if len(words) > 0 and (unique_words / len(words)) < 0.5:
            continue
            
        # Zlib compression ratio
        compressed = zlib.compress(doc.encode('utf-8', 'ignore'))
        if len(compressed) / len(doc.encode('utf-8', 'ignore')) > 0.2:
            continue

        # Deduplication
        key = get_dedup_key(doc)
        if key not in processed_keys:
            processed_keys.add(key)
            valid_candidates.append((i, doc))

    # Bigram analysis for shuffle detection
    def get_bigrams(text):
        words = re.findall(r'\w+', text.lower())
        return [(words[i], words[i+1]) for i in range(len(words)-1)]

    target_bigrams = set()
    target_doc_scores = []
    for ts in target_sample:
        t_bigrams = get_bigrams(ts)
        target_bigrams.update(t_biggrams)
        # Dummy score
        target_doc_scores.append(0)

    # Calculate median hit ratio for target
    def get_hit_ratio(text):
        b_list = get_bigrams(text[:1000])
        if not b_list: return 0
        hits = sum(1 for b in b_list if b in target_bigrams)
        return hits / len(b_list)

    target_hit_ratios = [get_hit_ratio(ts) for ts in target_sample]
    target_hit_ratios.sort()
    median_hit_ratio = target_hit_ratios[len(target_hit_ratios)//2] if target_hit_ratios else 0

    # Filtering valid candidates + Scoring
    scored_candidates = []
    enc = tiktoken.encoding_for_model("gpt2")
    
    for idx, doc in valid_candidates:
        # Word-shuffled artifacts
        b_list = get_bigrams(doc[:1000])
        if b_list:
            hits = sum(1 for b in b_list if b in target_bigrams)
            hit_ratio = hits / len(b_list)
            if hit_ratio < (median_hit_ratio * 0.5):
                continue
        
        # Score
        full_b_list = get_bigrams(doc)
        hits = sum(1 for b in full_b_list if b in target_bigrams)
        score = hits / len(full_b_list) if full_b_list else 0
        
        # Token count
        tokens = len(enc.encode(doc))
        scored_candidates.append({
            'idx': idx,
            'score': score,
            'tokens': tokens,
            'text': doc
        })

    # Sort and Filter
    # Add noise to score distribution
    if len(scored_candidates) > 1:
        scores = [c['score'] for c in scored_candidates]
        avg_score = sum(scores) / len(scores)
        std_dev = (sum((x - avg_score)**2 for x in scores) / len(scores))**0.5
        for c in scored_candidates:
            c['score'] += rng.uniform(-0.2 * std_dev, 0.2 * std_dev)

    scored_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    selected_indices = []
    current_tokens = 0
    
    # Threshold logic
    if scored_candidates:
        # Calculate 5th percentile of target-like scores
        scored_candidates.sort(key=lambda x: x['score'])
        p5_score = scored_candidates[max(0, int(len(scored_candidates)*0.05))]['score']
        # We want the scores from the high end first
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        threshold = 0.4 * p5_score
        
        # First pass: Top docs (descending score)
        for c in scored_candidates:
            if current_tokens + c['tokens'] <= T * 1.05:
                selected_indices.append(c['idx'])
                current_tokens += c['tokens']
            else:
                break
                
        # Top up if needed
        if current_tokens < T * 0.7:
            for c in scored_candidates:
                if c['idx'] not in selected_indices:
                    selected_indices.append(c['idx'])
                    current_tokens += c['tokens']
                    if current_tokens >= T: break
                if current_tokens >= T: break

    # Final check for 0.5 T
    if current_tokens < T * 0.5:
        # Add lowest score docs among kept ones up to 0.5 T
        # (Actually, the instructions say "lowest-score docs among kept ones" 
        # but implies we should take from pool to reach 0.5T if we are under it)
        remaining = [c for c in scored_candidates if c['idx'] not in selected_indices]
        remaining.sort(key=lambda x: x['score'])
        for c in remaining:
            selected_indices.append(c['idx'])
            current_tokens += c['tokens']
            if current_tokens >= T * 0.5: break

    # Final shuffle of indices
    final_indices = selected_indices[:]
    rng.shuffle(final_indices)
    return final_indices
