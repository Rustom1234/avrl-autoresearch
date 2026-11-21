import os
import random
import re

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    T_LIMIT = 9830400 * 1.03
    
    def get_shuffled_rate(text):
        toks = text.split()[:500]
        if not toks: return 0.0
        pairs = {('of','the'), ('in','the'), ('to','the'), ('on','the'), 
                  ('and','the'), ('at','the'), ('by','the'), ('for','the'), 
                  ('with','the'), ('from','the'), (',','and'), ('.','The'), 
                  ('as','a'), ('in','a'), ('of','a')}
        count = 0
        for i in range(len(toks) - 1):
            if (toks[i], toks[i+1]) in pairs:
                count += 1
        return count / len(toks)

    target_rates = [get_shuffled_rate(s) for s in target_sample]
    m = 0.0
    if target_rates:
        sorted_rates = sorted(target_rates)
        m = sorted_rates[len(sorted_rates) // 2]

    seen_hashes = set()
    tiers = {'A': [], 'B': [], 'C': [], 'D': []}
    
    for i, doc in enumerate(docs):
        h = hash(doc)
        if h in seen_hashes: continue
        seen_hashes.add(h)
        
        est_tokens = len(doc) / 4.0
        head = doc[:4000]
        spam_hits = sum(1 for kw in ['discount code', 'best price guaranteed', 'free shipping on all', 
                                      'Click here to order', 'online store |', 'Limited offer on'] if kw in head)
        is_spam = spam_hits >= 2
        is_junk = len(doc) < 200 or (head and (head.count(' ') / len(head) < 0.08 or sum(1 for c in head if ord(c) > 127) / len(head) > 0.20))
        rate = get_shuffled_rate(doc)
        is_shuffled = rate < 0.3 * m
        
        if is_spam or is_junk:
            tiers['D'].append((i, est_tokens))
        elif is_shuffled:
            tiers['C'].append((i, est_tokens))
        elif re.match(r'^\s*= [^=\n]+ =\s*$', doc.split('\n')[0]):
            tiers['A'].append((i, est_tokens))
        else:
            tiers['B'].append((i, est_tokens))
            
    selected_indices = []
    current_tokens = 0
    
    # Track counts for printing
    counts_taken = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    second_copy_count = 0

    tokA_total = sum(x[1] for x in tiers['A'])
    
    if tokA_total >= T_LIMIT:
        rng.shuffle(tiers['A'])
        for idx, tokens in tiers['A']:
            selected_indices.append(idx)
            current_tokens += tokens
            counts_taken['A'] += 1
            if current_tokens >= T_LIMIT: break
    else:
        # Take all of A
        rng.shuffle(tiers['A'])
        for idx, tokens in tiers['A']:
            selected_indices.append(idx)
            current_tokens += tokens
            counts_taken['A'] += 1
        
        # Second copy of A
        rng.shuffle(tiers['A'])
        for idx, tokens in tiers['A']:
            if current_tokens >= T_LIMIT: break
            selected_indices.append(idx)
            current_tokens += tokens
            counts_taken['A'] += 1
            second_copy_count += 1
            
        # Fill B, C, D
        for t in ['B', 'C', 'D']:
            tier_data = tiers[t]
            rng.shuffle(tier_data)
            for idx, tokens in tier_data:
                if current_tokens >= T_LIMIT: break
                selected_indices.append(idx)
                current_tokens += tokens
                counts_taken[t] += 1
                
    rng.shuffle(selected_indices)
    
    total_t = {t: sum(x[1] for x in tiers[t]) for t in ['A', 'B', 'C', 'D']}
    
    for t in ['A', 'B', 'C', 'D']:
        print(f"Tier {t}: count={len(tiers[t])}, tokens={total_t[t]:.1f}")
    
    print(f"tokA / T: {tokA_total / 9830400:.4f}")
    print(f"second copies: {second_copy_count}")
    print(f"B taken: {counts_taken['B']}")
    print(f"C taken: {counts_taken['C']}")
    print(f"D taken: {counts_taken['D']}")
    
    return selected_indices
