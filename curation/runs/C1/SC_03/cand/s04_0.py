import os
import random
import re

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    T_LIMIT = 9830400 * 1.03
    
    # 1. Estimate tokens and identify basic properties
    # 2. Handle exact duplicates using a set of hashes
    seen_hashes = set()
    processed_docs = []
    
    # Pre-calculate median shuffled rate for target_sample
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

    # Categorize docs
    tiers = {'A': [], 'B': [], 'C': [], 'D': []}
    tier_stats = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    tier_token_stats = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

    for i, doc in enumerate(docs):
        h = hash(doc)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        
        est_tokens = len(doc) / 4.0
        head = doc[:4000]
        
        # Spam check
        spam_keywords = ['discount code', 'best price guaranteed', 'free shipping on all', 
                         'Click here to order', 'online store |', 'Limited offer on']
        hits = sum(1 for kw in spam_keywords if kw in head)
        is_spam = hits >= 2
        
        # Junk check
        is_junk = False
        if len(doc) < 200:
            is_junk = True
        elif head:
            spaces = head.count(' ')
            if spaces / len(head) < 0.08:
                is_junk = True
            else:
                high_chars = sum(1 for char in head if ord(char) > 127)
                if high_chars / len(head) > 0.20:
                    is_junk = True
        
        # Shuffled check
        rate = get_shuffled_rate(doc)
        is_shuffled = rate < 0.3 * m
        
        # Tier Logic
        if is_spam or is_junk:
            tiers['D'].append((i, est_tokens))
        elif is_shuffled:
            tiers['C'].append((i, est_tokens))
        elif re.match(r'^\s*= [^=\n]+ =\s*$', doc.split('\n')[0]):
            tiers['A'].append((i, est_tokens))
        else:
            tiers['B'].append((i, est_tokens))
            
        # Update stats for tier assignment
        # Note: A doc can be D even if it matches A's regex. The rules imply order.
        # However, the prompt says "A: ... and not spam, junk or shuffled."
        # This implies mutually exclusive logic.
        
    # Re-evaluating logic to strictly follow prompt tiers:
    tiers = {'A': [], 'B': [], 'C': [], 'D': []}
    for i, doc in enumerate(docs):
        h = hash(doc)
        if h in seen_hashes: continue # This is slightly wrong because I already iterated, 
        # let's restart the loop clearly.
    
    # Restarting logic for clarity and correctness
    seen_hashes = set()
    tiers = {'A': [], 'B': [], 'C': [], 'D': []}
    tier_tokens = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    
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
            
    # Sort tiers and select
    order_tiers = ['A', 'B', 'C', 'D']
    selected_indices = []
    current_tokens = 0
    
    for t in order_tiers:
        # Shuffle within tier
        tier_data = tiers[t]
        rng.shuffle(tier_data)
        for idx, tokens in tier_data:
            selected_indices.append(idx)
            current_tokens += tokens
            if current_tokens >= T_LIMIT:
                break
        if current_tokens >= T_LIMIT:
            break
            
    # Final shuffle
    rng.shuffle(selected_indices)
    
    # Print stats
    for t in order_tiers:
        count = len(tiers[t])
        total_t = sum(x[1] for x in tiers[t])
        print(f"Tier {t}: count={count}, tokens={total_t:.1f}")
    
    return selected_indices
