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

    selected_data = []
    current_tokens = 0
    order_tiers = ['A', 'B', 'C', 'D']
    for t in order_tiers:
        tier_data = tiers[t]
        rng.shuffle(tier_data)
        for idx, tokens in tier_data:
            selected_data.append({'idx': idx, 'tokens': tokens, 'tier': t, 'is_a': (t == 'A')})
            current_tokens += tokens
            if current_tokens >= T_LIMIT: break
        if current_tokens >= T_LIMIT: break

    selected_data.sort(key=lambda x: rng.random() + (0.6 if x['is_a'] else 0.0))
    indices = [x['idx'] for x in selected_data]
    
    counts_taken = {t: 0 for t in order_tiers}
    for x in selected_data:
        counts_taken[x['tier']] += 1
        
    for t in order_tiers:
        total_t = sum(x[1] for x in tiers[t])
        print(f"Tier {t}: count={len(tiers[t])}, tokens={total_t:.1f}, taken={counts_taken[t]}")
        
    last_30_pct_size = len(indices) // 3
    start_idx = max(0, len(indices) - last_30_pct_size)
    a_in_last = sum(1 for x in selected_data[start_idx:] if x['is_a'])
    frac_a = a_in_last / (len(indices) - start_idx) if (len(indices) - start_idx) > 0 else 0
    print(f"Fraction of Tier A in last 30%: {frac_a:.4f}")

    return indices
