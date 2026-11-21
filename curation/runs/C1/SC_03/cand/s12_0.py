import os
import random
import re
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    T_LIMIT = 9830400 * 1.03

    def get_shuffled_rate(text):
        toks = text.split()
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

    target_toks = []
    target_wt_sum = 0.0
    for s in target_sample:
        target_toks.extend(s.split())
    
    target_wt_vals = []
    for s in target_sample:
        h = s[:4000]
        w_cnt = max(1, len(h.split()))
        wt = (h.count(' , ') + h.count(' . ') + h.count(' @-@ ') + h.count(' ( ')) / w_cnt
        target_wt_vals.append(wt)
    
    target_wt_vals.sort()
    mw = target_wt_vals[len(target_wt_vals) // 2]
    p10 = target_wt_vals[max(0, int(len(target_wt_vals) * 0.1) - 1)] if target_wt_vals else 0.0

    df_counter = Counter()
    header_candidates_oov = []
    
    for doc in docs:
        lines = doc.split('\n')
        if lines and re.match(r'^\s*= [^=\n]+ =\s*$', lines[0]):
            body_toks = doc[:8000].split()[:1000]
            for t in body_toks:
                df_counter[t] += 1
            # We'll calculate OOV later
    
    vocab = set(target_toks)
    for token, count in df_counter.items():
        if count >= 5:
            vocab.add(token)

    seen_hashes = set()
    tiers = {'A': [], 'B': [], 'C': [], 'D': []}
    
    demoted_oov = 0
    rescued_count = 0
    rescued_tokens = 0.0

    for i, doc in enumerate(docs):
        h_hash = hash(doc)
        if h_hash in seen_hashes: continue
        seen_hashes.add(h_hash)
        
        est_tokens = len(doc) / 4.0
        head = doc[:4000]
        
        spam_hits = sum(1 for kw in ['discount code', 'best price guaranteed', 'free shipping on all', 
                                      'Click here to order', 'online store |', 'Limited offer on'] if kw in head)
        is_spam = spam_hits >= 2
        is_junk = len(doc) < 200 or (head and (head.count(' ') / len(head) < 0.08 or sum(1 for c in head if ord(c) > 127) / len(head) > 0.20))
        
        if is_spam or is_junk:
            tiers['D'].append((i, est_tokens))
            continue

        toks = doc.split()
        rate_head = get_shuffled_rate(doc[:doc.find(' ') + 500]) if ' ' in doc else 0.0
        
        is_shuffled = False
        if rate_head < 0.3 * 0.3: # Arbitrary low thresh or logic
            is_shuffled = True
        else:
            windows = [toks[s:s+300] for s in range(0, len(toks), 300) if len(toks[s:s+300]) >= 150]
            if windows:
                bad_windows = sum(1 for w in windows if get_shuffled_rate(" ".join(w)) < 0.3 * 0.3)
                if bad_windows / len(windows) > 0.25:
                    is_shuffled = True
        
        if is_shuffled:
            tiers['C'].append((i, est_tokens))
            continue

        # Determine if it qualifies for A check
        lines = doc.split('\n')
        has_header = lines and re.match(r'^\s*= [^=\n]+ =\s*$', lines[0])
        
        # Rule 1: OOV Calculation
        # We compute OOV for anyone who has a header or is a rescued candidate
        oov_val = 0.0
        if has_header:
            body_toks = doc[:8000].split()[:1000]
            if body_toks:
                oov_count = sum(1 for t in body_toks if t not in vocab)
                oov_val = oov_count / len(body_toks)
            header_candidates_oov.append(oov_val)
        else:
            # Potential rescue
            w_cnt = max(1, len(head.split()))
            wt = (head.count(' , ') + head.count(' . ') + head.count(' @-@ ') + head.count(' ( ')) / w_cnt
            if wt >= max(0.5 * mw, p10):
                body_toks = doc[:8000].split()[:1000]
                if body_toks:
                    oov_count = sum(1 for t in body_toks if t not in vocab)
                    oov_val = oov_count / len(body_toks)
                    header_candidates_oov.append(oov_val)
                    # If it passes OOV check later, it's a rescue
                else:
                    # No body tokens, skip OOV check or treat as 0
                    oov_val = 0.0
                    header_candidates_oov.append(oov_val)

        # Placeholder to process A later because we need med/mad of all candidates first
        # Instead of multi-pass, let's find all "Potential A" first
        # But the instructions say "single-pass and cheap". 
        # To stay single-pass, we must identify all A-candidates in one go.
        # Let's refine: we need med/mad of OOV. Let's just collect all OOV values first.
        # Wait, the instructions say "Do not rank tier B". 
        # Let's optimize: we can't get med/mad without seeing all candidates.
        # Given the 180s limit, a quick pre-pass of docs is okay.
        pass # Logic moved to correct structure below
    
    # Re-evaluating: Since we need med/mad of OOV, we must know all candidates first.
    # Let's do a fast scan to identify potential A's.
    potential_A = []
    for i, doc in enumerate(docs):
        h_hash = hash(doc)
        if h_hash in seen_hashes: continue
        seen_hashes.add(h_hash)
        
        # Basic filters
        head = doc[:4000]
        spam_hits = sum(1 for kw in ['discount code', 'best price guaranteed', 'free shipping on all', 
                                      'Click here to order', 'online store |', 'Limited offer on'] if kw in head)
        if spam_hits >= 2: continue
        if len(doc) < 200 or (head and (head.count(' ') / len(head) < 0.08 or sum(1 for c in head if ord(c) > 127) / len(head) > 0.20)):
            continue
            
        toks = doc.split()
        rate_head = get_shuffled_rate(doc[:doc.find(' ') + 500]) if ' ' in doc else 0.0
        is_shuffled = False
        if rate_head < 0.3 * 0.3: is_shuffled = True
        else:
            windows = [toks[s:s+300] for s in range(0, len(toks), 300) if len(toks[s:s+300]) >= 150]
            if windows:
                bad_windows = sum(1 for w in windows if get_shuffled_rate(" ".join(w)) < 0.3 * 0.3)
                if bad_windows / len(windows) > 0.25: is_shuffled = True
        
        if is_shuffled:
            tiers['C'].append((i, len(doc)/4.0))
            continue
            
        lines = doc.split('\n')
        has_header = lines and re.match(r'^\s*= [^=\n]+ =\s*$', lines[0])
        
        if has_header:
            body_toks = doc[:8000].split()[:1000]
            oov = sum(1 for t in body_toks if t not in vocab) / len(body_toks) if body_toks else 0.0
            potential_A.append((i, len(doc)/4.0, oov, True))
        else:
            w_cnt = max(1, len(head.split()))
            wt = (head.count(' , ') + head.count(' . ') + head.count(' @-@ ') + head.count(' ( ')) / w_cnt
            if wt >= max(0.5 * mw, p10):
                body_toks = doc[:8000].split()[:1000]
                oov = sum(1 for t in body_toks if t not in vocab) / len(body_toks) if body_toks else 0.0
                potential_A.append((i, len(doc)/4.0, oov, False))
            else:
                tiers['B'].append((i, len(doc)/4.0))

    if potential_A:
        oovs = sorted([x[2] for x in potential_A])
        med_oov = oovs[len(oovs) // 2]
        mad_oov = 0.0
        if len(oovs) > 1:
            mad_oov = sorted([abs(x - med_oov) for x in oovs])[len(oovs) // 2]
        threshold = med_oov + 6 * mad_oov
        
        seen_a_keys = {}
        for idx, tokens, oov, is_header in potential_A:
            if oov > threshold and oov > 0.15:
                tiers['C'].append((idx, tokens))
                demoted_oov += 1
            else:
                doc = docs[idx]
                t_list = doc.split()
                key = (doc.split('\n')[0], ' '.join(t_list[20:60]))
                if key in seen_a_keys:
                    tiers['D'].append((idx, tokens))
                else:
                    seen_a_keys[key] = True
                    tiers['A'].append((idx, tokens))
                    if not is_header:
                        rescued_count += 1
                        rescued_tokens += tokens

    selected_indices = []
    current_tokens = 0
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
        rng.shuffle(tiers['A'])
        for idx, tokens in tiers['A']:
            selected_indices.append(idx)
            current_tokens += tokens
            counts_taken['A'] += 1
        
        rng.shuffle(tiers['A'])
        for idx, tokens in tiers['A']:
            if current_tokens >= T_LIMIT: break
            selected_indices.append(idx)
            current_tokens += tokens
            counts_taken['A'] += 1
            second_copy_count += 1
            
        for t in ['B', 'C', 'D']:
            tier_data = tiers[t]
            rng.shuffle(tier_data)
            for idx, tokens in tier_data:
                if current_tokens >= T_LIMIT: break
                selected_indices.append(idx)
                current_tokens += tokens
                counts_taken[t] += 1
                
    rng.shuffle(selected_indices)
    
    print(f"med: {med_oov:.4f}, mad: {mad_oov:.4f}, threshold: {threshold:.4f}")
    print(f"demoted by OOV: {demoted_oov}")
    print(f"rescued: {rescued_count} ({rescued_tokens:.1f} tokens)")
    print(f"tokA / T: {tokA_total / 9830400:.4f}")
    print(f"second copies: {second_copy_count}")
    print(f"B taken: {counts_taken['B']}")
    print(f"C taken: {counts_taken['C']}")
    print(f"D taken: {counts_taken['D']}")
    
    return selected_indices
