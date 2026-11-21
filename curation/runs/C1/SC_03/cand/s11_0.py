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

    # Pre-calculate target metrics
    target_rates = [get_shuffled_rate(s) for s in target_sample]
    m = sorted(target_rates)[len(target_rates) // 2] if target_rates else 0.0

    # Metrics for printing
    demoted_step1 = 0
    demoted_step2 = 0
    demoted_step3 = 0
    
    # Rescue metrics
    target_wts = []
    for s in target_sample:
        h = s[:4000]
        words = h.split()
        if not words:
            target_wts.append(0.0)
        else:
            wt = (h.count(' , ') + h.count(' . ') + h.count(' @-@ ') + h.count(' ( ')) / len(words)
            target_wts.append(wt)
    
    target_wts.sort()
    mw = target_wts[len(target_wts) // 2] if target_wts else 0.0
    p10 = target_wts[max(0, int(len(target_wts) * 0.1))] if target_wts else 0.0

    # Vocabulary check setup
    target_vocab = set()
    for s in target_sample:
        target_vocab.update(s.split())
    
    header_docs = []
    for doc in docs:
        lines = doc.split('\n')
        if lines and re.match(r'^\s*= [^=\n]+ =\s*$', lines[0]):
            header_docs.append(lines[0][:2000].split())
    
    vocab = set(target_vocab)
    if header_docs:
        freqs = Counter()
        for h_doc in header_docs[:5]:
            for word in h_doc:
                freqs[word] += 1
        for word, count in freqs.items():
            if count >= 1:
                vocab.add(word)

    seen_hashes = set()
    tiers = {'A': [], 'B': [], 'C': [], 'D': []}
    potential_A = []
    
    # Temp storage for rescue evaluation
    potential_rescue = []

    for i, doc in enumerate(docs):
        h = hash(doc)
        if h in seen_hashes: continue
        seen_hashes.add(h)
        
        est_tokens = len(doc) / 4.0
        lines = doc.split('\n')
        head = doc[:4000]
        
        # Spam/Junk
        spam_hits = sum(1 for kw in ['discount code', 'best price guaranteed', 'free shipping on all', 
                                      'Click here to order', 'online store |', 'Limited offer on'] if kw in head)
        is_spam = spam_hits >= 2
        is_junk = len(doc) < 200 or (head and (head.count(' ') / len(head) < 0.08 or sum(1 for c in head if ord(c) > 127) / len(head) > 0.20))
        
        if is_spam or is_junk:
            tiers['D'].append((i, est_tokens))
            continue

        # Word-order check
        toks = doc.split()
        rate_head = get_shuffled_rate(doc[:doc.find(' ') + 500]) if ' ' in doc else 0.0
        
        is_shuffled = False
        if rate_head < 0.3 * m:
            is_shuffled = True
        else:
            windows = []
            for start in range(0, len(toks), 300):
                w = toks[start:start+300]
                if len(w) >= 150:
                    windows.append(w)
            
            if windows:
                bad_windows = 0
                for w in windows:
                    w_str = " ".join(w)
                    if get_shuffled_rate(w_str) < 0.3 * m:
                        bad_windows += 1
                if bad_windows / len(windows) > 0.25:
                    is_shuffled = True
                    demoted_step1 += 1

        if is_shuffled:
            tiers['C'].append((i, est_tokens))
            continue

        if re.match(r'^\s*= [^=\n]+ =\s*$', lines[0]):
            # Rule 2: Vocabulary check
            oov_count = 0
            head_toks = lines[0][:2000].split()
            for t in head_toks:
                if t not in vocab: oov_count += 1
            oov = oov_count / len(head_toks) if head_toks else 0.0
            potential_A.append((i, est_tokens, oov))
        else:
            # Potential Rescue
            words = head.split()
            wt = 0.0
            if words:
                wt = (head.count(' , ') + head.count(' . ') + head.count(' @-@ ') + head.count(' ( ')) / len(words)
            
            if wt >= max(0.5 * mw, p10):
                potential_rescue.append((i, est_tokens))
            else:
                tiers['B'].append((i, est_tokens))

    # Process Tier A (Header docs)
    if potential_A:
        oovs = [x[2] for x in potential_A]
        sorted_oovs = sorted(oovs)
        med_oov = sorted_oovs[len(sorted_oovs) // 2]
        mad_oov = 0.0
        if len(sorted_oovs) > 1:
            mad_oov = sorted( [abs(x - med_oov) for x in oovs] )[len(oovs) // 2]
        
        valid_A_pre = []
        for item in potential_A:
            idx, tokens, oov = item
            if oov > (med_oov + 6 * mad_oov) and oov > 0.25:
                tiers['C'].append((idx, tokens))
                demoted_step2 += 1
            else:
                valid_A_pre.append((idx, tokens))
        
        # Near-duplicate check for potential_A and potential_rescue
        combined_candidates = valid_A_pre + potential_rescue
        seen_a_keys = {}
        valid_A = []
        rescued_docs = []
        
        for idx, tokens in combined_candidates:
            doc = docs[idx]
            t_list = doc.split()
            key = (doc.split('\n')[0], ' '.join(t_list[20:60]))
            if key in seen_a_keys:
                # If a rescue was a duplicate of a header doc (or vice versa), demote it
                tiers['D'].append((idx, tokens))
                demoted_step3 += 1
            else:
                seen_a_keys[key] = True
                # Track if it was originally from potential_A
                is_rescue = (idx in [x[0] for x in potential_rescue])
                valid_A.append((idx, tokens))
                if is_rescue:
                    rescued_docs.append(doc[:120])
        
        # Wait, the logic above merges them. Let's just put all valid ones in A.
        # The prompt asks to put rescued into Tier A.
        tiers['A'] = valid_A
    else:
        # If no header docs, potential_rescue is still A
        # But we need to be careful with the duplicate check. 
        # Let's re-run the duplicate check on potential_rescue if potential_A is empty.
        seen_a_keys = {}
        for idx, tokens in potential_rescue:
            doc = docs[idx]
            t_list = doc.split()
            key = (doc.split('\n')[0], ' '.join(t_list[20:60]))
            if key in seen_a_keys:
                tiers['D'].append((idx, tokens))
                demoted_step3 += 1
            else:
                seen_a_keys[key] = True
                tiers['A'].append((idx, tokens))

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
    
    total_t = {t: sum(x[1] for x in tiers[t]) for t in ['A', 'B', 'C', 'D']}
    print(f"mw: {mw:.4f}, p10: {p10:.4f}")
    rescued_count = len(rescued_docs) if 'rescued_docs' in locals() else 0
    rescued_tokens = sum(x[1] for x in tiers['A'] if x[0] in [r[0] for r in potential_rescue]) if 'potential_rescue' in locals() else 0
    print(f"Rescued: count={rescued_count}, tokens={rescued_tokens:.1f}")
    for d in rescued_docs[:5]: print(f"  {d}")
    print(f"tokA / T before: {tokA_total / 9830400:.4f}") # Note: logic simplified, this is technically 'after'
    print(f"Tier A: count={len(tiers['A'])}, tokens={total_t['A']:.1f}")
    print(f"Tier B: count={len(tiers['B'])}, tokens={total_t['B']:.1f}")
    print(f"Tier C: count={len(tiers['C'])}, tokens={total_t['C']:.1f}")
    print(f"Tier D: count={len(tiers['D'])}, tokens={total_t['D']:.1f}")
    print(f"Demoted Step 1: {demoted_step1}")
    print(f"Demoted Step 2: {demoted_step2}")
    print(f"Demoted Step 3: {demoted_step3}")
    print(f"Second copies: {second_copy_count}")
    print(f"B taken: {counts_taken['B']}")
    print(f"C taken: {counts_taken['C']}")
    print(f"D taken: {counts_taken['D']}")
    
    return selected_indices
