import os
import random
import re
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def get_tokens(text):
        return text.split()

    def get_bigrams(text):
        tokens = get_tokens(text)
        return [(tokens[i].lower(), tokens[i+1].lower()) for i in range(len(tokens)-1)]

    # 1. Format Filter
    candidate_indices = []
    for i, doc in enumerate(docs):
        tokens = get_tokens(doc)
        if not tokens: continue
        
        punctuation_count = sum(1 for t in tokens if t == ',' or t == '.')
        punct_frac = punctuation_count / len(tokens)
        
        lines = doc.splitlines()
        code_spam_count = 0
        for line in lines:
            if any(x in line for x in ['{', '};', 'import ', 'http', 'cheap', 'coupon']):
                code_spam_count += 1
        
        if punct_frac >= 0.04 and (code_spam_count / len(lines) if lines else 0) < 0.02:
            candidate_indices.append(i)

    # 2. Word-order Filter
    target_bigrams = set()
    for ts in target_sample:
        target_bigrams.update(get_bigrams(ts))
    
    real_rates = []
    shuffled_rates = []
    for i in range(len(target_sample)):
        doc = target_sample[i]
        others = [target_sample[j] for j in range(len(target_sample)) if i != j]
        others_bigrams = set()
        for o in others: others_bigrams.update(get_bigrams(o))
        
        current_bg = get_bigrams(doc)
        real_rates.append(sum(1 for b in current_bg if b in others_bigrams) / len(current_bg) if current_bg else 0)
        
        tokens = doc.split()
        rng.shuffle(tokens)
        shuffled_doc = " ".join(tokens)
        shuffled_bg = get_bigrams(shuffled_doc)
        shuffled_rates.append(sum(1 for b in shuffled_bg if b in others_bigrams) / len(shuffled_bg) if shuffled_bg else 0)

    real_rates.sort()
    shuffled_rates.sort()
    thr0 = (real_rates[int(len(real_rates) * 0.05)] + shuffled_rates[int(len(shuffled_rates) * 0.95)]) / 2

    s0_indices = []
    for idx in candidate_indices:
        doc = docs[idx]
        tokens = get_tokens(doc)
        sub_tokens = tokens[:2000]
        if len(sub_tokens) < 2:
            s0_indices.append(idx)
            continue
        bg_list = [(sub_tokens[i].lower(), sub_tokens[i+1].lower()) for i in range(len(sub_tokens)-1)]
        hit_rate = sum(1 for b in bg_list if b in target_bigrams) / len(bg_list)
        if hit_rate >= thr0:
            s0_indices.append(idx)

    # 3. Self-training recall
    H_indices = []
    s0_minus_h_indices = []
    for i, idx in enumerate(s0_indices):
        if i < min(300, len(s0_indices)):
            H_indices.append(idx)
        else:
            s0_minus_h_indices.append(idx)
    
    B = set()
    for idx in s0_minus_h_indices:
        B.update(get_bigrams(docs[idx][:2000]))
    for ts in target_sample:
        B.update(get_bigrams(ts))
    
    real_rates_2 = []
    shuffled_rates_2 = []
    for idx in H_indices:
        doc = docs[idx]
        tokens = get_tokens(doc)
        # Real
        bg = get_bigrams(doc[:2000])
        real_rates_2.append(sum(1 for b in bg if b in B) / len(bg) if bg else 0)
        # Shuffled
        tokens_shuffled = tokens[:2000]
        rng.shuffle(tokens_shuffled)
        shuffled_bg = get_bigrams(" ".join(tokens_shuffled))
        shuffled_rates_2.append(sum(1 for b in shuffled_bg if b in B) / len(shuffled_bg) if shuffled_bg else 0)
    
    real_rates_2.sort()
    shuffled_rates_2.sort()
    thr2 = (real_rates_2[int(len(real_rates_2) * 0.05)] + shuffled_rates_2[int(len(shuffled_rates_2) * 0.95)]) / 2
    
    s1_indices = list(s0_minus_h_indices)
    
    # Candidates for S1 addition: H or (C - S0)
    s0_set = set(s0_indices)
    for idx in (H_indices + candidate_indices):
        if idx not in s0_set:
            # Check if in H
            if idx in H_indices:
                pass # already handled by H_indices logic? No, logic says keep S0-H as is.
                # Wait, the prompt says: Keep S0 minus H as is. Keep a document of H or of C minus S0 only if hit rate...
            
            # Let's do it properly:
            pass

    # Refined S1 logic:
    # S1 = (S0 - H) + (subset of H and subset of (C - S0))
    s1_indices = list(s0_minus_h_indices)
    
    def check_s1_inclusion(idx):
        doc = docs[idx]
        tokens = get_tokens(doc)
        sub_tokens = tokens[:2000]
        if len(sub_tokens) < 2: return True
        bg_list = [(sub_tokens[i].lower(), sub_tokens[i+1].lower()) for i in range(len(sub_tokens)-1)]
        hit_rate = sum(1 for b in bg_list if b in B) / len(bg_list)
        if hit_rate >= thr2: return True
        return False

    # We need to check H and (C - S0)
    # Note: H is a subset of S0. The logic says: "Keep S0 minus H as is. Keep a document of H or of C minus S0 only if..."
    # This implies H itself is potentially discarded/filtered.
    
    # Let's iterate all candidates and filter
    # Wait, "Keep S0 minus H as is" means the elements in S0 \ H are preserved.
    # The new pool is (S0 \ H) U {x in H | score(x) >= thr2} U {x in (C \ S0) | score(x) >= thr2}
    
    added_h = []
    for idx in H_indices:
        if check_s1_inclusion(idx):
            added_h.append(idx)
            
    added_c_minus_s0 = []
    s0_set = set(s0_indices)
    for idx in candidate_indices:
        if idx not in s0_set:
            if check_s1_inclusion(idx):
                added_c_minus_s0.append(idx)
    
    s1_indices = list(s0_minus_h_indices) + added_h + added_c_minus_s0
    
    if len(s1_indices) < 0.9 * len(s0_indices) or len(s1_indices) > 1.5 * len(s0_indices):
        s1_indices = list(s0_indices)

    # 4. Experiment 7 checks
    def apply_exp7(indices):
        # (A) Windowed
        def filter_window(idx):
            doc = docs[idx]
            tokens = get_tokens(doc)
            if len(tokens) < 10: return True
            windows = []
            for i in range(0, len(tokens), 400):
                win = tokens[i:i+400]
                if len(win) < 200 and i > 0:
                    win = tokens[i-400:i+400] # merge last
                if len(win) < 2: continue
                bg = [(win[j].lower(), win[j+1].lower()) for j in range(len(win)-1)]
                
                # Use B0/thr0 for S0-H, B/thr2 for others
                # Since we don't know which is which easily, and the prompt says "for documents in S0-H... and for all others"
                # We must track if idx was in S0-H
                pass
            return True

        # The logic for S0-H vs others in Exp 7 is slightly complex to track.
        # Let's simplify: The prompt says "use B0 with thr0 for documents in S0 minus H, and B with thr2 for all others"
        # To do this, we need to know if an index is in S0-H.
        
        s0_minus_h_set = set(s0_minus_h_indices)
        
        # (A) Windowed check
        def window_filter(idx):
            tokens = get_tokens(docs[idx])
            if len(tokens) < 10: return True
            count_fail = 0
            total_windows = 0
            for i in range(0, len(tokens), 400):
                win = tokens[i:i+400]
                if len(win) < 200 and i > 0:
                    win = tokens[i-400:i+400]
                if len(win) < 2: continue
                total_windows += 1
                bg = [(win[j].lower(), win[j+1].lower()) for j in range(len(win)-1)]
                
                is_s0_minus_h = idx in s0_minus_h_set
                bg_set = target_bigrams
                threshold = thr0
                if not is_s0_minus_h:
                    bg_set = B
                    threshold = thr2
                
                hit_rate = sum(1 for b in bg if b in bg_set) / len(bg)
                if hit_rate < threshold:
                    count_fail += 1
            
            if total_windows == 0: return True
            return (count_fail / total_windows) <= 0.25

        # (B) Case check
        def case_filter(idx):
            doc = docs[idx]
            letters = [c for c in doc if c.isalpha()]
            if not letters: return True
            u = sum(1 for c in letters if c.isupper()) / len(letters)
            
            # Find target u
            target_u_vals = []
            for ts in target_sample:
                tls = [c for c in ts if c.isalpha()]
                if tls: target_u_vals.append(sum(1 for c in tls if c.isupper()) / len(tls))
            if not target_u_vals: return True
            
            return u >= 0.5 * min(target_u_vals) and u <= 1.5 * max(target_u_vals)

        # Apply independently
        res_a = []
        count_a = 0
        for idx in indices:
            if window_filter(idx):
                res_a.append(idx)
            else:
                count_a += 1
        
        if count_a > 0.2 * len(indices):
            res_a = indices
        
        res_b = []
        count_b = 0
        for idx in res_a:
            if case_filter(idx):
                res_b.append(idx)
            else:
                count_b += 1
        
        if count_b > 0.2 * len(res_a):
            return res_a
        return res_b

    s1_indices = apply_exp7(s1_indices)
    
    # Final Guard
    if len(s1_indices) < 0.9 * len(s0_indices):
        s1_indices = list(s0_indices)

    # 5. Dedupe
    seen_hashes = set()
    seen_titles = set()
    final_indices = []
    
    # Process S0 first to maintain priority
    s0_set_indices = set(s0_indices)
    
    # We need to process them in a way that respects S0 priority
    # The prompt says "processing S0 documents before the additions"
    # and "S0 documents" refers to the S1 components that originated from S0.
    # Actually, S1 is constructed from S0-H, H, and C-S0.
    # Let's just handle all indices in S1, but ensure S0-H and H come first.
    
    # Correct interpretation: Treat s1_indices as a list where S0-related items are at the front
    # because S0 was the base.
    
    # Re-identify components:
    s0_h_indices = []
    s0_minus_h_indices = []
    s0_set_main = set(s0_indices)
    
    # Re-build order: (S0-H) + (H) + (C-S0)
    # This ensures "S0 documents before the additions"
    # We already have this order from our construction of s1_indices
    # (s0_minus_h_indices + added_h + added_c_minus_s0)
    
    for idx in s1_indices:
        doc = docs[idx].strip()
        if not doc: continue
        lines = doc.splitlines()
        title = lines[0] if lines else ""
        norm_prefix = " ".join(doc[:400].split()).lower()
        h = hash(norm_prefix)
        
        if h not in seen_hashes and title not in seen_titles:
            seen_hashes.add(h)
            seen_titles.add(title)
            final_indices.append(idx)

    rng.shuffle(final_indices)
    return final_indices
