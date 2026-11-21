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
    p5_real = real_rates[int(len(real_rates) * 0.05)]
    p95_shuf = shuffled_rates[int(len(shuffled_rates) * 0.95)]
    threshold = (p5_real + p95_shuf) / 2

    filtered_indices = []
    for idx in candidate_indices:
        doc = docs[idx]
        tokens = get_tokens(doc)
        sub_tokens = tokens[:2000]
        if len(sub_tokens) < 2:
            filtered_indices.append(idx)
            continue
        
        bg_list = [(sub_tokens[i].lower(), sub_tokens[i+1].lower()) for i in range(len(sub_tokens)-1)]
        hit_rate = sum(1 for b in bg_list if b in target_bigrams) / len(bg_list)
        if hit_rate >= threshold:
            filtered_indices.append(idx)

    # (a) DF Table and Rare Rate
    doc_counts = Counter()
    # We need to look at pool candidates + target_sample to build the DF
    # However, the prompt says "survivors plus target_sample"
    # Survivors are the indices in filtered_indices
    for idx in filtered_indices:
        tokens = get_tokens(docs[idx])
        for t in tokens:
            doc_counts[t.lower()] += 1
    for ts in target_sample:
        tokens = get_tokens(ts)
        for t in tokens:
            doc_counts[t.lower()] += 1

    def get_rare_rate(doc_text):
        tokens = get_tokens(doc_text)
        sub = tokens[:3000]
        rare_count = 0
        alpha_count = 0
        for t in sub:
            low_t = t.lower()
            if low_t.isalpha() and len(low_t) >= 4:
                alpha_count += 1
                if doc_counts[low_t] <= 2:
                    rare_count += 1
        return rare_count / alpha_count if alpha_count > 0 else 0.0

    # (b) Calibration
    target_rare_rates = [get_rare_rate(ts) for ts in target_sample]
    target_rare_rates.sort()
    med = target_rare_rates[len(target_rare_rates)//2]
    p95 = target_rare_rates[int(len(target_rare_rates) * 0.95)]
    limit = max(p95 * 1.5, med + 0.02)
    
    survivors = []
    for idx in filtered_indices:
        rr = get_rare_rate(docs[idx])
        if rr <= limit:
            survivors.append(idx)

    # (c) Dedupe by group
    # Grouping keys: title line and hash of first 400 chars
    groups_title = {} # title -> list of (idx, rare_rate, hit_rate, length)
    groups_hash = {}  # hash -> list of (idx, rare_rate, hit_rate, length)
    
    target_bigrams_set = set()
    for ts in target_sample:
        target_bigrams_set.update(get_bigrams(ts))

    for idx in survivors:
        doc = docs[idx]
        rr = get_rare_rate(doc)
        
        # Calculate hit_rate for the tie-break
        tokens = get_tokens(doc)
        sub_tokens = tokens[:2000]
        bg_list = [(sub_tokens[i].lower(), sub_tokens[i+1].lower()) for i in range(len(sub_tokens)-1)]
        hit_rate = sum(1 for b in bg_list if b in target_bigrams_set) / len(bg_list) if bg_list else 0
        
        # Preparation for sorting: (rare_rate (low), -hit_rate (high), -length (high))
        # Since we want smallest RR, then largest hit_rate, then largest length
        data = (idx, rr, hit_rate, len(doc))
        
        lines = doc.splitlines()
        title = lines[0] if lines else ""
        norm_prefix = " ".join(doc[:400].split()).lower()
        h = hash(norm_prefix)
        
        groups_title.setdefault(title, []).append(data)
        groups_hash.setdefault(h, []).append(data)

    # We need to select the best from each group.
    # Since a doc can belong to two groups, we must ensure that if we pick a doc 
    # because it's the best in TitleGroup A, and it's also in HashGroup B, 
    # it doesn't prevent us from picking the best in HashGroup B if it's a different doc.
    # However, the prompt says: "keep only the single document...". 
    # To satisfy both conditions, a doc must be the "winner" of its Title group 
    # AND its Hash group to be uniquely identified? 
    # Standard interpretation for such "dedupe by group" is:
    # The final set contains docs that are the "best" in their respective Title AND Hash categories.
    
    winners_indices = set()
    
    def get_best(group_list):
        if not group_list: return None
        # Sort by RR (asc), HitRate (desc), Length (desc)
        best = min(group_list, key=lambda x: (x[1], -x[2], -x[3]))
        return best[0]

    best_titles = {}
    for title, items in groups_title.items():
        best_titles[title] = get_best(items)
    
    best_hashes = {}
    for h, items in groups_hash.items():
        best_hashes[h] = get_best(items)

    # The logic "keep only the single document" in two different group types 
    # implies the set of winners is the union of winners of Title groups and Hash groups, 
    # but only if they are the "unique" winners. 
    # Given "keep only the single", we'll collect winners of both and deduplicate.
    
    final_indices = set()
    for idx in best_titles.values():
        if idx is not None: final_indices.add(idx)
    for idx in best_hashes.values():
        if idx is not None: final_indices.add(idx)
        
    # The most robust way to interpret "keep the cleanest" across two groupings 
    # is to take the set of all winners from both.
    # If a doc is the best in its title group, it's a candidate.
    # If it's the best in its hash group, it's a candidate.
    
    # Let's refine: if it belongs to a group, it must be the best in that group.
    # If it is the best in Title AND its Hash group, it's definitely a winner.
    # If it is best in Title but NOT its Hash group, it means there's a cleaner copy 
    # of the same text content (hash), so we shouldn't pick it.
    
    # Revised logic: A doc is a winner if it is the best in its title group 
    # AND it is also the best in its hash group.
    
    final_indices = []
    for title, items in groups_title.items():
        best = min(items, key=lambda x: (x[1], -x[2], -x[3]))
        idx = best[0]
        # Check if it's also the best in its hash group
        # We need to find its hash
        doc_text = docs[idx]
        h_val = hash(" ".join(doc_text[:400].split()).lower())
        if best[0] == best_hashes[h_val]:
            final_indices.append(idx)

    rng.shuffle(final_indices)
    return final_indices
