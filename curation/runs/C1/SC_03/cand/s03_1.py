import os
import random
import re
import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Token counts
    enc = tiktoken.get_encoding("gpt2")
    token_counts = []
    for d in docs:
        try:
            token_counts.append(len(enc.encode(d)))
        except:
            token_counts.append(len(d) // 4)
            
    # 2. Exact duplicates
    unique_indices = []
    seen_hashes = set()
    for i, d in enumerate(docs):
        h = hash(' '.join(d.split()))
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(i)
    
    # Filtering logic
    is_spam = []
    is_junk = []
    is_shuffled = []
    
    spam_phrases = [
        'Click here to order', 'discount code', 'best price guaranteed',
        'free shipping on all', 'Limited offer on', 'online store |', 'Home | About | Products'
    ]
    
    for i in unique_indices:
        d = docs[i]
        # 3. Spam
        hits = 0
        for p in spam_phrases:
            hits += d.count(p)
        chars = len(d)
        is_spam.append(hits >= 2 or (chars > 0 and hits / (chars / 1500.0) >= 1.0))
        
        # 4. Junk
        non_ascii = sum(1 for c in d if ord(c) > 127)
        letters_spaces = sum(1 for c in d if c.isalpha() or c.isspace())
        is_junk.append(
            non_ascii > len(d) * 0.2 or 
            (len(d) > 0 and (letters_spaces / len(d)) < 0.6) or 
            len(d) < 200
        )
        
        # 5. Repeated-line boilerplate
        lines = d.splitlines()
        is_junk.append(len(lines) >= 6 and (len(set(lines)) / len(lines)) < 0.5)
        
        # 6. Word-order score
        words = d.split()
        score = 0.0
        if len(words) > 0:
            sub_words = words[:800]
            pairs = set(['of the', 'in the', ', and', 'to the', 'on the', 'and the', 'at the', 'by the', '. The', 'for the', 'with the', 'from the', 'was a', 'as a', 'in a', 'of a'])
            count = 0
            for k in range(len(sub_words) - 1):
                pair = f"{sub_words[k]} {sub_words[k+1]}"
                if pair in pairs:
                    count += 1
            if len(sub_words) > 100:
                score = count / (len(sub_words) / 100.0)
            else:
                score = count
        is_shuffled.append(score < 0.3) # Placeholder for now

    # Calculate median_score from target_sample
    target_scores = []
    for ts in target_sample:
        ts_words = ts.split()
        s = 0.0
        if len(ts_words) > 0:
            sub_words = ts_words[:800]
            pairs = set(['of the', 'in the', ', and', 'to the', 'on the', 'and the', 'at the', 'by the', '. The', 'for the', 'with the', 'from the', 'was a', 'as a', 'in a', 'of a'])
            count = 0
            for k in range(len(sub_words) - 1):
                pair = f"{sub_words[k]} {sub_words[k+1]}"
                if pair in pairs:
                    count += 1
            if len(sub_words) > 100:
                s = count / (len(sub_words) / 100.0)
            else:
                s = count
        target_scores.append(s)
    
    median_score = np.median(target_scores) if target_scores else 0.0
    
    # Re-evaluate is_shuffled
    for i, idx in enumerate(unique_indices):
        words = docs[idx].split()
        s = 0.0
        if len(words) > 0:
            sub_words = words[:800]
            pairs = set(['of the', 'in the', ', and', 'to the', 'on the', 'and the', 'at the', 'by the', '. The', 'for the', 'with the', 'from the', 'was a', 'as a', 'in a', 'of a'])
            count = 0
            for k in range(len(sub_words) - 1):
                pair = f"{sub_words[k]} {sub_words[k+1]}"
                if pair in pairs:
                    count += 1
            if len(sub_words) > 100:
                s = count / (len(sub_words) / 100.0)
            else:
                s = count
        is_shuffled[i] = s < 0.3 * median_score

    # 7. Tiers
    tier_indices = [[], [], [], []]
    
    # Tier A
    for i, idx in enumerate(unique_indices):
        d = docs[idx]
        if re.match(r'^\s*= [^=\n]+ =\s*\n', d) and not is_spam[i] and not is_junk[i] and not is_shuffled[i]:
            tier_indices[0].append(idx)
            
    # Tier B
    valid_b_indices = []
    for i, idx in enumerate(unique_indices):
        if not is_spam[i] and not is_junk[i] and not is_shuffled[i] and idx not in tier_indices[0]:
            valid_b_indices.append(idx)
            
    if valid_b_indices:
        vec = TfidfVectorizer(sublinear_tf=True, max_features=50000, lowercase=True)
        train_texts = [docs[i][:2000] for i in valid_b_indices] + [t[:2000] for t in target_sample]
        vec.fit(train_texts)
        
        target_vec = vec.transform([t[:2000] for t in target_sample]).mean(axis=0)
        b_texts = [docs[i][:2000] for i in valid_b_indices]
        b_vecs = vec.transform(b_texts)
        sims = np.dot(b_vecs, target_vec).flatten()
        
        sorted_b = sorted(valid_b_indices, key=lambda x: sims[valid_b_indices.index(x)], reverse=True)
        tier_indices[1] = sorted_b

    # Tier C
    for i, idx in enumerate(unique_indices):
        if is_shuffled[i] and idx not in tier_indices[0] and idx not in tier_indices[1]:
            tier_indices[2].append(idx)
            
    # Tier D
    used = set(tier_indices[0] + tier_indices[1] + tier_indices[2])
    for i, idx in enumerate(unique_indices):
        if idx not in used:
            tier_indices[3].append(idx)

    # 8. Concatenate and select
    final_indices = []
    total_tokens = 0
    for t_idx in range(4):
        for idx in tier_indices[t_idx]:
            if total_tokens < T * 1.01:
                final_indices.append(idx)
                total_tokens += token_counts[idx]
            else:
                break
        if total_tokens >= T * 1.01:
            break

    rng.shuffle(final_indices)
    return final_indices
