import os
import random
import re
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")

    def get_tokens(text):
        return text.split()

    def get_bigrams(text):
        tokens = get_tokens(text)
        return [(tokens[i].lower(), tokens[i+1].lower()) for i in range(len(tokens)-1)]

    def get_doc_stats(doc):
        tokens = get_tokens(doc)
        if not tokens: return None, 0, 0
        punct_count = sum(1 for t in tokens if t == ',' or t == '.')
        punct_frac = punct_count / len(tokens)
        lines = doc.splitlines()
        code_spam_count = 0
        for line in lines:
            if any(x in line for x in ['{', '};', 'import ', 'http', 'cheap', 'coupon']):
                code_spam_count += 1
        return punct_frac, code_spam_count / (len(lines) if lines else 1), len(tokens)

    # 1. Identify "Wiki Survivors" (Stage 1 & 2 from original)
    candidate_indices = []
    for i, doc in enumerate(docs):
        punct_frac, code_spam_frac, _ = get_doc_stats(doc)
        if punct_frac >= 0.04 and code_spam_frac < 0.02:
            candidate_indices.append(i)

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
    threshold = (real_rates[int(len(real_rates) * 0.05)] + shuffled_rates[int(len(shuffled_rates) * 0.95)]) / 2

    w_indices = []
    for idx in candidate_indices:
        doc = docs[idx]
        tokens = get_tokens(doc)
        sub_tokens = tokens[:2000]
        if len(sub_tokens) < 2:
            w_indices.append(idx)
            continue
        bg_list = [(sub_tokens[i].lower(), sub_tokens[i+1].lower()) for i in range(len(sub_tokens)-1)]
        hit_rate = sum(1 for b in bg_list if b in target_bigrams) / len(bg_list)
        if hit_rate >= threshold:
            w_indices.append(idx)

    # Dedupe W
    seen_hashes = set()
    seen_titles = set()
    w_indices = []
    for idx in w_indices:
        doc = docs[idx].strip()
        if not doc: continue
        lines = doc.splitlines()
        title = lines[0] if lines else ""
        norm_prefix = " ".join(doc[:400].split()).lower()
        h = hash(norm_prefix)
        if h not in seen_hashes and title not in seen_titles:
            seen_hashes.add(h)
            seen_titles.add(title)
            w_indices.append(idx)

    # 2. Build Prose P
    # Vocabulary from target_sample
    target_vocab = set()
    for ts in target_sample:
        # Clean words
        words = re.findall(r'\w+', ts.lower())
        target_vocab.update(words)

    p_candidates = []
    w_set = set(w_indices)
    for i, doc in enumerate(docs):
        if i in w_set: continue
        
        punct_frac, code_spam_frac, word_count = get_doc_stats(doc)
        if punct_frac >= 0.04 or code_spam_frac >= 0.02 or word_count < 200: continue
        if doc.startswith('= ') or (len(doc) > 0 and (doc[0] in ',.' or doc[-1] in ',.')): continue
        
        alpha_count = sum(1 for char in doc if char.isalpha() or char == ' ')
        if alpha_count / len(doc) < 0.85: continue
        
        tokens = get_tokens(doc)
        sub_tokens = tokens[:1000]
        clean_sub = [re.sub(r'\W+', '', t.lower()) for t in sub_tokens if re.sub(r'\W+', '', t.lower())]
        if not clean_sub: continue
        
        in_vocab = sum(1 for w in clean_sub if w in target_vocab)
        score = in_vocab / len(clean_sub)
        
        norm_prefix = " ".join(doc[:400].split()).lower()
        h = hash(norm_prefix)
        
        # This dedupe is specific to P to keep it clean
        # But for simplicity, we check hash against w and previous p
        if h not in seen_hashes:
            p_candidates.append((score, i))
            seen_hashes.add(h)

    p_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # 3. Calculate Sizes
    nw = sum(len(enc.encode(docs[i])) for i in w_indices)
    
    if nw >= 0.75 * T:
        final_indices = w_indices[:]
        rng.shuffle(final_indices)
        return final_indices

    # Build prefix
    p_indices = [idx for score, idx in p_candidates]
    prefix_indices = []
    current_p_tokens = 0
    max_prefix_tokens = min(0.25 * T, T - nw)
    
    for idx in p_indices:
        token_count = len(enc.encode(docs[idx]))
        if current_p_tokens + token_count <= max_prefix_tokens:
            prefix_indices.append(idx)
            current_p_tokens += token_count
        else:
            break
            
    # Assemble
    # Prefix (shuffled) + W (original) + W (reshuffled)
    rng.shuffle(prefix_indices)
    # We need to repeat W+W until we hit T.
    # The total length is sum(prefix) + sum(W) + sum(W)
    # If that's less than T, we cycle.
    
    full_list = prefix_indices + w_indices + w_indices
    
    # Ensure we cover T
    total_len = sum(len(enc.encode(docs[i])) for i in full_list)
    while total_len < T:
        full_list.extend(w_indices)
        total_len += sum(len(enc.encode(docs[i])) for i in w_indices)
        
    return full_list
