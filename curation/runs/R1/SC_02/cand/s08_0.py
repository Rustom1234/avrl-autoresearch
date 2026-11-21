import os
import random
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")

    def get_tokens(d):
        try:
            return len(enc.encode_ordinary(str(d)))
        except:
            return len(str(d)) // 4

    # Pre-calculate metadata
    doc_data = []
    for i, d in enumerate(docs):
        t_count = get_tokens(d)
        
        # Junk flag
        is_junk = False
        if len(d) < 10 or (len(d) > 100 and d.count('\n') == 0 and '.' not in d):
            is_junk = True
        
        # Wiki flag
        is_wiki = False
        if d.startswith('= '):
            first_3k = d[:3000]
            count = 0
            for char in [' ', ',', '.', '@', '-']:
                count += first_3k.count(char)
            if count >= 5:
                is_wiki = True
        
        p_clean = 0.5
        doc_data.append({
            'id': i,
            'tokens': t_count,
            'is_junk': is_junk,
            'is_wiki': is_wiki,
            'p_clean': p_clean
        })

    # Filter tiers
    tier_a = [] # Clean wiki
    tier_b = [] # Clean non-wiki
    tier_d = [] # Junk/Rest

    for item in doc_data:
        if item['is_junk']:
            tier_d.append(item)
        elif item['is_wiki']:
            tier_a.append(item)
        else:
            tier_b.append(item)
            
    # Similarity for Tier B
    target_texts = [str(s) for s in target_sample]
    vec = HashingVectorizer(ngram_range=(1, 2)).fit(target_texts)
    target_matrix = vec.transform(target_texts).toarray()
    centroid = np.mean(target_matrix, axis=0).reshape(1, -1)
    
    def get_sim(text):
        v = vec.transform([text]).toarray()
        return cosine_similarity(v, centroid)[0][0]

    for item in tier_b:
        item['sim'] = get_sim(docs[item['id']])
    
    tier_b.sort(key=lambda x: x['sim'], reverse=True)
    
    # Selection
    S = []
    a_tokens_total_pool = sum(item['tokens'] for item in tier_a)
    target_limit = 1.01 * T
    
    if a_tokens_total_pool >= target_limit:
        tier_a.sort(key=lambda x: x['p_clean'], reverse=True)
        current_s_tokens = 0
        for item in tier_a:
            if current_s_tokens + item['tokens'] <= target_limit * 1.01:
                S.append(item)
                current_s_tokens += item['tokens']
    else:
        S.extend(tier_a)
        current_s_tokens = a_tokens_total_pool
        for item in tier_b:
            if current_s_tokens + item['tokens'] <= target_limit * 1.01:
                S.append(item)
                current_s_tokens += item['tokens']
        for item in tier_d:
            if current_s_tokens + item['tokens'] <= target_limit * 1.01:
                S.append(item)
                current_s_tokens += item['tokens']

    # Tier A logic refinement
    s_a_docs = [item for item in S if item['is_wiki'] and not item['is_junk']]
    
    # Build G
    G = set()
    full_target_text = " ".join(target_sample).lower()
    words = full_target_text.split()
    for i in range(len(words) - 1):
        bg = (words[i], words[i+1])
        G.add(bg)
    # The prompt asks for "at least 3 times in all", but usually implies 
    # union of counts. To be safe with "all", we check occurrences.
    G = set()
    word_counts = {}
    words_all = full_target_text.split()
    for i in range(len(words_all) - 1):
        bg = (words_all[i], words_all[i+1])
        word_counts[bg] = word_counts.get(bg, 0) + 1
    G = {bg for bg, count in word_counts.items() if count >= 3}

    scores = []
    seen_hashes = set()
    
    for i, item in enumerate(s_a_docs):
        d = docs[item['id']]
        content = d.split('\n', 1)[1] if '\n' in d else d
        tokens = enc.encode_ordinary(content)
        
        # Compute r
        r_count = 0
        total_r = 0
        # Take tokens 0-600
        sub_tokens = tokens[:600]
        # Find bigrams in tokens
        # Note: "bigrams in its tokens" refers to the string bigrams 
        # created by splitting the text into words.
        text_words = content.lower().split()
        for j in range(len(text_words) - 1):
            bg = (text_words[j], text_words[j+1])
            total_r += 1
            if bg in G:
                r_count += 1
        r = r_count / total_r if total_r > 0 else 0
        
        # Compute p
        p_count = 0
        total_p = 0
        dots = [i for i, char in enumerate(d) if d[i:i+3] == ' . ']
        if len(dots) < 3:
            p = 0.5
        else:
            valid_p = 0
            for idx in dots:
                after = d[idx+3:]
                if not after:
                    valid_p += 1
                else:
                    first_char = after[0]
                    if first_char.isupper() or first_char == '\n' or first_char == '"' or first_char == '=':
                        valid_p += 1
            p = valid_p / len(dots)
        
        # Duplicate check
        h = hash(content.lower().split())
        if h in seen_hashes:
            score = -1
        else:
            seen_hashes.add(h)
            # Percentile ranks
            # Using sorted list to find rank
            r_rank = 0 # Placeholder
            p_rank = 0 # Placeholder
            # Actually we need ranks over all s_a_docs
            # To do this efficiently, we store the pair first
            scores.append((r, p, i))
        
    # Re-calculate scores with ranks
    # We need to collect all (r, p) pairs first
    all_r_p = []
    for i, item in enumerate(s_a_docs):
        d = docs[item['id']]
        content = d.split('\n', 1)[1] if '\n' in d else d
        tokens = enc.encode_ordinary(content)
        
        # r
        text_words = content.lower().split()
        r_count = 0
        total_r = 0
        for j in range(len(text_words) - 1):
            total_r += 1
            if (text_words[j], text_words[j+1]) in G:
                r_count += 1
        r = r_count / total_r if total_r > 0 else 0
        
        # p
        dots = [idx for idx, char in enumerate(d) if d[idx:idx+3] == ' . ']
        if len(dots) < 3:
            p = 0.5
        else:
            valid_p = 0
            for idx in dots:
                after = d[idx+3:]
                if not after or after[0].isupper() or after[0] == '\n' or after[0] == '"' or after[0] == '=':
                    valid_p += 1
            p = valid_p / len(dots)
        all_r_p.append((r, p))

    # Calculate ranks
    rs = sorted([x[0] for x in all_r_p])
    ps = sorted([x[1] for x in all_r_p])
    
    final_scores = []
    seen_hashes = set()
    for i, item in enumerate(s_a_docs):
        d = docs[item['id']]
        content = d.split('\n', 1)[1] if '\n' in d else d
        h = hash(content.lower().split())
        
        if h in seen_hashes:
            final_scores.append(-1)
        else:
            seen_hashes.add(h)
            r, p = all_r_p[i]
            # Rank calculation: percentile (0..1)
            r_rank = rs.index(r) / (len(rs) - 1) if len(rs) > 1 else 0
            p_rank = ps.index(p) / (len(ps) - 1) if len(ps) > 1 else 0
            final_scores.append(r_rank + 0.5 * p_rank)

    # Associate scores with s_a_docs
    scored_a = []
    for i, item in enumerate(s_a_docs):
        scored_a.append((final_scores[i], item))
    
    # Order logic
    scored_a.sort(key=lambda x: x[0], reverse=True)
    
    total_a_tokens = sum(item['tokens'] for item in s_a_docs)
    half_tokens = total_a_tokens / 2
    
    a_late = []
    a_early = []
    running_a_tokens = 0
    for score, item in scored_a:
        if running_a_tokens + item['tokens'] <= half_tokens:
            a_early.append(item)
            running_a_tokens += item['tokens']
        else:
            a_late.append(item)

    # Phase Construction
    s_non_a = [item for item in S if not (item['is_wiki'] and not item['is_junk'])]
    phase1 = a_early + s_non_a
    rng.shuffle(phase1)
    
    phase2 = a_late
    rng.shuffle(phase2)
    
    final_selection = [item['id'] for item in phase1] + [item['id'] for item in phase2]
    return final_selection
