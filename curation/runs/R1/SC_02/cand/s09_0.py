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

    # 1. Process initial metadata and flags
    doc_data = []
    for i, d in enumerate(docs):
        t_count = get_tokens(d)
        s_d = str(d)
        is_junk = False
        if len(s_d) < 10 or (len(s_d) > 100 and s_d.count('\n') == 0 and '.' not in s_d):
            is_junk = True
        
        is_wiki = False
        if s_d.startswith('= '):
            first_3k = s_d[:3000]
            count = 0
            for char in [' ', ',', '.', '@', '-']:
                count += first_3k.count(char)
            if count >= 5:
                is_wiki = True
        
        doc_data.append({
            'id': i,
            'text': s_d,
            'tokens': t_count,
            'is_junk': is_junk,
            'is_wiki': is_wiki
        })

    # 2. Shuffle detector on target_sample
    target_words = []
    for s in target_sample:
        target_words.extend(str(s).lower().split())
    
    bigrams = []
    for i in range(len(target_words) - 1):
        bigrams.append(f"{target_words[i]} {target_words[i+1]}")
    
    if bigrams:
        from collections import Counter
        counts = Counter(bigrams)
        fw = [pair for pair, count in counts.most_common(40)]
    else:
        fw = []

    def get_rate(text):
        words = text.lower().split()
        if len(words) < 2: return 0.0
        count = 0
        for i in range(len(words) - 1):
            if f"{words[i]} {words[i+1]}" in fw:
                count += 1
        return count / (len(words) - 1)

    rates = [get_rate(str(s)) for s in target_sample]
    m = np.median(rates) if rates else 0.0

    # 3. Group and Deduplicate Wiki Docs
    wiki_docs = [d for d in doc_data if d['is_wiki'] and not d['is_junk']]
    non_wiki_docs = [d for d in doc_data if not d['is_wiki'] and not d['is_junk']]
    junk_docs = [d for d in doc_data if d['is_junk']]
    
    # Deduplicate wiki based on hash of stripped joined words
    wiki_unique = {}
    duplicates = []
    for d in wiki_docs:
        key = hash(' '.join(d['text'].split()))
        if key not in wiki_unique:
            wiki_unique[key] = d
        else:
            duplicates.append(d)
            
    # Calculate clean/shuffled for wiki
    clean_wiki = []
    shuffled_wiki = []
    
    # Fallback logic
    clean_flag = True
    if wiki_unique:
        wiki_rates = [get_rate(d['text']) for d in wiki_unique.values()]
        clean_count = sum(1 for r in wiki_rates if r >= 0.5 * m)
        total_w = len(wiki_unique)
        if not (0.15 <= (clean_count / total_w) <= 0.97):
            clean_flag = True
        else:
            clean_flag = False
            
        for d in wiki_unique.values():
            if get_rate(d['text']) >= 0.5 * m:
                clean_wiki.append(d)
            else:
                shuffled_wiki.append(d)

    if clean_flag:
        clean_wiki = list(wiki_unique.values())
        shuffled_wiki = []

    # 4. Selection logic
    selected_docs = []
    current_tokens = 0
    budget = 1.01 * T

    # (i) Unique clean wiki
    rng.shuffle(clean_wiki)
    for d in clean_wiki:
        if current_tokens + d['tokens'] <= budget * 1.01:
            selected_docs.append(d)
            current_tokens += d['tokens']
        else:
            break
    
    # (ii) Unique shuffled wiki (by rate desc)
    if shuffled_wiki:
        shuffled_wiki.sort(key=lambda x: get_rate(x['text']), reverse=True)
        for d in shuffled_wiki:
            if current_tokens + d['tokens'] <= budget * 1.01:
                selected_docs.append(d)
                current_tokens += d['tokens']
            else:
                break

    # (iii) Tier B non-wiki (Cosine Similarity)
    target_texts = [str(s) for s in target_sample]
    vec = HashingVectorizer(ngram_range=(1, 2)).fit(target_texts)
    target_matrix = vec.transform(target_texts).toarray()
    centroid = np.mean(target_matrix, axis=0).reshape(1, -1)
    
    def get_sim(text):
        v = vec.transform([text]).toarray()
        return cosine_similarity(v, centroid)[0][0]

    non_wiki_docs.sort(key=lambda d: get_sim(d['text']), reverse=True)
    for d in non_wiki_docs:
        if current_tokens + d['tokens'] <= budget * 1.01:
            selected_docs.append(d)
            current_tokens += d['tokens']
        else:
            break

    # (iv) Duplicates then Junk
    for d in duplicates + junk_docs:
        if current_tokens + d['tokens'] <= budget * 1.01:
            selected_docs.append(d)
            current_tokens += d['tokens']
        else:
            break
            
    # Ensure 1.01*T
    while current_tokens < budget * 1.01 and (duplicates or junk_docs):
        # This handles the "Never return fewer than 1.01*T" if pool permits
        remaining = [d for d in (duplicates + junk_docs) if d not in selected_docs]
        if not remaining: break
        # Take any
        d = remaining[0]
        selected_docs.append(d)
        current_tokens += d['tokens']

    # 5. Ordering
    all_selected_wiki = [d for d in selected_docs if d['is_wiki'] and not d['is_junk']]
    total_s_a_tokens = sum(d['tokens'] for d in all_selected_wiki)
    half_tokens = total_s_a_tokens / 2
    
    a_early, a_late = [], []
    running_a_tokens = 0
    rng.shuffle(all_selected_wiki)
    for d in all_selected_wiki:
        if running_a_tokens + d['tokens'] <= half_tokens:
            a_early.append(d)
            running_a_tokens += d['tokens']
        else:
            a_late.append(d)
            
    if not a_early and a_late:
        a_early.append(a_late.pop(0))

    s_non_a = [d for d in selected_docs if not (d['is_wiki'] and not d['is_junk'])]
    
    phase1 = a_early + s_non_a
    rng.shuffle(phase1)
    phase2 = a_late
    rng.shuffle(phase2)
    
    return [d['id'] for d in phase1] + [d['id'] for d in phase2]
