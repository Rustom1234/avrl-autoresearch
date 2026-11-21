import os
import random
import re
import hashlib
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
import tiktoken

from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    def get_tokens(text_list):
        # The trainer gives us a list of strings, we need list of counts
        counts = []
        for text in text_list:
            counts.append(len(enc.encode(text)))
        return counts

    # 1. Exact dedupe
    unique_docs = {}
    for doc in docs:
        norm = doc.lower().split()
        h = hashlib.md5(" ".join(norm).encode("utf-8")).hexdigest()
        if h not in unique_docs:
            unique_docs[h] = doc
    docs = list(unique_docs.values())
    
    # 2. Token counts
    token_counts = get_tokens(docs)
    # doc_tokens_sum is no longer used directly in a way that causes the error
    
    # 3. Junk Flag
    def is_junk(doc):
        if len(doc) < 200: return True
        words = doc.split()
        first_500 = words[:500]
        if len(first_500) > 0:
            unique_ratio = len(set(first_500)) / len(first_500)
            if unique_ratio < 0.25: return True
        lines = doc.splitlines()
        if len(lines) >= 5:
            for line in lines:
                if len(line) / len(doc) > 0.3: return True
        alpha_count = sum(1 for c in doc if c.isalpha() or c == ' ')
        if alpha_count / len(doc) < 0.6: return True
        spam = ['discount code', 'coupon', 'free shipping', 'buy now', 'best price']
        if any(s in doc.lower() for s in spam): return True
        return False

    junk_flags = [is_junk(d) for d in docs]

    # 4. Shuffle Detector
    train_pos = []
    train_neg = []
    for ts in target_sample:
        words = ts.split()
        for i in range(0, len(words), 150):
            chunk = words[i:i+150]
            if len(chunk) > 0: 
                train_pos.append(" ".join(chunk))
                shuffled = list(chunk)
                rng.shuffle(shuffled)
                train_neg.append(" ".join(shuffled))

    vec = HashingVectorizer(analyzer='word', token_pattern=r'\S+', ngram_range=(2,2), lowercase=True, n_features=2**18, binary=True)
    X = vec.transform(train_pos + train_neg)
    y = [1] * len(train_pos) + [0] * len(train_neg)
    clf = LogisticRegression(C=1).fit(X, y)

    def get_p_clean(doc):
        words = doc.split()
        def get_score(chunk):
            if not chunk: return 0.5
            # Note: LogisticRegression.predict returns a 2D array if input is 2D
            # vec.transform returns a sparse matrix.
            pred = clf.predict(vec.transform([" ".join(chunk)]))[0]
            return float(pred) * 2 - 1
        
        mid = len(words) // 2
        s1 = get_score(words[:mid])
        s2 = get_score(words[mid:])
        p_clean = (s1 + s2) / 2
        
        count_signal = 0
        score_signal = 0
        for i in range(len(doc)-3):
            if doc[i:i+3] == " . ":
                count_signal += 1
                next_chars = doc[i+3:i+5]
                if next_chars and (next_chars[0].isupper() or next_chars[0] in ('\n', '"', "'")) or i+3 == len(doc):
                    score_signal += 1
        
        if count_signal > 0:
            if score_signal / count_signal < 0.6: return 0.0
            
        return p_clean

    shuffled_flags = []
    for i, doc in enumerate(docs):
        p_clean = get_p_clean(doc)
        shuffled_flags.append(p_clean < 0.5)

    # 5. Wiki Flag
    def is_wiki(doc):
        if not doc.startswith("= "): return False
        first_3k = doc[:3000]
        count = 0
        for sub in [' , ', ' . ', ' @-@ ']:
            count += first_3k.count(sub)
        return count >= 5

    wiki_flags = [is_wiki(d) for d in docs]

    # 6. Tiers
    target_words = Counter(" ".join(target_sample).split())
    
    tier_a = []
    tier_b = []
    tier_c = []
    tier_d = []

    for i in range(len(docs)):
        w = wiki_flags[i]
        s = shuffled_flags[i]
        j = junk_flags[i]
        
        if w and not s and not j: tier_a.append(i)
        elif not w and not s and not j: tier_b.append(i)
        elif w and s: tier_c.append(i)
        else: tier_d.append(i)

    def score_b(idx):
        doc_words = Counter(docs[idx].split())
        sim = 0
        for w, count in doc_words.items():
            if w in target_words:
                sim += (1 + np.log(count + 1)) * (1 + np.log(target_words[w] + 1))
        return sim

    tier_b.sort(key=score_b, reverse=True)

    # 7. Build list
    final_indices = []
    current_tokens = 0
    
    # Tier A
    for idx in tier_a:
        final_indices.append(idx)
        current_tokens += token_counts[idx]
    
    if current_tokens < 0.6 * T:
        for idx in tier_a:
            final_indices.append(idx)
            current_tokens += token_counts[idx]

    # Tier B
    for idx in tier_b:
        if current_tokens < 1.02 * T:
            final_indices.append(idx)
            current_tokens += token_counts[idx]
    
    # Tier C then D
    for idx in tier_c + tier_d:
        if current_tokens < 1.02 * T:
            final_indices.append(idx)
            current_tokens += token_counts[idx]

    # 8. Order
    rng.shuffle(final_indices)
    return final_indices
