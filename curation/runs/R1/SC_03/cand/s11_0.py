import os
import random
import re
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def get_tokens(text):
        return text.split()

    def is_s0_candidate(doc):
        tokens = get_tokens(doc)
        if not tokens: return False
        punct_count = sum(1 for t in tokens if t == ',' or t == '.')
        punct_frac = punct_count / len(tokens)
        
        spam_words = ['coupon', 'discount code', 'buy now', 'free shipping', 'click here', 'limited offer']
        lower_doc = doc.lower()
        spam_score = sum(lower_doc.count(sw) for sw in spam_words)
        
        # Spam phrase density: spam_score / (len(tokens)/1000)
        # This is approx spam_score * 1000 / len(tokens)
        density = (spam_score * 1000) / len(tokens) if len(tokens) > 0 else 0
        
        if punct_frac >= 0.04 and density <= 2.0:
            return True
        return False

    s0_indices = [i for i, doc in enumerate(docs) if is_s0_candidate(doc)]
    
    # Build positives
    pos_chunks = []
    for ts in target_sample:
        tokens = get_tokens(ts)
        # Target sample chunks of 300 words
        for i in range(0, len(tokens), 300):
            pos_chunks.append(" ".join(tokens[i:i+300]))
            
    s0_tokens = []
    for idx in s0_indices:
        tokens = get_tokens(docs[idx])
        if len(tokens) > 0:
            s0_tokens.append(rng.choice(tokens))
    
    # We need to pick one random chunk from each S0 doc (for the prompt's logic)
    # Actually, the prompt says "one random chunk from each of 1,500 seeded-random S0 documents"
    # Since S0 might be smaller or larger, we take min(1500, len(s0_indices))
    s0_limit = min(1500, len(s0_indices))
    s0_sampled_indices = rng.sample(s0_indices, s0_limit)
    
    pos_chunks_final = []
    for idx in s0_sampled_indices:
        tokens = get_tokens(docs[idx])
        # Pick a random chunk of 300
        if len(tokens) > 300:
            start = rng.randint(0, len(tokens) - 300)
            pos_chunks_final.append(" ".join(tokens[start:start+300]))
        else:
            pos_chunks_final.append(" ".join(tokens))
    
    # Combine targets + s0_sampled into pos_chunks
    pos_data = pos_chunks + pos_chunks_final
    
    # Negatives
    neg_data = []
    # Type 1
    for p in pos_data:
        dmg_type = rng.randint(0, 5)
        tokens = p.split()
        if dmg_type == 0: # Shuffle within sentences
            sentences = re.split(r' \. ', p)
            dmg = " ".join([" ".join(random.sample(s.split(), len(s.split()))) for s in sentences])
        elif dmg_type == 1: # Shuffle all
            dmg = " ".join(random.sample(tokens, len(tokens)))
        elif dmg_type == 2: # Char swap
            chars = list(p)
            for i in range(len(chars)):
                if rng.random() < 0.08 and chars[i].isalpha():
                    chars[i] = rng.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
            dmg = "".join(chars)
        elif dmg_type == 3: # Lower
            dmg = p.lower()
        elif dmg_type == 4: # Upper
            dmg = p.upper()
        else: # Failed punct doc
            dmg = docs[rng.randint(0, len(docs)-1)]
        neg_data.append(dmg)
        
    # Type 2: 1500 random chunks from failed punct
    fail_punct_docs = [docs[i] for i in range(len(docs)) if not is_s0_candidate(docs[i])]
    if fail_punct_docs:
        for _ in range(min(1500, len(fail_punct_docs))):
            doc = rng.choice(fail_punct_docs)
            tks = get_tokens(doc)
            if tks: neg_data.append(" ".join(tks[:300]))
            
    y = np.array([1]*len(pos_data) + [0]*len(neg_data))
    x_raw = pos_data + neg_data
    
    # Model
    # To keep it fast, we use a small subset of x_raw if it's massive
    x_subset = x_raw[:min(len(x_raw), 20000)]
    y_subset = y[:min(len(y), 20000)]
    
    v1 = HashingVectorizer(token_pattern=r'\S+', ngram_range=(1, 2), algorithm='once', nfeatures=2**18, binary=True)
    v2 = HashingVectorizer(analyzer='char', token_pattern=r'\S+', ngram_range=(1, 3), algorithm='once', nfeatures=2**16, binary=True)
    
    X1 = v1.transform(x_subset).toarray()
    X2 = v2.transform(x_subset).toarray()
    X = np.hstack([X1, X2])
    
    clf = LogisticRegression(C=1.0, max_iter=200, solver='liblinear').fit(X, y_subset)
    
    # Score C
    candidate_indices = [i for i, doc in enumerate(docs) if is_s0_candidate(doc)]
    
    # Cross-fitting (simplified as requested if slow)
    # For the sake of 180s, we skip heavy cross-fitting and use 0.5/0.2
    t_mean, t_min = 0.5, 0.2
    
    scored_docs = []
    for idx in candidate_indices:
        doc = docs[idx]
        tokens = get_tokens(doc)
        num_chunks = 6
        chunk_size = 300
        chunk_indices = []
        if len(tokens) <= chunk_size:
            chunk_indices = [0]
        else:
            # spread across whole doc
            step = len(tokens) // num_chunks
            for i in range(num_chunks):
                chunk_indices.append(i * step)
                if i == num_chunks - 1: break
        
        probs = []
        for start in chunk_indices:
            chunk_text = " ".join(tokens[start : start + chunk_size])
            X1_c = v1.transform([chunk_text]).toarray()
            X2_c = v2.transform([chunk_text]).toarray()
            Xc = np.hstack([X1_c, X2_c])
            probs.append(clf.predict_proba(Xc)[0][1])
            
        mean_p = np.mean(probs)
        min_p = np.min(probs)
        
        if mean_p >= t_mean and min_p >= t_min:
            scored_docs.append((idx, mean_p))
            
    if len(scored_docs) < 0.85 * len(s0_indices) or len(scored_docs) > 1.4 * len(s0_indices):
        final_indices = s0_indices
    else:
        # Dedupe
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        seen_hashes = set()
        seen_titles = set()
        final_indices = []
        for idx, score in scored_docs:
            doc = docs[idx].strip()
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
