import os
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import tiktoken
from common import T

def select(docs, target_sample):
    N = len(docs)
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = np.random.default_rng(seed)
    
    # 1. Dedupe mask
    keep = np.ones(N, dtype=bool)
    seen = set()
    for i in range(N):
        # Normalize for dedupe: strip whitespace and hash
        key = hash(" ".join(docs[i].split()))
        if key in seen:
            keep[i] = False
        else:
            seen.add(key)

    # 2. Word-order feature
    def get_bigrams(text):
        words = text[:4000].lower().split()
        if len(words) < 2:
            return set(), 0
        bigrams = set()
        for i in range(len(words) - 1):
            bigrams.add((words[i], words[i+1]))
        return bigrams, len(words)

    B = set()
    for t in target_sample:
        b_set, _ = get_bigrams(t)
        B.update(b_set)

    big = np.zeros(N)
    nw = np.zeros(N)
    for i in range(N):
        b_set, count = get_bigrams(docs[i])
        nw[i] = count
        if count < 2:
            big[i] = 0
        else:
            big[i] = len(b_set.intersection(B)) / len(b_set) if b_set else 0

    # Calibration
    B1 = set()
    for t in target_sample[:100]:
        b_set, _ = get_bigrams(t)
        B1.update(b_set)
    
    q_list = []
    for t in target_sample[100:]:
        b_set, _ = get_bigrams(t)
        hit_frac = len(b_set.intersection(B1)) / len(b_set) if b_set else 0
        q_list.append(hit_frac)
    
    q = np.percentile(q_list, 5) if q_list else 0.5
    thr_big = 0.6 * q

    # 3. Classifier
    texts = [d[:3000] for d in docs] + [t[:3000] for t in target_sample]
    # Use a smaller max_features to ensure we stay within memory/time limits if N is large
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1,3), 
                                  max_features=100000, sublinear_tf=True, 
                                  dtype=np.float32)
    X = vectorizer.fit_transform(texts)
    Xp = X[:N]
    Xt = X[N:]

    neg_indices = rng.choice(np.arange(N), size=min(N, 4000), replace=False)
    pos_indices = np.arange(N, N + 200)
    
    # Combine indices for training
    train_idx = np.concatenate([pos_indices, neg_indices])
    clf = LogisticRegression(C=10, class_weight='balanced', max_iter=200, 
                              solver='liblinear')
    clf.fit(X[train_idx], [1]*200 + [0]*len(neg_indices))
    
    score = clf.decision_function(Xp)

    # Self-training
    candidates = np.where((keep) & (big >= thr_big) & (nw >= 50))[0]
    # Score the candidates
    cand_scores = score[candidates]
    if len(candidates) > 0:
        # Select top 1500 candidates
        sorted_cand_idx = candidates[np.argsort(cand_scores)[::-1]]
        top_1500 = sorted_cand_idx[:1500]
        
        neg_self = np.where(score < np.percentile(score, 0.1))[0]
        neg_self = neg_self[neg_self < N]
        # Ensure unique
        neg_self = np.unique(np.concatenate([neg_self, neg_indices]))
        neg_self = neg_self[neg_self < N]

        train_idx_2 = np.concatenate([np.arange(N, N+200), top_1500, neg_self])
        # Re-filter indices for safety
        train_idx_2 = train_idx_2[train_idx_2 < (N + 200)] 
        # Since X is size (N+200), we just need the indices for X
        # Actually, X is shape (N+200, features). Xp is X[:N]
        # Just need the indices relative to the original list.
        # Let's simplify: 
        clf.fit(X[train_idx_2], [1]*200 + [1]*len(top_1500) + [0]*len(neg_self))
        score = clf.decision_function(Xp)

    # 4. Final Mask
    good = keep & (big >= thr_big) & (nw >= 50) & (score > 0)

    # 5. Title Dedupe
    # Use a dict to store {title_str: (best_big_score, original_index)}
    titles = {}
    for i in range(N):
        if good[i]:
            lines = docs[i].splitlines()
            if lines and lines[0].startswith('='):
                title = lines[0].strip()
                if title not in titles or big[i] > titles[title][0]:
                    titles[title] = (big[i], i)
                else:
                    good[i] = False
    
    # Re-set good based on titles
    good = np.zeros(N, dtype=bool)
    for title in titles:
        good[titles[title][1]] = True

    # 6. Token Counting and Selection
    sel_indices = np.flatnonzero(good)
    enc = tiktoken.get_encoding("gpt2")
    
    def count_tokens(idx_list):
        all_toks = []
        for idx in idx_list:
            all_toks.append(enc.encode(docs[idx]))
        return sum(len(t) for t in all_toks)

    # Selection logic
    current_sel = list(sel_indices)
    total_tokens = count_tokens(current_sel)
    
    if total_tokens > T:
        # Keep highest scores until T
        # scores[sel_indices] sorted descending
        scores_in_good = score[sel_indices]
        sorted_in_good = sel_indices[np.argsort(scores_in_good)[::-1]]
        current_sel = []
        running_total = 0
        for idx in sorted_in_good:
            # Need to count this specific doc
            doc_tokens = len(enc.encode(docs[idx]))
            if running_total + doc_tokens > T:
                break
            current_sel.append(idx)
            running_total += doc_tokens
    elif total_tokens < 0.4 * T:
        # Add from keep & big & nw in descending score
        fallback_pool = np.where((keep) & (big >= thr_big) & (nw >= 50))[0]
        scores_fallback = score[fallback_pool]
        sorted_fallback = fallback_pool[np.argsort(scores_fallback)[::-1]]
        
        # Remove what's already in current_sel to avoid duplicates
        current_sel_set = set(current_sel)
        for idx in sorted_fallback:
            if idx not in current_sel_set:
                doc_tokens = len(enc.encode(docs[idx]))
                if total_tokens + doc_tokens <= 0.4 * T:
                    current_sel.append(idx)
                    total_tokens += doc_tokens
                else:
                    break
                    
    # Final Shuffle
    final_indices = list(current_sel)
    # To satisfy "return what passed and let the trainer cycle", 
    # we need a result list that covers at least some tokens.
    # If current_sel is empty, just pick any from good.
    if not final_indices and len(sel_indices) > 0:
        final_indices = list(sel_indices)
    
    # Shuffle
    shuffler = random.Random(seed)
    shuffler.shuffle(final_indices)
    
    # Ensure indices are within bounds
    final_indices = [int(i) for i in final_indices if i < N]
    return final_indices
