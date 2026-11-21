import os
import random
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import tiktoken

from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Exact-dedupe
    unique_docs = []
    seen_hashes = set()
    for d in docs:
        normalized = " ".join(d.split())
        h = hash(normalized)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(d)
    
    # 2. Prepare texts for TF-IDF
    # Pre-filter for 50 words to avoid issues with empty/tiny strings in some estimators
    prep_docs = []
    for d in unique_docs:
        if len(re.findall(r'\w+', d)) >= 50:
            prep_docs.append(d[:3000])
            
    prep_target = [t[:3000] for t in target_sample]
    all_prep_texts = prep_docs + prep_target
    
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4), 
                                  max_features=200000, sublinear_tf=True, 
                                  dtype=np.float32)
    X = vectorizer.fit_transform(all_prep_texts)
    
    num_prep_docs = len(prep_docs)
    target_rows = np.arange(num_prep_docs, num_prep_docs + len(target_sample))
    
    # 3. Initial Logistic Regression
    neg_indices = rng.sample(np.arange(num_prep_docs), min(4000, num_prep_docs))
    X_train = X[np.concatenate([target_rows, neg_indices])]
    y_train = np.concatenate([np.ones(len(target_rows)), np.zeros(len(neg_indices))])
    
    clf = LogisticRegression(C=10, class_weight='balanced', max_iter=300, solver='liblinear')
    clf.fit(X_train, y_train)
    
    # 4. Self-training
    pool_scores = clf.decision_function(X[:num_prep_docs])
    top_pool_indices = np.argsort(pool_scores)[::-1][:1500]
    bottom_pool_indices = np.argsort(pool_scores)[-6000:]
    
    # Ensure indices are valid (handling case where pool < 6000)
    bottom_pool_indices = bottom_pool_indices[bottom_pool_indices < num_prep_docs]
    
    self_train_pos = np.concatenate([target_rows, top_pool_indices])
    self_train_neg = bottom_pool_indices
    X_self = X[np.concatenate([self_train_pos, self_train_neg])]
    y_self = np.concatenate([np.ones(len(self_train_pos)), np.zeros(len(self_train_neg))])
    
    clf.fit(X_self, y_self)
    final_pool_scores = clf.decision_function(X[:num_prep_docs])
    
    # 5. Junk filter
    target_words = set(re.findall(r'\w+', " ".join(target_sample).lower()))
    valid_pool_indices = []
    for i, d in enumerate(unique_docs):
        if len(re.findall(r'\w+', d)) < 50:
            continue
        first_3000 = d[:3000]
        words_in_doc = re.findall(r'\w+', first_3000.lower())
        if not words_in_doc:
            continue
        matches = sum(1 for w in words_in_doc if w in target_words)
        if matches / len(words_in_doc) < 0.6:
            continue
        valid_pool_indices.append(i)
    
    # 6. Near-duplicate filter
    # group by first line "= Title ="
    title_map = {}
    # We only filter indices that passed the Junk Filter
    for i in valid_pool_indices:
        d = unique_docs[i]
        lines = d.splitlines()
        title = ""
        for line in lines:
            if "= Title =" in line:
                title = line.strip()
                break
        if not title:
            # If no title line found, use first line
            title = lines[0].strip() if lines else ""
            
        if title not in title_map:
            title_map[title] = []
        title_map[title].append(i)
    
    final_keep_indices = []
    tk = tiktoken.get_encoding("gpt2")
    
    # Score calculation for ranking
    scored_indices = []
    for i in valid_pool_indices:
        scored_indices.append((i, final_pool_scores[i]))
    
    scored_indices.sort(key=lambda x: x[1], reverse=True)
    
    # Near-duplicate logic: for each title, keep one best
    # We process the scored_indices to respect the ranking
    processed_titles = set()
    refined_indices = []
    for idx, score in scored_indices:
        # Re-determine title for this doc
        d = unique_docs[idx]
        lines = d.splitlines()
        title = ""
        for line in lines:
            if "= Title =" in line:
                title = line.strip()
                break
        if not title: title = lines[0].strip() if lines else ""
        
        if title not in processed_titles:
            refined_indices.append(idx)
            processed_titles.add(title)

    # 7. Token count and thresholding
    # Sort by score descending
    refined_indices.sort(key=lambda i: final_pool_scores[i], reverse=True)
    
    selected_indices = []
    current_tokens = 0
    
    # Calculate total tokens first
    doc_tokens = []
    for idx in refined_indices:
        tokens = len(tk.encode(unique_docs[idx]))
        doc_tokens.append(tokens)
    
    total_available = sum(doc_tokens)
    
    # find score threshold
    # We need to select indices such that we get at least 0.3 T
    # and at most 1.0 T.
    
    # Helper to get tokens of a doc
    def get_t(i): return doc_tokens[refined_indices.index(i)]

    # First pass: try to get 1.0 T with score > 0
    temp_indices = []
    temp_tokens = 0
    for i, idx in enumerate(refined_indices):
        if final_pool_scores[idx] > 0:
            temp_indices.append(idx)
            temp_tokens += doc_tokens[i]
            if temp_tokens > T:
                break
    
    if temp_tokens >= T * 0.3:
        # If we have enough, but it's over 1.0 T, we need to prune 
        # but the requirement says "cut at 1.0 T"
        # Let's refine the selection
        final_selection = []
        running_tokens = 0
        for i, idx in enumerate(refined_indices):
            if final_pool_scores[idx] > 0:
                if running_tokens + doc_tokens[i] <= T:
                    final_selection.append(idx)
                    running_tokens += doc_tokens[i]
                else:
                    # Take partial if possible? No, must be whole docs.
                    # Just stop.
                    break
        # If final_selection is empty or too small, we use the 0.3T logic
        if len(final_selection) > 0:
            selected_indices = final_selection
        else:
            # Fallback to 0.3 T logic
            # ...
            pass
    
    # Let's simplify the logic to match the requirement precisely:
    # "Cut at 1.0 T. If fewer than 0.3 T, relax score threshold until 0.3 T"
    
    def build_selection(min_score):
        res = []
        count = 0
        for i, idx in enumerate(refined_indices):
            if final_pool_scores[idx] >= min_score:
                if count + doc_tokens[i] <= T:
                    res.append(idx)
                    count += doc_tokens[i]
                else:
                    break
                if count >= T * 0.3: return res
        return res

    # Try score > 0
    selected_indices = build_selection(0.0)
    
    # If count < 0.3 T, relax threshold
    # Since the list is sorted by score, relaxing means we just take more from the bottom
    if sum(doc_tokens[refined_indices.index(i)] for i in selected_indices) < T * 0.3:
        # Relaxing score threshold means we take docs regardless of score 
        # until 0.3 T is reached.
        res = []
        count = 0
        for i, idx in enumerate(refined_indices):
            res.append(idx)
            count += doc_tokens[i]
            if count >= T * 0.3:
                # Now we have 0.3 T. We continue adding until 1.0 T.
                # But we must respect the "cut at 1.0 T"
                for j in range(i + 1, len(refined_indices)):
                    if count + doc_tokens[j] <= T:
                        res.append(refined_indices[j])
                        count += doc_tokens[j]
                    else:
                        break
                return res
        selected_indices = res

    # 8. Shuffle and return
    rng.shuffle(selected_indices)
    return selected_indices
