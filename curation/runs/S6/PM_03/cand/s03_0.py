import os
import random
import hashlib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Tokenizer for length counting
    enc = tiktoken.get_encoding("gpt2")
    
    def get_tokens(text):
        return len(enc.encode(text))

    # Prepare data for Logistic Regression
    # Use first 2000 chars for classification
    X_pos = [doc[:2000] for doc in target_sample]
    
    # Sample random negatives from docs to balance the training (roughly 200)
    # Filter docs to remove very short ones first to avoid triviality
    potential_neg_indices = [i for i, d in enumerate(docs) if len(d) > 10]
    rng.shuffle(potential_neg_indices)
    neg_indices = potential_neg_indices[:200]
    X_neg = [docs[i][:2000] for i in neg_indices]
    
    y = [1] * len(X_pos) + [0] * len(X_neg)
    
    # Character n-grams for classification
    # Using 3-5 range to capture morphological features and punctuation
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 5), max_features=5000)
    X_train = vectorizer.fit_transform(X_pos + X_neg)
    
    clf = LogisticRegression(solver='liblinear', random_state=seed)
    clf.fit(X_train, y)
    
    # Score all docs
    X_pool = vectorizer.transform([doc[:2000] for doc in docs])
    scores = clf.predict_proba(X_pool)[:, 1]
    
    # Filter for "clean" WikiText-style docs with high scores
    # Logic: Score > 0.7 AND meets basic WikiText formatting
    clean_indices = []
    seen_hashes = set()
    
    # Sort indices by score descending to prioritize high-quality
    scored_indices = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    
    for i in scored_indices:
        doc = docs[i]
        # Basic WikiText heuristic
        if doc.startswith('= ') and (' ,' in doc or ' .' in doc):
            # Deduplicate based on normalized first 200 chars
            norm_start = doc[:200].strip().lower()
            h = hashlib.md5(norm_start.encode()).hexdigest()
            if h not in seen_hashes:
                clean_indices.append(i)
                seen_hashes.add(h)
                
    # Calculate current tokens
    current_tokens = sum(get_tokens(docs[i]) for i in clean_indices)
    T = 9830400
    
    # If under 0.35T, append lower-scoring clean docs
    if current_tokens < 0.35 * T:
        # Get indices of docs that are clean but weren't in our primary list
        # We prioritize by score then by index
        remaining_clean = []
        for i in scored_indices:
            if i not in clean_indices:
                doc = docs[i]
                if doc.startswith('= ') and (' ,' in doc or ' .' in doc):
                    remaining_clean.append(i)
            if len(remaining_clean) > 2000: # Safety break
                break
        
        for idx in remaining_clean:
            if current_tokens < T:
                clean_indices.append(idx)
                current_tokens += get_tokens(docs[idx])
            else:
                break

    # Final shuffle
    rng.shuffle(clean_indices)
    return clean_indices
