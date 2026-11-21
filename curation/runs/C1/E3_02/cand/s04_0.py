import os
import random
import re
from collections import Counter
import numpy as np

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Tokenize and identify target vocabulary
    target_words = set()
    for doc in target_sample:
        words = re.findall(r'\w{4,}', doc.lower())
        target_words.update(words)
    
    # 2. Pre-process all docs for scores and global word frequencies
    # We use a global frequency count to simulate the "Inverse Document Frequency" 
    # component of the TF-IDF-inspired score.
    global_word_counts = Counter()
    doc_data = []
    
    # Spam triggers
    spam_triggers = {'click here', 'order now', 'buy', 'subscribe', 'free gift', 'winner'}
    
    for i, doc in enumerate(docs):
        words = re.findall(r'\w+', doc.lower())
        if not words:
            continue
            
        # Count global frequencies
        global_word_counts.update(words)
        
        # Filter logic
        if len(words) < 10:
            continue
            
        # Spam penalty
        spam_penalty = 0
        doc_lower = doc.lower()
        for trigger in spam_triggers:
            if trigger in doc_lower:
                spam_penalty += 5
        
        # Similarity Score: Count overlap with target_words
        overlap_count = sum(1 for w in words if w in target_words)
        
        # TF-IDF inspired weighting
        # Weight = overlap_count * (Average Frequency of target words in docs)^-1
        # To simplify for this constraint, we use the inverse of the global frequency of the words in the doc
        # specifically for those that overlap with the target.
        tfidf_score = 0
        for w in words:
            if w in target_words:
                # Inverse frequency: 1 / (count + 1) to avoid division by zero
                tfidf_score += (1.0 / (global_word_counts[w] + 1))
        
        # Final score: (TF-IDF Score / sqrt(len)) - Spam Penalty
        # sqrt(len) normalizes for document length
        final_score = (tfidf_score / (len(words) ** 0.5)) - spam_penalty
        
        doc_data.append((final_score, i))

    # Sort by score descending
    doc_data.sort(key=lambda x: x[0], reverse=True)
    
    # 3. Selection
    # Take the top half of the best documents
    best_indices = [idx for score, idx in doc_data[:len(docs)//2]]
    
    # Shuffle to ensure the model doesn't see the same 10 documents in a row 
    # too frequently before cycling
    rng.shuffle(best_indices)
    
    return best_indices
