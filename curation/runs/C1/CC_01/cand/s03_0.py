import os
import random
import tiktoken
import numpy as np

def select(docs, target_sample):
    """
    Selects a subset of documents from the pool based on Jaccard similarity
    to the target sample and then filters by token length.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    encoding = tiktoken.get_encoding("gpt2")
    
    # Pre-process target_sample into sets of words
    target_word_sets = [set(d.split()) for d in target_sample]
    
    # Calculate Jaccard similarity for each document in docs against the target_sample
    # Score = Average Jaccard similarity to all documents in target_sample
    scores = []
    for d in docs:
        doc_words = set(d.split())
        sim_sum = 0
        for t_set in target_word_sets:
            intersection = len(doc_words.intersection(t_set))
            union = len(doc_words.union(t_set))
            sim_sum += (intersection / union) if union > 0 else 0
        scores.append(sim_sum / len(target_word_sets))
    
    # Identify indices of top 1000 documents by Jaccard similarity
    # Use stable sort to maintain order for tie-breaking
    indexed_scores = []
    for i, score in enumerate(scores):
        indexed_scores.append((score, i))
    
    # Sort descending by score
    indexed_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 1000 indices
    top_1000_indices = [idx for score, idx in indexed_scores[:1000]]
    
    # Filter these top indices by token length similarity to the target_sample
    pool_lens = []
    for i in top_1000_indices:
        pool_lens.append(len(encoding.encode(docs[i])))
        
    target_lens = []
    for d in target_sample:
        target_lens.append(len(encoding.encode(d)))
    avg_target_len = sum(target_lens) / len(target_lens)
    
    # Calculate absolute differences for the top 1000
    # We map back to the original index
    diffs = []
    for i, doc_idx in enumerate(top_1000_indices):
        diffs.append((abs(pool_lens[i] - avg_target_len), doc_idx))
    
    # Sort by difference (ascending)
    diffs.sort(key=lambda x: x[0])
    
    # Take a significant portion of the best matches from the top 1000
    num_to_pick = max(100, len(docs) // 4)
    num_to_pick = min(num_to_pick, len(diffs))
    selected_indices = [idx for diff, idx in diffs[:num_to_pick]]
    
    # Shuffle the selected indices
    rng.shuffle(selected_indices)
    
    return selected_indices
