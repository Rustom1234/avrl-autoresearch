import os
import random
import collections
import numpy as np
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Whitelist of common English words and general content terms
    # Fixed: Used a list instead of a set constructor to avoid TypeError
    words_list = ["the", "be", "to", "of", "and", "a", "in", "that", "is", "it", "you", 
                   "he", "was", "for", "on", "are", "with", "as", "i", "his", "they", 
                   "at", "be", "this", "have", "from", "or", "by", "an", "sh", "which", 
                   "one", "there", "your", "would", "do", "their", "said", "but", "not", 
                   "what", "all", "were", "we", "when", "could", "make", "like", "some", 
                   "can", "up", "one", "out", "other", "about", "into", "just", "time", 
                   "day", "way", "take", "only", "good", "get", "come", "these", "over", 
                   "know", "work", "first", "water", "before", "look", "use", "told", "find", 
                   "different", "many", "most", "eye", "know", "take", "think", "say", "help", 
                   "talk", "turn", "start", "still", "life", "give", "many", "back", "take",
                   "right", "move", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    
    content_terms = ["data", "info", "system", "process", "theory", "method", "result", 
                      "study", "history", "culture", "science", "object", "place", "event"]
    
    whitelist = set(words_list + content_terms)
    
    sorted_whitelist = sorted(list(whitelist))
    wh_to_idx = {word: i for i, word in enumerate(sorted_whitelist)}
    dim = len(sorted_whitelist)

    # Pre-fetch encoder
    encoder = tiktoken.get_encoding("gpt2")

    def get_word_vector(text):
        # Clean text and split into words
        words = text.lower().replace('"', '').replace("'", "").split()
        counts = collections.Counter(words)
        
        vec = np.zeros(dim)
        for word, count in counts.items():
            if word in wh_to_idx:
                vec[wh_to_idx[word]] += count
        return vec

    # Build profile from target sample
    target_vectors = [get_word_vector(doc) for doc in target_sample]
    if not target_vectors:
        return list(range(len(docs)))
        
    avg_vec = np.mean(target_vectors, axis=0)
    
    # Score docs in pool
    scored_indices = []
    for i, doc in enumerate(docs):
        v = get_word_vector(doc)
        
        dot = np.dot(v, avg_vec)
        mag_v = np.linalg.norm(v)
        mag_avg = np.linalg.norm(avg_vec)
        
        # Handle zero vectors by adding epsilon
        similarity = dot / (mag_v * mag_avg + 1e-9)
        scored_indices.append((similarity, i))
    
    # Sort by similarity descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Keep top 50% for variety, but prioritize similarity
    top_count = max(1, len(docs) // 2)
    top_indices = [idx for score, idx in scored_indices[:top_count]]
    
    # Shuffle top indices to ensure variety in the sequence
    rng.shuffle(top_indices)
    
    # Ensure we return enough to cover T if possible, but the problem 
    # says it cycles, so top_indices is sufficient.
    return top_indices
