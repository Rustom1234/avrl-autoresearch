"""Default curator ("weighted"): scores pool documents based on IDF of target_sample.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
import collections
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use gpt2 encoding as specified
    enc = tiktoken.encoding.get_encoding("gpt2")
    
    # 1. Tokenize the target sample and build a frequency map
    target_tokens = []
    for doc in target_sample:
        target_tokens.extend(enc.encode(doc))
    
    target_counts = collections.Counter(target_tokens)
    num_target_tokens = len(target_tokens)
    
    # 2. Calculate IDF for each token in the target sample
    # IDF(t) = log(Total_Target_Tokens / count(t_in_target))
    # We use a small epsilon to avoid division by zero, though Counter handles existence.
    # Using a simple weight: weight(t) = 1 / count(t_in_target) is also a common proxy.
    # The prompt asks for "sum of the 'inverse document frequency'", we will use 
    # a standard approach where higher frequency in target_sample *decreases* IDF, 
    # but the prompt says "higher frequency... increases the score". 
    # To follow the prompt's specific instruction: Score = Sum(count(t) in target_sample)
    # Wait, "sum of the IDF ... where higher frequency ... increases the score" 
    # is mathematically contradictory to standard IDF. 
    # To satisfy the prompt's requirement: Score = Sum(weight of token t in doc)
    # where weight(t) is proportional to its frequency in the target sample.
    
    # Let's re-read: "calculate a score based on the sum of the 'inverse document frequency' 
    # of its tokens, where the IDF is derived from the target_sample (higher frequency in the 
    # target_sample increases the score)."
    # This implies: Score = Sum over tokens in doc of (Frequency of token in target_sample).
    # If it strictly meant standard IDF, higher frequency would decrease the score.
    # We will follow the explicit instruction: Higher target frequency = Higher score.
    
    # Pre-calculate weights
    weights = {token: count for token, count in target_counts.items()}
    
    # 3. Score the documents in the pool
    doc_scores = []
    for i, doc in enumerate(docs):
        tokens = enc.encode(doc)
        score = sum(weights.get(t, 0) for t in tokens)
        doc_scores.append((score, i))
    
    # 4. Sort by score descending
    # doc_scores is a list of (score, index)
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Extract indices
    sorted_indices = [item[1] for item in doc_scores]
    
    # The trainer cycles the list, so we just need to provide the sorted list.
    # Since the trainer stops at T, we can just return the sorted indices.
    return sorted_indices
