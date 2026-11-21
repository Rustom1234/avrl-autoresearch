import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    """
    Curates documents by prioritizing those with the highest token overlap 
    (Jaccard similarity) with the target_sample.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    encoding = tiktoken.get_encoding("gpt2")
    
    # Tokenize target sample and create a global target set
    target_tokens = set()
    for doc in target_sample:
        target_tokens.update(encoding.encode(doc))
    
    # Score documents in the pool
    doc_scores = []
    for i, doc in enumerate(docs):
        doc_tokens = set(encoding.encode(doc))
        # Calculate Jaccard Similarity
        intersection = len(doc_tokens.intersection(target_tokens))
        union = len(doc_tokens.union(target_tokens))
        
        score = intersection / union if union > 0 else 0
        doc_scores.append((score, i))
    
    # Sort by score descending, then by original index to keep it stable
    doc_scores.sort(key=lambda x: (-x[0], x[1]))
    
    # To maximize the utility of the 300 steps, we want to prioritize 
    # high-quality docs but keep enough variety. 
    # Because we need to fill T tokens, we take the best docs.
    # However, since we can repeat indices, we can just return the best ones.
    # To ensure we don't just repeat one document, we can use a weighted 
    # selection or simply provide the best docs in descending order.
    
    # If the pool is large, we pick the top indices that represent the best content.
    # Given the constraints, a simple descending list of the best documents 
    # is the most effective way to increase the model's exposure to target domain.
    best_indices = [idx for score, idx in doc_scores]
    
    # Optional: Shuffle slightly to avoid overfitting to a specific sequence 
    # of high-score docs, but keep them at the front.
    # For a small model/short training, the best approach is the purest content.
    return best_indices
