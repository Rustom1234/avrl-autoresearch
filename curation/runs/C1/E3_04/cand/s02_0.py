import os
import random
import tiktoken

def select(docs, target_sample):
    """
    Selects documents from docs based on their Jaccard similarity to the 
    aggregate token set of the target_sample.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initialize the encoder for "gpt2"
    enc = tiktoken.get_encoding("gpt2")
    
    # Create the aggregate set of tokens from the target_sample
    target_tokens = set()
    for doc in target_sample:
        tokens = enc.encode(doc)
        target_tokens.update(tokens)
        
    # Calculate similarity scores for each document in docs
    scores = []
    for i, doc in enumerate(docs):
        doc_tokens = set(enc.encode(doc))
        
        # Jaccard Similarity = (A ∩ B) / (A ∪ B)
        intersection = len(doc_tokens.intersection(target_tokens))
        union = len(doc_tokens.union(target_tokens))
        
        similarity = intersection / union if union > 0 else 0.0
        scores.append((similarity, i))
    
    # Sort indices by similarity descending
    # We use a stable sort and then secondary sort by original index if needed, 
    # but a simple sort on the tuple works.
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Return the list of indices
    indices = [item[1] for item in scores]
    
    # Ensure we have enough to cover T tokens (though the loop in the 
    # trainer handles cycling, we provide the sorted list).
    return indices
