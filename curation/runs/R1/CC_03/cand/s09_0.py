import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    """
    Selects documents from the pool based on their 'purity' relative to the 
    vocabulary used in the target_sample.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initialize tokenizer
    try:
        encoder = tiktoken.get_encoding("gpt2")
    except:
        # Fallback for environments where gpt2 might not be directly accessible
        # but tiktoken is installed.
        encoder = tiktoken.encoding_for_model("gpt2")

    # Build a set of tokens present in the target sample to define the "high-quality" vocabulary
    target_tokens = set()
    for doc in target_sample:
        target_tokens.update(encoder.encode(doc))
    
    # Calculate purity scores for all documents in the pool
    # Score = (count of tokens in target_tokens) / (total tokens in document)
    doc_scores = []
    for i, doc in enumerate(docs):
        tokens = encoder.encode(doc)
        if not tokens:
            score = 0.0
        else:
            match_count = sum(1 for t in tokens if t in target_tokens)
            score = match_count / len(tokens)
        doc_scores.append((score, i))
    
    # Sort by score descending. We use the index as a tie-breaker for stability.
    doc_scores.sort(key=lambda x: (-x[0], x[1]))
    
    # Extract indices
    sorted_indices = [item[1] for item in doc_scores]
    
    # The trainer reads the list and cycles it. Since we want to maximize
    # the tokens seen before T is reached, we return the sorted indices.
    # To ensure we provide enough tokens to satisfy T, the list is cycled by the trainer.
    return sorted_indices
