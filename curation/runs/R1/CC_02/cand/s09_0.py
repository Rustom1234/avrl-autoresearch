import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    """
    Selects indices from docs based on a composite score of high-value 
    token density and uniqueness.
    """
    # Set up the encoder
    enc = tiktoken.get_encoding("gpt2")
    seed = int(os.environ.get("CURATE_SEED", 42))
    random.seed(seed)
    
    # Identify high-value tokens from the target_sample
    high_value_tokens = set()
    for doc in target_sample:
        tokens = enc.encode(doc)
        for t in tokens:
            high_value_tokens.add(t)
            
    scores = []
    for i, doc in enumerate(docs):
        doc_tokens = enc.encode(doc)
        if not doc_tokens:
            scores.append((0.0, i))
            continue
        
        # Count of high-value tokens in the document
        shared_count = sum(1 for t in doc_tokens if t in high_value_tokens)
        
        # Density = shared_count / total_tokens
        density = shared_count / len(doc_tokens)
        
        # Uniqueness = shared_count / unique_tokens_in_doc
        unique_tokens = set(doc_tokens)
        uniqueness = shared_count / len(unique_tokens)
        
        # Composite score calculation
        # Score = (shared_count * 0.6 + density * 0.4)
        # The prompt asks to use 'uniqueness' score logic but specific composite formula:
        # Score = (shared_count * 0.6 + density * 0.4)
        # However, it mentions "Calculate a 'uniqueness' score ... and use a composite score"
        # To satisfy both, we use the specific weighted formula provided:
        score = (shared_count * 0.6 + density * 0.4)
        
        # We also need to consider the uniqueness if the prompt implies it's part of the score,
        # but the formula provided is explicit. Let's stick to the explicit formula.
        # If the prompt meant (uniqueness * 0.6 + density * 0.4), it would have said so.
        # Actually, looking at the prompt, it says "use a composite score of...".
        # Let's include uniqueness in a way that respects the text provided.
        
        # Re-reading: "Calculate a 'uniqueness' score ... and use a composite score of (shared_count * 0.6 + density * 0.4)"
        # This implies shared_count is used in the final score.
        
        scores.append((score, i))
        
    # Sort by score descending, then by index to keep it stable
    scores.sort(key=lambda x: (-x[0], x[1]))
    
    # Extract the top 2500 indices
    top_indices = [item[1] for item in scores[:2500]]
    
    return top_indices
