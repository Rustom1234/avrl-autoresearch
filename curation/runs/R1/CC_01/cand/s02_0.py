import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    """
    Curates the training set by prioritizing documents with higher average 
    word lengths to filter out junk and favor content-rich text.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use gpt2 encoding as specified
    try:
        encoder = tiktoken.get_encoding("gpt2")
    except:
        # Fallback to a basic encoding if tiktoken fails for any reason
        encoder = None

    doc_scores = []
    for i, doc in enumerate(docs):
        # Basic junk filter: ignore documents that are too short
        if len(doc.split()) < 5:
            doc_scores.append((-1.0, i))
            continue
            
        # Calculate average word length
        words = doc.split()
        if not words:
            doc_scores.append((-1.0, i))
            continue
            
        avg_word_len = sum(len(w) for w in words) / len(words)
        doc_scores.append((avg_word_len, i))

    # Sort by average word length descending
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Extract indices of high-quality documents
    # We take a large enough pool to ensure we have many variations, 
    # but prioritize the top ones.
    best_indices = [item[1] for item in doc_scores]
    
    # To ensure we have enough variety and use the highest quality first,
    # we shuffle the best indices slightly or just return the sorted list.
    # Since the model stops at T tokens and cycles, a sorted high-quality 
    # list is often better for "front-loading" quality.
    
    # To satisfy the goal of "selecting top N", let's take the top 20%
    # of the pool as primary, then the rest.
    top_n = max(1, len(docs) // 5)
    primary = best_indices[:top_n]
    secondary = best_indices[top_n:]
    
    # Combine to ensure we provide a large pool if needed, but prioritize primary
    # However, the prompt asks to prioritize content-rich texts.
    # A sorted list of the best scores is the most direct implementation.
    
    # Final Selection: Just the sorted best indices to maximize quality density 
    # before the T token limit is hit.
    return best_indices
