import os
import random
import re
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    junk_keywords = [
        'discount code', 'buy now', 'save 70%', 'click here'
    ]
    
    def get_tokens(text):
        # Simple tokenizer to mimic gpt2-like tokenization behavior for similarity
        return set(re.findall(r'\w+', text.lower()))

    # Pre-process target sample to get collective token set
    target_tokens = set()
    for doc in target_sample:
        target_tokens.update(get_tokens(doc))
    
    # Filter junk and calculate similarities
    candidates = []
    for i, doc in enumerate(docs):
        # Check for junk keywords
        doc_lower = doc.lower()
        is_junk = any(kw in doc_lower for kw in junk_keywords)
        
        if not is_junk:
            doc_tokens = get_tokens(doc)
            if not doc_tokens:
                continue
            
            intersection = len(doc_tokens.intersection(target_tokens))
            union = len(doc_tokens.union(target_tokens))
            similarity = intersection / union if union > 0 else 0
            candidates.append((similarity, i))
    
    # Sort by similarity descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 2000 indices
    top_indices = [idx for score, idx in candidates[:2000]]
    
    # If we have fewer than 2000, just return what we have
    # Otherwise, we'll cycle them as required by the prompt
    # (The prompt says "return the top 2000 indices in a cycle")
    # If the prompt implies we only return 2000 if available:
    # We need to ensure we have at least some content.
    
    # To ensure we satisfy the "cycle" requirement for T tokens:
    # The trainer cycles the returned list. We return the top 2000.
    return top_indices
