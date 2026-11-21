import os
import random
import re
import numpy as np
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use gpt2 as a proxy for a high-quality pre-trained tokenizer
    # In a real scenario, this would be the domain-specific tiktoken model
    enc = tiktoken.get_encoding("gpt2")
    
    def count_tokens(text):
        return len(enc.encode(text))

    # Calculate target embeddings (simulated via high-dimensional semantic frequency)
    # Since we cannot actually load a heavy LLM for embeddings in this environment,
    # we use a sophisticated feature-based projection that mimics dot-product 
    # similarity in a dense space.
    def get_features(text):
        # Normalize and extract word-based features
        tokens = re.findall(r'\w+', text.lower())
        # Use a hash-based projection to simulate a fixed-dimension embedding
        # This captures local semantic structure and global frequency
        features = np.zeros(128)
        for i, t in enumerate(tokens):
            # Hash of the word mapped to the embedding dimension
            idx = hash(t) % 128
            features[idx] += 1.0
        return features / (np.linalg.norm(features) + 1e-9)

    # Compute mean "centroid" of the target sample
    target_embeddings = [get_features(s) for s in target_sample]
    target_centroid = np.mean(target_embeddings, axis=0)
    
    # Filter and score the pool
    scored_docs = []
    for i, doc in enumerate(docs):
        # Structural filter: Reward documents with header-like markers (Wiki/Markdown)
        structural_score = 1.0
        if not re.search(r'^[A-Z0-9\s]+$|\n[A-Z0-9\s]+', doc):
            structural_score = 0.7
            
        # Semantic Similarity Projection
        doc_emb = get_features(doc)
        similarity = np.dot(doc_emb, target_centroid)
        
        # Combined score
        final_score = similarity * structural_score
        scored_docs.append((i, final_score))
    
    # Sort by similarity score descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Deduplication and Selection
    selected_indices = []
    seen_hashes = set()
    current_tokens = 0
    
    # First pass: Select high-quality unique documents
    for idx, score in scored_docs:
        doc = docs[idx]
        # Basic content hashing to deduplicate
        h = hash(doc[:500]) 
        if h not in seen_hashes:
            count = count_tokens(doc)
            if current_tokens + count <= T:
                selected_indices.append(idx)
                seen_hashes.add(h)
                current_tokens += count
    
    # Second pass: Fill the remaining budget with the best available docs (repeats allowed)
    # Since we need to hit T, we cycle through the top-scoring docs
    best_indices = [item[0] for item in scored_docs[:100]]
    if current_tokens < T and not best_indices:
        best_indices = [item[0] for item in scored_docs]

    idx_ptr = 0
    while current_tokens < T and idx_ptr < len(best_indices):
        idx = best_indices[idx_ptr]
        count = count_tokens(docs[idx])
        if current_tokens + count <= T:
            selected_indices.append(idx)
            current_tokens += count
        idx_ptr += 1
        
    # Final fill for the remaining small gap
    while current_tokens < T:
        idx = best_indices[idx_ptr % len(best_indices)]
        count = count_tokens(docs[idx])
        if current_tokens + count <= T:
            selected_indices.append(idx)
            current_tokens += count
        else:
            break

    # Shuffle results to improve training distribution
    rng.shuffle(selected_indices)
    return selected_indices
