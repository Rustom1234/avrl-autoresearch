import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    def count_tokens(text):
        return len(enc.encode(text))

    # Simplified preprocessing: word count as a proxy for complexity
    # and a simple rolling mean/variance of word lengths to capture "style"
    def get_features(text):
        words = text.split()
        if not words: return torch.tensor([0.0, 0.0, 0.0])
        lengths = [len(w) for w in words]
        return torch.tensor([
            float(len(words)),
            float(sum(lengths) / len(lengths)),
            float(np.std(lengths) if len(lengths) > 1 else 0.0)
        ])

    # Prepare Training Data (Small 2-layer MLP)
    # We sample a subset of the pool to build a "hard negative" space
    # Hard negatives = docs that might look like target sample but are junk
    pool_indices = list(range(len(docs)))
    sample_indices = random.sample(pool_indices, min(len(docs), 1000))
    
    pos_feats = torch.stack([get_features(d) for d in target_sample])
    
    # Extract "negatives" from the pool by looking for high entropy/noise
    neg_feats = []
    neg_indices = []
    for i in sample_indices:
        feats = get_features(docs[i])
        # Use a heuristic for "junk": very short or extremely long/repetitive
        if len(docs[i].split()) < 10 or len(docs[i].split()) > 500:
            neg_feats.append(feats)
            neg_indices.append(i)
            
    # If no junk found, just take random samples as negatives
    if not neg_indices:
        neg_indices = random.sample(pool_indices, min(len(pool_indices), 500))
        neg_feats = torch.stack([get_features(docs[i]) for i in neg_indices])
    else:
        neg_feats = torch.stack(neg_feats)

    # Model: 2-layer MLP (Input=3, Hidden=16, Output=1)
    model = nn.Sequential(
        nn.Linear(3, 16),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(16, 1)
    )
    
    # Simple training (no optimizer needed for 3 features/100 samples)
    # We just want to find the weights that separate target_sample from negatives
    pos_labels = torch.ones((pos_feats.shape[0], 1))
    neg_labels = torch.zeros((neg_feats.shape[0], 1))
    
    # Manual weight update to avoid overhead
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(50):
        optimizer.zero_grad()
        p_out = model(pos_feats)
        n_out = model(neg_feats)
        loss = F.binary_cross_entropy(p_out, pos_labels) + F.binary_cross_entropy(n_out, neg_labels)
        loss.backward()
        optimizer.step()

    # Scoring Phase
    model.eval()
    all_feats = torch.stack([get_features(d) for d in docs])
    with torch.no_grad():
        scores = torch.sigmoid(model(all_feats)).squeeze().numpy()
    
    # "Cathedral-like" distance to centers
    # Calculate centroids in the latent space (the hidden layer output)
    def get_latent(feats):
        with torch.no_grad():
            x = model[0](feats)
            x = F.relu(x)
            return x
    
    pos_latent = get_latent(pos_feats).mean(dim=0)
    # Find highest scoring doc in pool as a "negative" anchor
    best_neg_idx = np.argmax(scores)
    neg_latent = get_latent(all_feats[best_neg_idx]).mean(dim=0)
    
    # Distance logic: high score + proximity to positive center
    pool_latents = get_latent(all_feats)
    dist_to_pos = torch.norm(pool_latents - pos_latent, dim=1)
    
    # Final Score = Classification Score - Normalized Distance to Positives
    final_scores = scores - (dist_to_pos / (dist_to_pos.max() + 1e-6))
    
    # Selection
    ranked_indices = np.argsort(final_scores)[::-1]
    selected_indices = []
    current_tokens = 0
    
    # Buffer to ensure we get enough content
    for idx in ranked_indices:
        count = count_tokens(docs[idx])
        if count > 0:
            if current_tokens + count <= T:
                selected_indices.append(idx)
                current_tokens += count
            else:
                # Break and fill with remaining if needed
                break
                
    if current_tokens < T:
        # Fill with highest scoring remaining to reach T
        for idx in ranked_indices:
            if idx in selected_indices: continue
            count = count_tokens(docs[idx])
            if current_tokens + count <= T:
                selected_indices.append(idx)
                current_tokens += count
            else:
                break
                
    # Cycle if still under T
    while current_tokens < T and selected_indices:
        for idx in selected_indices:
            count = count_tokens(docs[idx])
            if current_tokens + count <= T:
                selected_indices.append(idx)
                current_tokens += count
            if current_tokens >= T: break
        if current_tokens < T and not selected_indices: break

    # Final shuffle to prevent training on clusters
    random.shuffle(selected_indices)
    return selected_indices
