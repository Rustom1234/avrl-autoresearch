"""
Corrected curator: injects low-overhead n-gram statistics from target_sample into the docs order.
This function computes a relevance weight for each document based on character distribution
similarity to the target_sample. To ensure robustness and avoid infinite loops, we iterate
until we exceed the token budget T. We handle edge cases where weights might be zero or NaN
by capping them.
"""
import os
import random
from collections import Counter
from common import T
import numpy as np

# Set random seed from environment variable
seed_from_env = os.getenv("CURATE_SEED")
if seed_from_env:
    random.seed(int(seed_from_env))
    np.random.seed(int(seed_from_env))
else:
    random.seed(42)
    np.random.seed(42)

def select(docs, target_sample):
    # 1. Analyze target_sample to build a character frequency map.
    char_dist_source = Counter()
    for doc in target_sample:
        char_dist_source.update(doc)
    
    # Calculate normalization factor for target sample char counts to get relative probabilities
    total_chars_source = sum(char_dist_source.values())
    if total_chars_source == 0:
        total_chars_source = 1
    char_probs_source = {char: count / total_chars_source for char, count in char_dist_source.items()}

    # 2. Analyze the noisy pool (docs) and assign a relevance weight to each.
    # To avoid O(N*AvgLen) overhead if docs are massive, we compute a "signature" weight.
    # Weight = Sum of log probabilities of characters present in the document.
    # We use log probabilities to avoid high counts dominating too much, but since target char 
    # is rare in docs usually, we treat presence of target chars as positive weight.
    
    # Optimization: Precompute weights for characters that matter.
    # However, simply iterating is O(Total Tokens) which is acceptable since we only pass through docs once 
    # to generate the candidate list. If that cycle doesn't yield T tokens, we restart.
    # Note: The problem says docs is a list of strings mixing useful text with junk. 
    # We compute a score for each doc based on how "target-like" its character density is.
    
    scores = []
    for doc in docs:
        # Estimate relevance: sum of char probabilities from target_sample found in this doc
        score = 0.0
        for char in doc:
            if char in char_probs_source:
                score += char_probs_source[char]
        # Add a small uniform noise to break ties slightly if scores are identical
        # p=0 is not allowed in numpy choice, so we cap min at small epsilon
        score = max(score, 0.01) 
        scores.append(score)

    # 3. Convert scores to probabilities for weighted sampling.
    if len(scores) == 0:
        return []
    
    weights = np.array(scores)
    total_weight = weights.sum()
    
    if total_weight > 0:
        probs = weights / total_weight
    else:
        # Fallback if all scores were handled by max(0.01) logic above, so total should not be 0
        # Just in case, uniform distribution
        probs = np.ones(len(scores)) / len(scores)

    # Check for NaNs or infs
    if np.isnan(probs).any() or np.isinf(probs).any():
        probs = np.ones(len(scores)) / len(scores)

    # 4. Generate indices until we cover at least T tokens or until max iterations.
    # To avoid TLE, we check if we are getting T tokens efficiently.
    # However, the problem says docs is a pool of about 2T tokens.
    # Assuming average doc length is not 1 token.
    # We just need to generate a sequence of indices.
    
    generated_indices = []
    current_tokens_counted = 0
    max_iterations = 10000 # Safety break to prevent infinite loop in case of logic error
    
    while current_tokens_counted < T and len(generated_indices) < max_iterations:
        # Sample one document index based on probability
        idx = np.random.choice(len(docs), p=probs)
        
        # Check if this doc has enough tokens to cover overflow (just to minimize unnecessary samples)
        # We don't know exact token count without tokenization, so we assume 1 token per index if we haven't tried? 
        # No, standard practice: just append index. The training loop handles tokenization and counting.
        generated_indices.append(idx)
        
        # We are simulating the training process in terms of token consumption.
        # Since we can't count tokens inside select safely without tokenizing the whole doc list (slow)
        # AND we are selecting indices, we assume if we haven't hit T after max_iterations, fail gracefully
        # OR re-evaluate that the docs list is huge.
        # Let's assume the training loop counts tokens. If it needs > T tokens, it will loop the whole docs again if needed.
        # But we want to stop once T >= T.
        # Since we can't count tokens inside select efficiently for all docs, we rely on the loop limit or T
        # If we generate max indices and T is not reached, we stop. The trainer loops the list.
        generated_indices = generated_indices[:max_iterations] 
        if len(generated_indices) < max_iterations:
            # We need more indices to cover T. Keep retrying.
            pass
            
        # Debugging check
        if len(generated_indices) > max_iterations:
            break
            
        # We assume that if T is reached after a few passes over docs (which is typical),
        # we stop. But since we can't count tokens precisely here, the logic is to just return 
        # enough indices so that the training loop stops. 
        # To be safe and avoid TLE, we assume T is the hard limit. 
        # We might need to loop the generation process until we are confident T is reached?
        # Actually, the problem says "Training reads the documents in that order and stops after T tokens".
        # So if we return a list of indices, training iterates. If total length of selected indices > T, stop.
        # We just need to return enough indices to cover T tokens. 
        # To estimate tokens per index, we could sum lengths of docs? No, word boundaries unknown.
        # But docs strings are given. We could tokenize them once.
        # Let's optimize: Pre-tokenize lengths of all docs to estimate tokens per selection.
        # This is O(2T) which is fine.
    
    # OPTIMIZATION: Precompute estimated token counts per doc to ensure we stop exactly or close to T.
    # Since tiktoken is available, let's use it.
    if not generated_indices:
        return []
        
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    
    # Calculate token length for each document to avoid over-shooting T too much or wasting time
    token_lengths = []
    for doc in docs:
        tokens = enc.encode(doc)
        token_lengths.append(len(tokens))
    
    # Now we have estimated tokens per selected index. We can bail out more intelligently.
    # We generated 'generated_indices' based on avg length assumption before.
    # We should restart the generation loop with the correct stopping logic.
    
    generated_indices = []
    total_tokens_so_far = 0
    attempts = 0
    
    while total_tokens_so_far < T and (len(generated_indices) // len(docs)) < 500*len(docs) and attempts < 1000:
        idx = np.random.choice(len(docs), p=probs)
        token_len = token_lengths[idx]
        generated_indices.append(idx)
        
        # Accumulate estimated tokens
        total_tokens_so_far = total_tokens_so_far + token_len
        
        if total_tokens_so_far >= T:
            # Add remainder to not under-fill, but typically T is exact.
            # Remove excess if we overshoot significantly? 
            # Usually, it's fine to be slightly over. Or cut off the last partial doc.
            # We will return full list, trainer will take min.
            pass
        
        attempts += 1
        if total_tokens_so_far > T:
            break
    
    if len(generated_indices) > T: # crude check depending on doc length
        # Remove the tail if we exceeded T significantly
        idx_to_remove = len(generated_indices) - (total_tokens_so_far - T)
        # We need to remove indices, not tokens. 
        # This is tricky because min(len_idx) >= token_length[idx].
        # Simple approach: truncate the list to roughly T.
        # Assuming avg token doc length is approx 100-200. 
        # So we remove // 50 of the list? Not accurate.
        # Let's just return the most recent T-token approx.
        # Actually, most robust solution: just return the list, training loop stops exactly at T.
        # Overhead is minimal.
        pass

    return generated_indices
