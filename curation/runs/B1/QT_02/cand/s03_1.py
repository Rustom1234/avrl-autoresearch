import os
import random
from common import T

try:
    import tiktoken
    encoder = tiktoken.get_encoding("gpt2")
except ImportError:
    raise ImportError("tiktoken is required but not installed.")

# Get random seed from environment or default
SEED = int(os.environ.get("CURATE_SEED", 42))
random.seed(SEED)

def select(docs, target_sample):
    # Step 1: Extract target tokens from target_sample
    # Create a mapping from token ID to string, or just a set of token IDs?
    # To check membership efficiently, we need the token ID or the string representation of the token.
    # Since we are comparing counts, checking if a decoded token string matches any target token string is safe.
    # However, encoding might produce slightly different tokenizations if not identical.
    # Safest approach: Decode all tokens from target_sample to a set of strings.
    
    target_tokens_set = set()
    for target_str in target_sample:
        tokens = encoder.encode(target_str)
        for token_id, token_str in encoder.get_vocab().items():
            # Actually, encoding returns list of (type, string) or just strings?
            # get_encoding returns an object with encode/decode. encode returns list of tokens.
            # tokens[0][0] is type (1 for text, 209 for BOS etc), tokens[0][1] is string.
            pass
        
        # Reformulating:
        tokens = encoder.encode(target_str, allowed_special=False) # exclude special tokens
        current_strs = []
        for tok in tokens:
            # If tokens is a list of tuples (type, string) depending on implementation
            # Actually, encode returns list of tokens.
            # Let's re-verify tiktoken API.
            # encode returns a list of tokens. Each token is a tuple (`type`, `string`).
            # We iterate and extract the string.
            current_strs.append(tok[1])
        target_tokens_set.update(current_strs)

    # But wait, `encoder.encode` returns a list of tuples.
    # Let's just use the string representation of the tokens directly.
    # We need to re-iterate over target_sample carefully.
    
    target_tokens_set.clear()
    for s in target_sample:
        encoded = encoder.encode(s, allowed_special=False)
        for token_data in encoded:
            # token_data structure: (type, string) usually
            if len(token_data) >= 2:
                target_tokens_set.add(token_data[1])
    
    # Handle potential edge case where encoded string is just one token.
    # Actually, `encoded` is a list of tuples.
    
    # Re-implementation of extraction:
    target_tokens_set = set()
    temp_encoder = tiktoken.get_encoding("gpt2")
    for s in target_sample:
        encoded = temp_encoder.encode(s, allowed_special=False)
        for t in encoded:
            # t is (type, string)
            target_tokens_set.add(t[1])

    # Step 2: Analyze the entire docs pool to calculate global frequency (P(target_word))
    total_target_count_global = 0
    total_tokens_global = 0
    doc_stats = [] # List of (index, token_count, target_count, ratio)
    
    # Optimization: If the number of docs is very small but tokens are huge, we must process docs once.
    # We assume `docs` is a list of strings.
    
    for i, doc_text in enumerate(docs):
        tokens = temp_encoder.encode(doc_text, allowed_special=False)
        doc_token_count = len(tokens)
        doc_target_count = 0
        target_strs_in_doc = set()
        
        for t in tokens:
            target_strs_in_doc.add(t[1])
        # Compare against global target set
        doc_target_count = len(target_strs_in_doc & target_tokens_set)
        
        if doc_token_count > 0:
            ratio = doc_target_count / doc_token_count
        else:
            ratio = 1.0 # Avoid division by zero, though documents with 0 tokens are unlikely
            
        # Calculate global ratio
        # We compute global stats incrementally to avoid a second pass
        total_target_count_global += doc_target_count
        total_tokens_global += doc_token_count
        
        doc_stats.append((i, doc_text, doc_token_count, doc_target_count, ratio))

    # If total tokens global is 0, we cannot filter based on ratio (division by zero).
    # Assume the target domain exists in the pool. If not, we might have to default to keeping everything or filtering based on overlap.
    # Assuming valid input, total_tokens_global > 0.
    p_target_global = total_target_count_global / total_tokens_global if total_tokens_global > 0 else 0.0

    # Step 3: Filter documents
    # Filter condition: ratio / global_ratio < 0.01
    if p_target_global <= 0:
        # If no target words found globally, we can't distinguish. 
        # Assume keep all or filter based on a small epsilon? 
        # Safer: keep all if global ratio is 0.
        return list(range(len(docs)))

    filtered_indices = []
    # Sort by ratio descending to prioritize "pure" target docs, then by index for determinism if ratios are equal?
    # But the rule says "Prioritize the top 200 target domain documents deterministically (sorted by index)".
    # This implies we should collect candidates, and if we have >= 200, pick the first 200 by index.
    # So finding all "good" documents is necessary.
    
    # Let's filter out bad ones first.
    for idx_ptr, i_val, _, _, r in doc_stats:
        if p_target_global > 0:
            if (r / p_target_global) < 0.01:
                continue
        # Include in candidates
        # Store index
        filtered_indices.append(i_val)

    if not filtered_indices:
        # No documents passed filter? This is a degenerate case.
        # Fallback: return empty or indices 0..0?
        # Return empty list to stop training? Or return first doc?
        # Given the prompt "stops after T tokens", returning fewer than needed just cycles.
        # If we return nothing, it cycles through 0 docs.
        # Let's return a simple fallback.
        return [0]

    # Step 4: Sort filtered indices by their original index to satisfy "deterministically sorted by index" requirement for selection.
    # The list `filtered_indices` is naturally sorted if we appended in order, but we iterated `doc_stats` which was built in order.
    # So `filtered_indices` is already sorted by index.
    
    # Step 5: Select top 200 target documents.
    # We take the first 200 indices from `filtered_indices`.
    top_200_indices = filtered_indices[:200]
    
    # Step 6: Determine remaining tokens needed to reach exactly T tokens?
    # Or simply fill the quota T with the top 200 + shuffled rest?
    # The rule says: "stop after T tokens".
    # So we construct the sequence:
    # 1. Expand top_200_indices to cover at least some T tokens.
    #    If we need more tokens, we take the next documents from the shuffled pool.
    
    # Total tokens provided by top 200
    tokens_needed = 0
    selected_indices = top_200_indices.copy()
    
    # We need to calculate how many tokens we have from top 200.
    # We might need to append more from the shuffled non-top-200 pool.
    
    # Identify the pool for shuffling (indices in filtered_indices that are >= 200th index, i.e., indices from 200 to end in the sorted list).
    # Actually, we just need the indices starting from index 200 upwards in the `filtered_indices` list (since we took 0..199).
    # But we must ensure we only draw from documents that were NOT selected in top 200?
    # Yes, "fill the remaining tokens with a seeded random shuffle of the rest of the pool".
    # So we exclude the top 200 indices.
    
    # Rest of the pool
    remaining_indices = filtered_indices[200:]
    
    # If we still need tokens after expanding the initial 200 docs (or if we decide they are enough), we need to refill.
    # Actually, the instruction says "Prioritize the top 200... Fill the rest with shuffle".
    # This implies the order must be: [Top 200 docs] + [Shuffled Rest].
    # AND we stop when we hit T tokens.
    
    # Shuffle `remaining_indices`
    random.shuffle(remaining_indices)
    
    # Construct final list
    final_list = top_200_indices + remaining_indices
    
    # If we have fewer than T tokens in the result, we cycle.
    # Just return the list.
    return final_list
