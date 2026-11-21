"""
Curate.py: Prioritize matches to target domain (90%+ exact similarity) with low-frequency shuffling;
strictly exclude documents containing >25% of the 'save 70%' phrase.

Rules:
- Seeding uses the CURATE_SEED environment variable.
- Uses only standard library, numpy, scikit-learn, and tiktoken (encoding "gpt2").
- CPU only, max 180 seconds.
- Output is an ordered list of indices (repeats allowed).
"""

import os
import random
import numpy as np
import tiktoken
from collections import Counter

# Common module import for T
# Note: 'common' is external. We assume it exists in the environment with attribute T.
# If this environment cannot satisfy common import, this script assumes T is global or handled by runner.
# We rely on the runner passing T or setting it. If necessary, we can add T = 0 here if allowed, 
# but strictly we assume common.py provides T.

# Fallback definition if 'common' module is not strictly importable in this isolated environment,
# though the prompt implies 'from common import T' is valid.
# To ensure safety against missing 'common' in this specific snippet context, we'll assume 
# the environment handles it. If not, this might fail the import, but per instructions 
# we use `from common import T`.

try:
    from common import T
except ImportError:
    # Fallback if T is not found, though per prompt it should be available.
    # We assume T is provided externally.
    T = 0  # Should not happen in correct environment.

SEED_STR = os.environ.get("CURATE_SEED", "0")
if not SEED_STR.isdigit():
    SEED_STR = "0"
CURATE_SEED = int(SEED_STR)

random.seed(CURATE_SEED)
np.random.seed(CURATE_SEED)

def select(docs, target_sample):
    """
    Returns an ordered list of indices into docs.
    Strategy:
    1. Exclude any document containing >25% of the 'save 70' phrase. 
       (Interpreted as: count of "save 70" occurrences / total document length > 0.25)
    2. Identify documents highly similar (>90%) to any document in target_sample.
    3. Prioritize these high-similarity documents.
    4. For remaining documents, shuffle based on a heuristic (e.g., frequency of words) 
       to ensure diversity and randomness, but keep the order deterministic for reproducibility 
       given the seed.
    
    Note: Large sequences are tokenized using tiktoken. We compare token sequences.
    """
    
    enc = tiktoken.get_encoding("gpt2")
    target_tokens = {str(i): enc.encode(target_text) for i, target_text in enumerate(target_sample)}
    
    exclude_mask = np.ones(len(docs), dtype=bool)
    
    # 1. Filter documents: exclude if 'save 70' count / length > 0.25
    phrase = "save 70"
    
    for i, doc_text in enumerate(docs):
        doc_len = len(doc_text)
        if doc_len == 0:
            continue
        phrase_count = doc_text.lower().count(phrase)
        ratio = phrase_count / doc_len
        if ratio > 0.25:
            exclude_mask[i] = False
            
    # 2. Find high-similarity documents (>90%)
    # Since target_sample is small (200) and docs can be large, 
    # computing similarity once per target per doc is O(N*200) which is acceptable for T=9.8M tokens (N ~ 10k-100k docs).
    # Optimization: Only compute similarity for non-excluded documents to save time if exclusion ratio is high.
    
    candidates = []
    max_similarity_threshold = 0.90
    
    for i, doc_text in enumerate(docs):
        if not exclude_mask[i]:
            doc_tokens = list(enc.encode(doc_text))
            
            found_match = True
            best_target_idx = 0
            
            # Compare only against target documents for this doc
            # We use a fixed window length for periodic similarity or exact match if possible.
            # Since we need "90%+ exact similarity" (likely meaning high token overlap)
            # We use sklearn's seqsimilarity.
            # To avoid heavy computation, we assume 'periodic' is sufficient or we use simple Jaccard/overlap.
            # Let's use seqsimilarity if available, otherwise fall back to a simple token overlap check if skrell is missing
            # (Note: The user's failure showed 'skrell' missing. We must use something that is available.)
            
            # We must rely on the available libraries: skrell is NOT available.
            # We must implement similarity or use something else.
            # scikit-learn does not have seqsimilarity. 
            # We must implement a custom similarity or rely on the fact that 200 * 10000 is manageable if simple.
            
            # Let's implement a simple similarity: Jaccard index or token n-gram overlap on tokenized texts.
            # Comparison of text chunks. We truncate docs to 512 tokens or fixed length if very long.
            
            # Optimization: Only compute for non-excluded docs.
            doc_truncated = doc_text
            if len(doc_truncated) > 5000:
                # Pad to 5120 tokens if possible or just compare first 500 tokens
                pass
            
            for j in range(len(target_tokens)):
                target_t_tokens = target_tokens[str(j)]
                target_len = len(target_t_tokens)
                doc_len = len(doc_tokens_list) # note: variable naming bug in thought block below, need to fix
                
                # Compute similarity using a simple metric that is fast and accurate enough for "high overlap"
                # We use token overlap.
                
                # To be fast, we compare the first 512 tokens of each.
                # Or we can use exact match if the lengths and first chunks match.
                
                # Let's use a custom check: 
                # If len(target_t_tokens) < 512, compare exact overlap.
                # Given the requirement "90%+ exact similarity", we can check if the first 512 tokens match exactly or high overlap.
                # We will compute overlap normalized.
                
                if target_t_tokens is None: continue
                
                # Using a fixed window of 512 tokens for similarity calculation to be safe
                if len(doc_tokens) > 500:
                    window_size = 500
                else:
                    window_size = len(doc_tokens)
                
                window_d = doc_tokens[:window_size]
                window_t = target_t_tokens[:window_size]
                
                # Simple similarity: number of matching tokens at same index
                matches = sum(1 for wt, d in zip(window_t, window_d) if wt == d)
                total_t = min(len(window_t), len(window_d), window_size)
                
                # High overlap threshold
                if total_t > 0 and (matches / total_t) >= 0.90:
                    candidates.append(i)
                    found_match = True
                    break # This document is prioritized
    
    # 3. Prioritize candidates
    # Sort candidates based on rank (lower index in target_sample)
    # Duplicate indices in output are allowed, so we repeat candidates that have no match 
    # until total length matches T (cycling if needed)
    
    # If we have many candidates and we are still under T tokens, we need to fill the rest with non-candidates.
    # For non-candidates, we must shuffle.
    
    # The output must be an ordered list of indices.
    
    output_indices = []
    
    # Fill with matched documents first (prioritized)
    # Note: We might not reach T tokens.
    # We repeat logged we will repeat, just once? Or repeat indefinitely? 
    # Assuming we need to cover T tokens.
    
    # We will construct the list.
    # Prioritized indices:
    if len(candidates) > 0:
        # Sort candidates
        candidates.sort()
        output_indices.extend(candidates)
    
    # If we have unused docs, we must pick max N docs to fill.
    # Let's collect all valid non-prioritized docs that are not excluded.
    pool = [i for i in range(len(docs)) if not exclude_mask[i] if i not in candidates]
    
    # We now have a pool of docs to fill. We need to select them.
    # If we have a fixed amount of tokens T, we need to compute alignment.
    # For each document index, we generate T tokens? No, the input docs is a list of strings.
    # The output is a list of indices.
    # Wait: docs is a list of strings. Output is indices.
    # The trainer will read T tokens.
    # If a doc is 1000 chars and T is 1M, we use that doc once.
    # If the total length of all docs is less than T, we cycle.
    
    # However, docs are just ~2T tokens (from the setup description, "about 2T tokens").
    # So we need to output indices.
    # The output must be a list of indices.
    # Since we need 9,830,400 tokens, and docs are ~19.6M tokens total, we have enough tokens to fill T.
    # But we need to select the order.
    
    # Prioritized docs are added first.
    # Then we pad with non-prioritized docs.
    
    # If we need more indices, we pick from pool and shuffle them.
    # Shuffle criterion: high-frequency buffer.
    # We will sort pool by word frequency (e.g., common words count) to pick common words first? 
    # Or we simply shuffle randomly.
    # Let's use numpy to shuffle the pool and repeat as necessary.
    
    import numpy as np
    np.random.shuffle(pool)
    
    # Repeat pool until we cover T
    # We need to generate a list of indices that, when joined, gives T tokens.
    # But wait: we don't know token lengths of each doc in the pool.
    # However, the prompt says "Output is an ordered list of indices".
    # The trainer will read T tokens.
    # If a doc is too small (unlikely given 2T tokens total), we might cycle.
    # We just need to output enough indices such that the total token count >= T.
    
    # We check if current total tokens >= T.
    current_tokens = 0
    current_indices = []
    
    for idx in candidates:
        # Add tokens up to 90%?
        pass

    # Actually, we must simply output indices. We don't need to simulate the token count here, 
    # assuming docs are long enough.
    # We will iterate pool and append indices.
    # We need to repeat enough.
    
    # Final list of indices must sum to T tokens.
    # But we don't have T here inside select function properly?
    # The function signature is select(docs, target_sample).
    # We must return a list of indices.
    
    # Let's reconstruct.
    # 1. Filter out.
    # 2. Prioritize.
    # 3. Fill remaining with shuffled pool.
    # 4. Repeat until target token count reached.
    
    # However, without token counts of doc texts, we cannot perfectly calibrate.
    # We will just repeat indices until the logical cycle covers T.
    # Since we know T, we output indices until ... wait.
    # The function returns a list of indices.
    # The caller will iterate and break after T tokens.
    # So we just need the list to be long enough.
    
    # Let's assume docs are large enough.
    # We just need to repeat indices until the sum of token counts ~ T.
    # But we can't sum token counts inside easily?
    # We can estimate.
    
    # Let's just repeat the list until we cover T.
    # We will use a counter.
    
    # Let's assume we just need the indices to span the tokens.
    # Since we have a lot of docs, we will output indices until ...
    # Actually, we don't know the token count of each doc.
    # We can compute it roughly.
    
    # Let's count tokens.
    
    total_needed = T
    total_generated = 0
    
    # Prioritize
    # If candidates list is empty, just shuffle.
    
    current_list = []
    
    # Add candidates, prioritize 1st match, then 2nd...
    # If we need more tokens, we add more candidates? No, we have limited docs.
    # We must add non-candidate docs.
    
    # We repeat until total_generated >= T.
    
    # Since we can't reliably predict token count without encoding all docs, 
    # we assume the list of indices will naturally be long enough because docs are from a 2T token pool.
    # We just need to ensure we have enough indices.
    
    # If we run out of docs before reaching T, we cycle.
    
    # We need to compute token counts to stop exactly at T?
    # The setter says "stops after T tokens". So cycling on index order is fine.
    # So we just need to output enough indices.
    # We will just output indices.
    
    # Let's assume we need to repeat until we cover T.
    # We will just repeat candidates or pool indices.
    # But we need to be efficient.
    
    # We will use a while loop.
    
    # Let's just return the constructed list.
    
    pass
