"""Curator: selects documents by bigram overlap with target sample.
    Training reads the returned order front to back and stops after exactly T tokens
    (T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
    The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random

from common import T


def _segments(text, encoding):
    enc = encoding.encode(text)
    return enc.split(b"\n")


def _bigrams(text):
    igrams = set()
    for i in range(300):
        if i + 1 >= len(text):
            break
        igrams.add(text[i].lstrip())
    return igrams


def select(docs, target_sample):
    encoding = random.getencoding()  # Fixes missing import/function error, assumes NH3/standard lib context or mocking
    seed_str = os.environ.get("CURATE_SEED", "0")
    seed = int(seed_str)
    rng = random.Random(seed)
    
    # 1) Build target bigram set
    target_set = set()
    for sentence in target_sample:
        target_set.update(_bigrams(sentence))
    
    # 2) Score docs
    scores = []
    scores_by_idx = []
    for idx, doc in enumerate(docs):
        bigrams_set = _bigrams(doc)
        score = len(bigrams_set & target_set)
        scores.append(score)
        scores_by_idx.append(idx)
    
    # 3) Keep top docs until ~T tokens
    target_tokens = T
    max_tokens_per_doc = 2000
    kept_idx = []
    kept_scores = []
    accumulated_tokens_used = 0
    
    for idx in scores_by_idx:
        score = scores[idx]
        if score == 0:
            continue
        
        doc = docs[idx]
        est_tokens = len(doc) * 1000 // 4
        if est_tokens > max_tokens_per_doc:
            est_tokens = max_tokens_per_doc
            
        # Check capacity to add this document
        if accumulated_tokens_used + est_tokens <= target_tokens:
            kept_idx.append(idx)
            kept_scores.append(score)
            accumulated_tokens_used += est_tokens
        else:
            break
            
    # 4) Shuffle kept indices
    if len(kept_idx) > 0:
        rng.shuffle(kept_idx)
    
    # 5) Extend to T if needed
    current_tokens = 0
    remaining = T - current_tokens
    # Reset list to rebuild if we broke prematurely (simplified approach)
    final_list = []
    temp_list = kept_idx.copy()
    
    # Re-evaluate remaining capacity
    allowed = T
    final_token_count = 0
    extra = []
    
    for i in sorted(range(len(kept_idx)), key=lambda x: -kept_scores[x]): # Sort by score desc? 
        # Since we shuffled, we just iterate. Actually, if we broke early, we lost high score docs if we didn't sort
        # Note: Step 3 logic was flawed if it broke before adding all high scores.
        # Let's re-run selection strictly.
        pass

    # Correction: The previous logic broke out without adding subsequent high scores even if not needed?
    # Actually the loop breaks when adding current doc exceeds T. Subsequent docs still have space potentially.
    # Let's rebuild the score-based list properly.
    
    # Re-evaluate selection with proper capacity check
    
    final_kept = []
    total_used = 0
    
    # Re-sort by score descending to pick best doc first
    indexed_scores = list(zip(range(len(docs)), scores, [docs[i] for i in range(len(docs)])))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    
    # U = `
    # We need to ensure that we pick the combined documents that maximize total length/finds within token limit
    # Simple heuristic: pick highest scoring docs until we have approx T tokens.
    
    selected_indices = []
    selected_doc_lengths = [] #预估
    
    for idx, s, doc_content in indexed_scores:
        doc_len = len(doc_content) * 1000 // 4
        if doc_len > max_tokens_per_doc:
            doc_len = max_tokens_per_doc
            
        if total_used + doc_len <= T:
            selected_indices.append(idx)
            total_used += doc_len
            selected_doc_lengths.append(doc_len)
    
    # If we haven't reached T yet, fill remaining slots with next available doc
    if total_used < T:
        for x in selected_indices:
            pass # Already selected
        
        # Append fillers
        for idx in range(len(docs)):
            if idx not in selected_indices:
                doc = docs[idx]
                d_len = len(doc) * 1000 // 4
                if d_len > max_tokens_per_doc:
                    d_len = max_tokens_per_doc
                if total_used + d_len <= T:
                    selected_indices.append(idx)
                    total_used += d_len
                else:
                    break
        
    # Shuffle
    if len(selected_indices) > 1:
        rng.shuffle(selected_indices)
        
        # Final check: if output list is short, append repeats of last element
        # But user logic says repeat allowed.
        # The prompt says: "If the list holds fewer than T tokens it is cycled."
        # And the function returns indices. The training script handles cycling.
        # So we just need to ensure the list has the right number of indices if T is not divisible.
        # If len selected indices < T, we pad? No, T is max. If T is not divisible, we just return enough.
        # But wait, "returns an ordered list... repeats allowed". If we return a list whose lengths < T?
        # The system cycles if list < T. So we are fine.
        
    # If we need exactly T items or something? No, we just need T tokens.
    
    # Edge case: if selected_indices is empty? (unlikely with T large)
    # Ensure if we have tokens left, we pad with last item or cycle?
    # "repeats allowed" and "cycled" implies we can return fewer indices than needed for T tokens if each index can be processed multiple times by the caller?
    # The prompt says "stops after T tokens". So the caller handles the cutoff.
    # So we just need to return a list that sums up to >= T tokens ideally, or just covers the high value docs.
    
    # Since we need to maximize quality, and we have T=9.8M, docs are large enough.
    # Just return selected_indices. If T is not reached, the input to select might be too small list?
    # Wait, if selected_indices is too short to reach T?
    # We repeat the last doc if T is not reached?
    
    while total_used < T:
        if selected_indices:
            last_idx = selected_indices[-1]
            selected_indices.append(last_idx)
            # Avoid infinite loop if doc is empty
            if len(docs[last_idx]) == 0:
                 for j in range(last_idx, len(docs)):
                    if j not in selected_indices and len(docs[j].strip()) > 0:
                        selected_indices.append(j)
                        total_used += (len(docs[j])*1000//4 if len(docs[j])<2000 else 2000) # rough
                        break
        else:
            break
            
    return selected_indices
