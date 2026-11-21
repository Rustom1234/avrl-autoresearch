import os
import random
import math
import zlib
import numpy as np
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.encoding_for_model("gpt2")
    
    # (1) Build bigram set c from lowercased character bigrams of the whole target sample
    c_set = set()
    target_text = "".join(target_sample).lower()
    for i in range(len(target_text) - 1):
        b = target_text[i:i+2]
        c_set.add(zlib.adler32(b.encode('utf-8')) & 0xFFFFFFFF)
        
    # Calculate h = fraction of each pool doc's full-text character bigrams in c
    h_scores = []
    for doc in docs:
        doc_low = doc.lower()
        count = 0
        total = 0
        for i in range(len(doc_low) - 1):
            b = doc_low[i:i+2]
            total += 1
            if zlib.adler32(b.encode('utf-8')) & 0xFFFFFFFF in c_set:
                count += 1
        h_scores.append(count / total if total > 0 else 0.0)

    # (2) Unique-dom doc per doc: exact duplicates, lowercased whitespace-normalised, 
    # truncate to 3000 chars and hash the whitespace-encoded bytes.
    seen_hashes = set()
    unique_indices = []
    for i, doc in enumerate(docs):
        norm = " ".join(doc.lower().split())[:3000]
        h = zlib.adler32(norm.encode('utf-8')) & 0xFFFFFFFF
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(i)
    
    # Re-calculate scores for unique docs
    u_h_scores = [h_scores[i] for i in unique_indices]
    
    # (1) Calibration with half-half leave-out
    # Split unique indices into two halves
    mid_idx = len(unique_indices) // 2
    left = unique_indices[:mid_idx]
    right = unique_indices[mid_idx:]
    
    def get_score_list():
        # Scoring helper (simple probability or density isn't defined, 
        # but we use h as the score for calibration)
        return [u_h_scores[i] for i in unique_indices]

    # Logic for Lo/Hi calibration
    # Split the list into two halves to find lo and hi
    scores_left = sorted([u_h_scores[i] for i in left])
    scores_right = sorted([u_h_scores[i] for i in right])
    
    # Midpoints of [2.5th-95th] and [5th-95th]
    lo_score = 0.0
    if scores_left:
        lo_score = (scores_left[int(len(scores_left) * 0.025)] + 
                     scores_left[int(len(scores_left) * 0.95)]) / 2
    
    hi_score = 0.0
    if scores_right:
        hi_score = (scores_right[int(len(scores_right) * 0.05)] + 
                     scores_right[int(len(scores_right) * 0.95)]) / 2
        
    # Keep if h >= lo AND h >= hi
    kept_indices = []
    for i, idx in enumerate(unique_indices):
        if u_h_scores[i] >= lo_score and u_h_scores[i] >= hi_score:
            kept_indices.append(idx)

    # (3) Split into A and B by hash
    def get_doc_hash(idx):
        return zlib.adler32(docs[idx][:200].encode('utf-8')) & 0xFFFFFFFF
    
    kept_indices.sort(key=get_doc_hash)
    split_pt = len(kept_indices) // 2
    A = kept_indices[:split_pt]
    B = kept_indices[split_pt:]
    
    # (3) Populate selection as far as possible from A (weighted 2)
    # Sort A by token length descending
    a_data = sorted([(i, len(enc.encode(docs[i]))) for i in A], key=lambda x: x[1], reverse=True)
    b_data = sorted([(i, len(enc.encode(docs[i]))) for i in B], key=lambda x: x[1], reverse=True)
    
    selected_indices = []
    current_tokens = 0
    
    # Take A first (weight 2 effect: fill A completely if possible)
    for idx, count in a_data:
        if current_tokens + count <= T:
            selected_indices.append(idx)
            current_tokens += count
            
    # Top up with B until T
    for idx, count in b_data:
        if current_tokens + count <= T:
            selected_indices.append(idx)
            current_tokens += count
            
    # (5) Span filler if A was insufficient
    # This part triggers if we didn't reach T tokens with the primary A+B logic
    if current_tokens < T:
        # Check if A alone was insufficient
        a_tokens_count = sum(count for _, count in a_data)
        if a_tokens_count < T:
            # Fill remaining gap with raw clean B text (first 2000 chars)
            # B sorted by descending h
            b_sorted_by_h = sorted(B, key=lambda i: u_h_scores[unique_indices.index(i)], reverse=True)
            
            # We need to append these as distinct indices to the list
            # but the rule says "append the next doc's first 2000 chars"
            # To stay within the return type (indices into docs), we use the B indices
            # representing the "clean" parts.
            for b_idx in b_sorted_by_h:
                if current_tokens < T:
                    # Use the doc but it's effectively the "first 2000 chars"
                    # Since we return indices, we use the full index
                    # However, the requirement says "append first 2000 chars"
                    # We take the doc and let the model see it.
                    # To be safe with "append", we just include them.
                    # But if we must strictly only include 2000 chars, we can't 
                    # because we must return indices. We return the index.
                    # Actually, the logic is to fill the gap.
                    count_limit = min(len(enc.encode(docs[b_idx])), T - current_tokens)
                    # Since we can't truncate docs via indices, we add the index.
                    selected_indices.append(b_idx)
                    current_tokens += len(enc.encode(docs[b_idx]))
                else:
                    break
    
    # (4) If kept docs < T, cycle. This is handled by the return list 
    # being read/cycled by the trainer (trainer logic).
    # If A alone was insufficient and we couldn't reach T, the prompt says:
    # "If even that cannot reach T, keep only kept docs and let them cycle; 
    # emit NONE otherwise."
    # "emit NONE" usually implies returning an empty list in these contexts.
    
    # Wait, the prompt says "emit NONE otherwise" if "even that cannot reach T".
    # Let's check if we reached T.
    # The problem is current_tokens calculation is based on full doc length.
    # If we still haven't reached T, we keep selected_indices (the kept docs).
    # But if we didn't reach T even with the filler, we keep the kept docs.
    # Let's re-read: "If even that cannot reach T, keep only kept docs and let them cycle; emit NONE otherwise."
    # This "emit NONE" might refer to the gap-filling.
    
    # Corrected logic for (5):
    # If current_tokens < T:
    #   If we could fill it: keep selected_indices
    #   If we couldn't reach T even with fillers: keep ONLY the kept_indices
    #   Wait, the "emit NONE" is usually for a different type of task. 
    #   In these coding challenges, if it's unreachable, return [].
    
    # Let's stick to the logic:
    # 1. A + B
    # 2. If gap, B-filler
    # 3. If still gap, just kept_indices.
    # 4. If it's impossible to reach T at all (even with cycling), the trainer handles it.
    
    # Since "emit NONE" is ambiguous, and the trainer cycles, 
    # the most robust interpretation is to return selected_indices.
    
    # One more check: "If A alone holds fewer than T tokens... fill... If even that cannot reach T, 
    # keep only kept docs... emit NONE otherwise."
    # I will treat "emit NONE" as returning [] if T > 0 and we have no docs.
    
    if not selected_indices and T > 0 and docs:
        # This part is tricky. If we have no selected_indices, 
        # but we have kept_indices, we should return kept_indices.
        selected_indices = kept_indices
        
    if not selected_indices and T > 0:
        return []

    rng.shuffle(selected_indices)
    return selected_indices
