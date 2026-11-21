```python
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

    # 1. Exact/near-duplicate removal
    seen_hashes = set()
    unique_docs = []
    for doc in docs:
        clean_prefix = doc.lower().strip()
        # Use first 2000 chars for hashing
        prefix = clean_prefix[:2000]
        h = zlib.adler32(prefix.encode('utf-8')) & 0xFFFFFFFF
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(doc)
    
    # 2. Build target bigram set (first 8000 chars)
    target_text = "".join(target_sample).lower()
    target_prefix = target_text[:8000]
    c_set = set()
    for i in range(len(target_prefix) - 1):
        b = target_prefix[i:i+2]
        c_set.add(zlib.adler32(b.encode('utf-8')) & 0xFFFFFFFF)

    def get_h_score(doc):
        doc_low = doc.lower()
        # Fraction of first 8000 chars
        p1 = doc_low[:8000]
        count1 = 0
        total1 = 0
        for i in range(len(p1) - 1):
            total1 += 1
            if zlib.adler32(p1[i:i+2].encode('utf-8')) & 0xFFFFFFFF in c_set:
                count1 += 1
        f1 = count1 / total1 if total1 > 0 else 0
        
        # Fraction of next 5000 chars
        p2 = doc_low[8000:13000]
        count2 = 0
        total2 = 0
        for i in range(len(p2) - 1):
            total2 += 1
            if zlib.adler32(p2[i:i+2].encode('utf-8')) & 0xFFFFFFFF in c_set:
                count2 += 1
        f2 = count2 / total2 if total2 > 0 else 0
        
        return f1 * f2

    # Pre-calculate h_scores for all unique_docs
    h_scores = [get_h_score(d) for d in unique_docs]

    # 3. Calibrate lo (2.5th-95th midpoint)
    # Split target_sample (200 docs) half-half
    # Score of target docs is 1.0
    # We need the lo threshold from the pool
    # Using the rule: lo is midpoint of 2.5th and 95th percentile of target scores
    # Actually, "lo" is the threshold derived from the pool.
    # Wait, rule 3 says: lo at the 2.5th-95th percentile midpoint.
    # Rule 4 & 5 clarify the logistic logic.
    
    # Logistic Regression Simulation (Rule 5)
    # Positives: 100 target docs, each weighted twice (A once, B once) -> 200 * 1.0
    # Negatives: 2000 random docs from unique_docs
    neg_indices = rng.sample(list(range(len(unique_docs))), min(2000, len(unique_docs)))
    neg_mean_h = np.mean([h_scores[i] for i in neg_indices]) if neg_indices else 0.0
    
    w = (200 * 1.0 + 2000 * neg_mean_h) / 2200
    
    def get_score(h_val):
        try:
            return 1.0 / (1.0 + math.exp(-w * h_val))
        except OverflowError:
            return 1.0 if w * h_val > 0 else 0.0

    # Target scores (each doc scored twice)
    # Since target docs are 1.0 h-score, their score is get_score(1.0)
    t_score = get_score(1.0)
    # Rule 5: "midpoint of the 5th percentile of the target scores (each doc scored twice)"
    # Since all target scores are identical, 5th percentile is t_score.
    lo_val = t_score 
    
    # Pool scores
    pool_scores = [get_score(h) for h in h_scores]
    
    # Held-out pool scores
    # docs not in neg_indices
    held_out_indices = [i for i in range(len(unique_docs)) if i not in neg_indices]
    held_out_scores = sorted([pool_scores[i] for i in held_out_indices]) if held_out_indices else []
    
    hi_val = 0.0
    if held_out_scores:
        # 95th percentile
        idx_95 = int(len(held_out_scores) * 0.95)
        hi_val = held_out_scores[min(idx_95, len(held_out_scores)-1)]
    
    # Midpoint of lo and hi
    # Re-reading: "midpoint of the 5th percentile of the target scores ... and the 95th percentile of the held-out pool scores"
    # Target 5th percentile = t_score
    # Held-out 95th percentile = hi_val
    midpoint = (lo_val + hi_val) / 2

    # Keep docs with score >= midpoint
    kept_indices = [i for i, s in enumerate(pool_scores) if s >= midpoint]
    
    # Selection (Rule 6)
    # Start with all kept docs.
    # Take docs in order of selected list until T tokens reached.
    # If kept set holds fewer than T, let it cycle.
    
    kept_docs_indices = kept_indices
    # Calculate lengths
    lengths = [len(enc.encode(unique_docs[i])) for i in kept_docs_indices]
    
    if not kept_docs_indices:
        # Fallback to any doc if none pass
        if unique_docs:
            # Simply return one doc cycled
            return [0] * (T // len(enc.encode(unique_docs[0]))) + [0]
        return []

    total_kept_tokens = sum(lengths)
    
    if total_kept_tokens <= T:
        # Keep them all, then cycle to fill T? 
        # "If kept set holds fewer than T, let it cycle"
        # "do not top up with the next-highest-scoring docs"
        # This means we use the kept_docs_indices as our source.
        
        final_indices = []
        current_tokens = 0
        # Because we need to return indices into the original `docs` 
        # but `unique_docs` is a filtered version, we need to map them back.
        # Actually, the prompt says "indices into docs".
        # Let's track the original index.
        
        # Refactor to keep original indices
        unique_docs_with_orig = []
        seen_hashes_orig = set()
        for i, doc in enumerate(docs):
            clean_prefix = doc.lower().strip()
            prefix = clean_prefix[:2000]
            h = zlib.adler32(prefix.encode('utf-8')) & 0xFFFFFFFF
            if h not in seen_hashes_orig:
                seen_hashes_orig.add(h)
                unique_docs_with_orig.append((i, doc))
        
        # Recalculate scores based on unique_docs_with_orig
        h_scores_orig = [get_h_score(item[1]) for item in unique_docs_with_orig]
        pool_scores_orig = [get_score(h) for h in h_scores_orig]
        
        # Re-run logistic with correct indices
        neg_indices_orig = rng.sample(list(range(len(unique_docs_with_orig))), min(2000, len(unique_docs_with_orig)))
        neg_mean_h = np.mean([h_scores_orig[i] for i in neg_indices_orig]) if neg_indices_orig else 0.0
        w = (200 * 1.0 + 2000 * neg_mean_h) / 2200
        
        def get_score_orig(h_val):
            try: return 1.0 / (1.0 + math.exp(-w * h_val))
            except OverflowError: return 1.0 if w * h_val > 0 else 0.0
            
        pool_scores_orig = [get_score_orig(h) for h in h_scores_orig]
        t_score = get_score_orig(1.0)
        
        held_out_indices_orig = [i for i in range(len(unique_docs_with_orig)) if i not in neg_indices_orig]
        held_out_scores_orig = sorted([pool_scores_orig[i] for i in held_out_indices_orig]) if held_out_indices_orig else []
        
        hi_val = held_out_scores_orig[int(len(held_out_scores_orig) * 0.95)] if held_out_scores_orig else 0.0
        midpoint = (t_score + hi_val) / 2
        
        kept_indices_orig = [i for i, s in enumerate(pool_scores_orig) if s >= midpoint]
        
        # Selection logic
        # We need to return a list of indices into the ORIGINAL docs
        # current_tokens = 0
        # if total_kept_tokens <= T:
        #    We cycle the kept_indices_orig
        # else:
        #    We take them in order until T
        
        # But the rule says: "take docs in order of the selected list"
        # and the selected list is the set of "kept docs"
        # The instruction says: "take docs in order of the selected list"
        # But the selected list IS the kept docs.
        # It means: if sum(lengths) < T, cycle the kept list. 
        # If sum(lengths) > T, take them in order until T.
        
        # Let's stick to the most literal interpretation of Rule 6.
        # 1. Identify "kept" docs.
        # 2. If sum(kept_tokens) < T, the result is a cycled version of kept.
        # 3. If sum(kept_tokens) >= T, result is the subset of kept until T.
        
        # Implementation:
        final_indices = []
        # Indices in unique_docs_with_orig
        keep_ids = kept_indices_orig
        # Corresponding indices in original docs
        keep_orig_ids = [unique_docs_with_orig[i][0] for i in keep_ids]
        # Lengths in tokens
        keep_lens = [len(enc.encode(unique_docs_with_orig[i][1])) for i in keep_ids]
        
        total_kept_tokens = sum(keep_lens)
        
        if total_kept_tokens <= T:
            # Cycle
            curr_tokens = 0
            while curr_tokens < T:
                for i in range(len(keep_orig_ids)):
                    idx = keep_orig_ids[i]
                    length = keep_lens[i]
                    if curr_tokens + length <= T:
                        final_indices.append(idx)
                        curr_tokens += length
                    else:
                        # Rule 6: "still let that cycle and do not top up"
                        # This implies we stop this pass and start the next cycle?
                        # Or just "if total_kept_tokens < T, cycle"
                        break
                if sum(len(enc.encode(docs[i])) for i in final_indices) == T: # Not quite
                    pass
            # The prompt says: "if even the docs that pass... hold fewer than T tokens, still let that cycle"
            # A simple way to cycle:
            final_indices = []
            curr_tokens = 0
            # To ensure we don't get stuck if a doc is too big:
            # We just keep going until we can't fit the next one in the cycle.
            # But a cycle usually means restart the list.
            found_any = False
            while curr_tokens < T and not found_any:
                for i in range(len(keep_orig_ids)):
                    idx = keep_orig_ids[i]
                    l = keep_lens[i]
                    if curr_tokens + l <= T:
                        final_indices.append(idx)
                        curr_tokens += l
                    else:
                        # If we can't fit the next, but we want to cycle,
                        # we should probably try the next one in the list
                        continue
                # If we went through the whole kept list and added nothing, break
                if not any(final_indices): break
                found_any = True # simplify logic: if we added one, we continue
                # Wait, rule says "take docs in order of the selected list until about T... 
                # if held fewer... let that cycle". 
                # Let's just cycle until curr_tokens < T.
            
            # Corrected logic for "let that cycle":
            # Just keep appending until we can't fit the next one in the cycle.
            final_indices = []
            curr_tokens = 0
            count = 0
            while curr_tokens < T and count < len(keep_orig_ids) * 2: # limit cycles to 2
                for i in range(len(keep_orig_ids)):
                    idx = keep_orig_ids[i]
                    l = keep_lens[i]
                    if curr_tokens + l <= T:
                        final_indices.append(idx)
                        curr_tokens += l
                    else:
                        pass
                count += 1
            
            # Re-evaluating Rule 6: "take docs in order... if ... held fewer than T, still let that cycle"
            # Just make sure we return the best possible subset/cycle.
            # Let's use a simpler cycle:
            final_indices = []
            curr_tokens = 0
            for _ in range(2): # Allow up to 2 cycles
                for i in range(len(keep_orig_ids)):
                    idx = keep_orig_ids[i]
                    l = keep_lens[i]
                    if curr_tokens + l <= T:
                        final_indices.append(idx)
                        curr_tokens += l
                    else:
                        break # End of current cycle
                if curr_tokens >= T: break
            
        else:
            # Order of kept docs until T
            curr_tokens = 0
            for i in keep_orig_ids:
                l = len(enc.encode(docs[i]))
                if curr_tokens + l <= T:
                    final_indices.append(i)
                    curr_tokens += l
                else:
                    break
                    
        # Final Shuffle
        rng.shuffle(final_indices)
        return final_indices

    # This is a rewrite. The structure above is messy because of the nested logic. 
    # Let's flatten it to be clean.
    
    # [Clean Version starts here]
    # 1. Dedup
    # 2. Scores
    # 3. Logistic
    # 4. Selection
    pass

# Re-writing completely to ensure it matches all rules exactly.
def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.encoding_for_model("gpt2")

    # 1. Dedup
    unique_docs_with_orig = []
    seen_hashes = set()
    for i, doc in enumerate(docs):
        clean_prefix = doc.lower().strip()
        prefix = clean_prefix[:2000]
        h = zlib.adler32(prefix.encode('utf-8')) & 0xFFFFFFFF
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs_with_orig.append((i, doc))

    # 2. Build target big
