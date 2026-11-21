```python
import os
import random
import zlib
import collections
import numpy as np
from common import T

def get_doc_id(doc, length=300):
    # Hash of the first 300 chars or all lowercased characters and first 200 words
    # To ensure deterministic and stable ID, we'll use a simple hash of the slice
    # However, the prompt asks for two specific definitions: 
    # "H합 of the first 300 chars" or "hash of all lowercased characters and first 200 words".
    # We will use the hash of the first 300 chars as it is more standard for "doc id".
    content = doc[:length].encode('utf-8')
    return hash(content)

def get_bigrams(text):
    words = text.lower().split()
    bigrams = set()
    for i in range(len(words) - 1):
        bigrams.add((words[i], words[i+1]))
    return bigrams

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Build B
    B = set()
    for ts in target_sample:
        B.update(get_bigrams(ts))
        
    # 2. Split target docs
    # Use hash of first 200 chars to split
    def get_split_key(s):
        return hash(s[:200].encode('utf-8')) % 2
    
    half0_indices = []
    half1_indices = []
    for i, ts in enumerate(target_sample):
        if get_split_key(ts) == 0:
            half0_indices.append(i)
        else:
            half1_indices.append(i)
            
    # 3. Compute a0, a1
    def get_score(ts_list):
        scores = {}
        for idx in ts_list:
            ts = target_sample[idx]
            words = ts.lower().split()
            # first 3000 words
            subset = words[:3000]
            count = 0
            total = 0
            for i in range(len(subset) - 1):
                bi = (subset[i], subset[i+1])
                if bi in B:
                    count += 1
                total += 1
            scores[idx] = count / total if total > 0 else 0.0
        return scores

    a0_dict = get_score(half0_indices)
    a1_dict = get_score(half1_indices)
    
    # 4. Calculate thr
    all_scores = []
    for i in range(len(target_sample)):
        s0 = a0_dict.get(i, 0) if i in half0_indices else 0
        s1 = a1_dict.get(i, 0) if i in half1_indices else 0
        # The prompt says "score of half 0" and "half 1"
        # We use the specific scores for the docs in those halves
    
    # Actually, "a0 + a1 / 2 over the 200 target docs"
    # We need a0[i] or a1[i] for each of the 200 docs
    target_scores = []
    for i in range(len(target_sample)):
        if i in half0_indices:
            target_scores.append(a0_dict[i])
        else:
            target_scores.append(a1_dict[i])
    
    # The prompt says "sd the std of (a0 + a1)/2 over the 200 target docs"
    # This implies we should treat each of the 200 as having an a0 and an a1.
    # But docs only belong to one half. Based on instructions:
    # "a0 = dict of half 0 doc id -> fraction... a1 = same for half 1"
    # Let's assume if doc is in half 0, its score is a0_dict, else a1_dict.
    # To get (a0+a1)/2 for all 200, we'd need both for all.
    # Re-reading: "sd the std of (a0 + a1)/2 over the 200 target docs" 
    # Given the structure, it likely means the mean of scores of half 0 plus mean of half 1 divided by 2.
    # Or scores are pre-computed for all. Let's use the scores available.
    # Since we can't have both for one doc (they are partitioned), 
    # we use the mean of all 200 scores as the basis.
    
    # Let's interpret (a0+a1)/2 over the 200 docs as:
    # mean(scores_in_half0 + scores_in_half1) / 2
    # But we only have one score per doc. Let's just use the list of 200 scores.
    vals = np.array(target_scores)
    sd = np.std(vals)
    # The prompt says "a0 + 0.5*sd" where a0 is the score of a doc in half 0.
    # This is ambiguous. "thr = max(0.33, a0 + 0.5*sd)" - which a0? 
    # Likely the average of a0 values in half 0.
    avg_a0 = np.mean(list(a0_dict.values())) if a0_dict else 0
    thr = max(0.33, avg_a0 + 0.5 * sd)

    # 5. C0, C1
    def get_c(indices, score_dict):
        c = {}
        for idx in indices:
            doc = target_sample[idx] # This is wrong, we need to filter 'docs'
            pass
        # Wait, the target_sample is just used to build B and calculate thr.
        # The pool 'docs' is what we filter.
        return c

    # Correcting logic: docs is the pool to be filtered.
    # a0 and a1 were calculated on target_sample.
    # But the prompt says: "a0 = dict of half 0 doc id -> fraction... 
    # that are in B". This refers to the docs in the pool? 
    # "a0 = dict of half 0 doc id" -> implies we need to know which docs belong to half 0.
    # But "half 0" was defined by target_sample.
    # Interpretation: A doc in the pool belongs to half 0 if its hash is 0.
    
    def get_pool_scores(pool_docs):
        # We need to know which doc belongs to which half to calculate a0/a1
        # The prompt says "a0 = dict of half 0 doc id ... a1 = same for half 1"
        # "half 0" was "Split the target docs into two halves".
        # This is a contradiction. Let's assume:
        # a0_dict = {doc_id: score} for all docs in pool that would fall into half 0
        # if they were target docs (i.e., hash(first 200) % 2 == 0)
        s0 = {}
        s1 = {}
        for d in pool_docs:
            d_id = get_doc_id(d)
            # Use hash of first 200 chars to determine half
            split_key = hash(d[:200].encode('utf-8')) % 2
            words = d.lower().split()
            subset = words[:3000]
            count = 0
            total = 0
            for i in range(len(subset) - 1):
                bi = (subset[i], subset[i+1])
                if bi in B: count += 1
                total += 1
            score = count / total if total > 0 else 0.0
            if split_key == 0: s0[d_id] = score
            else: s1[d_id] = score
        return s0, s1

    # Let's re-read carefully: "a0 = dict of half 0 doc id... a1 = same for half 1"
    # This follows "Split the target docs...". It's highly likely a0/a1 are scores 
    # for the target docs.
    
    # Re-evaluating:
    # 3. a0 = scores of target docs in half 0.
    # 4. thr = max(0.33, avg(a0_scores) + 0.5*sd)
    # 5. C0 = docs in pool that belong to half 0 (hash % 2 == 0), 
    #    len >= 200, zlib <= 0.25, score >= thr.
    
    # Let's compute scores for ALL docs in the pool to make C0/C1 selection
    pool_scores = {}
    pool_ids = {}
    
    # Re-calculating thr correctly:
    # "a0 = dict of half 0 doc id -> fraction... over the 200 target docs"
    # This confirms a0/a1 are scores for the 200 target docs.
    # a0_dict is scores of docs in target_sample that have hash % 2 == 0
    a0_dict = get_score(half0_indices)
    a1_dict = get_score(half1_indices)
    
    # Calculate sd over (a0 + a1)/2 over the 200 target docs
    # This means for each of the 200 docs, we need its a0 and a1.
    # But they are split. This likely means (score_of_half0 + score_of_half1) / 2
    # where score_of_half0 is the average score of half 0.
    mean_a0 = np.mean(list(a0_dict.values())) if a0_dict else 0
    mean_a1 = np.mean(list(a1_dict.values())) if a1_dict else 0
    avg_score_target = (mean_a0 + mean_a1) / 2
    
    # To get sd, we need 200 values.
    # "sd the std of (a0 + a1)/2 over the 200 target docs"
    # Since each target doc only has one score, let's use that score as the value.
    # "over the 200 target docs"
    target_vals = []
    for i in range(len(target_sample)):
        if i in half0_indices: target_vals.append(a0_dict[i])
        else: target_vals.append(a1_dict[i])
    # If "a0 + a1 / 2" means (score_of_half0_avg + score_of_half1_avg) / 2 for every doc?
    # That would make sd = 0. That's unlikely.
    # It probably means the score of the doc itself.
    sd = np.std(target_vals)
    thr = max(0.33, mean_a0 + 0.5 * sd)

    # 5. C0, C1 from pool
    C = {}
    # We need doc_id as key to keep unique docs. 
    # "dedupe with hash of the first 500 lowercased chars"
    # Let's use a helper to get the 500-char hash.
    def get_dedupe_id(d):
        return hash(d.lower()[:500].encode('utf-8'))

    # We need to keep the "id" (H합 of first 300 chars)
    doc_info = [] # list of (original_index, doc_string, doc_id, dedupe_id)
    for i, d in enumerate(docs):
        d_id = get_doc_id(d)
        dedupe_id = get_dedupe_id(d)
        
        # Score it
        words = d.lower().split()
        subset = words[:3000]
        count = 0
        total = 0
        for j in range(len(subset) - 1):
            bi = (subset[j], subset[j+1])
            if bi in B: count += 1
            total += 1
        score = count / total if total > 0 else 0.0
        
        # Check conditions
        is_half0 = hash(d[:200].encode('utf-8')) % 2 == 0
        valid_len = len(d) >= 200
        valid_zlib = zlib.adler32(d.encode('utf-8')) % 1000000 # Use adler32 as ratio substitute if needed
        # Wait, "zlib ratio <= 0.25" usually means len(compressed) / len(original)
        import zlib
        compressed = zlib.compress(d.encode('utf-8'))
        valid_zlib = (len(compressed) / len(d)) <= 0.25 if len(d) > 0 else True
        
        if valid_len and valid_zlib:
            if is_half0 and score >= thr:
                C[d_id] = (i, d, dedupe_id)
            elif not is_half0 and score >= thr:
                C[d_id] = (i, d, dedupe_id)

    # 6. Dedupe by dedupe_id
    final_docs = {}
    for d_id, (orig_idx, d_str, ded_id) in C.items():
        if ded_id not in final_docs:
            final_docs[ded_id] = (orig_idx, d_str, d_id)
    
    # C is now the set of docs
    # 7. Rebuild if < 10
    # We need a way to iterate "next docs by id". Let's sort all docs by original index.
    # But the prompt says "rebuild C... take the next docs by id until 10 are found."
    # Let's sort all docs by their d_id.
    all_docs_sorted = sorted(enumerate(docs), key=lambda x: get_doc_id(x[1]))
    
    def rebuild_C(current_thr):
        new_C = {}
        for i, d in all_docs_sorted:
            d_id = get_doc_id(d)
            words = d.lower().split()
            subset = words[:3000]
            count = 0
            total = 0
            for j in range(len(subset) - 1):
                bi = (subset[j], subset[j+1])
                if bi in B: count += 1
                total += 1
            score = count / total if total > 0 else 0.0
            
            is_half0 = hash(d[:200].encode('utf-8')) % 2 == 0
            # This is a bit complex, let's simplify:
            # if it meets thr and is half 0 OR is half 1
            # Actually, the condition is "merged C0 and C1"
            # Let's just check if score >= thr (regardless of half for this part)
            # Wait, the prompt says "rebuild C as the union of the two halves' docs"
            # This implies the half-logic still applies.
            if score >= current_thr:
                dedupe_id = get_dedupe_id(d)
                if dedupe_id not in new_C:
                    new_C[dedupe_id] = (i, d, d_id)
        return new_C

    # Refined Rebuild Logic for step 7:
    # "If len(C) < 10, set thr = max(0.25, a0 + 0.2*sd) over the half scores and 
    # rebuild C as the union of the two halves' docs that meet it... 
    # if still < 10, take the next docs by id until 10 are found."
    
    # First, let's get the scores of all target docs to find a0 and a1 for thr
    # (already done: a0_dict and a1_dict)
    # Actually, "a0 + 0.2*sd" - is it mean_a0? Let's use mean_a0.
    
    # Let's restart the logic for C slightly to be more robust:
    # 5. C0 = docs in
