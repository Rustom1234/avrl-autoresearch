import os
import random
import hashlib
import tiktoken
import numpy as np
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.encoding_for_model("gpt2")
    
    def get_trigrams(text):
        # Whitespace-normalised, lowercased
        words = text.lower().split()
        if len(words) < 3:
            return set(), 0
        
        trigrams = set()
        for i in range(len(words) - 2):
            # ' ' + a + ' ' + b + ' ' + c
            tri_str = f" {words[i]} {words[i+1]} {words[i+2]} "
            h = hashlib.md5(tri_str.encode('utf-8')).digest()
            # Use raw bytes as hash for efficiency (FP16-like behavior)
            trigrams.add(h)
        return trigrams, len(trigrams)

    # 1 & 2. Build trigram set from target docs
    target_trigrams = set()
    target_doc_trigrams = []
    for doc in target_sample:
        trigs, count = get_trigrams(doc)
        target_doc_trigrams.append(count)
        target_trigrams.update(trigs)
    
    # Scoring function h
    def calc_h(doc):
        trigs, count = get_trigrams(doc)
        if count == 0: return 0.0
        hits = sum(1 for t in trigs if t in target_trigrams)
        return hits / count

    # Calculate m (baseline)
    # Since we are looking for matches against the set of target trigrams, 
    # m is effectively the average density of the target set itself.
    # In this context, m = sum(hits_per_doc / count_per_doc) / len(target_sample)
    # But instruction says "ratio for a replicated copy of the target set"
    # We can approximate m as the average h of the target_sample.
    m_vals = [calc_h(doc) for doc in target_sample]
    m = sum(m_vals) / len(m_vals) if m_vals else 0.0

    # 3. Calibrate threshold t
    m_vals.sort()
    if len(m_vals) > 0:
        t = m_vals[int(0.05 * len(m_vals))]
    else:
        t = 0.0
    
    # 4 & 5. Filter and Score
    junk_markers = {'coupon', 'discount code', 'free shipping', 'buy now', 'Order cannes', 'Get hazard'}
    scored_docs = []
    
    for i, doc in enumerate(docs):
        # Web-junk check
        if sum(1 for marker in junk_markers if marker in doc.lower()) >= 3:
            continue
            
        trigs, count = get_trigrams(doc)
        h = sum(1 for tr in trigs if tr in target_trigrams) / count if count > 0 else 0.0
        
        # Damage check
        if h < 0.4 * t and count < 100:
            continue
            
        # 6. Score with noise
        # Get std of pooled scores to add noise
        scored_docs.append({'idx': i, 'h': h, 'tokens': len(enc.encode(doc))})

    # Calculate standard deviation of h for noise
    if len(scored_docs) > 1:
        hs = [d['h'] for d in scored_docs]
        std_h = np.std(hs)
        for d in scored_docs:
            d['score'] = d['h'] + rng.uniform(-0.1 * std_h, 0.1 * std_h)
    else:
        for d in scored_docs:
            d['score'] = d['h']

    # 7. Select
    scored_docs.sort(key=lambda x: x['score'], reverse=True)
    
    selected_indices = []
    total_tokens = 0
    
    # Pass 1: Identify top docs within 1.025 T
    # Using token count directly as we have the encoder
    for d in scored_docs:
        if total_tokens + d['tokens'] <= 1.025 * T:
            selected_indices.append(d['idx'])
            total_tokens += d['tokens']
        else:
            break
            
    # 8. Refine selection based on h >= 0.5t
    high_score_indices = [d['idx'] for d in scored_docs if d['h'] >= 0.5 * t]
    high_score_tokens = sum(enc.encode(docs[i]).__len__() for i in high_score_indices)
    
    final_indices = []
    if high_score_tokens >= 0.7 * T:
        # Keep only high score docs and shuffle
        final_indices = high_score_indices[:]
        rng.shuffle(final_indices)
    else:
        # Hard fallback logic
        # Start with high score set
        current_indices = high_score_indices[:]
        current_tokens = high_score_tokens
        
        # Top up to 0.7 T with next-best clean docs
        if current_tokens < 0.7 * T:
            # Find docs that are NOT in high_score_indices but were not filtered
            # The current scored_docs is already sorted by score
            for d in scored_docs:
                if d['idx'] not in current_indices:
                    current_indices.append(d['idx'])
                    current_tokens += d['tokens']
                    if current_tokens >= 0.7 * T:
                        break
            rng.shuffle(current_indices)
            final_indices = current_indices
        else:
            # We have high_score_indices (which are already shuffled in the 0.7T case, 
            # but here we just use them as is or shuffle)
            rng.shuffle(current_indices)
            final_indices = current_indices

    # If final_indices total < T, repeat them
    # Calculate total tokens of the unique selected set
    final_tokens = sum(enc.encode(docs[i]).__len__() for i in final_indices)
    
    # The instruction says "repeat the kept list; do not bring in unselected pool docs"
    # This implies we fill until T is reached by repeating the sequence.
    # Since we need to return an ordered list that covers T tokens:
    # If total_tokens is 0, return an empty list or first doc.
    if final_tokens == 0:
        return []
    
    # Create the final sequence
    result = []
    while sum(enc.encode(docs[i]).__len__() for i in result) < T:
        result.extend(final_indices)
        # Safety break if repeating is impossible (shouldn't happen with valid docs)
        if len(result) > 1000000: break 
    
    # Final step: return in seeded random order
    # The instructions say "Return the selected indices in seeded random order (CURATE_SEED)"
    # This applies to the final list of indices that make up the training set.
    # Wait, the training reads "front to back". If I shuffle the result, I change the order.
    # Re-reading: "Return the selected indices in seeded random order" usually means
    # the final list should be shuffled. But "Training reads the returned order front to back"
    # means the order matters. I will shuffle the initial set and then repeat it 
    # to maintain the "seeded random" requirement while fulfilling the T tokens requirement.
    
    rng.shuffle(final_indices)
    
    # Re-calculate needed repetition to hit T exactly (or just past T)
    # and return the ordered list.
    res_final = []
    curr_t = 0
    # We need to be careful: the "ordered list" is what training reads.
    # If I shuffle at the end, I fulfill "return in seeded random order".
    # If I repeat, I fulfill "stops after exactly T".
    
    # Let's build the sequence:
    # To make it "ordered" for the trainer but "seeded random" for the selection:
    # 1. Take the best docs.
    # 2. Shuffle them.
    # 3. Repeat that shuffled sequence until T is met.
    
    # To be precise with "stops after exactly T", the repeat logic is:
    # If total_tokens_of_set < T, repeat the list until it exceeds T.
    # The trainer stops at T, so we don't need to truncate exactly to T, 
    # we just need to provide a list that contains at least T tokens.
    
    # The requirement "Return the selected indices in seeded random order" 
    # might mean the sequence itself is shuffled.
    
    # Let's stick to: Shuffle the valid set, then repeat to hit T.
    # The trainer cycles the list, so repeating is efficient.
    
    # Re-evaluating the "Return... in seeded random order" + "Training reads front to back"
    # This implies the result of select() should be a shuffled list of indices
    # such that the sum of lengths of docs at those indices is >= T.
    
    # Let's re-calculate the repeat count to be safe.
    # The trainer cycles, so we can just return the list once and it cycles.
    # However, the instructions say "repeat the kept list; do not bring in unselected".
    # This could mean repeat it in the list, OR it could mean the trainer cycles it.
    # "If the list holds fewer than T tokens it is cycled." 
    # This means we don't *need* to repeat in the list, but the instruction says "repeat the kept list".
    # I will repeat it in the list to be safe and satisfy both.
    
    final_indices_shuffled = final_indices[:]
    rng.shuffle(final_indices_shuffled)
    
    # Check if we need to repeat in the list to exceed T
    # (Though the trainer cycles, the instruction says "repeat the kept list")
    # I'll repeat until we exceed T to be safe.
    while sum(enc.encode(docs[i]).__len__() for i in final_indices_shuffled) < T:
        final_indices_shuffled.extend(final_indices_shuffled)
        
    return final_indices_shuffled
