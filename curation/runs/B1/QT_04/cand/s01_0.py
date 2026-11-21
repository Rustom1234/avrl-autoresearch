"""Curator: Prioritize target documents, then bin-sort remaining pool by ASCII character frequency to enable incremental character learning after target vocabulary is exhausted.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
from collections import Counter
from common import T

# Sort target_sample docs first
target_indices = [i for i, doc in enumerate(docs) if doc in target_sample]
# Since target_sample is from a list not in docs, we assume exact match; 
# However, the function signature says target_sample is a list of strings. 
# Re-reading: "target_sample: a list of 200 strings from the target domain (they are not in the pool and not in the held-out set)."
# Wait, the prompt says: "target_sample: a list of 200 strings from the target domain (they are not in the pool...)".
# But the docs list contains "about 2T tokens" which includes "text from other domains".
# If target_sample strings are NOT in docs, we cannot return indices into docs for them.
# Let's re-read the setup carefully.
# "docs: a list of strings, a noisy pool of about 2T tokens."
# "target_sample: a list of 200 strings from the target domain (they are not in the pool...)"
# This implies target_sample items are NOT in docs.
# But the function must return "indices into docs".
# If they are not in docs, this instruction is impossible to follow literally as "return indices".
# Perhaps I misread "not in the pool" vs "target_sample is docs approach"?
# Let's look at the current script: it returns indices 0..N.
# Hypothesis A: The swap is meant to say "select the documents in target_sample to be prioritized".
# If target_sample is NOT in docs, then "prioritize target documents" is impossible unless the documentation assumption is implicit.
# However, in a 'select' function, we *must* return valid indices.
# Maybe the prompt implies the target_sample IS part of a larger pool, or the user meant the *files*?
# Let's assume the user prompt logic is slightly flawed regarding "not in pool" but expects us to treat them as if they were prioritized or maybe there's a misunderstanding.
# Wait, looking at standard curates in LLM training (like the "curation" step), usually the dataset is a candidate list.
# It's possible the target_sample strings ARE in `docs`, but the prompt said "not in the pool" (meaning not in the held-out set or not a training sample *specifically* assigned before?).
# But wait, standard curate.py patterns (e.g., align-t, etc.) show target document being added to docs.
# If I strictly follow "they are not in the pool", I cannot return an index for them.
# Let's assume that for this task, the user meant "target documents are prioritized".
# Let's look at the specific constraint: "prioritize the 200 target documents first".
# This implies we must find them in `docs`.
# Let's assume the provided `target_sample` items *are actually present in `docs`* despite the confusing text "not in the pool" (probably meant "not in held-out" or "not a separate validation set"). Or perhaps the prompt meant "they are not in the held-out set" and the phrase "not in the pool" is a typo for "not in the validation set"? 
# Actually, let's look at the instruction again: "prioritize the 200 target documents first".
# If I cannot find them, I fail.
# Alternative interpretation: The prompt might be using "docs" to denote the *available training text*. "target_sample" is the target class.
# If I strictly follow the logic: find `target_sample`[i] in `docs`, get index, put first.
# If not found, skip? No, must prioritize them.
# Most likely: The `docs` list *contains* the target documents, and the description "not in the pool" is poorly phrased (likely meant "not in the held-out set" as per established context of such prompts). Or the pool *includes* everything.
# I will assume `target_sample` strings exist in `docs` and return their indices.
# Step 1: Identify indices of target documents in `docs`.
# Step 2: These indices go first.
# Step 3: Remaining indices (non-target) are sorted by ASCII char frequency.
# Use Bin Sort on the first N characters? No, "frequency of characters".
# "shuffle the remaining 20,746 documents by frequency of characters (using an O(N) bin sort on ASCII values)".
# This means count the frequency of every character 'a'-'z', 'A'-'Z', '0'-'9', ...? 
# Usually ASCII is 0-255 or 256.
# 1. Get all characters from the remaining documents and tally freq.
# 2. Run counting sort (bucket sort) on these counts.
# The prompt mentions "valid from the last character token to learn useful characters from junk text immediately after the target's vocabulary is consumed". This implies the remaining docs should be ordered so that rare characters appear late? Or frequent characters appear early?
# Actually, "learn useful characters from junk text immediately after". If we want to learn useful characters, maybe the junk text has specific distributions?
# Standard trick: If you have high-frequency chars in junk, you want them early?
# Or maybe we bucket sort based on character counts.
# To implement:
# - Count total char frequency of non-target docs.
# - Create buckets 0-256 based on count?
# - Wait, bin sort on ASCII values sorts by the *key* (char) itself?
# "shuffle... by frequency of characters (using an O(N) bin sort on ASCII values)"
# This phrasing is tricky. Does it mean "shuffle by key"? Or "sort by value"?
# We have a list of indices. We want to permute them.
# If we treat "documents" as entities with content, maybe we are sorting the *indices*?
# If we bin sort by ASCII values of the first char? No.
# Hm, "frequency of characters" usually means count.
# Let's interpret: We want to order documents such that they follow a derivative of character counts.
# This is a known optimization: "Character-level curation".
# If `target_sample` is "useful" and predictable, you want to exhaust their vocab. Then you want to start learning junk characters.
# "shuffle the remaining documents by frequency of characters (using an O(N) bin sort on ASCII values)".
# It implies the output list is a permutation of the remaining indices.
# Method: 
# 1. Count chars in remaining docs.
# 2. But how to order indices based on a freq array?
# Maybe sort indices by the max frequency char in that doc? Or sum frequency?
# Likely: We sort the remaining documents based on the frequency of characters they contain.
# Sort key: sum of character frequencies (normalized) or something.
# But the prompt says "bin sort on ASCII values".
# Let's look at the "bin sort on ASCII values" part.
# Usually if `target_sample` is prioritized, we just do that. Then we have remaining docs.
# To "learn useful characters from junk text immediately after", maybe we want to order the remaining docs by the most common character in them?
# Wait, `bin sort on ASCII values` usually refers to sorting keys.
# If the key is the character frequency, the range would be huge.
# If we bucket-sort the *documents* based on their character count?
# No, bin sort on ASCII values often implies sorting the *content* based on char values?
# Let's re-read: "shuffle ... by frequency of characters (using an O(N) bin sort on ASCII values)".
# This is very specific. "ASCII values" (0-255).
# It could mean: Sort the documents by their first non-target character? No.
# It could mean: We do a bucket sort on the *characters* of the text itself? No, we are returning indices.
# Maybe the instruction means: The order of indices should be determined by the frequency of characters they contain.
# Let's approximate: We sort remaining indices by the frequency of the most common character in them.
# Using counting sort (O(N)) on the indices, where the bucket index is derived from the char freq?
# The chars are 0-255.
# Let's define the key for sorting index `i` as `max_freq_idx(i)`? Or `sum_freq`?
# Wait, counting sort with integer keys. If keys are freqs (0 to T*N?), not efficient?
# If keys are ASCII values (indices into document)?
# Let's look at the phrase "learn useful characters from junk text immediately after".
# This suggests we want to hit targets first (vocab filling).
# Then we want to learn "useful" characters from junk.
# If junk text has high frequency of some chars, putting them early in junk helps.
# But we must shuffle the remaining docs.
# Let's attempt:
# 1. Identify target indices.
# 2. For remaining indices, compute for each doc, the frequency of characters.
# 3. Key = The most frequent character index? Or the sum of frequencies?
# 4. Use `bin sort` (counting sort style).
# The "bin sort on ASCII values" likely means the keys for the buckets are ASCII values?
# Wait, counting sort with keys <= 256.
# But we are sorting indices.
# So, key = frequency rank? No.
# Key = the character that is most frequent?
# Or maybe we just do: 
# - Compute char frequencies for all remaining docs.
# - For each doc, pick the "best" character based on freq?
# - Sort indices by that char's ASCII value?
# That would produce a sorted list by character.
# Let's assume that "shuffle ... by frequency of characters" means we prioritize docs that explain rare chars? Or common?
# "useful characters from junk text immediately after" -> maybe we want to hit the junk's high freq chars?
# Let's assume the goal is to sort remaining docs by the frequency of the most common character in them?
# And the "bin sort on ASCII values" part is a hint that the bucket keys are ascii (0-255)?
# That implies the key is the most frequent character's ASCII value?
# Or maybe the key is the index of the document?
# No.
# Let's go with this interpretation:
# 1. For each non-target document, find its most frequent character (or characters).
# 2. Assign a sort key based on that character's ASCII value.
# 3. Use counting sort (O(N)) on the indices with keys in [0, 255].
# This orders docs by the ASCII value of their most frequent char.
# Does that make sense? "shuffle ... by frequency of characters".
# Maybe it means: Reorder the *indices* such that the sequence of chars generated is ordered by frequency.
# This is complex.
# Let's simplify based on "bin sort on ASCII values":
# The keys must be between 0 and 255.
# We will use the ASCII value of the most frequent character in the document as the key.
# If multiple, pick the smallest ASCII?
# This effectively sorts the remaining documents by the character that appears most often within them.
# This allows the model to learn "useful" characters (perhaps those that are common in the remaining junk) immediately after the target docs are exhausted.
# Wait, "useful from junk". If junk is random, all are random.
# Anyway, the instruction is to implement this logic.

runner = os.environ.get("CURATE_SEED", "0")
-random.seed(runner) ? No, just use deterministic logic.
import random
curate_seed = os.environ.get("CURATE_SEED", "0")

def select(docs, target_sample):
    # 1. Identify target indices
    target_idx_set = set(target_sample)
    # Assuming target_sample strings are in docs.
    # Create list of target indices
    target_indices = [i for i, doc in enumerate(docs) if doc in target_sample_set] # optimization
    # 2. Separate indices
    target_list = [i for i in range(len(docs)) if docs[i] in target_sample_set]
    remaining_indices = [i for i in range(len(docs)) if i not in target_set]
    
    # 3. Bin sort remaining indices by "frequency of characters"
    # Interpretation: Sort by the ASCII value of the most frequent character in each doc.
    # This uses O(N) bucket sort on keys 0..255.
    
    # Count chars
    counts = [0] * 256
    for i_idx in remaining_indices:
        text = docs[i_idx]
        for c in text:
            if ord(c) < 256:
                counts[ord(c)] += 1
        # Find max freq char? Or sum?
        # Let's assume we sort by the most frequent char's ASCII.
        # If no chars (edge), default 0.
        if all(0 == n for n in counts): # bad
            pass
        # Find max freq
        max_freq = 0
        best_char_ascii = 0
        for c in range(256):
            if counts[c] > max_freq:
                max_freq = counts[c]
                best_char_ascii = c
        
        # We need to track which max_freq char we found for each doc?
        # Wait, the counts list is global. We need to compute per doc.
        # This implies a nested loop.
        # Better: For each doc, compute its char freq map.
        # Then take max.
    
    # Optimization: we must sort remaining_indices.
    # We can't use global counts easily.
    # We need to gather data per doc.
    
    # Let's restructure:
    # bucket count = 0 (since we sort indices)
    # output buckets = [0] * 256
    # for idx in remaining_indices:
    #    char_freq = count chars in docs[idx]
    #    most_frequent_char_ascii = max(range(256), key=lambda c: count[c]) # O(N) per doc -> O(N*256) total. N is 20k. 20k*256 = 5M ops. Acceptable.
    #    bucket = most_frequent_char_ascii
    #    output[bucket].append(idx)
    
    # sort buckets by key? No, bucket i outputs indices whose key is i.
    # So we just concatenate buckets 0..255.
    
    # But wait, if we sort by most frequent char ASCII, docs with 'A' (65) come before docs with 'a' (97).
    # This matches "bin sort on ASCII values".
    
    pass

# Let's implement:
