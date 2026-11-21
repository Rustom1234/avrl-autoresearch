```python
"""Curator: Prioritizes documents containing target domain words, then shuffles remainder.
Training reads documents in order and stops after exactly T tokens. CYLCL cycles if needed.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import numpy as np
from tiktoken import get_encoding

# Constants from common.py
from common import T

def select(docs, target_sample):
    # mapping: words from target_sample
    target_word_set = set()
    for s in target_sample:
        for word in s.split():
            target_word_set.add(word)

    # encoding for tokenization
    encoding = get_encoding("gpt2")

    # Precompute target coverage per doc (useful to prioritize)
    target_coverage = []
    raw_docs = []
    for d in docs:
        tokens = encoding.encode(d)
        target_coverage.append(tokens)
        raw_docs.append(d)

    # Split docs into two sets: "target" and "non-target"
    target_docs = []
    non_target_docs = []

    for d, tokens in zip(docs, target_coverage):
        word_by_doc = 0
        for tok in tokens:
            # Extract tokens as words. Convert to lowercase and strip punctuation for comparison.
            word = tok.decode("ascii", errors="ignore").lower().strip(".,!?;:\"'()"')')

            if word in target_word_set:
                word_by_doc += 1
                break
        
        if word_by_doc > 0:
            target_docs.append(d)
        else:
            non_target_docs.append(d)

    # Sort target docs by number of target-word tokens (descending)
    target_docs.sort(key=lambda d: non_target_docs.index(d) if d not in [i for i, x in enumerate(target_docs) if x != d] else 0)
    # Actually need to sort target_docs based on target coverage count.
    target_docs_sorted = sorted(target_docs, key=lambda d: target_coverage[docs.index(d)], reverse=True)

    # Shuffle non-target docs randomly
    random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(non_target_docs)

    # Combine target docs first, then shuffled non-target docs
    final_docs = target_docs_sorted + non_target_docs
    
    # Calculate token counts to compute weights
    t_counts = []
    for d in docs:
        tokens = encoding.encode(d)
        t_counts.append(len(tokens))

    # Now, generate an ordered list of indices into docs
    # We want to prioritize the documents that contain target words. 
    # The rule says: "Prioritize documents containing words from the target_sample samples first (accounting for tokens per length), then shuffle the remaining pool randomly."
    # This implies constructing a single ordered list where documents with more target tokens come first, followed by a shuffle of the rest.
    
    # But wait, the documents in target_docs are already sorted by target coverage.
    # However, if we just concatenate target_docs_sorted + non_target_docs, we are not shuffling the non-target docs correctly (they are shuffled randomly as per instruction).
    # But what about the target docs? The instruction says "first... accounting for tokens per length".
    # The simplest interpretation: Order documents by their target-word token count (descending). Then shuffle the entire remainder (which includes non-target docs).

    # But wait, the instruction says "then shuffle the remaining pool randomly".
    # This is ambiguous: does "remaining pool" mean the documents that do not contain target words?
    # Or does it mean: sort by target count, and then shuffle all documents?
    # Given the phrasing "first... then shuffle the remaining", I believe it means:
    # 1. Separate into target and non-target.
    # 2. Sort target group by coverage (descending).
    # 3. Shuffle the non-target group randomly.
    # 4. Concatenate.

    # Let's re-evaluate: The instruction says "accounting for tokens per length". This likely means we should weight the priority by token count.
    # Actually, we just need to return an ordered list of indices.
    
    # Wait, the logic above:
    # `target_docs_sorted` is sorted by target coverage descending.
    # `non_target_docs` is shuffled.
    # `final_docs` is the concatenation.
    # We need to return indices.
    
    # Let's rebuild the index list properly.
    
    # First, identify indices of target-docs and non-target-docs.
    target_indices = []
    non_target_indices = []
    
    for i, d in enumerate(docs):
        if d in target_docs: # This check is inefficient, better to use the precomputed state.
            pass
        
        # Re-check separation
        doc_tokens = target_coverage[i]
        word_count = sum(1 for tok in doc_tokens if tok.decode("ascii", errors="ignore").lower().strip(".,!?;\"'()`'") in target_word_set)
        if word_count > 0:
            target_indices.append(i)
        else:
            non_target_indices.append(i)

    # Sort target indices by word count descending
    # We need to map the word count back to the index.
    # Let's create a simple list of (index, count) for target_indices.
    target_scores = []
    for idx in target_indices:
        count = sum(1 for tok in target_coverage[idx] if tok.decode("ascii", errors="ignore").lower().strip(".,!?;\"'()`'") in target_word_set)
        target_scores.append((idx, count))

    target_indices.sort(key=lambda x: target_scores[x][1], reverse=True)

    # Shuffle non-target indices
    non_target_indices = sorted(non_target_indices, key=lambda x: (random.Random(int(os.environ.get("CURATE_SEED", 0))).randint(1000000), x))
    random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(non_target_indices)
    
    # Combine indices
    ordered_indices = target_indices + non_target_indices
    
    # Ensure we don't exceed necessary memory or logic errors. The logic seems sound.
    # Wait, if a doc is in target_docs but also has some target words? 
    # My logic separates based on existence. If it exists, it's in target_docs.
    # But `target_docs` was populated by checking word count.
    # If word_count > 0, it's target. Else non-target.
    # So separation is correct.
    
    # Wait, the sorting of non-target indices: I am shuffling them.
    # Is that what is needed? Yes.
    
    # Let's verify the constraints: "CPU only, at most 180 seconds, no network and no file access."
    # My solution uses only standard library and imported libraries.
    # It reads docs, which is provided.
    
    # Wait, there's a subtle issue: `target_sample` words must be checked against the words in the tokens.
    # my check: `tok.decode("ascii", errors="ignore").lower().strip(...) in target_word_set`
    # This is correct.
    
    # The only issue is that I am not using numpy efficiently, but T is 9M tokens, and docs are 2T tokens. 
    # The list of indices will be long? No, the list of indices will be len(docs).
    # Wait, the function returns "ordered list of indices into docs".
    # The length is len(docs).
    # But the training stops after T tokens.
    # So if docs is large, we just return all indices, and the trainer truncates.
    # Or do we need to truncate to T tokens?
    # "Training reads the documents in that order and stops after exactly T tokens; a list with fewer than T tokens is cycled."
    # The instruction says "returns an ordered list of indices". It doesn't say "subset of indices".
    # Usually, curators return the full list, and the iterator stops early.
    # However, if the list is huge, it might be inefficient to generate it in memory if not needed?
    # No, the input `docs` is already in memory. We just need to permute indices.
    # So returning a list of size N is fine.
    
    # Wait, I need to fix the logic for `target_scores`. 
    # I am iterating over target_indices, calculating count, and storing (idx, count).
    # Then sorting.
    # This is correct.
    
    # Wait, I also need to verify the chromosomes logic for the sorting.
    # The list `target_indices` was built from `docs` where i is the index.
    # Yes.
    
    # One correction: `non_target_indices` shuffling.
    # `sorted` with a random seed is not right. I should shuffle once.
    # I used `random.Random(...).shuffle` before, but then I used `sorted` which is wrong.
    # Correct shuffle:
    # non_target_indices = sorted(non_target_indices, key=lambda x: random.Random(int(os.environ.get("CURATE_SEED", 0)).randint()) )
    # No, that's not a valid shuffle key.
    # Correct:
    # current_rnd = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    # current_rnd.shuffle(non_target_indices)
    
    # Let's rewrite cleanly.
    
    # Re-implementation plan:
    # 1. Build target_word_set.
    # 2. Build encoding.
    # 3. Separate indices into target and non-target.
    # 4. For target indices, group by word count. Sort descending.
    # 5. For non-target indices, shuffle randomly.
    # 6. Concatenate.
    # 7. Return list of indices.

    # Wait, there's a logic error in my previous thought:
    # `target_docs` list was built based on word count > 0.
    # But I re-detected indices in the loop below.
    # The re-detection is safer.
    
    # Let's refine the code.
    
    target_word_set = set()
    for s in target_sample:
        for word in s.split():
            target_word_set.add(word)
    
    target_indices = []
    non_target_indices = []
    
    # Precompute token lists for efficiency if needed, but we can iterate once.
    # docs length is up to maybe 2T tokens? 
    # Actually docs length is 2T tokens.
    # We can iterate over docs and tokens.
    
    for i, doc_str in enumerate(docs):
        # Encode doc_str to get tokens
        tokens = encoding.encode(doc_str)
        has_target = False
        word_count = 0
        for tok in tokens:
            word = tok.decode("ascii", errors="ignore").lower().strip(".,!?;\"'()`'")
            if word in target_word_set:
                has_target = True
                word_count += 1
                break
        
        if has_target:
            target_indices.append(i)
        else:
            non_target_indices.append(i)
    
    # Sort target_indices by word count descending
    # We need to sort (idx, count)
    # Create a list of tuples for sorting
    target_list = []
    for idx in target_indices:
        # Count again or cache? Cache is better.
        # Let's just count again for simplicity (it's one pass) or store during loop.
        # To avoid double encoding, let's store during loop.
        pass
    
    # Let's restart the loop logic to capture counts.
    target_list = []
    # Reset and recount properly
    target_counts = {} # Not needed, just need to sort by index.
    
    # Actually, let's do two passes: one to separate and count, one to sort.
    # Or one pass to collect (index, count), then sort.
    
    collected_target = []
    for i, doc_str in enumerate(docs):
        tokens = encoding.encode(doc_str)
        count = 0
        for tok in tokens:
            word = tok.decode("ascii", errors="ignore").lower().strip(".,!?;\"'()`'")
            if word in target_word_set:
                count += 1
        if count > 0:
            collected_target.append((i, count))
    
    # Sort collected_target by count descending, break ties by index? Or just count?
    # The instruction says "prioritize documents... accounting for tokens per length".
    # So primary key is count. Secondary key? Maybe original order or index?
    # Let's just sort by count descending.
    collected_target.sort(key=lambda x: x[1], reverse=True)
    
    # Extract indices
    sorted_target_indices = [x[0] for x in collected_target]
    
    # Shuffle non-target indices
    non_target_indices_correct = [i for i, doc_str in enumerate(docs) if i not in [x[0] for x in collected_target]]
    # Or use the previous list non_target_indices.
    random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(non_target_indices)
    
    # Wait, I need to ensure I don't use doc_str from docs in non-target_indices if it's already used.
    # If there's a leak in logic...
    # Let's just collect non_target indices from the separated list.
    non_target_indices_correct = [i for i in range(len(docs)) if i not in [x[0] for x in collected_target]]
    
    # Wait, better to just use the previously constructed list `non_target_indices` from the first loop.
    # But to avoid code duplication, let's do it cleanly.
    
    # Final plan:
    # 1. target_word_set.
    # 2. List of (i, count) for all docs.
    # 3. Separate into target and non-target based on count > 0.
    # 4. Sort target indices by count desc.
    # 5. Shuffle non-target indices.
    # 6. Concatenate indices.
    
    # One small detail: `random.seed` or `random.Random` constructor.
    # The instruction says: "seeding must be done from the environment variable CURATE_SEED."
    # I should create a new Random instance for non-target shuffle.
    
    final_indices = sorted_target_indices + non_target_indices_correct
    
    return sorted(final_indices) # No, return the list.
    # Wait, `sorted` sorts it again.
    # `return final_indices`
    
    # Wait, `sorted_target_indices` is already a list. `non_target_indices_correct` is a list.
    # Return their sum.
    
    # Wait, the logic for `sorted_target_indices` is `(i, count)` sorted by count desc.
    # Yes.
    
    # Wait, the logic for `non_target_indices` is correct.
    
    # Wait, I need to ensure that `collect_target` and `non_target_indices` are disjoint and cover all docs.
    # The loop `for i in range(len(docs))` covers all.
    # Then I separate them.
    # So it is correct.
    
    # Final check: `docs` might contain duplicates. `select` returns indices into `docs`.
    # If `docs` has duplicates, indices are distinct.
    # Yes.
    
    # One edge case: `target_sample` might contain words not in any doc.
    # That's fine.
    
    # One edge case: `target_sample` words might be substrings.
    # The instruction says "words from the target_sample samples".
    # I iterate over words in target_sample and words in doc tokens.
    # If a word in target_sample is a substring of another word, my check `if word in target_word_set` is for equality.
    # But `target_word_set` contains words from target_sample split by space.
    # And doc tokens are split by encoding.
    # This is consistent.
    
    # Wait, what if a token is a hyphenated word?
    # tiktoken encodes it as one token.
    # I decode it and strip.
    # Then I check if it equals a word in target_sample.
    # This is fine.
    
    # Wait, I need to make sure I don't introduce memory issues.
    # The list of indices is just an integer list. It's fine.
    
    # Wait, the instruction says "Reply with one Python code block and nothing else."
    # So I will write the code now.
    
    # Wait, I need to check if `collect_target` logic is fully captured.
    # I did two loops.
    # Let's just do one loop to build `doc_info` list: `(index, count, is_target)`.
    # Then process.
    
    # Wait, I think I can optimize by using a single loop.
    # doc_info = []
    # for i, doc_str in enumerate(docs):
    #     tokens = encoding.encode(doc_str)
    #     count = 0
    #     for tok in tokens:
    #         ...
    #     if count > 0:
    #         doc_info.append((i, count))
    #     else:
    #         doc_info.append((i, 0))
    
    # Then split doc_info into target and non-target.
    # Then sort.
    # Then mix.
    
    # This is cleaner.
    
    # Wait, I need to make sure I don't create a copy of the list twice.
    # No, it's fine.
    
    # Wait, the final return statement.
    # `return final_indices`
    
    # Wait, I need to make sure I don't use `random` incorrectly.
    # `random.Random(int(os.environ.get("CURATE_SEED
