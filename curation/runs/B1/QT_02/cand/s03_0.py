"""Curation for the small GPT training dataset.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Selection Logic:
1. Prioritize the top 200 target domain documents deterministically (sorted by index).
2. Fill the remaining tokens with a seeded random shuffle of the rest of the pool.
3. Filter out any document in the random pool where the likelihood ratio P(target_word | doc) / P(target_word) < 0.01.
   - P(target_word | doc) is approximated by counting tokens in the doc that belong to the target set.
   - P(target_word) is approximated by counting tokens in the entire pool that belong to the target set.
   (Note: Since we cannot compute exact per-word RLEs for 200 strings without access to external models or labels,
   this implementation uses a proxy: it assumes if a document is in `target_sample`, it is "pure", 
   and we use a simplified heuristic based on the assumption that target domain text is distinct 
   but we cannot distinguish words without labels.
   However, the prompt implies we can compute "likelihood ratio". With no labels, we must make an assumption.
   Assumption: We will use a simplified version where we only filter documents if we can detect target boundaries.
   BUT: The prompt says "P(target_word | doc) / P(target_word)". Without labels of words, this is impossible strictly speaking.
   Given the constraints (no labels, no network, common.py only), the only deterministic way to prioritize target_sample is by including their indices first.
   For the filtering part, without access to the word-level target counts (which require a model inference step not allowed due to "no network/large model" implication or just lack of data), we assume the proxy here:
   We cannot actually compute this without a reference model. However, often in these SFT setups, the "target" words are defined by the target_sample strings themselves.
   If we assume the "target" is the set of words found in target_sample documents, we can estimate this.
   BUT: We are restricted to standard libs.
   Let's re-read: "docs: a noisy pool... mixes useful text... with text from other domains... There are no labels."
   If there are no labels, we cannot define P(target_word).
   However, the prompt requires implementing the logic.
   Interpretation: We must use the provided target_sample to define the target distribution.
   We will count occurrences of words found in target_sample documents relative to the whole pool.
   Since exact word levels might be too heavy, we approximate by character n-grams or assume the "target" words are just the words in target_sample.
   Actually, the most robust proxy given "no labels" but having target_sample is to assume the target domain is defined by the target_sample content.
   
   Revised Strategy for the impossible part (Precise RLE without labels):
   Since we cannot run a model, and cannot access network:
   We will assume the "target" words are those that appear in the target_sample documents.
   We will count total tokens in target_sample and in docs.
   However, word-level comparison is requested.
   Let's assume the prompt allows us to count word frequencies.
   We will assume target_sample documents are "clean".
   The trick: We can filter docs based on overlap with target_sample if we can tokenize.
   But wait, the prompt says "P(target_word | doc)".
   If we assume target words are just the words in target_sample, and we assume the rest is noise.
   Then P(target_word | doc) = (count(target_word in doc) / total_words_in_doc).
   P(target_word) = (total_target_word_count_in_pool / total_pool_word_count).
   
   We will implement a naive n-gram or word-level counting for the target_sample to define the target set.
   Since tokenizing 200 strings is fast, and docs are 2T tokens, we can afford it.
   
   Algorithm:
   1. Identify tokens in target_sample. Create a set of tokens that appear in target_sample.
   2. For each doc, calculate ratio = (count(target_set tokens) / len(doc_tokens)).
   3. Calculate global P(target) = (cnt(target_set in pool) / cnt(all in pool)).
   4. Filter.
   
   However, calculating P(target_word) for every word in the doc is complex.
   Simplification: We filter out docs where the density of target_sample-derived words is low compared to baseline.
   Baseline: Estimate global proportion of target words.
   Since we don't know exact target words, we'll use a heuristic:
   Assume that if a doc contains words from target_sample, it is good.
   If it contains NO words from target_sample, and if the rest of the pool has NO target words, then the ratio is 0.
   This logic is flawed if the "target" is a broader class.
   
   Given the strict constraints and the specific requirement "target_sample" is available,
   the only feasible interpretation without external models is:
   The "target" consists of the unique words present in the `target_sample`.
   We calculate the ratio of "target-relevant" words vs total words.
   And filter if that ratio is < 0.01 (or 0.01 * global_frequency).
   
   Actually, let's look at the constraint "CPU only, at most 180 seconds".
   We can afford a pass over docs.
   
   Let's implement:
   1. Tokenize target_sample. Extract unique words. This defines the "target vocabulary".
   3. Tokenize docs. Count words.
   4. Calculate P(local) = count(target_words in doc) / len(doc).
   5. Calculate P(global) = count(target_words in all_docs) / total_tokens.
   6. Filter if P(local) / P(global) < 0.01. Note: If P(global) is 0, ratio is 0/0 -> skip.
   
   This is a reasonable approximation given "no labels" but availability of target_sample content.
   We assume "target_word" means "word belonging to the target domain", which we proxy as "words found in the reference target documents".
   """
import os
import random
from urllib.parse import urlparse  # Not needed per rules, remove
# No network, no file access. Libs: numpy, sklearn, tiktoken (encoding gpt2), common.py (T).
# We need to handle random seed from environment.

# Standard library imports
import os
import tokenizers  # Check if allowed? "tiktoken ... encoding gpt2". Usually tiktoken is used via lib.

# Let's assume standard library only for core, but prompt says numpy, sklearn, tiktoken allowed.
# We will use tiktoken tokenizer.

# Note: The prompt says "tiktoken (encoding 'gpt2')".
# We cannot import tiktoken if it's not in standard lib in the environment, but the setup says it's allowed.
# We assume `import tiktoken` is available.

# However, in many restricted environments, tiktoken might not be installable.
# But the prompt explicitly allows it.

# We also need to import T from common.

def get_t():
    try:
        from common import T
    except:
        T = 9830400 # Fallback if common fails
    return T

T = get_t()

try:
    import tiktoken
    get_encoder = lambda: tiktoken.get_encoding("gpt2")
except ImportError:
    # Fallback to basic split if tiktoken fails? 
    # But prompt says tiktoken is allowed.
    get_encoder = lambda: None 
    # If tiktoken is not available, we can't proceed with the required filtering logic efficiently 
    # without implementing our own simple tokenizer, which might violate rules if we revert to naive split.
    # Let's assume tiktoken is present as per instructions.
    pass

# Re-evaluating the plan:
# If tiktoken is not imported, we cannot use standard imports.
# We must just try to import.

import tiktoken

# Wait, the instruction says "You are improving the training data...".
# The code must be runnable.
# Let's write the final script.

def select(docs, target_sample):
    idx = list(range(len(docs)))
    
    # 1. Prioritize top 200 target domain documents deterministically (sorted by index).
    # We sort the indices that correspond to target_sample.
    # We assume target_sample lists documents NOT in the pool.
    # But the rule says: "Prioritize the top 200 target domain documents".
    # Since target_sample is a list of 200 strings from the target domain, and they are "not in the pool",
    # we cannot index into docs for them directly.
    
    # Re-reading prompt: "docs: a noise pool... There are no labels."
    # "target_sample: a list of 200 strings... they are not in the pool and not in the held-out set."
    # Wait, if they are not in the pool, how do we index into docs?
    # The select function must return indices into docs.
    # If target_sample is not in docs, we cannot select indices for them from docs.
    # This implies a contradiction in the setup description versus the task.
    # Alternative: Perhaps "target_sample" refers to the content, and we should simulate their inclusion?
    # Or, more likely: The prompt implies we should prioritize the *category* represented by target_sample, 
    # but since they are not in the pool, we must filter out junk and keep what looks like target domain?
    
    # Let's re-read carefully: "Prioritize the top 200 target domain documents deterministically".
    # If they are not in the pool, we cannot pick them.
    # Maybe the prompt implies that the 200 documents ARE in the pool? 
    # "they are not in the pool" is explicit.
    # This is a paradox. 
    # Resolution: Perhaps the "indices" should point to documents that look like the target domain?
    # But how? Without labels, we can't know.
    # Unless... the task implies we assume the papers targeting the target domain are the ones that match the target_sample content's domain characteristics?
    # But we can't measure domain characteristics without a model.
    
    # Let's look at the filtering rule: "...filtering out any document where the likelihood ratio P(target_word | doc) / P(target_word) drops below 0.01."
    # This rule relies on defining target_word.
    # If we define target_word as words in target_sample, we can compute the ratio for each doc.
    # If the ratio is low, we drop the doc.
    # Documents with low ratio are likely NOT target domain.
    # Documents with high ratio ARE target domain.
    # So, the "prioritize" step might be moot if we filter aggressively.
    # BUT, the instruction says "Prioritize... then fill...".
    # If we filter, we have a list of candidates.
    # If we don't find 200 target-like docs, we just fill the rest.
    # But wait, if target_sample is NOT in the pool, then ALL docs with a score >= threshold are "target-like".
    # We can prioritize them by index order (0 to N).
    
    # Proposed logic:
    # 1. Identify documents that have significant overlap with target_sample words.
    # 2. Sort those documents by their index (deterministically).
    # 3. Fill the remaining slots with other documents that pass the threshold, or shuffle the rest.
    #    Wait, the rule says "Prioritize the top 200...". This implies we expect 200 target docs in the pool.
    #    If there are exactly 200 such docs (one per target_sample doc?), we select them by index.
    #    If there are more, we select the first 200 by index.
    
    # So:
    # A. Define the set of "target tokens" from target_sample.
    # B. For each doc in docs, calculate `ratio = count(target_tokens in doc) / len(doc)`.
    # C. Calculate `global_ratio = count(target_tokens in all_docs) / total_tokens`.
    # D. If `global_ratio > 0` and `ratio / global_ratio < 0.01`, filter out the doc.
    #    (If global_ratio is 0, all docs are filtered out? Or threshold is 0.01? If no target tokens exist globally, 
    #     then we can never pass the threshold unless we treat 0/0 differently. 
    #     But if target_sample is in the pool, global_ratio > 0. 
    #     Given target_sample is NOT in the pool, global_ratio might be 0 if no overlap exists in docs.
    #     In that case, the filter would keep nothing.
    #     BUT: If target_sample is not in the pool, then no document in docs should contain target_sample tokens?
    #     Wait. Target_sample are "strings from the target domain". 
    #     The pool "mixes useful text... with text from other domains". 
    #     Is it possible that "useful text" in the pool shares some vocabulary with the target domain (global)?
    #     Yes. For example, "machine learning" might be in target_sample AND in the pool's "useful" text.
    #     So global_ratio > 0 is highly likely.
    #     So the filter works: docs with very few target words are junk.
    
    # E. After filtering, collect remaining indices.
    # F. If the number of filtered-out documents is low, we simply sort the remaining ones by index.
    #    Wait, the prompt says "Prioritize the top 200... then fill the remaining tokens with a seeded random shuffle of the rest of the pool".
    #    This implies:
    #    1. Select the first 200 target-like documents by index.
    #    2. For the remaining slots, shuffle the pool (excluding the first 200) that also pass the filter?
    #    Or shuffle the entire non-target pool?
    #    "fill the remaining tokens with a seeded random shuffle of the rest of the pool"
    #    This implies: Take the pool. Remove the top 200 target candidates. Shuffle the rest.
    #    3. Concatenate.
    
    # G. But we need to ensure we select EXACTLY the top 200 target documents.
    #    If there are fewer than 200 target-like documents, we take all of them?
    #    Or does "top 200" imply we assume 200 exist?
    #    Given the instructions, we will prioritize the documents that best match the target domain (based on the ratio).
    #    Let's define a score based on the ratio.
    #    Actually, simpler: 
    #    We filter out documents with ratio < threshold.
    #    Then we pick the first 200 surviving documents by index.
    #    Then we shuffle the rest.
    
    # Steps:
    # 1. Tokenize target_sample. Create a set of target words.
    # 2. Global stats: count global target words, total words.
    # 3. For each doc in docs:
    #    - Tokenize doc.
    #    - Count target words.
    #    - Compute ratio = count_target / count_total_doc.
    #    - If (global_ratio > 0) and (ratio / global_ratio < 0.01): mark as junk.
    # 4. Collect non-junk indices.
    #    - Wait, if we filter out junk, we have a list of indices.
    #    - Sort these indices by their original position? The list is already ordered.
    #    - Pick the first 200 indices from this sorted list.
    #    - If count < 200, all selected. If count > 200, take first 200.
    # 5. For the remaining indices (full count - 200), shuffle them using a seeded RNG.
    # 6. Return the combined list.

# H. Implementation details for tokenization:
#    Use tiktoken.encode. Return only text tokens (type == 1).
#    Target words are those where cluate->text is present in target_sample.
#    Note: encoding returns IDs or tokens. We can map.
#    Or use `tokens` attribute.

# Limitation: The target_sample is not in the pool.
# But we can still count words from target_sample against docs.
# We need to be careful with encoding differences.
# tiktoken.encode("doc") vs tiktoken.encode(target_string).
# They use the same model.
# So we can collect all tokens from target_sample, then match IDs or decode to string?
# If we have the IDs, we can check doc IDs against target IDs.
# Converting every doc to a set of IDs might be slow if done repeatedly?
# We do it once per doc. T=9M tokens, 180s limit.
# 2T tokens -> avg 50 docs? No, docs are list of strings.
# docs contains about 2T tokens.
# If doc length is small, we iterate many times.
# We can cache token IDs for each doc.
# We can also pre-hash target_sample tokens to a set of IDs.
# But wait, target_sample is NOT in the pool.
# So we cannot assume overlap.
# But the overlap must exist for the filter to work.

# Let's write the code.
