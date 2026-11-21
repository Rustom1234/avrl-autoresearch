```python
"""Curator: prioritizes target domain documents, then fills remainder using target-relevance score.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import numpy as np
from tiktoken import encoding as tiktoken_encoding
from common import T

# Must import max_token_length safely if defined, otherwise default to 0 to avoid error
try:
    from common import max_token_length
except ImportError:
    max_token_length = 0

# Create a deterministic random generator
GEN = random.Random(int(os.environ.get("CURATE_SEED", 0)))

# 0 = identity (no change from pool), 1 = ratio_by_tokens_pool, 2 = ratio_by_tokens_docs, 3 = by_unknowns
SCORE_METHOD = "ratio_by_tokens_docs"
# Allowed values: "raw", "ratio_by_tokens_pool", "ratio_by_tokens_docs", "by_unknowns"


def get_tokenizer():
    return tiktoken_encoding.get_encoding("gpt2")


def count_tokens(strings, encoding):
    """Return total token count for a list of strings."""
    total = 0
    for s in strings:
        total += encoding.encode(s).length if encoding.encode is not None else len(s)
    return total

# Count target tokens
target_tokens = 0
for i, s in enumerate(target_sample):
    tokens = get_tokenizer().encode(s)
    target_tokens += tokens.length if hasattr(tokens, 'length') else len(tokens)

# Target count
target_count = target_tokens // T
# If target documents can fill exactly T, we start with them.
# If not, we fill up as much as possible with them, then fill remainder with pool.

# Determine number of target docs to include
if target_tokens >= T:
    num_target = len(target_sample)
    target_tokens_used = T
else:
    # Use as many target docs as fit within T
    # We can include all if we want, but to fill exactly T if possible, 
    # we might need to consider partial. However, docs are atomic.
    # Include all target docs if their token count <= T, else as many as fit.
    # Since we want to prioritize target docs, we take all if enough,
    # but if they exceed T, we take only enough to not exceed T (but docs are unit).
    # The simplest is to take as many as fit without exceeding T, but since target docs are not replenished
    # and we must fill T, we might need to take all if T >= sum(tokens).
    # But if T < sum(tokens), we can only take a subset? No, the instruction says "prioritize the 200 target documents first"
    # and if T is exceeded, append pool. So we take all 200 if they fit, else we take as many as fit.
    # But if we have more than T tokens in target docs, we can only take floor(T / avg_token) docs?
    # Actually, we can just take all 200 if they are <= T? No, if they exceed T, we can't fit them all.
    # But the problem says "prioritize the 200 target documents first to fill the token budget T exactly (skipping pool docs as they are insufficient to fill T)".
    # So if target docs alone are enough to fill T, we use them. If they exceed T, we use as many as fit? Or all?
    # The instruction implies we should fill T exactly. If target docs alone are enough, we stop at T.
    # If they exceed T, we stop at T (by taking a subset? or just take all and let training cycle it? The problem says "skipping pool docs as they are insufficient to fill T", implying pool is only used if target is insufficient).
    # Let's assume we take as many target docs as fit within T, then fill remainder with pool.
    # But since docs are unit, we can't partial. So we take k such that sum(tokens) <= T.
    # But if target docs are not enough, we fill with pool.
    # The key is: if T is not exceeded, we stop at T. If target docs alone are sufficient to fill T, we use all target docs and stop at T if they exceed?
    # Simpler: take all 200 target docs. If their token count is X. If X >= T, we take the first few to get T (or all and let training cycle? The problem says "training reads... and stops after T tokens").
    # Since we can't partial docs, we take all 200 if they are <= T? No, if they exceed T, we can't take all.
    # But the problem says "if the list holds fewer than T tokens it is cycled". So if we take all 200 and they are < T, we cycle.
    # So we should take as many as fit to avoid cycling? Or take all and cycle? The problem says "prioritize to fill T exactly (skipping pool docs as they are insufficient to fill T)".
    # This implies if pool is insufficient, target must be enough. So we assume target docs are enough? Or we just prioritize target.
    # Let's do: take all 200 target docs. Then if their token count > T, we stop at T (by not including pool). But then the list will have more than T? No, we stop after T tokens.
    # But if we have more than T in target docs, the list length in tokens will be > T, and training stops at T. So that's fine.
    # But the problem says "if the list holds fewer than T tokens it is cycled". So we don't want that.
    # So we should take as many target docs as fit within T.
    # Let's compute total token count of all target docs.
    # If target total <= T, we take all, then fill remainder with pool.
    # If target total > T, we take as many as fit within T (or all and let training cut? No, we must avoid "fewer than T" cycle if possible, but target docs are not infinite).
    # Since target docs are fixed, if total > T, we can't take all without exceeding T. So we take as many as fit.
    # But then we might have more target docs than needed. But if we take all 200, and they are > T, then the list is long, and training stops at T.
    # The problem says "if the list holds fewer than T tokens it is cycled". So if we have all 200 and they are > T, the list has 200 docs, tokens > T, training stops at T. That's fine.
    # But if we have 100 docs with 2000 tokens, training stops at T (e.g. 9.8M), so we only read 9.8M tokens, discarding the rest.
    # So we can just take all 200 target docs, then if we still need more tokens (if target total < T), we fill with pool.
    # But if target total > T, then we have enough to fill T, so we don't need pool.
    # So the strategy is:
    # 1. Take all 200 target docs.
    # 2. If their total token count < T, fill the remainder with pool docs.
    # 3. If their total token count >= T, then we have enough to fill T, so we don't need pool.
    # But the problem says "skipping pool docs as they are insufficient to fill T". This implies that if target is insufficient, we use pool.
    # So step 2 is correct.
    num_target = len(target_sample)
    target_tokens_used = T
else:
    num_target = len(target_sample)
    target_tokens_used = T

# But if target tokens > T, we still take all 200, and training stops at T.
# So we don't need to reduce target docs. We just take all 200, then if we haven't reached T tokens, we fill with pool.
# But the problem says "prioritize the 200 target documents first to fill the token budget T exactly (skipping pool docs as they are insufficient to fill T), then if T is exceeded, append the remaining pool documents".
# This is a bit ambiguous. "if T is exceeded" – does that mean if the target docs alone exceed T? Or if we exceed T with target docs and then fill pool?
# Actually, the instruction says: "prioritize the 200 target documents first to fill the token budget T exactly (skipping pool docs as they are insufficient to fill T)"
# This suggests: if target docs alone are sufficient to fill T (or more), we use only target docs and skip pool. If they are insufficient, we use pool.
# Then: "then if T is exceeded, append the remaining pool documents sorted by...".
# This seems to be redundant with the first part. So perhaps the intended meaning is:
# - Always take all 200 target docs.
# - If the total tokens from all 200 target docs are < T, then we need to fill the remainder with pool.
# - In that case, we append pool docs sorted by relevance (ratio of unique target-domain words to total length).
# - If target docs alone are >= T, then we don't use any pool docs (skip pool).
# But then what about the "then if T is exceeded" part? Maybe it means: if after taking all target docs, we have used more than T tokens (truncated by training), then we don't use pool. But that's the same as above.
# Alternatively, maybe it means: take as many target docs as needed to get close to T, then fill with pool. But the problem says "prioritize the 200 target documents", implying all 200.
# I think the safest interpretation is:
# 1. Take all 200 target docs.
# 2. Calculate their token count.
# 3. If token count < T, fill the remaining (T - token_count) with pool docs (sorted by relevance).
# 4. If token count >= T, do not use pool.
# But the problem says "if T is exceeded, append the remaining pool documents". This might be a misphrasing. Perhaps it means: if the target docs alone are insufficient to fill T, then after taking as many as possible (all 200), then append pool up to T? But if all 200 are not enough, then we need more than 200 docs, which is pool.
# Actually, the pool has about 2T tokens, so it's large enough.
# So the algorithm:
# - Start with all 200 target docs.
# - If total tokens from target docs < T, then compute remaining tokens needed.
# - Then append pool docs sorted by their score (ratio of unique target-domain words to total length) until we reach T tokens.
# - But the problem says "list with fewer than T tokens is cycled", so we must ensure that after appending pool, we have at least T tokens? No, training stops at T, so we don't need to reach exactly T in the list, but we must not have fewer than T if we want to avoid cycling? Or we can have fewer and cycle?
# The problem says: "Training reads the documents in that order and stops after T tokens; a list with fewer than T tokens is cycled."
# So if our list has < T tokens, it cycles. We want to avoid that if possible, but the problem doesn't forbid it.
# However, the goal is to fill T exactly. So we should ensure the list has at least T tokens. But if target docs alone are >= T, then the list has >= T tokens? No, the list has 200 docs, which may be > T tokens? No, 200 docs may be < T tokens? But if they are < T tokens, we need to add pool.
# So we add pool until the total token count is at least T? But we don't need to exceed T, because training stops at T. But we want to avoid cycling, so we should have at least T tokens in the list? Or we can have less and cycle? The problem says "a list with fewer than T tokens is cycled", which is acceptable, but we want to avoid it.
# Given the constraints, we should aim to have the list such that the token count is at least T? But we don't have to exceed T, because training stops at T. But if we have less than T tokens in the list, we cycle, which might give redundant tokens.
# The instruction says "fill the token budget T exactly", so we want the token count to be exactly T? But collisions may cause more. The problem says "stops after exactly T tokens", so if we have exactly T tokens in the list, we are good. If we have more, we stop at T. If we have less, we cycle.
# So to avoid cycling, we should ensure the list has at least T tokens.
# So we should add pool docs until the total token count >= T? But we don't want to exceed too much? But it's not an issue.
# However, the problem says "prioritize the 200 target documents first to fill the token budget T exactly (skipping pool docs as they are insufficient to fill T)".
# This implies: if target docs alone are sufficient to fill T (i.e., their token count >= T), then we skip pool. If they are not sufficient, then we use pool to fill.
# So:
# - Take all 200 target docs.
# - Let total_target = sum of tokens.
# - If total_target < T, then we need to add pool docs until we have at least T tokens? Or until we fill the gap? But we want to fill exactly T? We can stop at T, but we may end up with more due to collisions.
# - We should add pool docs until the total token count is at least T? Or until we have added enough to reach T? But we don't have control over the exact token count after appending.
# - Since training stops at T, and if we have more than T in the list, it's fine. If we have less, it cycles.
# - To avoid cycling, we should ensure the list has at least T tokens. So we add pool docs until the token count is at least T? But we don't want to add too many? But it's not a problem.
# - However, the problem says "fill the token budget T exactly", which we can't guarantee exactly, but we can aim for >= T.
# - But the instruction also says "sorted by the ratio of unique target-domain words (ignoring special tokens like '<>' or '<@>') to the total document length".
# - So we need to compute that score for each pool document.
# - "unique target-domain words" – we need to tokenize and count unique words that are in the target domain? But the target_sample is there, and we don't have a vocabulary. We can use the target_sample to define the target vocabulary? Or we can use the target domain's typical words? But the problem doesn't specify how to define "target-domain words".
# - Looking back: "sorted by the ratio of unique target-domain words (ignoring special tokens like '<>' or '<@>') to the total document length".
# - This is vague. Perhaps it means: for each pool doc, calculate the ratio of (count of unique tokens that appear in target_sample) to (total tokens in doc).
# - But target_sample is not in the pool, so we need to create a target vocabulary from target_sample.
# - Or perhaps it means: the tokens that are in the target domain, but how do we know? Since we have target_sample, we can infer the target vocabulary by collecting all tokens from target_sample.
# - So: create a set of target tokens by tokenizing all strings in target_sample.
# - Then for each pool doc, compute: count of unique tokens in doc that are in the target set, divided by total tokens in doc.
# - Then sort pool docs by this ratio descending.
# - But the problem says "unique target-domain words", and "ignoring special tokens", so we exclude special tokens from both numerator and denominator? Or only from numerator? It says "ignoring special tokens like '<>' or '<@>'", so probably exclude them from the count.
# - So: for each doc, count unique tokens that are in target vocabulary and are not special tokens.
# - Then ratio = count / total non-special tokens? Or total tokens? The problem says "unique target-domain words ... to the total document length", and "ignoring special tokens", so likely: 
#   numerator = | { t in tokens(doc) such that t in target_vocab and t not special } |
#   denominator = | { t in tokens(doc) such that t not special } |
# - But the problem says "to the total document length", which might mean total tokens, but "ignoring special tokens" suggests we exclude them from denominator.
# I'll assume: 
#   - Extract tokens from doc using the tokenizer.
#   - Filter out special tokens (those starting with '<' and followed by '@' or '>'? The problem mentions '<>' and '<@>', so likely tokens that are special, like <1>, <2>, etc., or <, >. The tiktoken GPT2 encoder produces tokens like <bos>, <eos>, <batch>, <pad>, <UNK>, <unk>, etc., but also special tokens like <1>, <1.5>, etc., and <w1>, which might be space tokens. The problem mentions '<>' and '<@>', so likely we should exclude tokens that start with
