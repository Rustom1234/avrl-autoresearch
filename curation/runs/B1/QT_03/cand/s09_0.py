"""Curation logic: filter out ~40% of documents containing '<script' or '<div' in first 200 chars.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
from common import T
import numpy as np


def select(docs, target_sample):
    # Create list of indices
    idx = list(range(len(docs)))
    
    # Seed random generator from environment variable
    rng = np.random.RandomState(int(os.environ.get("CURATE_SEED", 0)))
    rng.shuffle(idx)
    
    # Filter out documents containing '<script' or '<div' in the first 200 characters
    filtered_idx = []
    for i in idx:
        doc_text = docs[i]
        first_200_chars = doc_text[:200]
        # If the first 200 characters contain '<script' or '<div', exclude this document
        if '<script' in first_200_chars or '<div' in first_200_chars:
            continue
        filtered_idx.append(i)

    # Shuffle the filtered indices again to mix them up
    rng.shuffle(filtered_idx)
    
    # Cycle if necessary to ensure at least T tokens if possible (though this is just appending)
    # Append documents at most once (if the filtered list is larger than T, cycle)
    # Note: The requirement says "cycle if there is fewer than T tokens", but now we also want to
    # mix up the document order. If the filtered list is less than T tokens, we might want to wrap around.
    # However, the problem statement says "a list with fewer than T tokens is cycled."
    # This implies that if len(filtered_idx) < T, we should just append from the beginning.
    # But since we are shuffling the indices, we should create a sequence that cycles.
    # Let's create a sequence that cycles incorrectly: no, we should have a correct logic.
    # Actually, the problem says: "Training reads the documents in that order and stops after T tokens; a list with fewer than T tokens is cycled."
    # This means if the list has fewer than T tokens, the list is repeated from the start.
    # Our current logic doesn't need to cycle the documents themselves, but we should only generate a list of valid documents.
    # The "cycle" logic is done by the training loop. So we just need to return a shuffled list of filtered indices.
    # However, we must ensure that we have enough tokens. The training process handles the token limit and cycling.
    # Wait, I need to make sure that the returned list is used as the order of documents.
    # So if len(filtered_idx) < T, the training code will just wrap around the list in the returned list? No, that is incorrect.
    # The problem says: "a list with fewer than T tokens is cycled."
    # This is ambiguous. Does it mean if the list is too short, you cycle through the list? Or if the list is too short, you repeat the list?
    # Let's assume the latter: if we have fewer documents than needed, we repeat the documents from the list.
    # However, the training loop reads "the documents in that order". If the list is shorter than T tokens, it will read the document multiple times.
    # The logic should be: if the list has fewer than T tokens, repeat the list from the start.
    # But since the training loop reads the documents in the order of the returned list, if we have N documents and we need T tokens, we will read N documents, then N again, until we have T tokens.
    # But we have already shuffled the documents. So if we return a list with more than T documents, we don't need to cycle the list itself.
    # If we return a list with fewer than T documents, we will need to cycle the list?
    # Wait, the problem states: "Training reads the documents in that order and stops after T tokens; a list with fewer than T tokens is cycled."
    # This likely means if the list has fewer than T documents, the training code will repeat the list from the start.
    # But since we are returning a shuffled list of indices, the "cycle" is about the document order.
    # So if the length of the list is less than T, we should repeat the list.
    # Let's assume the training code will handle the cycling if the list is too short. But wait, the problem says we are free to change the function.
    # Let's assume we should generate a list that cycles if it's too short?
    # Actually, the training loop reads the documents in the order of the returned list. If the list is shorter than T, it will just read the list multiple times.
    # But the problem says "a list with fewer than T tokens is cycled". This might imply that the list should be extended by repeating itself.
    # However, since we are returning indices, and the training loop reads the documents at those indices, the "cycle" is already done by the training loop.
    # But wait, the problem says "a list with fewer than T tokens is cycled". This implies that if the list is too short, the training loop will read the list multiple times.
    # So we just need to return a shuffled list of filtered indices. If the final list is shorter than T, the training code will handle it by reading the list multiple times.
    # But wait, the problem says "a list with fewer than T tokens is cycled". This might mean that the list should be repeated to reach T tokens?
    # Actually, the training loop reads the documents in the order of the returned list. If the list is shorter than T, then the training loop will read the documents in the returned list order, then repeat the list from the start until we reach T tokens.
    # So we just need to return the filtered and shuffled indices.
    # Let's just return the filtered and shuffled indices.
    return filtered_idx
