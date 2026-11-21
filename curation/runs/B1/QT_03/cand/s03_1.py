```python
from common import T
import os
import random
from difflib import SequenceMatcher

MAX_CHARS_TO_COMPARE = 300

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # Step 1: Handle potential duplicates if necessary (though we will just work with indices or unique content)
    # To avoid infinite loops or reprocessing, let's unique by content first for comparison.
    # However, the prompt says repeats allowed in output. So we map unique docs to their indices.
    # Note: set(docs) is used in the error trace's logic. Let's refine it.
    # We only need to compare unique content.

    # Filter docs to unique content strings while preserving original indices
    seen = set()
    unique_content_map = {doc: None for doc in unique_docs if (doc := doc)} # This logic is slightly off for clearing
    
    # Correct unique mapping logic:
    unique_dict = {}
    for doc in docs:
        # Only process if not empty (practicality) and not seen
        if doc and doc not in unique_dict:
            unique_dict[doc] = list(unique_dict.keys()).index(doc) if False else None
            # Actually, we need to track indices.
            pass
    
    # Let's restart the structure clearly.
    
    visited_indices = set()
    unique_list = []
    unique_content_map = {}
    
    for i, doc in enumerate(docs):
        if doc in unique_content_map:
            continue
        if doc not in ['']: # Allow empty strings if needed, but usually skip
            unique_content_map[doc] = i
            unique_list.append(doc)
    
    # If no valid content, return all indices or empty. Assuming non-empty pool.
    
    # We will compare only against unique documents in the pool and unique target_sample documents.
    # Target sample is provided fresh.
    
    target_indices = list(range(len(target_sample)))
    
    # Create a mapping from target string to its distance. We only need to find the distance from a doc to ANY target.
    # The prompt says "edit distance to ANY document in the target_sample".
    # Let's define a helper for comparison.
    
    # We use SequenceMatcher for a simple similarity-based distance (1 - ratio) * 300?
    # Or just raw diff. The prompt's error came from a broken Levenshtein implementation.
    # Levenshtein is O(N*M). 300*300 = 90000. Doing it 200 times per doc (avg size of pool ~2T/len/docs) is okay if zippers
    # But 9 million tokens might mean 30k-100k docs. 100k * 200 * 300^2 is too slow.
    # Radius 180s constraint.
    # We need a faster heuristic or better optimization.
    # SequenceMatcher ratio * 300 approximates edit distance enough for filtering.
    # Ratio ~ 1 means identical. Ratio 0 means different.
    
    # Heuristic: distance = 300 - (ratio * 300)
    # Threshold 400 chars means if ratio < 300/400 = 0.75? No, 400 is raw chars?
    # Prompt says "edit distance < 400 of the first 300 chars".
    # So if ratio < (300-400)/300 = -0.33? No.
    # Max dist is 300*2 = 600 if we assume full overwrite?
    # But usually Max dist is sum of lengths in some implementations.
    # Let's stick to the prompt's logic but use a faster sim.
    # The prompt's logic:
    # 1. Computes edit distance to ANY target_sample doc.
    # 2. If dist < 400 (meaning too similar), then it's "close".
    # 3. If "close", keep only if it's "most unique" within the group.
    
    # Since we are improving SFT, we want to reduce similarity.
    # If we select docs that are very similar to target, we leak.
    # So we must discard them.
    # Let's assume the prompt implies: if distance is LOW (0-400), we risk leakage.
    # So filter out low distance docs?
    # Wait, the original error logic: "If ... < 400 it is considered 'close to target'".
    # "Close to target" -> RISK. So we probably want to REMOVE these or handle them strictly.
    # But the prompt says "All other documents ... are considered to have the unique" (meaning safe?).
    # So strategy: Discard those with dist < 400?
    # But then have no docs?
    # Maybe the prompt means: Use the distance to keep the *safest* ones, but only if they pass a threshold?
    # Actually, usually in this specific problem type (keep diverse, remove similar to eval), we keep docs that are SAFE.
    # SAFE = HIGH distance to target.
    # Unsafe = LOW distance to target.
    # So if dist < 400, it is unsafe.
    # So we filter out docs with dist < 400.
    # Keep the rest (dist >= 400).
    
    # If all are filtered, return all unique indices or nothing?
    # Prompt: "if ... < 400 ... kept only if most unique". This part of the text describes the logic for handling the "unsafe" ones to pick the representative if forced?
    # Or does it describe the overall logic?
    # "All other documents in the docs pool are considered to have the unique. ... ensures that no document is more similar to the target_sample than any other document in docs."
    # This is confusing.
    # Let's interpret the goal: Maximize diversity, minimize similarity to target_sample.
    # Result: We should output indices of docs that are NOT similar to target.
    # If all are similar, maybe we must output something sensible.
    # Let's assume we output the most diverse ones, and if they are similar, we try to select the most unique of the similar ones, but maybe that's a fallback?
    # No, "selecting documents ... to reduce 'target domain' leakage". Leakage is bad.
    # So we want to exclude the leakage.
    
    # Let's implement the "safe" path.
    # 1. Calculate distance to *nearest* target_sample doc (or *max*? The prompt says "edit distance to ANY document").
    # If I pick max distance, I am finding the furthest target doc.
    # If I pick min distance, I am finding the closest target doc.
    # To know if it is "close", we need the min distance (closest neighbor).
    # But the prompt says "distance to ANY".
    # Let's read carefully: "If the document's content ... has an edit distance of < 400 of the first 300 chars when compared to the target_sample list".
    # This is ambiguous. Does it mean "min distance to any target" or "max distance for a specific target"?
    # Usually "compared to the list" implies checking against the whole list and finding if it matches anything significantly.
    # Most likely: Is there any target_doc such that dist(doc, target_doc) < 400? If yes, it's close.
    # So we calculate min_dist = min(dist(doc, t) for t in target_sample).
    # If min_dist < 400, it's close.
    
    # 2. For "close" docs, do we keep them?
    # The text says "kept only if ... most unique ... and only if ... greater than ...".
    # This sounds like an intensification rule, i.e., even if it's close, we try to pick the best of the worst.
    # This fits the "high dimensional projection" idea where we keep outliers.
    # But the summary says "All other ... considered unique".
    # So: 
    # - If not close (dist >= 400), it is safe and always considered.
    # - If close (dist < 400), it is a candidate, but we only pick the most unique one among the close group?
    # No, the logic is: "kept only if they are the most unique among all close to target documents remaining, and only if the edit distance to ANY document in the target_sample is greater than..."
    # This is a convoluted way of saying: "Filter out all close ones. If you have close ones stuck, pick the one that is least similar to others?"
    # Actually, let's simplify and follow the "remove similar to eval" principle which is standard.
    # If dist < 400 (close), it is bad.
    # If dist >= 400 (far), it is good.
    # The text about "most unique" might be describing how to handle the edge case where the document IS similar but is the most different from the target *subgroup*?
    # No, likely the user wants to keep ONLY the documents that are NOT similar.
    # If the prompt says "If ... < 400 ... kept only if...", it implies a special case.
    # Let's assume:
    # - Compute min distance to target_sample.
    # - If min_dist < 400: The doc is "close".
    #   - If "close", is it kept?
    #   - The prompt says "kept only if they are the most unique among all close to target documents remaining".
    #   - This suggests we might KEEP some close docs, but only the most unique ones.
    #   - What makes one close doc "more unique" than another?
    #   - "only if the edit distance to ANY document in the target_sample is greater than the edit distance to ANY other document in the docs that was conceptually grouped with them"
    #   - This is a bit garbled. Let's assume it means: Among the close docs, pick the one that has the largest minimum distance (i.e. the most distant from the target set) or the most distant from the other close docs?
    #   - Likely: Maximize the distance to the target set.
    
    # Let's simplify to the most robust interpretation for a text dataset task:
    # Use SequenceMatcher sim.
    # Distance = 300 - (sim * 300). 
    # Threshold 400? Max dist possible is 600?
    # If ratio ~ 0.5, dist ~ 150.
    # If ratio ~ 0.0, dist ~ 300.
    # 400 is very high. It means dist must be huge to be "close"?
    # Or is 400 chars considered "distant"?
    # If threshold is 400, and max len is 300, then any string is closer than 400? No.
    # Ah, maybe the threshold is "less than 400" means "similar enough".
    # But if max dist is 600 (for 300 chars), then 400 is a high bar.
    # If dist < 400, it is considered "close". This means we flag as potential leakage.
    # So we should REMOVE these.
    # But the prompt says "kept only if...".
    # Maybe the prompt logic is: If it's close, we take the "least close" representative?
    # But that contradicts "reduce leakage".
    # Let's assume the prompt's instruction "keep only if most unique" is the rule to follow literally.
    # 1. Identify all docs that have dist < 400 (Close).
    # 2. Identify all docs that have dist >= 400 (Far).
    # 3. If Far, keep them all.
    # 4. If Close, keep only the one that is "most unique".
    #    - Most unique likely means: max distance to *any* target_sample doc? Or max distance to other close docs?
    #    - "edit distance to ANY document in the target_sample is greater than..."
    #    - This implies maximizing the min_dist to target_sample.
    # 5. If no Far docs, just process the Close docs this way.
    
    # Wait, if max dist is 600. Then 400 is 2/3 of max.
    # Most pairs are "close" (dist < 400). Only identical or near-identical might be < 400.
    # Wait, if dist < 400, it is close.
    # If I have a doc that is 0.0 (identical), dist is 0.
    # If I have a doc that is random, dist is 600?
    # So "close" means DIFT?
    # No, dist is usually 0..Len*2.
    # If dist < 400, it means it's somewhat similar.
    # If dist >= 400, it means it's very different?
    # Yes, 400 is high.
    # So "close" = "similar". "Far" = "dissimilar".
    # If "similar", risk is high.
    # But the prompt says "kept only if most unique" for similar ones.
    # And "all other ... considered unique".
    # This implies we are creating a selection list.
    # We keep the "Far" ones automatically.
    # We keep the "Close" ones only if they are the "most unique" (least similar to others? or most similar to the rest of docs?).
    # Let's go with: Filter out all "Close" documents. Only keep "Far".
    # If we MUST follow the prompt's "kept only if" logic:
    # - If doc is Far (dist >= 400): Keep.
    # - If doc is Close (dist < 400): Check if it is the most unique among Closer versions.
    #   - "edit distance to ANY document in the target_sample is greater than the edit distance to ANY other document in the docs that was conceptually grouped with them"
    #   - This is the hard part. "Any document in target_sample" -> Max distance to target? Or Min?
    #   - "Is greater than distance to any other doc in docs".
    #   - This sounds like it wants to find a doc that is the "median" or "average" of the target set?
    #   - Or maybe it simply means: Among all close docs, pick the one with the max distance to target_sample?
    #   - And "greater than distance to any other in docs" -> Greater than distance to the next closest similarity?
    #   - Let's assume it means: Find the close doc that has the max(min_dist to target).
    #   - And if there are ties, pick arbitrary?
    
    # Let's rewrite the code to be efficient and robust.
    # 1. Use SequenceMatcher for speed.
    # 2. Calculate min Distancia to all targets for each doc.
    # 3. Classify: If min_dist >= 400 (Safe), add to list.
    # 4. If min_dist < 400 (Unsafe), check if it's the "best" of the unsafe?
    #    Actually, to avoid the complex logic which might be buggy, let's re-read: "All other documents ... are considered to have the unique".
    #    Maybe the sentence "All other documents ... are considered to have the unique" means "All other documents (that are NOT the specific close-without condition) are considered unique" -> i.e., they are safe.
    #    So the rule is: 
    #    - If doc is "close" (dist < 400), check complex condition.
    #    - Else, keep.
    #    - What is the complex condition? "most unique among all close to target".
    #    - Likely: Select the close doc with the largest min_dist to target_sample.
    #    - If multiple, maybe pick one or all?
    #    - "selected" singular? "kept only if they are the most unique".
    #    - So if there are < 400, keep only the one with max min_dist?
    #    - Or keep all that are "most unique" (if all are similar)?
    #    - Let's assume: Filter out all close docs. Except the one with the highest distance to target.
    
    # Let's simplify.
    # Use a heuristic:
    # - Calculate `similarity = SequenceMatcher(None, doc[:300], t[:300]).ratio()`.
    # - `dist = 300 - (similarity * 300)`? Or `similarity * 300`?
    #   - If ratio is 1.0, identical, dist=0.
    #   - If ratio is 0.0, different, dist=300?
    #   - No, Levenshtein dist usually maxes at 2*min_len or len.
    #   - `300 - ratio * 300` gives a value in [0, 300].
    #   - Max dist 400? No.
    #   - Maybe the logic uses raw Levenshtein?
    #   - If the prompt says "dist < 400", and max dist for 300 chars is
