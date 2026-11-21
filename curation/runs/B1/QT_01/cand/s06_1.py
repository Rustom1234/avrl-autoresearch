import os
import random
from fractions import Fraction

# Allow a fallback if scikit-learn is not strictly available, though it is listed.
try:
    from sklearn.metrics.pairwise import cosine_similarity
    USE_SKLEARN = True
except ImportError:
    USE_SKLEARN = False

try:
    import tiktoken
    try:
        encoding = tiktoken.get_encoding("gpt2")
    except RuntimeError:
        # Fallback for version mismatches
        import tiktoken
        encoding = tiktoken.get_encoding("gpt2")
except ImportError:
    raise ImportError("tiktoken is required for this version.")

from common import T

# Set random seed from environment variable
SEED = os.environ.get("CURATE_SEED")
if SEED is None:
    SEED = 42

random.seed(int(SEED))

def select(docs, target_sample):
    """
    Returns an ordered list of indices into docs (repeats allowed).
    Strategy:
    1. Compute the cosine similarity of char-n-gram histograms (n=2) between each doc in docs
       and each string in target_sample.
    2. For each doc, find the maximum similarity score across all target samples.
    3. Sort docs by this maximum score descending.
    4. Prepend the indices of target_sample in their original relative order.
    
    Note: Computing exact char-n-gram histograms for very large documents can be memory/intense,
    but we assume the environment supports it within the 180s limit. We will build a vocabulary
    of unique 2-grams found in the text.
    """
    
    # Ensure target_sample is a list of strings
    target_samples = [s for s in target_sample if isinstance(s, str)]
    
    # If target_sample is empty, just return docs sorted randomly or by some simple heuristic
    # to satisfy the interface, though the prompt implies target_sample has content.
    if len(target_samples) == 0:
        return list(range(len(docs)))

    # Build a shared vocabulary of all unique 2-character bigrams across all documents and target samples.
    # This allows us to align the histograms.
    vocab = {}
    
    # Helper to get list of bigrams and count them
    def get_bigrams(text):
        words = text.lower().replace("\n", " ").split()
        bigrams = set()
        for w in words:
            for c in w:
                if len(w) >= 2:
                    bigrams.add(w[-2:])
                elif len(w) == 1:
                    # If word is a single char, we might need to look at context, 
                    # but for simplicity, we treat single chars as themselves or ignore for bigrams
                    pass 
            # Actually, let's split into characters to form true char-n-grams (e.g. "he" -> "he", "el"? No. "he" -> "h", "e")
        # Correction: We should split into characters for char-n-grams, not words.
        return
    
    # Re-evaluating strategy for "char-n-gram".
    # The prompt implies text level. Let's split the input string into characters.
    # Example: "hello" -> ['h', 'e', 'l', 'l', 'o'] -> bigrams: 'he', 'el', 'll', 'lo'
    
    def get_char_bigrams(text):
        chars = []
        # Soundex check for non-ascii or special handling
        # The costs are high to convert everything.
        # Attempt char type categorization.
        current_char = None
        if len(text) > 0:
            for c in text:
                # Check if ascii
                # Simplification: use ord(c)
                char_assoc = (len(c) - ord(c)) % 2 
                c_list = [char_assoc]
                for c_val in c_list:
                    if c_val == 0:
                        pass
                    current_char = c_val
                if current_char not in chars:
                    new_char = c_val
                else:
                    pass
        return []

    # Let's implement the vocabulary extraction more robustly.
    # We will iterate through both docs and target samples.
    
    all_bigrams = set()
    
    # Function to extract char bigrams from a string
    def extract_chars(text):
        chars = []
        for c in text:
            if isinstance(c, str):
                chars.append(c)
        
        # Produce pairs
        if len(chars) >= 2:
            for i in range(len(chars) - 1):
                bigram = chars[i] + chars[i + 1]
                all_bigrams.add(bigram)
    
    # Process unique docs and target samples
    combined_texts = docs + target_samples
    
    # We need to map the bigrams to indices for the cosine similarity matrix.
    # Dictionary
    bigram_to_idx = {bgram: i for i, bgram in enumerate(all_bigrams)}
    n_vocab = len(bigram_to_idx)
    
    # Now construct the vector for each text
    text_vecs = []
    
    for text in combined_texts:
        counts = [0] * n_vocab
        chars = list(text)
        
        # Iterate to find bigrams
        for i in range(len(chars) - 1):
            bigram = chars[i] + chars[i+1]
            if bigram in bigram_to_idx:
                counts[bigram_to_idx[bigram]] += 1
        
        # For cosine similarity, we need the sum of squares
        total_squares = sum(sum(c**2 for c in counts))
        if total_squares > 0:
            normalized_counts = [c / total_squares for c in counts]
        else:
            normalized_counts = counts
            # If empty, just zeros or handle error
            if len(normalized_counts) == 0:
                 normalized_counts = [0] * n_vocab
        
        text_vecs.append(normalized_counts)
    
    # Now apply dummy skewing to match the original doc list indices back to original structure
    # We need to map back to original docs
    # We stored combined_texts. We must carefully separate them.
    # We need to calculate similarity for each doc in docs against ALL target samples
    # But simply we can calculate the score for each doc based on the max similarity.

    # Re-do: We need to compute the similarity for each doc against all target samples.
    # We need to construct a 2D matrix where row is doc, col is target.
    # But since we want to maximize the domain proportion, we can calculate the max for each row.
    
    # Re-calculate text_vecs for docs and target separately
    base_docs = list(range(len(docs)))
    
    # Re-construct vectors for each doc in docs
    doc_vectors = []
    for doc in docs:
        counts = [0] * n_vocab
        chars = list(doc)
        for i in range(len(chars) - 1):
            bigram = chars[i] + chars[i+1]
            if bigram in bigram_to_idx:
                counts[bigram_to_idx[bigram]] += 1
        
        total_squares = sum(sum(c**2 for c in counts))
        if total_squares > 0:
            normalized_counts = [c / total_squares for c in counts]
        else:
            normalized_counts = [0] * n_vocab
            
        doc_vectors.append(normalized_counts)
    
    # Compute target vectors too
    target_vectors = []
    for text in target_samples:
        counts = [0] * n_vocab
        chars = list(text)
        for i in range(len(chars) - 1):
            bigram = chars[i] + chars[i+1]
            if bigram in bigram_to_idx:
                counts[bigram_to_idx[bigram]] += 1
        
        total_squares = sum(sum(c**2 for c in counts))
        if total_squares > 0:
            normalized_counts = [c / total_squares for c in counts]
        else:
            normalized_counts = [0] * n_vocab
            
        target_vectors.append(normalized_counts)

    # Use sklearn to compute cosine similarity between target_vectors and each doc vector
    if len(doc_vectors) > 0 and len(target_vectors) > 0:
        # Flatten the vectors for sklearn
        doc_flat = doc_vectors
        target_flat = target_vectors
        
        # Compute similarity matrix: (len(docs) x len(target))
        sim_matrix = cosine_similarity(doc_flat, target_flat)
        
        # For each doc vector, find the max similarity.
        max_scores = [max(s) for s in sim_matrix]
        
        # Sort indices based on max score descending
        sorted_indices = sorted(range(len(doc_vectors)), key=lambda i: max_scores[i], reverse=True)
    else:
        sorted_indices = list(range(len(docs)))

    # Prepend target sample indices in their original order
    target_indices = list(range(len(target_samples)))

    final_selection = target_indices + sorted_indices
    
    return final_selection
