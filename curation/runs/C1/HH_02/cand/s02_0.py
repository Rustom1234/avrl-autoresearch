"""Format-split curator: wiki-format docs first, then everything else. Seeded shuffle within each group.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os, random, re, hashlib

_HEADING = re.compile(r"^= .+ =\s*\n")
_JUNK_PHRASES = ("click here", "discount code", "coupon")


def _is_junk(doc):
    low = doc.lower()
    if any(p in low for p in _JUNK_PHRASES):
        return True
    if doc:
        non_ascii = sum(1 for ch in doc if ord(ch) > 127)
        if non_ascii / len(doc) > 0.05:
            return True
    return False


def _dedup_key(doc):
    normalized = " ".join(doc.lower().split())
    return hashlib.sha1(normalized.encode("utf-8", "replace")).hexdigest()


def _is_wiki_format(doc):
    if not _HEADING.match(doc):
        return False
    return " @-@ " in doc or doc.count(" , ") >= 5


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))

    # Drop junk and exact duplicates (keep first occurrence).
    seen = set()
    kept = []
    for i, doc in enumerate(docs):
        if _is_junk(doc):
            continue
        key = _dedup_key(doc)
        if key in seen:
            continue
        seen.add(key)
        kept.append(i)

    # Split by format test.
    group_a = [i for i in kept if _is_wiki_format(docs[i])]
    group_b_set = set(group_a)
    group_b = [i for i in kept if i not in group_b_set]

    # Seeded shuffle within each group, A then B.
    random.Random(seed).shuffle(group_a)
    random.Random(seed).shuffle(group_b)

    return group_a + group_b
