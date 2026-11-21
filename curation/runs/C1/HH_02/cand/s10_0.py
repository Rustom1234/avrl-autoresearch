"""Experiment 5 curator with exact-duplicate removal first.

Step 0: drop exact duplicates. Each doc is keyed by its whitespace-normalized,
lowercased text, and only the first occurrence in pool order is kept.

Group A: docs starting with a "= Title =" heading line, or containing " @-@ ".
Group B: all other docs.

Output order: a prefix of shuffled B, then all of shuffled A. The B prefix is
grown until B tokens + A tokens >= T, so the trainer reads exactly T tokens and
never cycles. Because whole documents are indexed, the total can overshoot T by
less than one B document; the trainer's T-token cutoff trims that overshoot from
the front of A, so A remains the final segment read. If A alone exceeds T, the
last whole documents of A that fit within T tokens are kept.
"""
import hashlib
import os, random, re

import tiktoken
from common import T

_HEADING = re.compile(r"^= .+ =\s*\n")


def _dedup_key(text):
    normalized = " ".join(text.lower().split())
    return hashlib.sha1(normalized.encode("utf-8")).digest()


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    enc = tiktoken.get_encoding("gpt2")

    def ntok(s):
        return len(enc.encode_ordinary(s))

    # Step 0: drop exact duplicates (normalized), keeping first occurrence.
    seen = set()
    keep = []
    for i, d in enumerate(docs):
        k = _dedup_key(d)
        if k in seen:
            continue
        seen.add(k)
        keep.append(i)

    group_a, group_b = [], []
    for i in keep:
        d = docs[i]
        if _HEADING.match(d) or " @-@ " in d:
            group_a.append(i)
        else:
            group_b.append(i)

    random.Random(seed).shuffle(group_a)
    random.Random(seed).shuffle(group_b)

    a_tok = {i: ntok(docs[i]) for i in group_a}
    total_a = sum(a_tok.values())

    if total_a > T:
        # Keep the last whole docs of A that fit within T tokens.
        out, s = [], 0
        for i in reversed(group_a):
            n = a_tok[i]
            if s + n > T:
                break
            out.append(i)
            s += n
        out.reverse()
        return out

    need = T - total_a
    prefix, s = [], 0
    for i in group_b:
        if s >= need:
            break
        prefix.append(i)
        s += ntok(docs[i])

    return prefix + group_a
