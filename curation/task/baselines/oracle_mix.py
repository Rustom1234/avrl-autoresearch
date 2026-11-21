"""Label oracle for graded pools: keep the listed kinds, shuffled, repeated to fill T. Analysis only.

CURATE_MODE = "mix:kind+kind+..."            epochs of the union, shuffled
CURATE_MODE = "mix:kind+kind+...:tail=kind"  the same, but documents of the tail kind are held back and fill the last
                                             part of the stream (one pass of everything else comes first)
Labels come from env CURATION_ORACLE_LABELS and are never available to a real curator.
"""
import json, os, random


def select(docs, target_sample):
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    parts = os.environ["CURATE_MODE"].split(":")
    kinds = parts[1].split("+")
    tail = parts[2].split("=")[1] if len(parts) > 2 else None
    kind = [l["kind"] for l in json.load(open(os.environ["CURATION_ORACLE_LABELS"]))]
    assert set(kinds) <= set(kind), "unknown kind in %s" % kinds

    def epochs(ids, n):
        out = []
        for _ in range(n):
            ids = list(ids); rng.shuffle(ids); out += ids
        return out

    if tail:
        return epochs([i for i, k in enumerate(kind) if k in kinds and k != tail], 1) + \
            epochs([i for i, k in enumerate(kind) if k == tail], 8)
    return epochs([i for i, k in enumerate(kind) if k in kinds], 8)
