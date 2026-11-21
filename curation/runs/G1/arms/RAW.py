"""Arm RAW of gate G1: a fixed list of documents (built by curation/skills/gate1.py) in a seeded random order."""
import json, os, random


def select(docs, target_sample):
    idx = json.load(open('/homes/rdubash/active-audit/curation/runs/G1/arms/RAW.json'))
    random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(idx)
    return idx
