"""L6a critic, part 3 (CPU, read-only): three-way split of the 12-batch deficit of the verdict-fold best arm vs carry_plain (seeds 8-15, mean errors per seed):
(1) calibration phase batches 1..k (carry_plain - arm), (2) after calibration, continued refit (carry_plain - freeze_plain_n, same n, same k), (3) after calibration, rule quality at equal labels (freeze_plain_n - arm)."""
import json, math
import numpy as np
R = "/homes/rdubash/avrl_live/FULL_R/avrl/runs/L6a/"
J = {"3B": "Qwen2.5-3B-Instruct", "7B": "Qwen2.5-7B-Instruct-AWQ"}
BEST = {("3B", 62): "amort_seed_500", ("3B", 125): "amort_seed_500", ("7B", 62): "amort_split_125", ("7B", 125): "amort_rand_125"}  # verdict-fold arms, re-derived by l6a_critic.py
for (l, B), arm in BEST.items():
    n = int(arm.split("_")[-1]); k = math.ceil(n / B)
    d = [json.load(open(f"{R}amort_{J[l]}_B{B}_o{o}.json"))["arms"] for o in range(8, 16)]
    M = lambda a: np.array([x[a] for x in d], float)
    cp, fz, am = M("carry_plain"), M(f"freeze_plain_{n}"), M(arm)
    tot = (cp - am).sum(1).mean(); cal = (cp - am)[:, :k].sum(1).mean(); refit = (cp - fz)[:, k:].sum(1).mean(); qual = (fz - am)[:, k:].sum(1).mean()
    print(f"{l} B={B} {arm} k={k}: total deficit {tot:.1f} = calibration phase {cal:.1f} ({cal/tot:.0%}) + continued refit past n labels {refit:.1f} ({refit/tot:.0%}) + rule quality at equal labels {qual:.1f} ({qual/tot:.0%}); post-cal batches {12-k}")
