"""L2b critic: rerun the l2b_replay.py job() on the L1b pool (p0 feats) for splits 4 to 7, compare to runs/L1b/result.json.
Also reruns the full l2b_replay.job on the L2b pool for splits 0 and 4 to confirm result.json is reproducible. CPU only, in memory."""
import sys, json
import numpy as np
sys.path.insert(0, "/homes/rdubash/avrl_live/ROUTED/avrl/exp")
import l2b_replay as W
L1 = json.load(open("/homes/rdubash/avrl_live/ROUTED/avrl/runs/L1b/result.json"))
P0 = "/homes/rdubash/avrl_live/ROUTED/avrl/runs/p0/feats.parquet"
res = {s: W.job((P0, s)) for s in (4, 5, 6, 7)}
for t, (name, key) in W.CONFIGS.items():
    gaps = [float((W.L1B.rates(res, s, name, key) - W.L1B.rates(res, s, W.L1B.FIXED, key)).mean()) for s in (4, 5, 6, 7)]
    sel = L1["per_budget"][key]["heldout_selected_minus_fixed"]["per_split_pp"]
    print(t, name, "wrapper on L1b pool:", [round(g, 3) for g in gaps], "stored L1b:", [round(g, 3) for g in sel], "max diff", float(np.max(np.abs(np.array(gaps) - np.array(sel)))))
