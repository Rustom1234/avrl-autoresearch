"""L10a reproduction gate (CPU). The new script exp/l10a_boot_xfit_fixed.py, run in its unchanged main mode (XFIT unset, MIN_CELL=1, BOOT_FRAC=0.25, FORCE_BEST7, B=200 x 4 seeds)
on the four pools, must reproduce the stored runs/{L9c-weak-policy-tiebreak,L8c-A-fresh-tiebreak}/boot/{q3_4b,rm}/boot_tb_f0.25.json EXACTLY: the three registered full-pool means,
the whole full_pool dict, the sanity dict and all 200 resamples. Usage: l10a_gate_repro.py <run_dir>; reads <run_dir>/gate_repro/<pool>/<proxy>/boot_tb_f0.25.json, writes <run_dir>/gate_repro/gate.json, exit 0 on pass, 5 on any mismatch."""
import json, os, sys
import numpy as np

R = sys.argv[1]; A = "/homes/rdubash/avrl_live/FULL/avrl"
EXPECT = {("L9c-weak-policy-tiebreak", "q3_4b"): (192.25, 214.25, 220.0), ("L9c-weak-policy-tiebreak", "rm"): (201.875, 222.75, 231.0),
          ("L8c-A-fresh-tiebreak", "q3_4b"): (136.5, 123.0, 151.0), ("L8c-A-fresh-tiebreak", "rm"): (142.0, 130.5, 148.0)}
KEYS = ("agree_acc", "blend50_acc", "greedy_blend7"); out = {"pass": True, "cells": {}}
for (pool, px), exp_means in EXPECT.items():
    c = f"{pool}/{px}"; d = {"expected": exp_means}
    try:
        new = json.load(open(f"{R}/gate_repro/{pool}/{px}/boot_tb_f0.25.json")); old = json.load(open(f"{A}/runs/{pool}/boot/{px}/boot_tb_f0.25.json"))
        d["new_means"] = tuple(float(np.mean(new["full_pool"][k])) for k in KEYS); d["old_means"] = tuple(float(np.mean(old["full_pool"][k])) for k in KEYS)
        d["means_equal_expected"] = d["new_means"] == exp_means and d["old_means"] == exp_means
        d["full_pool_identical"] = new["full_pool"] == old["full_pool"]; d["sanity_identical"] = new["sanity"] == old["sanity"]
        d["resamples_identical"] = new["resamples"] == old["resamples"]; d["n_resamples"] = (len(new["resamples"]), len(old["resamples"]))
        d["ok"] = bool(d["means_equal_expected"] and d["full_pool_identical"] and d["sanity_identical"] and d["resamples_identical"] and len(new["resamples"]) == 200)
    except Exception as e:  # missing or unreadable file is a gate failure, reported with the error text
        d["ok"] = False; d["error"] = repr(e)
    out["cells"][c] = d; out["pass"] = out["pass"] and d["ok"]; print(c, json.dumps(d), flush=True)
json.dump(out, open(f"{R}/gate_repro/gate.json", "w"), indent=1); print("REPRODUCTION GATE", "PASS" if out["pass"] else "FAILED", flush=True)
sys.exit(0 if out["pass"] else 5)
