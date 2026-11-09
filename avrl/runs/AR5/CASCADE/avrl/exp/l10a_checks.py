"""L10a input checks (CPU, read-only): the L9a assumptions plus the E01 join, before any replay is run.
Writes avrl/runs/L10a/assumption_checks.json."""
import json
import pandas as pd

E10 = "/homes/rdubash/avrl_live/CASCADE/avrl/runs/E10/feats.parquet"
E01 = "/homes/rdubash/avrl_live/CASCADE/avrl/runs/p0/feats.parquet"
OUT = "/homes/rdubash/avrl_live/CASCADE/avrl/runs/L10a/assumption_checks.json"

e10 = pd.read_parquet(E10)
e01 = pd.read_parquet(E01)
res = {}
res["rows"] = {"E10": len(e10), "E01": len(e01)}
res["pid_unique"] = {"E10": bool(e10.pid.is_unique), "E01": bool(e01.pid.is_unique)}
res["pid_sets_equal"] = bool(set(e10.pid) == set(e01.pid))
m = e10[["pid", "cell100", "proxy_error", "majority_correct"]].merge(e01[["pid", "cell100", "proxy_error", "majority_correct"]], on="pid", suffixes=("_e10", "_e01"), validate="one_to_one")
res["matched_pids"] = int(len(m))
res["cell100_identical_per_pid"] = bool((m.cell100_e10 == m.cell100_e01).all())
res["n_cell100_buckets"] = int(e10.cell100.nunique())
res["proxy_error_nan"] = {"E10": int(e10.proxy_error.isna().sum()), "E01": int(e01.proxy_error.isna().sum())}
res["proxy_error_equals_1_minus_majority_correct"] = {
    "E10": bool(((e10.proxy_error.astype(int)) == 1 - e10.majority_correct.astype(int)).all()),
    "E01": bool(((e01.proxy_error.astype(int)) == 1 - e01.majority_correct.astype(int)).all()),
}
res["proxy_error_error_count"] = {"E10": int(e10.proxy_error.sum()), "E01": int(e01.proxy_error.sum())}
res["e01_proxy_error_matches_on_join"] = int(m.proxy_error_e01.notna().sum())
res["join_asserts_pass"] = bool(res["matched_pids"] == 3000 and res["e01_proxy_error_matches_on_join"] == 3000 and res["pid_sets_equal"])
with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
