"""Stage A table: noise of the raw pool, every baseline and oracle, and the two headroom gates.
    python curation/task/pilot_table.py v1
G1 = (best no-LLM baseline minus best oracle) in noise sds: room for a curator that understands the documents.
G2 = (raw minus best no-LLM baseline) in noise sds: what simple rules already buy."""
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

pool = sys.argv[1] if len(sys.argv) > 1 else "v1"
res = defaultdict(list)
for f in sorted((Path(__file__).resolve().parent.parent / "results" / pool).glob("*.json")):
    res[f.stem.rsplit("_s", 1)[0]].append(json.loads(f.read_text()))
raw = res["raw"]
sd = {k: st.stdev(r[k] for r in raw) for k in ("dev_bpb", "sealed_bpb")}
mean = lambda name, k: st.mean(r[k] for r in res[name])
print("raw pool, %d seeds: dev %.4f (sd %.4f), sealed %.4f (sd %.4f)" % (
    len(raw), mean("raw", "dev_bpb"), sd["dev_bpb"], mean("raw", "sealed_bpb"), sd["sealed_bpb"]))
print("%-16s %2s %8s %8s %9s %9s %8s" % ("config", "n", "dev", "sealed", "d dev", "d sealed", "docs"))
for name in sorted(res, key=lambda n: mean(n, "sealed_bpb")):
    print("%-16s %2d %8.4f %8.4f %+9.4f %+9.4f %8d" % (
        name, len(res[name]), mean(name, "dev_bpb"), mean(name, "sealed_bpb"), mean(name, "dev_bpb") - mean("raw", "dev_bpb"),
        mean(name, "sealed_bpb") - mean("raw", "sealed_bpb"), st.mean(r["n_selected"] for r in res[name])))
orc = [n for n in res if n.startswith("oracle")]
base = [n for n in res if n != "raw" and not n.startswith("oracle")]
if orc and base:
    for k in ("dev_bpb", "sealed_bpb"):
        bo, bb = min(orc, key=lambda n: mean(n, k)), min(base, key=lambda n: mean(n, k))
        print("%s: G1 = %s minus %s = %.4f bpb = %.1f noise sds; G2 = raw minus %s = %.4f bpb = %.1f noise sds" % (
            k, bb, bo, mean(bb, k) - mean(bo, k), (mean(bb, k) - mean(bo, k)) / sd[k], bb,
            mean("raw", k) - mean(bb, k), (mean("raw", k) - mean(bb, k)) / sd[k]))
