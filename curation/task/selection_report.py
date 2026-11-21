"""Analysis only: what a curator selected, by hidden label, in units of T.  python selection_report.py CURATE_PATH [pool]"""
import json, subprocess, sys, os, collections, tempfile
import numpy as np
from common import DATA, T
curate, pool = sys.argv[1], DATA / f"pool_{sys.argv[2] if len(sys.argv) > 2 else 'v1'}"
with tempfile.NamedTemporaryFile(suffix=".json", dir=DATA) as tf:
    env = dict(os.environ, CURATION_ORACLE_LABELS=str(pool / "labels_HIDDEN.json"))
    subprocess.run([sys.executable, os.path.dirname(os.path.abspath(__file__)) + "/curate_runner.py", curate, str(pool), tf.name], env=env, check=True)
    idx = json.load(open(tf.name))
lab = [l["kind"] for l in json.load(open(pool / "labels_HIDDEN.json"))]
offs = np.load(pool / "pool_offsets.npy"); lens = np.diff(offs)
cnt, tot = collections.Counter(), 0
for i in idx:
    if tot >= T: break
    n = min(lens[i], T - tot); cnt[lab[i]] += n; tot += n
uniq = len(set(idx))
print(f"selected {len(idx)} docs ({uniq} unique), tokens before cut {int(lens[idx].sum())/T:.2f}T; used composition over T:")
print({k: round(v / T, 3) for k, v in sorted(cnt.items())})
