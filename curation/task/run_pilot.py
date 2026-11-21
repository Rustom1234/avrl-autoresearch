"""Driver for the pilot: runs this host's share of the job list sequentially (skips finished results).
    python run_pilot.py POOL SHARD NSHARDS     e.g.  python run_pilot.py v1 0 2   (on spruce)  /  1 2 (on willow)
Results: curation/results/<pool>/<config>_s<seed>.json"""
import os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
CUR = HERE / "baselines/curators.py"
CONFIGS = {  # name: (curate path, CURATE_MODE, seeds)
    "raw": (HERE / "curate.py", None, range(8)),
    **{n: (CUR, n, range(2)) for n in ["dedup", "random50", "heuristic", "tfidf0.3", "tfidf0.5", "tfidf0.8", "tfidf1.0",
                                         "oracle_wiki", "oracle_wiki_web", "oracle_web_wiki"]},
}
pool, shard, nshard = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
jobs = [(n, s) for n, (_, _, seeds) in CONFIGS.items() for s in seeds]
out_dir = HERE.parent / "results" / pool; out_dir.mkdir(parents=True, exist_ok=True)
for j, (name, seed) in enumerate(jobs):
    out = out_dir / f"{name}_s{seed}.json"
    if j % nshard != shard or out.exists():
        continue
    path, mode, _ = CONFIGS[name]
    env = dict(os.environ)
    if mode:
        env["CURATE_MODE"] = mode
    if name.startswith("oracle"):
        env["CURATION_ORACLE_LABELS"] = str(HERE.parent.parent / "data/curation" / f"pool_{pool}" / "labels_HIDDEN.json")
    print("RUN", name, seed, flush=True)
    subprocess.run([sys.executable, str(HERE / "run.py"), "--curate", str(path), "--seed", str(seed), "--out", str(out), "--pool", pool], env=env, check=True)
