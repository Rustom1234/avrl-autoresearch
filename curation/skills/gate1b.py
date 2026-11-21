#!/usr/bin/env python3
"""G1B (curation/design/G1B_PREREG.md): direct judge scores for every document, a combined arm, and 6 seeds.

    curation/.venv/bin/python curation/skills/gate1b.py judge --model gemma4-12b --threads 8   # every document
    curation/.venv/bin/python curation/skills/gate1b.py arms                                   # one curator file per new arm
    curation/.venv/bin/python curation/skills/gate1b.py report                                 # after the training runs
Reuses the prompt, the caller and the report helpers of gate1.py. Nothing written by gate1.py is changed.
"""
import argparse, json, random, statistics as st, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate1 as g                # noqa: E402  (also puts task/ and loop/models on the path)
import ops                       # noqa: E402
from common import DATA, T       # noqa: E402

OUT = HERE.parent / "runs/G1B"
DIRECT = {"gemma4-12b": "JC12D", "qwen35-4b": "JC4D", "fable": "JSD"}
OLD, NEW = ("BASE", "EMB", "JS", "JC12", "JC4"), ("JC12D", "JC4D", "COMB", "JSD")


def judge(a):
    """Scores every document kept after semantic dedup. Same call as gate1.judge; resumes from its own output file."""
    import call as call_mod
    call_mod.REGISTRY_PATH = HERE.parent / "loop/models.json"
    p = np.load(g.OUT / "prep.npz")
    rows = [int(r) for r in np.flatnonzero(p["sem"])]
    random.Random(g.SEED + 1).shuffle(rows)                          # who shares a call with whom is random
    if a.limit:
        rows = rows[:a.limit * g.PER_CALL]
    clean, _, _, target = g.load_pool()
    texts = {r: clean[int(p["ids"][r])][:g.SHOW] for r in rows}
    del clean
    OUT.mkdir(parents=True, exist_ok=True)
    out_f = OUT / ("direct_%s%s.json" % (a.model, "_test" if a.limit else ""))
    done = json.loads(out_f.read_text()) if out_f.exists() else {"scores": {}, "usd": 0.0, "calls": 0, "failed_calls": 0}
    todo = [r for r in rows if done["scores"].get(str(r)) is None]   # failed documents are tried again on a rerun
    batches = [todo[i:i + g.PER_CALL] for i in range(0, len(todo), g.PER_CALL)]
    is_open = call_mod.load_registry()[a.model]["provider"] != "claude"
    lock = threading.Lock()

    def one(k_batch):
        k, batch = k_batch
        user = g.RUBRIC % (target[0][:600], target[1][:600], len(batch), len(batch)) + "".join(
            "\n\n[%d]\n%s" % (j + 1, texts[r]) for j, r in enumerate(batch))
        r = call_mod.call(a.model, user, system=g.SYSTEM, schema=g.SCHEMA, max_tokens=300, tag="G1B:direct:%d" % k,
                          timeout_s=600, log_path=str(OUT / ("calls_%s.jsonl" % a.model)),
                          temperature=0.0 if is_open else None,
                          extra={"chat_template_kwargs": {"enable_thinking": False}} if is_open else None,
                          effort=None if is_open else "low")
        s = (r.obj or {}).get("scores") if r.ok else None
        good = isinstance(s, list) and len(s) == len(batch) and all(isinstance(x, int) and 1 <= x <= 5 for x in s)
        with lock:
            done["usd"] += r.cost_usd_nominal
            done["calls"] += 1
            done["failed_calls"] += not good
            for j, row in enumerate(batch):
                done["scores"][str(row)] = s[j] if good else None
            if done["calls"] % 50 == 0:
                out_f.write_text(json.dumps(done))
                print("%s: %d calls, %d failed, %.2f USD" % (a.model, done["calls"], done["failed_calls"], done["usd"]), flush=True)

    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(one, enumerate(batches)))
    out_f.write_text(json.dumps(done))
    got = [v for v in done["scores"].values() if v is not None]
    print("%s: scored %d of %d, mean %.2f, %d failed calls in total, %.2f USD" % (
        a.model, len(got), len(rows), st.mean(got), done["failed_calls"], done["usd"]))


def direct_scores(model):
    f = OUT / ("direct_%s.json" % model)
    return {int(k): v for k, v in json.loads(f.read_text())["scores"].items() if v is not None} if f.exists() else None


def z(x):
    return (x - x.mean()) / x.std()


def arms(a):
    p = np.load(g.OUT / "prep.npz")
    ids, emb, lens = p["ids"], p["emb"], p["lens"]
    kept = np.flatnonzero(p["sem"])
    n = len(np.load(DATA / ("pool_%s" % g.POOL) / "pool_offsets.npy")) // 2
    sel, info = {}, {}
    for model, arm in DIRECT.items():
        s = direct_scores(model)
        if s is None:
            continue
        rows = sorted(s)
        pred, r2 = ops.fit_scorer(emb[rows], np.array([s[r] for r in rows], dtype=float), emb)
        score = np.array([s.get(int(r), pred[r]) for r in kept], dtype=float)       # unscored: the prediction
        order = kept[np.lexsort((-pred[kept], -score))]                             # by score, ties by prediction
        sel[arm] = ops.take_top(order, lens, T)
        info[arm] = dict(labels=len(rows), unscored=int(len(kept) - len(set(rows) & set(kept.tolist()))), cv_r2=r2)
    js = g.judge_scores("fable")
    rows = sorted(js)
    pred_js, _ = ops.fit_scorer(emb[rows], np.array([js[r] for r in rows], dtype=float), emb)       # as in arm JS
    comb = z(p["tscore"][kept]) + z(pred_js[kept])
    sel["COMB"] = ops.take_top(kept[np.argsort(-comb, kind="stable")], lens, T)
    (OUT / "arms").mkdir(parents=True, exist_ok=True)
    jobs = []
    for arm, rws in sel.items():
        f = OUT / "arms" / (arm + ".json")
        f.write_text(json.dumps([int(n + ids[r]) for r in rws]))
        (OUT / "arms" / (arm + ".py")).write_text(
            '"""Arm %s of G1B: a fixed list of documents (built by curation/skills/gate1b.py) in a seeded random order."""\n'
            "import json, os, random\n\n\ndef select(docs, target_sample):\n"
            "    idx = json.load(open(%r))\n"
            '    random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(idx)\n    return idx\n' % (arm, str(f)))
        info.setdefault(arm, {}).update(docs=len(rws), rows=[int(r) for r in rws])
        jobs.append("G1B_%s=%s" % (arm, OUT / "arms" / (arm + ".py")))
        print("%-6s %6d docs %s" % (arm, len(rws), {k: v for k, v in info[arm].items() if k not in ("rows", "docs")}))
    (OUT / "arms.json").write_text(json.dumps(info))
    print("jobs: " + " ".join(jobs))


def report(a):
    from scipy.stats import spearmanr
    res_dir = HERE.parent / "results" / g.POOL
    sealed = {}
    for arm in OLD + NEW:
        fs = sorted(res_dir.glob("%s_%s_s*.json" % ("G1" if arm in OLD else "G1B", arm)))
        if fs:
            sealed[arm] = [json.loads(f.read_text())["sealed_bpb"] for f in fs]
    df = sum(len(v) - 1 for v in sealed.values())
    sd = (sum(st.variance(v) * (len(v) - 1) for v in sealed.values()) / df) ** 0.5
    m = {k: st.mean(v) for k, v in sealed.items()}
    print("noise sd of one run (pooled within arms, %d df): %.4f; 3 sds = %.4f\n" % (df, sd, 3 * sd))
    print("%-6s %2s %8s  %s" % ("arm", "n", "sealed", "sealed by seed"))
    for k, v in sealed.items():
        print("%-6s %2d %8.4f  %s" % (k, len(v), m[k], " ".join("%.4f" % x for x in v)))
    out = {"noise_sd": sd, "sealed": sealed, "pairs": {}, "answers": {}}

    def diff(x, y):
        d = m[x] - m[y]
        se = sd * (1 / len(sealed[x]) + 1 / len(sealed[y])) ** 0.5
        out["pairs"]["%s-%s" % (x, y)] = dict(diff=d, sds=d / sd, se=se)
        print("%s minus %s: %+.4f bpb = %+.1f noise sds (standard error %.4f)" % (x, y, d, d / sd, se))
        return d > 3 * sd

    print()
    best = min(("EMB", "JS"), key=m.get)
    qs = (("Q1, the judge was handicapped", "JC12", "JC12D"), ("Q2, direct judge beats embeddings", "EMB", "JC12D"),
          ("Q3, composing beats each skill alone", best, "COMB"), ("Q4, gate G on 6 seeds", "BASE", "JS"))
    for name, x, y in qs:
        if x in m and y in m:
            out["answers"][name[:2]] = bool(diff(x, y))
            print("  %s: %s\n" % (name, "YES" if out["answers"][name[:2]] else "NO"))
    for x, y in (("JC4D", "JC12D"), ("JC4", "JC4D"), ("EMB", "JS"), ("JC12", "JS"), ("JC12D", "JSD"), ("EMB", "JSD")):
        if x in m and y in m:
            diff(x, y)
    print()
    fable, g1b_info = g.judge_scores("fable"), json.loads((OUT / "arms.json").read_text())
    g1_info = json.loads((g.OUT / "arms.json").read_text())
    for model in DIRECT:
        d, old = direct_scores(model), g.judge_scores(model)
        if d is None:
            continue
        both = sorted(set(fable) & set(d))
        rho = spearmanr([fable[r] for r in both], [d[r] for r in both])[0]
        out["rho_fable_direct_%s" % model] = rho
        print("%s direct against fable (G1) on %d documents: rank agreement %.2f" % (model, len(both), rho))
        if old and model != "fable":
            both = sorted(set(old) & set(d))
            same = st.mean(old[r] == d[r] for r in both)
            rho = spearmanr([old[r] for r in both], [d[r] for r in both])[0]
            out["retest_%s" % model] = dict(same=same, rho=rho)
            print("%s against itself (G1 call and G1B call) on %d documents: same score %.0f percent, rank agreement %.2f" % (
                model, len(both), 100 * same, rho))
    rows = {k: set(v["rows"]) for k, v in list(g1_info.items()) + list(g1b_info.items()) if v.get("rows")}
    for x, y in (("JC12D", "JC12"), ("JC12D", "EMB"), ("JC12D", "JS"), ("JC12D", "JC4D"), ("COMB", "EMB"), ("COMB", "JS"), ("JSD", "JC12D")):
        if x in rows and y in rows:
            print("documents shared by %s and %s: %.0f percent of %s" % (x, y, 100 * len(rows[x] & rows[y]) / len(rows[x]), x))
    (OUT / "report.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--model", required=True, choices=list(DIRECT))
    j.add_argument("--threads", type=int, default=4)
    j.add_argument("--limit", type=int, default=0, help="test: only this many calls, written to a _test file")
    sub.add_parser("arms")
    sub.add_parser("report")
    a = ap.parse_args()
    {"judge": judge, "arms": arms, "report": report}[a.cmd](a)
