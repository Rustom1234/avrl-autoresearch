#!/usr/bin/env python3
"""Gate G1 (curation/design/G1_PREREG.md): a skill pipeline on a real web pool against the fixed standard pipeline.

    curation/.venv/bin/python curation/skills/gate1.py prep                 # GPU: heuristics, dedup, embeddings
    curation/.venv/bin/python curation/skills/gate1.py judge --model fable  # 200 calls, scores for the 2,000 sample
    curation/.venv/bin/python curation/skills/gate1.py arms                 # writes one curator file per arm
    curation/.venv/bin/python curation/skills/gate1.py report               # after the training runs
Training runs go through curation/task/run_recipes.py with the jobs that `arms` prints.
"""
import argparse, json, random, statistics as st, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent / "task"), str(ROOT / "loop/models")]
import ops                       # noqa: E402
from common import DATA, T       # noqa: E402

POOL, OUT, SEED = "web1", HERE.parent / "runs/G1", 20261112
N_SAMPLE, PER_CALL, SHOW = 2000, 10, 1500
JUDGES = {"fable": "JS", "gemma4-12b": "JC12", "qwen35-4b": "JC4"}
ARMS = ("RAW", "BASE", "EMB", "JS", "JC12", "JC4")
SYSTEM = "You grade web documents as training data for a language model. Reply with JSON only."
RUBRIC = """A small language model will be trained on a selection of web documents. It should get better at text like
these two target samples:

TARGET A: %s

TARGET B: %s

Score each of the %d documents below from 1 to 5 for how useful it is as training text for that goal.
5: clean, well-written, informative English prose of the same kind as the targets.
4: good English prose, somewhat different in kind or with a little noise.
3: readable English, but off-style or partly noisy.
2: mostly lists, fragments, ads or templates with little prose.
1: spam, machine filler, garbled text or not English.
Reply as {"scores": [...]} with exactly %d integers, in the order of the documents.
"""
SCHEMA = {"type": "object", "properties": {"scores": {"type": "array", "items": {"type": "integer"}}},
          "required": ["scores"]}


def load_pool():
    pool = DATA / ("pool_%s" % POOL)
    docs = json.load(open(pool / "docs.json"))
    n = len(docs) // 2
    lens = np.diff(np.load(pool / "pool_offsets.npy"))
    return docs[n:], n, lens[n:], json.load(open(pool / "target_sample.json"))


def prep(a):
    clean, n, lens, target = load_pool()
    ok = np.array([ops.doc_ok(t) for t in clean])
    ids = np.flatnonzero(ok)
    ids = ids[ops.exact_dedup([clean[i] for i in ids])]              # BASE: page ids, in pool order
    emb = ops.embed([clean[i] for i in ids])
    sem = ops.semantic_dedup(emb)
    tscore = ops.target_score(emb, ops.embed(target))
    sample = np.array(sorted(random.Random(SEED).sample(list(np.flatnonzero(sem)), N_SAMPLE)))   # rows of ids
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "prep.npz", ids=ids, emb=emb, sem=sem, tscore=tscore, sample=sample, lens=lens[ids])
    info = dict(pages=n, heuristics_ok=int(ok.sum()), base=len(ids), base_over_T=float(lens[ids].sum() / T),
                after_semantic_dedup=int(sem.sum()), sem_over_T=float(lens[ids][sem].sum() / T))
    (OUT / "prep.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def judge(a):
    import call as call_mod
    call_mod.REGISTRY_PATH = HERE.parent / "loop/models.json"
    clean, _, _, target = load_pool()
    p = np.load(OUT / "prep.npz")
    rows = [int(r) for r in p["sample"]]
    texts = {r: clean[int(p["ids"][r])][:SHOW] for r in rows}
    out_f = OUT / ("judge_%s.json" % a.model)
    done = json.loads(out_f.read_text()) if out_f.exists() else {"scores": {}, "usd": 0.0, "calls": 0, "failed_calls": 0}
    todo = [r for r in rows if str(r) not in done["scores"]]
    batches = [todo[i:i + PER_CALL] for i in range(0, len(todo), PER_CALL)]
    is_open = call_mod.load_registry()[a.model]["provider"] != "claude"
    lock = threading.Lock()

    def one(k_batch):
        k, batch = k_batch
        user = RUBRIC % (target[0][:600], target[1][:600], len(batch), len(batch)) + "".join(
            "\n\n[%d]\n%s" % (j + 1, texts[r]) for j, r in enumerate(batch))
        r = call_mod.call(a.model, user, system=SYSTEM, schema=SCHEMA, max_tokens=300, tag="G1:judge:%d" % k,
                          timeout_s=600, log_path=str(OUT / ("calls_%s.jsonl" % a.model)), temperature=0.0 if is_open else None,
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
            if done["calls"] % 10 == 0:
                out_f.write_text(json.dumps(done))
                print("%s: %d calls, %d failed, %.2f USD" % (a.model, done["calls"], done["failed_calls"], done["usd"]), flush=True)

    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(one, enumerate(batches)))
    out_f.write_text(json.dumps(done))
    got = [v for v in done["scores"].values() if v is not None]
    print("%s: scored %d of %d, mean %.2f, %d failed calls, %.2f USD" % (
        a.model, len(got), len(rows), st.mean(got), done["failed_calls"], done["usd"]))


def judge_scores(model):
    f = OUT / ("judge_%s.json" % model)
    return {int(k): v for k, v in json.loads(f.read_text())["scores"].items() if v is not None} if f.exists() else None


def arms(a):
    p = np.load(OUT / "prep.npz")
    ids, emb, sem, lens, n = p["ids"], p["emb"], p["sem"], p["lens"], len(np.load(DATA / ("pool_%s" % POOL) / "pool_offsets.npy")) // 2
    kept = np.flatnonzero(sem)

    def top(score):                                  # rows of ids, best first among the semantically deduplicated
        return ops.take_top(kept[np.argsort(-score[kept], kind="stable")], lens, T)

    sel, info = {"RAW": None, "BASE": list(range(len(ids))), "EMB": top(p["tscore"])}, {}
    for model, arm in JUDGES.items():
        s = judge_scores(model)
        if s is None:
            continue
        rows = sorted(s)
        pred, r2 = ops.fit_scorer(emb[rows], np.array([s[r] for r in rows], dtype=float), emb)
        sel[arm], info[arm] = top(pred), dict(labels=len(rows), cv_r2=r2)
    (OUT / "arms").mkdir(exist_ok=True)
    jobs = []
    for arm, rows in sel.items():
        idx = list(range(n)) if rows is None else [int(n + ids[r]) for r in rows]    # RAW: raw views; others: cleaned
        f = OUT / "arms" / (arm + ".json")
        f.write_text(json.dumps(idx))
        (OUT / "arms" / (arm + ".py")).write_text(
            '"""Arm %s of gate G1: a fixed list of documents (built by curation/skills/gate1.py) in a seeded random order."""\n'
            "import json, os, random\n\n\ndef select(docs, target_sample):\n"
            "    idx = json.load(open(%r))\n"
            '    random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(idx)\n    return idx\n' % (arm, str(f)))
        info.setdefault(arm, {}).update(docs=len(idx), rows=None if rows is None else [int(r) for r in rows])
        jobs.append("G1_%s=%s" % (arm, OUT / "arms" / (arm + ".py")))
    (OUT / "arms.json").write_text(json.dumps(info))
    print("\n".join("%-5s %6d docs %s" % (k, v["docs"], "" if "cv_r2" not in v else "cv R2 %.2f on %d labels" % (v["cv_r2"], v["labels"]))
                    for k, v in info.items()))
    print("jobs: " + " ".join(jobs))


def report(a):
    from scipy.stats import spearmanr
    res = {arm: [json.loads(f.read_text()) for f in sorted((HERE.parent / "results" / POOL).glob("G1_%s_s*.json" % arm))]
           for arm in ARMS}
    res = {k: v for k, v in res.items() if v}
    sealed = {k: [r["sealed_bpb"] for r in v] for k, v in res.items()}
    df = sum(len(v) - 1 for v in sealed.values())
    sd = (sum(st.variance(v) * (len(v) - 1) for v in sealed.values() if len(v) > 1) / df) ** 0.5
    print("noise sd of one run (pooled within arms, %d df): %.4f; 3 sds = %.4f\n" % (df, sd, 3 * sd))
    print("%-5s %2s %8s %8s  %s" % ("arm", "n", "dev", "sealed", "sealed by seed"))
    for k, v in res.items():
        print("%-5s %2d %8.4f %8.4f  %s" % (k, len(v), st.mean(r["dev_bpb"] for r in v), st.mean(sealed[k]),
                                            " ".join("%.4f" % x for x in sealed[k])))
    m = {k: st.mean(v) for k, v in sealed.items()}
    out = {"noise_sd": sd, "sealed": sealed, "pairs": {}}
    print()
    for x, y in (("RAW", "BASE"), ("BASE", "JS"), ("BASE", "EMB"), ("EMB", "JS"), ("JC12", "JS"), ("JC4", "JS")):
        if x in m and y in m:
            d = m[x] - m[y]
            out["pairs"]["%s-%s" % (x, y)] = d
            print("%s minus %s: %+.4f bpb = %+.1f noise sds" % (x, y, d, d / sd))
    if "BASE" in m and "JS" in m:
        out["G"] = m["BASE"] - m["JS"] > 3 * sd
        print("\nGate G (BASE minus JS above 3 noise sds): %s" % ("PASS" if out["G"] else "FAIL"))
    info = json.loads((OUT / "arms.json").read_text())
    js = {m_: judge_scores(m_) for m_ in JUDGES}
    print()
    for mod in ("gemma4-12b", "qwen35-4b"):
        if js.get("fable") and js.get(mod):
            both = sorted(set(js["fable"]) & set(js[mod]))
            rho = spearmanr([js["fable"][r] for r in both], [js[mod][r] for r in both])[0]
            out["rho_fable_%s" % mod] = rho
            print("rank agreement of %s with fable on %d documents: %.2f" % (mod, len(both), rho))
    for x, y in (("JS", "EMB"), ("JS", "JC12"), ("JS", "JC4"), ("JC12", "JC4")):
        if x in info and y in info and info[x].get("rows") and info[y].get("rows"):
            a_, b_ = set(info[x]["rows"]), set(info[y]["rows"])
            print("documents shared by %s and %s: %.0f percent of %s" % (x, y, 100 * len(a_ & b_) / len(a_), x))
    (OUT / "report.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    j = sub.add_parser("judge")
    j.add_argument("--model", required=True, choices=list(JUDGES))
    j.add_argument("--threads", type=int, default=4)
    sub.add_parser("arms")
    sub.add_parser("report")
    a = ap.parse_args()
    {"prep": prep, "judge": judge, "arms": arms, "report": report}[a.cmd](a)
