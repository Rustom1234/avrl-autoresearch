#!/usr/bin/env python3
"""Skip-filter replay on the six BG1 tasks (model loops of loop/runs/BG1_R).

    python loop/w1/filter_bg1.py

Question: with signals a loop has before the job runs, can it skip the lowest-scored third of proposals without
losing the good ones? Rows: every candidate that reached a job. Targets: promising (z of the first job at least
0.75), kept (accepted), wasted (crash or time-out). Features, none of which needs the job or an extra model call:
slot number, position of the idea in the planner's list, size of the candidate script and its change against the
starting script (lines added, removed, imports added), which model wrote it, and the words of the idea's one-line
description (TF-IDF, 300 terms). Tests: (a) inside a family, 5-fold with whole loops held out; (b) across tasks:
fit on the other tasks of the family, test on the held-out task; (c) across families: fit on one, test on the other.
"""
import difflib, glob, json, re
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "loop/runs/BG1_R"
plan = json.loads((ROOT / "plan.json").read_text())
rows = []
for lp in plan["loops"]:
    if lp["cmd"] != "loop":
        continue
    lid, cfg = lp["loop_id"], lp["config"]
    task = cfg["task"]
    base = (REPO / "loop/bench/base" / ("packing.py" if task.startswith("pack") else "admet_clf.py" if task == "bbb_martins" else "admet_reg.py")).read_text().splitlines()
    for l in open(ROOT / lid / "ledger.jsonl"):
        r = json.loads(l)
        if r["kind"] != "candidate" or not r.get("job_id") or r["decision"] == "infra":
            continue
        f = ROOT / lid / "slots" / str(r["slot"]) / "train.py"
        if not f.exists():
            continue
        cand = f.read_text().splitlines()
        d = [x for x in difflib.unified_diff(base, cand, lineterm="", n=0) if x[:1] in "+-" and x[:3] not in ("+++", "---")]
        m = re.match(r"i(\d+)", r.get("idea_id") or "")
        rows.append(dict(loop=lid, task=task, fam="packing" if task.startswith("pack") else "drug", cell=cfg["cell"],
                         text=r.get("description") or "",
                         x=[r["slot"], int(m.group(1)) if m else 30, np.log1p(len(cand)), np.log1p(sum(x[0] == "+" for x in d)),
                            np.log1p(sum(x[0] == "-" for x in d)), sum(x.startswith("+import") or x.startswith("+from") for x in d),
                            float(cfg["cell"] == "SS")],
                         promising=int(r.get("z1") is not None and r["z1"] >= 0.75), kept=int(r["decision"] == "accept"),
                         wasted=int(r.get("status") != "ok")))
X = np.array([r["x"] for r in rows])
T = [r["text"] for r in rows]
fam = np.array([r["fam"] for r in rows]); task = np.array([r["task"] for r in rows]); loops = np.array([r["loop"] for r in rows])
Y = {k: np.array([r[k] for r in rows]) for k in ("promising", "kept", "wasted")}
print("rows %d (drug %d, packing %d); promising %.3f, kept %.3f, wasted %.3f" % (
    len(rows), (fam == "drug").sum(), (fam == "packing").sum(), Y["promising"].mean(), Y["kept"].mean(), Y["wasted"].mean()))


def fit_predict(tr, te, y, words=True):
    sc = StandardScaler().fit(X[tr])
    a, b = csr_matrix(sc.transform(X[tr])), csr_matrix(sc.transform(X[te]))
    if words:
        tf = TfidfVectorizer(max_features=300, ngram_range=(1, 2), sublinear_tf=True).fit([T[i] for i in tr])
        a, b = hstack([a, tf.transform([T[i] for i in tr])]).tocsr(), hstack([b, tf.transform([T[i] for i in te])]).tocsr()
    return LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced").fit(a, y[tr]).decision_function(b)


def replay(s, idx):
    drop = s <= np.quantile(s, 1 / 3)
    f = lambda k: 100.0 * Y[k][idx][drop].sum() / max(1, Y[k][idx].sum())
    return "skipped %.0f%%: promising lost %.0f%%, kept lost %.0f%% (%d of %d), wasted avoided %.0f%%" % (
        100 * drop.mean(), f("promising"), f("kept"), Y["kept"][idx][drop].sum(), Y["kept"][idx].sum(), f("wasted"))


auc = lambda y, s: roc_auc_score(y, s) if 0 < y.sum() < len(y) else float("nan")
for F in ("drug", "packing"):
    idx = np.where(fam == F)[0]
    for words in (False, True):
        s = np.zeros(len(idx))
        for tr, te in GroupKFold(5).split(idx, groups=loops[idx]):
            s[te] = fit_predict(idx[tr], idx[te], Y["promising"], words)
        print("\n(a) %s, loops held out, %s: AUROC promising %.3f, kept %.3f\n    %s" % (
            F, "with idea words" if words else "numbers only", auc(Y["promising"][idx], s), auc(Y["kept"][idx], s), replay(s, idx)))
print()
for t in sorted(set(task[fam == "drug"])):
    tr, te = np.where((fam == "drug") & (task != t))[0], np.where(task == t)[0]
    s = fit_predict(tr, te, Y["promising"])
    print("(b) drug, held-out task %s: AUROC promising %.3f\n    %s" % (t, auc(Y["promising"][te], s), replay(s, te)))
for a, b in (("drug", "packing"), ("packing", "drug")):
    tr, te = np.where(fam == a)[0], np.where(fam == b)[0]
    s = fit_predict(tr, te, Y["promising"])
    print("(c) fit on %s, test on %s: AUROC promising %.3f\n    %s" % (a, b, auc(Y["promising"][te], s), replay(s, te)))
