#!/usr/bin/env python3
"""W1 analysis: do the reader's internal signals predict the outcome of a proposed experiment?

    python loop/w1/analyze.py

Targets: crash (the first run crashed or timed out) and promising (z of the first run against the incumbent is at
least 0.75, the engine's bar for not rejecting at once). Feature sets:
  free       what a loop has without any model: edit class, idea family, which coder wrote it, size of the diff,
             whether the diff touches the batch-size line (the one move that mattered on the GPT task)
  verdict    the reader's yes-minus-no logit, alone
  surprise   mean and max surprise of the diff under the reader, plus the verdict
  attention  the Lookback-style attention shares and entropies (reduced to 64 principal components)
  hidden     the reader's hidden states (reduced to 64 principal components)
  each of attention and hidden also together with free
In-task: GPT-task rows, 5-fold cross-validation with whole loops held out; AUROC with a bootstrap interval over
loops. Transfer: fit on all GPT rows, test on the CIFAR rows. Routing replay: on held-out predictions, drop the
lowest-scored third of proposals and count GPU runs saved and promising proposals lost. Rows of the harm-control
cell whose edit was replaced by a no-op line are left out.
"""
import glob, json, random
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
rows = [json.loads(x) for x in open(HERE / "data.jsonl")]
F = {}
for f in sorted(glob.glob(str(HERE / "features/shard_*.npz"))):
    z = np.load(f)
    arr = {k: z[k] for k in ("attn_mean", "attn_last", "ent", "hid", "scal", "idx")}   # each array is read once
    for j, i in enumerate(arr["idx"]):
        F[int(i)] = {k: arr[k][j] for k in ("attn_mean", "attn_last", "ent", "hid", "scal")}


def is_noop(r):
    cand = (HERE / "files" / (r["cand"] + ".py")).read_text()
    inc = (HERE / "files" / (r["inc"] + ".py")).read_text()
    return "# candidate edit withheld" in cand and "# candidate edit withheld" not in inc


def touches_batch(r):
    a = (HERE / "files" / (r["inc"] + ".py")).read_text().splitlines()
    b = (HERE / "files" / (r["cand"] + ".py")).read_text().splitlines()
    key = lambda ls: [l for l in ls if "BATCH_SIZE" in l.upper() and "=" in l]
    return float(key(a) != key(b))


keep = [i for i, r in enumerate(rows) if i in F and not (r["harm"] and is_noop(r))]
R = [rows[i] for i in keep]
cats = lambda key: sorted({str(r[key]) for r in R})
onehot = lambda key: np.array([[float(str(r[key]) == c) for c in cats(key)] for r in R])
scal = np.array([F[i]["scal"] for i in keep], dtype=np.float64)
X = {
    "free": np.hstack([onehot("edit_class"), onehot("family"), onehot("coder"), np.log1p(scal[:, 3:4]),
                       np.array([[touches_batch(r)] for r in R])]),
    "verdict": scal[:, :1],
    "surprise": scal[:, :3],
    "attention": np.hstack([np.array([F[i][k].astype(np.float32).ravel() for i in keep]) for k in ("attn_mean", "attn_last", "ent")]),
    "hidden": np.array([F[i]["hid"].astype(np.float32).ravel() for i in keep]),
}
task = np.array([r["task"] for r in R])
loops = np.array([r["run"] + "/" + r["loop"] for r in R])
Y = {"crash": np.array([r["crash"] for r in R]), "promising": np.array([r["promising"] for r in R])}


def model(name):
    steps = [StandardScaler()]
    if name.split("+")[0] in ("attention", "hidden"):
        steps.append(PCA(64, random_state=0))
        steps.append(StandardScaler())
    return make_pipeline(*steps, LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))


def feats(name, idx):
    return np.hstack([X[p][idx] for p in name.split("+")])


def fit_predict(name, tr, te, y):
    parts = name.split("+")
    if len(parts) == 1:
        return model(name).fit(X[name][tr], y[tr]).decision_function(X[name][te])
    # big block reduced by PCA first, then joined with the free features
    red = make_pipeline(StandardScaler(), PCA(64, random_state=0)).fit(X[parts[0]][tr])
    a, b = np.hstack([red.transform(X[parts[0]][tr]), X[parts[1]][tr]]), np.hstack([red.transform(X[parts[0]][te]), X[parts[1]][te]])
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")).fit(a, y[tr]).decision_function(b)


def boot_auc(y, s, groups, n=1000):
    rng, gs, vals = random.Random(0), sorted(set(groups)), []
    by = {g: np.where(groups == g)[0] for g in gs}
    for _ in range(n):
        idx = np.concatenate([by[rng.choice(gs)] for _ in gs])
        if 0 < y[idx].sum() < len(idx):
            vals.append(roc_auc_score(y[idx], s[idx]))
    return np.percentile(vals, [2.5, 97.5])


gpt, cif = np.where(task == "gpt")[0], np.where(task == "cifar")[0]
names = ["free", "verdict", "surprise", "attention", "hidden", "attention+free", "hidden+free"]
print("rows: GPT task %d (crash %.3f, promising %.3f), CIFAR %d (crash %.3f, promising %.3f)" % (
    len(gpt), Y["crash"][gpt].mean(), Y["promising"][gpt].mean(), len(cif), Y["crash"][cif].mean(), Y["promising"][cif].mean()))
held = {}
for target in ("crash", "promising"):
    y = Y[target].astype(int)
    print("\n== %s ==\n| features | AUROC in task (5-fold by loop) | 95%% interval | AUROC on CIFAR after training on GPT |\n|---|---|---|---|" % target)
    for name in names:
        s = np.zeros(len(gpt))
        for tr, te in GroupKFold(5).split(gpt, y[gpt], loops[gpt]):
            s[te] = fit_predict(name, gpt[tr], gpt[te], y)
        lo, hi = boot_auc(y[gpt], s, loops[gpt])
        st = fit_predict(name, gpt, cif, y)
        tr_auc = roc_auc_score(y[cif], st) if 0 < y[cif].sum() < len(cif) else float("nan")
        tlo, thi = boot_auc(y[cif], st, loops[cif])
        print("| %s | %.3f | %.3f to %.3f | %.3f (%.2f to %.2f) |" % (name, roc_auc_score(y[gpt], s), lo, hi, tr_auc, tlo, thi))
        held[(target, name)] = s
print("\n== routing replay on the GPT task: drop the lowest-scored third of proposals before the GPU ==")
print("| score | GPU runs saved | promising proposals lost | crashes avoided |\n|---|---|---|---|")
yp, yc = Y["promising"][gpt], Y["crash"][gpt]
for name in names:
    s = held[("promising", name)]
    drop = s <= np.quantile(s, 1 / 3)
    print("| %s | %.0f%% | %.0f%% | %.0f%% |" % (name, 100 * drop.mean(), 100 * yp[drop].sum() / yp.sum(), 100 * yc[drop].sum() / max(1, yc.sum())))
print("| random | 33% | 33% | 33% |")
