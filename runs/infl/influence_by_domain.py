"""H6: does oracle-style data influence (first-order: grad_i . grad_ref) cluster by domain, and is it
predictable from cheap features? If yes, a MATES-style prober could pick probes adaptively instead of at random.
Model: Qwen3-0.6B-Base. Candidates: docs from runs/hyp/scores.parquet (domain stratum). Two reference sets:
general (held-out random FineWeb) and edu (held-out docs with edu_score >= 3)."""
import argparse, os, sys, time, json
import numpy as np, pandas as pd, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from aa import data as D
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(ROOT, "models", "Qwen3-0.6B-Base"))
ap.add_argument("--n", type=int, default=16000)
ap.add_argument("--n_ref", type=int, default=256)
ap.add_argument("--max_tokens", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
t0 = time.time()
sc = pd.read_parquet(os.path.join(ROOT, "runs", "hyp", "scores.parquet"))
full = D.load_fineweb(columns=("id", "text"))
text = dict(zip(full.id, full.text)); del full
rng = np.random.default_rng(a.seed)
dom = sc[sc.stratum == "domain"]
doms = rng.choice(dom.domain.unique(), size=min(800, dom.domain.nunique()), replace=False)
cand = dom[dom.domain.isin(doms)].sample(frac=1, random_state=a.seed).groupby("domain").head(20)
cand = cand.iloc[:a.n].reset_index(drop=True)
uni = sc[sc.stratum == "uniform"]
uni = uni[~uni.id.isin(cand.id)]
ref_general = uni.sample(n=a.n_ref, random_state=a.seed)
ref_edu = uni[uni.edu_score >= 3].sample(n=a.n_ref, random_state=a.seed)
print(f"cand {len(cand)} docs / {cand.domain.nunique()} domains; refs {len(ref_general)}, {len(ref_edu)}; {time.time()-t0:.0f}s", flush=True)

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda()
model.gradient_checkpointing_disable(); model.train(False)
params = [p for p in model.parameters() if p.requires_grad]

def loss_of(t):
    b = tok(t, truncation=True, max_length=a.max_tokens, return_tensors="pt").to("cuda")
    return model(**b, labels=b.input_ids).loss

def ref_grad(ids):
    model.zero_grad(set_to_none=True)
    for i in ids:
        (loss_of(text[i]) / len(ids)).backward()
    g = [p.grad.detach().float().clone() for p in params]
    model.zero_grad(set_to_none=True)
    return g

g_gen = ref_grad(ref_general.id.tolist()); g_edu = ref_grad(ref_edu.id.tolist())
ng = torch.sqrt(sum((x * x).sum() for x in g_gen)); ne = torch.sqrt(sum((x * x).sum() for x in g_edu))
print(f"ref grads done {time.time()-t0:.0f}s |g_gen|={ng:.3f} |g_edu|={ne:.3f}", flush=True)

inf_gen = np.zeros(len(cand), np.float32); inf_edu = np.zeros(len(cand), np.float32); gnorm = np.zeros(len(cand), np.float32); nll = np.zeros(len(cand), np.float32)
for k, i in enumerate(cand.id.tolist()):
    model.zero_grad(set_to_none=True)
    l = loss_of(text[i]); l.backward()
    dg = de = nn = 0.0
    for p, a1, a2 in zip(params, g_gen, g_edu):
        if p.grad is None: continue
        g = p.grad.float()
        dg += (g * a1).sum().item(); de += (g * a2).sum().item(); nn += (g * g).sum().item()
    inf_gen[k], inf_edu[k], gnorm[k], nll[k] = dg, de, nn ** 0.5, l.item()
    if k % 1000 == 0:
        print(f"{k}/{len(cand)} {time.time()-t0:.0f}s", flush=True)
cand = cand.assign(infl_general=inf_gen, infl_edu=inf_edu, cos_general=inf_gen / (gnorm * ng.item() + 1e-9), cos_edu=inf_edu / (gnorm * ne.item() + 1e-9), grad_norm=gnorm, nll_06b=nll)
cand.to_parquet(os.path.join(ROOT, "runs", "infl", "influence.parquet"), index=False)
print(json.dumps({"n": len(cand), "wall_s": time.time() - t0}), flush=True)
