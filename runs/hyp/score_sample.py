"""Score a domain-stratified FineWeb sample with three candidate 'expensive' labels:
edu quality (FineWeb-Edu classifier), LM token loss (Qwen3-4B), and the regex PII proxy.
Output: runs/hyp/scores.parquet (no text). Then run analyze_hyp.py."""
import argparse, os, sys, time, json
import numpy as np, pandas as pd, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from aa import data as D
EDU = "HuggingFaceFW/fineweb-edu-classifier"

ap = argparse.ArgumentParser()
ap.add_argument("--n_domains", type=int, default=2000)
ap.add_argument("--per_domain", type=int, default=40)
ap.add_argument("--n_uniform", type=int, default=30000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_tokens", type=int, default=512)
ap.add_argument("--lm", default=os.path.join(ROOT, "models", "Qwen3-4B"))
ap.add_argument("--out", default=os.path.join(ROOT, "runs", "hyp", "scores.parquet"))
a = ap.parse_args()
t0 = time.time()

df = D.load_fineweb(columns=("id", "url", "text", "token_count", "language_score"))
df = D.add_domains(df)
df["host"] = df["url"].str.extract(r"^https?://([^/]+)")[0].str.lower()
rng = np.random.default_rng(a.seed)
counts = df["domain"].value_counts()
elig = counts[counts >= 10].index.values
chosen = rng.choice(elig, size=min(a.n_domains, len(elig)), replace=False)
sub = df[df["domain"].isin(chosen)].groupby("domain", group_keys=False).apply(
    lambda g: g.sample(n=min(len(g), a.per_domain), random_state=a.seed))
sub = sub.assign(stratum="domain")
uni = df.sample(n=a.n_uniform, random_state=a.seed + 1).assign(stratum="uniform")
s = pd.concat([sub, uni]).drop_duplicates("id").reset_index(drop=True)
del df
s = D.add_regex_features(s)
print(f"sample {len(s)} docs ({(s.stratum=='domain').sum()} stratified, {len(chosen)} domains) in {time.time()-t0:.0f}s", flush=True)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
texts = s["text"].astype(str).tolist()

tok = AutoTokenizer.from_pretrained(EDU)
m = AutoModelForSequenceClassification.from_pretrained(EDU, torch_dtype=torch.bfloat16).cuda().eval()
edu = []
for i in range(0, len(texts), 256):
    b = tok(texts[i:i + 256], truncation=True, max_length=512, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        edu.append(m(**b).logits.squeeze(-1).float().cpu())
s["edu_score"] = torch.cat(edu).numpy()
del m; torch.cuda.empty_cache()
print(f"edu done {time.time()-t0:.0f}s", flush=True)

tok2 = AutoTokenizer.from_pretrained(a.lm)
tok2.padding_side = "right"
lm = AutoModelForCausalLM.from_pretrained(a.lm, torch_dtype=torch.bfloat16).cuda().eval()
order = np.argsort([len(t) for t in texts])  # length-sort for padding efficiency
nll = np.zeros(len(texts), dtype=np.float32)
B = 8
for k in range(0, len(order), B):
    idx = order[k:k + B]
    b = tok2([texts[i] for i in idx], truncation=True, max_length=a.max_tokens, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = lm(**b).logits[:, :-1]
    tgt = b.input_ids[:, 1:]
    mask = b.attention_mask[:, 1:].float()
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1), reduction="none").view(tgt.shape)
    nll[idx] = ((ce * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()
    if k % (B * 500) == 0:
        print(f"nll {k}/{len(order)} {time.time()-t0:.0f}s", flush=True)
s["nll"] = nll
s.drop(columns=["text"]).to_parquet(a.out, index=False)
print(json.dumps({"n": len(s), "wall_s": time.time() - t0, "edu_mean": float(s.edu_score.mean()), "nll_mean": float(s.nll.mean())}), flush=True)
