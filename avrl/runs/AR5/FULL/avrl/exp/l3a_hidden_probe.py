"""L3a stage 0 (GPU): last-token hidden states of Qwen2.5-1.5B-Instruct on question + majority-answer completion, from the saved E21 rollouts.
No generation. For each pid 0..n-1 take the first rollout with proxy==1 (the completion carrying the answer the proxy accepts).
Input ids = chat-template prompt ids (system SYS from e21_lowT_rollouts.py + question, add_generation_prompt) + tokenizer(completion text) ids.
Saves <out>/hs.npz: hs [n, n_layers, 2, d] (positions: 0 = P last prompt token, 1 = A last completion token), pid, layers, prompt_len, comp_len, vllm_tokens.
Gates (<out>/gates.json, nonzero exit on failure): (a) re-tokenized length within 2 of vLLM `tokens` for >= 95% of non-truncated rows,
(b) 8 random sequences run unbatched match the batched states (cosine > 0.999, both positions, all layers), (c) finite values and correct shape.
Throughput abort: after the first 200 sequences, projected total > 35 min -> exit 3."""
import argparse, os, json, time, sys
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--rollouts", required=True); ap.add_argument("--groups", required=True)
ap.add_argument("--layers", default="14,20,28"); ap.add_argument("--out", required=True)
ap.add_argument("--n_prompts", type=int, default=2000); ap.add_argument("--bs", type=int, default=16)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
SYS = "Solve the math problem. Reason step by step, then give the final answer on its own line as: #### <number>"  # exact string from e21_lowT_rollouts.py
layers = [int(x) for x in a.layers.split(",")]
R = pd.read_parquet(a.rollouts); G = pd.read_parquet(a.groups).sort_values("pid").reset_index(drop=True)
assert len(G) == a.n_prompts and (G.pid.values == np.arange(a.n_prompts)).all()
first = R[R.proxy == 1].sort_values(["pid", "k"]).groupby("pid").head(1).sort_values("pid").reset_index(drop=True)
assert len(first) == a.n_prompts and (first.pid.values == np.arange(a.n_prompts)).all(), "every pid needs a proxy==1 rollout"
tok = AutoTokenizer.from_pretrained(a.model)
P_ids, C_ids = [], []
for q, txt in zip(G.question, first.text):
    ptxt = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    P_ids.append(tok(ptxt, add_special_tokens=False).input_ids); C_ids.append(tok(txt, add_special_tokens=False).input_ids)
plen = np.array([len(x) for x in P_ids]); clen = np.array([len(x) for x in C_ids]); seqs = [p + c for p, c in zip(P_ids, C_ids)]; slen = plen + clen
print(f"tokenized {len(seqs)} seqs, total tokens {slen.sum()}, max len {slen.max()}, mean prompt {plen.mean():.1f}, mean completion {clen.mean():.1f} ({time.time()-t0:.0f}s)", flush=True)
print("prompt tail of pid 0:", repr(tok.decode(P_ids[0][-12:])), "| completion head:", repr(tok.decode(C_ids[0][:8])), flush=True)

# gate (a) is data-only, evaluate now
vt = first.tokens.values; nt = first.truncated.values == 0
diff = np.abs(clen - vt)[nt]; frac_ok = float((diff <= 2).mean())
gates = {"a_len_match": {"frac_within_2": frac_ok, "n_nontruncated": int(nt.sum()), "median_signed_diff_retok_minus_vllm": float(np.median((clen - vt)[nt])), "max_abs_diff": int(diff.max()), "pass": bool(frac_ok >= 0.95)}}
print("gate a:", json.dumps(gates["a_len_match"]), flush=True)

dev = "cuda"
model = AutoModel.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()  # base model: no lm_head, so no 150k-vocab logits
pad = tok.pad_token_id if tok.pad_token_id is not None else 0
print(f"model loaded, layers {model.config.num_hidden_layers}, d {model.config.hidden_size} ({time.time()-t0:.0f}s)", flush=True)

@torch.no_grad()
def encode(idx):
    """hidden states at P and A for the sequences in idx (list of pid), right padding. returns float32 [len(idx), n_layers, 2, d] on cpu."""
    L = max(slen[i] for i in idx); ids = torch.full((len(idx), L), pad, dtype=torch.long); am = torch.zeros((len(idx), L), dtype=torch.long)
    for r, i in enumerate(idx): ids[r, :slen[i]] = torch.tensor(seqs[i]); am[r, :slen[i]] = 1
    hs = model(input_ids=ids.to(dev), attention_mask=am.to(dev), output_hidden_states=True).hidden_states  # tuple of n_layers+1; [0] = embeddings
    pp = torch.tensor([plen[i] - 1 for i in idx], device=dev); aa = torch.tensor([slen[i] - 1 for i in idx], device=dev); rr = torch.arange(len(idx), device=dev)
    out = torch.stack([torch.stack([hs[l][rr, pp], hs[l][rr, aa]], 1) for l in layers], 1)  # [b, n_layers, 2, d]
    return out.float().cpu().numpy()

order = np.argsort(-slen)  # longest first: early throughput is the conservative one, OOM would show immediately
n, d = a.n_prompts, model.config.hidden_size; H = np.zeros((n, len(layers), 2, d), np.float32); done_tok = 0; t1 = time.time(); checked = False
for s in range(0, n, a.bs):
    idx = [int(i) for i in order[s:s + a.bs]]; H[idx] = encode(idx); done_tok += int(slen[idx].sum())
    if not checked and s + a.bs >= 200:
        checked = True; el = time.time() - t1; proj = el * slen.sum() / done_tok
        print(f"throughput: {s + a.bs} seqs, {done_tok} tokens in {el:.1f}s ({done_tok/el:.0f} tok/s), projected total {proj/60:.1f} min", flush=True)
        if proj > 35 * 60: print("ABORT: projected total over 35 min", flush=True); sys.exit(3)
    if (s // a.bs) % 20 == 0: print(f"  {s + len(idx)}/{n} seqs, {time.time()-t1:.0f}s", flush=True)
print(f"forward passes done in {time.time()-t1:.0f}s", flush=True)

# gate (b): 8 random sequences unbatched
rng = np.random.default_rng(0); chk = [int(x) for x in rng.choice(n, 8, replace=False)]; cos = []
for i in chk:
    u = encode([i])[0]; b = H[i]  # [n_layers, 2, d]
    for li in range(len(layers)):
        for pi in range(2): cos.append(float(np.dot(u[li, pi], b[li, pi]) / (np.linalg.norm(u[li, pi]) * np.linalg.norm(b[li, pi]))))
gates["b_unbatched"] = {"pids": chk, "min_cos": float(min(cos)), "mean_cos": float(np.mean(cos)), "n_pairs": len(cos), "pass": bool(min(cos) > 0.999)}
print("gate b:", json.dumps(gates["b_unbatched"]), flush=True)

# gate (c): finite + shape; store float16 (lossless for bf16 values inside float16 range), fall back to float32 if any |value| would overflow
amax = float(np.abs(H).max()); sdt = np.float16 if amax < 6e4 else np.float32; Hs = H.astype(sdt)
fin = bool(np.isfinite(Hs).all()); shp = tuple(Hs.shape) == (n, len(layers), 2, d)
gates["c_finite_shape"] = {"finite": fin, "shape": list(Hs.shape), "expected": [n, len(layers), 2, d], "abs_max": amax, "stored_dtype": str(Hs.dtype), "pass": bool(fin and shp)}
print("gate c:", json.dumps(gates["c_finite_shape"]), flush=True)
np.savez(os.path.join(a.out, "hs.npz"), hs=Hs, pid=np.arange(n), layers=np.array(layers), prompt_len=plen, comp_len=clen, vllm_tokens=vt)
gates["all_pass"] = bool(all(g["pass"] for g in gates.values())); gates["wall_s"] = time.time() - t0
json.dump(gates, open(os.path.join(a.out, "gates.json"), "w"), indent=1)
if not gates["all_pass"]: print("GATES FAILED", json.dumps(gates), flush=True); sys.exit(2)
print(f"OK, wall {time.time()-t0:.0f}s", flush=True)
