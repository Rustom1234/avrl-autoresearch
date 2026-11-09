"""L3b stage 0 (GPU): Qwen3-4B Yes/No judge score for saved E21 T=0.25 rollouts. No generation, no gold in the prompt.
Per pid two completions are scored: 'maj' = first rollout with proxy==1 (the completion carrying the majority answer, as in L3a) and
'oth' = chosen label-free: the lowest-k rollout whose answer differs from the majority if the group has one, else the lowest-k rollout other than 'maj'.
Judge prompt = Qwen3 chat template, enable_thinking=False, user message: problem, proposed solution, 'Is the final answer correct? Reply Yes or No.'
Score s = logit(Yes) - logit(No) at the first generated position (one forward pass). The final norm output is cast to float32 and multiplied by an fp32
copy of the (tied) lm_head, so bf16 logit rounding (0.125 steps near 20) does not quantize s.
Writes <out>/scores.parquet (pid, k, role, s, mass, correct, proxy, tokens, truncated), <out>/gates.json. Nonzero exit if a gate fails (scores are saved first).
RUN 2 CHANGES (the single allowed fix after run 1 failed gates 2 and 3, see runs/L3b-judge-proxy/gates_run1.json): (i) prompt ends "Reply with a single word, Yes or No, and nothing else."
(run 1: "Reply Yes or No." gave mean Yes+No mass 0.837, 27% of rows below 0.9, because the judge starts explaining on uncertain rows); the change was chosen from the mass diagnostic alone;
(ii) rows are scored one at a time (--bs 1): run 1 batched vs unbatched differed by up to 0.69 (mean 0.25 on an s spread of SD 12), which is bf16 kernel-shape noise, not a masking bug, so gate 2 as written
(8 rows batched vs unbatched within 0.05) cannot hold at bf16; gate 2 now checks that rescoring 8 rows alone reproduces the stored scores within 0.05 (determinism), and the batch noise is reported, not gated
(gate 2b: 64 rows in 4 length-sorted batches of 16, the run 1 configuration, against the unbatched scores); (iii) the argmax token at the first position is saved (top_tok) to diagnose low-mass rows.
Gates: (1) timing probe after --probe_rows rows (longest first, conservative): projected total > --max_min -> keep only --sub_pids pids (seed 0) and continue in-process;
(2) 8 random rows unbatched vs batched: max |delta s| < 0.05; (3) all s finite and mean P(Yes)+P(No) over the full vocab > 0.9;
(4) sanity AUC of s for rollout correctness over all scored rows >= 0.70, else the judge is broken (failed experiment, not a verdict)."""
import argparse, os, json, time, sys
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--rollouts", required=True); ap.add_argument("--groups", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--per_group", type=int, default=2); ap.add_argument("--probe_rows", type=int, default=200); ap.add_argument("--max_min", type=float, default=25.0)
ap.add_argument("--sub_pids", type=int, default=1250); ap.add_argument("--bs", type=int, default=1)
ap.add_argument("--device", default="cuda"); ap.add_argument("--limit_pids", type=int, default=0)  # last two: CPU smoke test only
a = ap.parse_args(); assert a.per_group == 2; os.makedirs(a.out, exist_ok=True); t0 = time.time(); dev = a.device
R = pd.read_parquet(a.rollouts); G = pd.read_parquet(a.groups).sort_values("pid").reset_index(drop=True)
if a.limit_pids: G = G.iloc[:a.limit_pids]; R = R[R.pid < a.limit_pids]
n = len(G); assert (G.pid.values == np.arange(n)).all()
R = R.sort_values(["pid", "k"]); rows = []
for pid, g in R.groupby("pid"):
    maj = g[g.proxy == 1].iloc[0]; diff = g[g.proxy == 0]; oth = diff.iloc[0] if len(diff) else g[g.k != maj.k].iloc[0]
    rows += [dict(pid=pid, k=int(maj.k), role="maj", correct=int(maj.correct), proxy=int(maj.proxy), tokens=int(maj.tokens), truncated=int(maj.truncated), text=maj.text),
             dict(pid=pid, k=int(oth.k), role="oth", correct=int(oth.correct), proxy=int(oth.proxy), tokens=int(oth.tokens), truncated=int(oth.truncated), text=oth.text)]
D = pd.DataFrame(rows); D["question"] = G.question.values[D.pid.values]; assert len(D) == 2 * n
tok = AutoTokenizer.from_pretrained(a.model)
def prompt(q, c): return f"Problem:\n{q}\n\nProposed solution:\n{c}\n\nIs the final answer correct? Reply with a single word, Yes or No, and nothing else."
seqs = [tok(tok.apply_chat_template([{"role": "user", "content": prompt(q, c)}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False).input_ids
        for q, c in zip(D.question, D.text)]
slen = np.array([len(x) for x in seqs]); yes_id, no_id = tok.encode("Yes", add_special_tokens=False), tok.encode("No", add_special_tokens=False); assert len(yes_id) == len(no_id) == 1
yes_id, no_id = yes_id[0], no_id[0]
print(f"{len(seqs)} rows, total tokens {slen.sum()}, max len {slen.max()}, mean {slen.mean():.0f}; Yes id {yes_id} No id {no_id} ({time.time()-t0:.0f}s)", flush=True)
print("prompt tail:", repr(tok.decode(seqs[0][-24:])), flush=True)
pad = tok.pad_token_id if tok.pad_token_id is not None else 0
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
Wf = model.lm_head.weight.float()  # [V, d] fp32 copy (tied with the embeddings in Qwen3-4B)
print(f"model loaded ({time.time()-t0:.0f}s)", flush=True)

@torch.no_grad()
def score(idx):
    """s and Yes/No probability mass for rows idx (right padding, last real token)."""
    L = max(slen[i] for i in idx); ids = torch.full((len(idx), L), pad, dtype=torch.long); am = torch.zeros((len(idx), L), dtype=torch.long)
    for r, i in enumerate(idx): ids[r, :slen[i]] = torch.tensor(seqs[i]); am[r, :slen[i]] = 1
    h = model.model(input_ids=ids.to(dev), attention_mask=am.to(dev)).last_hidden_state  # after the final norm
    last = torch.tensor([slen[i] - 1 for i in idx], device=dev); h = h[torch.arange(len(idx), device=dev), last].float()
    lg = h @ Wf.T; lse = torch.logsumexp(lg, -1); ly, ln = lg[:, yes_id], lg[:, no_id]
    return (ly - ln).cpu().numpy(), torch.exp(torch.logaddexp(ly, ln) - lse).cpu().numpy(), lg.argmax(-1).cpu().numpy()

keep = np.arange(len(D)); gates = {"1_timing": {"subsampled": False, "note": "probe not reached"}}
order = [int(i) for i in np.argsort(-slen)]  # longest first: the early throughput is the conservative one
s_all = np.full(len(D), np.nan); m_all = np.full(len(D), np.nan); t_all = np.full(len(D), -1); done_tok = 0; t1 = time.time(); checked = False; pos = 0
while pos < len(order):
    idx = order[pos:pos + a.bs]; s, m, tt = score(idx); s_all[idx] = s; m_all[idx] = m; t_all[idx] = tt; done_tok += int(slen[idx].sum()); pos += len(idx)
    if not checked and pos >= a.probe_rows:
        checked = True; el = time.time() - t1; proj = el * slen.sum() / done_tok
        gates["1_timing"] = {"rows": pos, "tokens": done_tok, "s": el, "tok_per_s": done_tok / el, "projected_total_min": proj / 60, "limit_min": a.max_min, "subsampled": False}
        print(f"gate 1: {json.dumps(gates['1_timing'])}", flush=True)
        if proj / 60 > a.max_min:
            sp = np.sort(np.random.default_rng(0).choice(n, min(a.sub_pids, n), replace=False)); keep = np.where(np.isin(D.pid.values, sp))[0]; ks = set(keep.tolist())
            order = [i for i in order if i in ks and np.isnan(s_all[i])]; pos = 0; gates["1_timing"]["subsampled"] = True; gates["1_timing"]["n_pids_kept"] = int(len(sp))
            print(f"ABORT-AND-SUBSAMPLE: projected {proj/60:.1f} min > {a.max_min}; keeping {len(sp)} pids ({len(keep)} rows)", flush=True)
    if pos // 400 != (pos - len(idx)) // 400: print(f"  {pos}/{len(order)} rows, {time.time()-t1:.0f}s", flush=True)
print(f"scoring done in {time.time()-t1:.0f}s", flush=True)
D["s"] = s_all; D["mass"] = m_all; D["top_tok"] = t_all; Dk = D.iloc[keep].copy()
if gates["1_timing"]["subsampled"]: Dk = Dk[np.isfinite(Dk.s)]
Dk.drop(columns=["text", "question"]).to_parquet(os.path.join(a.out, "scores.parquet"), index=False)

# gate 2 (run 2): rescoring 8 random kept rows alone reproduces the stored scores (the main pass is unbatched, so this is a determinism check)
rng = np.random.default_rng(0); chk = [int(x) for x in rng.choice(Dk.index.values, 8, replace=False)]; dd = []
for i in chk: su, _, _ = score([i]); dd.append(float(abs(su[0] - s_all[i])))
gates["2_unbatched"] = {"rows": chk, "max_abs_diff": float(max(dd)), "mean_abs_diff": float(np.mean(dd)), "tol": 0.05, "pass": bool(max(dd) < 0.05), "note": "main pass is unbatched (bs=1), so this is a determinism check"}
print("gate 2:", json.dumps(gates["2_unbatched"]), flush=True)
# gate 2b (informational, not gating): bf16 batch noise, 4 consecutive length-sorted batches of 16 (the run 1 configuration) against the unbatched scores
ks = set(Dk.index.tolist()); srt = [int(i) for i in np.argsort(-slen) if int(i) in ks]; st = int(rng.integers(0, max(len(srt) - 64, 1))); bd = []
if len(srt) >= 64:
    for j in range(4):
        idx = srt[st + 16 * j: st + 16 * (j + 1)]; sb, _, _ = score(idx); bd += [float(abs(sb[r] - s_all[i])) for r, i in enumerate(idx)]
    gates["2b_batch_noise_info"] = {"n_rows": len(bd), "start_in_sorted_order": st, "max_abs_diff": float(max(bd)), "mean_abs_diff": float(np.mean(bd)), "s_sd": float(Dk.s.std()), "gating": False}
    print("gate 2b (info):", json.dumps(gates["2b_batch_noise_info"]), flush=True)
# gate 3: finite and Yes/No mass
fin = bool(np.isfinite(Dk.s.values).all() and np.isfinite(Dk.mass.values).all()); gates["3_finite_mass"] = {"finite": fin, "n_rows": int(len(Dk)), "mean_mass": float(Dk.mass.mean()), "min_mass": float(Dk.mass.min()), "pass": bool(fin and Dk.mass.mean() > 0.9)}
print("gate 3:", json.dumps(gates["3_finite_mass"]), flush=True)
lowm = Dk[Dk.mass < 0.9]; tc = lowm.top_tok.value_counts().head(8); gates["3b_low_mass_top_tokens"] = {"n_low_mass_rows": int(len(lowm)), "frac_low_mass": float(len(lowm) / len(Dk)), "top": {repr(tok.decode([int(t)])): int(c) for t, c in tc.items()}, "gating": False}
print("gate 3b (info):", json.dumps(gates["3b_low_mass_top_tokens"]), flush=True)
# gate 4: sanity AUC of s for rollout correctness (Mann-Whitney via average ranks)
y = Dk.correct.values; rk = pd.Series(Dk.s.values).rank(method="average").values; n1 = int(y.sum()); n0 = len(y) - n1
auc = float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
gates["4_sanity_auc"] = {"auc_s_vs_correct": auc, "n_rows": int(len(y)), "n_correct": n1, "threshold": 0.70, "pass": bool(auc >= 0.70)}
print("gate 4:", json.dumps(gates["4_sanity_auc"]), flush=True)
gates["all_pass"] = bool(all(g["pass"] for k, g in gates.items() if k in ("2_unbatched", "3_finite_mass", "4_sanity_auc"))); gates["wall_s"] = time.time() - t0
json.dump(gates, open(os.path.join(a.out, "gates.json"), "w"), indent=1)
if not gates["all_pass"]: print("GATES FAILED", json.dumps(gates), flush=True); sys.exit(2)
print(f"OK, wall {time.time()-t0:.0f}s", flush=True)
