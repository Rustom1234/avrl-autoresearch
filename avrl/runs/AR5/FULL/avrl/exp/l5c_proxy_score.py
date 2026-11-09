"""L5c stage 0 (GPU): two new proxies scored on the SAME 4,000 rows as L3b (saved E21 T=0.25 rollouts, Qwen2.5-1.5B-Instruct on GSM8K train pids 0..1999, G=8). No generation, no gold in any proxy input.
Copy of exp/l3b_judge_score.py with --kinds and one output dir per kind: <out>/<kind>/scores.parquet, gates.json (same schema as L3b run 2), failed.txt (exact traceback) if a kind raises.
Rows (identical to L3b): per pid 'maj' = first rollout with proxy==1 (carries the majority answer) and 'oth' = lowest-k rollout whose answer differs from the majority, else the lowest-k rollout other than 'maj'.
Kinds (run one at a time on the single A4000, model freed between kinds; a kind that fails to load or raises writes failed.txt and the next kind still runs):
  rm      Skywork-Reward-V2-Qwen3-0.6B via AutoModelForSequenceClassification (bf16, one label). Text = chat template [user: problem, assistant: completion], tokenized with add_special_tokens=False,
          one row at a time (bs=1, so the pad token never enters the sequence). s = the scalar logit. The final-norm hidden state of the last token is multiplied by an fp32 copy of the score head
          (same trick as L3b's fp32 lm_head), so bf16 output rounding (0.125 steps near 20) does not create ties; gate 2c (info, not gating) compares it with model(...).logits on 8 rows.
  judge3b Qwen2.5-3B-Instruct Yes/No judge, the exact L3b run-2 prompt ("... Is the final answer correct? Reply with a single word, Yes or No, and nothing else."), chat template with the generation prompt,
          s = logit(Yes) - logit(No) at the first generated position, fp32 copy of the (tied) lm_head, bs=1.
Gates per kind (same as L3b run 2; a failed gate = failed experiment for that proxy, it contributes no verdict):
  (1) timing probe after --probe_rows rows (longest first): projected total > --max_min (25) minutes -> keep only --sub_pids (1,250) pids (seed 0) and continue in-process;
  (2) determinism: rescoring 8 random kept rows alone reproduces the stored scores within 0.05 (2b: 64 rows in 4 length-sorted batches of 16 against the unbatched scores, informational);
  (3) all s finite; judge3b additionally mean P(Yes)+P(No) over the full vocabulary > 0.9 (rm: finite only, there is no Yes/No mass);
  (4) sanity AUC of s for rollout correctness over all scored rows >= 0.70.
REGISTERED BINS (L5c-proxies, fixed before any real score was read; the same text sits in l5c_analyze.py):
  PASS (claim 3 lead replicates, strong form): BOTH proxies pass their gates, have n_err in ACC >= 100, have top-half cell100 neighbor ratio >= 1.5 and above their own 1,000-draw null p95,
       AND delta AUC (JA + MiniLM pc0..15 minus JA) >= 0.03 with bootstrap CI lower end > 0 for at least one of them.
  weaker reading (NOT a pass, inconclusive): exactly one proxy has ratio >= 1.5 and above its null p95.
  KILL (claim 3 closed for every proxy type tried): BOTH proxies pass gates with n_err in ACC >= 100 and top-half cell100 ratio < 1.3 or <= own null p95 for BOTH.
  Anything else, or a proxy failing a gate (partial) = inconclusive. Cuts reported beside the verdict cut: top 25%, top 75%, all of ACC. The claim 2 replay never changes the verdict."""
import argparse, os, json, time, sys, traceback, gc
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
ap = argparse.ArgumentParser()
ap.add_argument("--kinds", default="rm,judge3b"); ap.add_argument("--rm_model", required=True); ap.add_argument("--judge_model", required=True)
ap.add_argument("--rollouts", required=True); ap.add_argument("--groups", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--per_group", type=int, default=2); ap.add_argument("--probe_rows", type=int, default=200); ap.add_argument("--max_min", type=float, default=25.0)
ap.add_argument("--sub_pids", type=int, default=1250); ap.add_argument("--bs", type=int, default=1)
ap.add_argument("--device", default="cuda"); ap.add_argument("--limit_pids", type=int, default=0)  # last two: CPU smoke test only
a = ap.parse_args(); assert a.per_group == 2; os.makedirs(a.out, exist_ok=True); T0 = time.time(); dev = a.device
R = pd.read_parquet(a.rollouts); G = pd.read_parquet(a.groups).sort_values("pid").reset_index(drop=True)
if a.limit_pids: G = G.iloc[:a.limit_pids]; R = R[R.pid < a.limit_pids]
n = len(G); assert (G.pid.values == np.arange(n)).all()
R = R.sort_values(["pid", "k"]); rows = []
for pid, g in R.groupby("pid"):
    maj = g[g.proxy == 1].iloc[0]; diff = g[g.proxy == 0]; oth = diff.iloc[0] if len(diff) else g[g.k != maj.k].iloc[0]
    rows += [dict(pid=pid, k=int(maj.k), role="maj", correct=int(maj.correct), proxy=int(maj.proxy), tokens=int(maj.tokens), truncated=int(maj.truncated), text=maj.text),
             dict(pid=pid, k=int(oth.k), role="oth", correct=int(oth.correct), proxy=int(oth.proxy), tokens=int(oth.tokens), truncated=int(oth.truncated), text=oth.text)]
D0 = pd.DataFrame(rows); D0["question"] = G.question.values[D0.pid.values]; assert len(D0) == 2 * n


def judge_prompt(q, c): return f"Problem:\n{q}\n\nProposed solution:\n{c}\n\nIs the final answer correct? Reply with a single word, Yes or No, and nothing else."


def run_kind(kind):
    out = os.path.join(a.out, kind); os.makedirs(out, exist_ok=True); t0 = time.time(); D = D0.copy()
    mpath = a.rm_model if kind == "rm" else a.judge_model; tok = AutoTokenizer.from_pretrained(mpath)
    if kind == "rm":
        seqs = [tok(tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": c}], tokenize=False), add_special_tokens=False).input_ids for q, c in zip(D.question, D.text)]
    else:
        seqs = [tok(tok.apply_chat_template([{"role": "user", "content": judge_prompt(q, c)}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids for q, c in zip(D.question, D.text)]
    slen = np.array([len(x) for x in seqs]); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    print(f"[{kind}] {len(seqs)} rows, total tokens {slen.sum()}, max len {slen.max()}, mean {slen.mean():.0f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[{kind}] prompt head:", repr(tok.decode(seqs[0][:24])), flush=True); print(f"[{kind}] prompt tail:", repr(tok.decode(seqs[0][-24:])), flush=True)
    if kind == "rm":
        model = AutoModelForSequenceClassification.from_pretrained(mpath, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval(); base = model.model; Wf = model.score.weight.float()  # [1, d]
        assert Wf.shape[0] == 1 and model.score.bias is None
    else:
        yes_id, no_id = tok.encode("Yes", add_special_tokens=False), tok.encode("No", add_special_tokens=False); assert len(yes_id) == len(no_id) == 1; yes_id, no_id = yes_id[0], no_id[0]
        print(f"[{kind}] Yes id {yes_id} No id {no_id}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(mpath, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval(); base = model.model; Wf = model.lm_head.weight.float()  # [V, d], tied
    print(f"[{kind}] model loaded ({time.time()-t0:.0f}s)", flush=True)

    @torch.no_grad()
    def score(idx):
        """s, Yes/No probability mass (nan for rm), argmax token (-1 for rm) for rows idx (right padding, last real token)."""
        L = max(slen[i] for i in idx); ids = torch.full((len(idx), L), pad, dtype=torch.long); am = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx): ids[r, :slen[i]] = torch.tensor(seqs[i]); am[r, :slen[i]] = 1
        h = base(input_ids=ids.to(dev), attention_mask=am.to(dev)).last_hidden_state  # after the final norm
        last = torch.tensor([slen[i] - 1 for i in idx], device=dev); h = h[torch.arange(len(idx), device=dev), last].float()
        if kind == "rm": return (h @ Wf.T)[:, 0].cpu().numpy(), np.full(len(idx), np.nan), np.full(len(idx), -1)
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
            print(f"[{kind}] gate 1: {json.dumps(gates['1_timing'])}", flush=True)
            if proj / 60 > a.max_min:
                sp = np.sort(np.random.default_rng(0).choice(n, min(a.sub_pids, n), replace=False)); keep = np.where(np.isin(D.pid.values, sp))[0]; ks = set(keep.tolist())
                order = [i for i in order if i in ks and np.isnan(s_all[i])]; pos = 0; gates["1_timing"]["subsampled"] = True; gates["1_timing"]["n_pids_kept"] = int(len(sp))
                print(f"[{kind}] ABORT-AND-SUBSAMPLE: projected {proj/60:.1f} min > {a.max_min}; keeping {len(sp)} pids ({len(keep)} rows)", flush=True)
        if pos // 400 != (pos - len(idx)) // 400: print(f"[{kind}]   {pos}/{len(order)} rows, {time.time()-t1:.0f}s", flush=True)
    print(f"[{kind}] scoring done in {time.time()-t1:.0f}s", flush=True)
    D["s"] = s_all; D["mass"] = m_all; D["top_tok"] = t_all; Dk = D.iloc[keep].copy()
    if gates["1_timing"]["subsampled"]: Dk = Dk[np.isfinite(Dk.s)]
    Dk.drop(columns=["text", "question"]).to_parquet(os.path.join(out, "scores.parquet"), index=False)
    # gate 2: rescoring 8 random kept rows alone reproduces the stored scores (the main pass is unbatched, so this is a determinism check)
    rng = np.random.default_rng(0); chk = [int(x) for x in rng.choice(Dk.index.values, 8, replace=False)]; dd = []
    for i in chk: su, _, _ = score([i]); dd.append(float(abs(su[0] - s_all[i])))
    gates["2_unbatched"] = {"rows": chk, "max_abs_diff": float(max(dd)), "mean_abs_diff": float(np.mean(dd)), "tol": 0.05, "pass": bool(max(dd) < 0.05), "note": "main pass is unbatched (bs=1), so this is a determinism check"}
    print(f"[{kind}] gate 2:", json.dumps(gates["2_unbatched"]), flush=True)
    ks = set(Dk.index.tolist()); srt = [int(i) for i in np.argsort(-slen) if int(i) in ks]; st = int(rng.integers(0, max(len(srt) - 64, 1))); bd = []
    if len(srt) >= 64:  # gate 2b (informational): bf16 batch noise, 4 consecutive length-sorted batches of 16 against the unbatched scores
        for j in range(4):
            idx = srt[st + 16 * j: st + 16 * (j + 1)]; sb, _, _ = score(idx); bd += [float(abs(sb[r] - s_all[i])) for r, i in enumerate(idx)]
        gates["2b_batch_noise_info"] = {"n_rows": len(bd), "start_in_sorted_order": st, "max_abs_diff": float(max(bd)), "mean_abs_diff": float(np.mean(bd)), "s_sd": float(Dk.s.std()), "gating": False}
        print(f"[{kind}] gate 2b (info):", json.dumps(gates["2b_batch_noise_info"]), flush=True)
    if kind == "rm":  # gate 2c (info): the fp32 score head against the model's own bf16 logits
        with torch.no_grad():
            dl = [float(abs(model(input_ids=torch.tensor([seqs[i]], device=dev)).logits[0, 0].float().item() - s_all[i])) for i in chk]
        gates["2c_fp32_head_vs_model_logits_info"] = {"rows": chk, "max_abs_diff": float(max(dl)), "mean_abs_diff": float(np.mean(dl)), "gating": False}
        print(f"[{kind}] gate 2c (info):", json.dumps(gates["2c_fp32_head_vs_model_logits_info"]), flush=True)
    fin = bool(np.isfinite(Dk.s.values).all()); mass_ok = True
    if kind == "judge3b": fin = bool(fin and np.isfinite(Dk.mass.values).all()); mass_ok = bool(Dk.mass.mean() > 0.9)
    gates["3_finite_mass"] = {"finite": fin, "n_rows": int(len(Dk)), "mass_required": kind == "judge3b", "mean_mass": float(Dk.mass.mean()) if kind == "judge3b" else None,
                              "min_mass": float(Dk.mass.min()) if kind == "judge3b" else None, "pass": bool(fin and mass_ok)}
    print(f"[{kind}] gate 3:", json.dumps(gates["3_finite_mass"]), flush=True)
    if kind == "judge3b":
        lowm = Dk[Dk.mass < 0.9]; tc = lowm.top_tok.value_counts().head(8); gates["3b_low_mass_top_tokens"] = {"n_low_mass_rows": int(len(lowm)), "frac_low_mass": float(len(lowm) / len(Dk)), "top": {repr(tok.decode([int(t)])): int(c) for t, c in tc.items()}, "gating": False}
        print(f"[{kind}] gate 3b (info):", json.dumps(gates["3b_low_mass_top_tokens"]), flush=True)
    y = Dk.correct.values; rk = pd.Series(Dk.s.values).rank(method="average").values; n1 = int(y.sum()); n0 = len(y) - n1  # gate 4: sanity AUC (Mann-Whitney via average ranks)
    auc = float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    gates["4_sanity_auc"] = {"auc_s_vs_correct": auc, "n_rows": int(len(y)), "n_correct": n1, "threshold": 0.70, "pass": bool(auc >= 0.70)}
    print(f"[{kind}] gate 4:", json.dumps(gates["4_sanity_auc"]), flush=True)
    gates["all_pass"] = bool(all(g["pass"] for k, g in gates.items() if k in ("2_unbatched", "3_finite_mass", "4_sanity_auc"))); gates["wall_s"] = time.time() - t0
    json.dump(gates, open(os.path.join(out, "gates.json"), "w"), indent=1)
    del model, base, Wf; gc.collect()
    if dev.startswith("cuda"): torch.cuda.empty_cache()
    return "ok" if gates["all_pass"] else "gates_failed"


status = {}
for kind in a.kinds.split(","):
    try: status[kind] = run_kind(kind)
    except Exception:
        tb = traceback.format_exc(); status[kind] = "exception"; os.makedirs(os.path.join(a.out, kind), exist_ok=True)
        open(os.path.join(a.out, kind, "failed.txt"), "w").write(tb); print(f"[{kind}] FAILED:\n{tb}", flush=True)
        gc.collect()
        if dev.startswith("cuda"): torch.cuda.empty_cache()
json.dump({"status": status, "wall_s": time.time() - T0}, open(os.path.join(a.out, "score_status.json"), "w"), indent=1); print("status:", json.dumps(status), f"wall {time.time()-T0:.0f}s", flush=True)
sys.exit(0 if all(v == "ok" for v in status.values()) else 2)
