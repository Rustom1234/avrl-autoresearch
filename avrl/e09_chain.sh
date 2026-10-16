#!/bin/bash
# E09 launcher: waits for the p1_goldberg_s1 training run to finish (avrl/runs/p1_goldberg_s1.flag
# contains "p1_goldberg_s1 done") and for the GPU to be idle (<2000 MiB used), then runs generation
# (vLLM, .venv-judge), judging (vLLM, .venv-judge, with 7B->3B fallback inside E09_judge_task.py),
# RM scoring + cheap features (.venv), analysis (.venv), and appends the RESEARCH_LOG.md ledger entry.
# Launch detached: setsid nohup bash avrl/e09_chain.sh > /dev/null 2>&1 < /dev/null & disown
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12
OUT=avrl/runs/E09
mkdir -p "$OUT"

while ! grep -q "p1_goldberg_s1 done" avrl/runs/p1_goldberg_s1.flag 2>/dev/null || \
      [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do
  sleep 60
done
echo "E09 start $(date -u)" >> "$OUT/done.flag"

timeout 2700 .venv-judge/bin/python avrl/exp/E09_judge_task.py --stage gen --out "$OUT" > "$OUT/gen.log" 2>&1
echo "gen exit $?" >> "$OUT/done.flag"

timeout 3600 .venv-judge/bin/python avrl/exp/E09_judge_task.py --stage judge --out "$OUT" > "$OUT/judge.log" 2>&1
echo "judge exit $?" >> "$OUT/done.flag"

.venv/bin/python avrl/exp/E09_judge_task.py --stage score_rm --out "$OUT" > "$OUT/score_rm.log" 2>&1
echo "score_rm exit $?" >> "$OUT/done.flag"

.venv/bin/python avrl/exp/E09_analyze.py --out "$OUT" > "$OUT/analyze.log" 2>&1
echo "analyze exit $?" >> "$OUT/done.flag"

# small python step: generate the RESEARCH_LOG.md ledger entry from result.json, no human needed
.venv/bin/python - "$OUT" <<'PYEOF'
import json, datetime, sys
P = sys.argv[1]
s = json.load(open(f"{P}/summary.json"))
r = json.load(open(f"{P}/result.json"))
p0 = json.load(open("avrl/runs/p0/result.json"))
e8 = json.load(open("avrl/runs/E08/result.json"))
nr = lambda d, k: d.get(k, float("nan"))
date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
verdict = r["verdict"]["label"]
p0_err = p0.get("n_proxy_error_groups", round(p0["proxy_error_rate"] * p0["n_groups"]))

L = []
L.append(f"## E09 {date} {verdict}")
L.append("")
L.append("Hypothesis: every point so far (E01, L5a, E05, E06, E08) has a self-consistency confidence "
          "signal (agreement of sampled answers) with AUC 0.82-0.93 for proxy error, which puts them in "
          "the threshold regime. Claim 3 can only survive on a task where that signal is unavailable or "
          "weak, so E09 uses an open-ended task (HelpSteer2 prompts, free-text answers, no majority vote) "
          "with a small reward model as the cheap proxy and a 7B judge as the expensive verifier.")
L.append("")
L.append(f"Run: Qwen2.5-1.5B-Instruct policy, {s['n_prompts']} HelpSteer2 prompts (nvidia/HelpSteer2, "
         f"deduped to unique prompts, seed 0), {s['G']} rollouts/prompt (vLLM, temperature 1.0, top_p 1.0, "
         f"max_tokens 384). Cheap proxy = Skywork-Reward-V2-Qwen3-0.6B scalar reward score; within-group "
         f"top half -> proxy=1; a separate global threshold ({s['rm_threshold']:.3f}) gives proxy_pass "
         f"positive rate {s['proxy_pass_positive_rate']:.3f} vs gold positive rate {s['gold_positive_rate']:.3f} "
         f"(chosen to match). Expensive verifier = {s['judge_model_used']} (fallback_used="
         f"{s['judge_fallback_used']}), fixed 1-10 helpfulness+correctness rubric, temperature 0, "
         f"max_tokens 8, gold = score>=7; judge output was unparseable (no digit in the 8-token completion) "
         f"in {s['frac_judge_unparseable']:.3%} of rollouts, scored gold_positive=0 in that case. "
         f"Group-level proxy_error = 1 if the group's top-RM response is not gold-positive. Files: "
         f"avrl/runs/E09/summary.json, result.json, feats.parquet, rollouts.parquet, groups.parquet, "
         f"gen/judge/score_rm/analyze logs, sim_*.json, done.flag. Scripts: avrl/exp/E09_judge_task.py "
         f"(gen/judge/score_rm stages, .venv-judge then .venv), avrl/exp/E09_analyze.py (.venv), "
         f"avrl/e09_chain.sh (launcher).")
L.append("")
L.append("| quantity | E01 (GSM8K, majority vote) | E08 (MBPP, output-sig vote) | E09 (HelpSteer2, RM proxy / judge) |")
L.append("|---|---|---|---|")
L.append(f"| n groups (n proxy-error groups) | {p0['n_groups']:,} ({p0_err:,}) | {e8['n_groups']:,} ({e8['n_proxy_error_groups']:,}) | {r['n_groups']:,} ({r['n_proxy_error_groups']:,}) |")
L.append(f"| group proxy-error rate | {p0['proxy_error_rate']:.1%} | {e8['proxy_error_rate']:.1%} | {r['proxy_error_rate']:.1%} |")
L.append(f"| cheap-signal-only AUC (agreement / agreement / consistency) | {p0['auc_agreement_only']:.3f} | {e8['auc_agreement_only']:.3f} | {r['auc_consistency_only']:.3f} |")
L.append(f"| AUC: all cheap / embedding only / cheap+emb | {p0['auc_cheap']:.3f} / {p0['auc_emb_only']:.3f} / {p0['auc_cheap_emb']:.3f} | {e8['auc_cheap']:.3f} / {e8['auc_emb_only']:.3f} / {e8['auc_cheap_emb']:.3f} | {r['auc_cheap']:.3f} / {r['auc_emb_only']:.3f} / {r['auc_cheap_emb']:.3f} |")
L.append(f"| AUC cheap+emb minus cheap | {p0['auc_cheap_emb']-p0['auc_cheap']:.4f} | {e8['auc_cheap_emb']-e8['auc_cheap']:.4f} | {r['auc_cheap_emb']-r['auc_cheap']:.4f} |")
L.append(f"| neighbor ratio by embedding cell (k=30/100/300) | {p0['neighbor_ratio']['cell30']:.2f} / {p0['neighbor_ratio']['cell100']:.2f} / {p0['neighbor_ratio']['cell300']:.2f} | {e8['neighbor_ratio']['cell30']:.2f} / {e8['neighbor_ratio']['cell100']:.2f} / {e8['neighbor_ratio']['cell300']:.2f} | {nr(r['neighbor_ratio'],'cell30'):.2f} / {nr(r['neighbor_ratio'],'cell100'):.2f} / {nr(r['neighbor_ratio'],'cell300'):.2f} |")
L.append(f"| confident subset (def. differs: agreement>=0.75 for E01/E08, top-quartile rm_gap12 for E09), n: error rate / cheap AUC / cheap+emb AUC / neighbor ratio | {p0['confident_subset']['proxy_error_rate']:.1%} / {p0['confident_subset']['auc_cheap']:.2f} / {p0['confident_subset']['auc_cheap_emb']:.2f} / {p0['confident_subset']['neighbor_ratio_cell100']:.2f} (n={p0['confident_subset']['n']:,}) | {e8['confident_subset']['proxy_error_rate']:.1%} / {e8['confident_subset']['auc_cheap']:.2f} / {e8['confident_subset']['auc_cheap_emb']:.2f} / {e8['confident_subset']['neighbor_ratio_cell100']:.2f} (n={e8['confident_subset']['n']:,}) | {r['confident_subset']['proxy_error_rate']:.1%} / {r['confident_subset']['auc_cheap']:.2f} / {r['confident_subset']['auc_cheap_emb']:.2f} / {r['confident_subset']['neighbor_ratio_cell100']:.2f} (n={r['confident_subset']['n']:,}) |")
L.append("")
auc_gap = r["verdict"]["auc_gap"]; nr100 = r["verdict"]["neighbor_ratio_cell100"]; n_err = r["verdict"]["n_proxy_error_groups"]
L.append(f"Verdict against the pre-registered rule (pass if auc_cheap_emb-auc_cheap>=0.05 or "
         f"neighbor_ratio_cell100>=1.5; kill if both fail; inconclusive if fewer than 60 proxy-error "
         f"groups): {verdict.upper()}. Observed auc gap is {auc_gap:.4f} and neighbor_ratio_cell100 is "
         f"{nr100:.3f}; n_proxy_error_groups is {n_err:,}. result.json's own verdict field agrees "
         f"(label {r['verdict']['label']!r}).")
L.append("")
if verdict == "pass":
    change = ("E09 is the first point where structure survives without a discrete self-consistency "
              "channel: embeddings (or the neighbor ratio) add real signal over the reward-model cheap "
              "features on an open-ended task. Claim 3 gets its first support here, and phase 1 should "
              "be rerun on this task/proxy pair before concluding the threshold regime is universal.")
elif verdict == "kill":
    change = ("Even with no discrete self-consistency signal available at all, proxy error (reward-model "
              "top-pick not gold-positive) still does not cluster in prompt-embedding space beyond what "
              "cheap scalar features (RM score stats, length, consistency) already capture. This is the "
              "third kill (E01, E08, E09) across three structurally different proxy mechanisms (majority "
              "vote over answers, majority vote over output signatures, reward-model score), which "
              "narrows claim 3's remaining room: no proxy mechanism tried so far gives embeddings "
              "anything to add.")
else:
    change = (f"Only {n_err} proxy-error groups were observed, below the 60-group inconclusive floor; the "
              "structure read is not reliable here. This point should be rerun with more prompts and/or "
              "a judge rubric threshold tuned to produce more errors before it can inform claim 3.")
L.append(f"What it changes: {change}")
L.append("")
L.append("Follow-ups:")
L.append("1. If kill or inconclusive: the remaining untried axis for claim 3 is a stronger/weaker "
         "judge-proxy capability gap (e.g. a 1.5B proxy against this same 7B judge, or this reward model "
         "against a weaker judge) or a multi-step agentic task, rather than another single-turn task.")
L.append("2. Judge parse failures under max_tokens=8 are an honest data-quality cost of this design; a "
         "targeted retry with more tokens on unparseable rows would tighten the gold labels if this point "
         "is extended or reused.")
L.append(f"3. judge_model_used={s['judge_model_used']} (fallback_used={s['judge_fallback_used']}): if the "
         "7B judge had to fall back to the 3B, the judge itself is weaker than planned and the gold "
         "labels should be treated as noisier than E01/E08's gold (exact answer / test execution).")
L.append("4. The allocation replay (sim_*.json under avrl/runs/E09/) should be folded into the claim-2 "
         "budget curve (L3a/L4a) alongside E01 and E08, same as E08's entry proposed.")
L.append("")

with open("avrl/RESEARCH_LOG.md", "a") as f:
    f.write("\n" + "\n".join(L) + "\n")
print("ledger entry appended for E09, verdict:", verdict)
PYEOF
echo "ledger exit $?" >> "$OUT/done.flag"

echo "E09 done $(date -u)" >> "$OUT/done.flag"
