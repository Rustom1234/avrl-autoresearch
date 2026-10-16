"""Phase 1 of verification-aware RL, v3: GRPO-train allenai/OLMo-2-0425-1B (base) on MBPP code
generation with the same pluggable, budget-capped verification allocator as p1_grpo.py and
p1_judge_grpo.py. Read p1_grpo.py's and allocators.py's module docstrings first; this file reuses
both unmodified (import only) and only swaps the reward mechanics for code: cheap proxy =
output-signature majority vote, expensive verifier = the problem's real test asserts, both
executed in the subprocess sandbox in code_rewards.py / avrl/exp/_e10_verify_worker.py (ported
from avrl/exp/E08_mbpp.py's prompt/extraction logic and exp/_e08_verify_worker.py's execution
protocol -- see code_rewards.py's module docstring for why those are ported, not imported).

OLMo-2-0425-1B base has NO chat template (confirmed: tokenizer_config.json has no "chat_template"
key, vs. the -Instruct checkpoint, which does). Every training prompt and every eval of the
base/trained policy therefore uses code_rewards.build_prompt(): a plain-text instruction +
problem (+ first test assert, so the function name is known -- MBPP's problem text alone does
not name it) + a bare "Answer:" line + an OPENED ```python fence, with no closing fence --
the model is expected to continue directly inside it (see extract_code()'s docstring for the
matching extraction, which also tolerates a model that echoes its own fresh fence). --eval_only
additionally supports evaluating a model that DOES have a chat template (the Instruct reference
bar): build_eval_prompt() below checks `tok.chat_template` and wraps the same instruction/problem
content as a chat "user" turn instead, via apply_chat_template, so the reference bar is not
unfairly handicapped by a prompt format built for a base model. This is a design choice the task
spec left open -- stated explicitly per Karpathy rule 1, not picked silently.

Same four arms, same n_verify()/select()/update() allocator contract, same per-step log schema
(expensive_calls, corrupted_reward_groups, proxy_error_rate_among_unverified) as p1_grpo.py.
Fairness rules from p1_grpo.py/allocators.py are re-marked inline where they recur here.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd

import code_rewards
from allocators import ALLOCATORS
from avrl_router.diagnose import diagnose, _grid_table_lines, _realizable_table_lines
from p1_grpo import PC_COLS, TimeLimit, build_grpo_config, completion_text, compute_prompt_feats

ROOT = os.path.dirname(os.path.abspath(__file__))  # avrl/ -- matches p1_grpo.py's --model/--data
                                                      # convention (hannah copies models/data here)


def make_hf_dataset(df):
    """df: MBPP rows with pid/text/test_list/test_setup_code. One record per training prompt;
    TRL's GRPOTrainer repeats each record G times (num_generations=G) before calling the reward
    function, matching p1_grpo.py's dataset convention exactly."""
    from datasets import Dataset

    records = [
        {
            "prompt": code_rewards.build_prompt(text, test_list[0]),
            "pid": int(pid),
            "test_list": list(test_list),
            "test_setup_code": setup or "",
        }
        for text, test_list, setup, pid in zip(df.text, df.test_list, df.test_setup_code, df.pid)
    ]
    return Dataset.from_list(records)


def load_eval_subset(test_path, n, seed, eval_seed=None):
    """Fixed, seeded subset of the MBPP *test* split only -- disjoint from training data by
    construction (different file, never merged with train_df), same convention as p1_grpo.py's
    load_eval_prompts. n >= len(df) returns the whole split (used by --eval_only's full-set
    reference-bar runs). eval_seed, when given, selects the subset instead of seed (see
    --eval_seed); everything else (training prompt order, generation sampling) is unaffected
    and still keyed on seed."""
    df = pd.read_parquet(test_path)
    n = min(n, len(df)) if n else len(df)
    pick = seed if eval_seed is None else eval_seed
    return df.sample(n=n, random_state=pick).reset_index(drop=True)


def build_eval_prompt(tok, text, first_assert=None):
    """Plain-text (code_rewards.build_prompt) if `tok` has no chat template -- OLMo-2-0425-1B
    base and every LoRA checkpoint trained from it, since LoRA does not add a template -- else
    wrap the SAME instruction/problem content as a chat 'user' turn (the Instruct reference
    bar). See this file's module docstring."""
    if getattr(tok, "chat_template", None):
        content = code_rewards.build_instruction_content(text, first_assert)
        return tok.apply_chat_template([{"role": "user", "content": content}],
                                        tokenize=False, add_generation_prompt=True)
    return code_rewards.build_prompt(text, first_assert)


def greedy_batch_generate(model, tok, prompts, max_new_tokens, batch_size):
    """Greedy-decode `prompts` in chunks of `batch_size`, left-padded, returning only the newly
    generated text per prompt (completion tokens, not the echoed prompt) -- same pattern as
    p1_grpo.py's PeriodicEval / p1_judge_grpo.py's JudgeEval."""
    import torch

    was_training = model.training
    model.eval()
    out_texts = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            out_texts.extend(tok.batch_decode(gen, skip_special_tokens=True))
    if was_training:
        model.train()
    return out_texts


def eval_mbpp_pass1(model, tok, df, max_new_tokens, batch_size, timeout, sandbox_workers):
    """Greedy pass@1 on an MBPP dataframe (all its own test_list asserts), through the exact same
    sandbox as training (code_rewards.run_batch): one flat task list for the whole eval set, run
    concurrently, not row by row."""
    prompts = [build_eval_prompt(tok, text, test_list[0]) for text, test_list in zip(df.text, df.test_list)]
    completions = greedy_batch_generate(model, tok, prompts, max_new_tokens, batch_size)
    codes = [code_rewards.extract_code(c) for c in completions]
    tasks = [code_rewards.build_mbpp_task(code, test_list, setup)
             for code, test_list, setup in zip(codes, df.test_list, df.test_setup_code)]
    results = code_rewards.run_batch(tasks, timeout=timeout, max_workers=sandbox_workers)
    return float(np.mean([r["correct"] for r in results])) if results else float("nan")


def eval_humaneval_pass1(model, tok, df, max_new_tokens, batch_size, timeout, sandbox_workers):
    """Greedy pass@1 on a HumanEval dataframe, via its own check(candidate) harness (see
    code_rewards.build_humaneval_task). No first_assert needed: HumanEval's `prompt` field
    already contains the full function signature, so the model is never told the function name
    separately."""
    prompts = [build_eval_prompt(tok, p) for p in df.prompt]
    completions = greedy_batch_generate(model, tok, prompts, max_new_tokens, batch_size)
    codes = [code_rewards.extract_code(c) for c in completions]
    tasks = [code_rewards.build_humaneval_task(code, test, ep)
             for code, test, ep in zip(codes, df.test, df.entry_point)]
    results = code_rewards.run_batch(tasks, timeout=timeout, max_workers=sandbox_workers)
    return float(np.mean([r["correct"] for r in results])) if results else float("nan")


def _format_burnin_summary(result, budget_mode, budget):
    """Plain-text summary of a burn-in diagnose() result: n groups, proxy error rate with CI,
    AUC with CI, corruption per budget (threshold vs random), recommended budget, regime and
    its one-line recommendation. Used for both the stdout print and burnin_report.md -- same
    numbers as diagnose.py's own --out json, formatted for a human skim rather than re-parsed."""
    per = result["proxy_error_rate"]
    c = result["confidence_columns"]["agreement"]
    auc = c["auc"]
    lines = [
        "# Burn-in diagnosis",
        "",
        f"n groups: {result['n_groups']}",
        f"proxy error rate: {per['point']:.4f} (95% CI {per['ci95_lo']:.4f} to {per['ci95_hi']:.4f})",
    ]
    if np.isnan(auc["point"]):
        lines.append("AUC: undefined (no error variance in the burn-in sample)")
    else:
        lines.append(f"AUC (low confidence predicts error): {auc['point']:.4f} "
                      f"(95% CI {auc['ci95_lo']:.4f} to {auc['ci95_hi']:.4f})")
    lines.append("")
    lines.append("corruption by budget, threshold vs random:")
    for b, br in c["budgets"].items():
        t, r = br["threshold"], br["random"]
        lines.append(f"  budget {b}: threshold {t['point']:.4f} "
                      f"(CI {t['ci95_lo']:.4f} to {t['ci95_hi']:.4f}) vs random {r['point']:.4f} "
                      f"(CI {r['ci95_lo']:.4f} to {r['ci95_hi']:.4f})")
    lines.append("")
    lines.extend(_grid_table_lines(c))
    if c.get("realizable"):
        lines.append("")
        lines.extend(_realizable_table_lines(c))
    lines.append("")
    lines.append(f"target: {c['target_mode']} {c['effective_target']:.4f} (base error {c['base_error']:.4f})")
    lines.append(f"recommended budget for target corruption {result['target_corruption']}: "
                 f"{c['recommended_budget']:.2f}")
    lines.append(f"regime: {c['regime']} - {c['recommendation']}")
    lines.append("")
    if budget_mode == "auto":
        lines.append(f"budget mode: auto, chosen budget for the rest of training: {budget:.2f}")
    else:
        lines.append(f"budget mode: fixed, training budget stays at {budget:.2f} as configured")
    return "\n".join(lines) + "\n"


class CodeRewardState:
    """GRPOTrainer reward function for E10 (MBPP, signature-verified). Same call contract and
    per-step bookkeeping as p1_grpo.py's RewardState (one call == one step's groups, self.call_idx
    lines up with the optimizer step 1:1); cheap proxy = output-signature majority vote, expensive
    verifier = the problem's real asserts, instead of GSM8K's free string-compare gold.

    The real-asserts check (`correct`) is computed for EVERY program in EVERY group, verified or
    not -- exactly like p1_grpo.py always has `gold` for free from the dataset -- because in this
    sandbox the signature probe and the real-asserts check are the SAME subprocess call
    (_e10_verify_worker.py computes sig, then runs the real asserts, in one already-exec'd
    namespace): there is no separate, literally-more-expensive code path to gate here. What the
    allocator's budget controls is which of the two results (gold `correct` vs. the cheap
    sig-match) actually BACKS the training reward and feeds LearnedAllocator.update() -- the same
    thing p1_grpo.py's budget controls. See its RewardState docstring and allocators.py's module
    docstring for the fairness invariants re-marked inline below."""

    def __init__(self, allocator, budget, G, feats, log_path, seed, max_completion_length,
                 timeout=5.0, sandbox_workers=None, proxy="signature", burnin_steps=0,
                 target_corruption=0.02, budget_mode="fixed", prompts_per_step=None,
                 budget_pick="point"):
        self.allocator = allocator
        self.budget = budget
        self.G = G
        self.feats = feats  # DataFrame indexed by pid: cell, q_len, pc_0..pc_15 (p1_grpo.py)
        self.proxy = proxy  # E10b: "signature" (default, unchanged) or "visible" -- see --proxy
        self.log_f = open(log_path, "a")
        groups_path = os.path.join(os.path.dirname(log_path), "groups.jsonl")
        self.groups_f = open(groups_path, "a")
        self.seed = seed
        self.max_completion_length = max_completion_length
        self.timeout = timeout
        self.sandbox_workers = sandbox_workers
        self.call_idx = 0
        self.__name__ = f"reward_{allocator.name}"  # TRL logs reward_funcs by __name__
        # Burn-in diagnostic (product piece 2). burnin_steps=0 (default) is a no-op everywhere
        # below -- see the in_burnin branch in __call__ and _run_burnin_diagnostic's docstring.
        self.burnin_steps = burnin_steps
        self.target_corruption = target_corruption
        self.budget_mode = budget_mode  # "fixed" (--budget is a number) or "auto"
        # PRODUCT-4: groups verified per step is fixed at prompts_per_step (n_groups per __call__),
        # so the burn-in recommendation must pick from budgets realizable at that granularity --
        # see _run_burnin_diagnostic and RESEARCH_LOG.md's E10e-rel entry for the bug this fixes.
        self.prompts_per_step = prompts_per_step
        self.budget_pick = budget_pick
        self.out_dir = os.path.dirname(log_path)
        self._burnin_rows = []  # group_row dicts from the burn-in steps only, for diagnose()

    def __call__(self, prompts, completions, completion_ids=None, **kwargs):
        pid_col = kwargs["pid"]
        test_list_col = kwargs["test_list"]
        setup_col = kwargs["test_setup_code"]
        G = self.G
        n_total = len(completions)
        assert n_total % G == 0, f"batch of {n_total} completions not divisible by G={G}"
        n_groups = n_total // G
        for gi in range(n_groups):
            sl = slice(gi * G, (gi + 1) * G)
            assert len(set(pid_col[sl])) == 1, \
                "reward fn assumes TRL groups completions contiguously by prompt; violated for this batch"
        assert completion_ids is not None, (
            "completion_ids is None; this TRL version does not supply it as this file's module "
            "docstring (and p1_grpo.py's) assumes, so frac_trunc cannot be computed correctly"
        )
        lens = [len(x) for x in completion_ids]

        texts = [completion_text(c) for c in completions]
        codes = [code_rewards.extract_code(t) for t in texts]

        # One flat task list for the WHOLE step's n_groups*G programs, run through ONE thread
        # pool, so all of them (64 at prompts_per_step=8, G=8) run concurrently on CPU cores, not
        # G at a time -- see code_rewards.run_batch's docstring.
        tasks = [
            code_rewards.build_mbpp_task(codes[i], test_list_col[i], setup_col[i])
            for i in range(n_total)
        ]
        # --proxy visible (E10b): a second sandbox call per completion, against test_list[:1]
        # only (the visible test) -- same two-call pattern exp/E10p0_rollouts.py uses, batched
        # into the SAME run_batch() call as the full-test tasks so the whole step's sandbox work
        # still runs concurrently (not two sequential waves). Default --proxy signature path is
        # untouched: one run_batch() call on `tasks` only, exactly as before.
        if self.proxy == "visible":
            visible_tasks = [
                code_rewards.build_mbpp_task(codes[i], test_list_col[i][:1], setup_col[i])
                for i in range(n_total)
            ]
            all_results = code_rewards.run_batch(tasks + visible_tasks, timeout=self.timeout,
                                                  max_workers=self.sandbox_workers)
            results, visible_results = all_results[:n_total], all_results[n_total:]
        else:
            results = code_rewards.run_batch(tasks, timeout=self.timeout, max_workers=self.sandbox_workers)
            visible_results = None

        records = []
        for gi in range(n_groups):
            sl = slice(gi * G, (gi + 1) * G)
            pid = pid_col[sl][0]
            group_results = results[sl]
            corrects = [r["correct"] for r in group_results]
            if self.proxy == "visible":
                visible_pass = [int(r["correct"]) for r in visible_results[sl]]
                agreement, n_distinct = code_rewards.visible_signature_stats(group_results, visible_pass)
                majority_sig = None
                # Corrupted iff some completion fools the cheap grader: passes the visible test
                # but fails the real (full-test) asserts. Group-level ("at least one"), not a
                # majority vote, because the reward itself is per-completion in this proxy mode.
                proxy_error_flag = int(any(vp and not c for vp, c in zip(visible_pass, corrects)))
            else:
                majority_sig, agreement, n_distinct, majority_correct = code_rewards.group_signature_stats(group_results, G)
                visible_pass = None
                proxy_error_flag = 1 - majority_correct
            group_lens = lens[sl]
            mean_len = float(np.mean(group_lens))
            std_len = float(np.std(group_lens))
            frac_trunc = float(np.mean([l >= self.max_completion_length for l in group_lens]))
            prow = self.feats.loc[pid]
            # FAIRNESS RULE 1: feat_vec is built only from completions + the gold-free startup
            # embedding cache; no gold (proxy_error_flag/sigs/corrects/visible_pass withheld below).
            cheap = [agreement, n_distinct, mean_len, std_len, frac_trunc, float(prow.q_len)]
            feat_vec = cheap + [float(prow[c]) for c in PC_COLS]
            records.append(dict(
                idx=gi, pid=pid, majority_sig=majority_sig, agreement=agreement, n_distinct=n_distinct,
                cell=int(prow.cell), feat_vec=feat_vec, proxy_error_flag=proxy_error_flag,
                sigs=[r["sig"] for r in group_results], corrects=corrects, visible_pass=visible_pass,
            ))

        in_burnin = self.call_idx < self.burnin_steps
        if in_burnin:
            # Burn-in diagnostic (product piece 2): every group verified with the expensive
            # grader for the first --burnin_steps steps, regardless of arm, so the policy gets
            # clean rewards here and every burn-in group carries a TRUE proxy_error label for
            # _run_burnin_diagnostic below. Bypasses the allocator entirely for these steps;
            # default behaviour (--burnin_steps 0) never takes this branch.
            n_verify = n_groups
            verify_idx = {r["idx"] for r in records}
        else:
            # FAIRNESS RULE 2: one pure function of (n_groups, budget), called fresh each step --
            # identical expensive-call count for every arm at the same budget, no carry-over.
            n_verify = self.allocator.n_verify(n_groups, self.budget)
            rng = np.random.default_rng(self.seed * 1_000_003 + self.call_idx)
            # FAIRNESS RULE 1 (structural): strip gold (proxy_error_flag/sigs/corrects) and the
            # visible-pass flags (not gold, but not needed by any allocator either) before handing
            # records to the allocator, so no select() implementation -- current or future -- can
            # read it.
            safe_records = [{k: v for k, v in r.items()
                              if k not in ("proxy_error_flag", "sigs", "corrects", "visible_pass")}
                             for r in records]
            verify_idx = set(self.allocator.select(safe_records, n_verify, rng))
            assert len(verify_idx) == n_verify, f"{self.allocator.name}.select returned {len(verify_idx)}, expected {n_verify}"

        rewards = [0.0] * n_total
        verified_for_update = []  # FAIRNESS RULE 3: only groups verified this step
        proxy_err_unverified = []
        corrupted = 0
        for rec in records:
            gi = rec["idx"]
            # gold (real asserts, baked into proxy_error_flag) used for logging/reward only, never a feature
            proxy_wrong = rec["proxy_error_flag"]
            verified = gi in verify_idx
            if verified:
                for k in range(G):
                    rewards[gi * G + k] = float(rec["corrects"][k])
                verified_for_update.append((rec["feat_vec"], proxy_wrong, rec["cell"]))
            else:
                if self.proxy == "visible":
                    for k in range(G):
                        rewards[gi * G + k] = float(rec["visible_pass"][k])
                else:
                    for k in range(G):
                        rewards[gi * G + k] = float(rec["sigs"][k] == rec["majority_sig"])
                proxy_err_unverified.append(proxy_wrong)
                corrupted += proxy_wrong
            group_row = dict(step=self.call_idx, idx=gi, agreement=rec["agreement"], n_distinct=rec["n_distinct"],
                              cell=rec["cell"], verified=bool(verified), proxy_error=bool(proxy_wrong))
            if self.proxy == "visible":
                group_row["visible_pass_count"] = int(sum(rec["visible_pass"]))
                group_row["visible_pass"] = rec["visible_pass"]
            if in_burnin:
                self._burnin_rows.append(group_row)
            self.groups_f.write(json.dumps(group_row) + "\n")
        self.groups_f.flush()
        self.allocator.update(verified_for_update)

        perr = float(np.mean(proxy_err_unverified)) if proxy_err_unverified else None
        log = dict(type="train_step", step=self.call_idx, arm=self.allocator.name, n_groups=n_groups,
                   expensive_calls=len(verify_idx), corrupted_reward_groups=corrupted,
                   proxy_error_rate_among_unverified=perr, train_reward_mean=float(np.mean(rewards)))
        self.log_f.write(json.dumps(log) + "\n")
        self.log_f.flush()
        self.call_idx += 1
        if self.burnin_steps > 0 and self.call_idx == self.burnin_steps:
            self._run_burnin_diagnostic()
        return rewards

    def _run_burnin_diagnostic(self):
        """Called once, exactly when self.call_idx first reaches burnin_steps (right after the
        K-th burn-in step). self._burnin_rows holds the K*n_groups group_row dicts already
        written to groups.jsonl for those steps -- all verified, so proxy_error there is a TRUE
        label, not the usual unverified-only proxy estimate. Runs avrl_router.diagnose.diagnose()
        on them and, in --budget auto mode, sets self.budget (read fresh by every later __call__)
        to its recommendation, clamped to [0.05, 1.0]."""
        df = pd.DataFrame(self._burnin_rows)
        result = diagnose(df, confidence_cols=["agreement"], error_col="proxy_error",
                           budgets=[0.1, 0.25, 0.5], target_corruption=self.target_corruption,
                           n_boot=1000, seed=self.seed,
                           n_groups_per_step=self.prompts_per_step, pick=self.budget_pick)
        c = result["confidence_columns"]["agreement"]

        if self.budget_mode == "auto":
            if c.get("realizable"):
                # Realizable by construction: recommended is k / prompts_per_step for some
                # integer k, so n_verify = round(budget * n_groups) returns exactly k.
                self.budget = c["realizable"]["recommended"]
            else:
                self.budget = min(1.0, max(0.05, c["recommended_budget"]))

        summary = _format_burnin_summary(result, self.budget_mode, self.budget)
        print(summary)
        with open(os.path.join(self.out_dir, "burnin_report.md"), "w") as f:
            f.write(summary)
        with open(os.path.join(self.out_dir, "burnin_diagnosis.json"), "w") as f:
            json.dump(result, f, indent=2)

        if self.budget_mode == "auto":
            burnin_log = dict(type="burnin", step=self.burnin_steps, chosen_budget=self.budget,
                               n_groups=result["n_groups"], proxy_error_rate=result["proxy_error_rate"],
                               auc=c["auc"], recommended_budget=c["recommended_budget"],
                               target_mode=c["target_mode"], effective_target=c["effective_target"],
                               base_error=c["base_error"],
                               regime=c["regime"], budgets=c["budgets"],
                               budget_pick=self.budget_pick, realizable=c.get("realizable"))
            self.log_f.write(json.dumps(burnin_log) + "\n")
            self.log_f.flush()
            print(f"[p1_code_grpo] burn-in diagnostic: budget auto -> {self.budget:.2f}")


class CodeEval:
    """TrainerCallback: greedy pass@1 on the fixed MBPP eval subset every `every` steps (and
    always at the final step too, even when that does not land on a multiple of `every`), plus a
    HumanEval pass@1 at the final step ONLY -- HumanEval is eval-only throughout this experiment
    (never a training-reward source), matching the task spec: 'full HumanEval at the end'. Both
    go through the exact same sandbox as training, logged as {"type": "eval", ...} lines to the
    same jsonl log CodeRewardState writes to -- same split p1_grpo.py's PeriodicEval has from
    RewardState."""

    def __new__(cls, mbpp_df, humaneval_df, model_path, every, max_new_tokens, log_path,
                batch_size=25, timeout=5.0, sandbox_workers=None):
        from transformers import AutoTokenizer, TrainerCallback

        class _CodeEval(TrainerCallback):
            def __init__(self):
                self.mbpp_df = mbpp_df
                self.humaneval_df = humaneval_df
                self.every = every
                self.max_new_tokens = max_new_tokens
                self.log_path = log_path
                self.batch_size = batch_size
                self.timeout = timeout
                self.sandbox_workers = sandbox_workers
                self.tok = AutoTokenizer.from_pretrained(model_path)
                self.tok.padding_side = "left"
                if self.tok.pad_token is None:
                    self.tok.pad_token = self.tok.eos_token

            def on_step_end(self, args, state, control, **kwargs):
                step = state.global_step
                if step == 0:
                    return control
                is_final = step >= state.max_steps
                if step % self.every != 0 and not is_final:
                    return control
                model = kwargs["model"]
                mbpp_acc = eval_mbpp_pass1(model, self.tok, self.mbpp_df, self.max_new_tokens,
                                            self.batch_size, self.timeout, self.sandbox_workers)
                row = {"type": "eval", "step": step, "eval_mbpp_pass1": mbpp_acc,
                       "n_eval_mbpp": len(self.mbpp_df)}
                if is_final:
                    he_acc = eval_humaneval_pass1(model, self.tok, self.humaneval_df, self.max_new_tokens,
                                                   self.batch_size, self.timeout, self.sandbox_workers)
                    row["eval_humaneval_pass1"] = he_acc
                    row["n_eval_humaneval"] = len(self.humaneval_df)
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                return control

        return _CodeEval()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ALLOCATORS), help="required unless --eval_only is set")
    ap.add_argument("--proxy", choices=["signature", "visible"], default="signature",
                     help="cheap grader for UNverified groups' reward (E10b): signature (default, "
                          "unchanged) = output-signature majority vote; visible = passes only "
                          "test_list[:1] (the visible test), checked via a second sandbox call. "
                          "Verified groups always use the full test_list either way.")
    ap.add_argument("--budget", type=str, default="0.25",
                     help="verification budget fraction in [0,1], or 'auto' to pick it from "
                          "the --burnin_steps diagnostic (requires --burnin_steps > 0)")
    ap.add_argument("--burnin_steps", type=int, default=0,
                     help="verify every group with the expensive grader for this many initial "
                          "steps (regardless of --arm), then run the burn-in diagnostic "
                          "(avrl_router.diagnose) once on those groups before resuming normal "
                          "arm behaviour; 0 (default) disables burn-in entirely")
    ap.add_argument("--target_corruption", type=str, default="rel:0.25",
                     help="target admitted-corruption rate for the burn-in diagnostic's "
                          "recommended budget: a float (absolute) or 'rel:F' (F times the "
                          "measured burn-in proxy error rate), default rel:0.25")
    ap.add_argument("--budget_pick", choices=["point", "upper"], default="upper",
                     help="PRODUCT-4: in --budget auto mode, require the burn-in's realizable "
                          "candidate budget (k / prompts_per_step) to meet target_corruption "
                          "under the point estimate or the bootstrap upper 95%% bound (default "
                          "upper, conservative -- see RESEARCH_LOG.md's E10e-rel entry)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_seed", type=int, default=None, help="seed for the held-out eval subset only; default None keeps current behaviour (eval subset picked by --seed)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-5, help="LoRA-appropriate peak learning rate (p1_grpo.py's --lr default is 1e-6, tuned for its own earlier full-run reproduction; this task's spec asks for 1e-5)")
    ap.add_argument("--no_vllm", action="store_true", help="force TRL HF generation (timing comparisons)")
    # Default targets <script_dir>/models/... (i.e. ~/avrl/models/... on hannah, where the model
    # is actually copied, per p1_grpo.py's own convention); pass --model explicitly on goldberg,
    # where models/ lives one directory up (active-audit/models/...).
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "OLMo-2-0425-1B"))
    ap.add_argument("--train_data", default=os.path.join(ROOT, "data", "e10_train.parquet"))
    ap.add_argument("--curriculum", default=None, metavar="PATH",
                     help="JSON list of task_id values (E10b); restrict the training prompt pool "
                          "to these before the seeded shuffle (FAIRNESS RULE 5 unaffected -- the "
                          "shuffle is still the only ordering, still keyed only on --seed)")
    ap.add_argument("--test_data", default=os.path.join(ROOT, "data", "e10_test.parquet"))
    ap.add_argument("--humaneval", default=os.path.join(ROOT, "data", "e10_humaneval.parquet"))
    ap.add_argument("--prompts_per_step", type=int, default=8)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--max_completion_length", type=int, default=512)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_n", type=int, default=200, help="fixed MBPP-test subset size for the periodic during-training eval")
    ap.add_argument("--ckpt_every", type=int, default=50)
    ap.add_argument("--max_hours", type=float, default=5.0)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=5.0, help="sandbox wall-clock timeout per program, seconds")
    ap.add_argument("--sandbox_workers", type=int, default=None, help="thread-pool size for code_rewards.run_batch; default min(n_tasks, cpu_count)")
    ap.add_argument("--eval_batch_size", type=int, default=25)
    ap.add_argument("--eval_max_new_tokens", type=int, default=512)
    ap.add_argument("--eval_only", default=None, metavar="MODEL_PATH",
                     help="evaluate this model path (e.g. the Instruct reference bar, or the untrained base) on --test_data + --humaneval and exit; skips training entirely")
    ap.add_argument("--eval_only_n", type=int, default=None,
                     help="--eval_only only: subset both eval sets to this many rows (quick smoke check); default is the FULL test_data (500) + FULL humaneval (164)")
    args = ap.parse_args()
    if not args.eval_only and not args.arm:
        ap.error("--arm is required unless --eval_only is set")
    if args.budget == "auto":
        if args.burnin_steps <= 0:
            ap.error("--budget auto requires --burnin_steps > 0")
        args.budget_mode = "auto"
        args.budget_value = 0.0  # placeholder; overwritten by the burn-in diagnostic before use
    else:
        args.budget_mode = "fixed"
        try:
            args.budget_value = float(args.budget)
        except ValueError:
            ap.error(f"--budget must be a float or 'auto', got {args.budget!r}")
    if args.smoke:
        args.steps, args.prompts_per_step, args.G, args.eval_n = 3, 4, 4, 20
        args.eval_every = args.steps
        args.ckpt_every = args.steps
    return args


def run_eval_only(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.out, exist_ok=True)
    model_path = args.eval_only
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    n = args.eval_only_n  # None => full sets
    mbpp_df = load_eval_subset(args.test_data, n, args.seed) if n else pd.read_parquet(args.test_data)
    he_df = pd.read_parquet(args.humaneval)
    if n:
        he_df = he_df.head(n).reset_index(drop=True)

    t0 = time.time()
    mbpp_acc = eval_mbpp_pass1(model, tok, mbpp_df, args.eval_max_new_tokens, args.eval_batch_size,
                                args.timeout, args.sandbox_workers)
    he_acc = eval_humaneval_pass1(model, tok, he_df, args.eval_max_new_tokens, args.eval_batch_size,
                                   args.timeout, args.sandbox_workers)
    result = dict(type="eval_only", model=model_path, eval_mbpp_pass1=mbpp_acc, n_eval_mbpp=len(mbpp_df),
                   eval_humaneval_pass1=he_acc, n_eval_humaneval=len(he_df), wall_s=time.time() - t0)
    with open(os.path.join(args.out, "log.jsonl"), "a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=1))
    return result


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
    except ImportError:
        pass

    if args.eval_only:
        run_eval_only(args)
        return

    train_df = pd.read_parquet(args.train_data).reset_index(drop=True)
    if args.curriculum:
        # E10b: restrict the prompt pool to a fixed task_id set BEFORE the seeded shuffle below,
        # so FAIRNESS RULE 5 still holds (the shuffle remains the only ordering, keyed only on
        # --seed) -- it just shuffles a smaller, fixed pool. astype coerces the JSON list's ids
        # (str or int) to match train_df's own task_id dtype, whichever it is.
        allowed_ids = json.load(open(args.curriculum))
        allowed = pd.Series(allowed_ids).astype(train_df["task_id"].dtype)
        train_df = train_df[train_df["task_id"].isin(allowed)].reset_index(drop=True)
        print(f"[p1_code_grpo] curriculum pool size: {len(train_df)} (from {args.curriculum})")
    train_df["pid"] = range(len(train_df))
    # FAIRNESS RULE 5: the only shuffle of prompt order; fixed by --seed, independent of --arm.
    train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Prompt-embedding features computed once for ALL training prompts (not just the smoke
    # subset) before any arm-specific slicing, so the cache is arm-independent.
    # compute_prompt_feats (p1_grpo.py, unmodified) expects a "question" column; MBPP's problem
    # text column is "text".
    feats = compute_prompt_feats(train_df.rename(columns={"text": "question"}), args.out, args.seed)

    if args.smoke:
        train_df = train_df.head(max(args.prompts_per_step * 2, args.G)).reset_index(drop=True)

    # FAIRNESS RULE 4: read only from --test_data, a different file than --train_data.
    mbpp_eval_df = load_eval_subset(args.test_data, args.eval_n, args.seed, args.eval_seed)
    print(f"[p1_code_grpo] eval_seed={args.eval_seed} (effective eval subset seed: {args.seed if args.eval_seed is None else args.eval_seed})")
    he_df = pd.read_parquet(args.humaneval)
    if args.smoke:
        he_df = he_df.head(args.eval_n).reset_index(drop=True)

    hf_dataset = make_hf_dataset(train_df)

    allocator = ALLOCATORS[args.arm]()
    log_path = os.path.join(args.out, "log.jsonl")
    reward_state = CodeRewardState(allocator, args.budget_value, args.G, feats, log_path, args.seed,
                                    args.max_completion_length, timeout=args.timeout,
                                    sandbox_workers=args.sandbox_workers, proxy=args.proxy,
                                    burnin_steps=args.burnin_steps, target_corruption=args.target_corruption,
                                    budget_mode=args.budget_mode, prompts_per_step=args.prompts_per_step,
                                    budget_pick=args.budget_pick)

    from peft import LoraConfig
    from trl import GRPOTrainer

    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                             bias="none", task_type="CAUSAL_LM", target_modules="all-linear")
    grpo_config = build_grpo_config(args, args.out)
    trainer = GRPOTrainer(model=args.model, args=grpo_config, reward_funcs=[reward_state],
                          train_dataset=hf_dataset, peft_config=peft_config)
    trainer.add_callback(CodeEval(mbpp_eval_df, he_df, args.model, args.eval_every, args.eval_max_new_tokens,
                                   log_path, batch_size=args.eval_batch_size, timeout=args.timeout,
                                   sandbox_workers=args.sandbox_workers))
    trainer.add_callback(TimeLimit(args.max_hours))

    train_start = time.time()
    trainer.train()
    elapsed = time.time() - train_start
    n_steps = max(trainer.state.global_step, 1)
    print(f"[p1_code_grpo] trained {n_steps} steps in {elapsed:.1f}s ({elapsed / n_steps:.2f}s/step)")
    import torch
    if torch.cuda.is_available():
        print(f"[p1_code_grpo] peak_gpu_mem_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}")
    trainer.save_model(os.path.join(args.out, "final"))


if __name__ == "__main__":
    main()
