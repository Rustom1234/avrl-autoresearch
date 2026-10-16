"""Phase 1 of verification-aware RL: GRPO-train Qwen2.5-1.5B-Instruct on GSM8K with a
pluggable, budget-capped verification allocator deciding which rollout groups get the
expensive (gold) reward versus the cheap (majority-vote proxy) reward. See PROJECT.md.

Written against TRL's GRPOTrainer per https://huggingface.co/docs/trl/main/grpo_trainer
(main branch, fetched 2026-09-30) and trl/trainer/grpo_trainer.py's _calculate_rewards:
reward functions are called as reward_func(prompts=..., completions=...,
completion_ids=..., **extra_dataset_columns) -- reward_kwargs is every dataset column
except prompt/completion/completion_ids -- and must return one float per completion.
GRPOConfig exposes use_vllm / vllm_mode="colocate" for in-process vLLM generation.

This file imports trl/peft/torch/datasets/sentence_transformers only inside the
functions that need them, so `python -m py_compile` and the allocator/feature-
extraction logic are checkable on a machine without those packages (goldberg, today);
the full run only happens on hannah.

Fairness rules from the task spec are each marked inline with "FAIRNESS RULE N"; see
allocators.py's module docstring for the allocator-side invariants.
"""
import argparse
import json
import os
import random
import re
import time

import numpy as np
import pandas as pd

from allocators import ALLOCATORS

ROOT = os.path.dirname(os.path.abspath(__file__))
SYS_PROMPT = ("Solve the math problem. Reason step by step, then give the final answer "
              "on its own line as: #### <number>")
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
PC_COLS = [f"pc_{i}" for i in range(16)]

def extract_answer(text):
    m = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else ""

def normalize(x):
    try:
        return f"{float(x):.6g}"
    except (TypeError, ValueError):
        return x

def completion_text(c):
    """Completions may be plain strings or conversational message lists depending on
    TRL version/config; handle both."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        return c[-1].get("content", "")
    return str(c)

def load_gold(df):
    df = df.copy()
    df["gold"] = df.answer.str.split("####").str[-1].str.strip().str.replace(",", "")
    df["gold"] = df["gold"].map(normalize)
    return df

def compute_prompt_feats(train_df, out_dir, seed):
    """Sentence-MiniLM embedding -> PCA(16) -> KMeans(100 cells), computed once for all
    training prompts at startup (never touches gold), cached to <out>/prompt_feats.parquet."""
    cache = os.path.join(out_dir, "prompt_feats.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache).set_index("pid")
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(train_df.question.tolist(), show_progress_bar=False, batch_size=128)
    pcs = PCA(n_components=16, random_state=seed).fit_transform(emb)
    cells = KMeans(n_clusters=100, random_state=seed, n_init=10).fit_predict(pcs)
    feats = pd.DataFrame(pcs, columns=PC_COLS)
    feats["cell"] = cells.astype(int)
    feats["pid"] = train_df["pid"].values
    feats["q_len"] = train_df["question"].str.len().values
    feats.to_parquet(cache, index=False)
    return feats.set_index("pid")

def load_eval_prompts(test_path, n, seed, eval_seed=None):
    """Fixed, seeded subset of the *test* split only -- disjoint from training data by
    construction (different file, never merged with train_df). eval_seed, when given,
    selects the subset instead of seed (see --eval_seed), so runs that only differ by
    --seed can still be evaluated on the same held-out rows; everything else (training
    prompt order, generation sampling) is unaffected and still keyed on seed."""
    df = load_gold(pd.read_parquet(test_path))
    n = min(n, len(df))
    pick = seed if eval_seed is None else eval_seed
    return df.sample(n=n, random_state=pick).reset_index(drop=True)

def make_hf_dataset(df):
    from datasets import Dataset

    records = [
        {
            "prompt": [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": q}],
            "gold": g,
            "pid": int(pid),
        }
        for q, g, pid in zip(df.question, df.gold, df.pid)
    ]
    return Dataset.from_list(records)

class RewardState:
    """Holds the allocator and all per-run bookkeeping; the GRPOTrainer calls the
    instance itself as the reward function. One call == one step's worth of groups
    under the batch config main() builds (generation_batch_size == prompts_per_step *
    G, gradient_accumulation_steps == prompts_per_step), so `self.call_idx` lines up
    with the optimizer step count 1:1 -- see build_grpo_config."""

    def __init__(self, allocator, budget, G, feats, log_path, seed, max_completion_length):
        self.allocator = allocator
        self.budget = budget
        self.G = G
        self.feats = feats  # DataFrame indexed by pid: cell, q_len, pc_0..pc_15
        self.log_f = open(log_path, "a")
        groups_path = os.path.join(os.path.dirname(log_path), "groups.jsonl")
        self.groups_f = open(groups_path, "a")
        self.seed = seed
        self.max_completion_length = max_completion_length
        self.call_idx = 0
        self.__name__ = f"reward_{allocator.name}"  # TRL logs reward_funcs by __name__

    def _group_record(self, gi, pid, gold, comps, comp_ids):
        texts = [completion_text(c) for c in comps]
        answers = [normalize(extract_answer(t)) for t in texts]
        assert comp_ids is not None, (
            "completion_ids is None; this TRL version does not supply it as this "
            "file's module docstring assumes, so frac_trunc (compared against a "
            "token-count threshold) cannot be computed correctly from word counts"
        )
        lens = [len(x) for x in comp_ids]
        vals, counts = np.unique(answers, return_counts=True)
        maj_i = int(np.argmax(counts))
        majority = vals[maj_i]
        agreement = float(counts.max()) / self.G
        n_distinct = int(len(vals))
        mean_len = float(np.mean(lens))
        std_len = float(np.std(lens))
        frac_trunc = float(np.mean([l >= self.max_completion_length for l in lens]))
        prow = self.feats.loc[pid]
        # FAIRNESS RULE 1: feat_vec is built only from completions + the gold-free startup embedding cache; no gold.
        cheap = [agreement, n_distinct, mean_len, std_len, frac_trunc, float(prow.q_len)]
        feat_vec = cheap + [float(prow[c]) for c in PC_COLS]
        return dict(
            idx=gi, pid=pid, gold=gold, majority=majority, agreement=agreement,
            n_distinct=n_distinct, cell=int(prow.cell), feat_vec=feat_vec, answers=answers,
        )

    def __call__(self, prompts, completions, completion_ids=None, **kwargs):
        gold_col = kwargs["gold"]
        pid_col = kwargs["pid"]
        G = self.G
        n_total = len(completions)
        assert n_total % G == 0, f"batch of {n_total} completions not divisible by G={G}"
        n_groups = n_total // G

        records = []
        for gi in range(n_groups):
            sl = slice(gi * G, (gi + 1) * G)
            pid_chunk, gold_chunk = pid_col[sl], gold_col[sl]
            assert len(set(pid_chunk)) == 1 and len(set(gold_chunk)) == 1, \
                "reward fn assumes TRL groups completions contiguously by prompt; violated for this batch"
            comp_ids_chunk = completion_ids[sl] if completion_ids is not None else None
            records.append(self._group_record(gi, pid_chunk[0], normalize(gold_chunk[0]), completions[sl], comp_ids_chunk))

        # FAIRNESS RULE 2: one pure function of (n_groups, budget), called fresh each
        # step -- identical expensive-call count for every arm at the same budget, no
        # carry-over.
        n_verify = self.allocator.n_verify(n_groups, self.budget)
        rng = np.random.default_rng(self.seed * 1_000_003 + self.call_idx)
        # FAIRNESS RULE 1 (structural): strip gold before handing records to the
        # allocator, so no select() implementation -- current or future -- can read it.
        safe_records = [{k: v for k, v in r.items() if k != "gold"} for r in records]
        verify_idx = set(self.allocator.select(safe_records, n_verify, rng))
        assert len(verify_idx) == n_verify, f"{self.allocator.name}.select returned {len(verify_idx)}, expected {n_verify}"

        rewards = [0.0] * n_total
        verified_for_update = []  # FAIRNESS RULE 3: only groups verified this step
        proxy_err_unverified = []
        corrupted = 0
        for rec in records:
            gi = rec["idx"]
            sl = slice(gi * G, (gi + 1) * G)
            maj_wrong = int(rec["majority"] != rec["gold"])  # gold used for logging/reward only, never features
            verified = gi in verify_idx
            if verified:
                for k in range(G):
                    rewards[gi * G + k] = float(rec["answers"][k] == rec["gold"])
                verified_for_update.append((rec["feat_vec"], maj_wrong, rec["cell"]))
            else:
                for k in range(G):
                    rewards[gi * G + k] = float(rec["answers"][k] == rec["majority"])
                proxy_err_unverified.append(maj_wrong)
                corrupted += maj_wrong
            group_row = dict(step=self.call_idx, idx=gi, agreement=rec["agreement"], n_distinct=rec["n_distinct"],
                              cell=rec["cell"], verified=bool(verified), proxy_correct=not bool(maj_wrong),
                              proxy_error=bool(maj_wrong), train_reward=float(np.mean(rewards[sl])))
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
        return rewards

def build_grpo_config(args, out_dir):
    """Only pass GRPOConfig fields the installed TRL version actually accepts (checked
    via introspection), so this runs against whatever TRL is on hannah: vLLM colocate
    if `use_vllm`/`vllm_mode` exist, else falls back to TRL's default HF generation."""
    import inspect

    from trl import GRPOConfig

    valid = set(inspect.signature(GRPOConfig.__init__).parameters)
    # training micro-batch: G sequences per forward by default; --micro_batch halves the full-vocabulary
    # logits memory (the peak for a 7B policy) while keeping one optimizer step per generation batch
    micro_batch = getattr(args, "micro_batch", None) or args.G
    assert (args.prompts_per_step * args.G) % micro_batch == 0, "prompts_per_step x G must be a multiple of micro_batch"
    cfg = dict(
        output_dir=out_dir,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=(args.prompts_per_step * args.G) // micro_batch,
        generation_batch_size=args.prompts_per_step * args.G,
        num_generations=args.G,
        max_completion_length=args.max_completion_length,
        max_steps=args.steps,
        bf16=True,
        learning_rate=args.lr,
        temperature=1.0,
        save_strategy="steps",
        save_steps=args.ckpt_every,
        logging_steps=1,
        seed=args.seed,
        report_to=[],
    )
    if getattr(args, "policy_single_gpu", False) and "model_init_kwargs" in valid:
        # with two visible GPUs TRL loads the policy with an automatic device map and spreads it over
        # both cards; pin it to cuda:0 so a judge on cuda:1 has its card to itself
        cfg["model_init_kwargs"] = {"device_map": {"": 0}}
    if getattr(args, "gradient_checkpointing", False):
        # needed for policies of 7B parameters and up on a 40 GB card (activations of 8 x 1400 tokens
        # without checkpointing exceed the card); use_reentrant=False is the PEFT-safe variant
        cfg["gradient_checkpointing"] = True
        if "gradient_checkpointing_kwargs" in valid:
            cfg["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if "use_vllm" in valid and not args.no_vllm:
        cfg["use_vllm"] = True
        print("[p1_grpo] vLLM generation requested (use_vllm=True, colocate)", flush=True)
        if "vllm_mode" in valid:
            cfg["vllm_mode"] = "colocate"
        if "vllm_gpu_memory_utilization" in valid:
            cfg["vllm_gpu_memory_utilization"] = 0.3
        if "vllm_max_model_length" in valid:
            cfg["vllm_max_model_length"] = 1024 + args.max_completion_length  # KV cache sized for prompt + completion, not the model's 32k default
    dropped = sorted(k for k in cfg if k not in valid)
    cfg = {k: v for k, v in cfg.items() if k in valid}
    if dropped:
        print(f"[p1_grpo] this TRL's GRPOConfig has no field(s) {dropped}; dropped, defaults apply")
    config = GRPOConfig(**cfg)
    if getattr(args, "policy_single_gpu", False):
        # with two visible GPUs the Trainer wraps the policy in DataParallel (n_gpu > 1) and tries to
        # replicate it onto the judge's card; report a single GPU so the policy trains on cuda:0 alone
        _ = config.n_gpu
        config._n_gpu = 1
        print(f"[p1_grpo] policy pinned to a single GPU (n_gpu forced to {config.n_gpu})", flush=True)
    return config

class PeriodicEval:
    """transformers.TrainerCallback subclass, built lazily (needs torch/transformers)
    so this module still imports with neither installed. Greedy-decodes the fixed
    held-out eval set every `every` steps and appends an {"type": "eval", ...} line to
    the same jsonl log RewardState writes to."""

    def __new__(cls, eval_df, model_path, every, max_new_tokens, log_path, batch_size=25):
        import torch
        from transformers import AutoTokenizer, TrainerCallback

        class _PeriodicEval(TrainerCallback):
            def __init__(self):
                self.eval_df = eval_df
                self.every = every
                self.max_new_tokens = max_new_tokens
                self.log_path = log_path
                self.batch_size = batch_size
                self.tok = AutoTokenizer.from_pretrained(model_path)
                self.tok.padding_side = "left"
                if self.tok.pad_token is None:
                    self.tok.pad_token = self.tok.eos_token

            def on_step_end(self, args, state, control, **kwargs):
                step = state.global_step
                if step == 0 or step % self.every != 0:
                    return control
                model = kwargs["model"]
                was_training = model.training
                model.eval()
                correct = 0
                with torch.no_grad():
                    for i in range(0, len(self.eval_df), self.batch_size):
                        chunk = self.eval_df.iloc[i:i + self.batch_size]
                        texts = [
                            self.tok.apply_chat_template(
                                [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": q}],
                                tokenize=False, add_generation_prompt=True,
                            )
                            for q in chunk.question
                        ]
                        enc = self.tok(texts, return_tensors="pt", padding=True).to(model.device)
                        out = model.generate(
                            **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                            pad_token_id=self.tok.pad_token_id,
                        )
                        gen = out[:, enc["input_ids"].shape[1]:]
                        decoded = self.tok.batch_decode(gen, skip_special_tokens=True)
                        for text, gold in zip(decoded, chunk.gold):
                            correct += int(normalize(extract_answer(text)) == gold)
                if was_training:
                    model.train()
                acc = correct / len(self.eval_df)
                with open(self.log_path, "a") as f:
                    f.write(json.dumps({"type": "eval", "step": step, "eval_accuracy": acc, "n_eval": len(self.eval_df)}) + "\n")
                return control

        return _PeriodicEval()

class TimeLimit:
    """Stops training after max_hours regardless of --steps ('400 steps or 5 hours,
    whichever first')."""

    def __new__(cls, max_hours):
        from transformers import TrainerCallback

        class _TimeLimit(TrainerCallback):
            def __init__(self):
                self.deadline = time.time() + max_hours * 3600

            def on_step_end(self, args, state, control, **kwargs):
                if time.time() > self.deadline:
                    control.should_training_stop = True
                return control

        return _TimeLimit()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ALLOCATORS))
    ap.add_argument("--budget", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_seed", type=int, default=None, help="seed for the held-out eval subset only; default None keeps current behaviour (eval subset picked by --seed)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-6, help="peak learning rate (1e-6 reproduces the Oct 1 runs; 1e-5 is the LoRA-appropriate value)")
    ap.add_argument("--no_vllm", action="store_true", help="force TRL HF generation (timing comparisons)")
    # Default targets <script_dir>/models/... (i.e. ~/avrl/models/... on hannah, where
    # the model is actually copied); pass --model explicitly if ever training from goldberg.
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "Qwen2.5-1.5B-Instruct"))
    ap.add_argument("--train_data", default=os.path.join(ROOT, "data", "gsm8k_train.parquet"))
    ap.add_argument("--test_data", default=os.path.join(ROOT, "data", "gsm8k_test.parquet"))
    ap.add_argument("--prompts_per_step", type=int, default=16)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--max_completion_length", type=int, default=512)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_n", type=int, default=500)
    ap.add_argument("--ckpt_every", type=int, default=100)
    ap.add_argument("--max_hours", type=float, default=5.0)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--micro_batch", type=int, default=None, help="sequences per training forward (default G); use 4 for a 7B policy on 40 GB")
    ap.add_argument("--policy_single_gpu", action="store_true", help="pin the policy to cuda:0 (use when a judge sits on another visible GPU)")
    ap.add_argument("--gradient_checkpointing", action="store_true", help="recompute activations in the backward pass (required for 7B policies on 40 GB)")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.prompts_per_step, args.G, args.eval_n = 3, 4, 4, 20
        args.eval_every = args.steps
        args.ckpt_every = args.steps
    return args

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

    train_df = load_gold(pd.read_parquet(args.train_data)).reset_index(drop=True)
    train_df["pid"] = range(len(train_df))
    # FAIRNESS RULE 5: the only shuffle of prompt order; fixed by --seed, independent of --arm.
    train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Prompt-embedding features computed once for ALL training prompts (not just the
    # smoke subset) before any arm-specific slicing, so the cache is arm-independent.
    feats = compute_prompt_feats(train_df, args.out, args.seed)

    if args.smoke:
        train_df = train_df.head(max(args.prompts_per_step * 2, args.G)).reset_index(drop=True)

    # FAIRNESS RULE 4: read only from --test_data, a different file than --train_data.
    eval_df = load_eval_prompts(args.test_data, args.eval_n, args.seed, args.eval_seed)
    print(f"[p1_grpo] eval_seed={args.eval_seed} (effective eval subset seed: {args.seed if args.eval_seed is None else args.eval_seed})")

    hf_dataset = make_hf_dataset(train_df)

    allocator = ALLOCATORS[args.arm]()
    log_path = os.path.join(args.out, "log.jsonl")
    reward_state = RewardState(allocator, args.budget, args.G, feats, log_path, args.seed, args.max_completion_length)

    from peft import LoraConfig
    from trl import GRPOTrainer

    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                             bias="none", task_type="CAUSAL_LM", target_modules="all-linear")
    grpo_config = build_grpo_config(args, args.out)
    trainer = GRPOTrainer(model=args.model, args=grpo_config, reward_funcs=[reward_state],
                          train_dataset=hf_dataset, peft_config=peft_config)
    trainer.add_callback(PeriodicEval(eval_df, args.model, args.eval_every, args.max_completion_length, log_path))
    trainer.add_callback(TimeLimit(args.max_hours))

    train_start = time.time()
    trainer.train()
    elapsed = time.time() - train_start
    n_steps = max(trainer.state.global_step, 1)
    print(f"[p1_grpo] trained {n_steps} steps in {elapsed:.1f}s ({elapsed / n_steps:.2f}s/step)")
    if torch.cuda.is_available():
        print(f"[p1_grpo] peak_gpu_mem_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}")
    trainer.save_model(os.path.join(args.out, "final"))

if __name__ == "__main__":
    main()
